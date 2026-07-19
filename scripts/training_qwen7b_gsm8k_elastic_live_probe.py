#!/usr/bin/env python3
"""Run two-generation elastic Qwen2.5-7B-Instruct GSM8K SFT on Kaggle."""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from crowdtensor.qwen15b_training import sha256_file, stable_hash
from crowdtensor.qwen7b_gsm8k_showcase import (
    DATASET_ID,
    DATASET_MANIFEST_SCHEMA,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_PARAMETER_COUNT,
    MODEL_REVISION,
    PRIVATE_TRAIN_SCHEMA,
    SOURCE_LAYOUT_SCHEMA,
)
from scripts.kaggle_gpu_token_weekly_quota_probe import clean_env
from scripts.training_cuda_kaggle_common import public_safety_errors, utc_now
from scripts.training_cuda_two_node_probe import (
    ensure_cloudflared,
    stop_process,
)
from scripts.training_qwen15b_elastic_live_probe import (
    ElasticQwenCoordinator,
    _delete_refs,
    _evaluate,
    _free_port,
    _get_json,
    _launch_generation,
)
from scripts.training_qwen15b_four_gpu_probe import (
    _credential_sections,
    _load,
    preflight_accounts,
    start_verified_tunnel,
)


SCHEMA = "crowdtensor_qwen7b_gsm8k_elastic_live_probe_v1"
ADAPTER_NAME = "training_qwen7b_standard_peft_adapter.zip"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_live(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    private = output / ".private-runtime"
    if private.exists():
        shutil.rmtree(private)
    private.mkdir(parents=True)
    private.chmod(0o700)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "training_ready": False,
        "live_run_performed": not args.preflight_only,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "parameter_count": MODEL_PARAMETER_COUNT,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "topology": "two-sequential-same-account-kaggle-t4x2-pairs",
        "target_steps": int(args.target_steps),
        "replacement_step": int(args.replacement_step),
        "microbatches_per_step": int(args.microbatches_per_step),
        "learning_rate": float(args.learning_rate),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "blockers": [],
        "started_at": utc_now(),
        "mock_runtime_used": False,
        "cpu_fallback_used": False,
        "tiny_or_random_model_used": False,
        "full_parameter_training_claimed": False,
        "physical_multi_host_verified": False,
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_tensor_values_public": False,
        "adapter_tensor_values_public": False,
        "credentials_public": False,
        "credential_paths_public": False,
        "coordinator_url_public": False,
        "session_tokens_public": False,
        "assignment_tokens_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
        "cleanup": {
            "all_four_kernels_deleted": False,
            "coordinator_stopped": False,
            "tunnel_stopped": False,
            "private_runtime_removed": False,
            "rendezvous_payloads_removed": False,
            "uncommitted_checkpoint_blobs_removed": False,
            "live_resources_left_running": True,
        },
    }
    coordinator: ElasticQwenCoordinator | None = None
    tunnel_process = None
    selected_env_values: dict[str, str] = {}
    all_refs: list[str] = []
    old_report: dict[str, Any] = {}
    new_report: dict[str, Any] = {}
    try:
        sections = _credential_sections(
            list(args.token_file or []),
            raw_token_file=str(args.raw_token_file),
            raw_token_username=str(args.raw_token_username),
        )
        if not sections:
            raise RuntimeError("qwen7b_private_kaggle_credentials_required")
        preflight, candidates = preflight_accounts(sections)
        report["account_preflight"] = preflight
        report["eligible_account_count"] = len(candidates)
        if args.preflight_only:
            report["blockers"].append("qwen7b_preflight_only_no_live_run")
            return report
        if not candidates:
            raise RuntimeError("qwen7b_same_account_t4x2_pair_unavailable")
        selected = candidates[0]
        selected_env_values = dict(selected["env_values"])
        owner = str(selected["owner"])
        report["selected_account"] = {
            "owner_hash": str(selected["owner_hash"]),
            "effective_remaining_seconds": float(selected["effective_remaining"]),
            "credential_values_public": False,
        }

        source_path = Path(args.source_layout).resolve()
        train_path = Path(args.train_payload).resolve()
        dataset_manifest_path = Path(args.dataset_manifest).resolve()
        source = _load(source_path)
        train = _load(train_path)
        dataset_manifest = _load(dataset_manifest_path)
        required_rows = int(args.target_steps) * int(args.microbatches_per_step)
        if (
            source.get("schema") != SOURCE_LAYOUT_SCHEMA
            or source.get("source_verified") is not True
            or source.get("model_id") != MODEL_ID
            or source.get("model_revision") != MODEL_REVISION
            or int(source.get("parameter_count") or 0) != MODEL_PARAMETER_COUNT
        ):
            raise RuntimeError("qwen7b_source_layout_invalid")
        if (
            train.get("schema") != PRIVATE_TRAIN_SCHEMA
            or train.get("model_id") != MODEL_ID
            or train.get("model_revision") != MODEL_REVISION
            or train.get("dataset_id") != DATASET_ID
            or train.get("dataset_revision") != DATASET_REVISION
            or len(train.get("train") or []) < required_rows
            or len(train.get("validation") or []) < 8
        ):
            raise RuntimeError("qwen7b_private_training_payload_invalid")
        if (
            dataset_manifest.get("schema") != DATASET_MANIFEST_SCHEMA
            or dataset_manifest.get("train_test_split_isolation_verified") is not True
        ):
            raise RuntimeError("qwen7b_dataset_manifest_invalid")
        sequence_length = int(train.get("sequence_length") or 0)
        selected_rows = list(train["train"][:required_rows])
        non_padding_tokens = sum(
            int(row.get("non_padding_token_count") or 0) for row in selected_rows
        )
        supervised_tokens = sum(
            int(row.get("supervised_token_count") or 0) for row in selected_rows
        )
        report["source"] = {
            "source_layout_hash": sha256_file(source_path),
            "source_content_hash": str(source.get("content_hash") or ""),
            "weight_bytes": int(source.get("weight_bytes") or 0),
            "tensor_count": int(source.get("tensor_count") or 0),
            "multi_file_source": True,
            "stage_selective_range_loading": True,
        }
        report["dataset"] = {
            "dataset_manifest_hash": sha256_file(dataset_manifest_path),
            "private_train_payload_hash": sha256_file(train_path),
            "sequence_length": sequence_length,
            "available_training_sequence_count": len(train["train"]),
            "selected_training_sequence_count": required_rows,
            "validation_sequence_count": len(train["validation"]),
            "train_test_split_isolation_verified": True,
            "train_token_hash": stable_hash(selected_rows),
            "validation_token_hash": stable_hash(train["validation"]),
            "raw_text_public": False,
            "token_ids_public": False,
        }
        report["training_budget"] = {
            "optimizer_steps": int(args.target_steps),
            "microbatches_per_step": int(args.microbatches_per_step),
            "sequence_length": sequence_length,
            "training_sequence_count": required_rows,
            "training_non_padding_token_count": non_padding_tokens,
            "training_supervised_token_count": supervised_tokens,
            "replacement_after_step": int(args.replacement_step),
        }
        config = dict(source["config"])
        run_id = f"qwen7b-gsm8k-{int(time.time())}-{secrets.token_hex(3)}"
        coordinator_token = secrets.token_urlsafe(32)
        port = _free_port()
        coordinator = ElasticQwenCoordinator(
            private_root=private,
            port=port,
            run_id=run_id,
            token=coordinator_token,
            target_steps=int(args.target_steps),
            microbatches_per_step=int(args.microbatches_per_step),
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
        )
        coordinator.start()
        local_url = f"http://127.0.0.1:{port}"
        cloudflared = ensure_cloudflared(private)
        tunnel_process, tunnel_url, route = start_verified_tunnel(
            cloudflared,
            local_url,
            private,
            token=coordinator_token,
            run_id=run_id,
            attempts=int(args.tunnel_attempts),
            timeout=float(args.route_timeout_seconds),
        )
        route_elastic = _get_json(
            f"{tunnel_url}/elastic-training/status", token=coordinator_token
        )
        report["route_preflight"] = {
            **route,
            "elastic_status_verified": route_elastic.get("schema")
            == "crowdtensor_elastic_training_status_v1",
        }
        if report["route_preflight"]["elastic_status_verified"] is not True:
            raise RuntimeError("qwen7b_tunnel_status_route_invalid")

        suffix = f"{str(int(time.time()))[-8:]}-{secrets.token_hex(2)}"
        with tempfile.TemporaryDirectory(
            prefix="ct-qwen7b-elastic-account-"
        ) as config_dir:
            env = clean_env(selected_env_values, config_dir=Path(config_dir))
            old_report, old_refs = _launch_generation(
                generation="old",
                expected_start_step=0,
                segment_end_step=int(args.replacement_step),
                target_steps=int(args.target_steps),
                microbatches_per_step=int(args.microbatches_per_step),
                owner=owner,
                suffix=suffix,
                config=config,
                tokenized=train_path,
                coordinator_url=tunnel_url,
                coordinator_token=coordinator_token,
                run_id=run_id,
                env=env,
                private=private,
                output=output,
                allocation_timeout=float(args.allocation_timeout_seconds),
                push_timeout=float(args.push_timeout_seconds),
                status_timeout=float(args.status_timeout_seconds),
                output_timeout=float(args.output_timeout_seconds),
                delete_timeout=float(args.delete_timeout_seconds),
                poll_interval=float(args.poll_interval_seconds),
                cleanup_registry=all_refs,
                learning_rate=float(args.learning_rate),
                lora_rank=int(args.lora_rank),
                lora_alpha=int(args.lora_alpha),
                kernel_timeout_seconds=int(args.kernel_timeout_seconds),
                model_id=MODEL_ID,
                model_revision=MODEL_REVISION,
                parameter_count=MODEL_PARAMETER_COUNT,
                source_layout_path=source_path,
                defer_evaluation=True,
                adapter_destination_name=ADAPTER_NAME,
            )
            report["old_generation"] = old_report
            if old_report.get("ok") is not True:
                raise RuntimeError("qwen7b_old_generation_acceptance_incomplete")
            midpoint = coordinator.runtime.public_status()
            report["midpoint_status"] = midpoint
            if not (
                int(midpoint.get("committed_step") or 0)
                == int(args.replacement_step)
                and midpoint.get("zero_live_miners") is True
                and midpoint.get("paused_waiting_for_miners") is True
                and old_report.get("all_kernels_deleted") is True
            ):
                raise RuntimeError("qwen7b_midpoint_pause_not_verified")
            report["midpoint_checkpoint_retention"] = (
                coordinator.runtime.enforce_checkpoint_retention()
            )
            report["midpoint_consumed_payload_cleanup"] = (
                coordinator.rendezvous.cleanup()
            )
            pause_started = time.monotonic()
            pause_observations = []
            while time.monotonic() - pause_started < float(args.pause_delay_seconds):
                current = coordinator.runtime.public_status()
                pause_observations.append(
                    {
                        "observed_at": utc_now(),
                        "runtime_state": current.get("runtime_state"),
                        "committed_step": current.get("committed_step"),
                        "live_miner_count": current.get("live_miner_count"),
                    }
                )
                time.sleep(1.0)
            pause_elapsed = time.monotonic() - pause_started
            report["full_offline_pause"] = {
                "requested_seconds": float(args.pause_delay_seconds),
                "observed_seconds": pause_elapsed,
                "observations": pause_observations,
                "new_kernel_launched_during_pause": False,
            }
            new_report, new_refs = _launch_generation(
                generation="new",
                expected_start_step=int(args.replacement_step),
                segment_end_step=int(args.target_steps),
                target_steps=int(args.target_steps),
                microbatches_per_step=int(args.microbatches_per_step),
                owner=owner,
                suffix=suffix,
                config=config,
                tokenized=train_path,
                coordinator_url=tunnel_url,
                coordinator_token=coordinator_token,
                run_id=run_id,
                env=env,
                private=private,
                output=output,
                allocation_timeout=float(args.allocation_timeout_seconds),
                push_timeout=float(args.push_timeout_seconds),
                status_timeout=float(args.status_timeout_seconds),
                output_timeout=float(args.output_timeout_seconds),
                delete_timeout=float(args.delete_timeout_seconds),
                poll_interval=float(args.poll_interval_seconds),
                cleanup_registry=all_refs,
                learning_rate=float(args.learning_rate),
                lora_rank=int(args.lora_rank),
                lora_alpha=int(args.lora_alpha),
                kernel_timeout_seconds=int(args.kernel_timeout_seconds),
                model_id=MODEL_ID,
                model_revision=MODEL_REVISION,
                parameter_count=MODEL_PARAMETER_COUNT,
                source_layout_path=source_path,
                defer_evaluation=True,
                adapter_destination_name=ADAPTER_NAME,
            )
            report["new_generation"] = new_report
            final_status = coordinator.runtime.public_status()
            rendezvous = coordinator.rendezvous.public_status()
            report["final_status"] = final_status
            report["rendezvous"] = rendezvous
            evidence = _evaluate(
                old=old_report,
                new=new_report,
                midpoint=midpoint,
                final=final_status,
                rendezvous=rendezvous,
                pause_observations=pause_observations,
                pause_seconds=pause_elapsed,
                target_steps=int(args.target_steps),
                replacement_step=int(args.replacement_step),
                allow_deferred_evaluation=True,
            )
            report["evidence"] = evidence
            adapter_path = output / ADAPTER_NAME
            report["adapter"] = {
                **dict(new_report.get("adapter_bundle") or {}),
                "archive_file": ADAPTER_NAME,
                "archive_hash": sha256_file(adapter_path)
                if adapter_path.is_file()
                else "",
                "isolated_benchmark_required": True,
            }
            report["training_ready"] = bool(
                evidence.get("verified") is True
                and adapter_path.is_file()
                and report["adapter"].get("verified") is True
            )
            report["ok"] = report["training_ready"]
            if not report["ok"]:
                report["blockers"].append("qwen7b_live_training_acceptance_incomplete")
    except BaseException as exc:
        report["blockers"].append(str(exc).split(":", 1)[0][:180])
        report["error_class"] = type(exc).__name__
    finally:
        generation_cleanup_verified = bool(
            all_refs
            and old_report.get("all_kernels_deleted") is True
            and (not new_report or new_report.get("all_kernels_deleted") is True)
        )
        if generation_cleanup_verified:
            report["cleanup"]["all_four_kernels_deleted"] = True
        elif selected_env_values and all_refs:
            with tempfile.TemporaryDirectory(
                prefix="ct-qwen7b-final-cleanup-"
            ) as config_dir:
                env = clean_env(selected_env_values, config_dir=Path(config_dir))
                deletions, deleted = _delete_refs(
                    all_refs,
                    env=env,
                    timeout=float(args.delete_timeout_seconds),
                    role_by_ref={ref: "" for ref in all_refs},
                )
                report["final_cleanup_deletions"] = deletions
                report["cleanup"]["all_four_kernels_deleted"] = deleted
        else:
            report["cleanup"]["all_four_kernels_deleted"] = not all_refs
        if coordinator is not None:
            report["final_checkpoint_retention"] = (
                coordinator.runtime.enforce_checkpoint_retention()
            )
            blob_cleanup = coordinator.runtime.cleanup_uncommitted_blobs()
            report["checkpoint_blob_cleanup"] = blob_cleanup
            report["cleanup"]["uncommitted_checkpoint_blobs_removed"] = (
                blob_cleanup.get("ok") is True
            )
            rendezvous_cleanup = coordinator.rendezvous.cleanup()
            report["rendezvous_cleanup"] = rendezvous_cleanup
            report["cleanup"]["rendezvous_payloads_removed"] = (
                rendezvous_cleanup.get("private_payloads_removed") is True
            )
            report["cleanup"]["coordinator_stopped"] = coordinator.stop()
        else:
            report["cleanup"]["uncommitted_checkpoint_blobs_removed"] = True
            report["cleanup"]["rendezvous_payloads_removed"] = True
            report["cleanup"]["coordinator_stopped"] = True
        report["cleanup"]["tunnel_stopped"] = stop_process(tunnel_process)
        shutil.rmtree(private, ignore_errors=True)
        report["cleanup"]["private_runtime_removed"] = not private.exists()
        report["cleanup"]["live_resources_left_running"] = not all(
            report["cleanup"].get(key) is True
            for key in (
                "all_four_kernels_deleted",
                "coordinator_stopped",
                "tunnel_stopped",
                "private_runtime_removed",
            )
        )
        if report["cleanup"]["live_resources_left_running"]:
            report["ok"] = False
            report["training_ready"] = False
            report["blockers"].append("qwen7b_live_cleanup_incomplete")
        report["blockers"] = sorted(set(report["blockers"]))
        report["finished_at"] = utc_now()
        safety = public_safety_errors(report)
        report["public_artifact_safe"] = not safety
        if safety:
            report["ok"] = False
            report["training_ready"] = False
            report["public_safety_error_count"] = len(safety)
        report["content_hash"] = stable_hash(
            {key: value for key, value in report.items() if key != "content_hash"}
        )
        _write(output / "training_qwen7b_gsm8k_elastic_live_probe.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token-file", action="append", default=[])
    parser.add_argument("--raw-token-file", default="")
    parser.add_argument("--raw-token-username", default="")
    parser.add_argument("--source-layout", required=True)
    parser.add_argument("--train-payload", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--target-steps", type=int, default=256)
    parser.add_argument("--replacement-step", type=int, default=128)
    parser.add_argument("--microbatches-per-step", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--pause-delay-seconds", type=float, default=10.0)
    parser.add_argument("--allocation-timeout-seconds", type=float, default=43200.0)
    parser.add_argument("--push-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--output-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--delete-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=20.0)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=43200)
    parser.add_argument("--tunnel-attempts", type=int, default=3)
    parser.add_argument("--route-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.target_steps <= 2048:
        parser.error("--target-steps must be in [2, 2048]")
    if not 1 <= args.replacement_step < args.target_steps:
        parser.error("--replacement-step must be in [1, target_steps)")
    if not 1 <= args.microbatches_per_step <= 16:
        parser.error("--microbatches-per-step must be in [1, 16]")
    if not 5.0 <= args.pause_delay_seconds <= 120.0:
        parser.error("--pause-delay-seconds must be in [5, 120]")
    if not 300 <= args.kernel_timeout_seconds <= 43200:
        parser.error("--kernel-timeout-seconds must be in [300, 43200]")
    report = run_live(args)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": report["ok"],
                    "training_ready": report["training_ready"],
                    "blockers": report["blockers"],
                    "cleanup": report["cleanup"],
                },
                sort_keys=True,
            )
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
