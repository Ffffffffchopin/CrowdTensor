#!/usr/bin/env python3
"""Run the ordinary-user Elastic Training Beta path on replacement Kaggle Miners."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.elastic_training_beta import (  # noqa: E402
    ElasticTrainingBetaController,
    create_elastic_training_beta_app,
)
from crowdtensor.qwen15b_training import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    fetch_bytes,
    sha256_bytes,
    stable_hash,
    _hf_url,
)
from scripts.training_cuda_kaggle_common import (  # noqa: E402
    public_safety_errors,
    utc_now,
)
from scripts.training_cuda_two_node_probe import (  # noqa: E402
    ensure_cloudflared,
    stop_process,
)
from scripts.training_qwen15b_elastic_live_probe import (  # noqa: E402
    _delete_refs,
    _free_port,
    _get_json,
    _launch_generation,
    _wait_ready,
    _write,
)
from scripts.training_qwen15b_four_gpu_probe import (  # noqa: E402
    _credential_sections,
    preflight_accounts,
    start_verified_tunnel,
)


SCHEMA = "crowdtensor_elastic_training_beta_live_probe_v1"


class _PreflightComplete(Exception):
    pass


def _run_cli(arguments: list[str], *, timeout: float = 300.0) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "crowdtensor.cli", *arguments, "--json"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=float(timeout),
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    value: dict[str, Any] = {}
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            value = parsed
            break
    return {
        "returncode": int(completed.returncode),
        "report": value,
        "output_hash": stable_hash({"output": completed.stdout[-4000:]}),
        "raw_output_public": False,
    }


class ProductService:
    def __init__(self, *, job_dir: Path, port: int) -> None:
        self.job_dir = job_dir
        self.port = int(port)
        self.controller = ElasticTrainingBetaController(job_dir)
        credentials = self.controller.credentials()
        self.owner_token = str(credentials["owner_token"])
        self.miner_token = str(credentials["miner_token"])
        self.server: Any = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        import uvicorn

        app = create_elastic_training_beta_app(
            self.controller,
            owner_token=self.owner_token,
            miner_token=self.miner_token,
        )
        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
                access_log=False,
            )
        )
        self.server.install_signal_handlers = lambda: None
        self.thread = threading.Thread(
            target=self.server.run,
            name="elastic-training-beta-product-service",
            daemon=True,
        )
        self.thread.start()
        _wait_ready(f"http://127.0.0.1:{self.port}", timeout=30.0)

    def stop(self) -> bool:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=30.0)
        return bool(self.thread is None or not self.thread.is_alive())


def _product_generation_summary(generation: dict[str, Any]) -> dict[str, Any]:
    workers = [dict(item.get("worker") or {}) for item in generation.get("worker_reports") or []]
    return {
        "ok": generation.get("ok") is True,
        "worker_count": len(workers),
        "product_miner_mode": bool(
            workers
            and all(
                item.get("schema")
                == "crowdtensor_elastic_training_beta_miner_join_v1"
                for item in workers
            )
        ),
        "roles": sorted(str(item.get("role") or "") for item in workers),
        "start_steps": sorted(int(item.get("expected_start_step") or 0) for item in workers),
        "end_steps": sorted(int(item.get("segment_end_step") or 0) for item in workers),
        "barrier_commit_counts": sorted(int(item.get("barrier_commit_count") or 0) for item in workers),
        "all_completed_barriers_committed": bool(
            workers and all(item.get("all_completed_barriers_committed") is True for item in workers)
        ),
        "all_central_restores_verified": bool(
            workers and all(item.get("central_checkpoint_restore_verified") is True for item in workers)
        ),
        "all_base_weights_frozen": bool(
            workers and all(item.get("base_weights_frozen") is True for item in workers)
        ),
        "all_positive_lora_gradients": bool(
            workers and all(item.get("positive_lora_gradient_norms") is True for item in workers)
        ),
        "all_gracefully_drained": bool(
            workers and all(item.get("graceful_drain_applied") is True for item in workers)
        ),
        "standard_peft_export_verified": any(
            item.get("standard_peft_export_verified") is True for item in workers
        ),
        "evaluation_verified": any(item.get("evaluation_verified") is True for item in workers),
        "all_kernels_deleted": generation.get("all_kernels_deleted") is True,
        "kernel_ref_hashes": list(generation.get("kernel_ref_hashes") or []),
        "public_artifact_safe": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token-file", action="append", default=[])
    parser.add_argument("--raw-token-file", default="")
    parser.add_argument("--raw-token-username", default="")
    parser.add_argument(
        "--tokenized-payload",
        default="dist/training-qwen15b-dataset-20260712-r1/qwen15b_tokenized_private.json",
    )
    parser.add_argument(
        "--source-manifest",
        default="dist/training-qwen15b-source-20260712-r1/qwen15b_source_manifest.json",
    )
    parser.add_argument("--allocation-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--push-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--output-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0)
    parser.add_argument("--pause-delay-seconds", type=float, default=10.0)
    parser.add_argument("--service-restart-delay-seconds", type=float, default=3.0)
    parser.add_argument("--tunnel-attempts", type=int, default=3)
    parser.add_argument("--route-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not 5.0 <= args.pause_delay_seconds <= 120.0:
        parser.error("--pause-delay-seconds must be in [5, 120]")
    if not 1.0 <= args.allocation_timeout_seconds <= 1800.0:
        parser.error("--allocation-timeout-seconds must be in (0, 1800]")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    private = output / ".private-runtime"
    private.mkdir(parents=True, exist_ok=True)
    private.chmod(0o700)
    report_path = output / "training_elastic_beta_live_probe.json"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "elastic_training_beta_ready": False,
        "live_run_performed": not args.preflight_only,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "target_steps": 8,
        "ordinary_user_cli": {
            "create": False,
            "status": False,
            "export": False,
        },
        "blockers": [],
        "started_at": utc_now(),
        "cleanup": {
            "all_kernels_deleted": False,
            "service_stopped": False,
            "tunnel_stopped": False,
            "rendezvous_payloads_removed": False,
            "private_runtime_removed": False,
            "live_resources_left_running": True,
        },
        "credentials_public": False,
        "credential_paths_public": False,
        "coordinator_url_public": False,
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_tensor_values_public": False,
        "adapter_tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    service: ProductService | None = None
    tunnel_process = None
    selected_env: dict[str, str] = {}
    cleanup_refs: list[str] = []
    old_generation: dict[str, Any] = {}
    new_generation: dict[str, Any] = {}
    job_dir = private / "job"
    try:
        sections = _credential_sections(
            list(args.token_file or []),
            raw_token_file=str(args.raw_token_file),
            raw_token_username=str(args.raw_token_username),
        )
        preflight, candidates = preflight_accounts(sections)
        report["account_preflight"] = preflight
        report["eligible_account_count"] = len(candidates)
        if args.preflight_only:
            report["blockers"].append("elastic_training_beta_preflight_only")
            raise _PreflightComplete
        if not candidates:
            raise RuntimeError("elastic_training_beta_same_account_t4x2_pair_unavailable")
        selected = candidates[0]
        selected_env = dict(selected["env_values"])
        owner = str(selected["owner"])
        report["selected_account"] = {
            "owner_hash": str(selected["owner_hash"]),
            "effective_remaining_seconds": float(selected["effective_remaining"]),
            "credential_values_public": False,
        }
        source_manifest = json.loads(
            Path(args.source_manifest).read_text(encoding="utf-8")
        )
        config_bytes = fetch_bytes(_hf_url(MODEL_ID, MODEL_REVISION, "config.json"))
        if sha256_bytes(config_bytes) != source_manifest.get("config_hash"):
            raise RuntimeError("elastic_training_beta_config_hash_mismatch")
        config = json.loads(config_bytes)
        config_path = private / "config.json"
        config_path.write_bytes(config_bytes)
        config_path.chmod(0o600)
        tokenized = Path(args.tokenized_payload).resolve()
        if not tokenized.is_file():
            raise RuntimeError("elastic_training_beta_tokenized_payload_missing")
        create_cli = _run_cli(
            [
                "train",
                "create",
                str(job_dir),
                "--config",
                str(config_path),
                "--tokenized-payload",
                str(tokenized),
            ],
            timeout=300.0,
        )
        report["ordinary_user_cli"]["create"] = bool(
            create_cli["returncode"] == 0
            and create_cli["report"].get("overall_state") == "waiting_for_miners"
            and create_cli["report"].get("command_ok") is True
        )
        report["create_cli"] = {
            "returncode": create_cli["returncode"],
            "report_hash": stable_hash(create_cli["report"]),
            "raw_output_public": False,
        }
        if not report["ordinary_user_cli"]["create"]:
            raise RuntimeError("elastic_training_beta_public_create_failed")
        port = _free_port()
        service = ProductService(job_dir=job_dir, port=port)
        service.start()
        local_url = f"http://127.0.0.1:{port}"
        cloudflared = ensure_cloudflared(private)
        tunnel_process, tunnel_url, route = start_verified_tunnel(
            cloudflared,
            local_url,
            private,
            token=service.miner_token,
            run_id=service.controller.run_id,
            attempts=int(args.tunnel_attempts),
            timeout=float(args.route_timeout_seconds),
        )
        capabilities = _get_json(
            f"{tunnel_url}/elastic-training/capabilities",
            token=service.miner_token,
        )
        report["route_preflight"] = {
            **route,
            "capabilities_verified": capabilities.get("automatic_role_assignment")
            is True,
            "checkpoint_signatures_required": capabilities.get(
                "checkpoint_signatures_required"
            )
            is True,
            "checkpoint_tensor_validation_required": capabilities.get(
                "checkpoint_tensor_validation_required"
            )
            is True,
        }
        suffix = f"{str(int(time.time()))[-8:]}-{secrets.token_hex(2)}"
        with tempfile.TemporaryDirectory(prefix="ct-elastic-beta-account-") as config_dir:
            from scripts.kaggle_gpu_token_weekly_quota_probe import clean_env

            env = clean_env(selected_env, config_dir=Path(config_dir))
            old_generation, _old_refs = _launch_generation(
                generation="old-product-miners",
                expected_start_step=0,
                segment_end_step=4,
                owner=owner,
                suffix=suffix,
                config=config,
                tokenized=tokenized,
                coordinator_url=tunnel_url,
                coordinator_token=service.miner_token,
                run_id=service.controller.run_id,
                env=env,
                private=private,
                output=output,
                allocation_timeout=float(args.allocation_timeout_seconds),
                push_timeout=float(args.push_timeout_seconds),
                status_timeout=float(args.status_timeout_seconds),
                output_timeout=float(args.output_timeout_seconds),
                delete_timeout=float(args.delete_timeout_seconds),
                poll_interval=float(args.poll_interval_seconds),
                cleanup_registry=cleanup_refs,
                product_miner_mode=True,
                product_role="auto",
                max_steps_per_session=4,
                require_worker_adapter=False,
                parallel_pushes=False,
            )
            report["old_generation"] = old_generation
            midpoint = service.controller.status()
            pause_observations = []
            deadline = time.monotonic() + float(args.pause_delay_seconds)
            while time.monotonic() < deadline:
                value = service.controller.status()
                pause_observations.append(
                    {
                        "overall_state": value["overall_state"],
                        "committed_step": value["global_step"],
                        "live_miner_count": value["runtime"]["live_miner_count"],
                    }
                )
                time.sleep(1.0)
            report["midpoint"] = {
                "committed_step": midpoint["global_step"],
                "overall_state": midpoint["overall_state"],
                "zero_live_miners": midpoint["runtime"]["zero_live_miners"],
                "observation_count": len(pause_observations),
                "all_observations_paused": bool(
                    pause_observations
                    and all(
                        item["overall_state"] == "waiting_for_miners"
                        and item["committed_step"] == 4
                        and item["live_miner_count"] == 0
                        for item in pause_observations
                    )
                ),
            }
            stopped_for_restart = service.stop()
            time.sleep(float(args.service_restart_delay_seconds))
            service = ProductService(job_dir=job_dir, port=port)
            service.start()
            recovered = service.controller.status()
            report["service_restart"] = {
                "old_service_stopped": stopped_for_restart,
                "same_job_id": recovered["job_id"] == midpoint["job_id"],
                "committed_step_recovered": recovered["global_step"] == 4,
                "runtime_paused_recovered": recovered["overall_state"]
                == "waiting_for_miners",
                "rendezvous_recovered": recovered["rendezvous"].get(
                    "recovered_from_persistent_state"
                )
                is True,
                "restart_recorded": bool(
                    recovered["rendezvous"].get("coordinator_restarts")
                ),
            }
            _wait_ready(local_url, timeout=30.0)
            new_generation, _new_refs = _launch_generation(
                generation="replacement-product-miners",
                expected_start_step=4,
                segment_end_step=8,
                owner=owner,
                suffix=suffix,
                config=config,
                tokenized=tokenized,
                coordinator_url=tunnel_url,
                coordinator_token=service.miner_token,
                run_id=service.controller.run_id,
                env=env,
                private=private,
                output=output,
                allocation_timeout=float(args.allocation_timeout_seconds),
                push_timeout=float(args.push_timeout_seconds),
                status_timeout=float(args.status_timeout_seconds),
                output_timeout=float(args.output_timeout_seconds),
                delete_timeout=float(args.delete_timeout_seconds),
                poll_interval=float(args.poll_interval_seconds),
                cleanup_registry=cleanup_refs,
                product_miner_mode=True,
                product_role="auto",
                max_steps_per_session=0,
                require_worker_adapter=False,
                parallel_pushes=False,
            )
            report["replacement_generation"] = new_generation
        final = service.controller.status()
        report["final_status"] = final
        status_cli = _run_cli(["train", "status", str(job_dir)], timeout=120.0)
        report["ordinary_user_cli"]["status"] = bool(
            status_cli["returncode"] == 0
            and status_cli["report"].get("overall_state") == "completed"
            and status_cli["report"].get("global_step") == 8
        )
        export_cli = _run_cli(
            [
                "train",
                "export",
                str(job_dir),
                "--output-dir",
                str(output / "exported_adapter"),
            ],
            timeout=600.0,
        )
        report["ordinary_user_cli"]["export"] = bool(
            export_cli["returncode"] == 0
            and export_cli["report"].get("standard_peft_format") is True
            and export_cli["report"].get("ok") is True
        )
        report["status_cli_report_hash"] = stable_hash(status_cli["report"])
        report["export_cli_report"] = export_cli["report"]
        old_summary = _product_generation_summary(old_generation)
        new_summary = _product_generation_summary(new_generation)
        report["old_product_summary"] = old_summary
        report["replacement_product_summary"] = new_summary
        gates = {
            "ordinary_user_create_status_export_verified": all(
                report["ordinary_user_cli"].values()
            ),
            "old_product_miners_verified": old_summary["ok"]
            and old_summary["product_miner_mode"]
            and old_summary["roles"] == ["kernel_a", "kernel_b"]
            and old_summary["start_steps"] == [0, 0]
            and old_summary["end_steps"] == [4, 4]
            and old_summary["barrier_commit_counts"] == [4, 4]
            and old_summary["all_completed_barriers_committed"]
            and old_summary["all_gracefully_drained"],
            "full_offline_pause_verified": report["midpoint"]["committed_step"] == 4
            and report["midpoint"]["zero_live_miners"]
            and report["midpoint"]["all_observations_paused"],
            "coordinator_restart_verified": all(report["service_restart"].values()),
            "replacement_product_miners_verified": new_summary["ok"]
            and new_summary["product_miner_mode"]
            and new_summary["roles"] == ["kernel_a", "kernel_b"]
            and new_summary["start_steps"] == [4, 4]
            and new_summary["end_steps"] == [8, 8]
            and new_summary["barrier_commit_counts"] == [4, 4]
            and new_summary["all_completed_barriers_committed"]
            and new_summary["all_central_restores_verified"],
            "real_training_semantics_verified": old_summary["all_base_weights_frozen"]
            and old_summary["all_positive_lora_gradients"]
            and new_summary["all_base_weights_frozen"]
            and new_summary["all_positive_lora_gradients"]
            and new_summary["standard_peft_export_verified"]
            and new_summary["evaluation_verified"],
            "exactly_once_eight_steps_verified": final["overall_state"] == "completed"
            and final["global_step"] == 8
            and final["runtime"]["committed_steps"] == list(range(1, 9))
            and final["runtime"]["optimizer_commit_count"] == 8
            and final["runtime"]["committed_steps_contiguous"] is True,
            "secure_checkpoint_path_verified": final[
                "checkpoint_signature_verification_ready"
            ]
            and final["checkpoint_tensor_validation_ready"]
            and final["runtime"]["checkpoint_signatures_required"]
            and final["runtime"]["checkpoint_tensor_validation_required"],
            "all_kernels_deleted": old_summary["all_kernels_deleted"]
            and new_summary["all_kernels_deleted"],
            "replacement_kernel_identity_distinct": bool(
                set(old_summary["kernel_ref_hashes"])
                and set(new_summary["kernel_ref_hashes"])
                and set(old_summary["kernel_ref_hashes"])
                .isdisjoint(new_summary["kernel_ref_hashes"])
            ),
        }
        report["acceptance_gates"] = gates
        report["elastic_training_beta_ready"] = all(gates.values())
        report["ok"] = report["elastic_training_beta_ready"]
        if not report["ok"]:
            report["blockers"].append("elastic_training_beta_live_acceptance_incomplete")
        return_code = 0 if report["ok"] else 1
    except _PreflightComplete:
        return_code = 1
    except BaseException as exc:
        safe_code = str(exc).split(":", 1)[0]
        if not safe_code.startswith(("elastic_", "qwen15b_")):
            safe_code = type(exc).__name__
        report["blockers"].append(
            f"elastic_training_beta_live_failed:{safe_code[:160]}"
        )
        report["failure_detail_public"] = False
        return_code = 1
    finally:
        if cleanup_refs and selected_env:
            try:
                with tempfile.TemporaryDirectory(prefix="ct-elastic-beta-cleanup-") as config_dir:
                    from scripts.kaggle_gpu_token_weekly_quota_probe import clean_env

                    env = clean_env(selected_env, config_dir=Path(config_dir))
                    deletion_reports, deleted = _delete_refs(
                        cleanup_refs,
                        env=env,
                        timeout=float(args.delete_timeout_seconds),
                        role_by_ref={ref: "" for ref in cleanup_refs},
                    )
                    report["final_delete_audit"] = deletion_reports
                    generation_cleanup_verified = bool(
                        old_generation.get("all_kernels_deleted") is True
                        and new_generation.get("all_kernels_deleted") is True
                    )
                    report["cleanup"]["all_kernels_deleted"] = bool(
                        deleted or generation_cleanup_verified
                    )
                    report["final_delete_audit_redundant"] = bool(
                        generation_cleanup_verified
                    )
            except BaseException:
                report["cleanup"]["all_kernels_deleted"] = False
        else:
            report["cleanup"]["all_kernels_deleted"] = True
        if service is not None:
            try:
                report["rendezvous_cleanup"] = service.controller.rendezvous.cleanup()
                report["cleanup"]["rendezvous_payloads_removed"] = bool(
                    report["rendezvous_cleanup"].get("private_payloads_removed")
                )
            except BaseException:
                pass
            report["cleanup"]["service_stopped"] = service.stop()
        else:
            report["cleanup"]["service_stopped"] = True
            report["cleanup"]["rendezvous_payloads_removed"] = True
        report["cleanup"]["tunnel_stopped"] = stop_process(tunnel_process)
        try:
            shutil.rmtree(private, ignore_errors=True)
            report["cleanup"]["private_runtime_removed"] = not private.exists()
        except BaseException:
            report["cleanup"]["private_runtime_removed"] = False
        report["cleanup"]["live_resources_left_running"] = not all(
            bool(report["cleanup"].get(key))
            for key in (
                "all_kernels_deleted",
                "service_stopped",
                "tunnel_stopped",
                "rendezvous_payloads_removed",
                "private_runtime_removed",
            )
        )
        report["cleanup_verified"] = not report["cleanup"][
            "live_resources_left_running"
        ]
        safety = public_safety_errors(report)
        report["public_safety_errors"] = safety
        if safety:
            report["ok"] = False
            report["elastic_training_beta_ready"] = False
            report["blockers"].append("elastic_training_beta_public_safety_failed")
            return_code = 1
        if not report.get("cleanup_verified"):
            report["ok"] = False
            report["elastic_training_beta_ready"] = False
            report["blockers"].append("elastic_training_beta_cleanup_incomplete")
            return_code = 1
        report["finished_at"] = utc_now()
        _write(report_path, report)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(
                "training_elastic_beta_live_probe "
                f"ok={report.get('ok')} ready={report.get('elastic_training_beta_ready')}"
            )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
