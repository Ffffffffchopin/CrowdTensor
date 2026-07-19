#!/usr/bin/env python3
"""Assemble the canonical Qwen 1.5B four-GPU Training Alpha artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from crowdtensor.qwen15b_training import MODEL_ID, MODEL_REVISION, sha256_file  # noqa: E402
from scripts.training_qwen15b_four_gpu_alpha_check import SCHEMA, check  # noqa: E402


def _load(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    if not source.is_file():
        return {}
    value = json.loads(source.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _artifact(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).is_file():
        return {"present": False, "file_hash": ""}
    source = Path(path)
    return {
        "present": True,
        "file_name": source.name,
        "file_hash": sha256_file(source),
        "byte_count": source.stat().st_size,
    }


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _allocation_history(ledger: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    attempts = list(ledger.get("qwen15b_four_gpu_attempts") or [])
    attempt_numbers = [int(item.get("attempt") or 0) for item in attempts]
    completed = [item for item in attempts if item.get("completed") is True]
    verified = [item for item in attempts if item.get("outcome") == "verified"]
    live_attempt = int(live.get("attempt") or 0)
    return {
        "schema": "crowdtensor_qwen15b_four_gpu_alpha_allocation_history_v1",
        "ledger_present": bool(ledger),
        "attempt_count": len(attempts),
        "completed_attempt_count": len(completed),
        "attempt_numbers_sequential": attempt_numbers == list(range(1, len(attempts) + 1)),
        "attempt_records_hash": _stable_hash(attempts) if attempts else "",
        "successful_attempt": live_attempt if live.get("qwen15b_four_gpu_alpha_verified") else 0,
        "verified_attempt_numbers": [int(item.get("attempt") or 0) for item in verified],
        "successful_attempt_matches_ledger": bool(
            live_attempt
            and any(
                int(item.get("attempt") or 0) == live_attempt
                and item.get("completed") is True
                and item.get("outcome") == "verified"
                for item in attempts
            )
        ),
        "immutable_history_preserved": bool(
            attempts
            and _dict(live.get("allocation_budget")).get("prior_attempts_preserved") is True
        ),
        "public_artifact_safe": True,
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _precision_path_verified(live: dict[str, Any]) -> bool:
    workers = list(live.get("worker_reports") or [])
    if len(workers) != 2:
        return False
    for outer in workers:
        smoke = _dict(outer.get("cuda_mixed_precision_smoke"))
        if any(
            smoke.get(key) is not True
            for key in (
                "verified",
                "cuda_live",
                "fp32_lora_parameters",
                "fp32_stable_compute",
                "fp16_stage_boundary",
                "grad_scaler_unscale_step_verified",
            )
        ):
            return False
        ready = _dict(_dict(outer.get("worker")).get("stage_ready")).get("baseline") or []
        if len(ready) != 2:
            return False
        for item in ready:
            load = _dict(item.get("load_report"))
            if (
                load.get("trainable_parameter_dtypes") != ["float32"]
                or load.get("fp32_lora_parameters_for_grad_scaler") is not True
                or load.get("cuda_fp16_autocast") is not False
                or load.get("cuda_fp32_stable_compute") is not True
                or load.get("stage_boundary_dtype") != "float16"
            ):
                return False
    return True


def pack(
    output_dir: str | Path,
    *,
    source_report: str | Path,
    dataset_report: str | Path,
    test_summary: str | Path,
    live_report: str | Path | None = None,
    allocation_ledger: str | Path | None = None,
    precision_failure_report: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = _load(source_report)
    dataset = _load(dataset_report)
    tests = _load(test_summary)
    live = _load(live_report)
    ledger = _load(allocation_ledger)
    precision_failure = _load(precision_failure_report)
    live_ready = bool(live.get("qwen15b_four_gpu_alpha_verified") and live.get("ok"))
    live_budget = _dict(live.get("allocation_budget"))
    unbounded_attempts = bool(
        live_budget.get("amendment_valid") is True
        and live_budget.get("total_attempt_limit_unbounded") is True
        and live_budget.get("effective_attempt_limit") is None
    )
    precision_path_verified = _precision_path_verified(live)
    precision_failure_text = json.dumps(precision_failure, sort_keys=True).lower()
    non_finite_fp16_activation_observed = "non_finite_stage_activation" in precision_failure_text
    live_log_text = json.dumps(live.get("kernel_logs") or {}, sort_keys=True).lower()
    torchao_pre_016_observed = bool(
        "torchao 0.10.0" in live_log_text
        or any(
            _dict(worker.get("dependencies")).get("torchao_before") == "0.10.0"
            for worker in live.get("worker_reports") or []
        )
    )
    blockers = list(live.get("blockers") or [])
    for worker in live.get("worker_reports") or []:
        blockers.extend(str(value) for value in worker.get("blockers") or [] if str(value))
    if "incompatible version of torchao" in json.dumps(live.get("kernel_logs") or {}).lower():
        blockers.append("qwen15b_kaggle_incompatible_torchao_pre_0_16")
    if not live:
        blockers.append("qwen15b_four_gpu_live_attempt_not_run")
    elif not live_ready and not blockers:
        blockers.append("qwen15b_four_gpu_live_acceptance_incomplete")
    if (
        live
        and not live_ready
        and not unbounded_attempts
        and int(live.get("attempt") or 0) >= int(live.get("attempt_limit") or 2)
    ):
        blockers.append("qwen15b_four_gpu_allocation_attempt_budget_exhausted")
    phases = {
        "model_source": {
            "state": "completed" if source.get("ok") else "blocked",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        },
        "dataset": {
            "state": "completed" if dataset.get("ok") else "blocked",
            "token_ids_public": False,
        },
        "stage_loading": {
            "state": "completed" if live_ready else "pending" if not live else "blocked",
        },
        "four_gpu_forward": {
            "state": "completed"
            if (live.get("evidence") or {}).get("four_stage_compute_overlap_verified")
            else "pending" if not live else "blocked",
        },
        "four_gpu_backward": {
            "state": "completed"
            if (live.get("evidence") or {}).get("gradient_payload_count") == 64
            else "pending" if not live else "blocked",
        },
        "checkpoint_resume": {
            "state": "completed"
            if (live.get("evidence") or {}).get("controlled_restart_verified")
            else "pending" if not live else "blocked",
        },
        "evaluation": {
            "state": "completed"
            if (live.get("evidence") or {}).get("evaluation_verified")
            else "pending" if not live else "blocked",
        },
        "export": {
            "state": "completed"
            if (live.get("evidence") or {}).get("standard_peft_export_verified")
            else "pending" if not live else "blocked",
        },
        "cleanup": {
            "state": "completed"
            if all(
                (live.get("cleanup") or {}).get(key) is True
                for key in (
                    "kernels_deleted",
                    "private_packages_removed",
                    "coordinator_stopped",
                    "tunnel_stopped",
                    "private_runtime_removed",
                )
            )
            else "pending" if not live else "blocked",
        },
    }
    report = {
        "schema": SCHEMA,
        "goal": "CrowdTensor Qwen 1.5B Four-GPU Pipeline Training Alpha",
        "goal_achieved": False,
        "qwen15b_four_gpu_alpha_ready": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "topology": "kaggle-2x-t4x2",
        "steps_per_run": 8,
        "run_kinds": ["baseline", "resumed"],
        "authoritative_cuda_training_rc": (
            "dist/training-cuda-two-node-rc-20260711-r5-live-achieved/"
            "training_cuda_two_node_rc.json"
        ),
        "authoritative_cuda_training_rc_reused_without_rewrite": True,
        "source": source,
        "dataset": dataset,
        "test_summary": tests,
        "live_report": live,
        "phase_status": phases,
        "blockers": sorted(set(blockers)),
        "runtime_remediation": {
            "incompatible_torchao_pre_0_16_observed": torchao_pre_016_observed,
            "kaggle_observed_torchao_version": "0.10.0" if torchao_pre_016_observed else "",
            "private_package_uninstalls_torchao_below_0_16": True,
            "dependency_lora_smoke_before_stage_materialization": True,
            "dependency_smoke_exercises_peft_forward_backward": True,
            "fp16_autocast_non_finite_activation_observed": non_finite_fp16_activation_observed,
            "fp16_autocast_abandoned": non_finite_fp16_activation_observed,
            "frozen_stage_weight_compute_dtype": "float32",
            "lora_parameter_dtype": "float32",
            "cuda_fp16_autocast": False,
            "cuda_fp32_stable_compute": precision_path_verified,
            "fp16_stage_boundary_transport": precision_path_verified,
            "grad_scaler_unscale_step_verified": precision_path_verified,
            "non_finite_activation_logits_loss_gradient_gates": True,
            "remediation_local_tests_passed": tests.get("ok") is True,
            "remediation_gpu_live_verified": False if not live_ready else True,
        },
        "engineering_hardening": {
            "fixed_dataset_rows_consumed_per_run": 32,
            "dataset_row_order_public_indexes_only": True,
            "strict_checker_recomputes_per_step_stage_records": True,
            "strict_checker_recomputes_payload_identity_and_hash_links": True,
            "strict_checker_recomputes_four_stage_overlap": True,
            "strict_checker_verifies_restart_pid_chronology": True,
            "strict_checker_verifies_checkpoint_manifest_summaries": True,
            "checkpoint_manifest_content_hashes_recomputed": True,
            "adapter_safetensors_header_and_layer_coverage_verified": True,
            "tensor_values_public": False,
            "token_ids_public": False,
            "private_paths_public": False,
        },
        "allocation_budget": {
            **live_budget,
            "attempts_used": int(live.get("attempt") or 0),
            "successful_attempt": int(live.get("attempt") or 0) if live_ready else 0,
            "probe_invocation_attempt_ceiling": int(live.get("attempt_limit") or 0),
            "budget_exhausted": bool(
                live
                and not live_ready
                and not unbounded_attempts
                and int(live.get("attempt") or 0) >= int(live.get("attempt_limit") or 2)
            ),
            "ledger_history_must_be_preserved": True,
            "additional_attempt_requires_explicit_user_amendment": not unbounded_attempts,
            "probe_invocation_ceiling_is_not_total_policy_limit": unbounded_attempts,
        },
        "allocation_history": _allocation_history(ledger, live),
        "artifacts": {
            "source_report": _artifact(source_report),
            "dataset_report": _artifact(dataset_report),
            "test_summary": _artifact(test_summary),
            "live_report": _artifact(live_report),
            "allocation_ledger": _artifact(allocation_ledger),
            "precision_failure_report": _artifact(precision_failure_report),
        },
        "commands": {
            "train": (
                "crowdtensor train lora --backend cuda --model Qwen/Qwen2.5-1.5B "
                "--topology kaggle-2x-t4x2 --steps 8"
            ),
            "status": "crowdtensor train status <job>",
            "resume": "crowdtensor train resume <job>",
            "export": "crowdtensor train export <job>",
            "cleanup": "crowdtensor train cleanup <job>",
        },
        "activation_values_public": False,
        "gradient_values_public": False,
        "adapter_tensor_values_public": False,
        "token_ids_public": False,
        "raw_training_text_public": False,
        "credentials_public": False,
        "coordinator_url_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    first = check(report)
    report["goal_achieved"] = bool(first["qwen15b_four_gpu_alpha_ready"])
    report["qwen15b_four_gpu_alpha_ready"] = report["goal_achieved"]
    report["blockers"] = [] if report["goal_achieved"] else report["blockers"]
    final = check(report)
    report["checker"] = final
    destination = output / "training_qwen15b_four_gpu_alpha.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--dataset-report", required=True)
    parser.add_argument("--test-summary", required=True)
    parser.add_argument("--live-report", default="")
    parser.add_argument("--allocation-ledger", default="")
    parser.add_argument("--precision-failure-report", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = pack(
        args.output_dir,
        source_report=args.source_report,
        dataset_report=args.dataset_report,
        test_summary=args.test_summary,
        live_report=args.live_report or None,
        allocation_ledger=args.allocation_ledger or None,
        precision_failure_report=args.precision_failure_report or None,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"training_qwen15b_four_gpu_alpha ready={report['goal_achieved']} "
            f"blockers={','.join(report['blockers']) or 'none'}"
        )
    return 0 if report["checker"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
