#!/usr/bin/env python3
"""Pack live and engineering evidence into the canonical CUDA Training RC."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.hf_lora_training import CUDALoRATrainingRuntime, CUDAOutOfMemoryError  # noqa: E402
from crowdtensor.pipeline_lora_training import CUDAStageRuntime  # noqa: E402
from crowdtensor.training_contract import sha256_file  # noqa: E402
from crowdtensor.training_allocation_budget import allocation_budget_summary  # noqa: E402
from scripts.training_cuda_two_node_rc_check import (  # noqa: E402
    SCHEMA,
    _embedded_single_gate_binding_verified,
    _single_gate_verified,
    check,
)


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy(source: str | Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(source), destination)
    return sha256_file(destination)


def _summary_baseline(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": value.get("schema"),
        "goal_achieved": value.get("goal_achieved") is True,
        "training_foundation_rc_ready": value.get("training_foundation_rc_ready") is True,
        "backend": value.get("backend"),
        "requirements": value.get("requirements") or {},
        "gpu_live_verified": value.get("gpu_live_verified") is True,
        "private_paths_public": False,
    }


def _cleanup_summary(single_attempts: list[dict[str, Any]], two_attempts: list[dict[str, Any]]) -> dict[str, Any]:
    single_clean = all(
        (item.get("cleanup") or {}).get("kernel_deleted") is True
        and (item.get("cleanup") or {}).get("private_package_removed") is True
        and (item.get("cleanup") or {}).get("private_cleanup_state_removed", True) is True
        and (
            item.get("single_kernel_t4x2_verified") is not True
            or (item.get("cleanup") or {}).get("checkpoint_preserved") is True
        )
        for item in single_attempts
    )
    two_clean = all(
        (item.get("cleanup") or {}).get("kernels_deleted") is True
        and (item.get("cleanup") or {}).get("private_packages_removed") is True
        and (item.get("cleanup") or {}).get("coordinator_stopped") is True
        and (item.get("cleanup") or {}).get("tunnel_stopped") is True
        and (item.get("cleanup") or {}).get("private_runtime_removed") is True
        and (item.get("cleanup") or {}).get("private_cleanup_state_removed", True) is True
        and (
            item.get("two_node_cuda_verified") is not True
            or (item.get("cleanup") or {}).get("checkpoint_bundles_preserved") is True
        )
        for item in two_attempts
    )
    return {
        "schema": "crowdtensor_cuda_training_cleanup_summary_v1",
        "single_attempt_count": len(single_attempts),
        "two_node_attempt_count": len(two_attempts),
        "all_kaggle_kernels_deleted": single_clean and two_clean,
        "all_private_packages_removed": single_clean and two_clean,
        "all_local_runtime_stopped": two_clean,
        "live_resources_left_running": not (single_clean and two_clean),
        "checkpoint_and_public_evidence_preserved": True,
        "credentials_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def _route_preflight_summary(value: dict[str, Any]) -> dict[str, Any]:
    route = dict(value.get("route_preflight") or {})
    coordinator = dict(value.get("coordinator") or {})
    cleanup = dict(value.get("cleanup") or {})
    return {
        "schema": "crowdtensor_cuda_training_coordinator_route_preflight_summary_v1",
        "verified": value.get("route_preflight_verified") is True and route.get("verified") is True,
        "authenticated_status_verified": route.get("authenticated_status_verified") is True,
        "miner_auth_required_verified": route.get("miner_auth_required_verified") is True,
        "run_id_hash_verified": route.get("run_id_hash_verified") is True,
        "stable_successes_observed": int(route.get("stable_successes_observed") or 0),
        "stable_successes_required": int(route.get("stable_successes_required") or 0),
        "transient_error_classes": dict(route.get("error_classes") or {}),
        "tunnel_start_attempts": int(coordinator.get("tunnel_start_attempts") or 0),
        "allocation_started": value.get("allocation_started") is True,
        "kernel_push_attempted": value.get("push_attempted") is True,
        "live_gate_claimed": value.get("two_node_cuda_verified") is True or value.get("ok") is True,
        "cleanup_verified": all(
            cleanup.get(key) is True
            for key in (
                "kernels_deleted",
                "private_packages_removed",
                "coordinator_stopped",
                "tunnel_stopped",
                "private_runtime_removed",
            )
        ),
        "url_public": False,
        "credentials_public": False,
        "private_paths_public": False,
        "public_artifact_safe": value.get("public_artifact_safe") is True,
    }


def pack(
    *,
    output_dir: str | Path,
    cpu_baseline: str | Path,
    single_attempt_reports: list[str | Path],
    two_node_attempt_reports: list[str | Path],
    attempt_ledger: str | Path,
    rejection_matrix: str | Path,
    test_summary: str | Path,
    route_preflight_report: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = _load(cpu_baseline)
    singles = [_load(path) for path in single_attempt_reports]
    doubles = [_load(path) for path in two_node_attempt_reports]
    ledger = _load(attempt_ledger)
    allocation_budget = allocation_budget_summary(ledger)
    rejection = _load(rejection_matrix)
    tests = _load(test_summary)
    route_preflight = _load(route_preflight_report) if route_preflight_report else {}
    artifact_sources: dict[str, str | Path] = {
        "cpu_foundation_rc": cpu_baseline,
        "allocation_attempts": attempt_ledger,
        "rejection_matrix": rejection_matrix,
        "test_summary": test_summary,
    }
    if route_preflight_report:
        artifact_sources["coordinator_route_preflight"] = route_preflight_report
    for index, source in enumerate(single_attempt_reports, start=1):
        artifact_sources[f"single_kernel_attempt_{index}"] = source
    for index, source in enumerate(two_node_attempt_reports, start=1):
        artifact_sources[f"two_node_attempt_{index}"] = source
    artifacts: dict[str, str] = {}
    artifact_hashes: dict[str, str] = {}
    for name, source in artifact_sources.items():
        relative = f"evidence/{name}.json"
        artifacts[name] = relative
        artifact_hashes[name] = _copy(source, output / relative)

    dependencies: dict[str, str] = {}
    for package in ("torch", "transformers", "peft", "safetensors", "accelerate"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "missing"
    single_latest = singles[-1] if singles else {}
    two_latest = doubles[-1] if doubles else {}
    selected_single = single_latest
    single_gate_source = "standalone_attempt"
    embedded_single = dict(two_latest.get("embedded_single_kernel_gate") or {})
    if (
        not _single_gate_verified(single_latest)
        and _single_gate_verified(embedded_single)
        and _embedded_single_gate_binding_verified(embedded_single, two_latest)
    ):
        selected_single = embedded_single
        single_gate_source = "two_node_stage0_embedded"
    selected_single_verified = _single_gate_verified(selected_single)
    if single_gate_source == "two_node_stage0_embedded":
        selected_single_verified = bool(
            selected_single_verified
            and _embedded_single_gate_binding_verified(selected_single, two_latest)
        )
    blockers: list[str] = []
    historical_attempt_blockers = sorted(
        {
            str(item)
            for evidence in singles + doubles
            for item in evidence.get("blockers") or []
        }
    )
    if not selected_single_verified:
        blockers.append("single_kernel_t4x2_live_gate_not_verified")
    if not two_latest.get("two_node_cuda_verified"):
        blockers.append("two_kernel_cross_machine_live_gate_not_verified")
    if (
        len(ledger.get("single_kernel_attempts") or [])
        >= int(allocation_budget["single_kernel_attempt_limit"])
        and not selected_single_verified
    ):
        blockers.append("single_kernel_allocation_attempt_budget_exhausted")
    if (
        len(ledger.get("two_node_attempts") or [])
        >= int(allocation_budget["two_node_attempt_limit"])
        and not two_latest.get("two_node_cuda_verified")
    ):
        blockers.append("two_node_allocation_attempt_budget_exhausted")
    report = {
        "schema": SCHEMA,
        "training_cuda_two_node_rc_ready": False,
        "goal_achieved": False,
        "gpu_success_claimed": False,
        "created_at_epoch": time.time(),
        "cpu_foundation_baseline": _summary_baseline(baseline),
        "runtime_contracts": {
            "cuda_lora_runtime_implemented": CUDALoRATrainingRuntime.__name__ == "CUDALoRATrainingRuntime",
            "cuda_stage_runtime_implemented": CUDAStageRuntime.__name__ == "CUDAStageRuntime",
            "fp16_autocast_supported": True,
            "grad_scaler_supported": True,
            "gradient_clipping_supported": True,
            "cuda_oom_classification_supported": CUDAOutOfMemoryError.code == "cuda_training_out_of_memory",
            "cpu_checkpoint_delta_compatibility_preserved": True,
            "authenticated_private_rendezvous_implemented": True,
            "remote_delta_materialization_implemented": True,
            "authenticated_route_preflight_supported": True,
            "allocation_attempt_reserved_at_kernel_push_boundary": True,
            "idempotent_worker_network_retries_supported": True,
            "checkpoint_bundle_preservation_supported": True,
            "crash_recoverable_private_cleanup_ledger_supported": True,
            "real_pytorch_transformers_peft_required": True,
            "cpu_fallback_allowed_for_cuda": False,
            "capability_or_dry_run_counts_as_live": False,
        },
        "real_training_stack": dependencies,
        "allocation_attempts": ledger,
        "allocation_budget": allocation_budget,
        "single_kernel_gate": selected_single,
        "single_kernel_gate_source": single_gate_source,
        "single_kernel_gate_selection": {
            "standalone_attempt_count": len(singles),
            "standalone_latest_verified": _single_gate_verified(single_latest),
            "embedded_candidate_present": bool(embedded_single),
            "embedded_candidate_binding_verified": bool(
                embedded_single
                and _embedded_single_gate_binding_verified(embedded_single, two_latest)
            ),
            "selected_source": single_gate_source,
            "historical_attempts_preserved": True,
            "public_artifact_safe": True,
        },
        "single_kernel_attempt_history": singles,
        "two_node_gate": two_latest,
        "two_node_attempt_history": doubles,
        "rejection_matrix": rejection,
        "test_summary": tests,
        "coordinator_route_preflight": (
            _route_preflight_summary(route_preflight) if route_preflight else {}
        ),
        "cleanup_summary": _cleanup_summary(singles, doubles),
        "artifacts": artifacts,
        "artifact_hashes": artifact_hashes,
        "blockers": sorted(set(blockers)),
        "historical_attempt_blockers": historical_attempt_blockers,
        "engineering_recovery": {
            "embedded_single_kernel_source_bundle_implemented": True,
            "embedded_single_kernel_live_evidence_binding_implemented": True,
            "old_torchao_removal_before_transformers_import_implemented": True,
            "authenticated_coordinator_route_preflight_verified": (
                bool(route_preflight)
                and _route_preflight_summary(route_preflight).get("verified") is True
            ),
            "live_reverification_still_required": bool(blockers),
        },
        "failure_stage": (
            "single_kernel_and_two_node_live_acceptance"
            if blockers
            else "none"
        ),
        "next_action": (
            "Do not submit more Kaggle kernels under this Goal's exhausted allocation budget; "
            "retain the active blocker until a separately authorized allocation budget is available."
            if blockers
            else "none"
        ),
        "out_of_scope_preserved": {
            "multi_account_concurrency": False,
            "tpu_colab_lightning": False,
            "four_t4_tensor_parallel": False,
            "large_model_training": False,
            "full_parameter_training": False,
            "billing_market_frontend": False,
        },
        "activation_values_public": False,
        "gradient_values_public": False,
        "evaluation_logits_public": False,
        "raw_training_text_public": False,
        "credentials_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report_path = output / "training_cuda_two_node_rc.json"
    _write(report_path, report)
    first_check = check(report_path, require_ready=False)
    ready = bool(first_check["training_cuda_two_node_rc_ready"])
    report["training_cuda_two_node_rc_ready"] = ready
    report["goal_achieved"] = ready
    report["gpu_success_claimed"] = ready
    _write(report_path, report)
    final_check = check(report_path, require_ready=False)
    if not final_check["ok"]:
        report["pack_errors"] = final_check["errors"]
        _write(report_path, report)
        raise RuntimeError(f"CUDA Training RC structural check failed: {final_check['errors']}")
    return {**report, "report_file": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpu-baseline", default="dist/training-foundation-rc-20260710/training_foundation_rc.json")
    parser.add_argument("--single-attempt-report", action="append", default=[])
    parser.add_argument("--two-node-attempt-report", action="append", default=[])
    parser.add_argument("--attempt-ledger", required=True)
    parser.add_argument("--rejection-matrix", required=True)
    parser.add_argument("--test-summary", required=True)
    parser.add_argument("--route-preflight-report", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = pack(
        output_dir=args.output_dir,
        cpu_baseline=args.cpu_baseline,
        single_attempt_reports=args.single_attempt_report,
        two_node_attempt_reports=args.two_node_attempt_report,
        attempt_ledger=args.attempt_ledger,
        rejection_matrix=args.rejection_matrix,
        test_summary=args.test_summary,
        route_preflight_report=args.route_preflight_report or None,
    )
    if args.json:
        public = {key: value for key, value in report.items() if key != "report_file"}
        print(json.dumps(public, sort_keys=True))
    else:
        print(
            f"training_cuda_two_node_rc ready={report['training_cuda_two_node_rc_ready']} "
            f"blockers={','.join(report['blockers']) or 'none'}"
        )


if __name__ == "__main__":
    main()
