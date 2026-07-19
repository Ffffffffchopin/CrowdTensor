#!/usr/bin/env python3
"""Independently validate Elastic Volunteer Training Beta product evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "crowdtensor_elastic_training_beta_live_probe_v1"
REQUIRED_GATES = {
    "ordinary_user_create_status_export_verified",
    "old_product_miners_verified",
    "full_offline_pause_verified",
    "coordinator_restart_verified",
    "replacement_product_miners_verified",
    "real_training_semantics_verified",
    "exactly_once_eight_steps_verified",
    "secure_checkpoint_path_verified",
    "all_kernels_deleted",
    "replacement_kernel_identity_distinct",
}


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("elastic_training_beta_report_invalid")
    return value


def _worker_values(generation: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item.get("worker") or {})
        for item in generation.get("worker_reports") or []
    ]


def check(report_path: str | Path, *, require_ready: bool = False) -> dict[str, Any]:
    report = _load(report_path)
    errors: list[str] = []

    def require(condition: bool, code: str) -> None:
        if not condition:
            errors.append(code)

    require(report.get("schema") == SCHEMA, "elastic_beta_schema_invalid")
    stored_hash = str(report.get("content_hash") or "")
    unhashed = {key: value for key, value in report.items() if key != "content_hash"}
    require(stored_hash == _stable_hash(unhashed), "elastic_beta_content_hash_invalid")
    gates = dict(report.get("acceptance_gates") or {})
    require(set(gates) == REQUIRED_GATES, "elastic_beta_gate_set_invalid")
    require(all(gates.get(key) is True for key in REQUIRED_GATES), "elastic_beta_gate_failed")
    require(report.get("live_run_performed") is True, "elastic_beta_live_run_missing")
    require(report.get("model_id") == "Qwen/Qwen2.5-1.5B", "elastic_beta_model_invalid")
    require(int(report.get("target_steps") or 0) == 8, "elastic_beta_target_steps_invalid")
    require(
        report.get("ordinary_user_cli")
        == {"create": True, "status": True, "export": True},
        "elastic_beta_ordinary_user_cli_incomplete",
    )
    old = dict(report.get("old_generation") or {})
    replacement = dict(report.get("replacement_generation") or {})
    old_workers = _worker_values(old)
    replacement_workers = _worker_values(replacement)
    for name, generation, workers, start, end in (
        ("old", old, old_workers, 0, 4),
        ("replacement", replacement, replacement_workers, 4, 8),
    ):
        require(generation.get("ok") is True, f"elastic_beta_{name}_generation_failed")
        require(
            generation.get("maximum_running_kernel_count") == 2,
            f"elastic_beta_{name}_two_kernel_concurrency_missing",
        )
        require(
            generation.get("terminal_states") == ["complete", "complete"],
            f"elastic_beta_{name}_terminal_state_invalid",
        )
        require(
            generation.get("all_kernels_deleted") is True,
            f"elastic_beta_{name}_cleanup_missing",
        )
        require(len(workers) == 2, f"elastic_beta_{name}_worker_count_invalid")
        require(
            sorted(str(item.get("role") or "") for item in workers)
            == ["kernel_a", "kernel_b"],
            f"elastic_beta_{name}_role_coverage_invalid",
        )
        require(
            all(
                item.get("schema")
                == "crowdtensor_elastic_training_beta_miner_join_v1"
                and item.get("ok") is True
                and int(item.get("expected_start_step", -1)) == start
                and int(item.get("segment_end_step", -1)) == end
                and int(item.get("barrier_commit_count") or 0) == 4
                and item.get("all_completed_barriers_committed") is True
                and item.get("base_weights_frozen") is True
                and item.get("positive_lora_gradient_norms") is True
                for item in workers
            ),
            f"elastic_beta_{name}_product_worker_contract_invalid",
        )
    require(
        all(item.get("graceful_drain_applied") is True for item in old_workers),
        "elastic_beta_old_graceful_drain_missing",
    )
    require(
        all(
            item.get("central_checkpoint_restore_verified") is True
            for item in replacement_workers
        ),
        "elastic_beta_replacement_restore_missing",
    )
    require(
        any(item.get("standard_peft_export_verified") is True for item in replacement_workers)
        and any(item.get("evaluation_verified") is True for item in replacement_workers),
        "elastic_beta_replacement_evaluation_missing",
    )
    midpoint = dict(report.get("midpoint") or {})
    require(
        midpoint.get("committed_step") == 4
        and midpoint.get("zero_live_miners") is True
        and midpoint.get("all_observations_paused") is True,
        "elastic_beta_full_offline_pause_missing",
    )
    restart = dict(report.get("service_restart") or {})
    require(
        restart and all(value is True for value in restart.values()),
        "elastic_beta_coordinator_restart_missing",
    )
    final = dict(report.get("final_status") or {})
    runtime = dict(final.get("runtime") or {})
    require(
        final.get("overall_state") == "completed"
        and final.get("global_step") == 8
        and runtime.get("committed_steps") == list(range(1, 9))
        and runtime.get("optimizer_commit_count") == 8
        and runtime.get("checkpoint_signatures_required") is True
        and runtime.get("checkpoint_tensor_validation_required") is True,
        "elastic_beta_final_runtime_invalid",
    )
    exported = dict(report.get("export_cli_report") or {})
    require(
        exported.get("ok") is True
        and exported.get("standard_peft_format") is True
        and exported.get("adapter_tensor_count") == 392
        and exported.get("layer_indexes") == list(range(28)),
        "elastic_beta_owner_export_invalid",
    )
    cleanup = dict(report.get("cleanup") or {})
    require(
        report.get("cleanup_verified") is True
        and cleanup.get("all_kernels_deleted") is True
        and cleanup.get("service_stopped") is True
        and cleanup.get("tunnel_stopped") is True
        and cleanup.get("rendezvous_payloads_removed") is True
        and cleanup.get("private_runtime_removed") is True
        and cleanup.get("live_resources_left_running") is False,
        "elastic_beta_cleanup_invalid",
    )
    repack = dict(report.get("repack") or {})
    require(
        repack.get("runtime_measurements_changed") is False
        and repack.get("generation_cleanup_verified") is True
        and repack.get("post_cleanup_account_audit_verified") is True
        and repack.get("selected_account_active_kernel_count") == 0,
        "elastic_beta_repack_provenance_invalid",
    )
    for key in (
        "credentials_public",
        "credential_paths_public",
        "coordinator_url_public",
        "raw_training_text_public",
        "token_ids_public",
        "activation_values_public",
        "gradient_values_public",
        "checkpoint_tensor_values_public",
        "adapter_tensor_values_public",
        "private_paths_public",
    ):
        require(report.get(key) is False, f"elastic_beta_public_safety_{key}")
    require(report.get("public_safety_errors") == [], "elastic_beta_public_safety_errors")
    require(report.get("blockers") == [], "elastic_beta_unexpected_blockers")
    ready = bool(
        not errors
        and report.get("ok") is True
        and report.get("elastic_training_beta_ready") is True
    )
    if require_ready and not ready:
        errors.append("elastic_training_beta_not_ready")
    return {
        "schema": "crowdtensor_elastic_training_beta_check_v1",
        "ok": not errors,
        "elastic_training_beta_ready": ready,
        "error_count": len(errors),
        "errors": errors,
        "report_content_hash": stored_hash,
        "public_artifact_safe": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(args.report, require_ready=args.require_ready)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            "training_elastic_beta_check "
            f"ok={result['ok']} ready={result['elastic_training_beta_ready']} "
            f"errors={result['error_count']}"
        )
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
