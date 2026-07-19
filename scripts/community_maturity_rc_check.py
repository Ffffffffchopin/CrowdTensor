#!/usr/bin/env python3
"""Strict, portable checker for the CrowdTensor Community Maturity RC."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.model_adapter import stable_hash
from crowdtensor.version import public_version
from scripts.community_cleanup_audit_check import check as check_cleanup
from scripts.community_kaggle_reliability_live_check import check as check_live
from scripts.community_release_check import check as check_release
from scripts.community_smollm_live_check import check as check_smollm


SCHEMA = "crowdtensor_community_maturity_rc_v1"
CHECK_SCHEMA = "crowdtensor_community_maturity_rc_check_v1"
REQUIRED_ARTIFACTS = {
    "local_gate",
    "release",
    "docs",
    "minio",
    "wheel_smoke",
    "kaggle_live",
    "smollm_live",
    "cleanup_audit",
    "gpu_diagnostic",
}
REQUIRED_REQUIREMENT_KEYS = {
    "P0": {
        "P0.1_architecture_boundaries",
        "P0.2_installable_release",
        "P0.3_ordinary_user_lifecycle",
        "P0.4_kaggle_clean_install_logical_nodes",
        "P0.5_executable_ci_and_local_gate",
    },
    "P1": {
        "P1.1_bounded_reliability_chaos",
        "P1.2_short_kaggle_reliability_gate",
        "P1.3_real_minio_integration",
        "P1.4_bound_benchmarks",
    },
    "P2": {
        "P2.1_security_and_trust_controls",
        "P2.2_restricted_execution_and_quarantine",
        "P2.3_negative_public_safety",
    },
    "P3": {
        "P3.1_versioned_model_adapter",
        "P3.2_second_model_kaggle_live",
        "P3.3_support_matrix_and_refusals",
    },
    "P4": {
        "P4.1_community_governance_and_license",
        "P4.2_release_bundle_and_compatibility",
        "P4.3_canonical_strict_rc_and_cleanup",
    },
}


def _read(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _content_hash_valid(value: dict[str, Any]) -> bool:
    supplied = str(value.get("content_hash") or "")
    return bool(
        supplied
        and supplied
        == stable_hash({key: item for key, item in value.items() if key != "content_hash"})
    )


def _local_gate_ok(value: dict[str, Any]) -> bool:
    return bool(
        value.get("schema") == "crowdtensor_community_local_gate_v1"
        and value.get("ok") is True
        and int((value.get("tests") or {}).get("passed_count") or 0) >= 1
        and int((value.get("chaos") or {}).get("scenario_count") or 0) >= 12
        and (value.get("chaos") or {}).get("ok") is True
        and (value.get("workflow") or {}).get("complete_action_contract") is True
        and value.get("public_artifact_safe") is True
        and _content_hash_valid(value)
    )


def _docs_ok(value: dict[str, Any]) -> bool:
    return bool(
        value.get("schema") == "crowdtensor_community_docs_check_v1"
        and value.get("ok") is True
        and not value.get("missing_files")
        and int(value.get("broken_link_count") or 0) == 0
        and value.get("public_artifact_safe") is True
    )


def _minio_ok(value: dict[str, Any]) -> bool:
    return bool(
        value.get("ok") is True
        and value.get("real_api_calls_performed") is True
        and value.get("content_addressed") is True
        and value.get("mirror_fallback_verified") is True
        and value.get("primary_repair_verified") is True
        and value.get("retention_verified") is True
        and value.get("service_restart_verified") is True
        and value.get("cleanup_verified") is True
        and value.get("container_left_running") is False
        and value.get("external_managed_object_storage_sla_verified") is False
        and _content_hash_valid(value)
    )


def _wheel_smoke_ok(value: dict[str, Any]) -> bool:
    kernel = value.get("kernel") if isinstance(value.get("kernel"), dict) else {}
    return bool(
        value.get("schema") == "crowdtensor_community_kaggle_wheel_smoke_v1"
        and value.get("ok") is True
        and value.get("full_live_gate") is False
        and value.get("gpu_used") is False
        and value.get("cleanup_verified") is True
        and kernel.get("fresh_install_kind") == "pip_target"
        and kernel.get("installed_package_under_install_root") is True
        and kernel.get("model_stack_import_verified") is True
        and kernel.get("runtime_requirements_exact_pins_verified") is True
        and kernel.get("golden_commands_passed") is True
        and kernel.get("workspace_import_used") is False
        and _content_hash_valid(value)
    )


def _gpu_diagnostic_ok(value: dict[str, Any]) -> bool:
    kernel = value.get("kernel") if isinstance(value.get("kernel"), dict) else {}
    dual = kernel.get("dual_stage") if isinstance(kernel.get("dual_stage"), dict) else {}
    return bool(
        value.get("schema") == "crowdtensor_community_kaggle_gpu_stage0_diagnostic_v1"
        and value.get("ok") is True
        and value.get("diagnostic_only") is True
        and value.get("full_live_gate") is False
        and value.get("cleanup_verified") is True
        and kernel.get("wheel_clean_install_verified") is True
        and kernel.get("runtime_requirements_exact_pins_verified") is True
        and kernel.get("stage0_model_loaded") is True
        and kernel.get("adapter_updated") is True
        and kernel.get("optimizer_step_applied") is True
        and dual.get("verified") is True
        and dual.get("devices") == ["cuda", "cuda"]
        and dual.get("both_stage_adapters_updated") is True
        and dual.get("adapter_reload_verified") is True
        and _content_hash_valid(value)
    )


def source_checks(values: dict[str, dict[str, Any]], paths: dict[str, Path]) -> dict[str, bool]:
    return {
        "local_gate": _local_gate_ok(values["local_gate"]),
        "release": check_release(paths["release"])["ok"] is True,
        "docs": _docs_ok(values["docs"]),
        "minio": _minio_ok(values["minio"]),
        "wheel_smoke": _wheel_smoke_ok(values["wheel_smoke"]),
        "kaggle_live": check_live(paths["kaggle_live"])["ok"] is True,
        "smollm_live": check_smollm(paths["smollm_live"])["ok"] is True,
        "cleanup_audit": check_cleanup(paths["cleanup_audit"])["ok"] is True,
        "gpu_diagnostic": _gpu_diagnostic_ok(values["gpu_diagnostic"]),
    }


def derive_requirements(
    values: dict[str, dict[str, Any]], checks: dict[str, bool]
) -> dict[str, dict[str, bool]]:
    local = values["local_gate"]
    workflow = local.get("workflow") if isinstance(local.get("workflow"), dict) else {}
    tests = local.get("tests") if isinstance(local.get("tests"), dict) else {}
    security = local.get("security") if isinstance(local.get("security"), dict) else {}
    registry = local.get("model_adapters") if isinstance(local.get("model_adapters"), dict) else {}
    release = values["release"].get("release_manifest") or {}
    release_artifacts = release.get("artifacts") if isinstance(release.get("artifacts"), dict) else {}
    live = values["kaggle_live"]
    benchmark = live.get("benchmark") if isinstance(live.get("benchmark"), dict) else {}
    unresolved = set(security.get("unresolved_security_boundaries") or [])
    expected_unresolved = {
        "byzantine_fault_tolerance",
        "confidential_computing_or_tee",
        "privacy_preserving_computation",
        "secure_aggregation",
        "semantic_poisoning_resistance",
        "sybil_resistance",
    }
    unsupported = set(registry.get("unsupported_capabilities") or [])
    expected_unsupported = {
        "arbitrary_architecture_partition",
        "data_parallel_training",
        "full_parameter_training",
        "in_flight_stage_migration",
        "parameter_limit_exploration",
    }
    wheel_hashes = {
        str((release_artifacts.get("wheel") or {}).get("sha256") or ""),
        str(values["wheel_smoke"].get("wheel_sha256") or ""),
        str((live.get("clean_install") or {}).get("wheel_hash") or ""),
        str(values["gpu_diagnostic"].get("wheel_sha256") or ""),
    }
    wheel_identity = bool(
        len(wheel_hashes) == 1
        and next(iter(wheel_hashes), "").startswith("sha256:")
    )
    security_core = bool(
        all(
            security.get(field) is True
            for field in (
                "tls_proxy_contract",
                "default_deny_rbac",
                "short_lived_rotatable_credentials",
                "replay_protection",
                "resource_limits",
                "public_safety_scanner",
            )
        )
        and security.get("task_signatures") == "hmac_sha256"
    )
    all_sources_public = all(
        value.get("public_artifact_safe") is True for value in values.values()
    )
    return {
        "P0": {
            "P0.1_architecture_boundaries": bool(checks["local_gate"] and checks["docs"]),
            "P0.2_installable_release": checks["release"],
            "P0.3_ordinary_user_lifecycle": bool(
                checks["local_gate"]
                and workflow.get("complete_action_contract") is True
                and workflow.get("idempotency_tested") is True
                and workflow.get("dry_run_tested") is True
                and workflow.get("explicit_exit_codes_tested") is True
                and workflow.get("run_id_and_safe_next_command_tested") is True
            ),
            "P0.4_kaggle_clean_install_logical_nodes": bool(
                checks["wheel_smoke"] and checks["kaggle_live"] and wheel_identity
            ),
            "P0.5_executable_ci_and_local_gate": bool(
                checks["local_gate"]
                and tests.get("security_negative_tests_included") is True
                and tests.get("protocol_compatibility_tests_included") is True
                and tests.get("scheduler_runtime_cli_tests_included") is True
                and tests.get("cuda_contract_tests_included") is True
                and tests.get("jax_contract_tests_included") is True
            ),
        },
        "P1": {
            "P1.1_bounded_reliability_chaos": bool(
                checks["local_gate"]
                and (local.get("chaos") or {}).get("ok") is True
                and int((local.get("chaos") or {}).get("scenario_count") or 0) >= 12
            ),
            "P1.2_short_kaggle_reliability_gate": bool(
                checks["kaggle_live"]
                and (live.get("acceptance") or {}).get("ok") is True
                and len(live.get("committed_step_ids") or []) == 100
                and live.get("worker_replacement_verified") is True
                and live.get("coordinator_restart_verified") is True
            ),
            "P1.3_real_minio_integration": checks["minio"],
            "P1.4_bound_benchmarks": bool(
                checks["kaggle_live"]
                and float(benchmark.get("steps_per_second") or 0.0) > 0
                and float(benchmark.get("p95_step_seconds") or 0.0) > 0
                and int(benchmark.get("forward_payload_bytes") or 0) > 0
                and int(benchmark.get("backward_payload_bytes") or 0) > 0
                and int(benchmark.get("checkpoint_bytes") or 0) > 0
            ),
        },
        "P2": {
            "P2.1_security_and_trust_controls": bool(checks["local_gate"] and security_core),
            "P2.2_restricted_execution_and_quarantine": bool(
                checks["local_gate"]
                and security.get("restricted_worker_execution") is True
                and security.get("anomaly_quarantine_interface") is True
                and unresolved == expected_unresolved
            ),
            "P2.3_negative_public_safety": bool(
                checks["local_gate"]
                and tests.get("security_negative_tests_included") is True
                and all_sources_public
            ),
        },
        "P3": {
            "P3.1_versioned_model_adapter": bool(
                checks["local_gate"]
                and registry.get("api_version") == "model_adapter_v1.0"
                and sorted(registry.get("supported_model_families") or [])
                == ["qwen2", "smollm2"]
            ),
            "P3.2_second_model_kaggle_live": bool(
                checks["smollm_live"] and checks["gpu_diagnostic"]
            ),
            "P3.3_support_matrix_and_refusals": bool(
                checks["docs"] and unsupported == expected_unsupported
            ),
        },
        "P4": {
            "P4.1_community_governance_and_license": bool(
                checks["docs"]
                and (release.get("license_audit") or {}).get("license_file_present") is True
                and (release.get("license_audit") or {}).get("dependency_unknowns_recorded") is True
            ),
            "P4.2_release_bundle_and_compatibility": bool(
                checks["release"]
                and len(release_artifacts) >= 6
                and (release.get("publishing") or {}).get("pypi_uploaded") is False
                and (release.get("publishing") or {}).get("github_release_created") is False
                and (release.get("publishing") or {}).get("container_registry_pushed") is False
            ),
            "P4.3_canonical_strict_rc_and_cleanup": bool(
                checks["cleanup_audit"] and all(checks.values())
            ),
        },
    }


def derive_readiness(
    values: dict[str, dict[str, Any]], checks: dict[str, bool]
) -> dict[str, Any]:
    requirements = derive_requirements(values, checks)
    release = values["release"].get("release_manifest") or {}
    release_artifacts = release.get("artifacts") if isinstance(release.get("artifacts"), dict) else {}
    live = values["kaggle_live"]
    wheel_hashes = {
        str((release_artifacts.get("wheel") or {}).get("sha256") or ""),
        str(values["wheel_smoke"].get("wheel_sha256") or ""),
        str((live.get("clean_install") or {}).get("wheel_hash") or ""),
        str(values["gpu_diagnostic"].get("wheel_sha256") or ""),
    }
    wheel_identity = bool(
        len(wheel_hashes) == 1
        and next(iter(wheel_hashes), "").startswith("sha256:")
    )
    p0 = all(requirements["P0"].values())
    p1 = all(requirements["P1"].values())
    p2 = all(requirements["P2"].values())
    p3 = all(requirements["P3"].values())
    p4 = all(requirements["P4"].values())
    cleanup = checks["cleanup_audit"]
    return {
        "p0_ready": p0,
        "p1_ready": p1,
        "p2_ready": p2,
        "p3_ready": p3,
        "p4_ready": p4,
        "cleanup_ready": cleanup,
        "wheel_identity_verified": wheel_identity,
        "community_maturity_rc_ready": bool(p0 and p1 and p2 and p3 and p4 and cleanup),
    }


def check_report(value: dict[str, Any], *, require_ready: bool = False) -> list[str]:
    errors: list[str] = []
    if value.get("schema") != SCHEMA:
        errors.append("community_maturity_schema_invalid")
    if value.get("versions") != public_version():
        errors.append("community_maturity_versions_invalid")
    if value.get("node_scope") != "Kaggle logical multi-node":
        errors.append("community_maturity_node_scope_invalid")
    if value.get("physical_multi_machine_verified") is not False:
        errors.append("community_maturity_physical_multi_machine_overclaim")
    gates = value.get("gates") if isinstance(value.get("gates"), dict) else {}
    requirements = value.get("requirements") if isinstance(value.get("requirements"), dict) else {}
    if (
        set(requirements) != set(REQUIRED_REQUIREMENT_KEYS)
        or any(
            not isinstance(requirements.get(section), dict)
            or set(requirements[section]) != keys
            or any(not isinstance(item, bool) for item in requirements[section].values())
            for section, keys in REQUIRED_REQUIREMENT_KEYS.items()
        )
    ):
        errors.append("community_maturity_requirement_matrix_invalid")
    else:
        for section in REQUIRED_REQUIREMENT_KEYS:
            if gates.get(section.lower() + "_ready") is not all(
                requirements[section].values()
            ):
                errors.append("community_maturity_requirement_gate_mismatch:" + section)
    expected_ready = all(
        gates.get(field) is True
        for field in ("p0_ready", "p1_ready", "p2_ready", "p3_ready", "p4_ready", "cleanup_ready")
    )
    if value.get("community_maturity_rc_ready") is not expected_ready:
        errors.append("community_maturity_ready_consistency_invalid")
    if gates.get("community_maturity_rc_ready") is not expected_ready:
        errors.append("community_maturity_nested_ready_consistency_invalid")
    blockers = list(value.get("blockers") or [])
    if expected_ready == bool(blockers):
        errors.append("community_maturity_blocker_consistency_invalid")
    if set((value.get("artifacts") or {})) != REQUIRED_ARTIFACTS:
        errors.append("community_maturity_artifact_set_invalid")
    if set((value.get("source_checks") or {})) != REQUIRED_ARTIFACTS:
        errors.append("community_maturity_source_check_set_invalid")
    supplied = str(value.get("content_hash") or "")
    if supplied != stable_hash({key: item for key, item in value.items() if key != "content_hash"}):
        errors.append("community_maturity_content_hash_invalid")
    if scan_public_value(value)["ok"] is not True:
        errors.append("community_maturity_public_safety_invalid")
    if require_ready and not expected_ready:
        errors.append("community_maturity_rc_not_ready")
    return sorted(set(errors))


def check(path: str | Path, *, require_ready: bool = False) -> dict[str, Any]:
    report_path = Path(path).expanduser().resolve()
    value = _read(report_path)
    errors = check_report(value, require_ready=require_ready)
    artifacts = value.get("artifacts") if isinstance(value.get("artifacts"), dict) else {}
    values: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for name in REQUIRED_ARTIFACTS:
        item = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
        relative = Path(str(item.get("relative_path") or ""))
        if not str(relative) or relative.is_absolute() or ".." in relative.parts:
            errors.append("community_maturity_artifact_path_invalid:" + name)
            continue
        source = report_path.parent / relative
        if not source.is_file():
            errors.append("community_maturity_artifact_missing:" + name)
            continue
        if _file_hash(source) != item.get("sha256"):
            errors.append("community_maturity_artifact_hash_invalid:" + name)
        paths[name] = source
        values[name] = _read(source)
    if set(values) == REQUIRED_ARTIFACTS:
        actual_checks = source_checks(values, paths)
        if actual_checks != value.get("source_checks"):
            errors.append("community_maturity_source_checks_mismatch")
        actual_gates = derive_readiness(values, actual_checks)
        if actual_gates != value.get("gates"):
            errors.append("community_maturity_gate_derivation_mismatch")
        actual_requirements = derive_requirements(values, actual_checks)
        if actual_requirements != value.get("requirements"):
            errors.append("community_maturity_requirement_derivation_mismatch")
        if require_ready and not all(actual_checks.values()):
            errors.append("community_maturity_source_evidence_not_ready")
    privacy = scan_public_value(value)
    return {
        "schema": CHECK_SCHEMA,
        "ok": not errors,
        "errors": sorted(set(errors)),
        "community_maturity_rc_ready": value.get("community_maturity_rc_ready") is True,
        "artifact_count": len(values),
        "public_safety": privacy,
        "public_artifact_safe": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(args.report, require_ready=args.require_ready)
    print(json.dumps(result, sort_keys=True) if args.json else f"ok={result['ok']} ready={result['community_maturity_rc_ready']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
