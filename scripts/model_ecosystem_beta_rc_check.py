#!/usr/bin/env python3
"""Strict checker for the pluggable Model Adapter Ecosystem Beta RC."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.model_adapter import stable_hash


SCHEMA = "crowdtensor_model_ecosystem_beta_rc_v1"
MODEL_ID = "Locutusque/TinyMistral-248M-v2"
MODEL_REVISION = "0f57b17cb317bb322c7c1466b669c681f80c058f"
HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")
REQUIRED_ARTIFACTS = {
    "adapter_config",
    "adapter_weights",
    "adapter_wheel",
    "attempt_ledger",
    "core_wheel",
    "live_check",
    "live_report",
    "local_gate",
    "plugin_smoke",
}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_hash(value: Any) -> bool:
    return bool(HASH_RE.fullmatch(str(value or "")))


def check_report(report: Any, *, artifact_root: str | Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        return {
            "schema": "crowdtensor_model_ecosystem_beta_rc_check_v1",
            "ok": False,
            "errors": ["model_ecosystem_rc_schema_invalid"],
            "error_count": 1,
            "goal_achieved": False,
        }
    if report.get("content_hash") != stable_hash(
        {key: item for key, item in report.items() if key != "content_hash"}
    ):
        errors.append("model_ecosystem_rc_content_hash_invalid")
    required_flags = (
        "model_adapter_ecosystem_ready",
        "entry_point_plugin_contract_ready",
        "mistral_adapter_ready",
        "mistral_real_heterogeneous_training_verified",
        "dual_wheel_clean_install_verified",
        "checkpoint_replacement_verified",
        "peft_export_reload_verified",
        "regression_gate_verified",
        "cleanup_verified",
        "goal_achieved",
    )
    if any(report.get(field) is not True for field in required_flags):
        errors.append("model_ecosystem_rc_readiness_incomplete")
    if report.get("supported_model_families") != ["mistral", "qwen2", "smollm2"]:
        errors.append("model_ecosystem_rc_family_registry_invalid")
    plugin = dict(report.get("plugin_smoke_summary") or {})
    if not all(
        (
            plugin.get("ok") is True,
            plugin.get("adapter_id") == "mistral_lora_v1",
            plugin.get("family") == "mistral",
            plugin.get("model_id") == MODEL_ID,
            plugin.get("model_revision") == MODEL_REVISION,
            plugin.get("registration_kind") == "entry_point_plugin",
            plugin.get("entry_point_group") == "crowdtensor.model_adapters.v1",
            plugin.get("conformance_verified") is True,
            plugin.get("partition_verified") is True,
            plugin.get("isolated_venv") is True,
            plugin.get("wheel_install_no_deps") is True,
            plugin.get("workspace_import_used") is False,
        )
    ):
        errors.append("model_ecosystem_rc_plugin_smoke_invalid")
    live = dict(report.get("live_summary") or {})
    if not all(
        (
            live.get("strict_check_ok") is True,
            live.get("mistral_live_verified") is True,
            live.get("live_run_performed") is True,
            live.get("committed_step_ids") == list(range(1, 9)),
            live.get("accepted_providers") == ["kaggle_cpu", "kaggle_cuda"],
            live.get("gpu_worker_replacement_verified") is True,
            live.get("restored_checkpoint_step") == 4,
            live.get("optimizer_state_restored") is True,
            live.get("adapter_tensor_count") == 168,
            live.get("adapter_reload_verified") is True,
            live.get("cleanup_verified") is True,
            live.get("public_safety_verified") is True,
        )
    ):
        errors.append("model_ecosystem_rc_live_summary_invalid")
    quality = dict(report.get("quality_summary") or {})
    if not all(
        (
            quality.get("ok") is True,
            int(quality.get("passed") or 0) >= 40,
            int(quality.get("failed") or 0) == 0,
            quality.get("py_compile_ok") is True,
            quality.get("plugin_registry_tests_included") is True,
            quality.get("mistral_architecture_tests_included") is True,
            quality.get("live_report_checker_tests_included") is True,
        )
    ):
        errors.append("model_ecosystem_rc_quality_invalid")
    artifacts = dict(report.get("artifacts") or {})
    if set(artifacts) != REQUIRED_ARTIFACTS or any(
        not isinstance(item, dict)
        or not str(item.get("relative_path") or "")
        or not _is_hash(item.get("sha256"))
        for item in artifacts.values()
    ):
        errors.append("model_ecosystem_rc_artifact_manifest_invalid")
    if artifact_root is not None and set(artifacts) == REQUIRED_ARTIFACTS:
        root = Path(artifact_root).expanduser().resolve()
        for name, item in artifacts.items():
            relative = Path(str(item.get("relative_path") or ""))
            try:
                path = (root / relative).resolve()
                path.relative_to(root)
            except (OSError, ValueError):
                errors.append("model_ecosystem_rc_artifact_path_invalid:" + name)
                continue
            if not path.is_file() or _hash_file(path) != item.get("sha256"):
                errors.append("model_ecosystem_rc_artifact_hash_invalid:" + name)
    cleanup = dict(report.get("cleanup") or {})
    if not all(
        (
            cleanup.get("all_remote_kernels_deleted") is True,
            cleanup.get("live_resources_left_running") is False,
            cleanup.get("private_runtime_removed") is True,
            cleanup.get("community_maturity_ledger_modified") is False,
        )
    ):
        errors.append("model_ecosystem_rc_cleanup_invalid")
    limitations = dict(report.get("unsupported_claims") or {})
    if set(limitations) != {
        "arbitrary_architecture_support_verified",
        "full_parameter_training_verified",
        "mistral_7b_live_verified",
        "physical_multi_machine_verified",
        "production_sla_verified",
    } or any(item is not False for item in limitations.values()):
        errors.append("model_ecosystem_rc_overclaim_invalid")
    safety = scan_public_value(
        {key: item for key, item in report.items() if key != "public_safety"}
    )
    if safety["ok"] is not True or report.get("public_artifact_safe") is not True:
        errors.append("model_ecosystem_rc_public_safety_invalid")
    result = {
        "schema": "crowdtensor_model_ecosystem_beta_rc_check_v1",
        "ok": not errors,
        "errors": sorted(set(errors)),
        "error_count": len(set(errors)),
        "goal_achieved": not errors,
        "public_artifact_safe": True,
    }
    result["content_hash"] = stable_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report = {}
    root = args.artifact_root or str(Path(args.report).expanduser().resolve().parent)
    result = check_report(report, artifact_root=root)
    print(json.dumps(result, sort_keys=True) if args.json else f"ok={result['ok']} errors={result['error_count']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
