#!/usr/bin/env python3
"""Strictly verify the One-Click Contributor Beta release candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.model_adapter import stable_hash
from crowdtensor.version import __version__
if __package__:
    from .one_click_contributor_release_pack import MANIFEST_SCHEMA, SCHEMA
else:
    from one_click_contributor_release_pack import MANIFEST_SCHEMA, SCHEMA


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def check(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path).expanduser().resolve()
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report = {}
    manifest = report.get("release_manifest") if isinstance(
        report.get("release_manifest"), dict
    ) else {}
    errors: list[str] = []
    if report.get("schema") != SCHEMA or manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append("one_click_release_schema_invalid")
    if report.get("ok") is not True:
        errors.append("one_click_release_not_ready")
    if manifest.get("package_version") != __version__:
        errors.append("one_click_release_version_invalid")
    supplied_manifest_hash = str(manifest.get("content_hash") or "")
    expected_manifest_hash = stable_hash(
        {key: value for key, value in manifest.items() if key != "content_hash"}
    )
    if supplied_manifest_hash != expected_manifest_hash:
        errors.append("one_click_release_manifest_hash_invalid")
    supplied_report_hash = str(report.get("content_hash") or "")
    expected_report_hash = stable_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    if supplied_report_hash != expected_report_hash:
        errors.append("one_click_release_report_hash_invalid")

    wheel = manifest.get("wheel_contract") or {}
    if not all(
        wheel.get(field) is True
        for field in (
            "required_one_click_files_present",
            "join_page_packaged",
            "webgpu_worker_packaged",
            "native_status_page_packaged",
        )
    ):
        errors.append("one_click_release_wheel_contract_invalid")
    clean = manifest.get("clean_install") or {}
    if not (
        clean.get("ok") is True
        and clean.get("fresh_venv") is True
        and clean.get("all_golden_commands_passed") is True
        and clean.get("workspace_import_used") is False
    ):
        errors.append("one_click_release_clean_install_invalid")
    container = manifest.get("container") or {}
    if not (
        container.get("image_built") is True
        and container.get("image_removed") is True
        and container.get("compose_config_valid") is True
        and container.get("published") is False
    ):
        errors.append("one_click_release_container_invalid")
    e2e = manifest.get("e2e") or {}
    if not (
        e2e.get("browser_task_accepted") is True
        and e2e.get("browser_runtime") in {"webgpu", "wasm-cpu", "cpu-js"}
        and e2e.get("browser_model_update") is False
        and e2e.get("native_lora_update_accepted") is True
        and e2e.get("real_peft_lora") is True
        and e2e.get("one_time_codes_persisted") is False
        and e2e.get("cleanup_complete") is True
    ):
        errors.append("one_click_release_e2e_invalid")
    install = manifest.get("install") or {}
    if not (
        install.get("first_contribution_maximum_steps") == 3
        and install.get("clone_required") is False
        and install.get("manual_invite_file_required") is False
        and install.get("native_agent_device_types") == ["cpu", "cuda"]
        and install.get("jax_tpu_one_click") is False
        and install.get("jax_tpu_advanced_workflow") is True
        and install.get("cpu_only_torch_on_cpu_hosts") is True
        and install.get("provider_torch_index_override") is True
        and install.get("pinned_contributor_runtime") is True
        and install.get("storage_extra_installed_by_default") is False
    ):
        errors.append("one_click_release_install_contract_invalid")
    boundaries = manifest.get("boundaries") or {}
    if not (
        boundaries.get("controlled_enrollment") is True
        and boundaries.get("permissionless") is False
        and boundaries.get("browser_lora_training") is False
        and boundaries.get("browser_large_model_sharding") is False
        and boundaries.get("kimi_k3") is False
        and boundaries.get("production_sla") is False
    ):
        errors.append("one_click_release_boundary_invalid")

    artifacts = manifest.get("artifacts") or {}
    for name, item in artifacts.items():
        target = path.parent / str((item or {}).get("file_name") or "")
        if (
            not target.is_file()
            or _hash(target) != item.get("sha256")
            or target.stat().st_size != int(item.get("size_bytes") or -1)
        ):
            errors.append("one_click_release_artifact_invalid:" + str(name))
    for field in ("release_json", "checksums", "bundle"):
        item = report.get(field) or {}
        target = path.parent / str(item.get("file_name") or "")
        if not target.is_file() or _hash(target) != item.get("sha256"):
            errors.append("one_click_release_" + field + "_invalid")
    sums_path = path.parent / "SHA256SUMS"
    if sums_path.is_file():
        for line in sums_path.read_text(encoding="ascii").splitlines():
            digest, file_name = line.split("  ", 1)
            target = path.parent / file_name
            if not target.is_file() or _hash(target) != "sha256:" + digest:
                errors.append("one_click_release_checksum_invalid:" + file_name)
    else:
        errors.append("one_click_release_checksums_missing")
    if scan_public_value(report)["ok"] is not True:
        errors.append("one_click_release_public_safety_invalid")
    return {
        "schema": "crowdtensor_one_click_contributor_release_check_v1",
        "ok": not errors,
        "error_count": len(set(errors)),
        "errors": sorted(set(errors)),
        "artifact_count": len(artifacts),
        "one_click_contributor_beta_ready": not errors,
        "public_artifact_safe": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(args.report)
    print(json.dumps(result, sort_keys=True) if args.json else f"ok={result['ok']} errors={result['error_count']}")
    return 0 if result["ok"] or not args.require_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
