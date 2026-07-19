#!/usr/bin/env python3
"""Verify the offline Community release manifest and artifact hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.version import __version__
from scripts.community_release_build import REPORT_SCHEMA, SCHEMA


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def check(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report = {}
    manifest = report.get("release_manifest") if isinstance(report.get("release_manifest"), dict) else {}
    errors: list[str] = []
    if report.get("schema") != REPORT_SCHEMA or manifest.get("schema") != SCHEMA:
        errors.append("community_release_schema_invalid")
    if report.get("ok") is not True:
        errors.append("community_release_build_not_ready")
    versions = manifest.get("versions") if isinstance(manifest.get("versions"), dict) else {}
    if versions.get("package_version") != __version__:
        errors.append("community_release_version_invalid")
    for field in ("community_protocol_version", "model_adapter_api_version", "evidence_api_version"):
        if not str(versions.get(field) or ""):
            errors.append("community_release_" + field + "_missing")
    wheel = manifest.get("wheel_contract") if isinstance(manifest.get("wheel_contract"), dict) else {}
    if not all(wheel.get(field) is True for field in ("required_modules_present", "license_file_present", "entry_points_present")):
        errors.append("community_release_wheel_contract_invalid")
    sdist = manifest.get("sdist_contract") if isinstance(manifest.get("sdist_contract"), dict) else {}
    if sdist.get("required_sources_present") is not True or sdist.get("absolute_members_present") is not False:
        errors.append("community_release_sdist_contract_invalid")
    provenance = manifest.get("python_artifact_provenance") if isinstance(
        manifest.get("python_artifact_provenance"), dict
    ) else {}
    if provenance:
        mode = provenance.get("mode")
        if mode not in {"built_from_source", "existing_python_artifacts_reused"}:
            errors.append("community_release_python_artifact_provenance_invalid")
        if mode == "existing_python_artifacts_reused" and (
            provenance.get("wheel_rebuilt") is not False
            or provenance.get("sdist_rebuilt") is not False
            or provenance.get("expected_wheel_sha256_enforced") is not True
        ):
            errors.append("community_release_python_artifact_reuse_invalid")
    clean = manifest.get("clean_install") if isinstance(manifest.get("clean_install"), dict) else {}
    if (
        clean.get("ok") is not True
        or clean.get("fresh_venv") is not True
        or clean.get("all_golden_commands_passed") is not True
        or clean.get("workspace_import_used") is not False
    ):
        errors.append("community_release_clean_install_invalid")
    container = manifest.get("container") if isinstance(manifest.get("container"), dict) else {}
    if (
        container.get("compose_config_valid") is not True
        or container.get("image_built") is not True
        or container.get("image_removed") is not True
        or container.get("published") is not False
        or not str(container.get("image_id_hash") or "").startswith("sha256:")
    ):
        errors.append("community_release_container_invalid")
    publishing = manifest.get("publishing") if isinstance(manifest.get("publishing"), dict) else {}
    if any(publishing.get(field) is not False for field in ("pypi_uploaded", "github_release_created", "container_registry_pushed")):
        errors.append("community_release_external_publish_detected")
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    if provenance and (
        provenance.get("wheel_sha256") != (artifacts.get("wheel") or {}).get("sha256")
        or provenance.get("sdist_sha256") != (artifacts.get("sdist") or {}).get("sha256")
        or provenance.get("source_paths_public") is not False
    ):
        errors.append("community_release_python_artifact_identity_invalid")
    for name in (
        "wheel",
        "sdist",
        "sbom",
        "dependency_inventory",
        "constraints",
        "kaggle_runtime_lock",
    ):
        item = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
        file_name = str(item.get("file_name") or "")
        candidates = [path.parent / file_name, path.parent / "artifacts" / file_name, Path.cwd() / "requirements" / file_name]
        target = next((candidate for candidate in candidates if candidate.is_file()), None)
        if target is None or _hash(target) != item.get("sha256") or target.stat().st_size != int(item.get("size_bytes") or -1):
            errors.append("community_release_artifact_hash_invalid:" + name)
    runtime_lock = artifacts.get("kaggle_runtime_lock") if isinstance(
        artifacts.get("kaggle_runtime_lock"), dict
    ) else {}
    lock_path = Path.cwd() / "requirements" / str(runtime_lock.get("file_name") or "")
    pins = [
        line.strip()
        for line in lock_path.read_text(encoding="utf-8").splitlines()
        if lock_path.is_file() and line.strip() and not line.startswith("#")
    ] if lock_path.is_file() else []
    if not pins or any("==" not in item for item in pins):
        errors.append("community_release_kaggle_runtime_exact_lock_invalid")
    bundle = report.get("bundle") if isinstance(report.get("bundle"), dict) else {}
    bundle_path = path.parent / str(bundle.get("file_name") or "")
    if not bundle_path.is_file() or _hash(bundle_path) != bundle.get("sha256"):
        errors.append("community_release_bundle_hash_invalid")
    docs = manifest.get("documentation") if isinstance(manifest.get("documentation"), dict) else {}
    if len(docs) < 10 or any(not (Path.cwd() / str(value)).is_file() for value in docs.values()):
        errors.append("community_release_documentation_incomplete")
    privacy = scan_public_value(report)
    if privacy["ok"] is not True:
        errors.append("community_release_public_safety_invalid")
    return {
        "schema": "crowdtensor_community_release_check_v1",
        "ok": not errors,
        "errors": sorted(set(errors)),
        "artifact_count": len(artifacts),
        "document_count": len(docs),
        "public_safety": privacy,
        "public_artifact_safe": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(args.report)
    print(json.dumps(result, sort_keys=True) if args.json else f"ok={result['ok']} errors={len(result['errors'])}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
