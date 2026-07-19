#!/usr/bin/env python3
"""Validate GLM 5.2 Kaggle stage worker package manifests."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


SCHEMA = "glm52_kaggle_stage_worker_package_check_v1"
PACKAGE_SCHEMA = "glm52_kaggle_stage_worker_package_v1"
MODEL_ID = "zai-org/GLM-5.2"
REQUIRED_PROVIDERS = {"kaggle_cuda", "kaggle_jax_tpu", "kaggle_cpu"}
RUNTIME_KINDS = {"value_op", "full_prefix_stage_decode"}
FULL_PREFIX_PROBE_MODES = {"default", "full-stage"}
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "kaggle-cookies",
    "kaggle-web-storage-state",
    "token=",
    "runtime_proxy",
    "jupyter-proxy",
    '"prompt":',
    '"raw_prompt":',
    '"generated_text":',
    '"raw_generated_text":',
    '"generated_token_ids":',
    '"activation":',
    '"hidden_state":',
    '"logits":',
    '"kv_cache":',
    '"weight_tensor_values":',
    '"safetensors_header_payload":',
)


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 71


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def _compile_errors(path: str) -> list[str]:
    p = Path(path)
    if not p.is_file():
        return ["kernel_missing"]
    try:
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    except SyntaxError as exc:
        return [f"kernel_syntax_error:{exc.lineno}"]
    return []


def _bundle_compile_errors(package_dir: str, bundled_files: list[Any]) -> list[str]:
    errors: list[str] = []
    root = Path(package_dir)
    for item in bundled_files:
        entry = _dict(item)
        relative = str(entry.get("relative_path") or "")
        if not relative:
            errors.append("bundle_path_missing")
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"bundle_file_missing:{relative}")
            continue
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                errors.append(f"bundle_syntax_error:{relative}:{exc.lineno}")
        if not _hash_ok(entry.get("sha256")):
            errors.append(f"bundle_sha_missing:{relative}")
    return errors


def validate_report(report: dict[str, Any], *, require_verified: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != PACKAGE_SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    if report.get("glm52_stage_worker_package_ready") is not True:
        errors.append("package_ready_missing")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))
    model = _dict(report.get("model"))
    if model.get("model_id") != MODEL_ID:
        errors.append("model_id_not_glm52")
    if model.get("fallback_model_allowed_for_success") is not False:
        errors.append("fallback_boundary_missing")
    request_hash_bound = report.get("coordinator_request_id_hash_bound") is True
    if request_hash_bound and not _hash_ok(report.get("coordinator_request_id_hash")):
        errors.append("coordinator_request_hash_bound_but_invalid")
    packages = [item for item in _list(report.get("packages")) if isinstance(item, dict)]
    providers = {str(item.get("provider") or "") for item in packages}
    if not REQUIRED_PROVIDERS.issubset(providers):
        errors.append("required_provider_packages_missing")
    manifest_runtime_kind = str(report.get("stage_runtime_package_kind") or "value_op")
    if manifest_runtime_kind not in RUNTIME_KINDS:
        errors.append("stage_runtime_package_kind_invalid")
    manifest_probe_mode = str(report.get("full_prefix_probe_mode") or "default")
    if manifest_probe_mode not in FULL_PREFIX_PROBE_MODES:
        errors.append("full_prefix_probe_mode_invalid")
    for package in packages:
        provider = str(package.get("provider") or "")
        if provider not in REQUIRED_PROVIDERS:
            errors.append(f"package_provider_not_required:{provider or 'missing'}")
        if _int(package.get("stage_id"), -1) < 0:
            errors.append(f"package_stage_id_missing:{provider or 'missing'}")
        if _int(package.get("stage_count"), 0) <= 0:
            errors.append(f"package_stage_count_missing:{provider or 'missing'}")
        if package.get("expected_stage_report_schema") != "glm52_kaggle_stage_runtime_report_v1":
            errors.append(f"package_stage_report_schema_missing:{provider or 'missing'}")
        owner = str(package.get("kaggle_owner") or "")
        if not owner:
            errors.append(f"package_owner_missing:{provider or 'missing'}")
        kernel_ref = str(package.get("kernel_ref") or "")
        if owner and not kernel_ref.startswith(f"{owner}/"):
            errors.append(f"package_owner_kernel_ref_mismatch:{provider or 'missing'}")
        if package.get("private_kernel") is not True:
            errors.append(f"package_not_private:{provider or 'missing'}")
        if package.get("public_artifact_safe") is not True:
            errors.append(f"package_public_artifact_unsafe:{provider or 'missing'}")
        if package.get("pushed_to_kaggle") is True or package.get("live_run_performed") is True:
            errors.append(f"package_overclaims_live_run:{provider or 'missing'}")
        if package.get("coordinator_request_id_hash_bound") is True and not request_hash_bound:
            errors.append(f"package_hash_bound_without_manifest_hash:{provider or 'missing'}")
        runtime_kind = str(package.get("stage_runtime_package_kind") or "")
        if runtime_kind not in RUNTIME_KINDS:
            errors.append(f"package_runtime_kind_invalid:{provider or 'missing'}")
        if runtime_kind != manifest_runtime_kind:
            errors.append(f"package_runtime_kind_mismatch:{provider or 'missing'}")
        package_probe_mode = str(package.get("full_prefix_probe_mode") or manifest_probe_mode)
        if package_probe_mode != manifest_probe_mode:
            errors.append(f"full_prefix_probe_mode_mismatch:{provider or 'missing'}")
        bundled_files = _list(package.get("bundled_runtime_files"))
        if runtime_kind == "full_prefix_stage_decode":
            if package.get("full_prefix_runtime_bundle_present") is not True:
                errors.append(f"full_prefix_runtime_bundle_missing:{provider or 'missing'}")
            if package.get("embedded_runtime_bundle_present") is not True:
                errors.append(f"embedded_runtime_bundle_missing:{provider or 'missing'}")
            if _int(package.get("embedded_runtime_bundle_file_count")) <= 0:
                errors.append(f"embedded_runtime_bundle_file_count_missing:{provider or 'missing'}")
            bundle_names = {Path(str(_dict(item).get("relative_path") or "")).name for item in bundled_files}
            if "glm52_full_prefix_stage_decode_probe.py" not in bundle_names:
                errors.append(f"full_prefix_probe_not_bundled:{provider or 'missing'}")
            package_dir = str(Path(str(package.get("kernel_path") or "")).parent)
            errors.extend(
                f"{error}:{provider or 'missing'}"
                for error in _bundle_compile_errors(package_dir, bundled_files)
            )
        elif bundled_files:
            errors.append(f"value_op_package_has_runtime_bundle:{provider or 'missing'}")
        layer_range = _list(package.get("stage_layer_range"))
        if len(layer_range) != 2 or _int(layer_range[1]) <= _int(layer_range[0]):
            errors.append(f"package_layer_range_invalid:{provider or 'missing'}")
        if runtime_kind == "full_prefix_stage_decode":
            probe_range = _list(package.get("full_prefix_probe_layer_range"))
            if (
                len(probe_range) != 2
                or _int(probe_range[0]) < _int(layer_range[0])
                or _int(probe_range[1]) > _int(layer_range[1])
                or _int(probe_range[1]) <= _int(probe_range[0])
            ):
                errors.append(f"full_prefix_probe_layer_range_invalid:{provider or 'missing'}")
            covers_full_stage = probe_range == layer_range
            if package.get("full_prefix_probe_covers_full_stage") is not covers_full_stage:
                errors.append(f"full_prefix_probe_full_stage_flag_mismatch:{provider or 'missing'}")
            if manifest_probe_mode == "full-stage" and not covers_full_stage:
                errors.append(f"full_prefix_probe_does_not_cover_full_stage:{provider or 'missing'}")
        errors.extend([f"{error}:{provider or 'missing'}" for error in _compile_errors(str(package.get("kernel_path") or ""))])
    boundary = _dict(report.get("completion_boundary"))
    for key in [
        "package_is_not_runtime_success",
        "kaggle_push_required",
        "live_stage_report_required",
        "same_request_probe_required",
    ]:
        if boundary.get(key) is not True:
            errors.append(f"completion_boundary_missing:{key}")
    verified = report.get("stage_runtime_adapter_verified") is True and report.get("same_request_route_verified") is True
    if require_verified and not verified:
        errors.append("stage_worker_package_not_verified")
    if verified and report.get("blockers"):
        errors.append("verified_package_has_blockers")
    return sorted(set(errors))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-verified", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = load_json(Path(args.report))
    errors = validate_report(report, require_verified=bool(args.require_verified))
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "stage_worker_package_ready": report.get("glm52_stage_worker_package_ready") is True,
        "stage_runtime_adapter_verified": report.get("stage_runtime_adapter_verified") is True,
        "same_request_route_verified": report.get("same_request_route_verified") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_kaggle_stage_worker_package_check: ok={result['ok']} "
            f"errors={len(errors)} ready={result['stage_worker_package_ready']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
