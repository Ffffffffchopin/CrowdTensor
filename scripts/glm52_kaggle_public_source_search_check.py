#!/usr/bin/env python3
"""Validate public Kaggle GLM 5.2 source-search evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_kaggle_public_source_search as search  # noqa: E402


SCHEMA = "glm52_kaggle_public_source_search_check_v1"


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


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != search.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    if report.get("glm52_kaggle_public_source_search_ready") is not True:
        errors.append("search_ready_missing")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = search.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))
    if _int(report.get("query_count")) < 1:
        errors.append("queries_missing")
    if _int(report.get("model_result_count")) < 0 or _int(report.get("dataset_result_count")) < 0:
        errors.append("result_count_invalid")

    model_candidates = _list(report.get("compatible_model_candidates"))
    dataset_candidates = _list(report.get("compatible_dataset_candidates"))
    source_verified = report.get("kaggle_attach_source_verified") is True
    if source_verified != bool(model_candidates or dataset_candidates):
        errors.append("source_verified_candidate_mismatch")
    if source_verified:
        if not _list(report.get("recommended_kaggle_kernel_model_sources")) and not dataset_candidates:
            errors.append("source_verified_without_attach_ref")
    else:
        blockers = set(str(item) for item in _list(report.get("blockers")))
        if "kaggle_models_glm52_weight_source_not_found" not in blockers:
            errors.append("missing_model_not_found_blocker")
        if "kaggle_datasets_glm52_weight_source_not_found" not in blockers:
            errors.append("missing_dataset_not_found_blocker")

    for item in _list(report.get("model_results")):
        row = _dict(item)
        if row.get("public_artifact_safe") is not True:
            errors.append("model_result_public_artifact_unsafe")
        if row.get("description_public_excerpt_included") is not False:
            errors.append("model_description_excerpt_public")
        if row.get("attachable_glm52_weight_source_candidate") is True:
            if row.get("glm52_text_match") is not True:
                errors.append("model_candidate_without_glm52_match")
            if row.get("weight_source_signal") is not True:
                errors.append("model_candidate_without_weight_signal")
    for item in _list(report.get("dataset_results")):
        row = _dict(item)
        if row.get("public_artifact_safe") is not True:
            errors.append("dataset_result_public_artifact_unsafe")
        if row.get("description_public_excerpt_included") is not False:
            errors.append("dataset_description_excerpt_public")
        if row.get("attachable_glm52_weight_source_candidate") is True:
            if row.get("glm52_text_match") is not True:
                errors.append("dataset_candidate_without_glm52_match")
            if row.get("weight_source_signal") is not True:
                errors.append("dataset_candidate_without_weight_signal")

    safety = _dict(report.get("safety"))
    for key in [
        "public_artifact_safe",
    ]:
        if safety.get(key) is not True:
            errors.append(f"safety_flag_not_true:{key}")
    for key in [
        "credentials_public",
        "cookies_public",
        "signed_url_public",
        "raw_prompt_public",
        "raw_generated_text_public",
        "generated_token_ids_public",
        "activation_public",
        "hidden_state_public",
        "logits_public",
        "kv_cache_public",
        "weight_tensor_values_public",
    ]:
        if safety.get(key) is not False:
            errors.append(f"safety_flag_not_false:{key}")
    return sorted(set(errors))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-source", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = load_json(Path(args.report))
    errors = validate_report(report)
    if args.require_source and report.get("kaggle_attach_source_verified") is not True:
        errors.append("required_kaggle_attach_source_missing")
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "kaggle_attach_source_verified": report.get("kaggle_attach_source_verified") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_kaggle_public_source_search_check: ok={result['ok']} "
            f"errors={len(errors)} source_verified={result['kaggle_attach_source_verified']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
