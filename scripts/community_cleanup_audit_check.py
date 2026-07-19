#!/usr/bin/env python3
"""Strict checker for the Community RC final cleanup audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.model_adapter import stable_hash
from scripts.community_cleanup_audit import SCHEMA


def check(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    errors: list[str] = []
    if value.get("schema") != SCHEMA:
        errors.append("community_cleanup_schema_invalid")
    for field in (
        "ok",
        "source_cleanup_verified",
        "kaggle_query_authenticated",
        "all_community_kaggle_resources_deleted",
        "all_evidence_private_runtimes_removed",
        "community_docker_resources_removed",
        "public_artifact_safe",
    ):
        if value.get(field) is not True:
            errors.append("community_cleanup_" + field + "_missing")
    if int(value.get("evidence_source_count") or 0) < 3:
        errors.append("community_cleanup_evidence_count_insufficient")
    if any(
        int(value.get(field) or 0) != 0
        for field in (
            "matching_kaggle_resource_count",
            "local_private_runtime_count",
            "community_docker_container_count",
            "community_docker_image_count",
        )
    ):
        errors.append("community_cleanup_resource_count_nonzero")
    if value.get("live_resources_left_running") is not False:
        errors.append("community_cleanup_live_resources_left_running")
    supplied = str(value.get("content_hash") or "")
    if supplied != stable_hash({key: item for key, item in value.items() if key != "content_hash"}):
        errors.append("community_cleanup_content_hash_invalid")
    privacy = scan_public_value(value)
    if privacy["ok"] is not True:
        errors.append("community_cleanup_public_safety_invalid")
    return {
        "schema": "crowdtensor_community_cleanup_audit_check_v1",
        "ok": not errors,
        "errors": sorted(set(errors)),
        "public_safety": privacy,
        "public_artifact_safe": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(args.report)
    print(json.dumps(result, sort_keys=True) if args.json else f"ok={result['ok']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
