#!/usr/bin/env python3
"""Audit that Community RC Kaggle and local temporary resources are gone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.model_adapter import stable_hash
from scripts.community_kaggle_reliability_live_probe import _authorized_kaggle_env
from scripts.training_cuda_kaggle_common import run_command


SCHEMA = "crowdtensor_community_cleanup_audit_v1"
PRIVATE_NAMES = (".private", ".private-live", ".private-runtime")


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("community_cleanup_evidence_invalid")
    return value


def _source_summary(path: str | Path) -> dict[str, Any]:
    value = _read(path)
    cleanup = value.get("cleanup") if isinstance(value.get("cleanup"), dict) else {}
    no_live = cleanup.get("live_resources_left_running") is not True
    return {
        "file_name": Path(path).name,
        "source_schema": str(value.get("schema") or ""),
        "source_content_hash": str(value.get("content_hash") or ""),
        "cleanup_verified": value.get("cleanup_verified") is True,
        "live_resources_left_running": not no_live,
        "public_artifact_safe": value.get("public_artifact_safe") is True,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [_source_summary(path) for path in args.evidence]
    private_roots = [
        candidate
        for path in args.evidence
        for candidate in (Path(path).expanduser().resolve().parent / name for name in PRIVATE_NAMES)
        if candidate.exists()
    ]
    with _authorized_kaggle_env(args) as env:
        listing = run_command(
            [
                "kaggle",
                "kernels",
                "list",
                "--mine",
                "--search",
                "ct-community",
                "--page-size",
                "200",
                "--csv",
            ],
            env=env,
            timeout=90,
        )
    matching = [
        line
        for line in str(listing.get("output_tail") or "").splitlines()[1:]
        if "ct-community-" in line.lower()
    ]
    docker_containers = run_command(
        ["docker", "ps", "-a", "--format", "{{.Names}}", "--filter", "name=ct-community"],
        env=dict(__import__("os").environ),
        timeout=60,
    )
    docker_images = run_command(
        [
            "docker",
            "images",
            "--filter",
            "reference=crowdtensor-community-rc:*",
            "--format",
            "{{.Repository}}:{{.Tag}}",
        ],
        env=dict(__import__("os").environ),
        timeout=60,
    )
    container_count = len(
        [line for line in str(docker_containers.get("output_tail") or "").splitlines() if line.strip()]
    ) if docker_containers.get("ok") else -1
    image_count = len(
        [
            line
            for line in str(docker_images.get("output_tail") or "").splitlines()
            if "crowdtensor-community-rc" in line.lower()
        ]
    ) if docker_images.get("ok") else -1
    sources_ok = bool(
        sources
        and all(
            item["cleanup_verified"]
            and item["live_resources_left_running"] is False
            and item["public_artifact_safe"]
            for item in sources
        )
    )
    report = {
        "schema": SCHEMA,
        "ok": False,
        "goal_resource_scope": "CrowdTensor Community RC only",
        "evidence_source_count": len(sources),
        "evidence_sources": sources,
        "source_cleanup_verified": sources_ok,
        "kaggle_query_authenticated": listing.get("ok") is True,
        "matching_kaggle_resource_count": len(matching),
        "all_community_kaggle_resources_deleted": listing.get("ok") is True and not matching,
        "local_private_runtime_count": len(private_roots),
        "all_evidence_private_runtimes_removed": not private_roots,
        "community_docker_container_count": container_count,
        "community_docker_image_count": image_count,
        "community_docker_resources_removed": container_count == 0 and image_count == 0,
        "live_resources_left_running": True,
        "credential_values_public": False,
        "credential_paths_public": False,
        "kaggle_account_identity_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["live_resources_left_running"] = not bool(
        sources_ok
        and report["all_community_kaggle_resources_deleted"]
        and report["all_evidence_private_runtimes_removed"]
        and report["community_docker_resources_removed"]
    )
    report["ok"] = not report["live_resources_left_running"]
    privacy = scan_public_value(report)
    report["public_safety"] = privacy
    report["public_artifact_safe"] = privacy["ok"] is True
    report["ok"] = bool(report["ok"] and privacy["ok"])
    report["content_hash"] = stable_hash(report)
    (output / "community_cleanup_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evidence", action="append", required=True)
    parser.add_argument("--kaggle-token-file", required=True)
    parser.add_argument("--kaggle-account-label", default="")
    parser.add_argument("--kaggle-username", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(args)
    print(json.dumps(report, sort_keys=True) if args.json else f"cleanup_ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
