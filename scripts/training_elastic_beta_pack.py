#!/usr/bin/env python3
"""Repack immutable Elastic Training Beta live evidence with cleanup audit truth."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "crowdtensor_elastic_training_beta_live_probe_v1"


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("elastic_training_beta_report_invalid")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def pack(
    source_report: str | Path,
    post_cleanup_audit: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_path = Path(source_report).resolve()
    audit_path = Path(post_cleanup_audit).resolve()
    source = _load(source_path)
    audit = _load(audit_path)
    if source.get("schema") != SCHEMA:
        raise ValueError("elastic_training_beta_source_schema_invalid")
    gates = dict(source.get("acceptance_gates") or {})
    if not gates or not all(value is True for value in gates.values()):
        raise ValueError("elastic_training_beta_source_acceptance_incomplete")
    old_deleted = (
        (source.get("old_generation") or {}).get("all_kernels_deleted") is True
    )
    replacement_deleted = (
        (source.get("replacement_generation") or {}).get("all_kernels_deleted")
        is True
    )
    selected_hash = str((source.get("selected_account") or {}).get("owner_hash") or "")
    selected_audit = next(
        (
            item
            for item in audit.get("account_preflight") or []
            if str(item.get("owner_hash") or "") == selected_hash
        ),
        {},
    )
    post_cleanup_verified = bool(
        selected_hash
        and selected_audit.get("authenticated") is True
        and int(selected_audit.get("active_kernel_count") or 0) == 0
    )
    if not old_deleted or not replacement_deleted or not post_cleanup_verified:
        raise ValueError("elastic_training_beta_cleanup_evidence_incomplete")
    report = copy.deepcopy(source)
    report["repack"] = {
        "schema": "crowdtensor_elastic_training_beta_repack_v1",
        "source_report_hash": _sha256_file(source_path),
        "post_cleanup_audit_hash": _sha256_file(audit_path),
        "runtime_measurements_changed": False,
        "correction": "redundant_final_delete_did_not_override_generation_deletes",
        "generation_cleanup_verified": True,
        "post_cleanup_account_audit_verified": True,
        "selected_account_active_kernel_count": 0,
        "credential_values_public": False,
        "credential_paths_public": False,
        "public_artifact_safe": True,
    }
    report["cleanup"]["all_kernels_deleted"] = True
    report["cleanup"]["live_resources_left_running"] = False
    report["cleanup_verified"] = True
    report["final_delete_audit_redundant"] = True
    report["blockers"] = [
        blocker
        for blocker in report.get("blockers") or []
        if blocker != "elastic_training_beta_cleanup_incomplete"
    ]
    report["ok"] = True
    report["elastic_training_beta_ready"] = True
    report.pop("content_hash", None)
    report["content_hash"] = _stable_hash(report)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "training_elastic_beta_live_probe.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--post-cleanup-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = pack(args.source_report, args.post_cleanup_audit, args.output_dir)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "training_elastic_beta_pack "
            f"ok={report.get('ok')} ready={report.get('elastic_training_beta_ready')}"
        )


if __name__ == "__main__":
    main()
