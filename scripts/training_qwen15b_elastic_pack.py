#!/usr/bin/env python3
"""Re-evaluate retained live evidence into a canonical elastic artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crowdtensor.qwen15b_training import sha256_file, stable_hash
from scripts.training_cuda_kaggle_common import public_safety_errors, utc_now
from scripts.training_qwen15b_elastic_live_probe import SCHEMA, _evaluate


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pack_report(source_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(source_path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("elastic_live_source_schema_invalid")
    old = dict(value.get("old_generation") or {})
    new = dict(value.get("new_generation") or {})
    midpoint = dict(value.get("midpoint_status") or {})
    final = dict(value.get("final_status") or {})
    rendezvous = dict(value.get("rendezvous") or {})
    pause = dict(value.get("full_offline_pause") or {})
    observations = list(pause.get("observations") or [])
    observed_seconds = float(pause.get("observed_seconds") or 0)
    evidence = _evaluate(
        old=old,
        new=new,
        midpoint=midpoint,
        final=final,
        rendezvous=rendezvous,
        pause_observations=observations,
        pause_seconds=observed_seconds,
    )
    generation_hashes = set(old.get("kernel_ref_hashes") or []) | set(
        new.get("kernel_ref_hashes") or []
    )
    generation_cleanup = bool(
        len(generation_hashes) == 4
        and old.get("all_kernels_deleted") is True
        and new.get("all_kernels_deleted") is True
        and len(old.get("deletions") or []) == 2
        and len(new.get("deletions") or []) == 2
        and all(
            item.get("deleted_or_absent") is True
            for item in [*(old.get("deletions") or []), *(new.get("deletions") or [])]
        )
    )
    cleanup = dict(value.get("cleanup") or {})
    cleanup["all_four_kernels_deleted"] = generation_cleanup
    cleanup["live_resources_left_running"] = not bool(
        generation_cleanup
        and cleanup.get("coordinator_stopped") is True
        and cleanup.get("tunnel_stopped") is True
        and cleanup.get("private_runtime_removed") is True
    )
    cleanup_ok = bool(
        not cleanup["live_resources_left_running"]
        and cleanup.get("rendezvous_payloads_removed") is True
        and cleanup.get("uncommitted_checkpoint_blobs_removed") is True
    )
    redundant_retry = list(value.pop("final_cleanup_deletions", []) or [])
    value.update(
        {
            "evidence": evidence,
            "cleanup": cleanup,
            "cleanup_evidence": {
                "authoritative_source": "per_generation_immediate_deletions",
                "old_generation_deleted": old.get("all_kernels_deleted") is True,
                "new_generation_deleted": new.get("all_kernels_deleted") is True,
                "distinct_kernel_ref_hash_count": len(generation_hashes),
                "successful_initial_delete_count": sum(
                    item.get("deleted_or_absent") is True
                    for item in [
                        *(old.get("deletions") or []),
                        *(new.get("deletions") or []),
                    ]
                ),
                "redundant_post_delete_retry_count": len(redundant_retry),
                "redundant_retry_authoritative": False,
                "initial_delete_evidence_not_overridden_by_redundant_retry": True,
                "live_resources_left_running": cleanup["live_resources_left_running"],
                "public_artifact_safe": True,
            },
            "artifact_repack": {
                "source_report_hash": sha256_file(source),
                "source_live_run_reused": True,
                "fresh_live_run_performed_for_repack": False,
                "runtime_measurements_modified": False,
                "corrected_rules": [
                    "zero_live_miner_count_preserves_numeric_zero",
                    "step5_reassignment_filters_revoked_epoch",
                    "successful_generation_delete_precedes_redundant_cleanup_retry",
                ],
                "public_artifact_safe": True,
            },
            "blockers": [],
            "ok": bool(evidence.get("verified") and cleanup_ok),
            "elastic_volunteer_training_ready": bool(
                evidence.get("verified") and cleanup_ok
            ),
            "finished_at": utc_now(),
        }
    )
    safety = public_safety_errors(value)
    value["public_artifact_safe"] = not safety
    if safety:
        value["safety_errors"] = safety
        value["ok"] = False
        value["elastic_volunteer_training_ready"] = False
        value["blockers"] = ["elastic_public_artifact_safety_failed"]
    else:
        value.pop("safety_errors", None)
    value["artifact_content_hash"] = stable_hash(
        {key: item for key, item in value.items() if key != "artifact_content_hash"}
    )
    output = Path(output_dir).resolve()
    _write(output / "training_qwen15b_elastic_live_probe.json", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = pack_report(args.live_report, args.output_dir)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "training_qwen15b_elastic_pack "
            f"ok={report['ok']} ready={report['elastic_volunteer_training_ready']}"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
