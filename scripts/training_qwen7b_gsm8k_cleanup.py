#!/usr/bin/env python3
"""Remove Qwen7B showcase private payloads and emit a public cleanup audit."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from crowdtensor.qwen15b_training import sha256_file, stable_hash
from crowdtensor.qwen7b_gsm8k_showcase import DATASET_MANIFEST_SCHEMA
from scripts.training_cuda_kaggle_common import public_safety_errors, utc_now


SCHEMA = "crowdtensor_qwen7b_gsm8k_cleanup_audit_v1"
REPORT_NAME = "training_qwen7b_gsm8k_cleanup_audit.json"
PRIVATE_PAYLOADS = {
    "train": ("qwen7b_gsm8k_train_private.json", "private_train_payload_hash"),
    "benchmark": (
        "qwen7b_gsm8k_benchmark_private.json",
        "private_benchmark_payload_hash",
    ),
}


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("cleanup input is not an object")
    return value


def _training_cleanup_ready(report: dict[str, Any]) -> bool:
    cleanup = dict(report.get("cleanup") or {})
    return bool(
        cleanup.get("live_resources_left_running") is False
        and all(
            cleanup.get(key) is True
            for key in (
                "all_four_kernels_deleted",
                "coordinator_stopped",
                "tunnel_stopped",
                "private_runtime_removed",
                "rendezvous_payloads_removed",
                "uncommitted_checkpoint_blobs_removed",
            )
        )
    )


def _benchmark_cleanup_ready(report: dict[str, Any]) -> bool:
    cleanup = dict(report.get("cleanup") or {})
    return bool(
        cleanup.get("live_resources_left_running") is False
        and all(
            cleanup.get(key) is True
            for key in (
                "kernel_deleted",
                "private_dataset_deleted",
                "private_runtime_removed",
            )
        )
    )


def _remove_private_payload(
    *,
    role: str,
    path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    expected_name: str,
    hash_field: str,
    remove: bool,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    expected_hash = str(manifest.get(hash_field) or "")
    location_valid = bool(
        path.name == expected_name and path.parent == manifest_path.parent
    )
    present_before = path.is_file()
    hash_verified = bool(
        present_before
        and expected_hash.startswith("sha256:")
        and sha256_file(path) == expected_hash
    )
    removed = False
    if remove and location_valid and hash_verified:
        path.unlink()
        removed = not path.exists()
    if not location_valid:
        blockers.append("cleanup_private_payload_location_invalid:" + role)
    elif not present_before:
        blockers.append("cleanup_private_payload_missing_before_audit:" + role)
    elif not hash_verified:
        blockers.append("cleanup_private_payload_hash_mismatch:" + role)
    elif not removed:
        blockers.append("cleanup_private_payload_not_removed:" + role)
    return (
        {
            "expected_hash": expected_hash,
            "present_before_cleanup": present_before,
            "hash_verified_before_cleanup": hash_verified,
            "removed": removed,
            "path_public": False,
            "raw_content_public": False,
        },
        blockers,
    )


def cleanup(
    output_dir: str | Path,
    *,
    dataset_manifest_path: str | Path,
    training_report_path: str | Path,
    baseline_report_path: str | Path,
    post_benchmark_report_path: str | Path,
    private_train_payload_path: str | Path,
    private_benchmark_payload_path: str | Path,
    additional_dataset_manifest_paths: list[str | Path] | None = None,
    remove: bool,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(dataset_manifest_path).resolve()
    training_path = Path(training_report_path).resolve()
    baseline_path = Path(baseline_report_path).resolve()
    post_path = Path(post_benchmark_report_path).resolve()
    manifest = _load(manifest_path)
    training = _load(training_path)
    baseline = _load(baseline_path)
    post = _load(post_path)
    payload_paths = {
        "train": Path(private_train_payload_path).resolve(),
        "benchmark": Path(private_benchmark_payload_path).resolve(),
    }
    blockers: list[str] = []
    if manifest.get("schema") != DATASET_MANIFEST_SCHEMA:
        blockers.append("cleanup_dataset_manifest_invalid")
    payload_evidence: dict[str, Any] = {}
    for role, (expected_name, hash_field) in PRIVATE_PAYLOADS.items():
        evidence, payload_blockers = _remove_private_payload(
            role=role,
            path=payload_paths[role],
            manifest_path=manifest_path,
            manifest=manifest,
            expected_name=expected_name,
            hash_field=hash_field,
            remove=remove,
        )
        payload_evidence[role] = evidence
        blockers.extend(payload_blockers)

    additional_manifest_paths = [
        Path(value).resolve() for value in (additional_dataset_manifest_paths or [])
    ]
    if len(set(additional_manifest_paths + [manifest_path])) != len(
        additional_manifest_paths
    ) + 1:
        blockers.append("cleanup_dataset_manifest_paths_not_unique")
    dataset_parents = {manifest_path.parent}
    additional_manifest_hashes: list[str] = []
    for index, additional_path in enumerate(additional_manifest_paths, start=1):
        additional = _load(additional_path)
        additional_manifest_hashes.append(sha256_file(additional_path))
        dataset_parents.add(additional_path.parent)
        if additional.get("schema") != DATASET_MANIFEST_SCHEMA:
            blockers.append(f"cleanup_additional_dataset_manifest_invalid:{index}")
        for role, (expected_name, hash_field) in PRIVATE_PAYLOADS.items():
            evidence_role = f"additional_{index}_{role}"
            evidence, payload_blockers = _remove_private_payload(
                role=evidence_role,
                path=additional_path.parent / expected_name,
                manifest_path=additional_path,
                manifest=additional,
                expected_name=expected_name,
                hash_field=hash_field,
                remove=remove,
            )
            payload_evidence[evidence_role] = evidence
            blockers.extend(payload_blockers)

    transient_names = (".private-raw-gsm8k", ".private-tokenizer-cache")
    if remove:
        for parent in dataset_parents:
            for name in transient_names:
                shutil.rmtree(parent / name, ignore_errors=True)
    transient_absent = all(
        not (parent / name).exists()
        for parent in dataset_parents
        for name in transient_names
    )
    runtime_private_absent = all(
        not (path.parent / ".private-runtime").exists()
        for path in (training_path, baseline_path, post_path)
    )
    training_cleanup = _training_cleanup_ready(training)
    baseline_cleanup = _benchmark_cleanup_ready(baseline)
    post_cleanup = _benchmark_cleanup_ready(post)
    all_payloads_removed = all(
        value.get("removed") is True for value in payload_evidence.values()
    )
    cleanup_ready = bool(
        remove
        and all_payloads_removed
        and transient_absent
        and runtime_private_absent
        and training_cleanup
        and baseline_cleanup
        and post_cleanup
        and not blockers
    )
    report = {
        "schema": SCHEMA,
        "ok": cleanup_ready,
        "cleanup_ready": cleanup_ready,
        "remove_requested": bool(remove),
        "generated_at": utc_now(),
        "evidence_hashes": {
            "dataset_manifest": sha256_file(manifest_path),
            "additional_dataset_manifests": stable_hash(
                additional_manifest_hashes
            ),
            "training_report": sha256_file(training_path),
            "baseline_report": sha256_file(baseline_path),
            "post_benchmark_report": sha256_file(post_path),
        },
        "private_payloads": payload_evidence,
        "dataset_manifest_count": 1 + len(additional_manifest_paths),
        "all_private_payloads_removed": all_payloads_removed,
        "dataset_transient_directories_absent": transient_absent,
        "runtime_private_directories_absent": runtime_private_absent,
        "training_live_cleanup_verified": training_cleanup,
        "baseline_live_cleanup_verified": baseline_cleanup,
        "post_benchmark_live_cleanup_verified": post_cleanup,
        "live_resources_left_running": not (
            training_cleanup and baseline_cleanup and post_cleanup
        ),
        "blockers": sorted(set(blockers)),
        "raw_text_public": False,
        "token_ids_public": False,
        "gold_answers_public": False,
        "credentials_public": False,
        "credential_paths_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    safety = public_safety_errors(report)
    if safety:
        report["ok"] = False
        report["cleanup_ready"] = False
        report["public_artifact_safe"] = False
        report["blockers"].append("cleanup_public_safety_failed")
    report["content_hash"] = stable_hash(report)
    (output / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--post-benchmark-report", required=True)
    parser.add_argument("--private-train-payload", required=True)
    parser.add_argument("--private-benchmark-payload", required=True)
    parser.add_argument("--additional-dataset-manifest", action="append", default=[])
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = cleanup(
        args.output_dir,
        dataset_manifest_path=args.dataset_manifest,
        training_report_path=args.training_report,
        baseline_report_path=args.baseline_report,
        post_benchmark_report_path=args.post_benchmark_report,
        private_train_payload_path=args.private_train_payload,
        private_benchmark_payload_path=args.private_benchmark_payload,
        additional_dataset_manifest_paths=list(args.additional_dataset_manifest),
        remove=args.remove,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "ok": report["ok"],
                    "cleanup_ready": report["cleanup_ready"],
                    "blockers": report["blockers"],
                },
                sort_keys=True,
            )
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
