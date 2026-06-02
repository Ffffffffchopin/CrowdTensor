#!/usr/bin/env python3
"""CI-safe checks for Petals-class P2P candidate artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import petals_class_p2p_candidate_pack as pack  # noqa: E402


SCHEMA = "petals_class_p2p_candidate_check_v1"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fake_real_p2p_report(*, mode: str, generated_tokens: int = 8, requeue: bool = False) -> dict[str, Any]:
    codes = [
        "real_p2p_swarm_inference_core_rc_ready",
        "libp2p_discovery_backend_ready",
        "p2p_peer_identity_ready",
        "p2p_provider_dht_ready",
        "external_libp2p_stage_discovery_ready",
        "external_libp2p_generate_ready",
        "peer_scoring_ready",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "kaggle_kernels_deleted",
        "real_p2p_kaggle_private_artifacts_cleaned",
    ]
    if mode == "local-smoke":
        codes.append("hivemind_petals_class_alpha_local_ready")
    if mode == "kaggle-runtime-smoke":
        codes.append("real_p2p_kaggle_runtime_smoke_ready")
    if requeue:
        codes.extend([
            "external_stage_requeue_ready",
            "live_stage0_requeue_ready",
            "rescue_miner_used",
            "accepted_result_after_requeue",
        ])
    report = {
        "schema": "real_p2p_swarm_inference_core_rc_v1",
        "ok": True,
        "mode": mode,
        "hf_model_id": "sshleifer/tiny-gpt2",
        "diagnosis_codes": codes,
        "p2p": {
            "discovery_backend": "libp2p-kad",
            "provider_count": 4,
            "stage0_provider_count": 1,
            "stage1_provider_count": 1,
            "peer_scoring": {"schema": "real_p2p_peer_scoring_v1", "ok": True},
        },
        "generation": {"generated_token_count": generated_tokens, "decoded_tokens_match": True},
        "stage_assignment": {"distinct_stage_miners": True},
        "ledger": {"accepted_rows": generated_tokens * 2},
        "kaggle_lifecycle": {"kernels_deleted": True, "local_private_artifacts_cleaned": True},
    }
    if requeue:
        report["live_requeue_summary"] = {
            "enabled": True,
            "failure_mode": "kill-stage0-after-claim",
            "target_stage": "stage0",
            "victim_miner_id": "real-p2p-rc-kaggle-stage0-victim",
            "rescue_miner_id": "real-p2p-rc-kaggle-stage0-rescue",
            "claim_observed": True,
            "victim_kernel_deleted": True,
            "lease_expired": True,
            "rescue_miner_used": True,
            "rescued_result": True,
            "accepted_result_after_requeue": True,
            "victim_result_accepted": False,
        }
    return report


def add_safe_batch_stream(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("diagnosis_codes", []).extend([
        "external_real_p2p_generate_batch_ready",
        "public_swarm_generate_batch_ready",
        "external_real_p2p_generate_stream_ready",
        "public_swarm_generate_stream_ready",
        "public_swarm_generate_stream_endpoint_ready",
    ])
    payload["batch"] = {
        "enabled": True,
        "expected_request_count": 2,
        "observed_request_count": 2,
        "max_request_count": 4,
        "prompt_hashes": ["sha256:a", "sha256:b"],
        "prompt_char_counts": [12, 13],
        "result_count": 2,
        "results": [
            {
                "request_id": "req-0",
                "prompt_hash": "sha256:a",
                "generated_token_count": 8,
                "max_new_tokens": 8,
                "generated_text_hash": "sha256:ga",
                "decoded_tokens_match": True,
                "multi_token_generation_ready": True,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            },
            {
                "request_id": "req-1",
                "prompt_hash": "sha256:b",
                "generated_token_count": 8,
                "max_new_tokens": 8,
                "generated_text_hash": "sha256:gb",
                "decoded_tokens_match": True,
                "multi_token_generation_ready": True,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            },
        ],
        "batch_generation_ready": True,
        "raw_prompts_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }
    payload["stream"] = {
        "enabled": True,
        "requested": True,
        "event_count": 16,
        "source": "admin-session-stream",
        "endpoint_ready": True,
        "progress": {
            "stream_progress_complete": True,
            "all_token_events_ready": True,
            "monotonic_progress": True,
            "expected_request_count": 2,
            "per_request_progress": [
                {
                    "request_key": "req-0",
                    "request_id": "req-0",
                    "prompt_hash": "sha256:a",
                    "event_count": 8,
                    "observed_token_counts": list(range(1, 9)),
                    "max_observed_token_count": 8,
                    "target_token_count": 8,
                    "monotonic_progress": True,
                    "stream_progress_complete": True,
                },
                {
                    "request_key": "req-1",
                    "request_id": "req-1",
                    "prompt_hash": "sha256:b",
                    "event_count": 8,
                    "observed_token_counts": list(range(1, 9)),
                    "max_observed_token_count": 8,
                    "target_token_count": 8,
                    "monotonic_progress": True,
                    "stream_progress_complete": True,
                },
            ],
            "per_request_progress_complete": True,
            "per_request_monotonic_progress": True,
            "observed_token_counts": list(range(1, 9)),
            "max_observed_token_count": 8,
            "max_new_tokens": 8,
            "source": "admin-session-stream",
        },
        "events": [
            {
                "schema": "session_stream_event_v1",
                "session_id": "session-1",
                "task_id": "task-1",
                "miner_id": "stage1",
                "stage_id": 1,
                "request_id": "req-0",
                "prompt_hash": "sha256:a",
                "generated_token_count": 8,
                "max_new_tokens": 8,
                "generation_step": 7,
                "generated_text_hash": "sha256:stream",
                "decoded_tokens_match": True,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            },
            {
                "schema": "session_stream_event_v1",
                "session_id": "session-1",
                "task_id": "task-2",
                "miner_id": "stage1",
                "stage_id": 1,
                "request_id": "req-1",
                "prompt_hash": "sha256:b",
                "generated_token_count": 8,
                "max_new_tokens": 8,
                "generation_step": 7,
                "generated_text_hash": "sha256:stream-b",
                "decoded_tokens_match": True,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            }
        ],
        "stream_generation_ready": True,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }
    return payload


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="crowdtensor_petals_candidate_check_")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / "local.json"
    runtime_path = output_dir / "runtime.json"
    external_path = output_dir / "external.json"
    requeue_path = output_dir / "requeue.json"
    batch_stream_path = output_dir / "batch-stream.json"
    write_json(local_path, fake_real_p2p_report(mode="local-smoke"))
    write_json(runtime_path, fake_real_p2p_report(mode="kaggle-runtime-smoke"))
    write_json(external_path, fake_real_p2p_report(mode="kaggle-auto"))
    write_json(requeue_path, fake_real_p2p_report(mode="kaggle-auto", requeue=True))
    write_json(batch_stream_path, add_safe_batch_stream(fake_real_p2p_report(mode="local-smoke")))
    report = pack.build_report(pack.parse_args([
        "evidence-import",
        "--output-dir",
        str(output_dir / "candidate"),
        "--local-report",
        str(local_path),
        "--runtime-smoke-report",
        str(runtime_path),
        "--external-report",
        str(external_path),
        "--requeue-report",
        str(requeue_path),
        "--batch-stream-report",
        str(batch_stream_path),
        "--max-new-tokens",
        "8",
    ]))
    errors: list[str] = []
    required = {
        "petals_class_p2p_candidate_ready",
        "external_libp2p_stage_discovery_ready",
        "external_libp2p_generate_ready",
        "external_stage_requeue_ready",
        "rescue_miner_used",
        "accepted_result_after_requeue",
        "p2p_live_requeue_rescue_ready",
        "p2p_victim_result_not_accepted",
        "peer_scoring_ready",
        "p2p_candidate_model_id_consistent",
        "p2p_candidate_batch_generation_ready",
        "p2p_candidate_stream_generation_ready",
    }
    codes = set(report.get("diagnosis_codes") or [])
    if report.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("candidate_not_ready")
    for code in sorted(required - codes):
        errors.append(f"missing_code:{code}")
    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    if int(candidate.get("external_generated_token_count") or 0) < 8:
        errors.append("generated_token_count_too_low")
    if candidate.get("p2p_live_requeue_ready") is not True:
        errors.append("live_requeue_not_ready")
    if candidate.get("victim_result_not_accepted") is not True:
        errors.append("victim_result_not_rejected")
    if candidate.get("model_id_consistent") is not True:
        errors.append("model_id_not_consistent")
    if candidate.get("batch_ready") is not True:
        errors.append("batch_not_ready")
    if candidate.get("stream_ready") is not True:
        errors.append("stream_not_ready")
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "output_dir": str(output_dir),
        "errors": errors,
        "candidate": {
            "ok": report.get("ok"),
            "diagnosis_codes": report.get("diagnosis_codes"),
            "generated_token_count": candidate.get("external_generated_token_count"),
            "batch_ready": candidate.get("batch_ready"),
            "stream_ready": candidate.get("stream_ready"),
        },
        "diagnosis_codes": ["petals_class_p2p_candidate_check_ready"] if not errors else ["petals_class_p2p_candidate_check_blocked"],
    }
    write_json(output_dir / "petals_class_p2p_candidate_check.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Petals-class P2P candidate contracts.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"Petals-class P2P candidate check ready: {result.get('ok')}")
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
