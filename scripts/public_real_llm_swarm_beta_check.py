#!/usr/bin/env python3
"""Acceptance checks for Public Real-LLM Swarm Inference Beta v1."""

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

import public_real_llm_swarm_beta_pack as pack  # noqa: E402
from petals_class_p2p_candidate_check import add_safe_batch_stream  # noqa: E402


SCHEMA = "public_real_llm_swarm_beta_check_v1"
ARTIFACT_SUMMARY_SCHEMA = "public_real_llm_swarm_beta_check_artifact_summary_v1"
REVIEW_SUMMARY_SCHEMA = "public_real_llm_swarm_beta_check_review_summary_v1"
REQUIRED_RELEASE_CODES = {
    "public_real_llm_swarm_beta_ready",
    "release_evidence_ready",
    "user_facing_serve_join_generate_ready",
    "public_real_llm_product_path_ready",
    "public_real_llm_swarm_beta_product_model_match_ready",
    "cpu_real_llm_default_ready",
    "external_kaggle_two_stage_ready",
    "external_real_llm_generate_ready",
    "external_stage_requeue_ready",
    "p2p_ready_product_beta",
    "real_p2p_discovery_candidate_ready",
    "peer_scoring_ready",
    "p2p_live_requeue_rescue_ready",
    "p2p_victim_result_not_accepted",
    "public_real_llm_swarm_beta_public_swarm_v2_ready",
    "public_real_llm_swarm_beta_p2p_user_path_ready",
    "public_swarm_inference_v2_ready",
    "public_swarm_v2_local_p2p_generate_ready",
    "public_swarm_v2_16_token_generation_ready",
    "public_swarm_v2_external_stage_rows_ready",
    "public_swarm_v2_dual_stage_kv_cache_ready",
    "public_swarm_v2_model_match_ready",
    "public_swarm_v2_signed_or_real_p2p_ready",
    "public_swarm_v2_real_p2p_local_ready",
    "public_swarm_v2_stage_requeue_rescue_ready",
    "public_real_llm_swarm_beta_v2_real_p2p_local_ready",
    "public_real_llm_swarm_beta_v2_batch_ready",
    "public_real_llm_swarm_beta_v2_stream_ready",
    "public_swarm_v2_batch_generation_ready",
    "public_swarm_v2_stream_generation_ready",
    "public_real_llm_swarm_beta_kv_cache_ready",
    "public_real_llm_swarm_beta_kv_cache_model_match_ready",
    "usable_real_llm_kv_cache_ready",
    "p2p_real_generate_kv_cache_ready",
    "real_llm_stage0_kv_cache_v1_ready",
    "real_llm_stage1_kv_cache_v1_ready",
    "stage0_kv_cache_hits_ready",
    "stage1_kv_cache_hits_ready",
    "optional_cuda_fail_closed_ready",
    "serve_join_generate_runbook_ready",
    "read_only_workload",
    "not_production",
    "not_coordinator_free",
    "not_hivemind_petals_production",
    "not_large_model_serving",
}
LOCAL_MODEL_VARIANT_REQUIRED_CODES = {
    "public_real_llm_swarm_beta_local_model_variant_ready",
    "public_real_llm_swarm_beta_local_model_variant_only",
    "external_validation_not_claimed",
    "public_real_llm_swarm_beta_local_model_variant_v2_ready",
    "public_swarm_inference_v2_local_model_variant_ready",
    "public_swarm_v2_local_model_variant_ready",
    "public_swarm_v2_external_validation_not_claimed",
    "public_swarm_v2_local_p2p_generate_ready",
    "public_swarm_v2_16_token_generation_ready",
    "public_swarm_v2_dual_stage_kv_cache_ready",
    "public_swarm_v2_model_match_ready",
    "public_swarm_v2_signed_or_real_p2p_ready",
    "public_swarm_v2_real_p2p_local_ready",
    "public_swarm_v2_real_p2p_local_requeue_ready",
    "public_swarm_v2_stage_requeue_rescue_ready",
    "public_real_llm_swarm_beta_v2_real_p2p_local_ready",
    "public_real_llm_swarm_beta_v2_real_p2p_local_requeue_ready",
    "public_real_llm_swarm_beta_v2_batch_ready",
    "public_real_llm_swarm_beta_v2_stream_ready",
    "public_real_llm_swarm_beta_kv_cache_ready",
    "public_real_llm_swarm_beta_kv_cache_model_match_ready",
    "usable_real_llm_kv_cache_ready",
    "optional_cuda_fail_closed_ready",
    "serve_join_generate_runbook_ready",
    "read_only_workload",
    "not_production",
    "not_coordinator_free",
    "not_hivemind_petals_production",
    "not_large_model_serving",
}
LOCAL_MODEL_VARIANT_BLOCKED_CODES = {
    "public_real_llm_swarm_beta_ready",
    "release_evidence_ready",
    "external_runtime_verified",
    "external_stage_requeue_ready",
    "p2p_ready_product_beta",
    "public_swarm_v2_external_stage_rows_ready",
    "cuda_runtime_available",
    "public_swarm_gpu_beta_ready",
    "gpu_runtime_ready",
}
SECRET_FRAGMENTS = [
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "CROWDTENSOR_P2P_PEER_SECRET=",
    "lease_token",
    "idempotency_key",
    "Bearer ",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
]


def write(path: Path, payload: str = "{}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def completed(payload: dict[str, Any], returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


def artifact_file_path(payload: dict[str, Any], name: str, fallback: str) -> Path:
    output_dir = Path(str(payload.get("output_dir") or ""))
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    artifact = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
    raw_path = str(artifact.get("path") or "")
    if raw_path:
        path = Path(raw_path)
        return path if path.is_absolute() else output_dir / path
    return output_dir / fallback


def artifact_text(payload: dict[str, Any], name: str, fallback: str) -> str:
    path = artifact_file_path(payload, name, fallback)
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def artifact_json(payload: dict[str, Any], name: str, fallback: str) -> dict[str, Any]:
    text = artifact_text(payload, name, fallback)
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def read_json_file(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        return {}, ["beta_report_missing"]
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, ["beta_report_invalid_json"]
    except OSError:
        return {}, ["beta_report_unreadable"]
    if not isinstance(loaded, dict):
        return {}, ["beta_report_not_object"]
    return loaded, []


def payload_for_existing_beta_report(report_path: Path) -> tuple[dict[str, Any], list[str]]:
    payload, errors = read_json_file(report_path)
    if errors:
        return {}, errors
    resolved_report = report_path.resolve()
    report_dir = resolved_report.parent
    payload = dict(payload)
    payload["output_dir"] = str(report_dir)
    artifacts = dict(payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {})
    artifacts["public_real_llm_swarm_beta_json"] = {
        "kind": "public_real_llm_swarm_beta",
        "path": str(resolved_report),
        "present": True,
        "schema": pack.SCHEMA,
        "ok": payload.get("ok"),
    }
    payload["artifacts"] = artifacts
    return payload, []


def append_sensitive_artifact_errors(errors: list[str], *, artifact_name: str, text: str) -> None:
    for fragment in SECRET_FRAGMENTS:
        if fragment in text:
            errors.append(f"sensitive_artifact:{artifact_name}:{fragment}")


def model_block(model_id: str) -> dict[str, Any]:
    return {
        "expected_hf_model_id": model_id,
        "observed_hf_model_id": model_id,
        "model_id_present": True,
        "model_id_match": True,
        "compatible": True,
        "default_model_retained_evidence": False,
    }


def fake_product_payload(model_id: str = "sshleifer/tiny-gpt2", *, tokens: int = 16) -> dict[str, Any]:
    return {
        "schema": pack.PRODUCT_SCHEMA,
        "ok": True,
        "mode": "local-loopback",
        "product_beta": {
            "ready": True,
            "workload_type": pack.WORKLOAD_TYPE,
            "hf_model_id": model_id,
            "max_new_tokens": tokens,
            "user_surface": ["serve", "join", "generate"],
        },
        "diagnosis_codes": [
            "public_swarm_product_beta_ready",
            "public_swarm_product_beta_user_path_ready",
            "serve_ready",
            "stage0_join_ready",
            "stage1_join_ready",
            "generate_ready",
            "serve_join_generate_loop_ready",
            "remote_generate_session_ready",
            "public_swarm_generate_ready",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "cpu_fallback_ready",
            "local_cpu_inference_ready",
            "read_only_workload",
            "not_production",
            "not_p2p",
            "not_large_model_serving",
        ],
    }


def fake_gpu_payload() -> dict[str, Any]:
    return {
        "schema": pack.GPU_SCHEMA,
        "ok": True,
        "mode": "local-smoke",
        "beta": {
            "ready": True,
            "backend": "hf_transformers_cuda",
            "cuda_available": False,
        },
        "diagnosis_codes": [
            "public_swarm_gpu_beta_smoke_ready",
            "gpu_runtime_smoke_ready",
            "cuda_runtime_unavailable",
            "read_only_workload",
            "not_production",
            "not_p2p",
        ],
    }


def fake_external_payload(*, tokens: int = 16) -> dict[str, Any]:
    return {
        "schema": pack.REAL_LLM_INTERNET_BETA_SCHEMA,
        "ok": True,
        "mode": "kaggle-auto",
        "workload": {
            "workload_type": pack.WORKLOAD_TYPE,
            "hf_model_id": "sshleifer/tiny-gpt2",
            "max_new_tokens": tokens,
        },
        "diagnosis_codes": [
            "real_llm_internet_beta_ready",
            "external_runtime_verified",
            "generation_complete",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "external_stage_requeue_ready",
            "live_stage0_requeue_ready",
            "kaggle_kernels_deleted",
        ],
        "generation": {
            "generated_token_count": tokens,
            "max_new_tokens": tokens,
            "generated_text_hash": "sha256:fake",
            "decoded_tokens_match": True,
            "multi_token_generation_ready": True,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        },
        "live_requeue_summary": {
            "enabled": True,
            "failure_mode": "kill-stage0-after-claim",
            "target_stage": "stage0",
            "victim_miner_id": "internet-real-llm-beta-stage0-victim",
            "rescue_miner_id": "internet-real-llm-beta-stage0-rescue",
            "claim_observed": True,
            "victim_kernel_deleted": True,
            "lease_expired": True,
            "rescued_result": True,
            "victim_result_accepted": False,
        },
    }


def fake_p2p_payload(*, tokens: int = 16) -> dict[str, Any]:
    payload = {
        "schema": pack.P2P_SCHEMA,
        "ok": True,
        "mode": "evidence-import",
        "candidate": {
            "peer_scoring_ready": True,
            "hf_model_id": "sshleifer/tiny-gpt2",
            "external_generated_token_count": tokens,
            "accepted_rows": tokens * 2,
            "external_stage_requeue_ready": True,
            "p2p_live_requeue_ready": True,
            "victim_result_not_accepted": True,
            "live_requeue_summary": {
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
            },
        },
        "generation": {
            "generated_token_count": tokens,
            "max_new_tokens": tokens,
            "generated_text_hash": "sha256:p2p-fake",
            "decoded_tokens_match": True,
            "multi_token_generation_ready": True,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        },
        "diagnosis_codes": [
            "petals_class_p2p_candidate_ready",
            "peer_scoring_ready",
            "external_multi_node_generation_ready",
            "external_stage_requeue_ready",
            "rescue_miner_used",
            "accepted_result_after_requeue",
            "p2p_live_requeue_rescue_ready",
            "p2p_victim_result_not_accepted",
            "libp2p_discovery_backend_ready",
        ],
    }
    candidate_batch_stream = add_safe_batch_stream({}).copy()
    payload["candidate"]["batch_ready"] = True
    payload["candidate"]["batch"] = candidate_batch_stream["batch"]
    payload["candidate"]["stream_ready"] = True
    payload["candidate"]["stream"] = candidate_batch_stream["stream"]
    payload["diagnosis_codes"].extend([
        "p2p_candidate_batch_generation_ready",
        "p2p_candidate_stream_generation_ready",
        "public_swarm_generate_batch_ready",
        "public_swarm_generate_stream_ready",
        "public_swarm_generate_stream_endpoint_ready",
    ])
    return payload


def fake_usable_payload(model_id: str = "sshleifer/tiny-gpt2", *, tokens: int = 16) -> dict[str, Any]:
    stage0_rows = [
        {
            "generation_step": step,
            "stage_id": 0,
            "miner_id": "p2p-v06-real-stage0",
            "cache_schema": "real_llm_stage0_kv_cache_v1",
            "cache_stage": "stage0_prefix",
            "cache_ready": True,
            "cache_hit": step > 0,
        }
        for step in range(tokens)
    ]
    stage1_rows = [
        {
            "generation_step": step,
            "stage_id": 1,
            "miner_id": "p2p-v06-real-stage1",
            "cache_schema": "real_llm_stage1_kv_cache_v1",
            "cache_stage": "stage1_suffix",
            "cache_ready": True,
            "cache_hit": step > 0,
        }
        for step in range(tokens)
    ]
    return {
        "schema": "usable_swarm_inference_v1",
        "ok": True,
        "mode": "evidence-import",
        "readiness": {
            "p2p_product_path": {
                "ready": True,
                "generated_token_count": tokens,
                "model": model_block(model_id),
                "generation": {
                    "generated_token_count": tokens,
                    "max_new_tokens": tokens,
                    "generated_text_hash": "sha256:usable",
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                },
                "kv_cache_ready": True,
                "kv_cache": {
                    "schema": "p2p_real_generate_dual_stage_kv_cache_v1",
                    "ready": True,
                    "expected_hit_count_per_stage": max(0, tokens - 1),
                    "process_scope": "single_miner_process_per_stage",
                    "raw_activations_public": False,
                    "raw_generated_values_public": False,
                    "raw_token_inputs_public": False,
                    "stage0": {
                        "schema": "real_llm_stage0_kv_cache_v1",
                        "stage": "stage0_prefix",
                        "ready": True,
                        "row_count": tokens,
                        "ready_count": tokens,
                        "hit_count": max(0, tokens - 1),
                        "expected_hit_count": max(0, tokens - 1),
                        "rows": stage0_rows,
                    },
                    "stage1": {
                        "schema": "real_llm_stage1_kv_cache_v1",
                        "stage": "stage1_suffix",
                        "ready": True,
                        "row_count": tokens,
                        "ready_count": tokens,
                        "hit_count": max(0, tokens - 1),
                        "expected_hit_count": max(0, tokens - 1),
                        "rows": stage1_rows,
                    },
                },
            },
        },
        "diagnosis_codes": [
            "usable_swarm_inference_ready",
            "usable_real_llm_kv_cache_ready",
            "p2p_real_generate_kv_cache_ready",
            "real_llm_stage0_kv_cache_v1_ready",
            "real_llm_stage1_kv_cache_v1_ready",
            "stage0_kv_cache_hits_ready",
            "stage1_kv_cache_hits_ready",
        ],
    }


def fake_public_swarm_v2_payload(
    tokens: int = 16,
    *,
    model_id: str = "sshleifer/tiny-gpt2",
    local_model_variant: bool = False,
) -> dict[str, Any]:
    model = model_block(model_id)
    external_model = model_block("sshleifer/tiny-gpt2" if local_model_variant else model_id)
    if local_model_variant and external_model["observed_hf_model_id"] != model_id:
        external_model["model_id_match"] = False
        external_model["compatible"] = False
    codes = [
        "public_swarm_v2_local_p2p_generate_ready",
        "public_swarm_v2_16_token_generation_ready",
        "public_swarm_v2_dual_stage_kv_cache_ready",
        "public_swarm_v2_model_match_ready",
        "public_swarm_v2_signed_or_real_p2p_ready",
        "public_swarm_v2_real_p2p_local_ready",
        "public_swarm_v2_real_p2p_local_requeue_ready",
        "public_swarm_v2_stage_requeue_rescue_ready",
        "public_swarm_v2_batch_generation_ready",
        "public_swarm_v2_stream_generation_ready",
    ]
    if local_model_variant:
        codes.extend([
            "public_swarm_inference_v2_local_model_variant_ready",
            "public_swarm_v2_local_model_variant_ready",
            "public_swarm_v2_local_model_variant_model_match_ready",
            "public_swarm_v2_external_validation_not_claimed",
        ])
    else:
        codes.extend([
            "public_swarm_inference_v2_ready",
            "public_swarm_v2_external_stage_rows_ready",
        ])
    batch = {
        "enabled": True,
        "request_count": 2,
        "expected_request_count": 2,
        "observed_request_count": 2,
        "max_request_count": 4,
        "prompt_hashes": ["sha256:p1", "sha256:p2"],
        "prompt_char_counts": [12, 13],
        "result_count": 2,
        "results": [
            {
                "request_id": "req-1",
                "prompt_hash": "sha256:p1",
                "generated_token_count": tokens,
                "max_new_tokens": tokens,
                "generated_text_hash": "sha256:v2-local-1",
                "decoded_tokens_match": True,
                "multi_token_generation_ready": True,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            },
            {
                "request_id": "req-2",
                "prompt_hash": "sha256:p2",
                "generated_token_count": tokens,
                "max_new_tokens": tokens,
                "generated_text_hash": "sha256:v2-local-2",
                "decoded_tokens_match": True,
                "multi_token_generation_ready": True,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            },
        ],
        "batch_identity_ready": True,
        "batch_generation_ready": True,
        "raw_prompts_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }
    stream = {
        "enabled": True,
        "requested": True,
        "event_count": tokens * 2,
        "source": "admin-session-stream",
        "endpoint_ready": True,
        "stream_generation_ready": True,
        "progress": {
            "stream_progress_complete": True,
            "all_token_events_ready": True,
            "monotonic_progress": True,
            "expected_request_count": 2,
            "per_request_progress": [
                {
                    "request_key": "req-1",
                    "request_id": "req-1",
                    "prompt_hash": "sha256:p1",
                    "event_count": tokens,
                    "observed_token_counts": list(range(1, tokens + 1)),
                    "max_observed_token_count": tokens,
                    "target_token_count": tokens,
                    "monotonic_progress": True,
                    "stream_progress_complete": True,
                },
                {
                    "request_key": "req-2",
                    "request_id": "req-2",
                    "prompt_hash": "sha256:p2",
                    "event_count": tokens,
                    "observed_token_counts": list(range(1, tokens + 1)),
                    "max_observed_token_count": tokens,
                    "target_token_count": tokens,
                    "monotonic_progress": True,
                    "stream_progress_complete": True,
                },
            ],
            "per_request_progress_complete": True,
            "per_request_monotonic_progress": True,
            "observed_token_counts": list(range(1, tokens + 1)),
            "max_observed_token_count": tokens,
            "max_new_tokens": tokens,
            "source": "admin-session-stream",
        },
        "events": [],
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }
    return {
        "schema": pack.PUBLIC_SWARM_V2_SCHEMA,
        "ok": True,
        "mode": pack.MODE_LOCAL_MODEL_VARIANT if local_model_variant else "local",
        "public_swarm_v2": {
            "ready": True,
            "hf_model_id": model_id,
            "max_new_tokens": tokens,
            "user_surface": ["p2pd", "serve", "join", "generate"],
            "local_model_variant_only": local_model_variant,
            "external_validation_claimed": not local_model_variant,
        },
        "readiness": {
            "local_p2p_generate": {
                "schema": "usable_swarm_inference_v1",
                "ok": True,
                "ready": True,
                "model": model,
                "route_source": "p2p-discovery",
                "generated_token_count": tokens,
                "accepted_rows": tokens * 2,
                "batch": batch,
                "batch_ready": True,
                "stream": stream,
                "stream_ready": True,
                "kv_cache_ready": True,
                "stage_requeue_rescue_ready": True,
                "kv_cache": {
                    "schema": "p2p_real_generate_dual_stage_kv_cache_v1",
                    "ready": True,
                    "stage0": {"schema": "real_llm_stage0_kv_cache_v1", "ready": True, "hit_count": max(0, tokens - 1)},
                    "stage1": {"schema": "real_llm_stage1_kv_cache_v1", "ready": True, "hit_count": max(0, tokens - 1)},
                },
            },
            "external_validation": {
                "schema": "real_p2p_swarm_inference_core_rc_v1",
                "ok": True,
                "ready": True,
                "model": external_model,
                "generated_token_count": tokens,
                "accepted_rows": tokens * 2,
                "accepted_rows_ready": True,
                "retained_external_evidence_ready": True,
                "fresh_external_runtime_verified": False,
            },
            "p2p_route_hardening": {
                "schema": "real_p2p_swarm_inference_core_rc_v1",
                "ok": True,
                "ready": True,
                "model": model,
                "route_source": "real-p2p-discovery",
            },
            "real_p2p_local_route_hardening": {
                "mode": "local-smoke",
                "ready": True,
                "required": True,
                "discovery_backend": "http-provider-store",
                "generated_token_count": tokens,
                "stage_requeue_ready": True,
                "stage_requeue_target": "stage1",
            },
            "performance": {
                "stage_latency_ready": True,
                "throughput_summary_ready": True,
                "memory_or_vram_summary_ready": True,
            },
        },
        "diagnosis_codes": codes,
    }


def validate_report(payload: dict[str, Any], *, mode: str, expected_tokens: int = 16) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if payload.get("ok") is not True:
        errors.append("beta_not_ready")
    if payload.get("mode") != mode:
        errors.append("mode_mismatch")
    beta = payload.get("beta") if isinstance(payload.get("beta"), dict) else {}
    if beta.get("ready") is not True:
        errors.append("beta_summary_not_ready")
    if beta.get("user_surface") != ["serve", "join", "generate"]:
        errors.append("user_surface_mismatch")
    if beta.get("cpu_default_ready") is not True:
        errors.append("cpu_default_not_ready")
    if beta.get("cuda_optional_fail_closed_ready") is not True:
        errors.append("cuda_fail_closed_not_ready")
    if int(beta.get("max_new_tokens") or 0) < expected_tokens:
        errors.append(f"beta_token_target_below_{expected_tokens}")
    local_model_variant = mode == pack.MODE_LOCAL_MODEL_VARIANT
    if local_model_variant:
        if beta.get("local_model_variant_only") is not True:
            errors.append("local_model_variant_not_marked")
        for field in ["release_evidence_ready", "external_two_stage_ready", "external_stage_requeue_ready", "p2p_ready_product_beta"]:
            if beta.get(field) is not False:
                errors.append(f"local_model_variant_claimed:{field}")
    else:
        if beta.get("external_two_stage_ready") is not True:
            errors.append("external_two_stage_not_ready")
        if beta.get("external_stage_requeue_ready") is not True:
            errors.append("external_requeue_not_ready")
        if beta.get("p2p_ready_product_beta") is not True:
            errors.append("p2p_product_not_ready")
        if beta.get("p2p_live_requeue_ready") is not True:
            errors.append("p2p_live_requeue_not_ready")
        if beta.get("p2p_victim_result_not_accepted") is not True:
            errors.append("p2p_victim_result_not_rejected")
    if beta.get("kv_cache_ready") is not True:
        errors.append("kv_cache_not_ready")
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    product = readiness.get("product_path") if isinstance(readiness.get("product_path"), dict) else {}
    external = readiness.get("external_kaggle") if isinstance(readiness.get("external_kaggle"), dict) else {}
    p2p = readiness.get("p2p_candidate") if isinstance(readiness.get("p2p_candidate"), dict) else {}
    public_swarm_v2 = readiness.get("public_swarm_v2") if isinstance(readiness.get("public_swarm_v2"), dict) else {}
    kv_cache = readiness.get("usable_p2p_kv_cache") if isinstance(readiness.get("usable_p2p_kv_cache"), dict) else {}
    product_model = product.get("model") if isinstance(product.get("model"), dict) else {}
    external_model = external.get("model") if isinstance(external.get("model"), dict) else {}
    p2p_model = p2p.get("model") if isinstance(p2p.get("model"), dict) else {}
    kv_cache_model = kv_cache.get("model") if isinstance(kv_cache.get("model"), dict) else {}
    if not product_model.get("observed_hf_model_id") or product_model.get("model_id_present") is not True:
        errors.append("product_model_id_missing")
    if product_model.get("compatible") is not True:
        errors.append("product_model_not_compatible")
    if local_model_variant:
        for name, summary in [("external_kaggle", external), ("p2p_candidate", p2p)]:
            if summary.get("claimed") is not False or summary.get("ready") is not False:
                errors.append(f"local_model_variant_claimed:{name}")
    else:
        if not external_model.get("observed_hf_model_id") or external_model.get("model_id_present") is not True:
            errors.append("external_model_id_missing")
        if external_model.get("compatible") is not True:
            errors.append("external_model_not_compatible")
        if not p2p_model.get("observed_hf_model_id") or p2p_model.get("model_id_present") is not True:
            errors.append("p2p_model_id_missing")
        if p2p_model.get("compatible") is not True:
            errors.append("p2p_model_not_compatible")
    if public_swarm_v2.get("ready") is not True:
        errors.append("public_swarm_v2_not_ready")
    v2_model = public_swarm_v2.get("model") if isinstance(public_swarm_v2.get("model"), dict) else {}
    if v2_model.get("compatible") is not True:
        errors.append("public_swarm_v2_model_not_compatible")
    if public_swarm_v2.get("generated_token_count", 0) < 1:
        errors.append("public_swarm_v2_generation_missing")
    if int(public_swarm_v2.get("generated_token_count") or 0) < expected_tokens:
        errors.append(f"public_swarm_v2_token_target_below_{expected_tokens}")
    if int(public_swarm_v2.get("required_generated_token_count") or 0) < expected_tokens:
        errors.append(f"public_swarm_v2_required_token_target_below_{expected_tokens}")
    if public_swarm_v2.get("accepted_rows_ready") is not True:
        errors.append("public_swarm_v2_stage_rows_not_ready")
    if local_model_variant:
        if public_swarm_v2.get("local_model_variant_only") is not True:
            errors.append("public_swarm_v2_local_model_variant_not_marked")
        if public_swarm_v2.get("external_validation_claimed") is not False:
            errors.append("public_swarm_v2_external_claimed_in_local_variant")
    elif public_swarm_v2.get("external_accepted_rows_ready") is not True:
        errors.append("public_swarm_v2_external_stage_rows_not_ready")
    if public_swarm_v2.get("kv_cache_ready") is not True:
        errors.append("public_swarm_v2_kv_cache_not_ready")
    if public_swarm_v2.get("stage_requeue_rescue_ready") is not True:
        errors.append("public_swarm_v2_requeue_not_ready")
    if public_swarm_v2.get("real_p2p_local_route_hardening_ready") is not True:
        errors.append("public_swarm_v2_real_p2p_local_not_ready")
    if public_swarm_v2.get("real_p2p_local_stage_requeue_ready") is not True:
        errors.append("public_swarm_v2_real_p2p_local_requeue_not_ready")
    if public_swarm_v2.get("batch_ready") is not True:
        errors.append("public_swarm_v2_batch_not_ready")
    if public_swarm_v2.get("stream_ready") is not True:
        errors.append("public_swarm_v2_stream_not_ready")
    if kv_cache.get("ready") is not True:
        errors.append("usable_kv_cache_not_ready")
    if int(kv_cache.get("generated_token_count") or 0) < expected_tokens:
        errors.append(f"usable_kv_cache_token_target_below_{expected_tokens}")
    if int(kv_cache.get("required_generated_token_count") or 0) < expected_tokens:
        errors.append(f"usable_kv_cache_required_token_target_below_{expected_tokens}")
    if not kv_cache_model.get("observed_hf_model_id") or kv_cache_model.get("model_id_present") is not True:
        errors.append("usable_kv_cache_model_id_missing")
    if kv_cache_model.get("compatible") is not True:
        errors.append("usable_kv_cache_model_not_compatible")
    stage0 = kv_cache.get("stage0") if isinstance(kv_cache.get("stage0"), dict) else {}
    stage1 = kv_cache.get("stage1") if isinstance(kv_cache.get("stage1"), dict) else {}
    if stage0.get("hit_count", 0) < 1:
        errors.append("stage0_kv_cache_hits_missing")
    if stage1.get("hit_count", 0) < 1:
        errors.append("stage1_kv_cache_hits_missing")
    expected_hits = max(1, expected_tokens - 1)
    if int(stage0.get("hit_count") or 0) < expected_hits:
        errors.append(f"stage0_kv_cache_hits_below_{expected_hits}")
    if int(stage1.get("hit_count") or 0) < expected_hits:
        errors.append(f"stage1_kv_cache_hits_below_{expected_hits}")
    if not local_model_variant:
        if int(external.get("generated_token_count") or 0) < expected_tokens:
            errors.append(f"external_token_target_below_{expected_tokens}")
        if int(external.get("required_generated_token_count") or 0) < expected_tokens:
            errors.append(f"external_required_token_target_below_{expected_tokens}")
        if int(p2p.get("generated_token_count") or 0) < expected_tokens:
            errors.append(f"p2p_token_target_below_{expected_tokens}")
        if int(p2p.get("required_generated_token_count") or 0) < expected_tokens:
            errors.append(f"p2p_required_token_target_below_{expected_tokens}")
    codes = set(payload.get("diagnosis_codes") or [])
    if payload.get("ok") is True:
        for code in sorted(code for code in codes if code.endswith("_blocked")):
            errors.append(f"unexpected_blocked_code:{code}")
    required_codes = LOCAL_MODEL_VARIANT_REQUIRED_CODES if local_model_variant else REQUIRED_RELEASE_CODES
    for code in sorted(required_codes - codes):
        errors.append(f"missing_code:{code}")
    if local_model_variant:
        for code in sorted(LOCAL_MODEL_VARIANT_BLOCKED_CODES & codes):
            errors.append(f"unexpected_code:{code}")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    for key in [
        "coordinator_backed_task_execution",
        "cpu_default",
        "cuda_optional",
        "cuda_fail_closed_expected",
        "real_p2p_discovery_optional",
        "not_coordinator_free",
        "not_hivemind_petals_production",
        "not_large_model_serving",
    ]:
        if safety.get(key) is not True:
            errors.append(f"safety_missing:{key}")
    output_request = payload.get("output_request") if isinstance(payload.get("output_request"), dict) else {}
    if output_request.get("include_output") is not False:
        errors.append("output_request_include_output_mismatch")
    if output_request.get("raw_prompt_public") is not False:
        errors.append("output_request_raw_prompt_public_mismatch")
    if output_request.get("raw_generated_text_public") is not False:
        errors.append("output_request_raw_generated_text_public_mismatch")
    if output_request.get("generated_token_ids_public") is not False:
        errors.append("output_request_generated_token_ids_public_mismatch")
    if output_request.get("public_artifact_safe") is not True:
        errors.append("output_request_public_artifact_safe_mismatch")
    prompt_scope = payload.get("prompt_scope") if isinstance(payload.get("prompt_scope"), dict) else {}
    if prompt_scope.get("source") not in {"prompt-text", "prompt-texts", "prompt-texts-file"}:
        errors.append("prompt_scope_source_mismatch")
    if not isinstance(prompt_scope.get("prompt_count"), int) or prompt_scope.get("prompt_count") < 1:
        errors.append("prompt_scope_count_mismatch")
    expected_inline = prompt_scope.get("source") in {"prompt-text", "prompt-texts"}
    if prompt_scope.get("inline_prompt_text") is not expected_inline:
        errors.append("prompt_scope_inline_prompt_text_mismatch")
    if prompt_scope.get("terminal_next_commands_local_private") is not expected_inline:
        errors.append("prompt_scope_terminal_next_commands_mismatch")
    if prompt_scope.get("saved_artifacts_prompt_placeholders") is not True:
        errors.append("prompt_scope_saved_placeholders_mismatch")
    if prompt_scope.get("saved_artifacts_public_safe") is not True:
        errors.append("prompt_scope_saved_artifacts_public_safe_mismatch")
    if prompt_scope.get("prefer_prompt_file_or_stdin_for_shareable_logs") is not True:
        errors.append("prompt_scope_shareable_log_guidance_mismatch")
    if prompt_scope.get("prompt_file_path_public") is not False:
        errors.append("prompt_scope_file_path_public_mismatch")
    if prompt_scope.get("raw_prompt_public") is not False:
        errors.append("prompt_scope_raw_prompt_public_mismatch")
    if prompt_scope.get("public_artifact_safe") is not True:
        errors.append("prompt_scope_public_artifact_safe_mismatch")
    answer_scope = payload.get("answer_scope") if isinstance(payload.get("answer_scope"), dict) else {}
    if answer_scope.get("scope_state") != "no-local-answer":
        errors.append("answer_scope_state_mismatch")
    if answer_scope.get("visible_in_terminal") is not False:
        errors.append("answer_scope_visible_in_terminal_mismatch")
    if answer_scope.get("terminal_only") is not False:
        errors.append("answer_scope_terminal_only_mismatch")
    if answer_scope.get("saved_json_display") != "hash-only":
        errors.append("answer_scope_saved_json_display_mismatch")
    if answer_scope.get("saved_markdown_display") != "hash-only":
        errors.append("answer_scope_saved_markdown_display_mismatch")
    if answer_scope.get("public_artifact_safe") is not True:
        errors.append("answer_scope_public_artifact_safe_mismatch")
    shareable = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
    if shareable.get("saved_artifacts_public_safe") is not True:
        errors.append("shareable_saved_artifacts_public_safe_mismatch")
    if shareable.get("raw_prompt_public") is not False:
        errors.append("shareable_raw_prompt_public_mismatch")
    if shareable.get("raw_generated_text_public") is not False:
        errors.append("shareable_raw_generated_text_public_mismatch")
    if shareable.get("generated_token_ids_public") is not False:
        errors.append("shareable_generated_token_ids_public_mismatch")
    if shareable.get("answer_scope_state") != "no-local-answer":
        errors.append("shareable_answer_scope_state_mismatch")
    if shareable.get("local_answer_terminal_only") is not False:
        errors.append("shareable_local_answer_terminal_only_mismatch")
    if shareable.get("public_artifact_safe") is not True:
        errors.append("shareable_public_artifact_safe_mismatch")
    for error in pack.validate_public_report(payload):
        if error not in errors:
            errors.append(error)
    artifact_summary = payload.get("artifact_summary") if isinstance(payload.get("artifact_summary"), dict) else {}
    if artifact_summary.get("schema") != pack.ARTIFACT_SUMMARY_SCHEMA:
        errors.append("artifact_summary_schema_mismatch")
    if artifact_summary.get("inspect_first") != "public_real_llm_swarm_beta.md":
        errors.append("artifact_summary_inspect_first_mismatch")
    if artifact_summary.get("machine_readable") != "public_real_llm_swarm_beta.json":
        errors.append("artifact_summary_machine_readable_mismatch")
    if artifact_summary.get("support_bundle") != "support_bundle.json":
        errors.append("artifact_summary_support_bundle_mismatch")
    if artifact_summary.get("runbook") != "PUBLIC_REAL_LLM_SWARM_BETA.md":
        errors.append("artifact_summary_runbook_mismatch")
    expected_shareable_paths = [
        "public_real_llm_swarm_beta.json",
        "public_real_llm_swarm_beta.md",
        "support_bundle.json",
    ]
    if artifact_summary.get("shareable_paths") != expected_shareable_paths:
        errors.append("artifact_summary_shareable_paths_mismatch")
    if artifact_summary.get("public_artifact_safe") is not True:
        errors.append("artifact_summary_public_artifact_safe_mismatch")
    if artifact_summary.get("raw_prompt_public") is not False:
        errors.append("artifact_summary_raw_prompt_public_mismatch")
    if artifact_summary.get("raw_generated_text_public") is not False:
        errors.append("artifact_summary_raw_generated_text_public_mismatch")
    if artifact_summary.get("generated_token_ids_public") is not False:
        errors.append("artifact_summary_generated_token_ids_public_mismatch")
    review_summary = payload.get("review_summary") if isinstance(payload.get("review_summary"), dict) else {}
    not_completed = payload.get("not_completed") if isinstance(payload.get("not_completed"), list) else []
    if review_summary.get("schema") != pack.REVIEW_SUMMARY_SCHEMA:
        errors.append("review_summary_schema_mismatch")
    if review_summary.get("inspect_first") != artifact_summary.get("inspect_first"):
        errors.append("review_summary_inspect_first_mismatch")
    if review_summary.get("machine_readable") != artifact_summary.get("machine_readable"):
        errors.append("review_summary_machine_readable_mismatch")
    if review_summary.get("support_bundle") != artifact_summary.get("support_bundle"):
        errors.append("review_summary_support_bundle_mismatch")
    if review_summary.get("shareable_paths") != expected_shareable_paths:
        errors.append("review_summary_shareable_paths_mismatch")
    if review_summary.get("not_completed_count") != len(not_completed):
        errors.append("review_summary_not_completed_count_mismatch")
    if payload.get("ok") is True and beta.get("ready") is True and not not_completed:
        if review_summary.get("state") != "ready" or review_summary.get("ready") is not True:
            errors.append("review_summary_ready_state_mismatch")
        if review_summary.get("next_step") != "run_beta_report_check":
            errors.append("review_summary_next_step_mismatch")
        recommended_check = payload.get("recommended_check_command") if isinstance(payload.get("recommended_check_command"), dict) else {}
        review_recommended_check = review_summary.get("recommended_check_command") if isinstance(review_summary.get("recommended_check_command"), dict) else {}
        command_line = str(recommended_check.get("command_line") or "")
        if not recommended_check:
            errors.append("recommended_check_command_missing")
        if review_recommended_check != recommended_check:
            errors.append("review_summary_recommended_check_mismatch")
        for fragment in [
            "crowdtensor public-real-llm-swarm-beta check",
            "--beta-report",
            "public_real_llm_swarm_beta.json",
            f"--max-new-tokens {expected_tokens}",
            "--json",
        ]:
            if fragment not in command_line:
                errors.append(f"recommended_check_command_missing:{fragment}")
        model_id = str(beta.get("hf_model_id") or pack.DEFAULT_HF_MODEL_ID)
        if model_id != pack.DEFAULT_HF_MODEL_ID and f"--hf-model-id {model_id}" not in command_line:
            errors.append("recommended_check_command_model_missing")
    if review_summary.get("public_artifact_safe") is not True:
        errors.append("review_summary_public_artifact_safe_mismatch")
    if review_summary.get("raw_prompt_public") is not False:
        errors.append("review_summary_raw_prompt_public_mismatch")
    if review_summary.get("raw_generated_text_public") is not False:
        errors.append("review_summary_raw_generated_text_public_mismatch")
    if review_summary.get("generated_token_ids_public") is not False:
        errors.append("review_summary_generated_token_ids_public_mismatch")
    user_status = payload.get("user_status") if isinstance(payload.get("user_status"), dict) else {}
    if user_status.get("state") not in {"ready", "blocked", "package-ready", "local-model-ready"}:
        errors.append("user_status_state_mismatch")
    if not user_status.get("headline"):
        errors.append("user_status_headline_missing")
    if not user_status.get("recommended_label"):
        errors.append("user_status_recommended_label_missing")
    if user_status.get("not_completed_count") != len(not_completed):
        errors.append("user_status_not_completed_count_mismatch")
    if user_status.get("public_artifact_safe") is not True:
        errors.append("user_status_public_artifact_safe_mismatch")
    recommended_next = payload.get("recommended_next_command") if isinstance(payload.get("recommended_next_command"), dict) else {}
    if not recommended_next.get("label") or not recommended_next.get("command_line"):
        errors.append("recommended_next_command_missing")
    if recommended_next.get("public_artifact_safe") is not True:
        errors.append("recommended_next_public_artifact_safe_mismatch")
    if review_summary.get("recommended_next_command") != recommended_next:
        errors.append("review_summary_recommended_next_mismatch")
    if review_summary.get("next_command") != recommended_next.get("command_line"):
        errors.append("review_summary_next_command_mismatch")
    if user_status.get("recommended_label") != recommended_next.get("label"):
        errors.append("user_status_recommended_label_mismatch")
    next_commands = payload.get("next_commands") if isinstance(payload.get("next_commands"), list) else []
    if not next_commands:
        errors.append("next_commands_missing")
    if not any(isinstance(item, dict) and item.get("label") == "inspect support bundle" for item in next_commands):
        errors.append("next_commands_support_bundle_missing")
    if recommended_next and not any(isinstance(item, dict) and item.get("command_line") == recommended_next.get("command_line") for item in next_commands):
        errors.append("next_commands_recommended_missing")
    operator_actions = [
        str(item)
        for item in (payload.get("operator_action") if isinstance(payload.get("operator_action"), list) else [])
    ]
    if not operator_actions:
        errors.append("operator_action_missing")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    required_artifacts = [
        "public_real_llm_swarm_beta_json",
        "public_real_llm_swarm_beta_markdown",
        "support_bundle_json",
        "runbook",
        "usable_swarm_kv_cache_report",
        "public_swarm_v2_report",
    ]
    if not local_model_variant:
        required_artifacts.extend(["external_real_llm_report", "p2p_candidate_report"])
    for name in required_artifacts:
        artifact = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
        if artifact.get("present") is not True:
            errors.append(f"missing_artifact:{name}")
    machine_text = artifact_text(
        payload,
        "public_real_llm_swarm_beta_json",
        "public_real_llm_swarm_beta.json",
    )
    if machine_text:
        append_sensitive_artifact_errors(errors, artifact_name="public_real_llm_swarm_beta_json", text=machine_text)
    machine = artifact_json(
        payload,
        "public_real_llm_swarm_beta_json",
        "public_real_llm_swarm_beta.json",
    )
    if not machine:
        errors.append("machine_readable_artifact_unreadable")
    else:
        if machine.get("schema") != payload.get("schema"):
            errors.append("machine_readable_schema_mismatch")
        if machine.get("ok") != payload.get("ok"):
            errors.append("machine_readable_ok_mismatch")
        if machine.get("operator_action") != payload.get("operator_action"):
            errors.append("machine_readable_operator_action_mismatch")
        if machine.get("not_completed") != payload.get("not_completed"):
            errors.append("machine_readable_not_completed_mismatch")
        if machine.get("runtime_provenance") != payload.get("runtime_provenance"):
            errors.append("machine_readable_runtime_provenance_mismatch")
        if machine.get("evidence_scope") != payload.get("evidence_scope"):
            errors.append("machine_readable_evidence_scope_mismatch")
        machine_review = machine.get("review_summary") if isinstance(machine.get("review_summary"), dict) else {}
        if machine_review.get("next_step") != review_summary.get("next_step"):
            errors.append("machine_readable_review_next_step_mismatch")
        if machine.get("recommended_check_command") != payload.get("recommended_check_command"):
            errors.append("machine_readable_recommended_check_mismatch")
    markdown = artifact_text(
        payload,
        "public_real_llm_swarm_beta_markdown",
        "public_real_llm_swarm_beta.md",
    )
    if not markdown:
        errors.append("markdown_artifact_unreadable")
    else:
        append_sensitive_artifact_errors(errors, artifact_name="public_real_llm_swarm_beta_markdown", text=markdown)
        required_markdown_sections = {
            "## Review": "review",
            "## Runtime Provenance": "runtime_provenance",
            "## Evidence Scope": "evidence_scope",
            "## Operator Action": "operator_action",
            "## Output Scope": "output_scope",
            "## Artifacts": "artifacts",
            "## Diagnosis": "diagnosis",
            "## Not Completed": "not_completed",
            "## Boundaries": "boundaries",
        }
        for section, label in required_markdown_sections.items():
            if section not in markdown:
                errors.append(f"markdown_section_missing:{label}")
        if "- inspect first: `public_real_llm_swarm_beta.md`" not in markdown:
            errors.append("markdown_review_inspect_first_missing")
        if "- support bundle: `support_bundle.json`" not in markdown:
            errors.append("markdown_support_bundle_missing")
        if "- machine readable: `public_real_llm_swarm_beta.json`" not in markdown:
            errors.append("markdown_machine_readable_missing")
        if "## What To Do Next" not in markdown:
            errors.append("markdown_next_commands_section_missing")
        if "inspect support bundle" not in markdown:
            errors.append("markdown_next_support_bundle_missing")
        if "- status: `" not in markdown:
            errors.append("markdown_status_missing")
        recommended_line = str((payload.get("recommended_check_command") if isinstance(payload.get("recommended_check_command"), dict) else {}).get("command_line") or "")
        if recommended_line and recommended_line not in markdown:
            errors.append("markdown_recommended_check_missing")
        recommended_next_line = str(recommended_next.get("command_line") or "")
        if recommended_next_line and recommended_next_line not in markdown:
            errors.append("markdown_recommended_next_missing")
        if "- answer scope: `no-local-answer`" not in markdown:
            errors.append("markdown_answer_scope_missing")
        if "- prompt scope: `" not in markdown or "raw_prompt_public=False" not in markdown:
            errors.append("markdown_prompt_scope_missing")
        evidence_scope = payload.get("evidence_scope") if isinstance(payload.get("evidence_scope"), dict) else {}
        if f"- level: `{evidence_scope.get('level')}`" not in markdown:
            errors.append("markdown_evidence_scope_level_missing")
        if f"- fresh Kaggle GPU verified: `{evidence_scope.get('fresh_kaggle_gpu_verified')}`" not in markdown:
            errors.append("markdown_evidence_scope_fresh_gpu_missing")
        for item in operator_actions[:3]:
            if item not in markdown:
                errors.append("markdown_operator_action_missing")
                break
        for item in [str(entry) for entry in not_completed[:3]]:
            if item not in markdown:
                errors.append("markdown_not_completed_item_missing")
                break
        if not not_completed and "- none" not in markdown:
            errors.append("markdown_not_completed_none_missing")
    runbook = artifact_text(payload, "runbook", "PUBLIC_REAL_LLM_SWARM_BETA.md")
    if not runbook:
        errors.append("runbook_artifact_unreadable")
    else:
        for fragment in [
            "## Review The Result",
            "`Review`, `Operator Action`, and `Not Completed`",
            "public_real_llm_swarm_beta.md",
            "support_bundle.json",
            "## Share Safely",
            "raw prompts, generated text, generated token ids, credentials, activations, and lease tokens are excluded",
        ]:
            if fragment not in runbook:
                errors.append("runbook_review_guidance_missing")
                break
    support = artifact_json(payload, "support_bundle_json", "support_bundle.json")
    support_text = artifact_text(payload, "support_bundle_json", "support_bundle.json")
    if support_text:
        append_sensitive_artifact_errors(errors, artifact_name="support_bundle_json", text=support_text)
    if not support:
        errors.append("support_bundle_unreadable")
    else:
        if support.get("operator_action") != payload.get("operator_action"):
            errors.append("support_bundle_operator_action_mismatch")
        if support.get("not_completed") != payload.get("not_completed"):
            errors.append("support_bundle_not_completed_mismatch")
        support_review = support.get("review_summary") if isinstance(support.get("review_summary"), dict) else {}
        if support_review.get("next_step") != review_summary.get("next_step"):
            errors.append("support_bundle_review_next_step_mismatch")
        if support.get("recommended_check_command") != payload.get("recommended_check_command"):
            errors.append("support_bundle_recommended_check_mismatch")
        if support.get("recommended_next_command") != payload.get("recommended_next_command"):
            errors.append("support_bundle_recommended_next_mismatch")
        if support.get("next_commands") != payload.get("next_commands"):
            errors.append("support_bundle_next_commands_mismatch")
        if support.get("user_status") != payload.get("user_status"):
            errors.append("support_bundle_user_status_mismatch")
        if support.get("runtime_provenance") != payload.get("runtime_provenance"):
            errors.append("support_bundle_runtime_provenance_mismatch")
        if support.get("evidence_scope") != payload.get("evidence_scope"):
            errors.append("support_bundle_evidence_scope_mismatch")
        support_artifacts = support.get("artifact_summary") if isinstance(support.get("artifact_summary"), dict) else {}
        if support_artifacts.get("inspect_first") != "public_real_llm_swarm_beta.md":
            errors.append("support_bundle_inspect_first_mismatch")
        if support.get("prompt_scope") != payload.get("prompt_scope"):
            errors.append("support_bundle_prompt_scope_mismatch")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    return errors


def build_fake_release(output_dir: Path, *, tokens: int) -> dict[str, Any]:
    external_path = output_dir / "source" / "real_llm_internet_beta.json"
    p2p_path = output_dir / "source" / "petals_class_p2p_candidate.json"
    usable_path = output_dir / "source" / "usable_swarm_inference.json"
    public_swarm_v2_path = output_dir / "source" / "public_swarm_inference_v2.json"
    write(external_path, json.dumps(fake_external_payload(tokens=tokens)) + "\n")
    write(p2p_path, json.dumps(fake_p2p_payload(tokens=tokens)) + "\n")
    write(usable_path, json.dumps(fake_usable_payload(tokens=tokens)) + "\n")
    write(public_swarm_v2_path, json.dumps(fake_public_swarm_v2_payload(tokens)) + "\n")
    args = pack.parse_args([
        "release",
        "--output-dir",
        str(output_dir / "beta"),
        "--external-report",
        str(external_path),
        "--p2p-report",
        str(p2p_path),
        "--usable-report",
        str(usable_path),
        "--public-swarm-v2-report",
        str(public_swarm_v2_path),
        "--base-port",
        "9430",
        "--port",
        "9430",
        "--timeout-seconds",
        "60",
        "--max-new-tokens",
        str(tokens),
    ])

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        joined = " ".join(command)
        if "public_swarm_product_beta_pack.py" in joined:
            write(output_dir / "beta" / "product-beta" / "public_swarm_product_beta.json", json.dumps(fake_product_payload(tokens=tokens)) + "\n")
            return completed(fake_product_payload(tokens=tokens))
        if "petals_class_p2p_candidate_pack.py" in joined:
            write(
                output_dir / "beta" / "p2p-candidate" / "petals_class_p2p_candidate.json",
                json.dumps(fake_p2p_payload(tokens=tokens)) + "\n",
            )
            return completed(fake_p2p_payload(tokens=tokens))
        if "public_swarm_inference_v2_pack.py" in joined:
            child_dir = output_dir / "beta" / "public-swarm-v2"
            write(child_dir / "public_swarm_inference_v2.json", json.dumps(fake_public_swarm_v2_payload(tokens)) + "\n")
            write(child_dir / "usable-v1-local" / "usable_swarm_inference.json", json.dumps(fake_usable_payload(tokens=tokens)) + "\n")
            return completed(fake_public_swarm_v2_payload(tokens))
        if "public_swarm_gpu_inference_beta_pack.py" in joined:
            write(
                output_dir / "beta" / "gpu-smoke" / "public_swarm_gpu_inference_beta_local_smoke.json",
                json.dumps(fake_gpu_payload()) + "\n",
            )
            return completed(fake_gpu_payload())
        raise AssertionError(command)

    return pack.build_report(args, runner=fake_runner)


def build_fake_local_model_variant(output_dir: Path, *, model_id: str, tokens: int = 16) -> dict[str, Any]:
    args = pack.parse_args([
        pack.MODE_LOCAL_MODEL_VARIANT,
        "--output-dir",
        str(output_dir / "beta"),
        "--base-port",
        "9530",
        "--port",
        "9530",
        "--public-swarm-v2-p2p-port",
        "9531",
        "--public-swarm-v2-coordinator-port",
        "9532",
        "--public-swarm-v2-real-p2p-port",
        "9533",
        "--public-swarm-v2-real-p2p-coordinator-port",
        "9534",
        "--hf-model-id",
        model_id,
        "--max-new-tokens",
        str(tokens),
        "--timeout-seconds",
        "60",
        "--stream-generation",
    ])
    product = fake_product_payload(model_id, tokens=tokens)
    usable = fake_usable_payload(model_id, tokens=tokens)
    public_swarm_v2 = fake_public_swarm_v2_payload(tokens, model_id=model_id, local_model_variant=True)
    gpu = fake_gpu_payload()

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        joined = " ".join(command)
        if "public_swarm_product_beta_pack.py" in joined:
            write(output_dir / "beta" / "product-beta" / "public_swarm_product_beta.json", json.dumps(product) + "\n")
            return completed(product)
        if "public_swarm_inference_v2_pack.py" in joined:
            child_dir = output_dir / "beta" / "public-swarm-v2"
            write(child_dir / "public_swarm_inference_v2.json", json.dumps(public_swarm_v2) + "\n")
            write(child_dir / "usable-v1-local" / "usable_swarm_inference.json", json.dumps(usable) + "\n")
            return completed(public_swarm_v2)
        if "public_swarm_gpu_inference_beta_pack.py" in joined:
            write(
                output_dir / "beta" / "gpu-smoke" / "public_swarm_gpu_inference_beta_local_smoke.json",
                json.dumps(gpu) + "\n",
            )
            return completed(gpu)
        raise AssertionError(command)

    return pack.build_report(args, runner=fake_runner)


def artifact_path(payload: dict[str, Any], name: str, fallback: str) -> str:
    output_dir = Path(str(payload.get("output_dir") or ""))
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    artifact = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
    raw_path = str(artifact.get("path") or "")
    if raw_path:
        path = Path(raw_path)
        return str(path if path.is_absolute() else output_dir / path)
    return str(output_dir / fallback)


def check_artifact_summary(result: dict[str, Any]) -> dict[str, Any]:
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    check_json = str(Path(str(result.get("output_dir") or "")) / "public_real_llm_swarm_beta_check.json")
    shareable_paths = [
        str(artifacts.get("public_real_llm_swarm_beta_json") or ""),
        str(artifacts.get("public_real_llm_swarm_beta_markdown") or ""),
        str(artifacts.get("support_bundle_json") or ""),
        check_json,
    ]
    return {
        "schema": ARTIFACT_SUMMARY_SCHEMA,
        "inspect_first": str(artifacts.get("public_real_llm_swarm_beta_markdown") or ""),
        "machine_readable": str(artifacts.get("public_real_llm_swarm_beta_json") or ""),
        "support_bundle": str(artifacts.get("support_bundle_json") or ""),
        "check_json": check_json,
        "runbook": str(artifacts.get("runbook") or ""),
        "shareable_paths": [path for path in shareable_paths if path],
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "summary": "Open inspect_first for the checked Beta report; use check_json for validation errors.",
    }


def recommended_check_command(result: dict[str, Any]) -> dict[str, Any]:
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    beta_report = str(result.get("checked_beta_report") or artifacts.get("public_real_llm_swarm_beta_json") or "").strip()
    if not beta_report:
        return {}
    command = [
        "crowdtensor",
        "public-real-llm-swarm-beta",
        "check",
    ]
    if str(result.get("mode") or "") == pack.MODE_LOCAL_MODEL_VARIANT:
        command.extend(["--hf-model-id", str(result.get("hf_model_id") or pack.DEFAULT_HF_MODEL_ID)])
    command.extend([
        "--beta-report",
        beta_report,
        "--output-dir",
        str(result.get("output_dir") or ""),
        "--max-new-tokens",
        str(result.get("max_new_tokens") or 16),
        "--json",
    ])
    return {
        "label": "rerun beta report check",
        "reason": "rerun_current_beta_report_check",
        "command_line": pack.shell_command(command),
        "beta_report": beta_report,
        "output_dir": str(result.get("output_dir") or ""),
        "check_source": "beta-report",
        "public_artifact_safe": True,
    }


def command_entry(label: str, command: list[str], *, reason: str = "") -> dict[str, Any]:
    entry: dict[str, Any] = {
        "label": label,
        "command": [str(part) for part in command],
        "command_line": pack.shell_command(command),
        "public_artifact_safe": True,
    }
    if reason:
        entry["reason"] = reason
    return entry


def inspect_command(path: str, *, lines: str = "1,220p") -> list[str]:
    return ["sed", "-n", lines, str(path)]


def recommended_next_command(result: dict[str, Any]) -> dict[str, Any]:
    artifact_summary = (
        result.get("artifact_summary")
        if isinstance(result.get("artifact_summary"), dict)
        else check_artifact_summary(result)
    )
    ready = bool(result.get("ok") is True and not result.get("errors"))
    if ready and artifact_summary.get("inspect_first"):
        return command_entry(
            "inspect checked Beta summary",
            inspect_command(str(artifact_summary.get("inspect_first"))),
            reason="review_checked_artifacts",
        )
    if artifact_summary.get("check_json"):
        return command_entry(
            "inspect check errors",
            inspect_command(str(artifact_summary.get("check_json"))),
            reason="fix_validation_errors",
        )
    recommended_check = result.get("recommended_check_command") if isinstance(result.get("recommended_check_command"), dict) else {}
    return dict(recommended_check)


def next_commands(result: dict[str, Any]) -> list[dict[str, Any]]:
    artifact_summary = (
        result.get("artifact_summary")
        if isinstance(result.get("artifact_summary"), dict)
        else check_artifact_summary(result)
    )
    commands: list[dict[str, Any]] = []
    if artifact_summary.get("inspect_first"):
        commands.append(command_entry(
            "inspect checked Beta summary",
            inspect_command(str(artifact_summary.get("inspect_first"))),
            reason="review_checked_artifacts",
        ))
    if artifact_summary.get("support_bundle"):
        commands.append(command_entry(
            "inspect support bundle",
            inspect_command(str(artifact_summary.get("support_bundle"))),
            reason="inspect_diagnostics",
        ))
    if artifact_summary.get("check_json"):
        commands.append(command_entry(
            "inspect check JSON",
            inspect_command(str(artifact_summary.get("check_json"))),
            reason="inspect_validation_record",
        ))
    recommended_next = result.get("recommended_next_command") if isinstance(result.get("recommended_next_command"), dict) else {}
    if recommended_next and all(item.get("command_line") != recommended_next.get("command_line") for item in commands):
        commands.append(dict(recommended_next))
    recommended_check = result.get("recommended_check_command") if isinstance(result.get("recommended_check_command"), dict) else {}
    if recommended_check and all(item.get("command_line") != recommended_check.get("command_line") for item in commands):
        commands.append(dict(recommended_check))
    return commands


def user_status(result: dict[str, Any]) -> dict[str, Any]:
    errors = result.get("errors") if isinstance(result.get("errors"), list) else []
    ready = bool(result.get("ok") is True and not errors)
    recommended = result.get("recommended_next_command") if isinstance(result.get("recommended_next_command"), dict) else {}
    return {
        "state": "ready" if ready else "blocked",
        "headline": (
            "Public Real-LLM Swarm Beta check passed."
            if ready
            else "Public Real-LLM Swarm Beta check needs attention."
        ),
        "next_step": "review_checked_artifacts" if ready else "fix_validation_errors",
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "error_count": len(errors),
        "public_artifact_safe": True,
    }


def check_review_summary(result: dict[str, Any]) -> dict[str, Any]:
    errors = [str(error) for error in (result.get("errors") if isinstance(result.get("errors"), list) else [])]
    artifact_summary = (
        result.get("artifact_summary")
        if isinstance(result.get("artifact_summary"), dict)
        else check_artifact_summary(result)
    )
    recommended = (
        result.get("recommended_check_command")
        if isinstance(result.get("recommended_check_command"), dict)
        else recommended_check_command(result)
    )
    recommended_next = (
        result.get("recommended_next_command")
        if isinstance(result.get("recommended_next_command"), dict)
        else recommended_next_command(result)
    )
    ready = bool(result.get("ok") is True and not errors)
    return {
        "schema": REVIEW_SUMMARY_SCHEMA,
        "state": "ready" if ready else "blocked",
        "ready": ready,
        "next_step": "review_checked_artifacts" if ready else "fix_validation_errors",
        "inspect_first": artifact_summary.get("inspect_first"),
        "machine_readable": artifact_summary.get("machine_readable"),
        "support_bundle": artifact_summary.get("support_bundle"),
        "check_json": artifact_summary.get("check_json"),
        "recommended_check_command": recommended,
        "recommended_next_command": recommended_next,
        "recommended_label": recommended_next.get("label") or recommended.get("label") or "none",
        "recommended_reason": recommended_next.get("reason") or recommended.get("reason") or "none",
        "next_command": recommended_next.get("command_line") or recommended.get("command_line") or "",
        "error_count": len(errors),
        "error_preview": errors[:8],
        "operator_action": (
            "Open inspect_first for the checked Markdown, support_bundle for diagnostics, and check_json for the validation record."
            if ready
            else f"Open check_json for the validation errors, inspect the checked Markdown Not Completed section, fix the listed items, then rerun: {recommended.get('command_line') or 'crowdtensor public-real-llm-swarm-beta check --beta-report <public_real_llm_swarm_beta.json> --json'}"
        ),
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }


def check_output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "public_artifact_safe": True,
        "summary": "The check artifact records validation status and paths only; raw prompts, generated text, and token ids are excluded.",
    }


def check_prompt_scope_summary(payload: dict[str, Any]) -> dict[str, Any]:
    prompt_scope = payload.get("prompt_scope") if isinstance(payload.get("prompt_scope"), dict) else {}
    return {
        "source": prompt_scope.get("source") or "unknown",
        "prompt_count": prompt_scope.get("prompt_count"),
        "inline_prompt_text": prompt_scope.get("inline_prompt_text"),
        "terminal_next_commands_local_private": prompt_scope.get("terminal_next_commands_local_private"),
        "terminal_logs_local_private": prompt_scope.get("terminal_logs_local_private"),
        "saved_artifacts_prompt_placeholders": prompt_scope.get("saved_artifacts_prompt_placeholders"),
        "saved_artifacts_public_safe": prompt_scope.get("saved_artifacts_public_safe"),
        "prefer_prompt_file_or_stdin_for_shareable_logs": prompt_scope.get("prefer_prompt_file_or_stdin_for_shareable_logs"),
        "raw_prompt_public": prompt_scope.get("raw_prompt_public", False),
        "public_artifact_safe": prompt_scope.get("public_artifact_safe", False),
        "summary": "The check artifact mirrors prompt source/count safety fields from the checked Beta report without raw prompt text.",
    }


def check_answer_scope_summary() -> dict[str, Any]:
    return {
        "scope_state": "no-local-answer",
        "terminal_only": False,
        "visible_in_terminal": False,
        "saved_json_display": "validation-only",
        "public_artifact_safe": True,
        "summary": "The check JSON is a validation record, not an answer transcript.",
    }


def check_shareable_summary() -> dict[str, Any]:
    return {
        "saved_artifacts_public_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "answer_scope_state": "no-local-answer",
        "summary": "Share the checked Beta JSON/Markdown, support bundle, and this check JSON; keep private env and runtime state local.",
    }


def check_result_from_payload(
    payload: dict[str, Any],
    *,
    output_dir: Path,
    mode: str,
    expected_tokens: int,
    check_source: str,
    checked_beta_report: str = "",
    hf_model_id: str = pack.DEFAULT_HF_MODEL_ID,
    load_errors: list[str] | None = None,
) -> dict[str, Any]:
    errors = list(load_errors or [])
    if payload:
        errors.extend(validate_report(payload, mode=mode, expected_tokens=expected_tokens))
    elif not errors:
        errors.append("beta_report_empty")
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "mode": mode,
        "max_new_tokens": expected_tokens,
        "output_dir": str(output_dir),
        "check_source": check_source,
        "checked_beta_report": checked_beta_report,
        "beta_output_dir": str(payload.get("output_dir") or "") if payload else "",
        "errors": errors,
        "beta_schema": payload.get("schema") if payload else None,
        "beta_ok": payload.get("ok") if payload else None,
        "hf_model_id": str((payload.get("beta") if isinstance(payload.get("beta"), dict) else {}).get("hf_model_id") or hf_model_id or pack.DEFAULT_HF_MODEL_ID),
        "diagnosis_codes": ["public_real_llm_swarm_beta_check_ready"] if not errors else ["public_real_llm_swarm_beta_check_blocked"],
        "artifacts": {
            "public_real_llm_swarm_beta_json": artifact_path(
                payload,
                "public_real_llm_swarm_beta_json",
                "public_real_llm_swarm_beta.json",
            ) if payload else checked_beta_report,
            "public_real_llm_swarm_beta_markdown": artifact_path(
                payload,
                "public_real_llm_swarm_beta_markdown",
                "public_real_llm_swarm_beta.md",
            ) if payload else "",
            "support_bundle_json": artifact_path(payload, "support_bundle_json", "support_bundle.json") if payload else "",
            "runbook": artifact_path(payload, "runbook", "PUBLIC_REAL_LLM_SWARM_BETA.md") if payload else "",
        },
    }
    result["artifact_summary"] = check_artifact_summary(result)
    result["recommended_check_command"] = recommended_check_command(result)
    result["recommended_next_command"] = recommended_next_command(result)
    result["next_commands"] = next_commands(result)
    result["user_status"] = user_status(result)
    result["review_summary"] = check_review_summary(result)
    result["operator_action"] = result["review_summary"]["operator_action"]
    result["output_request"] = check_output_request_summary()
    result["prompt_scope"] = check_prompt_scope_summary(payload or {})
    result["answer_scope"] = check_answer_scope_summary()
    result["shareable_summary"] = check_shareable_summary()
    check_json_errors = sensitive_check_json_errors(result)
    if check_json_errors:
        result["ok"] = False
        result["errors"] = list(result.get("errors") or []) + check_json_errors
        result["diagnosis_codes"] = ["public_real_llm_swarm_beta_check_blocked"]
        result["recommended_next_command"] = recommended_next_command(result)
        result["next_commands"] = next_commands(result)
        result["user_status"] = user_status(result)
        result["review_summary"] = check_review_summary(result)
        result["operator_action"] = result["review_summary"]["operator_action"]
    return result


def sensitive_check_json_errors(result: dict[str, Any]) -> list[str]:
    checked = {
        key: result.get(key)
        for key in [
            "schema",
            "ok",
            "mode",
            "max_new_tokens",
            "output_dir",
            "beta_schema",
            "beta_ok",
            "diagnosis_codes",
            "artifacts",
            "artifact_summary",
            "review_summary",
            "operator_action",
            "recommended_check_command",
            "recommended_next_command",
            "next_commands",
            "user_status",
            "output_request",
            "prompt_scope",
            "answer_scope",
            "shareable_summary",
            "check_source",
            "checked_beta_report",
            "beta_output_dir",
        ]
    }
    encoded = json.dumps(checked, sort_keys=True)
    return [f"sensitive_check_json:{fragment}" for fragment in SECRET_FRAGMENTS if fragment in encoded]


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    beta_report = Path(args.beta_report).expanduser() if args.beta_report else None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif beta_report:
        output_dir = beta_report.resolve().parent
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="crowdtensor_public_real_llm_beta_check_"))
    if beta_report:
        payload, load_errors = payload_for_existing_beta_report(beta_report)
        result = check_result_from_payload(
            payload,
            output_dir=output_dir,
            mode=args.mode,
            expected_tokens=args.max_new_tokens,
            check_source="beta-report",
            checked_beta_report=str(beta_report.resolve()),
            hf_model_id=args.hf_model_id,
            load_errors=load_errors,
        )
    else:
        if args.mode == pack.MODE_LOCAL_MODEL_VARIANT:
            payload = build_fake_local_model_variant(output_dir, model_id=args.hf_model_id, tokens=args.max_new_tokens)
        else:
            payload = build_fake_release(output_dir, tokens=args.max_new_tokens)
        result = check_result_from_payload(
            payload,
            output_dir=output_dir,
            mode=args.mode,
            expected_tokens=args.max_new_tokens,
            check_source="ci-fixture",
            hf_model_id=args.hf_model_id,
        )
    write(output_dir / "public_real_llm_swarm_beta_check.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def print_human_summary(result: dict[str, Any]) -> None:
    review = result.get("review_summary") if isinstance(result.get("review_summary"), dict) else {}
    artifact_summary = result.get("artifact_summary") if isinstance(result.get("artifact_summary"), dict) else {}
    user = result.get("user_status") if isinstance(result.get("user_status"), dict) else {}
    recommended = (
        result.get("recommended_check_command")
        if isinstance(result.get("recommended_check_command"), dict)
        else review.get("recommended_check_command") if isinstance(review.get("recommended_check_command"), dict) else {}
    )
    recommended_next = (
        result.get("recommended_next_command")
        if isinstance(result.get("recommended_next_command"), dict)
        else review.get("recommended_next_command") if isinstance(review.get("recommended_next_command"), dict) else {}
    )
    output_request = result.get("output_request") if isinstance(result.get("output_request"), dict) else {}
    prompt_scope = result.get("prompt_scope") if isinstance(result.get("prompt_scope"), dict) else {}
    answer_scope = result.get("answer_scope") if isinstance(result.get("answer_scope"), dict) else {}
    shareable = result.get("shareable_summary") if isinstance(result.get("shareable_summary"), dict) else {}
    print(f"Public Real-LLM Swarm Beta check ready: {result.get('ok')}")
    print(f"  mode: {result.get('mode')}")
    print(f"  max_new_tokens: {result.get('max_new_tokens')}")
    print(f"  check_source: {result.get('check_source')}")
    if result.get("checked_beta_report"):
        print(f"  checked_beta_report: {result.get('checked_beta_report')}")
    if user:
        print(
            "  status: "
            f"state={user.get('state')} "
            f"next={user.get('next_step')} "
            f"recommended={user.get('recommended_label')} "
            f"errors={user.get('error_count')} "
            f"public_artifact_safe={user.get('public_artifact_safe')}"
        )
    if review:
        print(
            "  review: "
            f"state={review.get('state')} "
            f"next={review.get('next_step')} "
            f"inspect={review.get('inspect_first')} "
            f"errors={review.get('error_count')} "
            f"public_artifact_safe={review.get('public_artifact_safe')}"
        )
    if artifact_summary:
        print(
            "  artifacts: "
            f"inspect={artifact_summary.get('inspect_first')} "
            f"json={artifact_summary.get('machine_readable')} "
            f"support={artifact_summary.get('support_bundle')} "
            f"check={artifact_summary.get('check_json')}"
        )
    if recommended:
        print(f"  recommended_check: {recommended.get('command_line')}")
    if recommended_next:
        print(f"  recommended_next: {recommended_next.get('command_line')}")
    for index, item in enumerate((result.get("next_commands") or []), start=1):
        if isinstance(item, dict):
            print(f"  next[{index}] {item.get('label')}: {item.get('command_line')}")
    if output_request:
        print(
            "  output_request: "
            f"include_output={output_request.get('include_output')} "
            f"raw_prompt_public={output_request.get('raw_prompt_public')} "
            f"raw_generated_text_public={output_request.get('raw_generated_text_public')} "
            f"generated_token_ids_public={output_request.get('generated_token_ids_public')} "
            f"public_artifact_safe={output_request.get('public_artifact_safe')}"
        )
    if prompt_scope:
        print(
            "  prompt_scope: "
            f"source={prompt_scope.get('source')} "
            f"count={prompt_scope.get('prompt_count')} "
            f"inline_prompt_text={prompt_scope.get('inline_prompt_text')} "
            f"raw_prompt_public={prompt_scope.get('raw_prompt_public')} "
            f"public_artifact_safe={prompt_scope.get('public_artifact_safe')}"
        )
    if answer_scope:
        print(
            "  answer_scope: "
            f"state={answer_scope.get('scope_state')} "
            f"saved_json={answer_scope.get('saved_json_display')} "
            f"public_artifact_safe={answer_scope.get('public_artifact_safe')}"
        )
    if shareable:
        print(
            "  shareable: "
            f"saved_artifacts={shareable.get('saved_artifacts_public_safe')} "
            f"raw_prompt_public={shareable.get('raw_prompt_public')} "
            f"raw_generated_text_public={shareable.get('raw_generated_text_public')} "
            f"generated_token_ids_public={shareable.get('generated_token_ids_public')} "
            f"answer_scope_state={shareable.get('answer_scope_state')}"
        )
    if result.get("operator_action"):
        print(f"  action: {result.get('operator_action')}")
    print(f"  output: {result.get('output_dir')}")
    print(f"  diagnosis: {', '.join(result.get('diagnosis_codes') or [])}")
    errors = result.get("errors") if isinstance(result.get("errors"), list) else []
    if errors:
        print("  errors:")
        for error in errors[:12]:
            print(f"    - {error}")
        if len(errors) > 12:
            print(f"    - ... {len(errors) - 12} more")
    for name, path in sorted((result.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {path}")
    print(f"  artifact public_real_llm_swarm_beta_check_json: {Path(str(result.get('output_dir') or '')) / 'public_real_llm_swarm_beta_check.json'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Public Real-LLM Swarm Inference Beta v1.")
    parser.add_argument("--mode", choices=["release", pack.MODE_LOCAL_MODEL_VARIANT], default="release")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--beta-report", default="", help="Validate an existing public_real_llm_swarm_beta.json instead of building the CI-safe fixture.")
    parser.add_argument("--hf-model-id", default=pack.DEFAULT_HF_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    return args


def main() -> None:
    args = parse_args()
    result = run_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print_human_summary(result)
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
