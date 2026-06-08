#!/usr/bin/env python3
"""Acceptance checks for Public Swarm Inference v2."""

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

import public_swarm_inference_v2_pack as pack  # noqa: E402


SCHEMA = "public_swarm_inference_v2_check_v1"
REQUIRED_CODES = {
    "public_swarm_inference_v2_ready",
    "public_swarm_inference_v2_preview_ready",
    "public_swarm_v2_local_p2p_generate_ready",
    "public_swarm_v2_16_token_generation_ready",
    "public_swarm_v2_dual_stage_kv_cache_ready",
    "public_swarm_v2_model_match_ready",
    "public_swarm_v2_signed_or_real_p2p_ready",
    "public_swarm_v2_external_evidence_ready",
    "public_swarm_v2_external_stage_rows_ready",
    "public_swarm_v2_stage_requeue_rescue_ready",
    "public_swarm_v2_cuda_optional_fail_closed_ready",
    "stage_latency_ready",
    "throughput_summary_ready",
    "memory_or_vram_summary_ready",
    "serve_join_generate_p2p_primary_path",
    "read_only_workload",
    "not_production",
    "not_coordinator_free",
    "not_hivemind_petals_production",
    "not_large_model_serving",
}
LOCAL_MODEL_VARIANT_REQUIRED_CODES = {
    "public_swarm_inference_v2_local_model_variant_ready",
    "public_swarm_v2_local_model_variant_ready",
    "public_swarm_v2_local_model_variant_model_match_ready",
    "public_swarm_v2_external_validation_not_claimed",
    "public_swarm_v2_local_p2p_generate_ready",
    "public_swarm_v2_16_token_generation_ready",
    "public_swarm_v2_dual_stage_kv_cache_ready",
    "public_swarm_v2_model_match_ready",
    "public_swarm_v2_signed_or_real_p2p_ready",
    "public_swarm_v2_real_p2p_local_ready",
    "public_swarm_v2_real_p2p_local_requeue_ready",
    "public_swarm_v2_stage_requeue_rescue_ready",
    "public_swarm_v2_cuda_optional_fail_closed_ready",
    "stage_latency_ready",
    "throughput_summary_ready",
    "memory_or_vram_summary_ready",
    "serve_join_generate_p2p_primary_path",
    "read_only_workload",
    "not_production",
    "not_coordinator_free",
    "not_hivemind_petals_production",
    "not_large_model_serving",
}
LOCAL_MODEL_VARIANT_BLOCKED_CODES = {
    "public_swarm_inference_v2_ready",
    "public_swarm_v2_external_stage_rows_ready",
    "external_runtime_verified",
    "external_stage_requeue_ready",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def completed(payload: dict[str, Any], returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


def fake_usable_report(tokens: int = 16, *, model_id: str = pack.DEFAULT_HF_MODEL_ID) -> dict[str, Any]:
    model = {
        "expected_hf_model_id": model_id,
        "observed_hf_model_id": model_id,
        "model_id_present": True,
        "model_id_match": True,
        "compatible": True,
        "default_model_retained_evidence": False,
    }
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
                "generated_text_hash": "sha256:fake-local-1",
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
                "generated_text_hash": "sha256:fake-local-2",
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
        "schema": pack.USABLE_SCHEMA,
        "ok": True,
        "mode": "local",
        "usable_swarm": {
            "ready": True,
            "hf_model_id": model_id,
            "user_surface": ["p2pd", "serve", "join", "generate"],
        },
        "readiness": {
            "p2p_product_path": {
                "ready": True,
                "model": model,
                "route_ready": True,
                "p2p_counts_ready": True,
                "real_generate_ready": True,
                "kv_cache_ready": True,
                "kv_cache": {
                    "schema": "p2p_real_generate_dual_stage_kv_cache_v1",
                    "ready": True,
                    "expected_hit_count_per_stage": max(0, tokens - 1),
                    "stage0": {
                        "schema": "real_llm_stage0_kv_cache_v1",
                        "stage": "stage0_prefix",
                        "ready": True,
                        "ready_count": tokens,
                        "hit_count": max(0, tokens - 1),
                    },
                    "stage1": {
                        "schema": "real_llm_stage1_kv_cache_v1",
                        "stage": "stage1_suffix",
                        "ready": True,
                        "ready_count": tokens,
                        "hit_count": max(0, tokens - 1),
                    },
                    "raw_activations_public": False,
                    "raw_token_inputs_public": False,
                },
                "generated_token_count": tokens,
                "max_new_tokens": tokens,
                "generation_target_ready": True,
                "accepted_rows": tokens * 2,
                "accepted_rows_ready": True,
                "route_source": "p2p-discovery",
                "batch": batch,
                "batch_ready": True,
                "stream": stream,
                "stream_ready": True,
                "distinct_stage_miners": True,
                "stage_rescue_ready": True,
                "real_stage_rescue_ready": True,
                "usable_evidence_source": "source_gate",
                "route": {
                    "route_source": "p2p-discovery",
                    "usable_now": True,
                    "matched_capabilities": {
                        "real_llm_sharded_stage0": "stage0",
                        "real_llm_sharded_stage1": "stage1",
                    },
                },
                "stage_assignment": {
                    "stage0_miner_id": "stage0",
                    "stage1_miner_id": "stage1",
                    "distinct_stage_miners": True,
                },
                "generation": {
                    "generated_token_count": tokens,
                    "max_new_tokens": tokens,
                    "generated_text_hash": "sha256:fake-local",
                    "decoded_tokens_match": True,
                    "multi_token_generation_ready": True,
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                },
            }
        },
        "diagnosis_codes": [
            "usable_swarm_inference_ready",
            "p2p_real_generate_ready",
            "p2p_stage_rescue_ready",
            "p2p_real_stage_rescue_ready",
            "distinct_stage_miners",
            "stage_assignment_valid",
        ],
    }


def fake_preview_report(tokens: int = 16) -> dict[str, Any]:
    return {
        "schema": pack.PREVIEW_V04_SCHEMA,
        "ok": True,
        "mode": "evidence-import",
        "preview": {
            "ready": True,
            "stage_latency_ready": True,
            "throughput_summary_ready": True,
            "memory_or_vram_summary_ready": True,
        },
        "performance": {
            "stage_latency": {
                "stage_latency_ready": True,
                "stage0_elapsed_ms": 1200.0,
                "stage1_elapsed_ms": 900.0,
                "stage_total_seconds": 2.1,
            },
            "throughput": {
                "throughput_summary_ready": True,
                "generated_tokens_per_stage_second": round(tokens / 2.1, 6),
            },
            "memory_or_vram": {
                "memory_or_vram_summary_ready": True,
                "stage_gpu_memory_reduced": True,
            },
        },
        "diagnosis_codes": [
            "public_swarm_preview_v04_ready",
            "stage_latency_ready",
            "throughput_summary_ready",
            "memory_or_vram_summary_ready",
            "multi_token_generation_ready",
        ],
    }


def fake_real_p2p_report(tokens: int = 16, *, model_id: str = pack.DEFAULT_HF_MODEL_ID) -> dict[str, Any]:
    return {
        "schema": pack.REAL_P2P_SCHEMA,
        "ok": True,
        "mode": "kaggle-auto",
        "hf_model_id": model_id,
        "expected_hf_model_id": model_id,
        "external": {"external_runtime_verified": True},
        "p2p": {
            "backend": "real",
            "discovery_backend": "http-provider-store",
            "route": {"route_source": "real-p2p-discovery"},
        },
        "generation": {
            "generated_token_count": tokens,
            "max_new_tokens": tokens,
            "generated_text_hash": "sha256:fake-external",
            "decoded_tokens_match": True,
        },
        "stage_assignment": {
            "completed_rows": tokens * 2,
            "distinct_stage_miners": True,
            "max_generation_step": max(0, tokens - 1),
            "stage0_miner_id": "external-stage0",
            "stage1_miner_id": "external-stage1",
        },
        "ledger": {"accepted_rows": tokens * 2, "error": ""},
        "diagnosis_codes": [
            "real_p2p_swarm_inference_core_rc_ready",
            "real_p2p_local_stage_requeue_ready",
            "local_stage_requeue_ready",
            "stage_requeue_ready",
            "external_runtime_verified",
            "external_stage_requeue_ready",
            "live_stage0_requeue_ready",
            "live_stage1_requeue_ready",
            "accepted_result_after_requeue",
            "rescue_miner_used",
            "kaggle_kernels_deleted",
            "real_p2p_kaggle_private_artifacts_cleaned",
            "token_rotation_required",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "libp2p_or_real_p2p_discovery_ready",
            "libp2p_discovery_backend_ready",
            "real_p2p_signed_provider_records_ready",
            "real_p2p_provider_store_ready",
            "real_p2p_core_rc_model_metadata_ready",
        ],
        "live_requeue_summary": {
            "enabled": True,
            "scope": "local-smoke",
            "failure_mode": "kill-stage1-after-claim",
            "target_stage": "stage1",
            "claim_observed": True,
            "victim_kernel_deleted": True,
            "victim_process_terminated": True,
            "lease_expired": True,
            "rescue_miner_used": True,
            "rescued_result": True,
            "accepted_result_after_requeue": True,
            "victim_result_accepted": False,
        },
    }


def fake_gpu_report(tokens: int = 16) -> dict[str, Any]:
    return {
        "schema": pack.GPU_SCHEMA,
        "ok": True,
        "mode": "kaggle-auto",
        "generation": {
            "generated_token_count": tokens,
            "max_new_tokens": tokens,
            "generated_text_hash": "sha256:fake-gpu",
        },
        "diagnosis_codes": [
            "public_swarm_gpu_beta_ready",
            "gpu_runtime_ready",
            "cuda_runtime_available",
            "hf_transformers_cuda_ready",
            "external_gpu_runtime_verified",
            "kaggle_kernels_deleted",
            "distinct_stage_miners",
            "stage_assignment_valid",
        ],
    }


def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    joined = " ".join(command)
    tokens = int(command[command.index("--max-new-tokens") + 1]) if "--max-new-tokens" in command else 16
    model_id = command[command.index("--hf-model-id") + 1] if "--hf-model-id" in command else pack.DEFAULT_HF_MODEL_ID
    if "usable_swarm_inference_pack.py" in joined:
        return completed(fake_usable_report(tokens, model_id=model_id))
    if "real_p2p_swarm_inference_core_rc_pack.py" in joined:
        payload = fake_real_p2p_report(tokens, model_id=model_id)
        payload["mode"] = "local-smoke"
        payload["external"] = {"external_runtime_verified": False}
        return completed(payload)
    raise AssertionError(command)


def validate_report(payload: dict[str, Any], *, mode: str) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if payload.get("ok") is not True:
        errors.append("v2_not_ready")
    if payload.get("mode") != mode:
        errors.append("mode_mismatch")
    v2 = payload.get("public_swarm_v2") if isinstance(payload.get("public_swarm_v2"), dict) else {}
    if v2.get("ready") is not True:
        errors.append("summary_not_ready")
    required_tokens = int(v2.get("max_new_tokens") or 16)
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    local = readiness.get("local_p2p_generate") if isinstance(readiness.get("local_p2p_generate"), dict) else {}
    if local.get("ready") is not True:
        errors.append("local_p2p_not_ready")
    if int(local.get("generated_token_count") or 0) < required_tokens:
        errors.append(f"local_token_count_below_{required_tokens}")
    if local.get("route_source") != "p2p-discovery":
        errors.append("local_route_not_p2p_discovery")
    if local.get("kv_cache_ready") is not True:
        errors.append("local_kv_cache_not_ready")
    external = readiness.get("external_validation") if isinstance(readiness.get("external_validation"), dict) else {}
    local_model_variant = mode == pack.MODE_LOCAL_MODEL_VARIANT
    if local_model_variant:
        if v2.get("local_model_variant_only") is not True:
            errors.append("local_model_variant_not_marked")
        if v2.get("external_validation_claimed") is not False:
            errors.append("external_validation_claimed_in_local_variant")
        if external.get("ready") is True:
            errors.append("external_claimed_ready_in_local_variant")
    else:
        if external.get("ready") is not True:
            errors.append("external_not_ready")
        if int(external.get("accepted_rows") or 0) < required_tokens * 2:
            errors.append("external_accepted_rows_below_target")
        if external.get("accepted_rows_ready") is not True:
            errors.append("external_accepted_rows_not_ready")
    p2p = readiness.get("p2p_route_hardening") if isinstance(readiness.get("p2p_route_hardening"), dict) else {}
    if p2p.get("ready") is not True:
        errors.append("real_p2p_not_ready")
    cuda = readiness.get("cuda_optional") if isinstance(readiness.get("cuda_optional"), dict) else {}
    if cuda.get("fail_closed_ready") is not True:
        errors.append("cuda_fail_closed_not_ready")
    perf = readiness.get("performance") if isinstance(readiness.get("performance"), dict) else {}
    for key in ["stage_latency_ready", "throughput_summary_ready", "memory_or_vram_summary_ready"]:
        if perf.get(key) is not True:
            errors.append(f"performance_missing:{key}")
    codes = set(payload.get("diagnosis_codes") or [])
    required_codes = LOCAL_MODEL_VARIANT_REQUIRED_CODES if local_model_variant else REQUIRED_CODES
    for code in sorted(required_codes - codes):
        errors.append(f"missing_code:{code}")
    if mode in {pack.MODE_LOCAL, pack.MODE_LOCAL_MODEL_VARIANT}:
        for code in [
            "public_swarm_v2_real_p2p_local_ready",
            "public_swarm_v2_real_p2p_local_requeue_ready",
        ]:
            if code not in codes:
                errors.append(f"missing_code:{code}")
    if local_model_variant:
        for code in sorted(LOCAL_MODEL_VARIANT_BLOCKED_CODES & codes):
            errors.append(f"unexpected_code:{code}")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    for key in [
        "coordinator_backed_task_execution",
        "p2p_discovery_primary_path",
        "persistent_dual_stage_kv_cache_required",
        "signed_or_real_p2p_preferred_when_available",
        "p2p_lite_fallback_explicit",
        "cuda_optional_fail_closed_ready",
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
    inline_prompt_text = prompt_scope.get("source") in {"prompt-text", "prompt-texts"}
    if prompt_scope.get("inline_prompt_text") is not inline_prompt_text:
        errors.append("prompt_scope_inline_prompt_text_mismatch")
    if prompt_scope.get("terminal_next_commands_local_private") is not inline_prompt_text:
        errors.append("prompt_scope_terminal_next_commands_mismatch")
    if prompt_scope.get("terminal_logs_local_private") is not inline_prompt_text:
        errors.append("prompt_scope_terminal_logs_mismatch")
    if prompt_scope.get("saved_artifacts_prompt_placeholders") is not True:
        errors.append("prompt_scope_saved_placeholders_mismatch")
    if prompt_scope.get("saved_artifacts_public_safe") is not True:
        errors.append("prompt_scope_saved_artifacts_public_safe_mismatch")
    if prompt_scope.get("prefer_prompt_file_or_stdin_for_shareable_logs") is not inline_prompt_text:
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
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    for name in ["public_swarm_inference_v2_json", "public_swarm_inference_v2_markdown", "support_bundle_json", "runbook"]:
        artifact = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
        if artifact.get("present") is not True:
            errors.append(f"artifact_missing:{name}")
    return sorted(set(errors))


def run_check(args: argparse.Namespace, *, runner: pack.Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="crowdtensor_public_swarm_v2_check_")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "sources"
    external_model_id = pack.DEFAULT_HF_MODEL_ID if args.mode == pack.MODE_LOCAL_MODEL_VARIANT else args.hf_model_id
    write_json(source_dir / "usable.json", fake_usable_report(args.max_new_tokens, model_id=args.hf_model_id))
    write_json(source_dir / "preview.json", fake_preview_report(args.max_new_tokens))
    write_json(source_dir / "real_p2p.json", fake_real_p2p_report(args.max_new_tokens, model_id=external_model_id))
    write_json(source_dir / "gpu.json", fake_gpu_report(args.max_new_tokens))
    argv = [
        args.mode,
        "--output-dir",
        str(output_dir / "v2"),
        "--usable-report",
        str(source_dir / "usable.json"),
        "--preview-report",
        str(source_dir / "preview.json"),
        "--real-p2p-report",
        str(source_dir / "real_p2p.json"),
        "--gpu-report",
        str(source_dir / "gpu.json"),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--hf-model-id",
        args.hf_model_id,
        "--p2p-port",
        str(args.p2p_port),
        "--coordinator-port",
        str(args.coordinator_port),
        "--real-p2p-port",
        str(args.real_p2p_port),
        "--real-p2p-coordinator-port",
        str(args.real_p2p_coordinator_port),
        "--real-p2p-libp2p-port",
        str(args.real_p2p_libp2p_port),
        "--startup-timeout",
        str(args.startup_timeout),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--http-timeout",
        str(args.http_timeout),
    ]
    if args.stream_generation:
        argv.append("--stream-generation")
    if args.prompt_texts:
        argv.extend(["--prompt-texts", args.prompt_texts])
    if args.hf_cache_dir:
        argv.extend(["--hf-cache-dir", args.hf_cache_dir])
    parsed = pack.parse_args(argv)
    real_local = bool(args.real_local and args.mode in {pack.MODE_LOCAL, pack.MODE_LOCAL_MODEL_VARIANT})
    report = pack.build_report(parsed, runner=runner if real_local else fake_runner)
    errors = validate_report(report, mode=args.mode)
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "mode": args.mode,
        "real_local": real_local,
        "max_new_tokens": args.max_new_tokens,
        "hf_model_id": args.hf_model_id,
        "output_dir": str(output_dir),
        "errors": errors,
        "public_swarm_inference_v2_schema": report.get("schema"),
        "public_swarm_inference_v2_ok": report.get("ok"),
        "diagnosis_codes": ["public_swarm_inference_v2_check_ready"] if not errors else ["public_swarm_inference_v2_check_failed"],
        "artifacts": {
            "public_swarm_inference_v2_json": str(output_dir / "v2" / "public_swarm_inference_v2.json"),
            "public_swarm_inference_v2_markdown": str(output_dir / "v2" / "public_swarm_inference_v2.md"),
            "support_bundle_json": str(output_dir / "v2" / "support_bundle.json"),
        },
    }
    write_json(output_dir / "public_swarm_inference_v2_check.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Public Swarm Inference v2.")
    parser.add_argument("--mode", choices=pack.MODES, default=pack.MODE_EVIDENCE_IMPORT)
    parser.add_argument("--real-local", action="store_true", help="run real local usable + real-P2P child gates for local modes")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--hf-model-id", default=pack.DEFAULT_HF_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--prompt-texts", default="")
    parser.add_argument("--stream-generation", action="store_true")
    parser.add_argument("--p2p-port", type=int, default=9888)
    parser.add_argument("--coordinator-port", type=int, default=9889)
    parser.add_argument("--real-p2p-port", type=int, default=pack.DEFAULT_REAL_P2P_LOCAL_P2P_PORT)
    parser.add_argument("--real-p2p-coordinator-port", type=int, default=pack.DEFAULT_REAL_P2P_LOCAL_COORDINATOR_PORT)
    parser.add_argument("--real-p2p-libp2p-port", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=420.0)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 8 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 8 and 32")
    if args.real_local and args.mode not in {pack.MODE_LOCAL, pack.MODE_LOCAL_MODEL_VARIANT}:
        raise SystemExit("--real-local is only valid with --mode local or --mode local-model-variant")
    for name in ["p2p_port", "coordinator_port", "real_p2p_port", "real_p2p_coordinator_port"]:
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.real_p2p_libp2p_port < 0:
        raise SystemExit("--real-p2p-libp2p-port must be non-negative")
    return args


def main() -> None:
    report = run_check(parse_args())
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
