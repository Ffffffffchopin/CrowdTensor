"""Public session/task protocol helpers for CrowdTensor product-facing flows.

This module deliberately contains no networking or model execution code.  It is
the narrow, safe summary layer between user-facing commands, Coordinator task
creation, P2P-lite route discovery, and public evidence artifacts.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .protocol import (
    DEFAULT_PROTOCOL_VERSION,
    REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
    REAL_LLM_SHARDED_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_STAGE1_CAPABILITY,
    WORKLOAD_REAL_LLM_SHARDED_INFER,
)


SESSION_PROTOCOL_SCHEMA = "session_protocol_v1"
SESSION_RESULT_SCHEMA = "session_result_summary_v1"
SESSION_STREAM_EVENT_SCHEMA = "session_stream_event_v1"
MAX_PROMPT_CHARS = 256
MAX_NEW_TOKENS = 32
MAX_BATCH_REQUESTS = 4

WORKLOAD_ALIASES = {
    "real-llm-sharded": WORKLOAD_REAL_LLM_SHARDED_INFER,
    "real-llm-shard": WORKLOAD_REAL_LLM_SHARDED_INFER,
    "gpu-generation": WORKLOAD_REAL_LLM_SHARDED_INFER,
    "cpu-real-llm": WORKLOAD_REAL_LLM_SHARDED_INFER,
}

RAW_PUBLIC_KEYS = {
    "prompt",
    "prompt_text",
    "prompt_texts",
    "output_text",
    "generated_text",
    "generated_token_ids",
    "next_token_text",
    "next_token_id",
    "lease_token",
    "idempotency_key",
    "activation_results",
    "hidden_state",
    "input_ids",
    "logits",
    "inference_result",
    "inference_results",
    "sharded_inference_result",
    "external_llm_result",
    "external_llm_results",
}
SECRET_VALUE_FRAGMENTS = (
    "CROWDTENSOR_ADMIN_TOKEN",
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "Bearer ",
)


def stable_hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def stable_hash_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_workload(workload: str | None) -> str:
    normalized = str(workload or WORKLOAD_REAL_LLM_SHARDED_INFER).strip()
    return WORKLOAD_ALIASES.get(normalized, normalized)


def normalize_backend(backend: str | None) -> str:
    normalized = str(backend or "cpu").strip().lower()
    if normalized in {"gpu", "cuda", "hf_transformers_cuda"}:
        return "cuda"
    if normalized in {"cpu", "hf_transformers_cpu", ""}:
        return "cpu"
    raise ValueError(f"unsupported session backend: {backend}")


def normalize_stage_mode(stage_mode: str | None) -> str:
    normalized = str(stage_mode or "split").strip().lower()
    if normalized not in {"stage0", "stage1", "both", "split"}:
        raise ValueError(f"unsupported stage_mode: {stage_mode}")
    return normalized


def required_stage_capabilities(*, backend: str, stage_mode: str = "split") -> list[str]:
    resolved_backend = normalize_backend(backend)
    resolved_stage = normalize_stage_mode(stage_mode)
    if resolved_backend == "cuda":
        stage0 = REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY
        stage1 = REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY
    else:
        stage0 = REAL_LLM_SHARDED_STAGE0_CAPABILITY
        stage1 = REAL_LLM_SHARDED_STAGE1_CAPABILITY
    if resolved_stage == "stage0":
        return [stage0]
    if resolved_stage == "stage1":
        return [stage1]
    return [stage0, stage1]


def build_session_request(
    *,
    prompt_text: str,
    prompt_texts: list[str] | None = None,
    workload_type: str = WORKLOAD_REAL_LLM_SHARDED_INFER,
    backend: str = "cpu",
    hf_model_id: str = "",
    stage_mode: str = "split",
    max_new_tokens: int = 16,
    scenario_id: str = "public-swarm-product-rc",
    route_source: str = "coordinator-url",
) -> dict[str, Any]:
    prompts = [str(item) for item in (prompt_texts if prompt_texts is not None else [prompt_text]) if str(item)]
    if not prompts:
        raise ValueError("prompt_text is required")
    if len(prompts) > MAX_BATCH_REQUESTS:
        raise ValueError(f"prompt_texts must contain at most {MAX_BATCH_REQUESTS} prompts")
    for prompt in prompts:
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt_text must be at most {MAX_PROMPT_CHARS} characters")
    token_limit = int(max_new_tokens)
    if token_limit < 1 or token_limit > MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")
    workload = normalize_workload(workload_type)
    resolved_backend = normalize_backend(backend)
    resolved_stage = normalize_stage_mode(stage_mode)
    if workload != WORKLOAD_REAL_LLM_SHARDED_INFER:
        raise ValueError(f"unsupported public session workload_type: {workload_type}")
    capabilities = required_stage_capabilities(backend=resolved_backend, stage_mode=resolved_stage)
    model_id = str(hf_model_id or "sshleifer/tiny-gpt2").strip() or "sshleifer/tiny-gpt2"
    request = {
        "schema": SESSION_PROTOCOL_SCHEMA,
        "workload_type": workload,
        "protocol_version": DEFAULT_PROTOCOL_VERSION,
        "backend": resolved_backend,
        "hf_model_id": model_id,
        "stage_mode": resolved_stage,
        "scenario_id": str(scenario_id or "public-swarm-product-rc"),
        "max_new_tokens": token_limit,
        "request_count": len(prompts),
        "prompt_hash": stable_hash_payload([stable_hash_text(prompt) for prompt in prompts]),
        "prompt_hashes": [stable_hash_text(prompt) for prompt in prompts],
        "prompt_chars": sum(len(prompt) for prompt in prompts),
        "prompt_char_counts": [len(prompt) for prompt in prompts],
        "batch": {
            "enabled": len(prompts) > 1,
            "max_request_count": MAX_BATCH_REQUESTS,
            "request_count": len(prompts),
        },
        "route_requirements": {
            "runtime": "python-cli",
            "backend": resolved_backend,
            "stage_count": 2,
            "required_capabilities": capabilities,
            "route_source": str(route_source or "coordinator-url"),
        },
        "safety": {
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "read_only_workload": True,
            "not_production": True,
            "not_full_p2p": True,
        },
    }
    assert_public_safe(request)
    return request


def coordinator_payload_for_request(
    session_request: dict[str, Any],
    *,
    prompt_text: str,
    prompt_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Build the private Coordinator payload corresponding to a public request."""
    prompts = [str(item) for item in (prompt_texts if prompt_texts is not None else [prompt_text]) if str(item)]
    prompts = prompts[:MAX_BATCH_REQUESTS]
    return {
        "workload_type": session_request.get("workload_type") or WORKLOAD_REAL_LLM_SHARDED_INFER,
        "request_count": max(1, min(int(session_request.get("request_count") or len(prompts) or 1), MAX_BATCH_REQUESTS)),
        "max_new_tokens": int(session_request.get("max_new_tokens") or 1),
        "runtime": "python-cli",
        "backend": session_request.get("backend") or "cpu",
        "hf_model_id": session_request.get("hf_model_id") or "sshleifer/tiny-gpt2",
        "prompt": prompts[0] if prompts else str(prompt_text or ""),
        "prompt_texts": prompts,
        "scenario_id": session_request.get("scenario_id") or "public-swarm-product-rc",
        "partition_mode": "stage-local",
    }


def peer_has_capability(peer: dict[str, Any], capability: str) -> bool:
    caps = peer.get("capabilities") if isinstance(peer.get("capabilities"), dict) else {}
    collections = [
        caps.get("real_llm_sharded_stage_capabilities"),
        caps.get("capabilities"),
        caps.get("supported_capabilities"),
    ]
    for values in collections:
        if isinstance(values, list) and capability in values:
            return True
    return False


def peer_model_compatible(peer: dict[str, Any], expected_model_id: str) -> bool:
    expected = str(expected_model_id or "sshleifer/tiny-gpt2").strip() or "sshleifer/tiny-gpt2"
    caps = peer.get("capabilities") if isinstance(peer.get("capabilities"), dict) else {}
    scalar_values = [
        peer.get("hf_model_id"),
        peer.get("model_id"),
        caps.get("hf_model_id"),
        caps.get("model_id"),
    ]
    for value in scalar_values:
        if isinstance(value, str) and value.strip():
            return value.strip() == expected
    collections = [
        caps.get("supported_hf_model_ids"),
        caps.get("supported_model_ids"),
        caps.get("hf_model_ids"),
        peer.get("supported_hf_model_ids"),
        peer.get("supported_model_ids"),
    ]
    for values in collections:
        if isinstance(values, list) and values:
            return expected in {str(item).strip() for item in values if str(item).strip()}
    return True


def peer_backend_compatible(peer: dict[str, Any], expected_backend: str) -> bool:
    expected = normalize_backend(expected_backend or "cpu")
    caps = peer.get("capabilities") if isinstance(peer.get("capabilities"), dict) else {}

    def safe_normalize(value: str) -> str:
        try:
            return normalize_backend(value)
        except ValueError:
            return ""

    scalar_values = [
        peer.get("backend"),
        caps.get("backend"),
        caps.get("real_llm_backend"),
        caps.get("runtime_backend"),
    ]
    for value in scalar_values:
        if isinstance(value, str) and value.strip():
            return safe_normalize(value) == expected
    collections = [
        caps.get("supported_backends"),
        caps.get("real_llm_backends"),
        peer.get("supported_backends"),
    ]
    for values in collections:
        if isinstance(values, list) and values:
            return expected in {safe_normalize(str(item)) for item in values if str(item).strip()}
    return True


def build_route_decision(
    session_request: dict[str, Any],
    *,
    coordinator_url: str = "",
    peer_catalog: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    requirements = session_request.get("route_requirements") if isinstance(session_request.get("route_requirements"), dict) else {}
    required = [str(item) for item in requirements.get("required_capabilities") or []]
    expected_model_id = str(session_request.get("hf_model_id") or "sshleifer/tiny-gpt2")
    expected_backend = normalize_backend(str(session_request.get("backend") or "cpu"))
    peers = [peer for peer in (peer_catalog or []) if isinstance(peer, dict)]
    matched: dict[str, str] = {}
    model_mismatched: dict[str, list[str]] = {}
    backend_mismatched: dict[str, list[str]] = {}
    for capability in required:
        peer_id = ""
        for peer in peers:
            if not peer_has_capability(peer, capability):
                continue
            if not peer_model_compatible(peer, expected_model_id):
                model_mismatched.setdefault(capability, []).append(str(peer.get("peer_id") or ""))
                continue
            if not peer_backend_compatible(peer, expected_backend):
                backend_mismatched.setdefault(capability, []).append(str(peer.get("peer_id") or ""))
                continue
            peer_id = str(peer.get("peer_id") or "")
            break
        if peer_id:
            matched[capability] = peer_id
    coordinator_candidates: list[dict[str, Any]] = []
    coordinator_mismatched: list[str] = []
    for peer in peers:
        if str(peer.get("role") or "") != "coordinator":
            continue
        if not ((peer.get("urls") or {}).get("coordinator") or peer.get("coordinator_url")):
            continue
        if not peer_model_compatible(peer, expected_model_id) or not peer_backend_compatible(peer, expected_backend):
            coordinator_mismatched.append(str(peer.get("peer_id") or ""))
            continue
        coordinator_candidates.append(peer)
    resolved_coordinator = str(coordinator_url or "")
    if not resolved_coordinator and coordinator_candidates:
        first = coordinator_candidates[0]
        urls = first.get("urls") if isinstance(first.get("urls"), dict) else {}
        resolved_coordinator = str(urls.get("coordinator") or first.get("coordinator_url") or "")
    missing = [capability for capability in required if capability not in matched]
    usable = bool(resolved_coordinator) and not missing
    model_aware_count = sum(
        1
        for peer in peers
        if (
            peer.get("hf_model_id")
            or peer.get("model_id")
            or peer.get("supported_hf_model_ids")
            or peer.get("supported_model_ids")
            or (
                isinstance(peer.get("capabilities"), dict)
                and (
                    peer["capabilities"].get("hf_model_id")
                    or peer["capabilities"].get("model_id")
                    or peer["capabilities"].get("supported_hf_model_ids")
                    or peer["capabilities"].get("supported_model_ids")
                )
            )
        )
    )
    backend_aware_count = sum(
        1
        for peer in peers
        if (
            peer.get("backend")
            or peer.get("supported_backends")
            or (
                isinstance(peer.get("capabilities"), dict)
                and (
                    peer["capabilities"].get("backend")
                    or peer["capabilities"].get("real_llm_backend")
                    or peer["capabilities"].get("runtime_backend")
                    or peer["capabilities"].get("supported_backends")
                    or peer["capabilities"].get("real_llm_backends")
                )
            )
        )
    )
    route_codes = ["session_route_ready"] if usable else ["coordinator_route_missing" if not resolved_coordinator else "stage_capability_missing"]
    if model_aware_count:
        route_codes.append("session_route_model_filter_ready")
    if model_mismatched:
        route_codes.append("session_route_model_mismatch")
    if backend_aware_count:
        route_codes.append("session_route_backend_filter_ready")
    if backend_mismatched:
        route_codes.append("session_route_backend_mismatch")
    if coordinator_mismatched:
        route_codes.append("session_route_coordinator_filter_ready")
        route_codes.append("session_route_coordinator_mismatch")
    return {
        "schema": "session_route_decision_v1",
        "usable_now": usable,
        "coordinator_url_present": bool(resolved_coordinator),
        "coordinator_url": resolved_coordinator,
        "route_source": requirements.get("route_source") or ("peer-bootstrap" if coordinator_candidates else "coordinator-url"),
        "backend": expected_backend,
        "hf_model_id": expected_model_id,
        "workload_type": session_request.get("workload_type"),
        "required_capabilities": required,
        "matched_capabilities": matched,
        "missing_capabilities": missing,
        "model_filter": {
            "enabled": True,
            "expected_hf_model_id": expected_model_id,
            "model_aware_peer_count": model_aware_count,
            "mismatched_capabilities": model_mismatched,
        },
        "backend_filter": {
            "enabled": True,
            "expected_backend": expected_backend,
            "backend_aware_peer_count": backend_aware_count,
            "mismatched_capabilities": backend_mismatched,
        },
        "coordinator_filter": {
            "enabled": True,
            "mismatched_peers": coordinator_mismatched,
            "compatible_candidate_count": len(coordinator_candidates),
        },
        "diagnosis_codes": route_codes,
    }


def find_generation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_score = -1
    queue: list[Any] = [payload]
    seen: set[int] = set()
    while queue:
        item = queue.pop(0)
        if not isinstance(item, dict):
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        generation = item.get("generation") if isinstance(item.get("generation"), dict) else {}
        validation = item.get("validation") if isinstance(item.get("validation"), dict) else {}
        for candidate in (generation, validation, item):
            if not isinstance(candidate, dict):
                continue
            score = 0
            if candidate.get("generated_text_hash"):
                score += 8
            if candidate.get("generated_token_count"):
                score += 4
            if candidate.get("decoded_tokens_match"):
                score += 2
            if score > best_score:
                best = candidate
                best_score = score
        for value in item.values():
            if isinstance(value, dict):
                queue.append(value)
            elif isinstance(value, list):
                queue.extend(entry for entry in value if isinstance(entry, dict))
    return best


def safe_generation_summary(payload: dict[str, Any], *, max_new_tokens: int | None = None) -> dict[str, Any]:
    generation = find_generation_summary(payload)
    count = int(generation.get("generated_token_count") or 0)
    limit = int(max_new_tokens or generation.get("max_new_tokens") or 0)
    ready = bool(count and (limit <= 0 or count >= limit))
    validation = payload.get("validation") if isinstance(payload.get("validation"), dict) else {}
    raw_results = validation.get("request_trace") or validation.get("inference_results") or []
    if not raw_results and isinstance(validation.get("inference_result"), dict):
        raw_results = [validation["inference_result"]]
    batch_results = []
    if isinstance(raw_results, list):
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            generated_count = int(item.get("generated_token_count") or count or 0)
            item_limit = int(max_new_tokens or item.get("max_new_tokens") or limit or 0)
            batch_results.append({
                "request_id": item.get("request_id"),
                "prompt_hash": item.get("prompt_hash"),
                "generated_token_count": generated_count,
                "max_new_tokens": item_limit or None,
                "generated_text_hash": item.get("generated_text_hash"),
                "decoded_tokens_match": item.get("decoded_tokens_match", generation.get("decoded_tokens_match")),
                "multi_token_generation_ready": bool(generated_count and (item_limit <= 0 or generated_count >= item_limit)),
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            })
    expected_request_count = max(1, int(validation.get("request_count") or len(batch_results) or 1))
    batch_identity_keys = [
        str(item.get("request_id") or item.get("prompt_hash") or "")
        for item in batch_results[:expected_request_count]
    ]
    batch_identity_ready = bool(
        expected_request_count <= 1
        or (
            len(batch_identity_keys) >= expected_request_count
            and all(batch_identity_keys)
            and len(set(batch_identity_keys)) == expected_request_count
        )
    )
    batch_ready = bool(
        batch_results
        and len(batch_results) >= expected_request_count
        and batch_identity_ready
        and all(bool(item.get("multi_token_generation_ready")) for item in batch_results[:expected_request_count])
    )
    summary = {
        "schema": SESSION_RESULT_SCHEMA,
        "generated_token_count": count,
        "max_new_tokens": limit or None,
        "generated_text_hash": generation.get("generated_text_hash"),
        "decoded_tokens_match": generation.get("decoded_tokens_match"),
        "multi_token_generation_ready": ready,
        "request_count": len(batch_results) or expected_request_count,
        "expected_request_count": expected_request_count,
        "observed_request_count": len(batch_results) or (1 if ready and expected_request_count == 1 else 0),
        "batch_identity_ready": batch_identity_ready,
        "batch_generation_ready": batch_ready if expected_request_count > 1 or batch_results else ready,
        "results": batch_results,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }
    assert_public_safe(summary)
    return summary


def safe_stream_event(
    payload: dict[str, Any],
    *,
    max_new_tokens: int | None = None,
    observed_at: float | None = None,
) -> dict[str, Any]:
    """Build a public-safe per-token progress event from a result ledger row."""

    generation = find_generation_summary(payload)
    validation = payload.get("validation") if isinstance(payload.get("validation"), dict) else {}
    count = int(generation.get("generated_token_count") or 0)
    limit = int(max_new_tokens or generation.get("max_new_tokens") or 0)
    try:
        step = int(generation.get("generation_step"))
    except (TypeError, ValueError):
        step = count - 1 if count > 0 else None
    event = {
        "schema": SESSION_STREAM_EVENT_SCHEMA,
        "task_id": payload.get("task_id"),
        "session_id": payload.get("session_id") or validation.get("session_id"),
        "miner_id": payload.get("miner_id"),
        "stage_id": validation.get("stage_id") if validation else payload.get("stage_id"),
        "request_id": generation.get("request_id") or validation.get("request_id") or payload.get("request_id"),
        "prompt_hash": generation.get("prompt_hash") or validation.get("prompt_hash") or payload.get("prompt_hash"),
        "generated_token_count": count,
        "max_new_tokens": limit or None,
        "generation_step": step,
        "generated_text_hash": generation.get("generated_text_hash"),
        "decoded_tokens_match": generation.get("decoded_tokens_match"),
        "observed_at": float(observed_at) if observed_at is not None else None,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }
    assert_public_safe(event)
    return event


def safe_stream_events(
    payload: dict[str, Any],
    *,
    max_new_tokens: int | None = None,
    observed_at: float | None = None,
) -> list[dict[str, Any]]:
    """Build public-safe progress events, expanding batch result rows per request."""

    validation = payload.get("validation") if isinstance(payload.get("validation"), dict) else {}
    raw_results = validation.get("request_trace") or validation.get("inference_results") or []
    if not raw_results and isinstance(validation.get("inference_result"), dict):
        raw_results = [validation["inference_result"]]
    events: list[dict[str, Any]] = []
    if isinstance(raw_results, list) and len(raw_results) > 1:
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            child = dict(payload)
            child_validation = dict(validation)
            child_validation.update({
                "request_id": item.get("request_id"),
                "prompt_hash": item.get("prompt_hash"),
                "generation_step": item.get("generation_step", validation.get("generation_step")),
                "generated_token_count": item.get("generated_token_count"),
                "max_new_tokens": item.get("max_new_tokens", validation.get("max_new_tokens")),
                "generated_text_hash": item.get("generated_text_hash"),
                "decoded_tokens_match": item.get("decoded_tokens_match", validation.get("decoded_tokens_match")),
            })
            child["validation"] = child_validation
            child["request_id"] = item.get("request_id")
            child["prompt_hash"] = item.get("prompt_hash")
            events.append(safe_stream_event(child, max_new_tokens=max_new_tokens, observed_at=observed_at))
    if not events:
        events = [safe_stream_event(payload, max_new_tokens=max_new_tokens, observed_at=observed_at)]
    return events


def assert_public_safe(value: Any) -> None:
    leaks = public_leak_paths(value)
    if leaks:
        raise ValueError("public payload contains raw or secret fields: " + ", ".join(leaks[:5]))


def public_leak_paths(value: Any, *, path: str = "$") -> list[str]:
    leaks: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in RAW_PUBLIC_KEYS:
                leaks.append(child_path)
            leaks.extend(public_leak_paths(item, path=child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            leaks.extend(public_leak_paths(item, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        for fragment in SECRET_VALUE_FRAGMENTS:
            if fragment in value:
                leaks.append(path)
    return leaks
