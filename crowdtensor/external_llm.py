"""Optional external LLM runtime adapter contract for CrowdTensorD.

The contract is intentionally runtime-agnostic: Coordinator leases prompt
requests, while Miner operators decide whether a local mock or external command
is allowed to execute them.
"""

from __future__ import annotations

import hashlib
import shlex
import subprocess
import time


SCHEMA_VERSION = "external_llm_infer_v1"
WORKLOAD_TYPE = "external_llm_infer"
DEFAULT_MAX_TOKENS = 16
MAX_OUTPUT_CHARS = 4096
PROMPTS = [
    "Explain CrowdTensor in one sentence.",
    "Name one safety property of a task lease.",
    "What does a Miner return after local inference?",
    "Why keep runtime adapters optional?",
]


def prompt_hash(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def _stable_offset(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def external_llm_inference_spec_for(
    task_id: str,
    miner_id: str,
    *,
    request_count: int = 1,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    count = max(1, min(int(request_count), len(PROMPTS)))
    token_budget = max(1, int(max_tokens))
    offset = _stable_offset(task_id, miner_id, WORKLOAD_TYPE) % len(PROMPTS)
    requests = []
    for index in range(count):
        prompt = PROMPTS[(offset + index) % len(PROMPTS)]
        requests.append({
            "request_id": f"req-{index + 1}",
            "prompt": prompt,
            "prompt_hash": prompt_hash(prompt),
            "max_tokens": token_budget,
        })
    return {
        "type": WORKLOAD_TYPE,
        "schema_version": SCHEMA_VERSION,
        "adapter_contract": "external_llm_runtime_v1",
        "runtime_policy": "miner_configured",
        "requests": requests,
        "request_count": len(requests),
        "max_tokens": token_budget,
    }


def _bounded_words(text: str, max_tokens: int) -> str:
    words = str(text).replace("\n", " ").split()
    return " ".join(words[:max(1, int(max_tokens))])


def run_mock_external_llm_inference(workload_spec: dict) -> dict:
    return run_external_llm_inference(
        workload_spec,
        adapter_kind="mock",
        model_id="mock-external-llm",
    )


def run_external_llm_command(
    *,
    command: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> str:
    args = shlex.split(command)
    if not args:
        raise ValueError("external LLM runtime command is empty")
    completed = subprocess.run(
        [*args, prompt, str(max_tokens)],
        text=True,
        capture_output=True,
        timeout=float(timeout),
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"external LLM runtime exited {completed.returncode}: {stderr}")
    return completed.stdout.strip()


def run_external_llm_inference(
    workload_spec: dict,
    *,
    adapter_kind: str,
    model_id: str,
    runtime_command: str = "",
    timeout: float = 30.0,
) -> dict:
    spec = dict(workload_spec or {})
    requests = list(spec.get("requests") or [])
    if not requests:
        raise ValueError("external_llm_infer requires at least one request")
    started = time.monotonic()
    results = []
    for request in requests:
        prompt = str(request.get("prompt", ""))
        max_tokens = max(1, int(request.get("max_tokens", spec.get("max_tokens", DEFAULT_MAX_TOKENS))))
        if runtime_command:
            text = run_external_llm_command(
                command=runtime_command,
                prompt=prompt,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        else:
            text = _bounded_words(f"mock completion for: {prompt}", max_tokens)
        text = str(text)[:MAX_OUTPUT_CHARS]
        results.append({
            "schema_version": SCHEMA_VERSION,
            "request_id": str(request.get("request_id")),
            "prompt_hash": str(request.get("prompt_hash") or prompt_hash(prompt)),
            "adapter_kind": adapter_kind,
            "model_id": model_id,
            "output_text": text,
            "output_chars": len(text),
            "max_tokens": max_tokens,
        })
    elapsed_ms = (time.monotonic() - started) * 1000.0
    elapsed_seconds = max(elapsed_ms / 1000.0, 1e-9)
    output_chars = sum(int(row["output_chars"]) for row in results)
    return {
        "schema_version": SCHEMA_VERSION,
        "external_llm_result": results[0],
        "external_llm_results": results,
        "adapter_kind": adapter_kind,
        "model_id": model_id,
        "request_count": len(results),
        "completion_count": len(results),
        "output_chars": output_chars,
        "elapsed_ms": elapsed_ms,
        "requests_per_second": len(results) / elapsed_seconds,
    }


def validate_external_llm_inference(
    external_llm_result: dict | None,
    *,
    external_llm_results: list[dict] | None = None,
    expected_requests: list[dict] | None = None,
) -> dict:
    if external_llm_results is None and external_llm_result is not None:
        external_llm_results = [external_llm_result]
    if not isinstance(external_llm_results, list) or not external_llm_results:
        return {
            "accepted": False,
            "code": "external_llm_inference_missing",
            "reason": "external_llm_infer requires external_llm_result or external_llm_results",
            "external_llm_results": external_llm_results,
        }
    expected = list(expected_requests or [])
    if expected and len(external_llm_results) != len(expected):
        return {
            "accepted": False,
            "code": "external_llm_request_count_mismatch",
            "reason": "external_llm_results length does not match claim-time requests",
            "expected_request_count": len(expected),
            "request_count": len(external_llm_results),
        }

    normalized = []
    for index, row in enumerate(external_llm_results):
        if not isinstance(row, dict):
            return {
                "accepted": False,
                "code": "external_llm_result_invalid",
                "reason": "external_llm_results rows must be objects",
                "request_index": index,
            }
        if str(row.get("schema_version")) != SCHEMA_VERSION:
            return {
                "accepted": False,
                "code": "external_llm_schema_mismatch",
                "reason": "external_llm_result schema_version does not match external_llm_infer_v1",
                "request_index": index,
                "external_llm_result": row,
            }
        expected_row = expected[index] if expected else {}
        request_id = str(row.get("request_id") or "")
        expected_request_id = str(expected_row.get("request_id") or request_id or f"req-{index + 1}")
        if request_id != expected_request_id:
            return {
                "accepted": False,
                "code": "external_llm_request_id_mismatch",
                "reason": "external_llm_result request_id does not match expected request order",
                "request_index": index,
                "external_llm_result": row,
            }
        row_prompt_hash = str(row.get("prompt_hash") or "")
        expected_prompt_hash = str(expected_row.get("prompt_hash") or row_prompt_hash)
        if row_prompt_hash != expected_prompt_hash:
            return {
                "accepted": False,
                "code": "external_llm_prompt_hash_mismatch",
                "reason": "external_llm_result prompt_hash does not match claim-time prompt",
                "request_index": index,
                "external_llm_result": row,
            }
        output_text = str(row.get("output_text") or "")
        if not output_text.strip():
            return {
                "accepted": False,
                "code": "external_llm_empty_output",
                "reason": "external_llm_result output_text must be non-empty",
                "request_index": index,
                "external_llm_result": row,
            }
        if len(output_text) > MAX_OUTPUT_CHARS:
            return {
                "accepted": False,
                "code": "external_llm_output_too_large",
                "reason": "external_llm_result output_text exceeds maximum length",
                "request_index": index,
                "external_llm_result": row,
            }
        normalized.append({
            **row,
            "request_id": request_id,
            "prompt_hash": row_prompt_hash,
            "output_text": output_text,
            "output_chars": len(output_text),
            "adapter_kind": str(row.get("adapter_kind") or "unknown"),
            "model_id": str(row.get("model_id") or "unknown"),
        })

    output_chars = sum(row["output_chars"] for row in normalized)
    first = normalized[0]
    return {
        "accepted": True,
        "code": "ok",
        "reason": "accepted",
        "external_llm_result": first,
        "external_llm_results": normalized,
        "request_count": len(normalized),
        "completion_count": len(normalized),
        "output_chars": output_chars,
        "adapter_kind": first["adapter_kind"],
        "model_id": first["model_id"],
        "output_preview": first["output_text"][:120],
    }


def safe_external_llm_metrics(metrics: dict | None, validation: dict) -> dict:
    source = dict(metrics or {})
    return {
        "request_count": validation.get("request_count", source.get("request_count")),
        "completion_count": validation.get("completion_count", source.get("completion_count")),
        "output_chars": validation.get("output_chars", source.get("output_chars")),
        "elapsed_ms": source.get("elapsed_ms"),
        "requests_per_second": source.get("requests_per_second"),
        "adapter_kind": validation.get("adapter_kind", source.get("adapter_kind")),
        "model_id": validation.get("model_id", source.get("model_id")),
    }
