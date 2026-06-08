#!/usr/bin/env python3
"""Run and package a local CPU-only real tiny-LLM sharded inference proof."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import sharded_inference_evidence_pack as base  # noqa: E402
from crowdtensor.protocol import (  # noqa: E402
    REAL_LLM_SHARDED_BOTH_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_BOTH_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
    REAL_LLM_SHARDED_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_STAGE1_CAPABILITY,
)
from crowdtensor.real_llm import DEFAULT_MODEL_ID, DEFAULT_PROMPTS, inspect_real_llm_artifact  # noqa: E402
from crowdtensor.real_llm import BACKEND_CPU as REAL_LLM_BACKEND_CPU  # noqa: E402
from crowdtensor.real_llm import BACKEND_CUDA as REAL_LLM_BACKEND_CUDA  # noqa: E402
from crowdtensor.real_llm import normalize_backend as normalize_real_llm_backend  # noqa: E402
from crowdtensor.real_llm import normalize_partition_mode as normalize_real_llm_partition_mode  # noqa: E402
from product_swarm_mvp_check import parse_prompt_texts_arg, read_prompt_texts_file  # noqa: E402


SCHEMA = "real_llm_sharded_evidence_v1"
OBSERVABILITY_SCHEMA = "real_llm_sharded_observability_v1"
WORKLOAD_TYPE = "real_llm_sharded_infer"
DEFAULT_ADMIN_TOKEN = "real-llm-sharded-admin"
DEFAULT_OBSERVER_TOKEN = "real-llm-sharded-observer"


def configure_base_module() -> None:
    base.SCHEMA = SCHEMA
    base.OBSERVABILITY_SCHEMA = OBSERVABILITY_SCHEMA
    base.WORKLOAD_TYPE = WORKLOAD_TYPE
    base.DEFAULT_ADMIN_TOKEN = DEFAULT_ADMIN_TOKEN
    base.DEFAULT_OBSERVER_TOKEN = DEFAULT_OBSERVER_TOKEN
    base.create_session = create_session
    base.build_report = build_report
    base.render_markdown = render_markdown


def stage_capability_advertised(task: dict[str, Any], *, required_capability: str, stage_role: str) -> bool:
    capabilities = task.get("capabilities") if isinstance(task.get("capabilities"), dict) else {}
    advertised = capabilities.get("real_llm_sharded_stage_capabilities")
    if advertised is None:
        advertised_set: set[str] = set()
    elif isinstance(advertised, str):
        advertised_set = {advertised}
    else:
        advertised_set = {str(item) for item in advertised}
    role = str(capabilities.get("real_llm_sharded_stage_role") or "both").strip().lower()
    if required_capability in {REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY, REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY}:
        runtime = capabilities.get("real_llm_runtime") if isinstance(capabilities.get("real_llm_runtime"), dict) else {}
        return (
            str(runtime.get("adapter_kind") or "") == REAL_LLM_BACKEND_CUDA
            and (
                required_capability in advertised_set
                or REAL_LLM_SHARDED_CUDA_BOTH_CAPABILITY in advertised_set
            )
        )
    return (
        required_capability in advertised_set
        or REAL_LLM_SHARDED_BOTH_CAPABILITY in advertised_set
        or role in {stage_role, "both"}
    )


def create_session(args: argparse.Namespace) -> dict[str, Any]:
    resolved_backend = normalize_real_llm_backend(getattr(args, "real_llm_backend", REAL_LLM_BACKEND_CPU))
    payload: dict[str, Any] = {
        "request_count": args.request_count,
        "workload_type": WORKLOAD_TYPE,
        "backend": "cuda" if resolved_backend == REAL_LLM_BACKEND_CUDA else "cpu",
        "partition_mode": normalize_real_llm_partition_mode(getattr(args, "real_llm_partition_mode", "full")),
    }
    prompt_texts = prompt_list_from_args(args)
    if prompt_texts:
        payload["prompt_texts"] = prompt_texts
    return base.request_json(
        "POST",
        args.base_url,
        "/admin/inference-sessions",
        payload=payload,
        admin_token=args.admin_token,
        timeout=max(30.0, float(args.startup_timeout)),
    )


def prompt_list_from_args(args: argparse.Namespace) -> list[str]:
    prompt_list = getattr(args, "prompt_texts_list", None)
    if isinstance(prompt_list, list) and prompt_list:
        return [str(prompt) for prompt in prompt_list]
    prompt_texts_file = str(getattr(args, "prompt_texts_file", "") or "")
    if prompt_texts_file:
        return read_prompt_texts_file(prompt_texts_file)
    return parse_prompt_texts_arg("", str(getattr(args, "prompt_texts", "") or ""))


def redacted_token_summary(token_ids: Any) -> dict[str, Any]:
    if not isinstance(token_ids, list):
        return {"count": 0, "redacted": True}
    return {"count": len(token_ids), "redacted": True}


def redacted_request_trace(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        sanitized.append({
            "request_id": item.get("request_id"),
            "prompt_hash": item.get("prompt_hash"),
            "activation_hash": item.get("activation_hash"),
            "output_hash": item.get("output_hash"),
            "baseline_match": item.get("baseline_match"),
            "generation_step": item.get("generation_step"),
            "max_new_tokens": item.get("max_new_tokens"),
            "generated_token_count": item.get("generated_token_count"),
            "generated_text_hash": item.get("generated_text_hash"),
            "next_token_redacted": "next_token_id" in item or "next_token_text" in item,
        })
    return sanitized


def task_generation_step(task: dict[str, Any]) -> int:
    metadata = task.get("workload_metadata") if isinstance(task.get("workload_metadata"), dict) else {}
    validation = task.get("validation") if isinstance(task.get("validation"), dict) else {}
    for source in (metadata, validation):
        try:
            return int(source.get("generation_step", 0))
        except (TypeError, ValueError):
            continue
    return 0


def latest_stage_task(tasks: list[dict[str, Any]], stage_id: int) -> dict[str, Any]:
    candidates = [
        task
        for task in tasks
        if int((task.get("workload_metadata") or {}).get("stage_id", -1)) == stage_id
    ]
    if not candidates:
        return {}
    return max(candidates, key=task_generation_step)


def build_report(
    *,
    args: argparse.Namespace,
    session: dict[str, Any],
    state: dict[str, Any],
    stage_processes: list[dict[str, Any]],
    requeue_summary: dict[str, Any],
    ledger_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    configure_base_module()
    session_id = str(session.get("session_id") or "")
    tasks = base.session_tasks(state, session_id)
    rows = list(ledger_rows) if ledger_rows is not None else (base.admin_results(args, limit=10).get("results") or [])
    stage0 = latest_stage_task(tasks, 0)
    stage1 = latest_stage_task(tasks, 1)
    stage0_validation = stage0.get("validation") or {}
    stage1_validation = stage1.get("validation") or {}
    raw_public = base.safe_state_text(state)
    forbidden_fragments = [
        "activation_results",
        "activation_result",
        "hidden_state",
        "input_ids",
        "logits",
        "sharded_inference_result",
        "inference_results",
        "inference_result",
        "CrowdTensor routes",
        "A miner returns",
    ]
    redaction_failures = [fragment for fragment in forbidden_fragments if fragment in raw_public]
    redaction_ok = not redaction_failures
    read_only = (
        state.get("model", {}).get("global_step") == 0
        and state.get("model_updates") == 0
        and not any(row.get("model_updated") for row in rows)
    )
    stage0_ok = stage0.get("status") == "completed" and stage0_validation.get("code") == "ok"
    stage1_ok = stage1.get("status") == "completed" and stage1_validation.get("code") == "ok"
    baseline_match = bool(stage1_validation.get("baseline_match"))
    decoded_tokens_match = bool(stage1_validation.get("decoded_tokens_match"))
    activation_ready = bool(
        stage0_validation.get("activation_transport_ready")
        and stage1_validation.get("activation_transport_ready")
    )
    stage0_miner_id = str(stage0.get("miner_id") or "")
    stage1_miner_id = str(stage1.get("miner_id") or "")
    distinct_stage_miners = bool(stage0_miner_id and stage1_miner_id and stage0_miner_id != stage1_miner_id)
    max_new_tokens = max(1, int(getattr(args, "max_new_tokens", session.get("max_new_tokens") or 1)))
    generated_token_ids = stage1_validation.get("generated_token_ids")
    generated_token_count = int(
        stage1_validation.get("generated_token_count")
        or (len(generated_token_ids) if isinstance(generated_token_ids, list) else 0)
    )
    if max_new_tokens == 1 and generated_token_count == 0 and decoded_tokens_match:
        generated_token_count = 1
    generation_step = task_generation_step(stage1)
    completed_generation_steps = len([
        task
        for task in tasks
        if int((task.get("workload_metadata") or {}).get("stage_id", -1)) == 1
        and task.get("status") == "completed"
        and (task.get("validation") or {}).get("code") == "ok"
    ])
    generation_complete = bool(generated_token_count >= max_new_tokens and completed_generation_steps >= max_new_tokens)
    multi_token_generation_ready = bool(max_new_tokens > 1 and generation_complete)
    artifact = {
        "schema": session.get("artifact_schema") or stage0_validation.get("artifact_schema") or stage1_validation.get("artifact_schema"),
        "artifact_hash": session.get("artifact_hash") or stage0_validation.get("artifact_hash") or stage1_validation.get("artifact_hash"),
        "model_id": session.get("model_id") or stage0_validation.get("model_id") or stage1_validation.get("model_id"),
        "backend": session.get("backend") or stage0_validation.get("backend") or stage1_validation.get("backend"),
        "partition_mode": (
            session.get("partition_mode")
            or stage0_validation.get("partition_mode")
            or stage1_validation.get("partition_mode")
            or normalize_real_llm_partition_mode(getattr(args, "real_llm_partition_mode", "full"))
        ),
        "split_index": session.get("split_index") or stage0_validation.get("split_index") or stage1_validation.get("split_index"),
        "num_hidden_layers": session.get("num_hidden_layers"),
        "hidden_size": session.get("hidden_size"),
        "loaded": bool(stage0_validation.get("real_llm_artifact_ready") and stage1_validation.get("real_llm_artifact_ready")),
        "stage0_parameter_count": stage0_validation.get("stage_parameter_count"),
        "stage1_parameter_count": stage1_validation.get("stage_parameter_count"),
        "full_model_parameter_count": (
            stage0_validation.get("full_model_parameter_count")
            or stage1_validation.get("full_model_parameter_count")
        ),
        "partition_parameter_split_valid": bool(
            stage0_validation.get("partition_parameter_split_valid")
            and stage1_validation.get("partition_parameter_split_valid")
        ),
        "stage_local_partition_ready": bool(
            stage0_validation.get("stage_local_partition_ready")
            and stage1_validation.get("stage_local_partition_ready")
        ),
    }
    partition_mode = normalize_real_llm_partition_mode(artifact.get("partition_mode"))
    stage0_partition_loaded = bool(stage0_validation.get("stage0_partition_loaded"))
    stage1_partition_loaded = bool(stage1_validation.get("stage1_partition_loaded"))
    partition_ready = bool(
        partition_mode != "stage_local"
        or (
            artifact["stage_local_partition_ready"]
            and artifact["partition_parameter_split_valid"]
            and stage0_partition_loaded
            and stage1_partition_loaded
        )
    )
    stage_assignment_valid = bool(
        stage0_miner_id
        and stage1_miner_id
        and stage_capability_advertised(
            stage0,
            required_capability=(
                REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY
                if artifact.get("backend") == REAL_LLM_BACKEND_CUDA
                else REAL_LLM_SHARDED_STAGE0_CAPABILITY
            ),
            stage_role="stage0",
        )
        and stage_capability_advertised(
            stage1,
            required_capability=(
                REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY
                if artifact.get("backend") == REAL_LLM_BACKEND_CUDA
                else REAL_LLM_SHARDED_STAGE1_CAPABILITY
            ),
            stage_role="stage1",
        )
    )
    codes = []
    if stage0_ok:
        codes.append("stage_0_accepted")
    if stage1_ok:
        codes.append("stage_1_accepted")
    if baseline_match:
        codes.append("baseline_match")
    if decoded_tokens_match:
        codes.append("decoded_tokens_match")
    if generation_complete:
        codes.append("generation_complete")
    if multi_token_generation_ready:
        codes.append("multi_token_generation_ready")
        codes.append("decoded_text_ready")
    if activation_ready:
        codes.append("activation_transport_ready")
    if artifact["loaded"]:
        codes.append("real_llm_artifact_ready")
    if partition_mode == "stage_local":
        if stage0_partition_loaded:
            codes.append("stage0_partition_loaded")
        if stage1_partition_loaded:
            codes.append("stage1_partition_loaded")
        if artifact["partition_parameter_split_valid"]:
            codes.append("partition_parameter_split_valid")
        if artifact["stage_local_partition_ready"]:
            codes.append("stage_local_partition_ready")
        if stage0_validation.get("stage_gpu_memory_reduced") or stage1_validation.get("stage_gpu_memory_reduced"):
            codes.append("stage_gpu_memory_reduced")
        if stage0_validation.get("stage_cpu_partition_ready") and stage1_validation.get("stage_cpu_partition_ready"):
            codes.append("stage_cpu_partition_ready")
    if artifact.get("backend") == REAL_LLM_BACKEND_CUDA:
        codes.extend(["cuda_runtime_available", "hf_transformers_cuda_ready"])
    if distinct_stage_miners:
        codes.append("distinct_stage_miners")
    if stage_assignment_valid:
        codes.append("stage_assignment_valid")
    if requeue_summary.get("enabled") and requeue_summary.get("rescued_result"):
        codes.append("stage_requeue_ready")
    distinct_requirement_met = not getattr(args, "require_distinct_stage_miners", False) or distinct_stage_miners
    if (
        stage0_ok
        and stage1_ok
        and baseline_match
        and decoded_tokens_match
        and generation_complete
        and activation_ready
        and artifact["loaded"]
        and partition_ready
        and read_only
        and redaction_ok
        and distinct_requirement_met
    ):
        codes.append("real_llm_sharded_ready")
    else:
        if not stage0_ok:
            codes.append("stage_0_missing")
        if not stage1_ok:
            codes.append("stage_1_missing")
        if not baseline_match:
            codes.append("baseline_mismatch")
        if not decoded_tokens_match:
            codes.append("decoded_tokens_mismatch")
        if not generation_complete:
            codes.append("multi_token_generation_incomplete")
        if not activation_ready:
            codes.append("activation_transport_failed")
        if not artifact["loaded"]:
            codes.append("real_llm_artifact_missing")
        if not partition_ready:
            codes.append("stage_local_partition_missing")
        if not distinct_requirement_met:
            codes.append("distinct_stage_miners_missing")
        if not read_only or not redaction_ok:
            codes.append("safety_failed")
    return {
        "schema": SCHEMA,
        "generated_at": base.utc_now(),
        "ok": "real_llm_sharded_ready" in codes and (
            not requeue_summary.get("enabled") or "stage_requeue_ready" in codes
        ),
        "base_url": args.base_url,
        "workload_type": WORKLOAD_TYPE,
        "session": {
            "schema": session.get("schema"),
            "session_id": session_id,
            "stage_count": session.get("stage_count"),
            "stage_0_task_id": stage0.get("task_id") or session.get("stage_0_task_id"),
            "stage_1_task_id": stage1.get("task_id") or session.get("stage_1_task_id"),
            "request_count": session.get("request_count"),
            "artifact_hash": session.get("artifact_hash"),
            "model_id": session.get("model_id"),
            "prompt_request_count": session.get("prompt_request_count"),
            "partition_mode": artifact.get("partition_mode"),
            "max_new_tokens": max_new_tokens,
            "final_generation_step": generation_step,
        },
        "artifact": artifact,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "completed_generation_steps": completed_generation_steps,
            "final_generation_step": generation_step,
            "generated_token_count": generated_token_count,
            "generated_token_summary": redacted_token_summary(stage1_validation.get("generated_token_ids")),
            "generated_text_hash": stage1_validation.get("generated_text_hash"),
            "generated_text_redacted": bool(stage1_validation.get("generated_text") is not None),
            "multi_token_generation_ready": multi_token_generation_ready,
        },
        "stage_summary": {
            "stage_0": {
                "task_id": stage0.get("task_id"),
                "miner_id": stage0.get("miner_id"),
                "attempt": stage0.get("attempt"),
                "accepted": stage0_ok,
                "generation_step": task_generation_step(stage0),
                "activation_count": stage0_validation.get("activation_count"),
                "activation_bytes": stage0_validation.get("activation_bytes"),
                "activation_hashes": stage0_validation.get("activation_hashes"),
                "artifact_hash": stage0_validation.get("artifact_hash"),
                "backend": stage0_validation.get("backend"),
                "partition_mode": stage0_validation.get("partition_mode"),
                "stage_layer_range": stage0_validation.get("stage_layer_range"),
                "stage_parameter_count": stage0_validation.get("stage_parameter_count"),
                "full_model_parameter_count": stage0_validation.get("full_model_parameter_count"),
                "stage_parameter_fraction": stage0_validation.get("stage_parameter_fraction"),
                "partition_parameter_split_valid": stage0_validation.get("partition_parameter_split_valid"),
                "stage_local_partition_ready": stage0_validation.get("stage_local_partition_ready"),
                "stage0_partition_loaded": stage0_partition_loaded,
                "stage_gpu_memory_reduced": stage0_validation.get("stage_gpu_memory_reduced"),
                "stage_cpu_partition_ready": stage0_validation.get("stage_cpu_partition_ready"),
                "elapsed_ms": (stage0.get("metrics") or {}).get("elapsed_ms"),
            },
            "stage_1": {
                "task_id": stage1.get("task_id"),
                "miner_id": stage1.get("miner_id"),
                "attempt": stage1.get("attempt"),
                "accepted": stage1_ok,
                "generation_step": generation_step,
                "baseline_match": baseline_match,
                "decoded_tokens_match": decoded_tokens_match,
                "request_count": stage1_validation.get("request_count"),
                "generated_token_summary": redacted_token_summary(stage1_validation.get("generated_token_ids")),
                "generated_text_redacted": bool(stage1_validation.get("generated_text") is not None),
                "request_trace": redacted_request_trace(stage1_validation.get("request_trace")),
                "artifact_hash": stage1_validation.get("artifact_hash"),
                "backend": stage1_validation.get("backend"),
                "partition_mode": stage1_validation.get("partition_mode"),
                "baseline_device": stage1_validation.get("baseline_device"),
                "stage_layer_range": stage1_validation.get("stage_layer_range"),
                "stage_parameter_count": stage1_validation.get("stage_parameter_count"),
                "full_model_parameter_count": stage1_validation.get("full_model_parameter_count"),
                "stage_parameter_fraction": stage1_validation.get("stage_parameter_fraction"),
                "partition_parameter_split_valid": stage1_validation.get("partition_parameter_split_valid"),
                "stage_local_partition_ready": stage1_validation.get("stage_local_partition_ready"),
                "stage1_partition_loaded": stage1_partition_loaded,
                "stage_gpu_memory_reduced": stage1_validation.get("stage_gpu_memory_reduced"),
                "stage_cpu_partition_ready": stage1_validation.get("stage_cpu_partition_ready"),
                "elapsed_ms": (stage1.get("metrics") or {}).get("elapsed_ms"),
            },
        },
        "stage_assignment": {
            "mode": getattr(args, "stage_mode", "both"),
            "required_distinct_stage_miners": bool(getattr(args, "require_distinct_stage_miners", False)),
            "stage0_miner_id": stage0_miner_id,
            "stage1_miner_id": stage1_miner_id,
            "distinct_stage_miners": distinct_stage_miners,
            "stage_assignment_valid": stage_assignment_valid,
        },
        "observability": {
            "schema": OBSERVABILITY_SCHEMA,
            "stage_count": len(tasks),
            "generation_task_count": len(tasks),
            "accepted_ledger_rows": len(rows),
            "miner_ids": sorted({str(task.get("miner_id")) for task in tasks if task.get("miner_id")}),
            "processes": stage_processes,
            "requeue_summary": requeue_summary,
        },
        "diagnosis_codes": sorted(set(codes)),
        "safety": {
            "read_only": read_only,
            "redaction_ok": redaction_ok,
            "redaction_failures": redaction_failures,
            "raw_activation_redacted": "activation_results" not in raw_public and "hidden_state" not in raw_public,
            "generated_text_redacted": True,
            "generated_token_ids_redacted": True,
            "not_production": True,
            "gpu_backend_selected": artifact.get("backend") == REAL_LLM_BACKEND_CUDA,
            "stage_local_partition": partition_mode == "stage_local",
        },
        "limitations": [
            "Tiny Hugging Face GPT two-stage pipeline; optional CUDA backend is explicit and not production Swarm Inference",
            "Downloads or uses an operator-provided HF cache; not GGUF/llama.cpp, GPU pooling marketplace, or large-model serving",
            "Not P2P routing, NAT traversal, payments, or arbitrary public prompt serving",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    session = report.get("session") or {}
    stage = report.get("stage_summary") or {}
    return "\n".join([
        "# CrowdTensor Real Tiny-LLM Sharded Inference Evidence",
        "",
        f"Schema: `{report.get('schema')}`",
        f"OK: `{report.get('ok')}`",
        f"Session: `{session.get('session_id')}`",
        f"Workload: `{report.get('workload_type')}`",
        f"Model: `{session.get('model_id')}`",
        f"Partition mode: `{session.get('partition_mode')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []),
        "",
        "## Stages",
        "",
        f"- Stage 0: task `{(stage.get('stage_0') or {}).get('task_id')}`, miner `{(stage.get('stage_0') or {}).get('miner_id')}`, activations `{(stage.get('stage_0') or {}).get('activation_count')}`",
        f"- Stage 1: task `{(stage.get('stage_1') or {}).get('task_id')}`, miner `{(stage.get('stage_1') or {}).get('miner_id')}`, baseline match `{(stage.get('stage_1') or {}).get('baseline_match')}`",
        "",
        "CPU-only tiny Hugging Face GPT two-stage pipeline; not production Swarm Inference, GPU pooling, P2P routing, or GGUF/llama.cpp serving.",
        "",
    ])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CPU-only real tiny-LLM sharded inference evidence.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9880)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--prompt-texts", default=",".join(DEFAULT_PROMPTS))
    parser.add_argument("--prompt-texts-file", default="", help="newline-delimited bounded batch of up to 4 prompts")
    parser.add_argument("--hf-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--real-llm-backend",
        choices=["hf_transformers_cpu", "hf_transformers_cuda", "cpu", "cuda", "auto"],
        default=REAL_LLM_BACKEND_CPU,
    )
    parser.add_argument("--real-llm-partition-mode", choices=["full", "stage-local", "stage_local"], default="full")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--failure-mode", choices=sorted(base.FAILURE_MODES), default=base.FAILURE_NONE)
    parser.add_argument("--stage-mode", choices=["both", "split"], default="both")
    parser.add_argument("--require-distinct-stage-miners", action="store_true")
    parser.add_argument("--miner-prefix", default="real-llm-shard-miner")
    parser.add_argument("--invite-token-prefix", default="real-llm-sharded-token")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--compute-seconds", type=float, default=0.0)
    parser.add_argument("--victim-compute-seconds", type=float, default=15.0)
    parser.add_argument("--heartbeat-interval", type=float, default=0.2)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--miner-timeout", type=float, default=120.0)
    parser.add_argument("--stage-timeout", type=float, default=30.0)
    parser.add_argument("--claim-observe-timeout", type=float, default=10.0)
    parser.add_argument("--requeue-timeout", type=float, default=20.0)
    parser.add_argument("--admin-token", default=DEFAULT_ADMIN_TOKEN)
    parser.add_argument("--observer-token", default=DEFAULT_OBSERVER_TOKEN)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    args = parser.parse_args(argv)
    if args.request_count < 1 or args.request_count > 4:
        raise SystemExit("--request-count must be between 1 and 4")
    if args.max_new_tokens < 1 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 1 and 32")
    if args.failure_mode != base.FAILURE_NONE and args.max_new_tokens != 1:
        raise SystemExit("--max-new-tokens greater than 1 is only supported with --failure-mode none")
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    prompt_texts_explicit = "--prompt-texts" in raw_argv or any(item.startswith("--prompt-texts=") for item in raw_argv)
    if args.prompt_texts_file and prompt_texts_explicit:
        raise SystemExit("real_llm_sharded_inference_evidence accepts either --prompt-texts or --prompt-texts-file, not both")
    try:
        if args.prompt_texts_file:
            args.prompt_texts_list = read_prompt_texts_file(args.prompt_texts_file)
            args.prompt_texts = ""
        else:
            args.prompt_texts_list = parse_prompt_texts_arg("", args.prompt_texts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.requeue_timeout <= args.lease_seconds:
        args.requeue_timeout = args.lease_seconds + 5.0
    if args.victim_compute_seconds <= args.lease_seconds:
        args.victim_compute_seconds = args.lease_seconds + 3.0
    args.base_url = f"http://{args.host}:{args.port}"
    args.real_llm_backend = normalize_real_llm_backend(args.real_llm_backend)
    args.real_llm_partition_mode = normalize_real_llm_partition_mode(args.real_llm_partition_mode)
    inspect_real_llm_artifact(
        model_id=args.hf_model_id,
        cache_dir=args.hf_cache_dir,
        backend=args.real_llm_backend,
    )
    return args


configure_base_module()


def run_evidence(args: argparse.Namespace) -> dict[str, Any]:
    configure_base_module()
    if args.stage_mode == "split":
        args.require_distinct_stage_miners = True
    return base.run_evidence(args)


def main() -> None:
    try:
        args = parse_args()
        report = run_evidence(args)
        base.write_json(report, args.json_out)
        if args.markdown_out:
            output = Path(args.markdown_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(render_markdown(report), encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(0 if report.get("ok") else 1)
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
