#!/usr/bin/env python3
"""Run and package a local CPU-only micro-LLM pipeline-sharded inference proof."""

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
from crowdtensor.micro_llm_artifact import DEFAULT_PROMPTS, inspect_micro_llm_artifact  # noqa: E402
from crowdtensor.protocol import (  # noqa: E402
    MICRO_LLM_SHARDED_BOTH_CAPABILITY,
    MICRO_LLM_SHARDED_STAGE0_CAPABILITY,
    MICRO_LLM_SHARDED_STAGE1_CAPABILITY,
)


SCHEMA = "micro_llm_sharded_evidence_v1"
OBSERVABILITY_SCHEMA = "micro_llm_sharded_observability_v1"
WORKLOAD_TYPE = "micro_llm_sharded_infer"
DEFAULT_ADMIN_TOKEN = "micro-llm-sharded-admin"
DEFAULT_OBSERVER_TOKEN = "micro-llm-sharded-observer"
FAILURE_KILL_STAGE0_AFTER_CLAIM = "kill-stage0-after-claim"
FAILURE_KILL_STAGE1_AFTER_CLAIM = "kill-stage1-after-claim"


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
    advertised = capabilities.get("micro_llm_sharded_stage_capabilities")
    if advertised is None:
        advertised_set: set[str] = set()
    elif isinstance(advertised, str):
        advertised_set = {advertised}
    else:
        advertised_set = {str(item) for item in advertised}
    role = str(capabilities.get("micro_llm_sharded_stage_role") or "both").strip().lower()
    return (
        required_capability in advertised_set
        or MICRO_LLM_SHARDED_BOTH_CAPABILITY in advertised_set
        or role in {stage_role, "both"}
    )


def create_session(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_count": args.request_count,
        "decode_steps": args.decode_steps,
        "workload_type": WORKLOAD_TYPE,
    }
    prompt_texts = [item for item in str(getattr(args, "prompt_texts", "") or "").split(",") if item]
    if prompt_texts:
        payload["prompt_texts"] = prompt_texts
    return base.request_json(
        "POST",
        args.base_url,
        "/admin/inference-sessions",
        payload=payload,
        admin_token=args.admin_token,
    )


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
    stage_tasks = {int((task.get("workload_metadata") or {}).get("stage_id", -1)): task for task in tasks}
    rows = list(ledger_rows) if ledger_rows is not None else (base.admin_results(args, limit=10).get("results") or [])
    stage0 = stage_tasks.get(0, {})
    stage1 = stage_tasks.get(1, {})
    stage0_validation = stage0.get("validation") or {}
    stage1_validation = stage1.get("validation") or {}
    raw_public = base.safe_state_text(state)
    redaction_ok = all(fragment not in raw_public for fragment in [
        "activation_results",
        "activation_result",
        "hidden_state",
        "logits",
        "sharded_inference_result",
        "inference_results",
        "inference_result",
        '"lease_token": "<redacted>"',
    ])
    micro = state.get("model", {}).get("micro_transformer", {})
    artifact_summary = {
        "schema": micro.get("artifact_schema") or "",
        "artifact_id": micro.get("artifact_id") or "",
        "artifact_version": micro.get("artifact_version"),
        "artifact_hash": micro.get("artifact_hash") or stage1_validation.get("artifact_hash") or stage0_validation.get("artifact_hash"),
        "tokenizer_schema": micro.get("tokenizer_schema") or "",
        "loaded": bool(micro.get("artifact_schema") and micro.get("artifact_hash")),
    }
    read_only = (
        micro.get("version") == 0
        and micro.get("optimizer_step") == 0
        and state.get("model", {}).get("global_step") == 0
        and state.get("model_updates") == 0
        and not any(row.get("model_updated") or row.get("micro_transformer_updated") for row in rows)
    )
    stage0_ok = stage0.get("status") == "completed" and stage0_validation.get("code") == "ok"
    stage1_ok = stage1.get("status") == "completed" and stage1_validation.get("code") == "ok"
    baseline_match = bool(stage1_validation.get("baseline_match"))
    decoded_tokens_match = bool(stage1_validation.get("decoded_tokens_match"))
    activation_ready = bool(
        stage0_validation.get("activation_transport_ready")
        and stage1_validation.get("activation_transport_ready")
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
    if activation_ready:
        codes.append("activation_transport_ready")
    if requeue_summary.get("enabled") and requeue_summary.get("rescued_result"):
        codes.append("stage_requeue_ready")
    stage0_miner_id = str(stage0.get("miner_id") or "")
    stage1_miner_id = str(stage1.get("miner_id") or "")
    distinct_stage_miners = bool(stage0_miner_id and stage1_miner_id and stage0_miner_id != stage1_miner_id)
    stage_assignment_valid = bool(
        stage0_miner_id
        and stage1_miner_id
        and stage_capability_advertised(
            stage0,
            required_capability=MICRO_LLM_SHARDED_STAGE0_CAPABILITY,
            stage_role="stage0",
        )
        and stage_capability_advertised(
            stage1,
            required_capability=MICRO_LLM_SHARDED_STAGE1_CAPABILITY,
            stage_role="stage1",
        )
    )
    if distinct_stage_miners:
        codes.append("distinct_stage_miners")
    if stage_assignment_valid:
        codes.append("stage_assignment_valid")
    if artifact_summary["loaded"]:
        codes.append("artifact_loaded")
    if artifact_summary["schema"] == "micro_llm_artifact_v1":
        codes.append("micro_llm_artifact_ready")
    distinct_requirement_met = not getattr(args, "require_distinct_stage_miners", False) or distinct_stage_miners
    if stage0_ok and stage1_ok and baseline_match and decoded_tokens_match and activation_ready and read_only and redaction_ok:
        if distinct_requirement_met:
            codes.append("micro_llm_sharded_ready")
        else:
            codes.append("distinct_stage_miners_missing")
    else:
        if not stage0_ok:
            codes.append("stage_0_missing")
        if not stage1_ok:
            codes.append("stage_1_missing")
        if not baseline_match:
            codes.append("baseline_mismatch")
        if not decoded_tokens_match:
            codes.append("decoded_tokens_mismatch")
        if not activation_ready:
            codes.append("activation_transport_failed")
        if not read_only or not redaction_ok:
            codes.append("safety_failed")
    return {
        "schema": SCHEMA,
        "generated_at": base.utc_now(),
        "ok": "micro_llm_sharded_ready" in codes and (
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
            "decode_steps": session.get("decode_steps"),
            "artifact_hash": session.get("artifact_hash"),
            "artifact_id": session.get("artifact_id"),
            "prompt_request_count": session.get("prompt_request_count"),
        },
        "artifact": artifact_summary,
        "stage_summary": {
            "stage_0": {
                "task_id": stage0.get("task_id"),
                "miner_id": stage0.get("miner_id"),
                "attempt": stage0.get("attempt"),
                "accepted": stage0_ok,
                "activation_count": stage0_validation.get("activation_count"),
                "activation_bytes": stage0_validation.get("activation_bytes"),
                "activation_hashes": stage0_validation.get("activation_hashes"),
                "artifact_hash": stage0_validation.get("artifact_hash"),
                "elapsed_ms": (stage0.get("metrics") or {}).get("elapsed_ms"),
            },
            "stage_1": {
                "task_id": stage1.get("task_id"),
                "miner_id": stage1.get("miner_id"),
                "attempt": stage1.get("attempt"),
                "accepted": stage1_ok,
                "baseline_match": baseline_match,
                "decoded_tokens_match": decoded_tokens_match,
                "request_count": stage1_validation.get("request_count"),
                "decode_steps": stage1_validation.get("decode_steps"),
                "generated_token_count": stage1_validation.get("generated_token_count"),
                "generated_text": stage1_validation.get("generated_text"),
                "request_trace": stage1_validation.get("request_trace"),
                "artifact_hash": stage1_validation.get("artifact_hash"),
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
            "accepted_ledger_rows": len(rows),
            "miner_ids": sorted({str(task.get("miner_id")) for task in tasks if task.get("miner_id")}),
            "processes": stage_processes,
            "requeue_summary": requeue_summary,
        },
        "diagnosis_codes": sorted(set(codes)),
        "safety": {
            "read_only": read_only,
            "redaction_ok": redaction_ok,
            "raw_activation_redacted": "activation_results" not in raw_public and "hidden_state" not in raw_public,
            "not_production": True,
        },
        "limitations": [
            "CPU-only deterministic micro-LLM two-stage pipeline; not production Swarm Inference",
            "Not GPU/TPU pooling, P2P routing, GGUF/llama.cpp serving, or arbitrary prompt serving",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    session = report.get("session") or {}
    stage = report.get("stage_summary") or {}
    return "\n".join([
        "# CrowdTensor Micro-LLM Sharded Inference Evidence",
        "",
        f"Schema: `{report.get('schema')}`",
        f"OK: `{report.get('ok')}`",
        f"Session: `{session.get('session_id')}`",
        f"Workload: `{report.get('workload_type')}`",
        f"Decode steps: `{session.get('decode_steps')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []),
        "",
        "## Stages",
        "",
        f"- Stage 0: task `{(stage.get('stage_0') or {}).get('task_id')}`, miner `{(stage.get('stage_0') or {}).get('miner_id')}`, activations `{(stage.get('stage_0') or {}).get('activation_count')}`",
        f"- Stage 1: task `{(stage.get('stage_1') or {}).get('task_id')}`, miner `{(stage.get('stage_1') or {}).get('miner_id')}`, baseline match `{(stage.get('stage_1') or {}).get('baseline_match')}`, decoded tokens match `{(stage.get('stage_1') or {}).get('decoded_tokens_match')}`",
        "",
        "CPU-only deterministic micro-LLM two-stage pipeline; not production Swarm Inference, GPU pooling, P2P routing, or GGUF/llama.cpp serving.",
        "",
    ])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CPU-only micro-LLM sharded inference evidence.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9860)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--micro-llm-artifact", default="")
    parser.add_argument("--prompt-texts", default=",".join(DEFAULT_PROMPTS))
    parser.add_argument("--failure-mode", choices=sorted(base.FAILURE_MODES), default=base.FAILURE_NONE)
    parser.add_argument("--stage-mode", choices=["both", "split"], default="both")
    parser.add_argument("--require-distinct-stage-miners", action="store_true")
    parser.add_argument("--miner-prefix", default="micro-llm-shard-miner")
    parser.add_argument("--invite-token-prefix", default="micro-llm-sharded-token")
    parser.add_argument("--lease-seconds", type=float, default=5.0)
    parser.add_argument("--compute-seconds", type=float, default=0.0)
    parser.add_argument("--victim-compute-seconds", type=float, default=8.0)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=30.0)
    parser.add_argument("--stage-timeout", type=float, default=10.0)
    parser.add_argument("--claim-observe-timeout", type=float, default=5.0)
    parser.add_argument("--requeue-timeout", type=float, default=10.0)
    parser.add_argument("--admin-token", default=DEFAULT_ADMIN_TOKEN)
    parser.add_argument("--observer-token", default=DEFAULT_OBSERVER_TOKEN)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    args = parser.parse_args(argv)
    if args.request_count < 1 or args.request_count > 8:
        raise SystemExit("--request-count must be between 1 and 8")
    if args.decode_steps < 1 or args.decode_steps > 4:
        raise SystemExit("--decode-steps must be between 1 and 4")
    if args.requeue_timeout <= args.lease_seconds:
        args.requeue_timeout = args.lease_seconds + 5.0
    if args.victim_compute_seconds <= args.lease_seconds:
        args.victim_compute_seconds = args.lease_seconds + 3.0
    args.base_url = f"http://{args.host}:{args.port}"
    if args.micro_llm_artifact:
        inspect_micro_llm_artifact(args.micro_llm_artifact)
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
