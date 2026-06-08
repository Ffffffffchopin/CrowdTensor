#!/usr/bin/env python3
"""Build the CPU-only remote micro-LLM pipeline-sharded inference Beta report."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import micro_llm_sharded_inference_evidence_pack as micro_pack  # noqa: E402
import remote_sharded_inference_beta_pack as base  # noqa: E402


SCHEMA = "remote_micro_llm_sharded_beta_v1"
WORKLOAD_TYPE = "micro_llm_sharded_infer"
FAILURE_NONE = "none"
FAILURE_KILL_STAGE_AFTER_CLAIM = "kill-stage-after-claim"
FAILURE_KILL_STAGE0_AFTER_CLAIM = "kill-stage0-after-claim"
FAILURE_KILL_STAGE1_AFTER_CLAIM = "kill-stage1-after-claim"
SECRET_FRAGMENTS = tuple(sorted(set(base.SECRET_FRAGMENTS + ("hidden_state",))))


def configure_base_module() -> None:
    base.SCHEMA = SCHEMA
    base.WORKLOAD_TYPE = WORKLOAD_TYPE
    base.SECRET_FRAGMENTS = SECRET_FRAGMENTS
    base.sharded_payload_summary = payload_summary
    base.build_local = build_local
    base.build_remote_loopback = build_remote_loopback
    base.build_remote_existing = build_remote_existing
    base.mode_ready_code = mode_ready_code


def payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    session = payload.get("session") if isinstance(payload.get("session"), dict) else {}
    stage = payload.get("stage_summary") if isinstance(payload.get("stage_summary"), dict) else {}
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    assignment = payload.get("stage_assignment") if isinstance(payload.get("stage_assignment"), dict) else {}
    artifact = payload.get("artifact") if isinstance(payload.get("artifact"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "diagnosis_codes": base.diagnosis_codes(payload),
        "session": {
            "schema": session.get("schema"),
            "session_id": session.get("session_id"),
            "stage_count": session.get("stage_count"),
            "stage_0_task_id": session.get("stage_0_task_id"),
            "stage_1_task_id": session.get("stage_1_task_id"),
            "request_count": session.get("request_count"),
            "decode_steps": session.get("decode_steps"),
            "artifact_hash": session.get("artifact_hash"),
            "artifact_id": session.get("artifact_id"),
            "prompt_request_count": session.get("prompt_request_count"),
        },
        "artifact": {
            "schema": artifact.get("schema"),
            "artifact_id": artifact.get("artifact_id"),
            "artifact_hash": artifact.get("artifact_hash"),
            "tokenizer_schema": artifact.get("tokenizer_schema"),
            "loaded": artifact.get("loaded"),
        },
        "stage_summary": {
            "stage_0": stage.get("stage_0") or {},
            "stage_1": stage.get("stage_1") or {},
        },
        "stage_assignment": {
            "mode": assignment.get("mode"),
            "required_distinct_stage_miners": assignment.get("required_distinct_stage_miners"),
            "stage0_miner_id": assignment.get("stage0_miner_id"),
            "stage1_miner_id": assignment.get("stage1_miner_id"),
            "distinct_stage_miners": assignment.get("distinct_stage_miners"),
            "stage_assignment_valid": assignment.get("stage_assignment_valid"),
        },
        "safety": {
            "read_only": safety.get("read_only"),
            "redaction_ok": safety.get("redaction_ok"),
            "raw_activation_redacted": safety.get("raw_activation_redacted"),
            "not_production": safety.get("not_production"),
        },
    }


def build_local(args: argparse.Namespace, *, runner: base.Runner) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    local_dir = output_dir / "local-micro-llm-shard-infer"
    command = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "micro-llm-shard-infer",
        "--output-dir",
        str(local_dir),
        "--port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--decode-steps",
        str(args.decode_steps),
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    if getattr(args, "micro_llm_artifact", ""):
        command.extend(["--micro-llm-artifact", str(args.micro_llm_artifact)])
    if getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", str(args.prompt_texts)])
    step, payload = base.run_json_step(
        "local_micro_llm_sharded_inference",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    return {
        "mode": "local",
        "steps": [step],
        "payload_summaries": {"local_micro_llm_sharded_inference": payload_summary(payload)},
        "artifacts": {
            "local_micro_llm_sharded_cli_summary": base.artifact_entry(
                local_dir / "micro_llm_sharded_cli_summary.json",
                output_dir,
                kind="micro_llm_sharded_cli_summary",
                schema="micro_llm_sharded_cli_v1",
                ok=payload.get("ok") if payload else None,
            ),
            "local_micro_llm_sharded_evidence_json": base.artifact_entry(
                local_dir / "micro_llm_sharded_evidence.json",
                output_dir,
                kind="micro_llm_sharded_evidence",
                schema="micro_llm_sharded_evidence_v1",
            ),
        },
        "diagnosis_codes": base.diagnosis_codes(payload),
    }


def build_remote_loopback(args: argparse.Namespace, *, runner: base.Runner) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    loopback_dir = output_dir / "remote-loopback-micro-llm-shard-infer"
    evidence_json = loopback_dir / "micro_llm_sharded_evidence.json"
    evidence_md = loopback_dir / "micro_llm_sharded_evidence.md"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "micro_llm_sharded_inference_evidence_pack.py"),
        "--port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--decode-steps",
        str(args.decode_steps),
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--miner-prefix",
        "remote-micro-llm-shard-miner",
        "--invite-token-prefix",
        "remote-micro-llm-sharded-token",
        "--json-out",
        str(evidence_json),
        "--markdown-out",
        str(evidence_md),
    ]
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    if getattr(args, "micro_llm_artifact", ""):
        command.extend(["--micro-llm-artifact", str(args.micro_llm_artifact)])
    if getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", str(args.prompt_texts)])
    step, payload = base.run_json_step(
        "remote_loopback_micro_llm_sharded_inference",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    return {
        "mode": "remote-loopback",
        "steps": [step],
        "payload_summaries": {"remote_loopback_micro_llm_sharded_inference": payload_summary(payload)},
        "artifacts": {
            "remote_loopback_micro_llm_sharded_evidence_json": base.artifact_entry(
                evidence_json,
                output_dir,
                kind="micro_llm_sharded_evidence",
                schema="micro_llm_sharded_evidence_v1",
                ok=payload.get("ok") if payload else None,
            ),
            "remote_loopback_micro_llm_sharded_evidence_markdown": base.artifact_entry(
                evidence_md,
                output_dir,
                kind="micro_llm_sharded_evidence_markdown",
            ),
        },
        "diagnosis_codes": base.diagnosis_codes(payload),
    }


def build_remote_existing(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    existing_dir = output_dir / "remote-existing-micro-llm-shard-infer"
    evidence_json = existing_dir / "micro_llm_sharded_evidence.json"
    evidence_md = existing_dir / "micro_llm_sharded_evidence.md"
    step: dict[str, Any] = {
        "name": "remote_existing_micro_llm_sharded_inference",
        "ok": False,
        "returncode": None,
    }
    started = base.time.monotonic()
    try:
        session_payload: dict[str, Any] = {
            "request_count": args.request_count,
            "decode_steps": args.decode_steps,
            "workload_type": WORKLOAD_TYPE,
        }
        prompt_texts = [item for item in str(getattr(args, "prompt_texts", "") or "").split(",") if item]
        if prompt_texts:
            session_payload["prompt_texts"] = prompt_texts
        session = base.request_json(
            "POST",
            args.coordinator_url,
            "/admin/inference-sessions",
            payload=session_payload,
            admin_token=args.admin_token,
            timeout=args.http_timeout,
        )
        session_id = str(session.get("session_id") or "")
        completed, state = base.wait_for_remote_completion(args, session_id)
        rows = base.admin_results_for_session(args, session_id) if session_id else []
        report_args = argparse.Namespace(
            base_url=args.coordinator_url,
            admin_token=args.admin_token,
            observer_token=args.observer_token,
            stage_mode=args.stage_mode,
            require_distinct_stage_miners=args.require_distinct_stage_miners,
        )
        evidence = micro_pack.build_report(
            args=report_args,
            session=session,
            state=state,
            stage_processes=[],
            requeue_summary={
                "enabled": False,
                "failure_mode": FAILURE_NONE,
                "victim_stage_id": None,
                "victim_task_id": "",
                "rescue_miner_id": "",
                "lease_expired": False,
                "rescued_result": False,
                "victim_result_accepted": False,
            },
            ledger_rows=rows,
        )
        evidence_json.parent.mkdir(parents=True, exist_ok=True)
        evidence_json.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        evidence_md.write_text(micro_pack.render_markdown(evidence), encoding="utf-8")
        step["ok"] = bool(completed and evidence.get("ok"))
        step["payload_schema"] = evidence.get("schema")
        step["payload_ok"] = evidence.get("ok")
        if not completed:
            step["error"] = "remote_timeout_waiting_for_stages"
    except Exception as exc:
        evidence = {}
        step["error"] = str(exc)
    step["duration_seconds"] = round(base.time.monotonic() - started, 3)
    return {
        "mode": "remote-existing",
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "steps": [step],
        "payload_summaries": {"remote_existing_micro_llm_sharded_inference": payload_summary(evidence)},
        "artifacts": {
            "remote_existing_micro_llm_sharded_evidence_json": base.artifact_entry(
                evidence_json,
                output_dir,
                kind="micro_llm_sharded_evidence",
                schema="micro_llm_sharded_evidence_v1",
                ok=evidence.get("ok") if evidence else None,
            ),
            "remote_existing_micro_llm_sharded_evidence_markdown": base.artifact_entry(
                evidence_md,
                output_dir,
                kind="micro_llm_sharded_evidence_markdown",
            ),
        },
        "diagnosis_codes": base.diagnosis_codes(evidence),
    }


def mode_ready_code(mode: str) -> str:
    if mode == "remote-loopback":
        return "remote_micro_llm_sharded_loopback_ready"
    if mode == "remote-existing":
        return "remote_micro_llm_sharded_existing_ready"
    return "local_micro_llm_sharded_inference_ready"


configure_base_module()


def build_report(args: argparse.Namespace, *, runner: base.Runner = subprocess.run) -> dict[str, Any]:
    configure_base_module()
    report = base.build_report(args, runner=runner)
    codes = set(report.get("diagnosis_codes") or [])
    if report.get("ok"):
        codes.discard("remote_sharded_inference_ready")
        codes.add("remote_micro_llm_sharded_ready")
    else:
        codes.discard("remote_sharded_inference_failed")
        codes.add("remote_micro_llm_sharded_failed")
    prompt_values = [item for item in str(getattr(args, "prompt_texts", "") or "").split(",") if item]
    prompt_hash = hashlib.sha256(json.dumps(prompt_values, sort_keys=True).encode("utf-8")).hexdigest() if prompt_values else ""
    report.update({
        "schema": SCHEMA,
        "decode_steps": args.decode_steps,
        "stage_mode": args.stage_mode,
        "require_distinct_stage_miners": bool(args.require_distinct_stage_miners),
        "micro_llm_artifact": getattr(args, "micro_llm_artifact", ""),
        "prompt_text_count": len(prompt_values),
        "prompt_texts_hash": f"sha256:{prompt_hash}" if prompt_hash else "",
        "prompt_texts_redacted": True,
        "diagnosis_codes": sorted(codes),
        "safety": {
            **(report.get("safety") or {}),
            "read_only_workload": WORKLOAD_TYPE,
        },
        "limitations": [
            "CPU-only deterministic micro-LLM two-stage pipeline-sharded inference Beta; not production Swarm Inference",
            "Uses fixed tiny Transformer requests and activation hashes; not GGUF/llama.cpp or large-model serving",
            "Does not provide GPU/TPU pooling, P2P routing, NAT traversal, payments, or arbitrary prompt serving",
        ],
        "recommended_next_commands": [
            "crowdtensor micro-llm-shard-infer --json",
            "python3 scripts/remote_micro_llm_sharded_beta_check.py --mode remote-loopback",
        ],
    })
    output_dir = Path(args.output_dir).resolve()
    json_out = Path(args.json_out).resolve() if args.json_out else output_dir / "remote_micro_llm_sharded_beta.json"
    markdown_out = Path(args.markdown_out).resolve() if args.markdown_out else output_dir / "remote_micro_llm_sharded_beta.md"
    support_path = output_dir / "support_bundle.json"
    report["artifacts"]["remote_micro_llm_sharded_beta_json"] = base.artifact_entry(
        json_out,
        output_dir,
        kind="remote_micro_llm_sharded_beta",
        schema=SCHEMA,
        ok=report.get("ok"),
    )
    report["artifacts"]["remote_micro_llm_sharded_beta_markdown"] = base.artifact_entry(
        markdown_out,
        output_dir,
        kind="remote_micro_llm_sharded_beta_markdown",
    )
    return base.finalize_report_artifacts(
        report,
        args,
        output_dir=output_dir,
        json_out=json_out,
        markdown_out=markdown_out,
        support_path=support_path,
        json_artifact_key="remote_micro_llm_sharded_beta_json",
        markdown_artifact_key="remote_micro_llm_sharded_beta_markdown",
        support_schema="remote_micro_llm_sharded_beta_support_bundle_v1",
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CrowdTensor Remote Micro-LLM Sharded Inference Beta",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- failure_mode: `{report.get('failure_mode')}`",
        f"- decode_steps: `{report.get('decode_steps')}`",
        f"- output_dir: `{report.get('output_dir')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Steps",
        "",
    ]
    for step in report.get("steps") or []:
        lines.append(f"- `{step.get('name')}`: `{step.get('ok')}` schema=`{step.get('payload_schema')}`")
    lines.extend(["", "## Boundaries", ""])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    decode_steps = 4
    stage_mode = "both"
    require_distinct_stage_miners = False
    micro_llm_artifact = ""
    prompt_texts = "arn,ten"
    cleaned: list[str] = []
    index = 0
    while index < len(raw):
        item = raw[index]
        if item == "--decode-steps":
            index += 1
            if index >= len(raw):
                raise SystemExit("--decode-steps requires a value")
            decode_steps = int(raw[index])
        elif item.startswith("--decode-steps="):
            decode_steps = int(item.split("=", 1)[1])
        elif item == "--stage-mode":
            index += 1
            if index >= len(raw):
                raise SystemExit("--stage-mode requires a value")
            stage_mode = str(raw[index])
        elif item.startswith("--stage-mode="):
            stage_mode = item.split("=", 1)[1]
        elif item == "--require-distinct-stage-miners":
            require_distinct_stage_miners = True
        elif item == "--micro-llm-artifact":
            index += 1
            if index >= len(raw):
                raise SystemExit("--micro-llm-artifact requires a value")
            micro_llm_artifact = str(raw[index])
        elif item.startswith("--micro-llm-artifact="):
            micro_llm_artifact = item.split("=", 1)[1]
        elif item == "--prompt-texts":
            index += 1
            if index >= len(raw):
                raise SystemExit("--prompt-texts requires a value")
            prompt_texts = str(raw[index])
        elif item.startswith("--prompt-texts="):
            prompt_texts = item.split("=", 1)[1]
        else:
            cleaned.append(item)
        index += 1
    args = base.parse_args(cleaned)
    args.decode_steps = decode_steps
    args.stage_mode = stage_mode
    args.require_distinct_stage_miners = require_distinct_stage_miners or stage_mode == "split"
    args.micro_llm_artifact = micro_llm_artifact
    args.prompt_texts = prompt_texts
    if args.decode_steps < 1 or args.decode_steps > 4:
        raise SystemExit("--decode-steps must be between 1 and 4")
    if args.stage_mode not in {"both", "split"}:
        raise SystemExit("--stage-mode must be both or split")
    return args


def print_human(report: dict[str, Any]) -> None:
    print("CrowdTensor remote micro-LLM sharded inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  failure_mode: {report.get('failure_mode')}")
    print(f"  decode_steps: {report.get('decode_steps')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for step in report.get("steps") or []:
        print(f"  step {step.get('name')}: {step.get('ok')} schema={step.get('payload_schema')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def main() -> None:
    configure_base_module()
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human(report)
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
