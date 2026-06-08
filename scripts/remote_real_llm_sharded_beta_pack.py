#!/usr/bin/env python3
"""Build the CPU-only remote real tiny-LLM sharded inference Beta report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import real_llm_sharded_inference_evidence_pack as real_pack  # noqa: E402
import remote_sharded_inference_beta_pack as base  # noqa: E402
from crowdtensor.real_llm import BACKEND_CPU as REAL_LLM_BACKEND_CPU  # noqa: E402
from crowdtensor.real_llm import BACKEND_CUDA as REAL_LLM_BACKEND_CUDA  # noqa: E402
from crowdtensor.real_llm import DEFAULT_MODEL_ID  # noqa: E402
from crowdtensor.real_llm import normalize_backend as normalize_real_llm_backend  # noqa: E402
from crowdtensor.real_llm import normalize_partition_mode as normalize_real_llm_partition_mode  # noqa: E402
from product_swarm_mvp_check import parse_prompt_texts_arg, read_prompt_texts_file  # noqa: E402


SCHEMA = "remote_real_llm_sharded_beta_v1"
WORKLOAD_TYPE = "real_llm_sharded_infer"
FAILURE_NONE = "none"
SECRET_FRAGMENTS = tuple(sorted(set(base.SECRET_FRAGMENTS + ("hidden_state", "input_ids"))))


def configure_base_module() -> None:
    base.SCHEMA = SCHEMA
    base.WORKLOAD_TYPE = WORKLOAD_TYPE
    base.SECRET_FRAGMENTS = SECRET_FRAGMENTS
    base.sharded_payload_summary = payload_summary
    base.build_local = build_local
    base.build_remote_loopback = build_remote_loopback
    base.build_remote_existing = build_remote_existing
    base.mode_ready_code = mode_ready_code


def generation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_score = -1
    pending: list[Any] = [payload]
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        if not isinstance(item, dict):
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        generation = item.get("generation")
        if isinstance(generation, dict) and generation:
            score = 0
            if generation.get("generated_text_hash"):
                score += 8
            if generation.get("generated_token_count"):
                score += 4
            if generation.get("multi_token_generation_ready"):
                score += 2
            if generation.get("generated_text_redacted"):
                score += 1
            if score > best_score:
                best = generation
                best_score = score
        for value in item.values():
            if isinstance(value, dict):
                pending.append(value)
            elif isinstance(value, list):
                pending.extend(entry for entry in value if isinstance(entry, dict))
    return best


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
            "model_id": session.get("model_id"),
            "artifact_hash": session.get("artifact_hash"),
            "max_new_tokens": session.get("max_new_tokens"),
            "final_generation_step": session.get("final_generation_step"),
        },
        "generation": generation_summary(payload),
        "artifact": {
            "schema": artifact.get("schema"),
            "model_id": artifact.get("model_id"),
            "backend": artifact.get("backend"),
            "artifact_hash": artifact.get("artifact_hash"),
            "split_index": artifact.get("split_index"),
            "loaded": artifact.get("loaded"),
            "partition_mode": artifact.get("partition_mode"),
            "stage_local_partition_ready": artifact.get("stage_local_partition_ready"),
            "partition_parameter_split_valid": artifact.get("partition_parameter_split_valid"),
            "stage0_parameter_count": artifact.get("stage0_parameter_count"),
            "stage1_parameter_count": artifact.get("stage1_parameter_count"),
            "full_model_parameter_count": artifact.get("full_model_parameter_count"),
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
    local_dir = output_dir / "local-real-llm-shard-infer"
    command = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "real-llm-shard-infer",
        "--output-dir",
        str(local_dir),
        "--port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--hf-model-id",
        args.hf_model_id,
        "--real-llm-backend",
        args.real_llm_backend,
        "--real-llm-partition-mode",
        args.real_llm_partition_mode,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    if getattr(args, "hf_cache_dir", ""):
        command.extend(["--hf-cache-dir", str(args.hf_cache_dir)])
    if getattr(args, "prompt_texts_file", ""):
        command.extend(["--prompt-texts-file", str(args.prompt_texts_file)])
    elif getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", str(args.prompt_texts)])
    step, payload = base.run_json_step(
        "local_real_llm_sharded_inference",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    return {
        "mode": "local",
        "steps": [step],
        "payload_summaries": {"local_real_llm_sharded_inference": payload_summary(payload)},
        "artifacts": {
            "local_real_llm_sharded_cli_summary": base.artifact_entry(
                local_dir / "real_llm_sharded_cli_summary.json",
                output_dir,
                kind="real_llm_sharded_cli_summary",
                schema="real_llm_sharded_cli_v1",
                ok=payload.get("ok") if payload else None,
            ),
            "local_real_llm_sharded_evidence_json": base.artifact_entry(
                local_dir / "real_llm_sharded_evidence.json",
                output_dir,
                kind="real_llm_sharded_evidence",
                schema="real_llm_sharded_evidence_v1",
            ),
        },
        "diagnosis_codes": base.diagnosis_codes(payload),
    }


def build_remote_loopback(args: argparse.Namespace, *, runner: base.Runner) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    loopback_dir = output_dir / "remote-loopback-real-llm-shard-infer"
    evidence_json = loopback_dir / "real_llm_sharded_evidence.json"
    evidence_md = loopback_dir / "real_llm_sharded_evidence.md"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "real_llm_sharded_inference_evidence_pack.py"),
        "--port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--hf-model-id",
        args.hf_model_id,
        "--real-llm-backend",
        args.real_llm_backend,
        "--real-llm-partition-mode",
        args.real_llm_partition_mode,
        "--miner-prefix",
        "remote-real-llm-shard-miner",
        "--invite-token-prefix",
        "remote-real-llm-sharded-token",
        "--json-out",
        str(evidence_json),
        "--markdown-out",
        str(evidence_md),
    ]
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    if getattr(args, "hf_cache_dir", ""):
        command.extend(["--hf-cache-dir", str(args.hf_cache_dir)])
    if getattr(args, "prompt_texts_file", ""):
        command.extend(["--prompt-texts-file", str(args.prompt_texts_file)])
    elif getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", str(args.prompt_texts)])
    step, payload = base.run_json_step(
        "remote_loopback_real_llm_sharded_inference",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    return {
        "mode": "remote-loopback",
        "steps": [step],
        "payload_summaries": {"remote_loopback_real_llm_sharded_inference": payload_summary(payload)},
        "artifacts": {
            "remote_loopback_real_llm_sharded_evidence_json": base.artifact_entry(
                evidence_json,
                output_dir,
                kind="real_llm_sharded_evidence",
                schema="real_llm_sharded_evidence_v1",
                ok=payload.get("ok") if payload else None,
            ),
            "remote_loopback_real_llm_sharded_evidence_markdown": base.artifact_entry(
                evidence_md,
                output_dir,
                kind="real_llm_sharded_evidence_markdown",
            ),
        },
        "diagnosis_codes": base.diagnosis_codes(payload),
    }


def build_remote_existing(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    existing_dir = output_dir / "remote-existing-real-llm-shard-infer"
    evidence_json = existing_dir / "real_llm_sharded_evidence.json"
    evidence_md = existing_dir / "real_llm_sharded_evidence.md"
    step: dict[str, Any] = {
        "name": "remote_existing_real_llm_sharded_inference",
        "ok": False,
        "returncode": None,
    }
    started = base.time.monotonic()
    error_codes: list[str] = []
    try:
        session_payload: dict[str, Any] = {
            "request_count": args.request_count,
            "workload_type": WORKLOAD_TYPE,
            "backend": "cuda" if normalize_real_llm_backend(args.real_llm_backend) == REAL_LLM_BACKEND_CUDA else "cpu",
            "partition_mode": args.real_llm_partition_mode,
            "max_new_tokens": args.max_new_tokens,
        }
        prompt_texts = prompt_list_from_args(args)
        if prompt_texts:
            session_payload["prompt_texts"] = prompt_texts
        session = base.request_json(
            "POST",
            args.coordinator_url,
            "/admin/inference-sessions",
            payload=session_payload,
            admin_token=args.admin_token,
            timeout=max(float(args.http_timeout), 30.0),
        )
    except Exception as exc:
        evidence = {}
        step["error"] = str(exc)
        error_codes.append("session_create_failed")
        if isinstance(exc, HTTPError):
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            if "requires optional Hugging Face dependencies" in detail or "transformers" in detail:
                error_codes.append("hf_dependencies_missing")
                step["error"] = "real_llm_sharded_infer requires optional Hugging Face dependencies; install with python -m pip install -e .[hf]"
            step["http_status"] = exc.code
        elif "requires optional Hugging Face dependencies" in str(exc) or "transformers" in str(exc):
            error_codes.append("hf_dependencies_missing")
            step["error"] = "real_llm_sharded_infer requires optional Hugging Face dependencies; install with python -m pip install -e .[hf]"
    else:
        try:
            session_id = str(session.get("session_id") or "")
            completed, state = base.wait_for_remote_completion(args, session_id)
            rows = base.admin_results_for_session(args, session_id) if session_id else []
            report_args = argparse.Namespace(
                base_url=args.coordinator_url,
                admin_token=args.admin_token,
                observer_token=args.observer_token,
                stage_mode=args.stage_mode,
                require_distinct_stage_miners=args.require_distinct_stage_miners,
                max_new_tokens=args.max_new_tokens,
                real_llm_partition_mode=args.real_llm_partition_mode,
            )
            evidence = real_pack.build_report(
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
            evidence_md.write_text(real_pack.render_markdown(evidence), encoding="utf-8")
            step["ok"] = bool(completed and evidence.get("ok"))
            step["payload_schema"] = evidence.get("schema")
            step["payload_ok"] = evidence.get("ok")
            if not completed:
                step["error"] = "remote_timeout_waiting_for_stages"
                error_codes.append("remote_timeout_waiting_for_stages")
        except Exception as exc:
            evidence = {}
            step["error"] = str(exc)
            error_codes.append("remote_stage_wait_failed")
    step["duration_seconds"] = round(base.time.monotonic() - started, 3)
    return {
        "mode": "remote-existing",
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "steps": [step],
        "payload_summaries": {"remote_existing_real_llm_sharded_inference": payload_summary(evidence)},
        "artifacts": {
            "remote_existing_real_llm_sharded_evidence_json": base.artifact_entry(
                evidence_json,
                output_dir,
                kind="real_llm_sharded_evidence",
                schema="real_llm_sharded_evidence_v1",
                ok=evidence.get("ok") if evidence else None,
            ),
            "remote_existing_real_llm_sharded_evidence_markdown": base.artifact_entry(
                evidence_md,
                output_dir,
                kind="real_llm_sharded_evidence_markdown",
            ),
        },
        "diagnosis_codes": sorted(set(base.diagnosis_codes(evidence) + error_codes)),
    }


def mode_ready_code(mode: str) -> str:
    if mode == "remote-loopback":
        return "remote_real_llm_sharded_loopback_ready"
    if mode == "remote-existing":
        return "remote_real_llm_sharded_existing_ready"
    return "local_real_llm_sharded_inference_ready"


configure_base_module()


def build_report(args: argparse.Namespace, *, runner: base.Runner = subprocess.run) -> dict[str, Any]:
    configure_base_module()
    report = base.build_report(args, runner=runner)
    codes = set(report.get("diagnosis_codes") or [])
    if report.get("ok"):
        codes.discard("remote_sharded_inference_ready")
        codes.add("remote_real_llm_sharded_ready")
        if args.real_llm_backend == REAL_LLM_BACKEND_CUDA:
            codes.update({"gpu_runtime_ready", "cuda_runtime_available", "hf_transformers_cuda_ready"})
    else:
        codes.discard("remote_sharded_inference_failed")
        codes.add("remote_real_llm_sharded_failed")
    report.update({
        "schema": SCHEMA,
        "hf_model_id": args.hf_model_id,
        "real_llm_backend": args.real_llm_backend,
        "real_llm_partition_mode": args.real_llm_partition_mode,
        "stage_mode": args.stage_mode,
        "require_distinct_stage_miners": bool(args.require_distinct_stage_miners),
        "prompt_text_count": len(prompt_list_from_args(args)),
        "max_new_tokens": args.max_new_tokens,
        "diagnosis_codes": sorted(codes),
        "safety": {
            **(report.get("safety") or {}),
            "read_only_workload": WORKLOAD_TYPE,
        },
        "limitations": [
            "Tiny Hugging Face GPT two-stage pipeline-sharded inference Beta; optional CUDA backend is explicit and not production Swarm Inference",
            "Uses a tiny HF model and activation hashes; not GGUF/llama.cpp, GPU pooling marketplace, or large-model serving",
            "Does not provide P2P routing, NAT traversal, payments, or arbitrary public prompt serving",
        ],
        "recommended_next_commands": [
            "crowdtensor real-llm-shard-infer --stage-mode split --json",
            "python3 scripts/remote_real_llm_sharded_beta_check.py --mode remote-loopback",
        ],
    })
    output_dir = Path(args.output_dir).resolve()
    json_out = Path(args.json_out).resolve() if args.json_out else output_dir / "remote_real_llm_sharded_beta.json"
    markdown_out = Path(args.markdown_out).resolve() if args.markdown_out else output_dir / "remote_real_llm_sharded_beta.md"
    support_path = output_dir / "support_bundle.json"
    report["artifacts"]["remote_real_llm_sharded_beta_json"] = base.artifact_entry(
        json_out,
        output_dir,
        kind="remote_real_llm_sharded_beta",
        schema=SCHEMA,
        ok=report.get("ok"),
    )
    report["artifacts"]["remote_real_llm_sharded_beta_markdown"] = base.artifact_entry(
        markdown_out,
        output_dir,
        kind="remote_real_llm_sharded_beta_markdown",
    )
    return base.finalize_report_artifacts(
        report,
        args,
        output_dir=output_dir,
        json_out=json_out,
        markdown_out=markdown_out,
        support_path=support_path,
        json_artifact_key="remote_real_llm_sharded_beta_json",
        markdown_artifact_key="remote_real_llm_sharded_beta_markdown",
        support_schema="remote_real_llm_sharded_beta_support_bundle_v1",
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CrowdTensor Remote Real Tiny-LLM Sharded Inference Beta",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- failure_mode: `{report.get('failure_mode')}`",
        f"- hf_model_id: `{report.get('hf_model_id')}`",
        f"- partition_mode: `{report.get('real_llm_partition_mode')}`",
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


def prompt_list_from_args(args: argparse.Namespace) -> list[str]:
    prompt_list = getattr(args, "prompt_texts_list", None)
    if isinstance(prompt_list, list) and prompt_list:
        return [str(prompt) for prompt in prompt_list]
    prompt_texts_file = str(getattr(args, "prompt_texts_file", "") or "")
    if prompt_texts_file:
        return read_prompt_texts_file(prompt_texts_file)
    return parse_prompt_texts_arg("", str(getattr(args, "prompt_texts", "") or ""))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    hf_model_id = DEFAULT_MODEL_ID
    hf_cache_dir = ""
    real_llm_backend = REAL_LLM_BACKEND_CPU
    real_llm_partition_mode = "full"
    prompt_texts = "CrowdTensor routes home CPU,A miner returns one token"
    prompt_texts_file = ""
    prompt_texts_file_seen = False
    prompt_texts_seen = False
    max_new_tokens = 1
    cleaned: list[str] = []
    index = 0
    while index < len(raw):
        item = raw[index]
        if item == "--hf-model-id":
            index += 1
            if index >= len(raw):
                raise SystemExit("--hf-model-id requires a value")
            hf_model_id = str(raw[index])
        elif item.startswith("--hf-model-id="):
            hf_model_id = item.split("=", 1)[1]
        elif item == "--hf-cache-dir":
            index += 1
            if index >= len(raw):
                raise SystemExit("--hf-cache-dir requires a value")
            hf_cache_dir = str(raw[index])
        elif item.startswith("--hf-cache-dir="):
            hf_cache_dir = item.split("=", 1)[1]
        elif item == "--real-llm-backend":
            index += 1
            if index >= len(raw):
                raise SystemExit("--real-llm-backend requires a value")
            real_llm_backend = str(raw[index])
        elif item.startswith("--real-llm-backend="):
            real_llm_backend = item.split("=", 1)[1]
        elif item == "--real-llm-partition-mode":
            index += 1
            if index >= len(raw):
                raise SystemExit("--real-llm-partition-mode requires a value")
            real_llm_partition_mode = str(raw[index])
        elif item.startswith("--real-llm-partition-mode="):
            real_llm_partition_mode = item.split("=", 1)[1]
        elif item == "--prompt-texts":
            index += 1
            if index >= len(raw):
                raise SystemExit("--prompt-texts requires a value")
            prompt_texts = str(raw[index])
            prompt_texts_seen = True
        elif item.startswith("--prompt-texts="):
            prompt_texts = item.split("=", 1)[1]
            prompt_texts_seen = True
        elif item == "--prompt-texts-file":
            index += 1
            if index >= len(raw):
                raise SystemExit("--prompt-texts-file requires a value")
            prompt_texts_file = str(raw[index])
            prompt_texts_file_seen = True
        elif item.startswith("--prompt-texts-file="):
            prompt_texts_file = item.split("=", 1)[1]
            prompt_texts_file_seen = True
        elif item == "--max-new-tokens":
            index += 1
            if index >= len(raw):
                raise SystemExit("--max-new-tokens requires a value")
            max_new_tokens = int(raw[index])
        elif item.startswith("--max-new-tokens="):
            max_new_tokens = int(item.split("=", 1)[1])
        else:
            cleaned.append(item)
        index += 1
    args = base.parse_args(cleaned)
    args.hf_model_id = hf_model_id
    args.hf_cache_dir = hf_cache_dir
    args.real_llm_backend = normalize_real_llm_backend(real_llm_backend)
    args.real_llm_partition_mode = normalize_real_llm_partition_mode(real_llm_partition_mode)
    if prompt_texts_file_seen and prompt_texts_seen:
        raise SystemExit("remote_real_llm_sharded_beta accepts either --prompt-texts or --prompt-texts-file, not both")
    try:
        if prompt_texts_file_seen:
            args.prompt_texts_file = prompt_texts_file
            args.prompt_texts_list = read_prompt_texts_file(prompt_texts_file)
            args.prompt_texts = ""
        else:
            args.prompt_texts_file = ""
            args.prompt_texts = prompt_texts
            args.prompt_texts_list = parse_prompt_texts_arg("", prompt_texts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.max_new_tokens = max_new_tokens
    if args.request_count > 4:
        raise SystemExit("--request-count must be at most 4 for real LLM sharded inference")
    if args.max_new_tokens < 1 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 1 and 32")
    if args.failure_mode != FAILURE_NONE and args.max_new_tokens != 1:
        raise SystemExit("--max-new-tokens greater than 1 is only supported with --failure-mode none")
    if args.timeout_seconds < 240:
        args.timeout_seconds = 240
    if args.http_timeout < 30:
        args.http_timeout = 30
    return args


def print_human(report: dict[str, Any]) -> None:
    print("CrowdTensor remote real tiny-LLM sharded inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  failure_mode: {report.get('failure_mode')}")
    print(f"  hf_model_id: {report.get('hf_model_id')}")
    print(f"  partition_mode: {report.get('real_llm_partition_mode')}")
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
