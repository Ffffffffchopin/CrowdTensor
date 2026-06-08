#!/usr/bin/env python3
"""Build and operate the Public Swarm Inference Beta evidence pack.

This layer is the public, user-facing wrapper around the current Coordinator
backed inference product surface.  The legacy CPU-only tiny-GPT stage0/stage1
paths remain available, while the product Beta aggregates serve/join/generate,
session_protocol_v1, P2P-lite discovery, CPU fallback, and retained GPU
generation evidence into one stable schema.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import public_swarm_inference_alpha_rc_pack as alpha_rc  # noqa: E402
import support_bundle  # noqa: E402
from crowdtensor.real_llm import DEFAULT_MODEL_ID, DEFAULT_PROMPTS  # noqa: E402
from product_swarm_mvp_check import parse_prompt_texts_arg, read_prompt_texts_file  # noqa: E402


SCHEMA = "public_swarm_inference_beta_v1"
REMOTE_REAL_SCHEMA = "remote_real_llm_sharded_beta_v1"
SWARM_BETA_SCHEMA = "swarm_inference_beta_v1"
PRODUCT_RC_SCHEMA = "public_swarm_product_rc_v1"
CPU_BETA_SCHEMA = "cpu_inference_beta_v1"
WORKLOAD_TYPE = "real_llm_sharded_infer"
DEFAULT_OUTPUT_DIR = "dist/public-swarm-inference-beta"
DEFAULT_ALPHA_RC_REPORT = "dist/public-swarm-inference-alpha-rc/public_swarm_inference_alpha_rc.json"
DEFAULT_GPU_REPORT = (
    "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
    "gpu_sharded_generation_beta_kaggle_auto.json"
)
DEFAULT_STAGE0_REPORT = alpha_rc.DEFAULT_STAGE0_REPORT
DEFAULT_STAGE1_REPORT = alpha_rc.DEFAULT_STAGE1_REPORT
DEFAULT_SUMMARY_REPORT = alpha_rc.DEFAULT_SUMMARY_REPORT
DEFAULT_COORDINATOR_URL = "http://127.0.0.1:9200"
DEFAULT_PUBLIC_HOST = "24.199.118.54"
DEFAULT_PORT = 9200
DEFAULT_BASE_PORT = 9290
PUBLIC_ACTIONS = (
    "prepare",
    "coordinator",
    "miner",
    "verify",
    "collect",
    "clean",
    "local-loopback",
    "evidence-import",
    "product-beta",
)
SWARM_WRAPPED_ACTIONS = {"prepare", "coordinator", "miner", "verify", "collect", "clean"}
SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "Bearer ",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    "activation_result",
    "real_llm_sharded_result",
    "sharded_inference_result",
    "inference_results",
    '"generated_text":',
    '"generated_token_ids":',
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


def redact_text(text: str, secret_values: list[str] | None = None) -> str:
    redacted = str(text)
    for value in secret_values or []:
        if value:
            redacted = redacted.replace(value, "<redacted>")
    for fragment in SECRET_FRAGMENTS:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def redact_values(value: Any, secret_values: list[str] | None = None) -> Any:
    if isinstance(value, str):
        return redact_text(value, secret_values)
    if isinstance(value, list):
        return [redact_values(item, secret_values) for item in value]
    if isinstance(value, dict):
        return {key: redact_values(item, secret_values) for key, item in value.items()}
    return value


def diagnosis_codes(*payloads: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    codes: set[str] = set(extra or [])
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for code in payload.get("diagnosis_codes") or []:
            if isinstance(code, str):
                codes.add(code)
        summary = payload.get("diagnosis_summary")
        if isinstance(summary, dict):
            for code in summary.get("codes") or []:
                if isinstance(code, str):
                    codes.add(code)
        summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
        for item in summaries.values():
            if isinstance(item, dict):
                for code in item.get("diagnosis_codes") or []:
                    if isinstance(code, str):
                        codes.add(code)
    return sorted(codes)


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Public Swarm Inference Beta artifacts summarize product readiness, "
            "operator workflow status, route/discovery evidence, CPU fallback, "
            "retained generation hashes/counts, and diagnostics only. They do "
            "not include answer text."
        ),
    }


def prompt_scope_summary(args: argparse.Namespace) -> dict[str, Any]:
    prompts = prompt_list_from_args(args)
    prompt_texts_file = str(getattr(args, "prompt_texts_file", "") or "")
    has_prompt_source = bool(prompts)
    if prompt_texts_file:
        source = "prompt-texts-file"
    elif str(getattr(args, "prompt_texts", "") or ""):
        source = "prompt-texts"
    elif str(getattr(args, "prompt_text", "") or ""):
        source = "prompt-text"
    else:
        source = "inherited-or-fixed-evidence"
    inline_prompt_text = source in {"prompt-text", "prompt-texts"}
    return {
        "source": source,
        "prompt_count": len(prompts) if has_prompt_source else 0,
        "inline_prompt_text": inline_prompt_text,
        "terminal_next_commands_local_private": inline_prompt_text,
        "terminal_logs_local_private": inline_prompt_text,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": source in {"prompt-text", "prompt-texts", "prompt-texts-file"},
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This Public Swarm Inference Beta report records prompt source/count "
            "and placeholder safety only; raw prompt text is excluded from public "
            "JSON, Markdown, and support artifacts."
        ),
    }


def prompt_list_from_args(args: argparse.Namespace) -> list[str]:
    prompt_list = getattr(args, "prompt_texts_list", None)
    if isinstance(prompt_list, list) and prompt_list:
        return [str(prompt) for prompt in prompt_list]
    prompt_texts_file = str(getattr(args, "prompt_texts_file", "") or "")
    if prompt_texts_file:
        return read_prompt_texts_file(prompt_texts_file)
    prompt_text = str(getattr(args, "prompt_text", "") or "")
    prompt_texts = str(getattr(args, "prompt_texts", "") or "")
    if not prompt_text and not prompt_texts:
        return []
    return parse_prompt_texts_arg(prompt_text, prompt_texts)


def prompt_secret_values(args: argparse.Namespace) -> list[str]:
    return prompt_list_from_args(args)


def answer_scope_summary() -> dict[str, Any]:
    return {
        "scope_state": "no-local-answer",
        "terminal_only": False,
        "visible_in_terminal": False,
        "saved_json_display": "hash-only",
        "saved_markdown_display": "hash-only",
        "json_stdout_display": "hash-only-json",
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This Public Swarm Inference Beta report is shareable operator "
            "evidence, not a local answer transcript; raw prompts, generated "
            "text, token ids, activations, leases, credentials, and private "
            "runtime state are excluded."
        ),
    }


def shareable_summary() -> dict[str, Any]:
    return {
        "saved_artifacts_public_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "answer_scope_state": "no-local-answer",
        "local_answer_terminal_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Share public_swarm_inference_beta*.json/md artifacts; they contain "
            "readiness evidence, route status, hashes, counts, cleanup state, "
            "and diagnostics, not raw prompts or answers."
        ),
    }


def run_json_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    timeout_seconds: float,
    secret_values: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        completed = runner(
            command,
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "ok": False,
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "timeout",
        }, {}
    step: dict[str, Any] = {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    payload: dict[str, Any] = {}
    try:
        payload = json_from_stdout(completed.stdout)
    except ValueError as exc:
        step["ok"] = False
        step["error"] = str(exc)
    else:
        step["payload_schema"] = payload.get("schema")
        step["payload_ok"] = payload.get("ok")
        step["ok"] = bool(step.get("ok") and payload.get("ok"))
    if not step.get("ok"):
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1600:], secret_values)
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1600:], secret_values)
    return step, payload


def base_safety(
    *,
    local_private_artifacts_removed: bool | None = None,
    cpu_only: bool = True,
    p2p_lite_discovery: bool = False,
    gpu_generation_evidence_imported: bool = False,
) -> dict[str, Any]:
    safety = {
        "cpu_only": bool(cpu_only),
        "read_only_workload": WORKLOAD_TYPE,
        "activation_payloads_redacted": True,
        "credential_values_redacted": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "coordinator_backed_task_execution": True,
        "p2p_lite_discovery_only": bool(p2p_lite_discovery),
        "gpu_generation_evidence_imported": bool(gpu_generation_evidence_imported),
        "not_production": True,
        "not_p2p": not bool(p2p_lite_discovery),
        "not_libp2p": True,
        "not_dht": True,
        "not_nat_traversal": True,
        "not_gpu_tpu_pooling": True,
        "not_large_model_serving": True,
        "not_public_prompt_serving": True,
        "not_training": True,
    }
    if local_private_artifacts_removed is not None:
        safety["local_private_artifacts_removed"] = bool(local_private_artifacts_removed)
    return safety


def limitations(*, product_beta: bool = False) -> list[str]:
    if product_beta:
        return [
            "Coordinator-backed Public Swarm Inference Beta product surface; not production Swarm Inference.",
            "P2P-lite is HTTP-gossip route discovery only; not libp2p, DHT, NAT traversal, decentralized security, or a P2P execution layer.",
            "GPU generation is retained tiny GPT evidence import; not a GPU pooling marketplace, Hivemind/Petals-level serving, or large-model serving.",
            "CPU fallback is deterministic read-only local inference evidence, not arbitrary public prompt serving or training.",
        ]
    return [
        "CPU-only read-only tiny GPT stage0/stage1 split inference Beta; not production Swarm Inference.",
        "No P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, payments, or staking.",
        "Fixed operator proof and public evidence boundary; not arbitrary public prompt serving.",
    ]


def report_filenames(mode: str) -> tuple[str, str]:
    if mode in {"local-loopback", "evidence-import", "product-beta"}:
        return "public_swarm_inference_beta.json", "public_swarm_inference_beta.md"
    safe_mode = mode.replace("-", "_")
    return f"public_swarm_inference_beta_{safe_mode}.json", f"public_swarm_inference_beta_{safe_mode}.md"


def render_markdown(report: dict[str, Any]) -> str:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    lines = [
        "# CrowdTensor Public Swarm Inference Beta",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- output_dir: `{report.get('output_dir')}`",
        f"- ready: `{beta.get('ready')}`",
        f"- workload_type: `{WORKLOAD_TYPE}`",
        "",
        "## Output Scope",
        "",
        f"- include output: `{output_request.get('include_output')}`",
        f"- prompt scope: `source={prompt_scope.get('source')} count={prompt_scope.get('prompt_count')} inline_prompt_text={prompt_scope.get('inline_prompt_text')} terminal_next_commands_local_private={prompt_scope.get('terminal_next_commands_local_private')} saved_artifacts_prompt_placeholders={prompt_scope.get('saved_artifacts_prompt_placeholders')} prompt_file_path_public={prompt_scope.get('prompt_file_path_public')} raw_prompt_public={prompt_scope.get('raw_prompt_public')} public_artifact_safe={prompt_scope.get('public_artifact_safe')}`",
        f"- prompt scope note: {prompt_scope.get('summary') or 'Public artifacts record prompt source/count only and exclude raw prompt text.'}",
        f"- answer scope: `{answer_scope.get('scope_state')}`",
        f"- answer scope note: {answer_scope.get('summary') or 'Public artifacts contain no local answer transcript or raw generated text.'}",
        f"- saved JSON display: `{answer_scope.get('saved_json_display')}`",
        f"- saved Markdown display: `{answer_scope.get('saved_markdown_display')}`",
        f"- shareable: `saved_artifacts={shareable.get('saved_artifacts_public_safe')} raw_prompt_public={shareable.get('raw_prompt_public')} raw_generated_text_public={shareable.get('raw_generated_text_public')} generated_token_ids_public={shareable.get('generated_token_ids_public')} answer_scope_state={shareable.get('answer_scope_state')} local_answer_terminal_only={shareable.get('local_answer_terminal_only')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
    ]
    if report.get("steps"):
        lines.extend(["## Steps", ""])
        for step in report.get("steps") or []:
            lines.append(f"- `{step.get('name')}`: `{step.get('ok')}` schema=`{step.get('payload_schema')}`")
        lines.append("")
    if report.get("artifacts"):
        lines.extend(["## Artifacts", ""])
        for name, artifact in sorted((report.get("artifacts") or {}).items()):
            lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
        lines.append("")
    lines.extend(["## Boundaries", ""])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def persist_report(report: dict[str, Any], *, output_dir: Path, mode: str, secret_values: list[str] | None = None) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault("prompt_scope", {
        "source": "inherited-or-fixed-evidence",
        "prompt_count": 0,
        "inline_prompt_text": False,
        "terminal_next_commands_local_private": False,
        "terminal_logs_local_private": False,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": False,
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
        "summary": (
            "Public Swarm Beta artifacts record prompt source/count and placeholder "
            "safety only; raw prompt text is excluded from public JSON, Markdown, and "
            "support bundles."
        ),
    })
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report = support_bundle.sanitize(redact_values(report, secret_values))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "Public Swarm Inference Beta report contained secret-like fragments"
    json_name, markdown_name = report_filenames(mode)
    json_path = output_dir / json_name
    markdown_path = output_dir / markdown_name
    report.setdefault("artifacts", {})
    report["artifacts"]["public_swarm_inference_beta_json"] = artifact_entry(
        json_path,
        output_dir,
        kind="public_swarm_inference_beta",
        schema=SCHEMA,
        ok=report.get("ok"),
    )
    report["artifacts"]["public_swarm_inference_beta_markdown"] = artifact_entry(
        markdown_path,
        output_dir,
        kind="public_swarm_inference_beta_markdown",
    )
    write_json(json_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    report["artifacts"]["public_swarm_inference_beta_json"]["present"] = True
    report["artifacts"]["public_swarm_inference_beta_markdown"]["present"] = True
    write_json(json_path, report)
    return report


def remote_real_llm_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
    inner = next((item for item in summaries.values() if isinstance(item, dict)), {})
    session = inner.get("session") if isinstance(inner.get("session"), dict) else {}
    assignment = inner.get("stage_assignment") if isinstance(inner.get("stage_assignment"), dict) else {}
    artifact = inner.get("artifact") if isinstance(inner.get("artifact"), dict) else {}
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "failure_mode": payload.get("failure_mode"),
        "diagnosis_codes": diagnosis_codes(payload),
        "session": {
            "stage_count": session.get("stage_count"),
            "request_count": session.get("request_count"),
            "model_id": session.get("model_id"),
            "artifact_hash": session.get("artifact_hash"),
        },
        "artifact": {
            "schema": artifact.get("schema"),
            "backend": artifact.get("backend"),
            "model_id": artifact.get("model_id"),
            "artifact_hash": artifact.get("artifact_hash"),
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
            "cpu_only_default": safety.get("cpu_only_default"),
            "read_only_workload": safety.get("read_only_workload"),
            "activation_payloads_redacted": safety.get("activation_payloads_redacted"),
        },
    }


def swarm_child_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "diagnosis_codes": diagnosis_codes(payload),
        "output_dir": payload.get("output_dir"),
        "workload": payload.get("workload") if isinstance(payload.get("workload"), dict) else {},
        "stage_join_pack_count": len(payload.get("stage_join_packs") or []),
        "cleanup_mode": payload.get("cleanup_mode"),
        "candidate_count": payload.get("candidate_count"),
    }


def product_rc_summary(payload: dict[str, Any]) -> dict[str, Any]:
    product_surface = payload.get("product_surface") if isinstance(payload.get("product_surface"), dict) else {}
    session_protocol = payload.get("session_protocol") if isinstance(payload.get("session_protocol"), dict) else {}
    p2p_lite = payload.get("p2p_lite") if isinstance(payload.get("p2p_lite"), dict) else {}
    gpu_generation = payload.get("gpu_generation_import") if isinstance(payload.get("gpu_generation_import"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "diagnosis_codes": diagnosis_codes(payload),
        "product_surface": {
            "serve_ok": (product_surface.get("serve") or {}).get("ok") if isinstance(product_surface.get("serve"), dict) else None,
            "join_stage0_ok": (product_surface.get("join_stage0") or {}).get("ok") if isinstance(product_surface.get("join_stage0"), dict) else None,
            "join_stage1_ok": (product_surface.get("join_stage1") or {}).get("ok") if isinstance(product_surface.get("join_stage1"), dict) else None,
            "generate_ok": (product_surface.get("generate") or {}).get("ok") if isinstance(product_surface.get("generate"), dict) else None,
            "peer_check_ok": (product_surface.get("peer_check") or {}).get("ok") if isinstance(product_surface.get("peer_check"), dict) else None,
        },
        "session_protocol": {
            "ok": session_protocol.get("ok"),
            "schema": session_protocol.get("schema"),
            "route_usable": session_protocol.get("route_usable"),
        },
        "p2p_lite": {
            "ok": p2p_lite.get("ok"),
            "schema": p2p_lite.get("schema"),
            "cpu_route_ok": p2p_lite.get("cpu_route_ok"),
            "cuda_route_ok": p2p_lite.get("cuda_route_ok"),
        },
        "gpu_generation_import": {
            "ok": gpu_generation.get("ok"),
            "schema": gpu_generation.get("schema"),
            "mode": gpu_generation.get("mode"),
            "generated_token_count": gpu_generation.get("generated_token_count"),
            "generated_text_hash": gpu_generation.get("generated_text_hash"),
            "raw_generated_text_public": gpu_generation.get("raw_generated_text_public"),
        },
    }


def cpu_beta_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "workload": payload.get("workload"),
        "diagnosis_codes": diagnosis_codes(payload),
        "step_count": len(payload.get("steps") or []),
        "safety": {
            "cpu_only_default": (payload.get("safety") or {}).get("cpu_only_default")
            if isinstance(payload.get("safety"), dict)
            else None,
            "summary_excludes_raw_inference_payloads": (payload.get("safety") or {}).get("summary_excludes_raw_inference_payloads")
            if isinstance(payload.get("safety"), dict)
            else None,
        },
    }


def build_product_beta(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    product_dir = output_dir / "product-rc"
    cpu_dir = output_dir / "cpu-fallback"
    prompts = prompt_list_from_args(args)
    product_command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_product_rc_pack.py"),
        "--output-dir",
        str(product_dir),
        "--gpu-report",
        args.gpu_report,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--prompt-text",
        prompts[0] if prompts else "CrowdTensor public beta",
        "--timeout-seconds",
        str(int(args.timeout_seconds)),
        "--json",
    ]
    product_step, product_payload = run_json_step(
        "public_swarm_product_rc",
        product_command,
        runner=runner,
        timeout_seconds=float(args.timeout_seconds) + 180.0,
        secret_values=prompt_secret_values(args),
    )
    product_ready_codes = {
        "public_swarm_product_rc_ready",
        "coordinator_product_surface_ready",
        "session_protocol_ready",
        "p2p_lite_discovery_ready",
        "gpu_generation_evidence_import_ready",
    }
    product_codes = set(diagnosis_codes(product_payload))
    product_missing = sorted(product_ready_codes - product_codes)
    product_ready = bool(product_step.get("ok") and product_payload.get("schema") == PRODUCT_RC_SCHEMA and not product_missing)

    cpu_command = [
        sys.executable,
        str(ROOT / "scripts" / "cpu_inference_beta_pack.py"),
        "--mode",
        "local",
        "--output-dir",
        str(cpu_dir),
        "--base-port",
        str(args.base_port),
        "--request-count",
        str(args.cpu_request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--scenario-id",
        args.scenario_id,
        "--timeout-seconds",
        str(int(args.cpu_timeout_seconds)),
        "--json",
    ]
    cpu_step, cpu_payload = run_json_step(
        "cpu_inference_beta_local_fallback",
        cpu_command,
        runner=runner,
        timeout_seconds=float(args.cpu_timeout_seconds) + 90.0,
    )
    cpu_codes = set(diagnosis_codes(cpu_payload))
    cpu_ready = bool(cpu_step.get("ok") and cpu_payload.get("schema") == CPU_BETA_SCHEMA and "cpu_inference_beta_ready" in cpu_codes)

    codes = set(product_codes) | set(cpu_codes)
    if product_ready:
        codes.add("product_rc_ready")
    if cpu_ready:
        codes.update({"cpu_fallback_ready", "local_cpu_inference_ready"})
    ready = bool(product_ready and cpu_ready)
    if ready:
        codes.update({
            "public_swarm_inference_beta_ready",
            "public_swarm_product_beta_ready",
            "coordinator_product_surface_ready",
            "session_protocol_ready",
            "p2p_lite_discovery_ready",
            "gpu_generation_evidence_import_ready",
            "read_only_workload",
            "not_production",
        })
    else:
        codes.add("public_swarm_inference_beta_blocked")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": "product-beta",
        "output_dir": str(output_dir),
        "beta": {
            "ready": ready,
            "product_beta": True,
            "product_rc_ready": product_ready,
            "cpu_fallback_ready": cpu_ready,
            "gpu_generation_evidence_imported": bool(product_ready),
            "p2p_lite_discovery": bool(product_ready),
            "session_protocol": bool(product_ready),
            "workload_type": WORKLOAD_TYPE,
            "model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "cpu_request_count": args.cpu_request_count,
            "external_llm_request_count": args.external_llm_request_count,
            "missing_codes": {
                "product_rc": product_missing,
                "cpu_fallback": [] if cpu_ready else sorted({"cpu_inference_beta_ready"} - cpu_codes),
            },
        },
        "steps": [product_step, cpu_step],
        "payload_summaries": {
            "public_swarm_product_rc": product_rc_summary(product_payload),
            "cpu_inference_beta": cpu_beta_summary(cpu_payload),
        },
        "diagnosis_codes": sorted(codes),
        "prompt_scope": prompt_scope_summary(args),
        "artifacts": {
            "public_swarm_product_rc_json": artifact_entry(
                product_dir / "public_swarm_product_rc.json",
                output_dir,
                kind="public_swarm_product_rc",
                schema=PRODUCT_RC_SCHEMA,
                ok=product_payload.get("ok") if product_payload else None,
            ),
            "gpu_generation_import_json": artifact_entry(
                product_dir / "gpu-generation-import" / "gpu_sharded_generation_beta_evidence_import.json",
                output_dir,
                kind="gpu_sharded_generation_beta_evidence_import",
                schema="gpu_sharded_generation_beta_v1",
            ),
            "cpu_inference_beta_json": artifact_entry(
                cpu_dir / "cpu_inference_beta.json",
                output_dir,
                kind="cpu_inference_beta",
                schema=CPU_BETA_SCHEMA,
                ok=cpu_payload.get("ok") if cpu_payload else None,
            ),
            "cpu_inference_beta_markdown": artifact_entry(
                cpu_dir / "cpu_inference_beta.md",
                output_dir,
                kind="cpu_inference_beta_markdown",
            ),
        },
        "safety": base_safety(
            cpu_only=False,
            p2p_lite_discovery=True,
            gpu_generation_evidence_imported=True,
        ),
        "operator_action": [
            "Run crowdtensor serve to start the Coordinator, crowdtensor peer daemon for HTTP-gossip discovery, two crowdtensor join commands for stage Miners, and crowdtensor generate for bounded session creation.",
            "Use product-beta as the shareable Beta acceptance artifact; use local-loopback or evidence-import when validating legacy CPU-only split paths.",
        ],
        "limitations": limitations(product_beta=True),
    }


def build_local_loopback(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    child_dir = output_dir / "local-loopback-real-llm"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_real_llm_sharded_beta_pack.py"),
        "--mode",
        "remote-loopback",
        "--output-dir",
        str(child_dir),
        "--base-port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--failure-mode",
        "none",
        "--stage-mode",
        "split",
        "--require-distinct-stage-miners",
        "--hf-model-id",
        args.hf_model_id,
        "--timeout-seconds",
        str(int(args.timeout_seconds)),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.prompt_texts_file:
        command.extend(["--prompt-texts-file", args.prompt_texts_file])
    elif args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    step, payload = run_json_step(
        "remote_real_llm_sharded_loopback",
        command,
        runner=runner,
        timeout_seconds=float(args.timeout_seconds) + 120.0,
        secret_values=prompt_secret_values(args),
    )
    codes = set(diagnosis_codes(payload))
    required = {
        "remote_real_llm_sharded_ready",
        "remote_real_llm_sharded_loopback_ready",
        "real_llm_sharded_ready",
        "real_llm_artifact_ready",
        "activation_transport_ready",
        "baseline_match",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
    }
    missing = sorted(required - codes)
    ready = bool(step.get("ok") and payload.get("schema") == REMOTE_REAL_SCHEMA and not missing)
    if ready:
        codes.update({
            "public_swarm_inference_beta_ready",
            "two_stage_split_inference_ready",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "local_loopback_ready",
            "read_only_workload",
            "not_production",
            "not_p2p",
        })
    else:
        codes.add("public_swarm_inference_beta_blocked")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": "local-loopback",
        "output_dir": str(output_dir),
        "beta": {
            "ready": ready,
            "local_loopback": True,
            "evidence_imported": False,
            "workload_type": WORKLOAD_TYPE,
            "model_id": args.hf_model_id,
            "request_count": args.request_count,
            "stage_count": 2,
            "stage_mode": "split",
            "requires_distinct_stage_miners": True,
            "missing_codes": missing,
        },
        "steps": [step],
        "payload_summaries": {"remote_real_llm_sharded_beta": remote_real_llm_summary(payload)},
        "diagnosis_codes": sorted(codes),
        "prompt_scope": prompt_scope_summary(args),
        "artifacts": {
            "remote_real_llm_sharded_beta_json": artifact_entry(
                child_dir / "remote_real_llm_sharded_beta.json",
                output_dir,
                kind="remote_real_llm_sharded_beta",
                schema=REMOTE_REAL_SCHEMA,
                ok=payload.get("ok") if payload else None,
            ),
            "remote_real_llm_sharded_beta_markdown": artifact_entry(
                child_dir / "remote_real_llm_sharded_beta.md",
                output_dir,
                kind="remote_real_llm_sharded_beta_markdown",
            ),
        },
        "safety": base_safety(),
        "operator_action": [
            "Install the optional HF runtime with python -m pip install -e '.[hf]' before running local-loopback.",
            "Use prepare/coordinator/miner/verify/collect for a controlled two-machine operator demo.",
        ],
        "limitations": limitations(),
    }


def validate_alpha_rc_report(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    required = {
        "public_swarm_inference_alpha_rc_ready",
        "public_swarm_alpha_rc_evidence_imported",
        "stage0_live_requeue_evidence_ready",
        "stage1_live_requeue_evidence_ready",
        "public_swarm_live_requeue_evidence_ready",
        "public_swarm_alpha_private_artifacts_absent",
    }
    missing = sorted(required - codes)
    failed: list[str] = []
    if not path.is_file():
        failed.append("alpha_rc_report_missing")
    if payload.get("schema") != alpha_rc.SCHEMA:
        failed.append("schema_mismatch")
    if payload.get("ok") is not True:
        failed.append("report_not_ok")
    if missing:
        failed.append("missing_readiness_codes")
    private_artifacts = alpha_rc.find_private_artifacts(path.parent) if path.parent.exists() else []
    if private_artifacts:
        failed.append("private_artifacts_present")
    return {
        "path": str(path),
        "present": path.is_file(),
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": not failed,
        "missing_codes": missing,
        "failed_checks": sorted(set(failed)),
        "diagnosis_codes": sorted(codes),
        "private_artifacts": private_artifacts,
    }


def build_evidence_import(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    alpha_rc_path = Path(args.alpha_rc_report).resolve()
    stage0_path = Path(args.stage0_report).resolve()
    stage1_path = Path(args.stage1_report).resolve()
    summary_path = Path(args.summary_report).resolve()
    alpha_payload = load_json(alpha_rc_path)
    stage0_payload = load_json(stage0_path)
    stage1_payload = load_json(stage1_path)
    summary_payload = load_json(summary_path)

    rc_summary = validate_alpha_rc_report(alpha_rc_path, alpha_payload)
    stage0 = alpha_rc.validate_stage_report(stage0_path, stage0_payload, stage="stage0")
    stage1 = alpha_rc.validate_stage_report(stage1_path, stage1_payload, stage="stage1")
    summary = alpha_rc.validate_summary_report(summary_path, summary_payload)
    codes = set(diagnosis_codes(alpha_payload, stage0_payload, stage1_payload))
    if rc_summary.get("ready"):
        codes.add("public_swarm_alpha_rc_imported")
    if stage0.get("ready"):
        codes.add("stage0_live_requeue_evidence_ready")
    if stage1.get("ready"):
        codes.add("stage1_live_requeue_evidence_ready")
    if summary.get("ready"):
        codes.add("public_swarm_live_requeue_summary_ready")
    private_clear = not rc_summary.get("private_artifacts") and not stage0.get("private_artifacts") and not stage1.get("private_artifacts")
    ready = bool(rc_summary.get("ready") and stage0.get("ready") and stage1.get("ready") and summary.get("ready") and private_clear)
    if ready:
        codes.update({
            "public_swarm_inference_beta_ready",
            "public_swarm_beta_evidence_import_ready",
            "external_live_evidence_imported",
            "public_swarm_alpha_rc_evidence_imported",
            "two_stage_split_inference_ready",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "read_only_workload",
            "not_production",
            "not_p2p",
        })
    elif args.allow_missing_live_evidence and not alpha_rc_path.is_file():
        codes.add("public_swarm_beta_evidence_import_skipped")
    else:
        codes.add("public_swarm_inference_beta_blocked")
    if args.allow_missing_live_evidence and not alpha_rc_path.is_file():
        ready = False
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": "evidence-import",
        "output_dir": str(output_dir),
        "beta": {
            "ready": ready,
            "local_loopback": False,
            "evidence_imported": ready,
            "alpha_rc_ready": bool(rc_summary.get("ready")),
            "stage0_live_requeue_ready": bool(stage0.get("ready")),
            "stage1_live_requeue_ready": bool(stage1.get("ready")),
            "summary_ready": bool(summary.get("ready")),
            "workload_type": WORKLOAD_TYPE,
            "model_id": "sshleifer/tiny-gpt2",
            "stage_count": 2,
        },
        "imported_reports": {
            "alpha_rc": rc_summary,
            "stage0": stage0,
            "stage1": stage1,
            "summary": summary,
        },
        "diagnosis_codes": sorted(codes),
        "prompt_scope": prompt_scope_summary(argparse.Namespace(prompt_text="", prompt_texts="", prompt_texts_file="", prompt_texts_list=[])),
        "artifacts": {
            "public_swarm_inference_alpha_rc_json": artifact_entry(
                alpha_rc_path,
                output_dir,
                kind="public_swarm_inference_alpha_rc",
                schema=alpha_rc.SCHEMA,
                ok=alpha_payload.get("ok") if alpha_payload else None,
            ),
            "stage0_public_swarm_report": artifact_entry(
                stage0_path,
                output_dir,
                kind="public_swarm_inference_alpha_stage0",
                schema=alpha_rc.ALPHA_SCHEMA,
                ok=stage0_payload.get("ok") if stage0_payload else None,
            ),
            "stage1_public_swarm_report": artifact_entry(
                stage1_path,
                output_dir,
                kind="public_swarm_inference_alpha_stage1",
                schema=alpha_rc.ALPHA_SCHEMA,
                ok=stage1_payload.get("ok") if stage1_payload else None,
            ),
            "live_requeue_summary": artifact_entry(
                summary_path,
                output_dir,
                kind="public_swarm_inference_alpha_live_requeue_summary",
                schema=alpha_rc.SUMMARY_SCHEMA,
                ok=summary_payload.get("ok") if summary_payload else None,
            ),
        },
        "safety": base_safety(local_private_artifacts_removed=private_clear),
        "operator_action": [
            "Use evidence-import to turn retained live stage0/stage1 requeue evidence into a Public Beta artifact.",
            "Use local-loopback when you need a fresh CPU-only tiny GPT split proof on this machine.",
        ],
        "limitations": limitations(),
    }


def swarm_common_command(args: argparse.Namespace, *, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "swarm_inference_beta_pack.py"),
        args.mode,
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    if args.mode != "clean":
        command.extend([
            "--coordinator-url",
            args.coordinator_url,
            "--port",
            str(args.port),
            "--bind-host",
            args.bind_host,
            "--miner-id-prefix",
            args.miner_id_prefix,
            "--request-count",
            str(args.request_count),
            "--hf-model-id",
            args.hf_model_id,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--http-timeout",
            str(args.http_timeout),
        ])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.mode == "prepare":
        command.extend([
            "--lease-seconds",
            str(args.lease_seconds),
            "--compute-seconds",
            str(args.compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
        if args.replace:
            command.append("--replace")
    elif args.mode == "coordinator":
        command.extend(["--lease-seconds", str(args.lease_seconds)])
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
        if args.run:
            command.append("--run")
    elif args.mode == "miner":
        command.extend([
            "--stage",
            args.stage,
            "--compute-seconds",
            str(args.compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
        if args.run:
            command.append("--run")
    elif args.mode == "verify":
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
        if args.prompt_texts_file:
            command.extend(["--prompt-texts-file", args.prompt_texts_file])
        elif args.prompt_texts:
            command.extend(["--prompt-texts", args.prompt_texts])
        if args.real_internet_beta_report:
            command.extend(["--real-internet-beta-report", args.real_internet_beta_report])
    elif args.mode == "collect":
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
        if args.miner_id:
            command.extend(["--miner-id", args.miner_id])
        command.extend(["--artifact-timeout", str(args.artifact_timeout)])
    elif args.mode == "clean":
        if args.apply:
            command.append("--apply")
        if args.include_private:
            command.append("--include-private")
        if args.remove_empty_dir:
            command.append("--remove-empty-dir")
    return command


def build_swarm_wrapped_action(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    command = swarm_common_command(args, output_dir=output_dir)
    secret_values = [args.observer_token, args.admin_token, *prompt_secret_values(args)]
    step, payload = run_json_step(
        f"swarm_inference_beta_{args.mode}",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), 60.0) + 60.0,
        secret_values=secret_values,
    )
    codes = set(diagnosis_codes(payload))
    child_ready = bool(step.get("ok") and payload.get("schema") == SWARM_BETA_SCHEMA)
    if child_ready:
        codes.update({
            "public_swarm_beta_operator_workflow_ready",
            f"public_swarm_beta_{args.mode.replace('-', '_')}_ready",
            "read_only_workload",
            "not_production",
            "not_p2p",
        })
        if args.mode == "prepare":
            codes.update({"two_stage_join_pack_ready", "stage0_join_pack_ready", "stage1_join_pack_ready", "miner_registry_hashed"})
    else:
        codes.add("public_swarm_inference_beta_blocked")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": child_ready,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "beta": {
            "ready": child_ready,
            "operator_workflow": True,
            "workload_type": WORKLOAD_TYPE,
            "model_id": args.hf_model_id,
            "stage_count": 2,
        },
        "steps": [step],
        "payload_summaries": {"swarm_inference_beta": swarm_child_summary(payload)},
        "diagnosis_codes": sorted(codes),
        "prompt_scope": prompt_scope_summary(args),
        "artifacts": {
            "swarm_inference_beta_json": artifact_entry(
                output_dir / f"swarm_inference_beta_{args.mode.replace('-', '_')}.json",
                output_dir,
                kind="swarm_inference_beta",
                schema=SWARM_BETA_SCHEMA,
                ok=payload.get("ok") if payload else None,
            ),
            "swarm_inference_beta_markdown": artifact_entry(
                output_dir / f"swarm_inference_beta_{args.mode.replace('-', '_')}.md",
                output_dir,
                kind="swarm_inference_beta_markdown",
            ),
        },
        "safety": base_safety(),
        "operator_action": [
            "Use this Public Beta wrapper for ordinary prepare/coordinator/miner/verify/collect/clean workflows.",
            "Keep operator.private.env, miner.private.env, and miner_registry.json private.",
        ],
        "limitations": limitations(),
    }


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "product-beta":
        report = build_product_beta(args, output_dir=output_dir, runner=runner)
    elif args.mode == "local-loopback":
        report = build_local_loopback(args, output_dir=output_dir, runner=runner)
    elif args.mode == "evidence-import":
        report = build_evidence_import(args, output_dir=output_dir)
    elif args.mode in SWARM_WRAPPED_ACTIONS:
        report = build_swarm_wrapped_action(args, output_dir=output_dir, runner=runner)
    else:
        raise SystemExit(f"unknown Public Swarm Inference Beta mode: {args.mode}")
    return persist_report(
        report,
        output_dir=output_dir,
        mode=args.mode,
        secret_values=[getattr(args, "observer_token", ""), getattr(args, "admin_token", "")],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or operate Public Swarm Inference Beta evidence.")
    parser.add_argument("action", nargs="?", choices=PUBLIC_ACTIONS)
    parser.add_argument("--mode", choices=PUBLIC_ACTIONS, default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coordinator-url", default=DEFAULT_COORDINATOR_URL)
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--miner-id-prefix", default="public-swarm-beta")
    parser.add_argument("--stage", choices=["stage0", "stage1"], default="")
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--cpu-request-count", type=int, default=2)
    parser.add_argument("--external-llm-request-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--hf-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--prompt-text", default="")
    parser.add_argument("--prompt-texts", default=",".join(DEFAULT_PROMPTS))
    parser.add_argument("--prompt-texts-file", default="", help="newline-delimited bounded batch of up to 4 prompts")
    parser.add_argument("--scenario-id", default="route-baseline")
    parser.add_argument("--observer-token", default="")
    parser.add_argument("--admin-token", default="")
    parser.add_argument("--miner-id", default="")
    parser.add_argument("--real-internet-beta-report", default="")
    parser.add_argument("--gpu-report", default=DEFAULT_GPU_REPORT)
    parser.add_argument("--alpha-rc-report", default=DEFAULT_ALPHA_RC_REPORT)
    parser.add_argument("--stage0-report", default=DEFAULT_STAGE0_REPORT)
    parser.add_argument("--stage1-report", default=DEFAULT_STAGE1_REPORT)
    parser.add_argument("--summary-report", default=DEFAULT_SUMMARY_REPORT)
    parser.add_argument("--allow-missing-live-evidence", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--artifact-timeout", type=float, default=60.0)
    parser.add_argument("--lease-seconds", type=float, default=15.0)
    parser.add_argument("--compute-seconds", type=float, default=0.2)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--max-request-attempts", type=int, default=120)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--include-private", action="store_true")
    parser.add_argument("--remove-empty-dir", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    args.mode = args.mode or args.action or "local-loopback"
    if args.request_count < 1 or args.request_count > 4:
        raise SystemExit("--request-count must be between 1 and 4")
    if args.cpu_request_count < 1 or args.cpu_request_count > 4:
        raise SystemExit("--cpu-request-count must be between 1 and 4")
    if args.external_llm_request_count < 1 or args.external_llm_request_count > 4:
        raise SystemExit("--external-llm-request-count must be between 1 and 4")
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    if args.port < 1 or args.base_port < 1:
        raise SystemExit("--port and --base-port must be positive")
    for name in ["timeout_seconds", "cpu_timeout_seconds", "remote_timeout_seconds", "http_timeout", "artifact_timeout", "lease_seconds", "heartbeat_interval"]:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.compute_seconds < 0:
        raise SystemExit("--compute-seconds must be non-negative")
    if args.max_request_attempts < 1:
        raise SystemExit("--max-request-attempts must be at least 1")
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    prompt_texts_explicit = "--prompt-texts" in raw_argv or any(item.startswith("--prompt-texts=") for item in raw_argv)
    if args.prompt_texts_file and prompt_texts_explicit:
        raise SystemExit("public_swarm_inference_beta accepts either --prompt-texts or --prompt-texts-file, not both")
    try:
        if args.prompt_texts_file:
            args.prompt_texts_list = read_prompt_texts_file(args.prompt_texts_file)
            args.prompt_texts = ""
        else:
            args.prompt_texts_list = parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.mode == "miner" and not args.stage:
        raise SystemExit("miner requires --stage stage0|stage1")
    if args.mode not in PUBLIC_ACTIONS:
        raise SystemExit(f"unknown Public Swarm Inference Beta mode: {args.mode}")
    return args


def print_human(report: dict[str, Any]) -> None:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    print("CrowdTensor Public Swarm Inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {beta.get('ready')}")
    print(f"  output: {report.get('output_dir')}")
    if answer_scope:
        print(f"  answer_scope: {answer_scope.get('scope_state')}")
        print(f"  answer_scope_note: {answer_scope.get('summary') or 'Public artifacts contain no local answer transcript or raw generated text.'}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human(report)
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
