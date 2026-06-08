#!/usr/bin/env python3
"""Build the Public P2P Swarm Inference v1.0 RC evidence artifact.

This RC turns the v0.6 Coordinator-to-P2P prototype into a product-shaped
public preview: signed peer announcements, a public bootstrap peer, P2P route
selection for serve/join/generate, and Coordinator lease/result fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
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

import p2p_swarm_inference_v06_pack as v06_pack  # noqa: E402
import support_bundle  # noqa: E402
from crowdtensor.session_protocol import public_leak_paths  # noqa: E402


SCHEMA = "public_p2p_swarm_inference_v1_rc_v1"
SUPPORT_SCHEMA = "public_p2p_swarm_inference_v1_rc_support_bundle_v1"
MODE_LOCAL_SMOKE = "local-smoke"
MODE_PACKAGE = "package"
MODE_EVIDENCE_IMPORT = "evidence-import"
MODE_KAGGLE_AUTO = "kaggle-auto"
MODES = [MODE_LOCAL_SMOKE, MODE_PACKAGE, MODE_EVIDENCE_IMPORT, MODE_KAGGLE_AUTO]
DEFAULT_OUTPUT_DIR = "dist/public-p2p-swarm-inference-v1-rc"
DEFAULT_V06_LOCAL_REPORT = "dist/p2p-swarm-inference-v06-local-smoke-refresh2/p2p_swarm_inference_v06.json"
DEFAULT_V06_EXTERNAL_REPORT = "dist/p2p-swarm-inference-v06-kaggle-auto-final/p2p_swarm_inference_v06.json"
DEFAULT_V06_KAGGLE_REPORT = "dist/p2p-swarm-inference-v06-kaggle-auto-final/kaggle-auto/p2p_v06_kaggle_auto.json"
DEFAULT_PUBLIC_HOST = "24.199.118.54"
DEFAULT_P2P_PORT = 9660
DEFAULT_COORDINATOR_PORT = 9661
WORKLOAD_TYPE = "real_llm_sharded_infer"
DEFAULT_HF_MODEL_ID = v06_pack.DEFAULT_HF_MODEL_ID
Runner = Callable[..., subprocess.CompletedProcess[str]]

SECRET_FRAGMENTS = (
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
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in str(stdout or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


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


def effective_peer_secret(args: argparse.Namespace) -> str:
    secret = str(getattr(args, "peer_secret", "") or "")
    if secret:
        return secret
    if not hasattr(args, "_generated_peer_secret"):
        setattr(args, "_generated_peer_secret", f"public-p2p-v1-rc-{secrets.token_hex(16)}")
    return str(getattr(args, "_generated_peer_secret"))


def diagnosis_codes(*payloads: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    codes: set[str] = set(extra or [])
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for code in payload.get("diagnosis_codes") or []:
            if isinstance(code, str):
                codes.add(code)
        summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
        for item in summaries.values():
            if isinstance(item, dict):
                for code in item.get("diagnosis_codes") or []:
                    if isinstance(code, str):
                        codes.add(code)
    return sorted(codes)


def first_string_value(payload: dict[str, Any], key: str) -> str:
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
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
        for nested in item.values():
            if isinstance(nested, dict):
                pending.append(nested)
            elif isinstance(nested, list):
                pending.extend(entry for entry in nested if isinstance(entry, dict))
    return ""


def model_compatibility(payload: dict[str, Any], expected_model_id: str) -> dict[str, Any]:
    observed = first_string_value(payload, "hf_model_id")
    return {
        "expected_hf_model_id": expected_model_id,
        "observed_hf_model_id": observed,
        "model_id_present": bool(observed),
        "model_id_match": bool(observed and observed == expected_model_id),
        "compatible": bool(observed and observed == expected_model_id),
        "default_model_retained_evidence": False,
    }


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
            "Public P2P v1 RC artifacts summarize signed peer discovery, route "
            "selection, Coordinator fallback, safe generation hashes/counts, external "
            "runtime readiness, and stage rescue evidence only. Run `crowdtensor "
            "generate --p2p` in local human mode to display answer text."
        ),
    }


def safe_prompt_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, count)


def normalize_prompt_scope(scope: dict[str, Any]) -> dict[str, Any]:
    source = str(scope.get("source") or "imported-or-built-in-validation-prompts")
    inline_prompt_text = bool(scope.get("inline_prompt_text"))
    return {
        "source": source,
        "prompt_count": safe_prompt_count(scope.get("prompt_count")),
        "inline_prompt_text": inline_prompt_text,
        "terminal_next_commands_local_private": bool(scope.get("terminal_next_commands_local_private")),
        "terminal_logs_local_private": bool(scope.get("terminal_logs_local_private")),
        "saved_artifacts_prompt_placeholders": scope.get("saved_artifacts_prompt_placeholders") is not False,
        "saved_artifacts_public_safe": scope.get("saved_artifacts_public_safe") is not False,
        "prefer_prompt_file_or_stdin_for_shareable_logs": bool(scope.get("prefer_prompt_file_or_stdin_for_shareable_logs")),
        "prompt_file_path_public": bool(scope.get("prompt_file_path_public")),
        "raw_prompt_public": bool(scope.get("raw_prompt_public")),
        "public_artifact_safe": scope.get("public_artifact_safe") is not False,
        "summary": (
            "Public P2P v1 RC inherits prompt source/count metadata from v0.6 "
            "evidence when present; public artifacts keep raw prompt text and "
            "prompt file paths out of JSON, Markdown, and support bundles."
        ),
    }


def inherited_prompt_scope(args: argparse.Namespace, *payloads: dict[str, Any]) -> dict[str, Any]:
    for payload in payloads:
        prompt_scope = payload.get("prompt_scope") if isinstance(payload.get("prompt_scope"), dict) else {}
        if prompt_scope:
            return normalize_prompt_scope(prompt_scope)
    prompt_count = 1 if str(getattr(args, "prompt_text", "") or "").strip() else 0
    return normalize_prompt_scope({
        "source": "imported-or-built-in-validation-prompts",
        "prompt_count": prompt_count,
        "inline_prompt_text": False,
        "terminal_next_commands_local_private": False,
        "terminal_logs_local_private": False,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": False,
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
    })


def prompt_scope_text(prompt_scope: dict[str, Any]) -> str:
    return (
        f"source={prompt_scope.get('source') or 'unknown'} "
        f"count={prompt_scope.get('prompt_count')} "
        f"inline_prompt_text={bool(prompt_scope.get('inline_prompt_text'))} "
        f"terminal_next_commands_local_private={bool(prompt_scope.get('terminal_next_commands_local_private'))} "
        f"saved_artifacts_prompt_placeholders={bool(prompt_scope.get('saved_artifacts_prompt_placeholders'))} "
        f"prompt_file_path_public={bool(prompt_scope.get('prompt_file_path_public'))} "
        f"raw_prompt_public={bool(prompt_scope.get('raw_prompt_public'))} "
        f"public_artifact_safe={bool(prompt_scope.get('public_artifact_safe'))}"
    )


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
            "This Public P2P v1 RC report is shareable route and readiness evidence, "
            "not an answer transcript. Raw prompts, generated text, generated token ids, "
            "activations, lease tokens, peer secrets, private env files, and runtime "
            "state are excluded."
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
            "Share `public_p2p_swarm_inference_v1_rc.json`, "
            "`public_p2p_swarm_inference_v1_rc.md`, and `support_bundle.json`; they "
            "contain signed P2P route evidence, hashes, counts, and readiness summaries, "
            "not raw prompts or answers."
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
    payload = json_from_stdout(completed.stdout)
    if not payload:
        step["ok"] = False
        step["error"] = "command emitted no JSON object"
    if not step["ok"]:
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1000:], secret_values)
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1000:], secret_values)
    return step, redact_values(payload, secret_values) if payload else {}


def summarize_v06(payload: dict[str, Any]) -> dict[str, Any]:
    p2p = payload.get("p2p") if isinstance(payload.get("p2p"), dict) else payload
    payload_summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
    local_p2p = payload_summaries.get("local_p2p_discovery") if isinstance(payload_summaries.get("local_p2p_discovery"), dict) else {}
    codes = set(diagnosis_codes(payload))
    registry = p2p.get("registry") if isinstance(p2p.get("registry"), dict) else {}
    if not registry and isinstance(local_p2p.get("registry"), dict):
        registry = local_p2p["registry"]
    lifecycle = p2p.get("kaggle_lifecycle") if isinstance(p2p.get("kaggle_lifecycle"), dict) else payload.get("kaggle_lifecycle") if isinstance(payload.get("kaggle_lifecycle"), dict) else {}
    generation = p2p.get("generation") if isinstance(p2p.get("generation"), dict) else payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
    if not generation and isinstance(local_p2p.get("real_generate_probe"), dict):
        real_generation = local_p2p["real_generate_probe"].get("generation")
        if isinstance(real_generation, dict):
            generation = real_generation
    return {
        "schema": payload.get("schema"),
        "ok": bool(payload.get("ok")),
        "p2p_ready": bool(p2p.get("ready") or payload.get("ok")),
        "catalog_peer_count": p2p.get("catalog_peer_count") or payload.get("catalog_peer_count") or local_p2p.get("catalog_peer_count"),
        "signed_peer_count": registry.get("signed_peer_count") or p2p.get("signed_peer_count") or local_p2p.get("signed_peer_count") or 0,
        "healthy_peer_count": registry.get("healthy_peer_count") or p2p.get("healthy_peer_count") or local_p2p.get("healthy_peer_count") or 0,
        "stage0_peer_count": p2p.get("stage0_peer_count") or payload.get("stage0_peer_count"),
        "stage1_peer_count": p2p.get("stage1_peer_count") or payload.get("stage1_peer_count"),
        "external_runtime_verified": bool(p2p.get("external_runtime_verified") or payload.get("external_runtime_verified")),
        "external_generate_verified": bool(p2p.get("external_generate_verified") or payload.get("external_generate_verified")),
        "stage_rescue_ready": bool(p2p.get("stage_rescue_ready") or (local_p2p.get("rescue_probe") or {}).get("ok")),
        "real_stage_rescue_ready": bool(p2p.get("real_stage_rescue_ready") or (local_p2p.get("real_stage_rescue_probe") or {}).get("ok")),
        "real_generate_ready": bool(p2p.get("real_generate_ready") or (local_p2p.get("real_generate_probe") or {}).get("ok")),
        "generated_token_count": generation.get("generated_token_count"),
        "decoded_tokens_match": generation.get("decoded_tokens_match"),
        "hf_model_id": p2p.get("hf_model_id") or payload.get("hf_model_id") or first_string_value(local_p2p, "hf_model_id"),
        "observed_hf_model_id": p2p.get("observed_hf_model_id") or p2p.get("hf_model_id") or payload.get("hf_model_id") or first_string_value(local_p2p, "hf_model_id"),
        "model_id_match": p2p.get("model_id_match"),
        "kaggle_kernels_deleted": bool(lifecycle.get("kernels_deleted")),
        "token_rotation_required": bool(lifecycle.get("token_rotation_required")),
        "diagnosis_codes": sorted(codes),
    }


def run_signed_local_smoke(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    peer_secret = effective_peer_secret(args)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "p2p_swarm_inference_v06_pack.py"),
        v06_pack.MODE_LOCAL_SMOKE,
        "--output-dir",
        str(output_dir / "signed-local-v06"),
        "--swarm-id",
        args.swarm_id,
        "--public-host",
        "127.0.0.1",
        "--p2p-port",
        str(args.p2p_port),
        "--coordinator-port",
        str(args.coordinator_port),
        "--backend",
        args.backend,
        "--hf-model-id",
        args.hf_model_id,
        "--prompt-text",
        args.prompt_text,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--startup-timeout",
        str(args.startup_timeout),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--http-timeout",
        str(args.http_timeout),
        "--peer-secret",
        peer_secret,
        "--require-signed",
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return run_json_step(
        "signed_local_v06",
        command,
        runner=runner,
        timeout_seconds=max(args.timeout_seconds, args.startup_timeout, 60.0) + 900.0,
        secret_values=[peer_secret],
    )


def write_runbook(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    path = output_dir / "PUBLIC_P2P_SWARM_INFERENCE_V1_RC.md"
    lines = [
        "# CrowdTensor Public P2P Swarm Inference v1.0 RC",
        "",
        "This runbook uses signed P2P discovery for route selection while the Coordinator remains the lease/result-ledger fallback.",
        "",
        "```bash",
        "export CROWDTENSOR_P2P_PEER_SECRET='replace-with-random-shared-secret'",
        f"crowdtensor p2pd --host 0.0.0.0 --port {args.p2p_port} --peer-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --require-signed --run",
        f"crowdtensor serve --p2p --peer-bootstrap http://{args.public_host}:{args.p2p_port} --peer-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --public-host {args.public_host} --port {args.coordinator_port} --run",
        f"crowdtensor join --p2p --peer-bootstrap http://{args.public_host}:{args.p2p_port} --peer-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --stage stage0 --miner-id public-p2p-stage0 --run",
        f"crowdtensor join --p2p --peer-bootstrap http://{args.public_host}:{args.p2p_port} --peer-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --stage stage1 --miner-id public-p2p-stage1 --run",
        f"crowdtensor generate --p2p --peer-bootstrap http://{args.public_host}:{args.p2p_port} --prompt 'CrowdTensor public P2P v1 RC' --max-new-tokens {args.max_new_tokens} --json",
        "```",
        "",
        "Rotate tokens and the P2P peer secret after every temporary public HTTP or Kaggle proof.",
        "Do not publish raw prompts, generated text, token ids, activations, private env files, runtime state, or inline private Kaggle payloads.",
        "",
        "Boundary: this is a Petals-style public preview shape, not libp2p/DHT/NAT traversal, decentralized security, tokenomics, production Hivemind/Petals parity, or large-model throughput.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifact_entry(path, output_dir, kind="public_p2p_swarm_inference_v1_rc_runbook")


def build_common_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    mode: str,
    signed_local: dict[str, Any],
    external_v06: dict[str, Any],
    kaggle_v06: dict[str, Any],
    steps: list[dict[str, Any]],
    runbook: dict[str, Any],
    signed_local_report_path: str | Path | None = None,
) -> dict[str, Any]:
    local_summary = summarize_v06(signed_local)
    external_summary = summarize_v06(external_v06)
    kaggle_summary = summarize_v06(kaggle_v06)
    local_model = model_compatibility(signed_local, args.hf_model_id)
    external_model = model_compatibility(external_v06, args.hf_model_id)
    kaggle_model = model_compatibility(kaggle_v06, args.hf_model_id)
    local_model_compatible = bool(local_model.get("compatible") or not signed_local)
    external_model_compatible = bool(external_model.get("compatible") or not external_v06)
    kaggle_model_compatible = bool(kaggle_model.get("compatible") or not kaggle_v06)
    local_required = mode in {MODE_LOCAL_SMOKE, MODE_EVIDENCE_IMPORT, MODE_KAGGLE_AUTO}
    external_required = mode in {MODE_EVIDENCE_IMPORT, MODE_KAGGLE_AUTO}
    package_only = mode == MODE_PACKAGE
    local_model_ready = bool((not local_required) or (bool(signed_local) and local_model_compatible))
    external_model_ready = bool(
        (not external_required)
        or (bool(external_v06 or kaggle_v06) and external_model_compatible and kaggle_model_compatible)
    )
    model_metadata_ready = bool(not package_only and local_model_ready and external_model_ready)
    signed_local_ready = bool(
        local_summary.get("ok")
        and int(local_summary.get("signed_peer_count") or 0) >= 3
        and int(local_summary.get("healthy_peer_count") or 0) >= 3
        and local_model_compatible
    )
    external_ready = bool(
        external_summary.get("ok")
        and (external_summary.get("external_runtime_verified") or kaggle_summary.get("external_runtime_verified"))
        and external_model_compatible
        and kaggle_model_compatible
    )
    generation_ready = bool(
        local_summary.get("real_generate_ready")
        or int(kaggle_summary.get("generated_token_count") or 0) >= args.max_new_tokens
    )
    rescue_ready = bool(local_summary.get("real_stage_rescue_ready") or "p2p_real_stage_rescue_ready" in set(local_summary.get("diagnosis_codes") or []))
    package_ready = bool(runbook.get("present"))
    ready = bool(
        not package_only
        and (signed_local_ready or not local_required)
        and (external_ready or not external_required)
        and (generation_ready or not local_required)
        and (rescue_ready or not local_required)
        and model_metadata_ready
        and package_ready
    )
    codes = set(diagnosis_codes(signed_local, external_v06, kaggle_v06))
    if signed_local_ready:
        codes.update({
            "signed_peer_announcement_ready",
            "peer_identity_ready",
            "peer_registry_health_ready",
            "ttl_refresh_ready",
            "local_signed_p2p_discovery_ready",
        })
    if generation_ready:
        codes.add("tiny_gpt2_multi_token_ready")
    if rescue_ready:
        codes.add("p2p_stage_rescue_ready")
    if external_ready:
        codes.update({"external_p2p_runtime_verified", "external_p2p_generate_verified"})
    if model_metadata_ready:
        codes.add("public_p2p_v1_rc_model_metadata_ready")
    else:
        if local_required and not local_model_ready:
            codes.add("public_p2p_v1_rc_local_model_mismatch")
        if external_required and not external_model_compatible:
            codes.add("public_p2p_v1_rc_external_model_mismatch")
        if external_required and not kaggle_model_compatible:
            codes.add("public_p2p_v1_rc_kaggle_model_mismatch")
    if package_ready:
        codes.update({"public_p2p_v1_rc_runbook_ready", "serve_join_generate_p2p_commands_ready"})
    if ready:
        codes.update({
            "public_p2p_swarm_inference_v1_rc_ready",
            "petals_style_public_preview_ready",
            "coordinator_ledger_fallback_ready",
            "not_libp2p",
            "not_dht",
            "not_nat_traversal",
            "not_decentralized_security",
            "not_economic_system",
            "not_large_model_throughput",
            "not_hivemind_petals_production_parity",
        })
    else:
        codes.add("public_p2p_swarm_inference_v1_rc_blocked")

    artifacts = {
        "public_p2p_swarm_inference_v1_rc_json": artifact_entry(output_dir / "public_p2p_swarm_inference_v1_rc.json", output_dir, kind="public_p2p_swarm_inference_v1_rc", schema=SCHEMA, ok=ready),
        "public_p2p_swarm_inference_v1_rc_markdown": artifact_entry(output_dir / "public_p2p_swarm_inference_v1_rc.md", output_dir, kind="public_p2p_swarm_inference_v1_rc_markdown"),
        "support_bundle_json": artifact_entry(output_dir / "support_bundle.json", output_dir, kind="public_p2p_swarm_inference_v1_rc_support_bundle", schema=SUPPORT_SCHEMA, ok=ready),
        "runbook": runbook,
    }
    if signed_local:
        signed_local_path = Path(signed_local_report_path) if signed_local_report_path else output_dir / "signed-local-v06" / "p2p_swarm_inference_v06.json"
        artifacts["signed_local_v06_json"] = artifact_entry(signed_local_path, output_dir, kind="p2p_swarm_inference_v06_signed_local", schema=v06_pack.SCHEMA, ok=signed_local.get("ok"))
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": mode,
        "output_dir": str(output_dir),
        "rc": {
            "ready": ready,
            "package_only": package_only,
            "signed_local_ready": signed_local_ready,
            "external_runtime_ready": external_ready,
            "generation_ready": generation_ready,
            "stage_rescue_ready": rescue_ready,
            "runbook_ready": package_ready,
            "model_metadata_ready": model_metadata_ready,
        },
        "p2p": {
            "swarm_id": args.swarm_id,
            "hf_model_id": args.hf_model_id,
            "local_model": local_model,
            "external_model": external_model,
            "kaggle_model": kaggle_model,
            "signed_announcement_required": True,
            "peer_identity": "shared-secret-hmac",
            "signed_peer_count": local_summary.get("signed_peer_count"),
            "healthy_peer_count": local_summary.get("healthy_peer_count"),
            "catalog_peer_count": local_summary.get("catalog_peer_count"),
            "stage0_peer_count": local_summary.get("stage0_peer_count"),
            "stage1_peer_count": local_summary.get("stage1_peer_count"),
            "external_runtime_verified": external_ready,
            "external_generate_verified": bool(kaggle_summary.get("external_generate_verified") or external_summary.get("external_generate_verified")),
            "coordinator_ledger_fallback": True,
        },
        "inference": {
            "workload_type": WORKLOAD_TYPE,
            "backend": args.backend,
            "model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "tiny_gpt2_multi_token_ready": generation_ready,
            "optional_distilgpt2_or_larger_attempt": "distilgpt2_attempt_ready" in codes or "optional_distilgpt2_or_gpt2_strict_ready" in codes,
        },
        "steps": steps,
        "payload_summaries": {
            "signed_local_v06": local_summary,
            "external_v06": external_summary,
            "kaggle_v06": kaggle_summary,
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": artifacts,
        "prompt_scope": inherited_prompt_scope(args, signed_local, external_v06, kaggle_v06),
        "safety": {
            "signed_peer_announcement": True,
            "peer_secret_gossiped": False,
            "tokens_gossiped": False,
            "raw_prompts_gossiped": False,
            "activations_gossiped": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "coordinator_result_fallback": True,
            "token_rotation_required": bool(kaggle_summary.get("token_rotation_required")),
            "not_production": True,
            "not_libp2p": True,
            "not_dht": True,
            "not_nat_traversal": True,
            "not_decentralized_security": True,
            "not_economic_system": True,
            "not_large_model_throughput": True,
        },
        "completed": [
            "Signed P2P discovery/routing RC surface",
            "Product serve/join/generate commands for P2P route selection",
            "Coordinator lease/result-ledger fallback",
        ],
        "not_completed": [
            "Production NAT traversal",
            "libp2p DHT routing",
            "Decentralized identity and Sybil resistance",
            "Economic system",
            "Hivemind/Petals production parity",
            "Large-model throughput",
        ],
        "operator_action": [
            "Use PUBLIC_P2P_SWARM_INFERENCE_V1_RC.md for a public-preview rehearsal.",
            "Run a fresh kaggle-auto proof before making new external runtime claims.",
            "Rotate tokens and CROWDTENSOR_P2P_PEER_SECRET after temporary public proofs.",
        ],
    }
    return support_bundle.sanitize(redact_values(report, [effective_peer_secret(args)]))


def render_markdown(report: dict[str, Any]) -> str:
    rc = report.get("rc") if isinstance(report.get("rc"), dict) else {}
    lines = [
        "# CrowdTensor Public P2P Swarm Inference v1.0 RC",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- signed local ready: `{rc.get('signed_local_ready')}`",
        f"- external runtime ready: `{rc.get('external_runtime_ready')}`",
        f"- generation ready: `{rc.get('generation_ready')}`",
        f"- stage rescue ready: `{rc.get('stage_rescue_ready')}`",
        "",
        "## Output Scope",
        "",
        f"- output request: `include_output={bool((report.get('output_request') or {}).get('include_output'))} raw_prompt_public={bool((report.get('output_request') or {}).get('raw_prompt_public'))} raw_generated_text_public={bool((report.get('output_request') or {}).get('raw_generated_text_public'))} generated_token_ids_public={bool((report.get('output_request') or {}).get('generated_token_ids_public'))} public_artifact_safe={bool((report.get('output_request') or {}).get('public_artifact_safe'))}`",
        f"- prompt scope: `{prompt_scope_text((report.get('prompt_scope') or {}) if isinstance(report.get('prompt_scope'), dict) else {})}`",
        f"- answer scope: `state={(report.get('answer_scope') or {}).get('scope_state')} terminal_only={bool((report.get('answer_scope') or {}).get('terminal_only'))} visible_in_terminal={bool((report.get('answer_scope') or {}).get('visible_in_terminal'))} saved_json={(report.get('answer_scope') or {}).get('saved_json_display')} saved_markdown={(report.get('answer_scope') or {}).get('saved_markdown_display')} public_artifact_safe={bool((report.get('answer_scope') or {}).get('public_artifact_safe'))}`",
        f"- shareable: `saved_artifacts={bool((report.get('shareable_summary') or {}).get('saved_artifacts_public_safe'))} raw_prompt_public={bool((report.get('shareable_summary') or {}).get('raw_prompt_public'))} raw_generated_text_public={bool((report.get('shareable_summary') or {}).get('raw_generated_text_public'))} generated_token_ids_public={bool((report.get('shareable_summary') or {}).get('generated_token_ids_public'))} answer_scope_state={(report.get('shareable_summary') or {}).get('answer_scope_state')} local_answer_terminal_only={bool((report.get('shareable_summary') or {}).get('local_answer_terminal_only'))}`",
        f"- note: {(report.get('answer_scope') or {}).get('summary')}",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Boundaries",
        "",
    ]
    lines.extend(f"- {item}" for item in report.get("not_completed") or [])
    lines.extend(["", "## Artifacts", ""])
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    return "\n".join(lines) + "\n"


def validate_public_report(report: dict[str, Any]) -> list[str]:
    encoded = json.dumps(report, sort_keys=True)
    errors: list[str] = []
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    for path in public_leak_paths(report):
        if path.endswith(".prompt_hash") or ".safety." in path:
            continue
        errors.append(f"public_leak:{path}")
    return sorted(set(errors))


def persist_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault(
        "prompt_scope",
        normalize_prompt_scope({"source": "imported-or-built-in-validation-prompts", "prompt_count": 0}),
    )
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    errors = validate_public_report(report)
    if errors:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"public_report_safety_failed"})
        report["safety_errors"] = errors
    report = support_bundle.sanitize(report)
    write_json(output_dir / "public_p2p_swarm_inference_v1_rc.json", report)
    (output_dir / "public_p2p_swarm_inference_v1_rc.md").write_text(render_markdown(report), encoding="utf-8")
    report["artifacts"]["public_p2p_swarm_inference_v1_rc_json"] = artifact_entry(
        output_dir / "public_p2p_swarm_inference_v1_rc.json",
        output_dir,
        kind="public_p2p_swarm_inference_v1_rc",
        schema=SCHEMA,
        ok=report.get("ok"),
    )
    report["artifacts"]["public_p2p_swarm_inference_v1_rc_markdown"] = artifact_entry(
        output_dir / "public_p2p_swarm_inference_v1_rc.md",
        output_dir,
        kind="public_p2p_swarm_inference_v1_rc_markdown",
    )
    bundle = support_bundle.sanitize({
        "schema": SUPPORT_SCHEMA,
        "ok": report.get("ok"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "rc": report.get("rc"),
        "p2p": report.get("p2p"),
        "inference": report.get("inference"),
        "artifacts": report.get("artifacts"),
        "output_request": report.get("output_request"),
        "prompt_scope": report.get("prompt_scope"),
        "answer_scope": report.get("answer_scope"),
        "shareable_summary": report.get("shareable_summary"),
        "safety": report.get("safety"),
    })
    write_json(output_dir / "support_bundle.json", bundle)
    report["artifacts"]["support_bundle_json"] = artifact_entry(output_dir / "support_bundle.json", output_dir, kind="public_p2p_swarm_inference_v1_rc_support_bundle", schema=SUPPORT_SCHEMA, ok=bundle.get("ok"))
    (output_dir / "public_p2p_swarm_inference_v1_rc.md").write_text(render_markdown(report), encoding="utf-8")
    write_json(output_dir / "public_p2p_swarm_inference_v1_rc.json", report)
    return report


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    peer_secret = effective_peer_secret(args)
    steps: list[dict[str, Any]] = []
    signed_local: dict[str, Any] = {}
    signed_local_report_path: str | Path | None = None
    if args.mode == MODE_LOCAL_SMOKE:
        step, signed_local = run_signed_local_smoke(args, output_dir=output_dir, runner=runner)
        signed_local_report_path = output_dir / "signed-local-v06" / "p2p_swarm_inference_v06.json"
        steps.append(step)
    elif args.mode == MODE_EVIDENCE_IMPORT:
        signed_local_report_path = args.signed_local_report or args.v06_local_report
        signed_local = load_json(signed_local_report_path)
    elif args.mode == MODE_KAGGLE_AUTO:
        step, signed_local = run_signed_local_smoke(args, output_dir=output_dir, runner=runner)
        signed_local_report_path = output_dir / "signed-local-v06" / "p2p_swarm_inference_v06.json"
        steps.append(step)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "p2p_swarm_inference_v06_pack.py"),
            v06_pack.MODE_KAGGLE_AUTO,
            "--output-dir",
            str(output_dir / "kaggle-v06"),
            "--swarm-id",
            args.swarm_id,
            "--public-host",
            args.public_host,
            "--p2p-port",
            str(args.p2p_port),
            "--coordinator-port",
            str(args.coordinator_port),
            "--backend",
            args.backend,
            "--hf-model-id",
            args.hf_model_id,
            "--prompt-text",
            args.prompt_text,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--startup-timeout",
            str(args.startup_timeout),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--http-timeout",
            str(args.http_timeout),
            "--peer-secret",
            peer_secret,
            "--require-signed",
            "--kaggle-stage-timeout-seconds",
            str(args.kaggle_stage_timeout_seconds),
            "--kaggle-push-timeout-seconds",
            str(args.kaggle_push_timeout_seconds),
            "--kaggle-delete-timeout-seconds",
            str(args.kaggle_delete_timeout_seconds),
            "--json",
        ]
        if args.kaggle_owner:
            command.extend(["--kaggle-owner", args.kaggle_owner])
        if args.kernel_slug_prefix:
            command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
        if args.skip_kaggle_cleanup:
            command.append("--skip-kaggle-cleanup")
        step, kaggle_payload = run_json_step(
            "v06_kaggle_auto",
            command,
            runner=runner,
            timeout_seconds=max(args.timeout_seconds, args.kaggle_stage_timeout_seconds, 60.0) + 1200.0,
            secret_values=[peer_secret],
        )
        steps.append(step)
        external_v06 = kaggle_payload
        kaggle_v06 = load_json(output_dir / "kaggle-v06" / "kaggle-auto" / "p2p_v06_kaggle_auto.json")
        runbook = write_runbook(args, output_dir)
        report = build_common_report(args, output_dir=output_dir, mode=args.mode, signed_local=signed_local, external_v06=external_v06, kaggle_v06=kaggle_v06, steps=steps, runbook=runbook, signed_local_report_path=signed_local_report_path)
        return persist_report(report, output_dir=output_dir)
    else:
        signed_local = {}

    external_v06 = load_json(args.v06_external_report)
    kaggle_v06 = load_json(args.v06_kaggle_report)
    runbook = write_runbook(args, output_dir)
    report = build_common_report(args, output_dir=output_dir, mode=args.mode, signed_local=signed_local, external_v06=external_v06, kaggle_v06=kaggle_v06, steps=steps, runbook=runbook, signed_local_report_path=signed_local_report_path)
    return persist_report(report, output_dir=output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Public P2P Swarm Inference v1.0 RC evidence.")
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--swarm-id", default="public-p2p-v1-rc")
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--p2p-port", type=int, default=DEFAULT_P2P_PORT)
    parser.add_argument("--coordinator-port", type=int, default=DEFAULT_COORDINATOR_PORT)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--prompt-text", default="CrowdTensor public P2P v1 RC")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--http-timeout", type=float, default=10.0)
    parser.add_argument("--peer-secret", default=os.environ.get("CROWDTENSOR_P2P_PEER_SECRET", ""))
    parser.add_argument("--signed-local-report", default="")
    parser.add_argument("--v06-local-report", default=DEFAULT_V06_LOCAL_REPORT)
    parser.add_argument("--v06-external-report", default=DEFAULT_V06_EXTERNAL_REPORT)
    parser.add_argument("--v06-kaggle-report", default=DEFAULT_V06_KAGGLE_REPORT)
    parser.add_argument("--kaggle-owner", default=os.environ.get("KAGGLE_USERNAME", ""))
    parser.add_argument("--kernel-slug-prefix", default="")
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--kaggle-stage-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.p2p_port < 1 or args.coordinator_port < 1:
        raise SystemExit("--p2p-port and --coordinator-port must be positive")
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    for name in ["startup_timeout", "timeout_seconds", "http_timeout", "kaggle_push_timeout_seconds", "kaggle_delete_timeout_seconds", "kaggle_stage_timeout_seconds"]:
        if float(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.mode == MODE_KAGGLE_AUTO and not args.kaggle_owner:
        raise SystemExit("kaggle-auto requires --kaggle-owner or KAGGLE_USERNAME")
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"Public P2P Swarm Inference v1.0 RC ready: {report.get('ok')}")
        print(f"Diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
