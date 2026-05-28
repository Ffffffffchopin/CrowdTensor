#!/usr/bin/env python3
"""Build the GPU multi-machine sharded generation Beta evidence wrapper."""

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

import public_swarm_gpu_inference_beta_pack as gpu_pack  # noqa: E402
from crowdtensor.real_llm import DEFAULT_MODEL_ID  # noqa: E402


SCHEMA = "gpu_sharded_generation_beta_v1"
CLI_SCHEMA = "gpu_sharded_generation_beta_cli_v1"
DEFAULT_OUTPUT_DIR = "dist/gpu-sharded-generation-beta"
MODES = ("local-loopback", "kaggle-auto", "evidence-import")
Runner = Callable[..., subprocess.CompletedProcess[str]]

SECRET_FRAGMENTS = tuple(
    fragment
    for fragment in gpu_pack.SECRET_FRAGMENTS
    if fragment not in {"generated_text", "generated_token_ids", "next_token_text", "next_token_id"}
)
RAW_GENERATION_KEYS = {
    "generated_text",
    "generated_token_ids",
    "next_token_text",
    "next_token_id",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


def redact_text(text: str) -> str:
    redacted = str(text)
    for fragment in SECRET_FRAGMENTS:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def has_raw_generation_payload(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in RAW_GENERATION_KEYS:
                return True
            if has_raw_generation_payload(item):
                return True
    if isinstance(value, list):
        return any(has_raw_generation_payload(item) for item in value)
    return False


def diagnosis_codes(*payloads: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    codes: set[str] = set(extra or [])
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for code in payload.get("diagnosis_codes") or []:
            if isinstance(code, str):
                codes.add(code)
        summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
        for summary in summaries.values():
            if isinstance(summary, dict):
                for code in summary.get("diagnosis_codes") or []:
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


def run_json_step(name: str, command: list[str], *, runner: Runner, timeout_seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
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
            step["stdout_tail"] = redact_text(completed.stdout[-1200:])
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1200:])
    return step, payload


def find_generation_summary(payload: dict[str, Any]) -> dict[str, Any]:
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
        generation = item.get("generation") if isinstance(item.get("generation"), dict) else {}
        if generation:
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


def imported_generation_count(payload: dict[str, Any], max_new_tokens: int) -> int:
    generation = find_generation_summary(payload)
    if generation:
        return int(generation.get("generated_token_count") or 0)
    if "multi_token_generation_ready" in diagnosis_codes(payload):
        return max_new_tokens
    return 0


def base_command(args: argparse.Namespace, output_dir: Path, mode: str) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_gpu_inference_beta_pack.py"),
        mode,
        "--output-dir",
        str(output_dir / "public-gpu-beta"),
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--hf-model-id",
        args.hf_model_id,
        "--real-llm-partition-mode",
        args.real_llm_partition_mode,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if mode in {"local-loopback", "kaggle-auto"}:
        command.extend([
            "--public-host",
            args.public_host,
            "--port",
            str(args.port),
            "--base-port",
            str(args.base_port),
            "--bind-host",
            args.bind_host,
            "--miner-id-prefix",
            args.miner_id_prefix,
            "--http-timeout",
            str(args.http_timeout),
        ])
    if mode == "kaggle-auto":
        command.extend([
            "--kaggle-owner",
            args.kaggle_owner,
            "--dataset-title",
            args.dataset_title,
            "--kernel-title-prefix",
            args.kernel_title_prefix,
            "--startup-timeout",
            str(args.startup_timeout),
            "--process-exit-timeout",
            str(args.process_exit_timeout),
            "--poll-interval",
            str(args.poll_interval),
            "--kaggle-push-timeout-seconds",
            str(args.kaggle_push_timeout_seconds),
            "--kaggle-delete-timeout-seconds",
            str(args.kaggle_delete_timeout_seconds),
            "--kaggle-status-timeout-seconds",
            str(args.kaggle_status_timeout_seconds),
            "--kaggle-status-poll-interval",
            str(args.kaggle_status_poll_interval),
            "--lease-seconds",
            str(args.lease_seconds),
            "--compute-seconds",
            str(args.compute_seconds),
            "--victim-compute-seconds",
            str(args.victim_compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--idle-sleep",
            str(args.idle_sleep),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
        if args.dataset_slug:
            command.extend(["--dataset-slug", args.dataset_slug])
        if args.kernel_slug_prefix:
            command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
        command.append("--inline-kernel-payload" if args.inline_kernel_payload else "--no-inline-kernel-payload")
        if args.skip_kaggle_cleanup:
            command.append("--skip-kaggle-cleanup")
    return command


def wrap_payload(args: argparse.Namespace, *, output_dir: Path, mode: str, step: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    generation = find_generation_summary(payload)
    generated_count = imported_generation_count(payload, args.max_new_tokens)
    generation_ready = bool(generated_count >= args.max_new_tokens and "multi_token_generation_ready" in codes)
    required = {
        "public_swarm_gpu_beta_ready",
        "hf_transformers_cuda_ready",
        "stage0_partition_loaded",
        "stage1_partition_loaded",
        "partition_parameter_split_valid",
        "stage_local_partition_ready",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
    }
    if mode == "kaggle-auto":
        required.update({"public_swarm_gpu_beta_kaggle_auto_ready", "external_gpu_runtime_verified", "kaggle_kernels_deleted"})
    missing = sorted(required - codes)
    ready = bool(step.get("ok") and payload.get("ok") and not missing and generation_ready)
    if ready:
        codes.update({
            "gpu_sharded_generation_ready",
            "gpu_multi_machine_generation_ready" if mode == "kaggle-auto" else "gpu_loopback_generation_ready",
            "read_only_workload",
            "not_production",
            "not_p2p",
        })
    else:
        codes.add("gpu_sharded_generation_blocked")
        if not generation_ready:
            codes.add("gpu_multi_token_generation_missing")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": mode,
        "output_dir": str(output_dir),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "generated_token_count": generated_count,
            "generated_text_hash": generation.get("generated_text_hash"),
            "generated_text_redacted": True,
            "multi_token_generation_ready": generation_ready,
            "raw_generated_text_public": False,
        },
        "gpu": {
            "backend": "hf_transformers_cuda",
            "model_id": args.hf_model_id,
            "partition_mode": args.real_llm_partition_mode,
            "stage_count": 2,
            "distinct_stage_miners": "distinct_stage_miners" in codes,
            "kaggle_auto": mode == "kaggle-auto",
        },
        "steps": [step],
        "payload_summary": {
            "schema": payload.get("schema"),
            "ok": payload.get("ok"),
            "mode": payload.get("mode"),
            "diagnosis_codes": diagnosis_codes(payload),
        },
        "diagnosis_codes": sorted(codes),
        "missing_codes": missing,
        "artifacts": {
            "public_swarm_gpu_beta_json": artifact_entry(
                output_dir / "public-gpu-beta" / f"public_swarm_gpu_inference_beta_{mode.replace('-', '_')}.json",
                output_dir,
                kind="public_swarm_gpu_inference_beta",
                schema="public_swarm_gpu_inference_beta_v1",
                ok=payload.get("ok") if payload else None,
            ),
        },
        "safety": {
            "read_only": True,
            "tiny_model_only": True,
            "raw_generated_text_redacted": True,
            "generated_token_ids_redacted": True,
            "not_production": True,
            "not_p2p": True,
            "not_gpu_pooling_marketplace": True,
            "not_large_model_serving": True,
            "not_training": True,
        },
        "limitations": [
            "Tiny GPT CUDA stage-local sharded generation Beta; not production Hivemind-level serving.",
            "Two-stage task-level generation proof; no P2P routing, NAT traversal, GPU marketplace, or large-model serving.",
            "Kaggle mode uses temporary private kernels and requires token rotation after runs.",
        ],
    }
    return persist_report(report, output_dir=output_dir, mode=mode)


def build_evidence_import(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    source = Path(args.gpu_report).resolve()
    payload = load_json(source)
    step = {
        "name": "gpu_sharded_generation_evidence_import",
        "ok": source.is_file() and bool(payload),
        "source_path": str(source),
        "payload_schema": payload.get("schema"),
        "payload_ok": payload.get("ok"),
    }
    report = wrap_payload(args, output_dir=output_dir, mode="evidence-import", step=step, payload=payload)
    if has_raw_generation_payload(payload):
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "Imported GPU generation evidence contained raw generated token payloads"
    report.setdefault("artifacts", {})["imported_gpu_report"] = artifact_entry(
        source,
        output_dir,
        kind="gpu_sharded_generation_import_source",
        schema=str(payload.get("schema") or ""),
        ok=payload.get("ok") if payload else None,
    )
    return persist_report(report, output_dir=output_dir, mode="evidence-import")


def persist_report(report: dict[str, Any], *, output_dir: Path, mode: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "GPU sharded generation report contained secret-like fragments"
    json_path = output_dir / f"gpu_sharded_generation_beta_{mode.replace('-', '_')}.json"
    md_path = output_dir / f"gpu_sharded_generation_beta_{mode.replace('-', '_')}.md"
    report.setdefault("artifacts", {})
    report["artifacts"]["gpu_sharded_generation_beta_json"] = artifact_entry(
        json_path,
        output_dir,
        kind="gpu_sharded_generation_beta",
        schema=SCHEMA,
        ok=report.get("ok"),
    )
    report["artifacts"]["gpu_sharded_generation_beta_markdown"] = artifact_entry(
        md_path,
        output_dir,
        kind="gpu_sharded_generation_beta_markdown",
    )
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    report["artifacts"]["gpu_sharded_generation_beta_json"]["present"] = True
    report["artifacts"]["gpu_sharded_generation_beta_markdown"]["present"] = True
    write_json(json_path, report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    generation = report.get("generation") if isinstance(report.get("generation"), dict) else {}
    gpu = report.get("gpu") if isinstance(report.get("gpu"), dict) else {}
    lines = [
        "# CrowdTensor GPU Sharded Generation Beta",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- backend: `{gpu.get('backend')}`",
        f"- model: `{gpu.get('model_id')}`",
        f"- generated_token_count: `{generation.get('generated_token_count')}`",
        f"- max_new_tokens: `{generation.get('max_new_tokens')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Boundaries",
        "",
    ]
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "evidence-import":
        return build_evidence_import(args, output_dir=output_dir)
    command = base_command(args, output_dir, args.mode)
    step, payload = run_json_step(
        f"public_swarm_gpu_beta_{args.mode}",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.kaggle_status_timeout_seconds)) + 600.0,
    )
    return wrap_payload(args, output_dir=output_dir, mode=args.mode, step=step, payload=payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GPU sharded multi-token generation Beta evidence.")
    parser.add_argument("mode", nargs="?", choices=MODES, default="local-loopback")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--hf-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--real-llm-partition-mode", choices=["stage-local", "stage_local"], default="stage-local")
    parser.add_argument("--gpu-report", default="")
    parser.add_argument("--public-host", default="24.199.118.54")
    parser.add_argument("--port", type=int, default=9340)
    parser.add_argument("--base-port", type=int, default=9341)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--miner-id-prefix", default="gpu-sharded-generation-beta")
    parser.add_argument("--kaggle-owner", default="")
    parser.add_argument("--dataset-slug", default="")
    parser.add_argument("--dataset-title", default="CrowdTensor GPU Sharded Generation Beta Package")
    parser.add_argument("--kernel-slug-prefix", default="")
    parser.add_argument("--kernel-title-prefix", default="CrowdTensor GPU Sharded Generation Beta Miner")
    parser.add_argument("--inline-kernel-payload", dest="inline_kernel_payload", action="store_true", default=True)
    parser.add_argument("--no-inline-kernel-payload", dest="inline_kernel_payload", action="store_false")
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--process-exit-timeout", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=15.0)
    parser.add_argument("--lease-seconds", type=float, default=15.0)
    parser.add_argument("--compute-seconds", type=float, default=0.2)
    parser.add_argument("--victim-compute-seconds", type=float, default=45.0)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--idle-sleep", type=float, default=0.5)
    parser.add_argument("--max-request-attempts", type=int, default=240)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.request_count != 1:
        raise SystemExit("--request-count must be 1 for gpu-generate Beta")
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    if args.mode == "evidence-import" and not args.gpu_report:
        raise SystemExit("evidence-import requires --gpu-report")
    if args.mode == "kaggle-auto" and not args.kaggle_owner:
        args.kaggle_owner = gpu_pack.cpu_beta.default_kaggle_owner() if hasattr(gpu_pack.cpu_beta, "default_kaggle_owner") else ""
    args.real_llm_partition_mode = "stage_local"
    return args


def main() -> None:
    report = build_report(parse_args())
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
