#!/usr/bin/env python3
"""Build and operate the optional CUDA Public Swarm Inference Beta pack."""

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

import public_swarm_inference_beta_pack as cpu_beta  # noqa: E402
import support_bundle  # noqa: E402
from crowdtensor.real_llm import BACKEND_CUDA, DEFAULT_MODEL_ID, DEFAULT_PROMPTS, cuda_runtime_summary  # noqa: E402
from crowdtensor.real_llm import normalize_partition_mode as normalize_real_llm_partition_mode  # noqa: E402
from kaggle_real_llm_live_package import DEFAULT_CUDA_TORCH_INDEX_URL, DEFAULT_CUDA_TORCH_RUNTIME_SPEC  # noqa: E402
from kaggle_real_llm_live_package import DEFAULT_TRANSFORMERS_SPEC  # noqa: E402


SCHEMA = "public_swarm_gpu_inference_beta_v1"
CLI_SCHEMA = "public_swarm_gpu_inference_beta_cli_v1"
REMOTE_REAL_SCHEMA = "remote_real_llm_sharded_beta_v1"
WORKLOAD_TYPE = "real_llm_sharded_infer"
DEFAULT_OUTPUT_DIR = "dist/public-swarm-gpu-inference-beta"
PUBLIC_ACTIONS = (
    "local-smoke",
    "local-loopback",
    "kaggle-package",
    "kaggle-auto",
    "evidence-import",
    "prepare",
    "coordinator",
    "miner",
    "verify",
    "collect",
    "clean",
)
SECRET_FRAGMENTS = cpu_beta.SECRET_FRAGMENTS + (
    "real_llm_sharded_result",
    "generated_token_ids",
    "generated_text",
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


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
    except (OSError, json.JSONDecodeError):
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
        summary = payload.get("diagnosis_summary") if isinstance(payload.get("diagnosis_summary"), dict) else {}
        for code in summary.get("codes") or []:
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


def base_safety() -> dict[str, Any]:
    return {
        "read_only_workload": WORKLOAD_TYPE,
        "backend": BACKEND_CUDA,
        "credential_values_redacted": True,
        "activation_payloads_redacted": True,
        "not_production": True,
        "not_p2p": True,
        "not_nat_traversal": True,
        "not_gpu_pooling_marketplace": True,
        "not_large_model_serving": True,
        "not_public_prompt_serving": True,
        "not_training": True,
        "stage_local_partition_beta": True,
    }


def limitations() -> list[str]:
    return [
        "Optional CUDA tiny GPT stage0/stage1 split inference Beta; stage-local mode is a device-side module partition proof, not production Swarm Inference.",
        "CUDA backend is explicit; the Coordinator may schedule metadata-only CUDA sessions, while participating Miners require torch CUDA.",
        "No P2P routing, NAT traversal, GPU pooling marketplace, GGUF/llama.cpp serving, large-model serving, training, payments, or staking.",
    ]


def report_filenames(mode: str) -> tuple[str, str]:
    safe = str(mode).replace("-", "_")
    return f"public_swarm_gpu_inference_beta_{safe}.json", f"public_swarm_gpu_inference_beta_{safe}.md"


def render_markdown(report: dict[str, Any]) -> str:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    lines = [
        "# CrowdTensor Public Swarm GPU Inference Beta",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- ready: `{beta.get('ready')}`",
        f"- backend: `{beta.get('backend')}`",
        f"- output_dir: `{report.get('output_dir')}`",
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


def persist_report(report: dict[str, Any], *, output_dir: Path, mode: str, secret_values: list[str] | None = None) -> dict[str, Any]:
    report = support_bundle.sanitize(redact_values(report, secret_values))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "Public Swarm GPU Inference Beta report contained secret-like fragments"
    json_name, markdown_name = report_filenames(mode)
    json_path = output_dir / json_name
    markdown_path = output_dir / markdown_name
    report.setdefault("artifacts", {})
    report["artifacts"]["public_swarm_gpu_inference_beta_json"] = artifact_entry(
        json_path,
        output_dir,
        kind="public_swarm_gpu_inference_beta",
        schema=SCHEMA,
        ok=report.get("ok"),
    )
    report["artifacts"]["public_swarm_gpu_inference_beta_markdown"] = artifact_entry(
        markdown_path,
        output_dir,
        kind="public_swarm_gpu_inference_beta_markdown",
    )
    write_json(json_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    report["artifacts"]["public_swarm_gpu_inference_beta_json"]["present"] = True
    report["artifacts"]["public_swarm_gpu_inference_beta_markdown"]["present"] = True
    write_json(json_path, report)
    return report


def remote_real_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
    inner = next((item for item in summaries.values() if isinstance(item, dict)), {})
    artifact = inner.get("artifact") if isinstance(inner.get("artifact"), dict) else {}
    assignment = inner.get("stage_assignment") if isinstance(inner.get("stage_assignment"), dict) else {}
    generation = generation_summary(payload)
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "diagnosis_codes": diagnosis_codes(payload),
        "generation": generation,
        "artifact": {
            "backend": artifact.get("backend"),
            "model_id": artifact.get("model_id"),
            "artifact_hash": artifact.get("artifact_hash"),
            "partition_mode": artifact.get("partition_mode"),
            "stage_local_partition_ready": artifact.get("stage_local_partition_ready"),
            "partition_parameter_split_valid": artifact.get("partition_parameter_split_valid"),
            "stage0_parameter_count": artifact.get("stage0_parameter_count"),
            "stage1_parameter_count": artifact.get("stage1_parameter_count"),
            "full_model_parameter_count": artifact.get("full_model_parameter_count"),
        },
        "stage_assignment": {
            "stage0_miner_id": assignment.get("stage0_miner_id"),
            "stage1_miner_id": assignment.get("stage1_miner_id"),
            "distinct_stage_miners": assignment.get("distinct_stage_miners"),
            "stage_assignment_valid": assignment.get("stage_assignment_valid"),
        },
    }


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


def build_local_smoke(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    cuda = cuda_runtime_summary()
    available = bool(cuda.get("cuda_available"))
    codes = set(diagnosis_codes(cuda))
    codes.update({
        "public_swarm_gpu_beta_smoke_ready",
        "gpu_runtime_smoke_ready",
        "read_only_workload",
        "not_production",
        "not_p2p",
    })
    if available:
        codes.update({"cuda_runtime_available", "gpu_runtime_ready"})
    else:
        codes.add("cuda_runtime_unavailable")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "mode": "local-smoke",
        "output_dir": str(output_dir),
        "beta": {
            "ready": True,
            "backend": BACKEND_CUDA,
            "smoke_only": True,
            "cuda_available": available,
            "model_id": args.hf_model_id,
            "request_count": args.request_count,
            "max_new_tokens": args.max_new_tokens,
        },
        "gpu_runtime": cuda,
        "diagnosis_codes": sorted(codes),
        "safety": base_safety(),
        "operator_action": [
            "Run local-loopback on a CUDA host to produce gpu-ready evidence.",
            "Run kaggle-package to prepare GPU-stage Kaggle artifacts for an external proof.",
        ],
        "limitations": limitations(),
    }


def build_local_loopback(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    child_dir = output_dir / "local-loopback-real-llm-cuda"
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
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--failure-mode",
        "none",
        "--stage-mode",
        "split",
        "--require-distinct-stage-miners",
        "--hf-model-id",
        args.hf_model_id,
        "--real-llm-backend",
        BACKEND_CUDA,
        "--real-llm-partition-mode",
        args.real_llm_partition_mode,
        "--timeout-seconds",
        str(int(args.timeout_seconds)),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    step, payload = run_json_step(
        "remote_real_llm_sharded_cuda_loopback",
        command,
        runner=runner,
        timeout_seconds=float(args.timeout_seconds) + 180.0,
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
        "cuda_runtime_available",
        "hf_transformers_cuda_ready",
    }
    if args.max_new_tokens > 1:
        required.add("multi_token_generation_ready")
    if args.real_llm_partition_mode == "stage_local":
        required.update({
            "stage_local_partition_ready",
            "stage0_partition_loaded",
            "stage1_partition_loaded",
            "partition_parameter_split_valid",
        })
    missing = sorted(required - codes)
    ready = bool(step.get("ok") and payload.get("schema") == REMOTE_REAL_SCHEMA and not missing)
    if ready:
        codes.update({
            "public_swarm_gpu_beta_ready",
            "two_stage_split_inference_ready",
            "gpu_runtime_ready",
            "gpu_stage0_ready",
            "gpu_stage1_ready",
            "local_loopback_ready",
            "read_only_workload",
            "not_production",
            "not_p2p",
            "not_large_model",
        })
    else:
        codes.add("public_swarm_gpu_beta_blocked")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": "local-loopback",
        "output_dir": str(output_dir),
        "beta": {
            "ready": ready,
            "backend": BACKEND_CUDA,
            "local_loopback": True,
            "model_id": args.hf_model_id,
            "request_count": args.request_count,
            "max_new_tokens": args.max_new_tokens,
            "stage_count": 2,
            "partition_mode": args.real_llm_partition_mode,
            "missing_codes": missing,
        },
        "steps": [step],
        "payload_summaries": {"remote_real_llm_sharded_beta": remote_real_summary(payload)},
        "gpu_runtime": cuda_runtime_summary(),
        "diagnosis_codes": sorted(codes),
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
            "Install the optional HF runtime with CUDA-enabled torch before running local-loopback.",
            "Use local-smoke on CPU-only hosts; it does not claim GPU readiness.",
        ],
        "limitations": limitations(),
    }


def build_kaggle_package(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    package_dir = output_dir / "kaggle-gpu-package"
    package_dir.mkdir(parents=True, exist_ok=True)
    runbook = package_dir / "KAGGLE_GPU_RUNBOOK.md"
    kernel_template = package_dir / "kaggle_gpu_stage_miner.py"
    stage0_env = package_dir / "stage0.miner.private.env.example"
    stage1_env = package_dir / "stage1.miner.private.env.example"
    stage0_env.write_text(
        "CROWDTENSOR_MINER_TOKEN=<stage0-token>\nCROWDTENSOR_REAL_LLM_BACKEND=hf_transformers_cuda\n",
        encoding="utf-8",
    )
    stage1_env.write_text(
        "CROWDTENSOR_MINER_TOKEN=<stage1-token>\nCROWDTENSOR_REAL_LLM_BACKEND=hf_transformers_cuda\n",
        encoding="utf-8",
    )
    kernel_template.write_text(
        "\n".join([
            "from __future__ import annotations",
            "import os, subprocess, sys",
            "STAGE = os.environ.get('CROWDTENSOR_REAL_LLM_STAGE_ROLE', 'stage0')",
            "cmd = [sys.executable, 'miner_cli.py', '--coordinator', os.environ['CROWDTENSOR_COORDINATOR_URL'],",
            "       '--miner-id', os.environ.get('CROWDTENSOR_MINER_ID', f'kaggle-gpu-{STAGE}'),",
            f"       '--max-tasks', '{args.max_new_tokens}', '--enable-hf-tiny-gpt-runtime', '--real-llm-backend', 'hf_transformers_cuda',",
            "       '--real-llm-stage-role', STAGE, '--real-llm-partition-mode', os.environ.get('CROWDTENSOR_REAL_LLM_PARTITION_MODE', 'stage_local'),",
            "       '--hf-model-id', os.environ.get('CROWDTENSOR_HF_MODEL_ID', 'sshleifer/tiny-gpt2')]",
            "raise SystemExit(subprocess.call(cmd))",
            "",
        ]),
        encoding="utf-8",
    )
    runbook.write_text(
        "\n".join([
            "# CrowdTensor Public Swarm GPU Inference Beta Kaggle Package",
            "",
            "Use this package only for a controlled private proof.",
            "",
            "1. Create a public Coordinator with `--real-llm-backend hf_transformers_cuda`; it can schedule CUDA sessions without local CUDA by using metadata-only artifact inspection.",
            "2. Create two private Kaggle script kernels with Accelerator set to GPU.",
            "3. Upload only the stage-specific `miner.private.env` values to the matching private kernel.",
            "4. Run stage0 and stage1 with `--real-llm-backend hf_transformers_cuda` and `--real-llm-stage-role stage0|stage1`.",
            "5. Import retained evidence with `crowdtensor public-swarm-gpu-beta evidence-import --gpu-report <report>`.",
            "",
            "Boundaries: not production, not P2P, not GPU pooling marketplace, not large-model serving.",
            "",
        ]),
        encoding="utf-8",
    )
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "mode": "kaggle-package",
        "output_dir": str(output_dir),
        "beta": {
            "ready": True,
            "backend": BACKEND_CUDA,
            "package_only": True,
            "model_id": args.hf_model_id,
            "request_count": args.request_count,
            "max_new_tokens": args.max_new_tokens,
            "partition_mode": args.real_llm_partition_mode,
        },
        "diagnosis_codes": [
            "public_swarm_gpu_beta_package_ready",
            "kaggle_gpu_package_ready",
            "kaggle_gpu_requires_private_kernels",
            "read_only_workload",
            "not_production",
            "not_p2p",
        ],
        "artifacts": {
            "kaggle_gpu_runbook": artifact_entry(runbook, output_dir, kind="kaggle_gpu_runbook"),
            "kaggle_gpu_stage_miner_template": artifact_entry(kernel_template, output_dir, kind="kaggle_gpu_stage_miner_template"),
            "stage0_env_example": artifact_entry(stage0_env, output_dir, kind="stage_env_example"),
            "stage1_env_example": artifact_entry(stage1_env, output_dir, kind="stage_env_example"),
        },
        "safety": {
            **base_safety(),
            "package_only": True,
            "private_kernel_expected": True,
            "raw_token_values_excluded": True,
        },
        "operator_action": [
            "Set Kaggle Accelerator to GPU for both private kernels.",
            "Do not upload operator.private.env or miner_registry.json to Kaggle.",
        ],
        "limitations": limitations(),
    }
    return report


def build_evidence_import(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    source = Path(args.gpu_report).resolve()
    payload = load_json(source)
    codes = set(diagnosis_codes(payload))
    ready = bool(
        source.is_file()
        and payload.get("ok") is True
        and "public_swarm_gpu_beta_ready" in codes
        and "hf_transformers_cuda_ready" in codes
    )
    if ready:
        codes.update({
            "public_swarm_gpu_beta_evidence_import_ready",
            "external_gpu_runtime_verified",
            "read_only_workload",
            "not_production",
            "not_p2p",
        })
    else:
        codes.add("public_swarm_gpu_beta_evidence_import_blocked")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": "evidence-import",
        "output_dir": str(output_dir),
        "beta": {
            "ready": ready,
            "backend": BACKEND_CUDA,
            "evidence_imported": ready,
            "source_schema": payload.get("schema"),
            "source_ok": payload.get("ok"),
        },
        "imported_report": {
            "path": str(source),
            "present": source.is_file(),
            "schema": payload.get("schema"),
            "ok": payload.get("ok"),
            "diagnosis_codes": sorted(codes),
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "imported_gpu_report": artifact_entry(
                source,
                output_dir,
                kind="public_swarm_gpu_inference_beta_import_source",
                schema=str(payload.get("schema") or ""),
                ok=payload.get("ok") if payload else None,
            ),
        },
        "safety": base_safety(),
        "operator_action": ["Run local-loopback on a CUDA host or import retained external GPU evidence."],
        "limitations": limitations(),
    }


def build_kaggle_auto(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    child_dir = output_dir / "kaggle-auto-real-llm-cuda"
    coordinator_url = "" if args.coordinator_url == "http://127.0.0.1:9300" else args.coordinator_url
    command = [
        sys.executable,
        str(ROOT / "scripts" / "real_llm_internet_beta_pack.py"),
        "--output-dir",
        str(child_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--miner-id",
        args.miner_id_prefix,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--hf-model-id",
        args.hf_model_id,
        "--real-llm-backend",
        BACKEND_CUDA,
        "--real-llm-partition-mode",
        args.real_llm_partition_mode,
        "--kaggle-owner",
        args.kaggle_owner,
        "--dataset-title",
        args.dataset_title,
        "--kernel-title-prefix",
        args.kernel_title_prefix,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
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
        "--json",
    ]
    if coordinator_url:
        command.extend(["--coordinator-url", coordinator_url])
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.torch_spec:
        command.extend(["--torch-spec", args.torch_spec])
    if args.torch_index_url:
        command.extend(["--torch-index-url", args.torch_index_url])
    if args.transformers_spec:
        command.extend(["--transformers-spec", args.transformers_spec])
    if args.dataset_slug:
        command.extend(["--dataset-slug", args.dataset_slug])
    if args.kernel_slug_prefix:
        command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
    if args.inline_kernel_payload:
        command.append("--inline-kernel-payload")
    else:
        command.append("--no-inline-kernel-payload")
    if args.skip_kaggle_cleanup:
        command.append("--skip-kaggle-cleanup")
    step, payload = run_json_step(
        "real_llm_internet_beta_kaggle_cuda_auto",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.kaggle_status_timeout_seconds)) + 240.0,
    )
    codes = set(diagnosis_codes(payload))
    required = {
        "real_llm_internet_beta_ready",
        "external_runtime_verified",
        "cuda_runtime_available",
        "hf_transformers_cuda_ready",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "kaggle_kernels_deleted",
        "token_rotation_required",
    }
    if args.max_new_tokens > 1:
        required.add("multi_token_generation_ready")
    if args.real_llm_partition_mode == "stage_local":
        required.update({
            "stage_local_partition_ready",
            "stage0_partition_loaded",
            "stage1_partition_loaded",
            "partition_parameter_split_valid",
        })
    missing = sorted(required - codes)
    ready = bool(step.get("ok") and payload.get("ok") and not missing)
    if ready:
        codes.update({
            "public_swarm_gpu_beta_ready",
            "public_swarm_gpu_beta_kaggle_auto_ready",
            "external_gpu_runtime_verified",
            "gpu_runtime_ready",
            "gpu_stage0_ready",
            "gpu_stage1_ready",
            "two_stage_split_inference_ready",
            "read_only_workload",
            "not_production",
            "not_p2p",
        })
    else:
        codes.add("public_swarm_gpu_beta_kaggle_auto_blocked")
    lifecycle = payload.get("kaggle_lifecycle") if isinstance(payload.get("kaggle_lifecycle"), dict) else {}
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": "kaggle-auto",
        "output_dir": str(output_dir),
        "beta": {
            "ready": ready,
            "backend": BACKEND_CUDA,
            "kaggle_auto": True,
            "model_id": args.hf_model_id,
            "request_count": args.request_count,
            "max_new_tokens": args.max_new_tokens,
            "stage_count": 2,
            "partition_mode": args.real_llm_partition_mode,
            "torch_spec": args.torch_spec,
            "torch_index_url": args.torch_index_url,
            "transformers_spec": args.transformers_spec,
            "missing_codes": missing,
        },
        "steps": [step],
        "payload_summaries": {"real_llm_internet_beta": {
            "schema": payload.get("schema"),
            "ok": payload.get("ok"),
            "mode": payload.get("mode"),
            "diagnosis_codes": diagnosis_codes(payload),
            "generation": generation_summary(payload),
            "runtime_classification": payload.get("runtime_classification") if isinstance(payload.get("runtime_classification"), dict) else {},
            "workload": payload.get("workload") if isinstance(payload.get("workload"), dict) else {},
            "partition_mode": args.real_llm_partition_mode,
        }},
        "kaggle_lifecycle": {
            "owner": lifecycle.get("owner"),
            "kernel_slug_prefix": lifecycle.get("kernel_slug_prefix"),
            "expected_push_count": lifecycle.get("expected_push_count"),
            "pushed_refs": lifecycle.get("pushed_refs") if isinstance(lifecycle.get("pushed_refs"), dict) else {},
            "kernels_deleted": lifecycle.get("kernels_deleted"),
            "cleanup_required": lifecycle.get("cleanup_required"),
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "real_llm_internet_beta_json": artifact_entry(
                child_dir / "real_llm_internet_beta.json",
                output_dir,
                kind="real_llm_internet_beta",
                schema="real_llm_internet_beta_v1",
                ok=payload.get("ok") if payload else None,
            ),
            "external_remote_real_llm_sharded_beta_json": artifact_entry(
                child_dir / "external-alpha" / "live-rc" / "remote-real-llm-runtime" / "remote_real_llm_sharded_beta.json",
                output_dir,
                kind="remote_real_llm_sharded_beta",
                schema=REMOTE_REAL_SCHEMA,
            ),
        },
        "safety": {
            **base_safety(),
            "kaggle_auto": True,
            "temporary_public_http": True,
            "kaggle_cleanup_required": True,
            "kaggle_cleanup_ok": bool(lifecycle.get("kernels_deleted")),
            "token_rotation_required": True,
            "coordinator_cuda_runtime_required": False,
            "miner_cuda_runtime_required": True,
            "cuda_torch_wheel_pinned": bool(args.torch_spec),
            "stage_local_partition": args.real_llm_partition_mode == "stage_local",
        },
        "operator_action": [
            "Rotate generated Coordinator and Miner tokens after every temporary public HTTP run.",
            "If cleanup failed, delete temporary private Kaggle kernels manually before retrying.",
            "Do not claim production Swarm Inference from this tiny CUDA proof.",
        ],
        "limitations": limitations(),
    }


def build_cpu_wrapped_action(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    action = args.mode
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_inference_beta_pack.py"),
        action,
        "--output-dir",
        str(output_dir / "cpu-operator-wrapper"),
        "--request-count",
        str(args.request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    if action != "clean":
        command.extend([
            "--coordinator-url",
            args.coordinator_url,
            "--port",
            str(args.port),
            "--base-port",
            str(args.base_port),
            "--bind-host",
            args.bind_host,
            "--miner-id-prefix",
            args.miner_id_prefix,
            "--hf-model-id",
            args.hf_model_id,
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--http-timeout",
            str(args.http_timeout),
        ])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if action in {"prepare", "coordinator", "miner"}:
        command.extend(["--lease-seconds", str(args.lease_seconds)])
    if action in {"prepare", "miner"}:
        command.extend([
            "--compute-seconds",
            str(args.compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
    if action in {"prepare", "coordinator", "verify", "collect"}:
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
    if action == "prepare" and args.replace:
        command.append("--replace")
    if action == "coordinator" and args.run:
        command.append("--run")
    if action == "miner":
        command.extend(["--stage", args.stage])
        if args.run:
            command.append("--run")
    if action in {"verify"} and args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    if action == "collect":
        if args.miner_id:
            command.extend(["--miner-id", args.miner_id])
        command.extend(["--artifact-timeout", str(args.artifact_timeout)])
    if action == "clean":
        if args.apply:
            command.append("--apply")
        if args.include_private:
            command.append("--include-private")
        if args.remove_empty_dir:
            command.append("--remove-empty-dir")
    secret_values = [args.observer_token, args.admin_token]
    step, payload = run_json_step(
        f"public_swarm_beta_{action}",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), 60.0) + 120.0,
        secret_values=secret_values,
    )
    ready = bool(step.get("ok") and payload.get("ok"))
    codes = set(diagnosis_codes(payload))
    if ready:
        codes.update({
            "public_swarm_gpu_beta_operator_workflow_ready",
            f"public_swarm_gpu_beta_{action.replace('-', '_')}_ready",
            "gpu_operator_overlay_ready",
            "read_only_workload",
            "not_production",
            "not_p2p",
        })
    else:
        codes.add("public_swarm_gpu_beta_operator_workflow_blocked")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": action,
        "output_dir": str(output_dir),
        "beta": {
            "ready": ready,
            "backend": BACKEND_CUDA,
            "operator_workflow": True,
            "cpu_wrapper": True,
            "model_id": args.hf_model_id,
            "request_count": args.request_count,
            "max_new_tokens": args.max_new_tokens,
            "partition_mode": args.real_llm_partition_mode,
        },
        "steps": [step],
        "payload_summaries": {"public_swarm_inference_beta": cpu_beta.swarm_child_summary(payload)},
        "diagnosis_codes": sorted(codes),
        "safety": base_safety(),
        "operator_action": [
            "Use local-loopback for actual CUDA validation.",
            "This operator wrapper preserves the existing Public Beta workflow shape while GPU-specific live execution remains explicit.",
        ],
        "limitations": limitations(),
    }


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "local-smoke":
        report = build_local_smoke(args, output_dir=output_dir)
    elif args.mode == "local-loopback":
        report = build_local_loopback(args, output_dir=output_dir, runner=runner)
    elif args.mode == "kaggle-package":
        report = build_kaggle_package(args, output_dir=output_dir)
    elif args.mode == "evidence-import":
        report = build_evidence_import(args, output_dir=output_dir)
    elif args.mode == "kaggle-auto":
        report = build_kaggle_auto(args, output_dir=output_dir, runner=runner)
    elif args.mode in {"prepare", "coordinator", "miner", "verify", "collect", "clean"}:
        report = build_cpu_wrapped_action(args, output_dir=output_dir, runner=runner)
    else:
        raise SystemExit(f"unknown Public Swarm GPU Inference Beta mode: {args.mode}")
    return persist_report(
        report,
        output_dir=output_dir,
        mode=args.mode,
        secret_values=[getattr(args, "observer_token", ""), getattr(args, "admin_token", "")],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or operate Public Swarm GPU Inference Beta evidence.")
    parser.add_argument("action", nargs="?", choices=PUBLIC_ACTIONS)
    parser.add_argument("--mode", choices=PUBLIC_ACTIONS, default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coordinator-url", default="http://127.0.0.1:9300")
    parser.add_argument("--public-host", default="24.199.118.54")
    parser.add_argument("--port", type=int, default=9320)
    parser.add_argument("--base-port", type=int, default=9321)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--miner-id-prefix", default="public-swarm-gpu-beta")
    parser.add_argument("--stage", choices=["stage0", "stage1"], default="")
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--hf-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--real-llm-partition-mode", choices=["full", "stage-local", "stage_local"], default="stage-local")
    parser.add_argument("--torch-spec", default="")
    parser.add_argument("--torch-index-url", default="")
    parser.add_argument("--transformers-spec", default=DEFAULT_TRANSFORMERS_SPEC)
    parser.add_argument("--prompt-texts", default=",".join(DEFAULT_PROMPTS))
    parser.add_argument("--observer-token", default="")
    parser.add_argument("--admin-token", default="")
    parser.add_argument("--miner-id", default="")
    parser.add_argument("--gpu-report", default="")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--artifact-timeout", type=float, default=60.0)
    parser.add_argument("--lease-seconds", type=float, default=15.0)
    parser.add_argument("--compute-seconds", type=float, default=0.2)
    parser.add_argument("--victim-compute-seconds", type=float, default=45.0)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--idle-sleep", type=float, default=0.5)
    parser.add_argument("--max-request-attempts", type=int, default=120)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--process-exit-timeout", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--kaggle-owner", default="")
    parser.add_argument("--dataset-slug", default="")
    parser.add_argument("--dataset-title", default="CrowdTensor Public Swarm GPU Beta Package")
    parser.add_argument("--kernel-slug-prefix", default="")
    parser.add_argument("--kernel-title-prefix", default="CrowdTensor Public Swarm GPU Beta Miner")
    parser.add_argument("--inline-kernel-payload", dest="inline_kernel_payload", action="store_true", default=True)
    parser.add_argument("--no-inline-kernel-payload", dest="inline_kernel_payload", action="store_false")
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=15.0)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--include-private", action="store_true")
    parser.add_argument("--remove-empty-dir", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    args.mode = args.mode or args.action or "local-smoke"
    if args.mode == "kaggle-auto" and not args.torch_spec:
        args.torch_spec = DEFAULT_CUDA_TORCH_RUNTIME_SPEC
        args.torch_index_url = args.torch_index_url or DEFAULT_CUDA_TORCH_INDEX_URL
    if args.request_count < 1 or args.request_count > 4:
        raise SystemExit("--request-count must be between 1 and 4")
    if args.max_new_tokens < 1 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 1 and 32")
    args.real_llm_partition_mode = normalize_real_llm_partition_mode(args.real_llm_partition_mode)
    if args.port < 1 or args.base_port < 1:
        raise SystemExit("--port and --base-port must be positive")
    for name in [
        "timeout_seconds",
        "remote_timeout_seconds",
        "http_timeout",
        "artifact_timeout",
        "lease_seconds",
        "heartbeat_interval",
        "idle_sleep",
        "startup_timeout",
        "process_exit_timeout",
        "poll_interval",
        "kaggle_push_timeout_seconds",
        "kaggle_delete_timeout_seconds",
        "kaggle_status_timeout_seconds",
        "kaggle_status_poll_interval",
    ]:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.compute_seconds < 0:
        raise SystemExit("--compute-seconds must be non-negative")
    if args.victim_compute_seconds <= 0:
        raise SystemExit("--victim-compute-seconds must be positive")
    if args.max_request_attempts < 1:
        raise SystemExit("--max-request-attempts must be at least 1")
    if args.mode == "kaggle-auto" and not args.kaggle_owner:
        import os
        args.kaggle_owner = os.environ.get("KAGGLE_USERNAME", "")
        if not args.kaggle_owner:
            kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
            if kaggle_config.is_file():
                try:
                    args.kaggle_owner = str((json.loads(kaggle_config.read_text(encoding="utf-8")) or {}).get("username") or "")
                except json.JSONDecodeError:
                    args.kaggle_owner = ""
        if not args.kaggle_owner:
            raise SystemExit("kaggle-auto requires --kaggle-owner or KAGGLE_USERNAME/~/.kaggle/kaggle.json")
    if args.mode == "miner" and not args.stage:
        raise SystemExit("miner requires --stage stage0|stage1")
    if args.mode == "evidence-import" and not args.gpu_report:
        raise SystemExit("evidence-import requires --gpu-report")
    return args


def print_human(report: dict[str, Any]) -> None:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    print("CrowdTensor Public Swarm GPU Inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {beta.get('ready')}")
    print(f"  backend: {beta.get('backend')}")
    print(f"  output: {report.get('output_dir')}")
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
