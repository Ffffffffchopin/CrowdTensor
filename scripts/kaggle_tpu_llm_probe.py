#!/usr/bin/env python3
"""Run a bounded Kaggle TPU v5e LLM-runtime probe."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_32b_stage_owned_safetensors_probe as loading_probe


SCHEMA = "kaggle_tpu_llm_probe_v1"
STAGE_SCHEMA = "kaggle_tpu_llm_runtime_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-tpu-llm-probe"
DEFAULT_ACCELERATORS = "tpuV5e8,TPU_V5E_8,TPU v5e-8,tpu1vmV38,Tpu1VmV38,TPUv5e-8"
Runner = Callable[..., subprocess.CompletedProcess[str]]


KERNEL_TEMPLATE = r'''
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "__STAGE_SCHEMA__"
REQUESTED_ACCELERATOR = __REQUESTED_ACCELERATOR_JSON__
OUT = Path("/kaggle/working")
REPORT_PATH = OUT / "kaggle_tpu_llm_probe_report.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_payload(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def safe_tail(value, limit=1200):
    text = str(value or "")[-limit:]
    for fragment in ["KAGGLE_KEY", "KAGGLE_USERNAME", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "Bearer "]:
        text = text.replace(fragment, "<redacted>")
    return text


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(command, timeout=120):
    started = time.monotonic()
    try:
        completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "duration_seconds": round(time.monotonic() - started, 3)}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_tail": safe_tail(completed.stdout),
        "stderr_tail": safe_tail(completed.stderr),
    }


def env_summary():
    keys = [
        "TPU_NAME",
        "TPU_WORKER_ID",
        "TPU_WORKER_HOSTNAMES",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "TPU_HOST_BOUNDS",
        "COLAB_TPU_ADDR",
        "PJRT_DEVICE",
        "XLA_FLAGS",
    ]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "env_present": {key: bool(os.environ.get(key)) for key in keys},
        "requested_accelerator": REQUESTED_ACCELERATOR,
    }


def jax_probe():
    started = time.monotonic()
    report = {
        "ok": False,
        "backend": "jax",
        "jax_imported": False,
        "tpu_device_count": 0,
        "devices_public": [],
        "simple_op_ready": False,
        "synthetic_llm_ready": False,
        "generated_token_count": 0,
        "generated_token_ids_public": False,
        "raw_prompt_public": False,
        "activation_public": False,
        "kv_cache_public": False,
        "diagnosis_codes": [],
        "blockers": [],
    }
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:
        report.update({
            "error_type": type(exc).__name__,
            "error_public": safe_tail(str(exc), 240),
            "diagnosis_codes": ["kaggle_tpu_jax_missing"],
            "blockers": ["jax_import_failed"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        return report
    report["jax_imported"] = True
    report["jax_version"] = str(getattr(jax, "__version__", ""))
    try:
        devices = list(jax.devices())
        tpu_devices = [device for device in devices if str(getattr(device, "platform", "")).lower() == "tpu"]
        report["devices_public"] = [
            {
                "platform": str(getattr(device, "platform", "")),
                "device_kind": str(getattr(device, "device_kind", "")),
            }
            for device in devices
        ]
        report["tpu_device_count"] = len(tpu_devices)
        if not tpu_devices:
            report["diagnosis_codes"].append("kaggle_tpu_jax_tpu_device_missing")
            report["blockers"].append("jax_tpu_device_missing")
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            return report
        device = tpu_devices[0]
        a = jax.device_put(jnp.arange(16, dtype=jnp.float32).reshape(4, 4), device)
        b = jax.device_put(jnp.eye(4, dtype=jnp.float32), device)
        simple = jnp.matmul(a, b).block_until_ready()
        report["simple_op_ready"] = bool(simple.shape == (4, 4))

        vocab_size = 128
        hidden_size = 64
        seq_len = 8
        key = jax.random.PRNGKey(17)
        keys = jax.random.split(key, 8)
        scale = jnp.array(0.02, dtype=jnp.float32)
        params = {
            "embed": jax.random.normal(keys[0], (vocab_size, hidden_size), dtype=jnp.float32) * scale,
            "wq": jax.random.normal(keys[1], (hidden_size, hidden_size), dtype=jnp.float32) * scale,
            "wk": jax.random.normal(keys[2], (hidden_size, hidden_size), dtype=jnp.float32) * scale,
            "wv": jax.random.normal(keys[3], (hidden_size, hidden_size), dtype=jnp.float32) * scale,
            "wo": jax.random.normal(keys[4], (hidden_size, hidden_size), dtype=jnp.float32) * scale,
            "w1": jax.random.normal(keys[5], (hidden_size, hidden_size * 4), dtype=jnp.float32) * scale,
            "w2": jax.random.normal(keys[6], (hidden_size * 4, hidden_size), dtype=jnp.float32) * scale,
            "lm_head": jax.random.normal(keys[7], (hidden_size, vocab_size), dtype=jnp.float32) * scale,
        }
        params = jax.device_put(params, device)
        token_ids = jax.device_put(jnp.arange(seq_len, dtype=jnp.int32)[None, :] % vocab_size, device)

        @jax.jit
        def tiny_causal_lm_generate(p, ids):
            hidden = p["embed"][ids]
            q = hidden @ p["wq"]
            k = hidden @ p["wk"]
            v = hidden @ p["wv"]
            scores = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(jnp.array(hidden_size, dtype=jnp.float32))
            causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
            scores = jnp.where(causal[None, :, :], scores, jnp.array(-1.0e9, dtype=jnp.float32))
            attn = jax.nn.softmax(scores, axis=-1)
            hidden = hidden + (attn @ v) @ p["wo"]
            mlp = jax.nn.gelu(hidden @ p["w1"]) @ p["w2"]
            hidden = hidden + mlp
            logits = hidden[:, -1, :] @ p["lm_head"]
            return jnp.argmax(logits, axis=-1)

        next_token = tiny_causal_lm_generate(params, token_ids).block_until_ready()
        token_hash = sha_payload({"next_token_id": int(next_token[0])})
        report.update({
            "ok": True,
            "simple_op_ready": True,
            "synthetic_llm_ready": True,
            "synthetic_llm_runtime": "jax_tiny_causal_lm_jit",
            "generated_token_count": 1,
            "next_token_hash": token_hash,
            "next_token_id_public": False,
            "diagnosis_codes": ["kaggle_tpu_jax_runtime_ready", "kaggle_tpu_synthetic_llm_ready"],
            "blockers": [],
        })
    except Exception as exc:
        report.update({
            "ok": False,
            "error_type": type(exc).__name__,
            "error_public": safe_tail(str(exc), 240),
            "error_digest": sha_payload(str(exc)),
            "diagnosis_codes": ["kaggle_tpu_jax_runtime_failed"],
            "blockers": ["jax_tpu_runtime_failed"],
        })
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


def torch_xla_probe():
    report = {
        "backend": "torch_xla",
        "torch_xla_imported": False,
        "xla_device_public": "",
        "simple_op_ready": False,
        "diagnosis_codes": [],
        "blockers": [],
    }
    try:
        import torch
        import torch_xla.core.xla_model as xm
    except Exception as exc:
        report.update({
            "error_type": type(exc).__name__,
            "error_public": safe_tail(str(exc), 240),
            "diagnosis_codes": ["kaggle_tpu_torch_xla_missing"],
            "blockers": ["torch_xla_import_failed"],
        })
        return report
    try:
        device = xm.xla_device()
        value = torch.arange(8, dtype=torch.float32, device=device).sum()
        xm.mark_step()
        report.update({
            "torch_xla_imported": True,
            "xla_device_public": str(device),
            "simple_op_ready": bool(float(value.cpu()) == 28.0),
            "diagnosis_codes": ["kaggle_tpu_torch_xla_runtime_ready"],
        })
    except Exception as exc:
        report.update({
            "torch_xla_imported": True,
            "error_type": type(exc).__name__,
            "error_public": safe_tail(str(exc), 240),
            "diagnosis_codes": ["kaggle_tpu_torch_xla_runtime_failed"],
            "blockers": ["torch_xla_runtime_failed"],
        })
    return report


def tensorflow_probe():
    report = {
        "backend": "tensorflow",
        "tensorflow_imported": False,
        "tpu_logical_device_count": 0,
        "simple_op_ready": False,
        "diagnosis_codes": [],
        "blockers": [],
    }
    try:
        import tensorflow as tf
    except Exception as exc:
        report.update({
            "error_type": type(exc).__name__,
            "error_public": safe_tail(str(exc), 240),
            "diagnosis_codes": ["kaggle_tpu_tensorflow_missing"],
            "blockers": ["tensorflow_import_failed"],
        })
        return report
    report["tensorflow_imported"] = True
    report["tensorflow_version"] = str(getattr(tf, "__version__", ""))
    try:
        tpus = tf.config.list_logical_devices("TPU")
        report["tpu_logical_device_count"] = len(tpus)
        if not tpus:
            report["diagnosis_codes"] = ["kaggle_tpu_tensorflow_tpu_device_missing"]
            report["blockers"] = ["tensorflow_tpu_device_missing"]
            return report
        with tf.device(tpus[0].name):
            value = tf.reduce_sum(tf.range(8, dtype=tf.float32)).numpy().item()
        report.update({
            "simple_op_ready": bool(value == 28.0),
            "diagnosis_codes": ["kaggle_tpu_tensorflow_runtime_ready"],
        })
    except Exception as exc:
        report.update({
            "error_type": type(exc).__name__,
            "error_public": safe_tail(str(exc), 240),
            "diagnosis_codes": ["kaggle_tpu_tensorflow_runtime_failed"],
            "blockers": ["tensorflow_tpu_runtime_failed"],
        })
    return report


def main():
    started = time.monotonic()
    report = {
        "schema": SCHEMA,
        "requested_accelerator": REQUESTED_ACCELERATOR,
        "ok": False,
        "tpu_runtime_ready": False,
        "llm_inference_ready": False,
        "fresh_kaggle_run_performed": True,
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_public": False,
        "kv_cache_public": False,
        "credentials_public": False,
        "started_at": utc_now(),
        "env": env_summary(),
    }
    report["jax"] = jax_probe()
    report["torch_xla"] = torch_xla_probe()
    report["tensorflow"] = tensorflow_probe()
    report["tpu_runtime_ready"] = bool(
        report["jax"].get("synthetic_llm_ready")
        or report["torch_xla"].get("simple_op_ready")
        or report["tensorflow"].get("simple_op_ready")
    )
    report["llm_inference_ready"] = bool(report["jax"].get("synthetic_llm_ready"))
    report["ok"] = bool(report["llm_inference_ready"])
    diagnosis = []
    blockers = []
    for key in ["jax", "torch_xla", "tensorflow"]:
        child = report.get(key) if isinstance(report.get(key), dict) else {}
        diagnosis.extend(list(child.get("diagnosis_codes") or []))
        blockers.extend(list(child.get("blockers") or []))
    if report["ok"]:
        diagnosis.append("kaggle_tpu_llm_probe_ready")
        blockers = []
    else:
        diagnosis.append("kaggle_tpu_llm_probe_not_ready")
        if not report["llm_inference_ready"]:
            blockers.append("tpu_llm_inference_not_ready")
    report["diagnosis_codes"] = sorted(set(diagnosis))
    report["blockers"] = sorted(set(blockers))
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["updated_at"] = utc_now()
    write_json(REPORT_PATH, report)
    print(json.dumps({"schema": SCHEMA, "ok": report["ok"], "llm_inference_ready": report["llm_inference_ready"], "diagnosis_codes": report["diagnosis_codes"]}, sort_keys=True))


main()
'''


def utc_now() -> str:
    return loading_probe.utc_now()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    loading_probe.write_json(path, payload)


def load_json(path: Path) -> dict[str, Any]:
    return loading_probe.load_json(path)


def parse_accelerators(value: str) -> list[str]:
    rows = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return rows or [item.strip() for item in DEFAULT_ACCELERATORS.split(",") if item.strip()]


def push_accepted(step: dict[str, Any]) -> bool:
    output = f"{step.get('stdout_tail') or ''}\n{step.get('stderr_tail') or ''}"
    return bool(step.get("ok")) and "Kernel version" in output and "successfully pushed" in output


def render_kernel(accelerator: str) -> str:
    return KERNEL_TEMPLATE.replace("__STAGE_SCHEMA__", STAGE_SCHEMA).replace(
        "__REQUESTED_ACCELERATOR_JSON__",
        json.dumps(accelerator),
    )


def build_package(args: argparse.Namespace, *, output_dir: Path, accelerator: str) -> dict[str, Any]:
    owner = args.kaggle_owner or loading_probe.default_kaggle_owner()
    if not owner:
        raise SystemExit("--kaggle-owner or ~/.kaggle/kaggle.json username is required")
    suffix = str(int(time.time()))[-8:]
    accel_slug = loading_probe.safe_slug(accelerator, default="tpu")[:12]
    slug = f"{loading_probe.safe_slug(args.kernel_slug_prefix)[:24]}-{accel_slug}-{suffix}"[:45].strip("-")
    kernel_dir = output_dir / "private-kaggle-tpu-kernels" / accel_slug
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text(render_kernel(accelerator), encoding="utf-8")
    metadata = {
        "id": f"{owner}/{slug}",
        "title": f"CT TPU LLM Probe {accel_slug} {suffix}",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "true",
        "enable_internet": "true",
        "machine_shape": accelerator,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    write_json(kernel_dir / "kernel-metadata.json", metadata)
    return {
        "accelerator": accelerator,
        "kernel_dir": kernel_dir,
        "declared_kernel_ref": metadata["id"],
        "kernel_ref": metadata["id"],
        "metadata": metadata,
        "report_filename": "kaggle_tpu_llm_probe_report.json",
    }


def public_step(step: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(step)
    for key in ("command_line", "stdout_tail", "stderr_tail"):
        if key in cleaned:
            cleaned[key] = str(cleaned[key]).replace("private-kaggle-tpu-kernels", "<private-payload-dir>")
    command = cleaned.get("command_public")
    if isinstance(command, list):
        cleaned["command_public"] = [
            "<private-payload-dir>" if "private-kaggle-tpu-kernels" in str(part) else part
            for part in command
        ]
    return cleaned


def missing_runtime_report_if_needed(
    *,
    package: dict[str, Any],
    stage_output: Path,
    steps: list[dict[str, Any]],
) -> dict[str, Any] | None:
    report_path = stage_output / str(package["report_filename"])
    if report_path.exists():
        return None
    status_steps = [step for step in steps if step.get("name") == "kaggle_kernel_status"]
    last_status = status_steps[-1] if status_steps else {}
    diagnosis_codes = ["kaggle_tpu_report_missing"]
    blockers = ["kaggle_tpu_report_missing"]
    if last_status.get("terminal") is False and str(last_status.get("status") or "").upper() == "QUEUED":
        diagnosis_codes.append("kaggle_tpu_kernel_queued_timeout")
        blockers.insert(0, "kaggle_tpu_kernel_queued_timeout")
    return {
        "schema": STAGE_SCHEMA,
        "requested_accelerator": package["accelerator"],
        "ok": False,
        "tpu_runtime_ready": False,
        "llm_inference_ready": False,
        "diagnosis_codes": diagnosis_codes,
        "blockers": blockers,
    }


def run_package(
    args: argparse.Namespace,
    *,
    package: dict[str, Any],
    output_dir: Path,
    runner: Runner,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    push_command = [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(package["kernel_dir"]),
        "-t",
        str(args.kernel_timeout_seconds),
        "--accelerator",
        str(package["accelerator"]),
    ]
    print(f"[{utc_now()}] pushing private Kaggle TPU kernel {package['declared_kernel_ref']} accelerator={package['accelerator']}", flush=True)
    push_step = loading_probe.run_step(
        "kaggle_kernel_push",
        push_command,
        runner=runner,
        timeout_seconds=args.kaggle_push_timeout_seconds,
    )
    push_step["accepted"] = push_accepted(push_step)
    steps.append(push_step)
    if not push_step.get("accepted"):
        return {}, steps
    kernel_ref, resolve_step = loading_probe.resolve_pushed_kernel_ref(
        package,
        push_step,
        runner=runner,
        timeout_seconds=args.kaggle_push_timeout_seconds,
    )
    if resolve_step:
        steps.append(resolve_step)
    package["kernel_ref"] = kernel_ref
    print(f"[{utc_now()}] waiting for {kernel_ref}", flush=True)
    status_step = loading_probe.wait_kaggle_terminal(
        kernel_ref,
        runner=runner,
        timeout_seconds=args.kaggle_status_timeout_seconds,
        poll_interval=args.kaggle_status_poll_interval,
    )
    steps.append(status_step)
    stage_output = output_dir / "kaggle-output" / loading_probe.safe_slug(str(package["accelerator"]), default="tpu")
    output_step = loading_probe.run_step(
        "kaggle_kernel_output",
        [
            "kaggle",
            "kernels",
            "output",
            kernel_ref,
            "-p",
            str(stage_output),
            "--force",
            "--file-pattern",
            str(package["report_filename"]),
        ],
        runner=runner,
        timeout_seconds=args.kaggle_output_timeout_seconds,
    )
    steps.append(output_step)
    if not args.skip_kaggle_cleanup:
        print(f"[{utc_now()}] deleting private Kaggle TPU kernel {kernel_ref}", flush=True)
        delete_step = loading_probe.run_step(
            "kaggle_kernel_delete",
            ["kaggle", "kernels", "delete", kernel_ref, "-y"],
            runner=runner,
            timeout_seconds=args.kaggle_delete_timeout_seconds,
        )
        steps.append(delete_step)
    if not args.keep_kaggle_logs:
        for log_path in stage_output.glob("*.log"):
            log_path.unlink(missing_ok=True)
    report_path = stage_output / str(package["report_filename"])
    missing_report = missing_runtime_report_if_needed(package=package, stage_output=stage_output, steps=steps)
    if missing_report is not None:
        return missing_report, steps
    return load_json(report_path), steps


def build_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    accelerator_attempts: list[dict[str, Any]],
    selected_report: dict[str, Any],
) -> dict[str, Any]:
    accepted = [attempt for attempt in accelerator_attempts if any(step.get("name") == "kaggle_kernel_push" and step.get("accepted") for step in attempt.get("steps", []))]
    kernels_deleted = all(
        any(step.get("name") == "kaggle_kernel_delete" and step.get("ok") for step in attempt.get("steps", []))
        for attempt in accepted
    ) if accepted else False
    private_packages_removed = not (output_dir / "private-kaggle-tpu-kernels").exists()
    ready = bool(
        selected_report.get("ok") is True
        and selected_report.get("llm_inference_ready") is True
        and kernels_deleted
        and private_packages_removed
    )
    blockers: list[str] = []
    selected_blockers = list(selected_report.get("blockers") or []) if isinstance(selected_report.get("blockers"), list) else []
    if not accepted:
        blockers.append("kaggle_tpu_accelerator_not_accepted")
    if accepted and selected_report.get("llm_inference_ready") is not True:
        blockers.extend(selected_blockers or ["kaggle_tpu_llm_inference_not_ready"])
    if not kernels_deleted:
        blockers.append("kaggle_tpu_kernel_cleanup_not_verified")
    if not private_packages_removed:
        blockers.append("kaggle_tpu_private_package_retained")
    diagnosis = [
        "kaggle_tpu_llm_probe_ready" if ready else "kaggle_tpu_llm_probe_not_ready",
        "kaggle_tpu_accelerator_accepted" if accepted else "kaggle_tpu_accelerator_not_accepted",
        "kaggle_tpu_kernel_deleted" if kernels_deleted else "kaggle_tpu_kernel_cleanup_not_verified",
    ]
    if selected_report.get("llm_inference_ready") is True:
        diagnosis.append("kaggle_tpu_synthetic_llm_ready")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "fresh_kaggle_run_performed": bool(accepted),
        "requested_accelerators": parse_accelerators(args.accelerators),
        "selected_accelerator": selected_report.get("requested_accelerator") or "",
        "tpu_runtime_ready": bool(selected_report.get("tpu_runtime_ready")),
        "llm_inference_ready": bool(selected_report.get("llm_inference_ready")),
        "synthetic_llm_runtime": (selected_report.get("jax") or {}).get("synthetic_llm_runtime") if isinstance(selected_report.get("jax"), dict) else "",
        "generated_token_count": int((selected_report.get("jax") or {}).get("generated_token_count") or 0) if isinstance(selected_report.get("jax"), dict) else 0,
        "blockers": sorted(set(blockers)),
        "blocked_reason": "" if ready else (blockers[0] if blockers else "kaggle_tpu_llm_probe_not_ready"),
        "diagnosis_codes": sorted(set(diagnosis + list(selected_report.get("diagnosis_codes") or []))),
        "runtime_report": selected_report,
        "accelerator_attempts": [
            {
                **attempt,
                "steps": [public_step(step) for step in list(attempt.get("steps") or []) if isinstance(step, dict)],
            }
            for attempt in accelerator_attempts
        ],
        "kaggle_lifecycle": {
            "actual_push_count": len(accepted),
            "kernels_deleted": kernels_deleted,
            "private_packages_removed": private_packages_removed,
        },
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "credentials_public": False,
            "private_kernel_payload_public": False,
        },
        "limitations": [
            "This proves a minimal JAX synthetic causal-LM inference path on a Kaggle TPU accelerator, not full Hugging Face Qwen deployment.",
            "PyTorch CUDA stage runtimes are not automatically portable to TPU; production TPU support needs a JAX/Flax, TensorFlow, torch_xla, MaxText, or vLLM-TPU backend.",
            "This is not a large-model capacity proof, not production serving, and not P2P routing.",
        ],
    }


def run_probe(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    selected_report: dict[str, Any] = {}
    try:
        for accelerator in parse_accelerators(args.accelerators):
            package = build_package(args, output_dir=output_dir, accelerator=accelerator)
            report, steps = run_package(args, package=package, output_dir=output_dir, runner=runner)
            attempts.append({
                "accelerator": accelerator,
                "kernel_ref": package.get("kernel_ref"),
                "accepted": any(step.get("name") == "kaggle_kernel_push" and step.get("accepted") for step in steps),
                "steps": steps,
                "runtime_ok": report.get("ok") is True,
                "llm_inference_ready": report.get("llm_inference_ready") is True,
            })
            if report.get("llm_inference_ready") is True:
                selected_report = report
                break
            if any(step.get("name") == "kaggle_kernel_push" and step.get("accepted") for step in steps):
                selected_report = report
                break
    finally:
        if not args.keep_private_package:
            shutil.rmtree(output_dir / "private-kaggle-tpu-kernels", ignore_errors=True)
    final = build_report(args, output_dir=output_dir, accelerator_attempts=attempts, selected_report=selected_report)
    write_json(output_dir / "kaggle_tpu_llm_probe.json", final)
    return final


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded Kaggle TPU LLM runtime probe.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-owner", default=loading_probe.default_kaggle_owner())
    parser.add_argument("--kernel-slug-prefix", default="cttpu-llm")
    parser.add_argument("--accelerators", default=DEFAULT_ACCELERATORS)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=900)
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=60.0)
    parser.add_argument("--kaggle-output-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--keep-private-package", action="store_true")
    parser.add_argument("--keep-kaggle-logs", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.kernel_timeout_seconds > 3600:
        raise SystemExit("--kernel-timeout-seconds must be <= 3600")
    if args.kaggle_status_timeout_seconds > 4500:
        raise SystemExit("--kaggle-status-timeout-seconds must be <= 4500")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"{SCHEMA}: ok={bool(report.get('ok'))} "
            f"accelerator={report.get('selected_accelerator') or 'none'} "
            f"llm={bool(report.get('llm_inference_ready'))} "
            f"blocked={report.get('blocked_reason') or 'none'}"
        )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
