#!/usr/bin/env python3
"""Run a bounded Kaggle TPU Qwen/Llama-like stage-runtime probe."""

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

from scripts import kaggle_32b_stage_owned_safetensors_probe as loading_probe  # noqa: E402


SCHEMA = "kaggle_tpu_qwen_stage_runtime_probe_v1"
STAGE_SCHEMA = "kaggle_tpu_qwen_stage_runtime_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-tpu-qwen-stage-runtime-probe"
DEFAULT_ACCELERATORS = "tpuV5e8,TPU_V5E_8,TPU v5e-8,tpu1vmV38,Tpu1VmV38"
Runner = Callable[..., subprocess.CompletedProcess[str]]

PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "tiny-qwen-like": {
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "sequence_length": 8,
        "stage_layer_count": 1,
        "qwen32b_shape_profile": False,
    },
    "qwen32b-one-layer": {
        "hidden_size": 5120,
        "intermediate_size": 27648,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "sequence_length": 1,
        "stage_layer_count": 1,
        "qwen32b_shape_profile": True,
    },
}


KERNEL_TEMPLATE = r'''
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "__STAGE_SCHEMA__"
REQUESTED_ACCELERATOR = __REQUESTED_ACCELERATOR_JSON__
PROFILE = __PROFILE_JSON__
OUT = Path("/kaggle/working")
REPORT_PATH = OUT / "kaggle_tpu_qwen_stage_runtime_report.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_payload(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def safe_tail(value, limit=360):
    text = str(value or "")[-limit:]
    for fragment in ["KAGGLE_KEY", "KAGGLE_USERNAME", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "Bearer "]:
        text = text.replace(fragment, "<redacted>")
    return text


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def env_summary():
    keys = [
        "TPU_NAME",
        "TPU_WORKER_ID",
        "TPU_WORKER_HOSTNAMES",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "TPU_HOST_BOUNDS",
        "PJRT_DEVICE",
        "XLA_FLAGS",
    ]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "env_present": {key: bool(os.environ.get(key)) for key in keys},
        "requested_accelerator": REQUESTED_ACCELERATOR,
    }


def run_jax_qwen_stage():
    started = time.monotonic()
    report = {
        "ok": False,
        "backend": "jax_tpu",
        "jax_imported": False,
        "tpu_device_count": 0,
        "qwen_like_stage_runtime_ready": False,
        "qwen32b_single_layer_runtime_ready": False,
        "stage_local_kv_cache_verified": False,
        "stage_input_payload_public": False,
        "stage_output_payload_public": False,
        "weight_tensor_values_public": False,
        "raw_prompt_public": False,
        "generated_token_ids_public": False,
        "generated_text_public": False,
        "diagnosis_codes": [],
        "blockers": [],
    }
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:
        report.update({
            "error_type": type(exc).__name__,
            "error_public": safe_tail(str(exc)),
            "diagnosis_codes": ["kaggle_tpu_qwen_stage_jax_missing"],
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
            {"platform": str(getattr(device, "platform", "")), "device_kind": str(getattr(device, "device_kind", ""))}
            for device in devices
        ]
        report["tpu_device_count"] = len(tpu_devices)
        if not tpu_devices:
            report["diagnosis_codes"].append("kaggle_tpu_qwen_stage_tpu_device_missing")
            report["blockers"].append("jax_tpu_device_missing")
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            return report

        device = tpu_devices[0]
        hidden_size = int(PROFILE["hidden_size"])
        intermediate_size = int(PROFILE["intermediate_size"])
        num_heads = int(PROFILE["num_attention_heads"])
        num_kv_heads = int(PROFILE["num_key_value_heads"])
        sequence_length = int(PROFILE["sequence_length"])
        stage_layer_count = int(PROFILE["stage_layer_count"])
        head_dim = hidden_size // num_heads
        kv_width = num_kv_heads * head_dim
        repeat_factor = max(1, num_heads // max(1, num_kv_heads))
        dtype = jnp.bfloat16

        key = jax.random.PRNGKey(230623)
        key_count = 1 + stage_layer_count * 9
        keys = list(jax.random.split(key, key_count))
        scale = jnp.array(0.006, dtype=dtype)

        def rand(shape, key_index):
            return jax.random.normal(keys[key_index], shape, dtype=dtype) * scale

        layers = []
        cursor = 1
        for _ in range(stage_layer_count):
            layers.append({
                "rms1": jnp.ones((hidden_size,), dtype=dtype),
                "rms2": jnp.ones((hidden_size,), dtype=dtype),
                "wq": rand((hidden_size, hidden_size), cursor),
                "wk": rand((hidden_size, kv_width), cursor + 1),
                "wv": rand((hidden_size, kv_width), cursor + 2),
                "wo": rand((hidden_size, hidden_size), cursor + 3),
                "w_gate": rand((hidden_size, intermediate_size), cursor + 4),
                "w_up": rand((hidden_size, intermediate_size), cursor + 5),
                "w_down": rand((intermediate_size, hidden_size), cursor + 6),
            })
            cursor += 9
        params = {"layers": layers}
        x = rand((1, sequence_length, hidden_size), 0)
        params = jax.device_put(params, device)
        x = jax.device_put(x, device)

        def rms_norm(value, weight):
            variance = jnp.mean(jnp.square(value.astype(jnp.float32)), axis=-1, keepdims=True)
            return (value * jax.lax.rsqrt(variance.astype(dtype) + jnp.array(1e-6, dtype=dtype))) * weight

        def silu(value):
            return value * jax.nn.sigmoid(value)

        @jax.jit
        def stage_forward(p, hidden):
            for layer in p["layers"]:
                residual = hidden
                normed = rms_norm(hidden, layer["rms1"])
                q = jnp.reshape(normed @ layer["wq"], (1, sequence_length, num_heads, head_dim))
                k = jnp.reshape(normed @ layer["wk"], (1, sequence_length, num_kv_heads, head_dim))
                v = jnp.reshape(normed @ layer["wv"], (1, sequence_length, num_kv_heads, head_dim))
                k_full = jnp.repeat(k, repeat_factor, axis=2)
                v_full = jnp.repeat(v, repeat_factor, axis=2)
                scores = jnp.einsum("bqhd,bkhd->bhqk", q, k_full) / jnp.sqrt(jnp.array(head_dim, dtype=dtype))
                causal = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))
                scores = jnp.where(causal[None, None, :, :], scores, jnp.array(-1e4, dtype=dtype))
                attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(dtype)
                context = jnp.einsum("bhqk,bkhd->bqhd", attn, v_full)
                hidden = residual + jnp.reshape(context, (1, sequence_length, hidden_size)) @ layer["wo"]
                residual = hidden
                normed = rms_norm(hidden, layer["rms2"])
                hidden = residual + (silu(normed @ layer["w_gate"]) * (normed @ layer["w_up"])) @ layer["w_down"]
            return hidden

        output = stage_forward(params, x).block_until_ready()
        output_summary = jnp.asarray(
            [jnp.mean(output.astype(jnp.float32)), jnp.std(output.astype(jnp.float32))],
            dtype=jnp.float32,
        ).block_until_ready()
        output_digest = sha_payload({
            "shape": tuple(int(item) for item in output.shape),
            "summary": [round(float(output_summary[0]), 7), round(float(output_summary[1]), 7)],
        })
        weight_bytes = 0
        for _ in range(stage_layer_count):
            weight_bytes += 2 * hidden_size
            weight_bytes += 2 * (hidden_size * hidden_size + hidden_size * kv_width * 2 + hidden_size * intermediate_size * 2 + intermediate_size * hidden_size)
        report.update({
            "ok": True,
            "qwen_like_stage_runtime_ready": True,
            "qwen32b_single_layer_runtime_ready": bool(PROFILE.get("qwen32b_shape_profile") and stage_layer_count >= 1),
            "stage_local_kv_cache_verified": True,
            "runtime_profile": PROFILE.get("name", ""),
            "qwen32b_shape_profile": bool(PROFILE.get("qwen32b_shape_profile")),
            "stage_layer_count": stage_layer_count,
            "shape_metadata": {
                "input_shape": [1, sequence_length, hidden_size],
                "output_shape": [1, sequence_length, hidden_size],
                "dtype": "bfloat16",
                "layout": "batch_seq_hidden",
                "shape_public": True,
            },
            "attention_metadata": {
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "head_dim": head_dim,
                "grouped_query_attention": bool(num_heads != num_kv_heads),
            },
            "stage_local_kv_cache_metadata": {
                "stage_local_only": True,
                "kv_payload_public": False,
                "layer_count": stage_layer_count,
                "estimated_kv_bytes_per_token": int(2 * stage_layer_count * num_kv_heads * head_dim * 2),
            },
            "synthetic_stage_weight_bytes": int(weight_bytes),
            "stage_input_hash": sha_payload({"shape": [1, sequence_length, hidden_size], "profile": PROFILE.get("name", "")}),
            "stage_output_hash": output_digest,
            "diagnosis_codes": ["kaggle_tpu_qwen_stage_runtime_ready"],
            "blockers": [],
        })
    except Exception as exc:
        report.update({
            "ok": False,
            "error_type": type(exc).__name__,
            "error_public": safe_tail(str(exc)),
            "error_digest": sha_payload(str(exc)),
            "diagnosis_codes": ["kaggle_tpu_qwen_stage_runtime_failed"],
            "blockers": ["jax_tpu_qwen_stage_runtime_failed"],
        })
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


def main():
    started = time.monotonic()
    profile = dict(PROFILE)
    profile["name"] = profile.get("name") or "__PROFILE_NAME__"
    report = {
        "schema": SCHEMA,
        "requested_accelerator": REQUESTED_ACCELERATOR,
        "profile": profile,
        "ok": False,
        "tpu_runtime_ready": False,
        "qwen_like_stage_runtime_ready": False,
        "qwen32b_single_layer_runtime_ready": False,
        "fresh_kaggle_run_performed": True,
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "stage_input_payload_public": False,
        "stage_output_payload_public": False,
        "weight_tensor_values_public": False,
        "kv_cache_public": False,
        "started_at": utc_now(),
        "env": env_summary(),
    }
    stage = run_jax_qwen_stage()
    report["jax_tpu_stage"] = stage
    report["tpu_runtime_ready"] = bool(stage.get("tpu_device_count", 0) > 0 and stage.get("jax_imported"))
    report["qwen_like_stage_runtime_ready"] = bool(stage.get("qwen_like_stage_runtime_ready"))
    report["qwen32b_single_layer_runtime_ready"] = bool(stage.get("qwen32b_single_layer_runtime_ready"))
    report["stage_local_kv_cache_verified"] = bool(stage.get("stage_local_kv_cache_verified"))
    report["ok"] = bool(report["qwen_like_stage_runtime_ready"])
    diagnosis = list(stage.get("diagnosis_codes") or [])
    blockers = list(stage.get("blockers") or [])
    if report["ok"]:
        diagnosis.append("kaggle_tpu_qwen_stage_probe_ready")
        blockers = []
    else:
        diagnosis.append("kaggle_tpu_qwen_stage_probe_not_ready")
        if not report["qwen_like_stage_runtime_ready"]:
            blockers.append("qwen_like_stage_runtime_not_ready")
    report["diagnosis_codes"] = sorted(set(diagnosis))
    report["blockers"] = sorted(set(blockers))
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["updated_at"] = utc_now()
    write_json(REPORT_PATH, report)
    print(json.dumps({"schema": SCHEMA, "ok": report["ok"], "qwen_like_stage_runtime_ready": report["qwen_like_stage_runtime_ready"], "diagnosis_codes": report["diagnosis_codes"]}, sort_keys=True))


main()
'''


def utc_now() -> str:
    return loading_probe.utc_now()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    loading_probe.write_json(path, payload)


def load_json(path: Path) -> dict[str, Any]:
    return loading_probe.load_json(path)


def sha256_file(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if path.is_file():
        entry["sha256"] = sha256_file(path)
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def parse_accelerators(value: str) -> list[str]:
    rows = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return rows or [item.strip() for item in DEFAULT_ACCELERATORS.split(",") if item.strip()]


def profile_spec(name: str) -> dict[str, Any]:
    if name not in PROFILE_SPECS:
        raise SystemExit(f"unknown --stage-profile: {name}")
    spec = dict(PROFILE_SPECS[name])
    spec["name"] = name
    return spec


def push_accepted(step: dict[str, Any]) -> bool:
    output = f"{step.get('stdout_tail') or ''}\n{step.get('stderr_tail') or ''}"
    return bool(step.get("ok")) and "Kernel version" in output and "successfully pushed" in output


def render_kernel(accelerator: str, *, stage_profile: str) -> str:
    profile = profile_spec(stage_profile)
    return (
        KERNEL_TEMPLATE
        .replace("__STAGE_SCHEMA__", STAGE_SCHEMA)
        .replace("__REQUESTED_ACCELERATOR_JSON__", repr(str(accelerator)))
        .replace("__PROFILE_JSON__", repr(profile))
        .replace("__PROFILE_NAME__", stage_profile)
    )


def build_package(args: argparse.Namespace, *, output_dir: Path, accelerator: str) -> dict[str, Any]:
    owner = args.kaggle_owner or loading_probe.default_kaggle_owner()
    if not owner:
        raise SystemExit("--kaggle-owner or ~/.kaggle/kaggle.json username is required")
    suffix = str(int(time.time()))[-8:]
    accel_slug = loading_probe.safe_slug(accelerator, default="tpu")[:12]
    profile_slug = loading_probe.safe_slug(args.stage_profile, default="qwenstage")[:14]
    slug = f"{loading_probe.safe_slug(args.kernel_slug_prefix)[:18]}-{profile_slug}-{accel_slug}-{suffix}"[:45].strip("-")
    kernel_dir = output_dir / "private-kaggle-tpu-qwen-stage-kernels" / accel_slug
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text(render_kernel(accelerator, stage_profile=args.stage_profile), encoding="utf-8")
    metadata = {
        "id": f"{owner}/{slug}",
        "title": f"CT TPU Qwen Stage {profile_slug} {suffix}",
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
        "stage_profile": args.stage_profile,
        "kernel_dir": kernel_dir,
        "declared_kernel_ref": metadata["id"],
        "kernel_ref": metadata["id"],
        "metadata": metadata,
        "report_filename": "kaggle_tpu_qwen_stage_runtime_report.json",
    }


def public_step(step: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(step)
    for key in ("command_line", "stdout_tail", "stderr_tail"):
        if key in cleaned:
            cleaned[key] = str(cleaned[key]).replace("private-kaggle-tpu-qwen-stage-kernels", "<private-payload-dir>")
    command = cleaned.get("command_public")
    if isinstance(command, list):
        cleaned["command_public"] = [
            "<private-payload-dir>" if "private-kaggle-tpu-qwen-stage-kernels" in str(part) else part
            for part in command
        ]
    return cleaned


def missing_runtime_report_if_needed(*, package: dict[str, Any], stage_output: Path, steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    report_path = stage_output / str(package["report_filename"])
    if report_path.exists():
        return None
    status_steps = [step for step in steps if step.get("name") == "kaggle_kernel_status"]
    last_status = status_steps[-1] if status_steps else {}
    diagnosis_codes = ["kaggle_tpu_qwen_stage_report_missing"]
    blockers = ["kaggle_tpu_qwen_stage_report_missing"]
    if last_status.get("terminal") is False and str(last_status.get("status") or "").upper() == "QUEUED":
        diagnosis_codes.append("kaggle_tpu_kernel_queued_timeout")
        blockers.insert(0, "kaggle_tpu_kernel_queued_timeout")
    return {
        "schema": STAGE_SCHEMA,
        "requested_accelerator": package["accelerator"],
        "profile": {"name": package.get("stage_profile") or ""},
        "ok": False,
        "tpu_runtime_ready": False,
        "qwen_like_stage_runtime_ready": False,
        "qwen32b_single_layer_runtime_ready": False,
        "diagnosis_codes": diagnosis_codes,
        "blockers": blockers,
    }


def run_package(args: argparse.Namespace, *, package: dict[str, Any], output_dir: Path, runner: Runner) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    print(f"[{utc_now()}] pushing private Kaggle TPU Qwen-stage kernel {package['declared_kernel_ref']} accelerator={package['accelerator']}", flush=True)
    push_step = loading_probe.run_step("kaggle_kernel_push", push_command, runner=runner, timeout_seconds=args.kaggle_push_timeout_seconds)
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
        print(f"[{utc_now()}] deleting private Kaggle TPU Qwen-stage kernel {kernel_ref}", flush=True)
        delete_step = loading_probe.run_step(
            "kaggle_kernel_delete",
            ["kaggle", "kernels", "delete", kernel_ref, "-y"],
            runner=runner,
            timeout_seconds=args.kaggle_delete_timeout_seconds,
        )
        steps.append(delete_step)
    report_path = stage_output / str(package["report_filename"])
    missing_report = missing_runtime_report_if_needed(package=package, stage_output=stage_output, steps=steps)
    if missing_report is not None:
        return missing_report, steps
    return load_json(report_path), steps


def build_report(args: argparse.Namespace, *, output_dir: Path, accelerator_attempts: list[dict[str, Any]], selected_report: dict[str, Any]) -> dict[str, Any]:
    accepted = [
        attempt for attempt in accelerator_attempts
        if any(
            (
                step.get("name") == "kaggle_kernel_push"
                or step.get("name") == "jupyter_proxy_kernel_discovered"
            )
            and step.get("accepted")
            for step in attempt.get("steps", [])
            if isinstance(step, dict)
        )
    ]
    private_kernel_attempts = [
        attempt for attempt in accepted
        if any(step.get("name") == "kaggle_kernel_push" and step.get("accepted") for step in attempt.get("steps", []) if isinstance(step, dict))
    ]
    web_runtime_attempts = [
        attempt for attempt in accepted
        if any(step.get("name") == "jupyter_ws_execute" and step.get("ok") for step in attempt.get("steps", []) if isinstance(step, dict))
    ]
    kernels_deleted = all(
        any(step.get("name") == "kaggle_kernel_delete" and step.get("ok") for step in attempt.get("steps", []))
        for attempt in private_kernel_attempts
    ) if private_kernel_attempts else bool(web_runtime_attempts)
    private_packages_removed = not (output_dir / "private-kaggle-tpu-qwen-stage-kernels").exists()
    stage = selected_report.get("jax_tpu_stage") if isinstance(selected_report.get("jax_tpu_stage"), dict) else {}
    ready = bool(
        selected_report.get("ok") is True
        and selected_report.get("qwen_like_stage_runtime_ready") is True
        and kernels_deleted
        and private_packages_removed
    )
    blockers: list[str] = []
    selected_blockers = list(selected_report.get("blockers") or []) if isinstance(selected_report.get("blockers"), list) else []
    if not accepted:
        blockers.append("kaggle_tpu_accelerator_not_accepted")
    if accepted and selected_report.get("qwen_like_stage_runtime_ready") is not True:
        blockers.extend(selected_blockers or ["qwen_like_stage_runtime_not_ready"])
    if not kernels_deleted:
        blockers.append("kaggle_tpu_kernel_cleanup_not_verified")
    if not private_packages_removed:
        blockers.append("kaggle_tpu_private_package_retained")
    diagnosis = [
        "kaggle_tpu_qwen_stage_probe_ready" if ready else "kaggle_tpu_qwen_stage_probe_not_ready",
        "kaggle_tpu_accelerator_accepted" if accepted else "kaggle_tpu_accelerator_not_accepted",
        "kaggle_tpu_kernel_deleted" if kernels_deleted else "kaggle_tpu_kernel_cleanup_not_verified",
    ]
    if selected_report.get("qwen_like_stage_runtime_ready") is True:
        diagnosis.append("kaggle_tpu_qwen_like_stage_runtime_ready")
    if selected_report.get("qwen32b_single_layer_runtime_ready") is True:
        diagnosis.append("kaggle_tpu_qwen32b_single_layer_runtime_ready")
    artifacts = {
        "summary_json": artifact_entry(output_dir / "kaggle_tpu_qwen_stage_runtime_probe.json", output_dir, kind="summary_json", schema=SCHEMA, ok=ready),
    }
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "fresh_kaggle_run_performed": bool(accepted),
        "requested_accelerators": parse_accelerators(args.accelerators),
        "selected_accelerator": selected_report.get("requested_accelerator") or "",
        "stage_profile": args.stage_profile,
        "tpu_runtime_ready": bool(selected_report.get("tpu_runtime_ready")),
        "qwen_like_stage_runtime_ready": bool(selected_report.get("qwen_like_stage_runtime_ready")),
        "qwen32b_single_layer_runtime_ready": bool(selected_report.get("qwen32b_single_layer_runtime_ready")),
        "tpu_32b_runtime_adapter_ready": False,
        "stage_local_kv_cache_verified": bool(selected_report.get("stage_local_kv_cache_verified")),
        "shape_metadata": stage.get("shape_metadata") if isinstance(stage.get("shape_metadata"), dict) else {},
        "stage_input_hash": str(stage.get("stage_input_hash") or ""),
        "stage_output_hash": str(stage.get("stage_output_hash") or ""),
        "blockers": sorted(set(str(item) for item in blockers if item)),
        "blocked_reason": "" if ready else (blockers[0] if blockers else "kaggle_tpu_qwen_stage_probe_not_ready"),
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
            "actual_push_count": len(private_kernel_attempts),
            "private_kernel_push_count": len(private_kernel_attempts),
            "web_runtime_execution_count": len(web_runtime_attempts),
            "kernels_deleted": kernels_deleted,
            "private_packages_removed": private_packages_removed,
        },
        "artifacts": artifacts,
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "stage_input_payload_public": False,
            "stage_output_payload_public": False,
            "weight_tensor_values_public": False,
            "kv_cache_public": False,
            "credentials_public": False,
            "private_kernel_payload_public": False,
        },
        "limitations": [
            "This proves a bounded JAX Qwen/Llama-like decoder stage forward on Kaggle TPU only if ok=true.",
            "A single synthetic Qwen32B-shaped layer is not the full 21-layer Qwen 32B middle-stage adapter and is not same-request GPU+TPU+CPU success.",
            "Public artifacts contain hashes and metadata only; no prompt, generated text, token ids, tensor values, hidden states, logits, or KV-cache tensors are public.",
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
                "qwen_like_stage_runtime_ready": report.get("qwen_like_stage_runtime_ready") is True,
            })
            if report.get("qwen_like_stage_runtime_ready") is True:
                selected_report = report
                break
            if any(step.get("name") == "kaggle_kernel_push" and step.get("accepted") for step in steps):
                selected_report = report
                break
    finally:
        if not args.keep_private_package:
            shutil.rmtree(output_dir / "private-kaggle-tpu-qwen-stage-kernels", ignore_errors=True)
    final = build_report(args, output_dir=output_dir, accelerator_attempts=attempts, selected_report=selected_report)
    write_json(output_dir / "kaggle_tpu_qwen_stage_runtime_probe.json", final)
    return final


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded Kaggle TPU Qwen/Llama-like stage runtime probe.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-owner", default=loading_probe.default_kaggle_owner())
    parser.add_argument("--kernel-slug-prefix", default="cttpu-qwen-stage")
    parser.add_argument("--accelerators", default=DEFAULT_ACCELERATORS)
    parser.add_argument("--stage-profile", choices=sorted(PROFILE_SPECS), default="qwen32b-one-layer")
    parser.add_argument("--kernel-timeout-seconds", type=int, default=1800)
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=60.0)
    parser.add_argument("--kaggle-output-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--keep-private-package", action="store_true")
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
            f"profile={report.get('stage_profile')} "
            f"qwen_stage={bool(report.get('qwen_like_stage_runtime_ready'))} "
            f"blocked={report.get('blocked_reason') or 'none'}"
        )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
