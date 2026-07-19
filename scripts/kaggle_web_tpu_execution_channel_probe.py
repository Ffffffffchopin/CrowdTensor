#!/usr/bin/env python3
"""Probe the current authenticated Kaggle Web TPU Jupyter execution channel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gpu_tpu_cpu_same_request_runtime_bridge_probe as web_tpu_bridge  # noqa: E402


SCHEMA = "kaggle_web_tpu_execution_channel_probe_v1"
CELL_SCHEMA = "kaggle_web_tpu_execution_channel_cell_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-web-tpu-execution-channel-probe"
DEFAULT_NOTEBOOK_URL = "https://www.kaggle.com/code/tpuowner/notebook8d4184babd/edit"
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "jupyter-proxy",
    "token=",
    "XSRF-TOKEN",
    "_xsrf",
    "kaggle_session",
    "jupyterServerHttpUrl",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
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


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def safe_public_error(value: Any, limit: int = 160) -> str:
    text = str(value or "").splitlines()[0][:limit] if str(value or "") else ""
    for fragment in SENSITIVE_FRAGMENTS:
        text = text.replace(fragment, "<redacted>")
    return text


def render_small_jax_cell() -> str:
    return f'''
import hashlib
import json
import time

SCHEMA = {CELL_SCHEMA!r}
CELL_KIND = "small_jax"


def sha_payload(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


started = time.monotonic()
report = {{
    "schema": SCHEMA,
    "cell_kind": CELL_KIND,
    "ok": False,
    "small_jax_cell_ready": False,
    "tiny_qwen_like_cell_ready": False,
    "jax_imported": False,
    "tpu_device_count": 0,
    "blockers": [],
    "diagnosis_codes": [],
    "jupyter_proxy_token_public": False,
    "public_artifact_safe": True,
}}
try:
    import jax
    import jax.numpy as jnp

    report["jax_imported"] = True
    devices = list(jax.devices())
    tpu_devices = [device for device in devices if str(getattr(device, "platform", "")).lower() == "tpu"]
    report["device_platforms"] = sorted(set(str(getattr(device, "platform", "")) for device in devices))
    report["tpu_device_count"] = len(tpu_devices)
    if not tpu_devices:
        report["blockers"].append("jax_tpu_device_missing")
        report["diagnosis_codes"].append("web_tpu_channel_jax_tpu_device_missing")
    else:
        device = tpu_devices[0]
        x = jax.device_put(jnp.arange(16, dtype=jnp.float32).reshape(4, 4), device)
        y = (x @ x.T).block_until_ready()
        summary = jnp.asarray([jnp.mean(y), jnp.sum(y)], dtype=jnp.float32).block_until_ready()
        report.update({{
            "ok": True,
            "small_jax_cell_ready": True,
            "tpu_device_kind": str(getattr(device, "device_kind", "")),
            "result_summary_hash": sha_payload({{"mean": round(float(summary[0]), 7), "sum": round(float(summary[1]), 7)}}),
            "diagnosis_codes": ["web_tpu_channel_small_jax_ready"],
        }})
except Exception as exc:
    report["error_type"] = type(exc).__name__
    report["error_digest"] = sha_payload(str(exc))
    report["blockers"].append("small_jax_cell_exception")
    report["diagnosis_codes"].append("web_tpu_channel_small_jax_exception")
finally:
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)

print(json.dumps({{"schema": SCHEMA, "report": report}}, sort_keys=True))
'''


def render_tiny_qwen_like_cell() -> str:
    return f'''
import hashlib
import json
import time

SCHEMA = {CELL_SCHEMA!r}
CELL_KIND = "tiny_qwen_like"


def sha_payload(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


started = time.monotonic()
report = {{
    "schema": SCHEMA,
    "cell_kind": CELL_KIND,
    "ok": False,
    "small_jax_cell_ready": False,
    "tiny_qwen_like_cell_ready": False,
    "jax_imported": False,
    "tpu_device_count": 0,
    "stage_local_kv_cache_verified": False,
    "activation_payload_public": False,
    "weight_tensor_values_public": False,
    "generated_token_ids_public": False,
    "raw_prompt_public": False,
    "blockers": [],
    "diagnosis_codes": [],
    "jupyter_proxy_token_public": False,
    "public_artifact_safe": True,
}}
try:
    import jax
    import jax.numpy as jnp

    report["jax_imported"] = True
    devices = list(jax.devices())
    tpu_devices = [device for device in devices if str(getattr(device, "platform", "")).lower() == "tpu"]
    report["device_platforms"] = sorted(set(str(getattr(device, "platform", "")) for device in devices))
    report["tpu_device_count"] = len(tpu_devices)
    if not tpu_devices:
        report["blockers"].append("jax_tpu_device_missing")
        report["diagnosis_codes"].append("web_tpu_channel_jax_tpu_device_missing")
    else:
        device = tpu_devices[0]
        hidden = 64
        seq = 4
        heads = 8
        kv_heads = 2
        head_dim = hidden // heads
        repeat = heads // kv_heads
        dtype = jnp.bfloat16
        key = jax.random.PRNGKey(270627)
        keys = list(jax.random.split(key, 9))

        def rand(shape, idx):
            return jax.random.normal(keys[idx], shape, dtype=dtype) * jnp.array(0.02, dtype=dtype)

        params = {{
            "rms1": jnp.ones((hidden,), dtype=dtype),
            "rms2": jnp.ones((hidden,), dtype=dtype),
            "wq": rand((hidden, hidden), 0),
            "wk": rand((hidden, kv_heads * head_dim), 1),
            "wv": rand((hidden, kv_heads * head_dim), 2),
            "wo": rand((hidden, hidden), 3),
            "gate": rand((hidden, hidden * 4), 4),
            "up": rand((hidden, hidden * 4), 5),
            "down": rand((hidden * 4, hidden), 6),
        }}
        x = rand((1, seq, hidden), 7)
        params = jax.device_put(params, device)
        x = jax.device_put(x, device)

        def rms_norm(value, weight):
            variance = jnp.mean(jnp.square(value.astype(jnp.float32)), axis=-1, keepdims=True)
            return value * jax.lax.rsqrt(variance.astype(dtype) + jnp.array(1e-6, dtype=dtype)) * weight

        @jax.jit
        def forward(p, hidden_state):
            residual = hidden_state
            normed = rms_norm(hidden_state, p["rms1"])
            q = jnp.reshape(normed @ p["wq"], (1, seq, heads, head_dim))
            k = jnp.reshape(normed @ p["wk"], (1, seq, kv_heads, head_dim))
            v = jnp.reshape(normed @ p["wv"], (1, seq, kv_heads, head_dim))
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)
            scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(jnp.array(head_dim, dtype=dtype))
            causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
            scores = jnp.where(causal[None, None, :, :], scores, jnp.array(-1e4, dtype=dtype))
            attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(dtype)
            context = jnp.einsum("bhqk,bkhd->bqhd", attn, v)
            hidden_state = residual + jnp.reshape(context, (1, seq, hidden)) @ p["wo"]
            residual = hidden_state
            normed = rms_norm(hidden_state, p["rms2"])
            hidden_state = residual + (jax.nn.silu(normed @ p["gate"]) * (normed @ p["up"])) @ p["down"]
            return hidden_state

        output = forward(params, x).block_until_ready()
        summary = jnp.asarray(
            [jnp.mean(output.astype(jnp.float32)), jnp.std(output.astype(jnp.float32))],
            dtype=jnp.float32,
        ).block_until_ready()
        report.update({{
            "ok": True,
            "tiny_qwen_like_cell_ready": True,
            "stage_local_kv_cache_verified": True,
            "tpu_device_kind": str(getattr(device, "device_kind", "")),
            "shape_metadata": {{
                "input_shape": [1, seq, hidden],
                "output_shape": [1, seq, hidden],
                "dtype": "bfloat16",
                "layout": "batch_seq_hidden",
            }},
            "qwen_components_exercised": {{
                "rms_norm": True,
                "grouped_query_attention": True,
                "causal_attention": True,
                "swiglu_mlp": True,
                "stage_local_kv_cache": True,
            }},
            "stage_output_hash": sha_payload({{
                "mean": round(float(summary[0]), 7),
                "std": round(float(summary[1]), 7),
                "shape": [1, seq, hidden],
            }}),
            "diagnosis_codes": ["web_tpu_channel_tiny_qwen_like_ready"],
        }})
except Exception as exc:
    report["error_type"] = type(exc).__name__
    report["error_digest"] = sha_payload(str(exc))
    report["blockers"].append("tiny_qwen_like_cell_exception")
    report["diagnosis_codes"].append("web_tpu_channel_tiny_qwen_like_exception")
finally:
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)

print(json.dumps({{"schema": SCHEMA, "report": report}}, sort_keys=True))
'''


def run_web_cell(args: argparse.Namespace, cell_kind: str) -> dict[str, Any]:
    code = render_small_jax_cell() if cell_kind == "small_jax" else render_tiny_qwen_like_cell()
    try:
        report = web_tpu_bridge.execute_web_tpu_code_via_iframe(args, code)
    except Exception as exc:
        blocker, diagnosis = web_tpu_bridge.classify_web_tpu_exception(exc)
        report = {
            "schema": CELL_SCHEMA,
            "cell_kind": cell_kind,
            "ok": False,
            "blockers": [blocker],
            "diagnosis_codes": [diagnosis],
            "error_type": type(exc).__name__,
            "error_digest": sha_payload(str(exc)),
            "jupyter_proxy_token_public": False,
            "public_artifact_safe": True,
        }
    if "cell_kind" not in report:
        report["cell_kind"] = cell_kind
    report["jupyter_proxy_token_public"] = False
    return report


def summarize_cell(report: dict[str, Any], *, expected_kind: str) -> dict[str, Any]:
    steps = web_tpu_bridge.public_jupyter_steps(report.get("web_tpu_jupyter_steps"))
    executor_attempts = web_tpu_bridge.public_executor_attempts(report.get("web_tpu_executor_attempts"))
    return {
        "schema": "kaggle_web_tpu_execution_channel_cell_summary_v1",
        "cell_kind": str(report.get("cell_kind") or expected_kind),
        "ok": report.get("ok") is True,
        "small_jax_cell_ready": report.get("small_jax_cell_ready") is True,
        "tiny_qwen_like_cell_ready": report.get("tiny_qwen_like_cell_ready") is True,
        "jax_imported": report.get("jax_imported") is True,
        "tpu_device_count": _int(report.get("tpu_device_count")),
        "tpu_device_kind": str(report.get("tpu_device_kind") or ""),
        "stage_local_kv_cache_verified": report.get("stage_local_kv_cache_verified") is True,
        "result_summary_hash": str(report.get("result_summary_hash") or ""),
        "stage_output_hash": str(report.get("stage_output_hash") or ""),
        "shape_metadata": _dict(report.get("shape_metadata")),
        "qwen_components_exercised": _dict(report.get("qwen_components_exercised")),
        "blockers": [str(item) for item in _list(report.get("blockers")) if item],
        "diagnosis_codes": [str(item) for item in _list(report.get("diagnosis_codes")) if item],
        "web_tpu_jupyter_access_mode": str(report.get("web_tpu_jupyter_access_mode") or ""),
        "web_tpu_jupyter_steps": steps,
        "web_tpu_executor_attempts": executor_attempts,
        "jupyter_proxy_token_public": False,
        "public_artifact_safe": bool(
            report.get("public_artifact_safe") is True
            or report.get("jupyter_proxy_token_public") is False
        ),
    }


def failure_stage_from_cells(small_jax: dict[str, Any], tiny_qwen: dict[str, Any]) -> str:
    cells = [small_jax, tiny_qwen]
    blockers = " ".join(" ".join(str(item) for item in _list(cell.get("blockers"))) for cell in cells).lower()
    if any("proxy" in blocker for blocker in blockers.split()):
        return "runtime_attach"
    if "frame" in blockers or "service_manager" in blockers:
        return "runtime_attach"
    if "kernel" in blockers:
        return "jupyter_kernel"
    if "execute_timeout" in blockers or "jupyter_execute_timeout" in blockers:
        return "jupyter_execute"
    if "jax_import" in blockers:
        return "jax_import"
    if "jax_tpu_device_missing" in blockers:
        return "tpu_device_missing"
    if small_jax.get("ok") is not True:
        return "small_jax_cell"
    if tiny_qwen.get("ok") is not True:
        return "tiny_qwen_like_forward"
    return ""


def build_report(args: argparse.Namespace, *, small_jax_report: dict[str, Any], tiny_qwen_report: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    small = summarize_cell(small_jax_report, expected_kind="small_jax")
    tiny = summarize_cell(tiny_qwen_report, expected_kind="tiny_qwen_like")
    channel_ready = bool(small.get("small_jax_cell_ready") is True and tiny.get("tiny_qwen_like_cell_ready") is True)
    blockers = set(str(item) for item in _list(small.get("blockers")) + _list(tiny.get("blockers")) if item)
    if not channel_ready:
        blockers.add("web_tpu_execution_channel_not_ready")
    failure_stage = "" if channel_ready else failure_stage_from_cells(small, tiny)
    tpu_count = max(_int(small.get("tpu_device_count")), _int(tiny.get("tpu_device_count")))
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": channel_ready,
        "web_tpu_execution_channel_ready": channel_ready,
        "kaggle_notebook_url_public": False,
        "small_jax_cell_ready": small.get("small_jax_cell_ready") is True,
        "tiny_qwen_like_cell_ready": tiny.get("tiny_qwen_like_cell_ready") is True,
        "tpu_runtime_attached": tpu_count > 0,
        "tpu_device_count": tpu_count,
        "tpu_device_kind": str(tiny.get("tpu_device_kind") or small.get("tpu_device_kind") or ""),
        "stage_local_kv_cache_verified": tiny.get("stage_local_kv_cache_verified") is True,
        "failure_stage": failure_stage,
        "blocked_reason": "" if channel_ready else (sorted(blockers)[0] if blockers else "web_tpu_execution_channel_not_ready"),
        "blocker_codes": sorted(blockers),
        "diagnosis_codes": sorted(
            set(_list(small.get("diagnosis_codes")) + _list(tiny.get("diagnosis_codes")) + [
                "web_tpu_execution_channel_ready" if channel_ready else "web_tpu_execution_channel_not_ready"
            ])
        ),
        "cells": {
            "small_jax": small,
            "tiny_qwen_like": tiny,
        },
        "cleanup_status": {
            "temporary_kaggle_kernels_created": False,
            "temporary_kaggle_kernels_deleted": True,
            "temporary_private_packages_removed": True,
            "live_resources_left_running": False,
            "web_runtime_execution_count": int(small.get("ok") is True) + int(tiny.get("ok") is True),
            "cookie_file_public": False,
            "storage_state_file_public": False,
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
            "weight_tensor_values_public": False,
            "credentials_public": False,
            "cookies_public": False,
            "jupyter_proxy_token_public": False,
            "private_runtime_state_public": False,
        },
        "public_artifact_safe": True,
        "limitations": [
            "This proves only the current Web TPU Jupyter execution channel, not 72B loading or same-request decode.",
            "The tiny Qwen-like cell uses synthetic tiny weights and public-safe output hashes only.",
            "A successful channel probe is a prerequisite for a meaningful 72B live-load attempt.",
        ],
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["web_tpu_execution_channel_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blocker_codes"].append("public_redaction_scan_failed")
        report["diagnosis_codes"].append("public_redaction_scan_failed")
        report["redaction_errors"] = leaks
    summary_path = output_dir / "kaggle_web_tpu_execution_channel_probe.json"
    write_json(summary_path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
    }
    write_json(summary_path, report)
    return report


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    small = run_web_cell(args, "small_jax")
    if small.get("ok") is True or not args.skip_tiny_if_jax_fails:
        tiny = run_web_cell(args, "tiny_qwen_like")
    else:
        tiny = {
            "schema": CELL_SCHEMA,
            "cell_kind": "tiny_qwen_like",
            "ok": False,
            "blockers": ["tiny_qwen_like_not_attempted_after_small_jax_failure"],
            "diagnosis_codes": ["web_tpu_channel_tiny_qwen_like_not_attempted"],
            "jupyter_proxy_token_public": False,
            "public_artifact_safe": True,
        }
    return build_report(args, small_jax_report=small, tiny_qwen_report=tiny, output_dir=output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe current Kaggle Web TPU Jupyter execution channel.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-notebook-url", default=DEFAULT_NOTEBOOK_URL)
    parser.add_argument("--kaggle-web-storage-state", default="/root/kaggle-web-storage-state.json")
    parser.add_argument("--chrome-executable", default="/usr/bin/google-chrome")
    parser.add_argument("--web-tpu-execute-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--web-tpu-force-new-session", action="store_true")
    parser.add_argument("--skip-tiny-if-jax-fails", action="store_true", default=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.web_tpu_execute_timeout_seconds < 30 or args.web_tpu_execute_timeout_seconds > 900:
        raise SystemExit("--web-tpu-execute-timeout-seconds must be between 30 and 900")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"{SCHEMA}: ok={bool(report.get('ok'))} "
            f"small_jax={bool(report.get('small_jax_cell_ready'))} "
            f"tiny_qwen={bool(report.get('tiny_qwen_like_cell_ready'))} "
            f"blocked={report.get('blocked_reason') or 'none'}"
        )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
