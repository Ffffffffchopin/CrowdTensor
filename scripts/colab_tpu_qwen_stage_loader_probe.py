#!/usr/bin/env python3
"""Run Qwen stage-owned loader code inside a Colab TPU runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import colab_cli_runtime  # noqa: E402
from scripts import kaggle_tpu_32b_stage_owned_loader_probe as loader_probe  # noqa: E402


SCHEMA = "colab_tpu_qwen_stage_loader_probe_v1"


def sha256_short(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def load_session(config: Path, session_name: str) -> dict[str, Any]:
    data = json.loads(config.read_text())
    session = data.get(session_name)
    if not isinstance(session, dict):
        raise SystemExit(f"Session {session_name!r} not found")
    missing = [key for key in ("url", "token", "endpoint") if not session.get(key)]
    if missing:
        raise SystemExit(f"Session {session_name!r} missing {missing}")
    return session


def extract_runtime_report(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    text = "\n".join(str(item.get("text") or "") for item in outputs if isinstance(item, dict))
    for line in text.splitlines()[::-1]:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("report"), dict):
            return dict(parsed["report"])
        if isinstance(parsed, dict) and parsed.get("schema") == loader_probe.SCHEMA:
            return parsed
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="ct-colab-tpu-v5e1")
    parser.add_argument("--config", default=os.path.expanduser("~/.config/colab-cli/sessions.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-repo", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--stage-start", type=int, required=True)
    parser.add_argument("--stage-end", type=int, required=True)
    parser.add_argument("--execute-layer-count", type=int, required=True)
    parser.add_argument("--tensor-key", default="")
    parser.add_argument("--max-header-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--timeout", type=float, default=1500.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    session = load_session(Path(args.config), args.session)
    loader_args = argparse.Namespace(
        model_repo=args.model_repo,
        stage_start=args.stage_start,
        stage_end=args.stage_end,
        tensor_key=args.tensor_key or f"model.layers.{args.stage_start}.input_layernorm.weight",
        max_header_bytes=args.max_header_bytes,
        max_tensor_bytes=args.max_tensor_bytes,
        execute_layer_count=args.execute_layer_count,
        input_activation_private={},
        return_output_activation_private=False,
    )
    runtime = None
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "session_name": args.session,
        "model_repo": args.model_repo,
        "stage_layer_range": [args.stage_start, args.stage_end],
        "execute_layer_count": args.execute_layer_count,
        "public_artifact_safe": True,
        "runtime_proxy_token_public": False,
        "runtime_proxy_url_public": False,
        "endpoint_public": False,
        "endpoint_hash": sha256_short(str(session.get("endpoint") or "")),
        "runtime_proxy_host_hash": sha256_short(urlparse(str(session.get("url") or "")).netloc),
    }
    try:
        ColabRuntime = colab_cli_runtime.load_colab_runtime_class()
        runtime = ColabRuntime(session["url"], session["token"], kernel_id=session.get("kernel_id"), session_id=session.get("session_id"))
        outputs = runtime.execute_code(loader_probe.render_web_probe_code(loader_args), timeout=float(args.timeout))
        runtime_report = extract_runtime_report(outputs)
        if not runtime_report:
            report.update(
                {
                    "ok": False,
                    "blockers": ["colab_loader_report_missing"],
                    "diagnosis_codes": ["colab_loader_report_missing"],
                    "output_hashes": [
                        hashlib.sha256(str(item.get("text") or "").encode("utf-8")).hexdigest()
                        for item in outputs
                        if isinstance(item, dict) and item.get("text")
                    ],
                }
            )
        else:
            full_layer_count = max(0, args.stage_end - args.stage_start)
            full_ready = bool(
                runtime_report.get("ok") is True
                and runtime_report.get("full_stage_owned_tpu_loader_ready") is True
                and int(runtime_report.get("executed_layer_count") or 0) >= full_layer_count
                and int(runtime_report.get("missing_stage_key_count") or 0) == 0
            )
            report.update(
                {
                    "ok": runtime_report.get("ok") is True,
                    "colab_qwen_stage_loader_ready": full_ready,
                    "runtime_report": {
                        key: value
                        for key, value in runtime_report.items()
                        if key
                        not in {
                            "errors_public",
                            "web_tpu_jupyter_steps",
                            "web_tpu_executor_attempts",
                        }
                    },
                    "executed_layer_count": int(runtime_report.get("executed_layer_count") or 0),
                    "loaded_execution_tensor_gb": float(runtime_report.get("loaded_execution_tensor_gb") or 0.0),
                    "loaded_execution_tensor_key_count": int(runtime_report.get("loaded_execution_tensor_key_count") or 0),
                    "missing_stage_key_count": int(runtime_report.get("missing_stage_key_count") or 0),
                    "tpu_device_count": int(runtime_report.get("tpu_device_count") or 0),
                    "stage_output_hash": str(runtime_report.get("stage_output_hash") or ""),
                }
            )
    except Exception as exc:  # noqa: BLE001
        report.update(
            {
                "ok": False,
                "colab_qwen_stage_loader_ready": False,
                "blockers": ["colab_qwen_stage_loader_exception"],
                "diagnosis_codes": ["colab_qwen_stage_loader_exception"],
                "error_type": type(exc).__name__,
                "error_digest": "sha256:" + hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
            }
        )
    finally:
        if runtime is not None:
            try:
                runtime.stop()
            except Exception:
                pass
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "colab_tpu_qwen_stage_loader_probe.json"
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(output_path)
        if not report.get("ok"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
