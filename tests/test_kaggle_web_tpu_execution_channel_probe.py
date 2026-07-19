from __future__ import annotations

import argparse
import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_web_tpu_execution_channel_check as check
from scripts import kaggle_web_tpu_execution_channel_probe as probe


def _args(tmp_path: Path) -> argparse.Namespace:
    return probe.parse_args(["--output-dir", str(tmp_path)])


def _cell(kind: str, *, ok: bool = True) -> dict:
    ready_field = "small_jax_cell_ready" if kind == "small_jax" else "tiny_qwen_like_cell_ready"
    payload = {
        "schema": probe.CELL_SCHEMA,
        "cell_kind": kind,
        "ok": ok,
        "jax_imported": ok,
        "tpu_device_count": 8 if ok else 0,
        "tpu_device_kind": "TPU v5 lite" if ok else "",
        "blockers": [] if ok else ["web_tpu_jupyter_execute_timeout"],
        "diagnosis_codes": [f"web_tpu_channel_{kind}_ready"] if ok else ["bridge_web_tpu_jupyter_execute_timeout"],
        "web_tpu_jupyter_steps": [{"name": "web_tpu_execute_subprocess", "ok": ok, "timeout": not ok}],
        "jupyter_proxy_token_public": False,
        "public_artifact_safe": True,
        ready_field: ok,
    }
    if kind == "small_jax" and ok:
        payload["result_summary_hash"] = "sha256:small"
    if kind == "tiny_qwen_like" and ok:
        payload.update(
            {
                "stage_output_hash": "sha256:tiny",
                "stage_local_kv_cache_verified": True,
                "shape_metadata": {"input_shape": [1, 4, 64], "output_shape": [1, 4, 64]},
                "qwen_components_exercised": {
                    "rms_norm": True,
                    "grouped_query_attention": True,
                    "causal_attention": True,
                    "swiglu_mlp": True,
                    "stage_local_kv_cache": True,
                },
            }
        )
    return payload


def test_rendered_cells_compile(tmp_path: Path) -> None:
    for name, source in {
        "small_jax.py": probe.render_small_jax_cell(),
        "tiny_qwen.py": probe.render_tiny_qwen_like_cell(),
    }.items():
        path = tmp_path / name
        path.write_text(source, encoding="utf-8")
        py_compile.compile(str(path), doraise=True)
        rendered = path.read_text(encoding="utf-8")
        assert "jupyter_proxy_token_public" in rendered
        assert "public_artifact_safe" in rendered


def test_build_report_marks_channel_ready(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        small_jax_report=_cell("small_jax"),
        tiny_qwen_report=_cell("tiny_qwen_like"),
        output_dir=tmp_path,
    )

    assert report["ok"] is True
    assert report["web_tpu_execution_channel_ready"] is True
    assert report["small_jax_cell_ready"] is True
    assert report["tiny_qwen_like_cell_ready"] is True
    assert report["tpu_runtime_attached"] is True
    assert report["tpu_device_count"] == 8
    assert report["blocker_codes"] == []
    assert check.validate_report(report) == []


def test_build_report_records_timeout_blocker(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        small_jax_report=_cell("small_jax", ok=False),
        tiny_qwen_report={
            "schema": probe.CELL_SCHEMA,
            "cell_kind": "tiny_qwen_like",
            "ok": False,
            "blockers": ["tiny_qwen_like_not_attempted_after_small_jax_failure"],
            "diagnosis_codes": ["web_tpu_channel_tiny_qwen_like_not_attempted"],
            "jupyter_proxy_token_public": False,
            "public_artifact_safe": True,
        },
        output_dir=tmp_path,
    )

    assert report["ok"] is False
    assert report["web_tpu_execution_channel_ready"] is False
    assert "web_tpu_execution_channel_not_ready" in report["blocker_codes"]
    assert report["failure_stage"] == "jupyter_execute"
    assert check.validate_report(report) == []


def test_cell_summary_keeps_public_executor_attempts(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        small_jax_report={
            **_cell("small_jax", ok=False),
            "web_tpu_executor_attempts": [
                {
                    "executor_name": "browser_iframe_service_manager_ws",
                    "errors_public": [{"ename": "Timeout", "message_public": "jupyter_ws_execute_timeout"}],
                    "steps": [{"name": "service_manager_ws_execute", "ok": False, "timeout": True, "wsUrl": "wss://secret"}],
                }
            ],
        },
        tiny_qwen_report={
            "schema": probe.CELL_SCHEMA,
            "cell_kind": "tiny_qwen_like",
            "ok": False,
            "blockers": ["tiny_qwen_like_not_attempted_after_small_jax_failure"],
            "diagnosis_codes": ["web_tpu_channel_tiny_qwen_like_not_attempted"],
            "jupyter_proxy_token_public": False,
            "public_artifact_safe": True,
        },
        output_dir=tmp_path,
    )

    attempts = report["cells"]["small_jax"]["web_tpu_executor_attempts"]
    assert attempts[0]["executor_name"] == "browser_iframe_service_manager_ws"
    assert attempts[0]["error_names"] == ["Timeout"]
    assert "wsUrl" not in attempts[0]["steps"][0]
    assert check.validate_report(report) == []


def test_checker_rejects_ready_without_tpu_device(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        small_jax_report=_cell("small_jax"),
        tiny_qwen_report=_cell("tiny_qwen_like"),
        output_dir=tmp_path,
    )
    report["tpu_device_count"] = 0

    errors = check.validate_report(report)

    assert "channel_ready_without_tpu_device" in errors


def test_public_artifacts_are_redacted(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        small_jax_report=_cell("small_jax"),
        tiny_qwen_report=_cell("tiny_qwen_like"),
        output_dir=tmp_path,
    )

    assert probe.public_redaction_errors(report) == []
    scanned = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.rglob("*") if path.is_file())
    for fragment in [
        "KAGGLE_KEY",
        "HF_TOKEN",
        "Bearer ",
        "Cookie:",
        "jupyter-proxy",
        "token=",
        '"generated_token_ids":',
        '"activation":',
    ]:
        assert fragment not in scanned
