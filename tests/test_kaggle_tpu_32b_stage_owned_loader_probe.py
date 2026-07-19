from __future__ import annotations

import argparse
import json
import py_compile
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_tpu_32b_stage_owned_loader_probe as probe


def _args(tmp_path: Path) -> argparse.Namespace:
    return probe.parse_args(["--output-dir", str(tmp_path)])


def test_qwen_stage_keys_selects_only_requested_layers() -> None:
    weight_map = {
        "model.layers.20.input_layernorm.weight": "before.safetensors",
        "model.layers.21.input_layernorm.weight": "stage.safetensors",
        "model.layers.41.mlp.down_proj.weight": "stage.safetensors",
        "model.layers.42.input_layernorm.weight": "after.safetensors",
        "model.embed_tokens.weight": "embed.safetensors",
    }

    keys = probe.qwen_stage_keys(weight_map, stage_start=21, stage_end=42)

    assert keys == [
        "model.layers.21.input_layernorm.weight",
        "model.layers.41.mlp.down_proj.weight",
    ]


def test_rendered_web_probe_code_compiles(tmp_path: Path) -> None:
    args = _args(tmp_path)
    path = tmp_path / "web_probe.py"
    path.write_text(probe.render_web_probe_code(args), encoding="utf-8")

    py_compile.compile(str(path), doraise=True)

    rendered = path.read_text(encoding="utf-8")
    assert "stage_owned_header_verified" in rendered
    assert "partial_tensor_to_tpu_verified" in rendered
    assert "full_stage_owned_tpu_loader_ready" in rendered
    assert "weight_tensor_values_public" in rendered


def test_default_tensor_key_follows_stage_start(tmp_path: Path) -> None:
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--model-repo",
        "Qwen/Qwen2.5-72B-Instruct",
        "--stage-start",
        "32",
        "--stage-end",
        "40",
    ])

    assert args.tensor_key == "model.layers.32.input_layernorm.weight"


def test_build_report_keeps_partial_loader_boundary(tmp_path: Path) -> None:
    args = _args(tmp_path)
    runtime_report = {
        "schema": probe.SCHEMA,
        "ok": True,
        "stage_owned_header_verified": True,
        "partial_tensor_to_tpu_verified": True,
        "full_stage_owned_tpu_loader_ready": False,
        "assigned_weight_key_count": 252,
        "assigned_weight_file_count": 6,
        "present_stage_key_count": 252,
        "selected_tensor_key_hash": "sha256:key",
        "selected_tensor_value_hash": "sha256:value",
        "selected_tensor_tpu_summary_hash": "sha256:summary",
        "selected_tensor_shape": [5120],
        "selected_tensor_dtype": "BF16",
        "selected_tensor_bytes": 10240,
        "tpu_device_count": 8,
        "tpu_device_kind": "TPU v5 lite",
        "executed_layer_count": 0,
        "full_stage_layer_count": 21,
        "loaded_execution_tensor_key_count": 0,
        "loaded_execution_tensor_bytes": 0,
        "stage_output_hash": "",
        "stage_local_kv_cache_verified": False,
        "diagnosis_codes": ["kaggle_web_tpu_32b_partial_tensor_to_tpu_verified"],
        "blockers": ["full_stage_owned_tpu_loader_not_executed"],
    }

    report = probe.build_report(
        args,
        runtime_report=runtime_report,
        steps=[{"name": "jupyter_ws_execute", "ok": True}],
        output_dir=tmp_path,
    )

    assert report["ok"] is True
    assert report["stage_owned_header_verified"] is True
    assert report["partial_tensor_to_tpu_verified"] is True
    assert report["full_stage_owned_tpu_loader_ready"] is False
    assert report["tpu_32b_runtime_adapter_ready"] is False
    assert report["executed_layer_count"] == 0
    assert report["full_stage_layer_count"] == 21
    assert "full_stage_owned_tpu_loader_not_executed" in report["blockers"]
    assert probe.public_redaction_errors(report) == []
    saved = json.loads((tmp_path / "kaggle_tpu_32b_stage_owned_loader_probe.json").read_text(encoding="utf-8"))
    assert saved["schema"] == probe.SCHEMA


def test_execute_web_code_prefers_bridge_iframe_executor(tmp_path: Path) -> None:
    args = _args(tmp_path)
    bridge_report = {
        "schema": probe.SCHEMA,
        "ok": False,
        "blockers": ["web_tpu_jupyter_proxy_not_found"],
        "diagnosis_codes": ["bridge_web_tpu_jupyter_proxy_not_found"],
        "web_tpu_jupyter_steps": [{"name": "web_tpu_execute_subprocess", "ok": False}],
        "jupyter_proxy_token_public": False,
        "public_artifact_safe": True,
    }

    with mock.patch.object(probe.web_tpu_bridge, "execute_web_tpu_code_via_iframe", return_value=bridge_report) as mocked:
        runtime, steps = probe.execute_web_code(args, "print('unused')")

    mocked.assert_called_once()
    assert runtime["blockers"] == ["web_tpu_jupyter_proxy_not_found"]
    assert steps == [{"name": "web_tpu_execute_subprocess", "ok": False}]
