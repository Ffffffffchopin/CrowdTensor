from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

from scripts import deepseek_v4_flash_safetensors_stage_header_check as check
from scripts import deepseek_v4_flash_safetensors_stage_header_probe as probe


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_dsv4_header_probe_"))


def _stage_keys() -> list[str]:
    return [
        "layers.16.attn.wq_a.weight",
        "layers.16.attn.wq_b.weight",
        "layers.16.attn.wkv_a.weight",
        "layers.16.attn.wo_a.weight",
        "layers.16.attn.wo_b.weight",
        "layers.16.ffn.gate.weight",
        "layers.16.ffn.shared_experts.0.up_proj.weight",
        "layers.16.ffn.experts.0.up_proj.weight",
        "layers.16.hc_attn.weight",
        "layers.16.hc_ffn.weight",
        "layers.16.attn_norm.weight",
        "layers.16.ffn_norm.weight",
        "layers.17.attn.wq_a.weight",
        "layers.17.ffn.experts.1.down_proj.weight",
    ]


def _config() -> dict:
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "moe_intermediate_size": 2048,
        "torch_dtype": "bfloat16",
    }


def _index() -> dict:
    weight_map = {}
    for index, key in enumerate(_stage_keys()):
        weight_map[key] = "model-00001-of-00002.safetensors" if index < 8 else "model-00002-of-00002.safetensors"
    weight_map["layers.18.attn.wq_a.weight"] = "model-00002-of-00002.safetensors"
    return {
        "metadata": {"total_size": 123456789},
        "weight_map": weight_map,
    }


def _header_for(keys: list[str]) -> dict:
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for key in keys:
        size = 8
        header[key] = {"dtype": "BF16", "shape": [2, 2], "data_offsets": [offset, offset + size]}
        offset += size
    return header


def _patch_hf(monkeypatch, *, omit_key: str = "") -> None:
    index = _index()

    def fake_fetch_hf_json(model_id: str, filename: str, *, timeout_seconds: float = 120.0) -> dict:
        assert model_id == probe.DEFAULT_MODEL_ID
        if filename == "config.json":
            return _config()
        if filename == "model.safetensors.index.json":
            return index
        raise AssertionError(filename)

    def fake_load_safetensors_header(
        model_id: str,
        filename: str,
        *,
        timeout_seconds: float,
        max_header_bytes: int,
    ) -> tuple[int, dict]:
        keys = [key for key, file_name in index["weight_map"].items() if file_name == filename]
        if omit_key:
            keys = [key for key in keys if key != omit_key]
        header = _header_for(keys)
        return len(json.dumps(header).encode("utf-8")), header

    monkeypatch.setattr(probe, "fetch_hf_json", fake_fetch_hf_json)
    monkeypatch.setattr(probe, "load_safetensors_header", fake_load_safetensors_header)


def test_parse_safetensors_header_from_bytes() -> None:
    header = _header_for(["layers.16.attn.wq_a.weight"])
    payload = json.dumps(header).encode("utf-8")
    blob = struct.pack("<Q", len(payload)) + payload + b"tensor-bytes"

    header_len, loaded = probe.parse_safetensors_header_from_bytes(blob, max_header_bytes=4096)

    assert header_len == len(payload)
    assert loaded["layers.16.attn.wq_a.weight"]["dtype"] == "BF16"


def test_stage_header_probe_ready_with_mocked_hf_headers(monkeypatch) -> None:
    _patch_hf(monkeypatch)
    out = _tmp_dir()

    report = probe.build_report(
        probe.parse_args([
            "--output-dir",
            str(out),
            "--layer-start",
            "16",
            "--layer-end",
            "18",
        ])
    )

    assert report["safetensors_header_ready"] is True
    assert report["stage_header_shape_ready"] is True
    assert report["stage_mapping"]["selected_key_count"] == len(_stage_keys())
    assert report["headers"]["header_file_count"] == 2
    assert report["headers"]["missing_header_key_count"] == 0
    assert report["headers"]["dtype_counts"]["BF16"] == len(_stage_keys())
    assert report["headers"]["real_weight_tensor_values_loaded"] is False
    assert check.validate_report(report, require_ready=True) == []


def test_stage_header_probe_records_missing_header_key_without_overclaim(monkeypatch) -> None:
    _patch_hf(monkeypatch, omit_key="layers.16.attn.wq_a.weight")
    out = _tmp_dir()

    report = probe.build_report(
        probe.parse_args([
            "--output-dir",
            str(out),
            "--layer-start",
            "16",
            "--layer-end",
            "18",
        ])
    )

    assert report["safetensors_header_ready"] is False
    assert "safetensors_header_key_missing" in report["blockers"]
    assert check.validate_report(report) == []
    assert "safetensors_header_not_ready" in check.validate_report(report, require_ready=True)


def test_checker_rejects_weight_value_overclaim(monkeypatch) -> None:
    _patch_hf(monkeypatch)
    out = _tmp_dir()
    report = probe.build_report(probe.parse_args(["--output-dir", str(out)]))
    report["headers"]["real_weight_tensor_values_loaded"] = True

    errors = check.validate_report(report, require_ready=True)

    assert "real_weight_tensor_values_loaded_overclaim" in errors
