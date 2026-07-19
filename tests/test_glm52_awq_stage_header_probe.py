from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

from scripts import glm52_awq_stage_header_check as check
from scripts import glm52_awq_stage_header_probe as probe


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_awq_header_"))


def _config() -> dict:
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 4,
        "hidden_size": 16,
    }


def _index() -> dict:
    weight_map = {
        "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
        "model.layers.0.self_attn.q_proj.qweight": "model-00001-of-00003.safetensors",
        "model.layers.0.self_attn.q_proj.scales": "model-00001-of-00003.safetensors",
        "model.layers.1.mlp.gate_proj.qweight": "model-00002-of-00003.safetensors",
        "model.layers.1.mlp.gate_proj.qzeros": "model-00002-of-00003.safetensors",
        "model.layers.2.self_attn.o_proj.qweight": "model-00002-of-00003.safetensors",
        "model.layers.3.mlp.down_proj.qweight": "model-00003-of-00003.safetensors",
        "model.norm.weight": "model-00003-of-00003.safetensors",
        "lm_head.weight": "model-00003-of-00003.safetensors",
    }
    return {"metadata": {"total_size": 440_335_957_008}, "weight_map": weight_map}


def _header_for(filename: str) -> dict:
    idx = _index()["weight_map"]
    keys = [key for key, file_name in idx.items() if file_name == filename]
    return {
        key: {
            "dtype": "I32" if key.endswith("qweight") or key.endswith("qzeros") else "F16",
            "shape": [4, 4] if "weight" in key or "qweight" in key else [4],
            "data_offsets": [offset * 32, (offset + 1) * 32],
        }
        for offset, key in enumerate(keys)
    }


def test_stage_selection_covers_layer_prefixes_and_boundaries() -> None:
    selection = probe.build_stage_selection(_config(), _index(), stage_id=0, stage_count=2)

    assert selection["stage_layer_range"] == [0, 2]
    assert selection["assigned_weight_key_count"] == 5
    assert "model.embed_tokens.weight" in selection["assigned_weight_keys"]
    assert "model.layers.1.mlp.gate_proj.qweight" in selection["assigned_weight_keys"]
    assert "model.layers.3.mlp.down_proj.qweight" not in selection["assigned_weight_keys"]


def test_build_report_verifies_header_without_tensor_values() -> None:
    def fake_fetch(repo: str, filename: str, *, timeout_seconds: float = 90.0) -> dict:
        return _config() if filename == "config.json" else _index()

    def fake_header(repo: str, filename: str, *, timeout_seconds: float = 90.0, max_header_bytes: int = 1024) -> dict:
        return _header_for(filename)

    with (
        mock.patch.object(probe, "fetch_hf_json", side_effect=fake_fetch),
        mock.patch.object(probe, "load_safetensors_header", side_effect=fake_header),
    ):
        report = probe.build_report(
            probe.parse_args([
                "--output-dir",
                str(_tmp_dir()),
                "--stage-id",
                "0",
                "--stage-count",
                "2",
            ])
        )

    assert report["ok"] is True
    assert report["glm52_awq_stage_header_ready"] is True
    assert report["present_stage_key_count"] == report["assigned_weight_key_count"]
    assert report["weight_tensor_values_loaded"] is False
    assert report["weight_tensor_values_public"] is False
    assert report["stage_runtime_adapter_verified"] is False
    assert report["stage_family_hits"]["awq_quantized_tensors"] is True
    assert check.validate_report(report, require_ready=True) == []


def test_checker_rejects_runtime_overclaim() -> None:
    report = {
        "schema": probe.SCHEMA,
        "ok": True,
        "glm52_awq_stage_header_ready": True,
        "model_repo": probe.DEFAULT_MODEL_REPO,
        "base_model_id": probe.BASE_MODEL_ID,
        "model_type": "glm_moe_dsa",
        "assigned_weight_key_count": 1,
        "present_stage_key_count": 1,
        "missing_stage_key_count": 0,
        "header_file_count": 1,
        "dtype_counts": {"I32": 1},
        "total_selected_tensor_storage_bytes": 32,
        "stage_family_hits": {"awq_quantized_tensors": True},
        "weight_tensor_values_loaded": False,
        "weight_tensor_values_public": False,
        "safetensors_header_payload_public": False,
        "stage_runtime_adapter_verified": True,
        "same_request_route_verified": False,
        "public_artifact_safe": True,
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "signed_url_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
            "activation_public": False,
            "generated_token_ids_public": False,
        },
    }

    errors = check.validate_report(report, require_ready=True)

    assert "stage_runtime_adapter_overclaim" in errors
