from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

from scripts import glm52_model_source_resolver as resolver


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_source_"))


def _fake_model_api(repo: str, *, timeout_seconds: float = 90.0) -> dict:
    tags = ["text-generation", "glm_moe_dsa", "license:mit"]
    if repo != resolver.MODEL_ID:
        tags.append(f"base_model:{resolver.MODEL_ID}")
    siblings = [{"rfilename": "config.json"}]
    if repo == resolver.MODEL_ID:
        siblings.extend({"rfilename": f"model-{idx:05d}-of-00282.safetensors"} for idx in range(1, 4))
    elif "AWQ" in repo:
        siblings.extend({"rfilename": f"model-{idx:05d}-of-00083.safetensors"} for idx in range(1, 3))
    else:
        siblings.append({"rfilename": "GLM-5.2-Q2_K.gguf"})
    return {
        "id": repo,
        "private": False,
        "gated": False,
        "downloads": 123,
        "likes": 9,
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "tags": tags,
        "siblings": siblings,
    }


def _fake_tree_api(repo: str, *, timeout_seconds: float = 90.0) -> list[dict]:
    if repo == resolver.MODEL_ID:
        return [
            {"path": "model-00001-of-00282.safetensors", "type": "file", "lfs": {"size": 5_000_000_000}},
            {"path": "model-00002-of-00282.safetensors", "type": "file", "lfs": {"size": 5_000_000_000}},
            {"path": "model-00003-of-00282.safetensors", "type": "file", "lfs": {"size": 5_000_000_000}},
        ]
    if "AWQ" in repo:
        return [
            {"path": "model-00001-of-00083.safetensors", "type": "file", "lfs": {"size": 2_000_000_000}},
            {"path": "model-00002-of-00083.safetensors", "type": "file", "lfs": {"size": 2_000_000_000}},
        ]
    return [{"path": "GLM-5.2-Q2_K.gguf", "type": "file", "lfs": {"size": 80_000_000_000}}]


def _fake_raw_json(repo: str, filename: str, *, timeout_seconds: float = 90.0) -> dict:
    if filename == "config.json":
        return {
            "architectures": ["GlmMoeDsaForCausalLM"],
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 6,
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 32,
            "n_routed_experts": 8,
            "moe_intermediate_size": 8,
        }
    if filename == "model.safetensors.index.json" and repo == resolver.MODEL_ID:
        weight_map = {
            "model.embed_tokens.weight": "model-00001-of-00282.safetensors",
            "model.layers.0.input_layernorm.weight": "model-00001-of-00282.safetensors",
            "model.layers.2.mlp.gate.weight": "model-00002-of-00282.safetensors",
            "model.layers.5.self_attn.q_proj.weight": "model-00003-of-00282.safetensors",
            "model.norm.weight": "model-00003-of-00282.safetensors",
            "lm_head.weight": "model-00003-of-00282.safetensors",
        }
        return {"metadata": {"total_size": 1_506_659_919_872}, "weight_map": weight_map}
    return {}


def test_build_report_resolves_public_glm52_source_and_budget_blockers() -> None:
    with (
        mock.patch.object(resolver, "hf_model_api", side_effect=_fake_model_api),
        mock.patch.object(resolver, "hf_tree_api", side_effect=_fake_tree_api),
        mock.patch.object(resolver, "hf_raw_json", side_effect=_fake_raw_json),
        mock.patch.object(resolver, "hf_readme", return_value="# GLM-5.2"),
    ):
        report = resolver.build_report(
            resolver.parse_args([
                "--output-dir",
                str(_tmp_dir()),
                "--runtime-disk-budget-gb",
                "120",
            ])
        )

    assert report["ok"] is True
    assert report["glm52_source_resolver_ready"] is True
    assert report["model"]["model_id"] == resolver.MODEL_ID
    assert report["model"]["model_type"] == "glm_moe_dsa"
    assert report["model"]["official_weight_key_count"] == 6
    assert report["model"]["official_weight_total_size_gb"] > 1500
    assert "glm52_full_weights_exceed_kaggle_runtime_disk_budget" in report["blockers"]
    assert report["stage_adapter_plan"]["assigned_key_count_total"] == 6
    assert report["stage_adapter_plan"]["stage_runtime_adapter_verified"] is False
    assert report["kaggle_attach_plan"]["hf_source_verified"] is True
    assert resolver.public_redaction_errors(report) == []
