#!/usr/bin/env python3
"""Validate stage-selective safetensors materialization without real model downloads."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor import real_llm  # noqa: E402


SCHEMA = "stage_selective_weight_loading_check_v1"
DEFAULT_OUTPUT_DIR = "dist/stage-selective-weight-loading-check"
REDACTION_FRAGMENTS = (
    '"generated_text":',
    '"generated_token_ids":',
    '"activation_results":',
    '"hidden_state":',
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "CROWDTENSOR_ADMIN_TOKEN",
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return [fragment for fragment in REDACTION_FRAGMENTS if fragment in encoded]


def synthetic_metadata() -> dict[str, Any]:
    return {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "num_hidden_layers": 4,
        "hidden_size": 8,
        "split_index": 2,
        "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.2.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
            "model.layers.3.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
            "model.norm.weight": "model-00002-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors",
        },
    }


def build_synthetic_weights(root: Path) -> None:
    import torch  # type: ignore
    from safetensors.torch import save_file  # type: ignore
    from transformers import LlamaConfig, LlamaForCausalLM  # type: ignore

    model = LlamaForCausalLM(synthetic_llama_config())
    state = model.state_dict()

    save_file(
        {
            "model.embed_tokens.weight": torch.ones_like(state["model.embed_tokens.weight"]),
            "model.layers.0.self_attn.q_proj.weight": torch.full_like(state["model.layers.0.self_attn.q_proj.weight"], 2.0),
            "model.layers.1.self_attn.q_proj.weight": torch.full_like(state["model.layers.1.self_attn.q_proj.weight"], 3.0),
            "model.layers.2.self_attn.q_proj.weight": torch.full_like(state["model.layers.2.self_attn.q_proj.weight"], 4.0),
        },
        root / "model-00001-of-00002.safetensors",
    )
    save_file(
        {
            "model.layers.2.self_attn.q_proj.weight": torch.full_like(state["model.layers.2.self_attn.q_proj.weight"], 5.0),
            "model.layers.3.self_attn.q_proj.weight": torch.full_like(state["model.layers.3.self_attn.q_proj.weight"], 6.0),
            "model.norm.weight": torch.ones_like(state["model.norm.weight"]),
            "lm_head.weight": torch.full_like(state["lm_head.weight"], 7.0),
            "model.layers.1.self_attn.q_proj.weight": torch.full_like(state["model.layers.1.self_attn.q_proj.weight"], 8.0),
        },
        root / "model-00002-of-00002.safetensors",
    )


def synthetic_llama_config() -> Any:
    from transformers import LlamaConfig  # type: ignore

    return LlamaConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )


def synthetic_stage_model() -> Any:
    from transformers import LlamaForCausalLM  # type: ignore

    return LlamaForCausalLM(synthetic_llama_config())


def build_report(output_dir: Path) -> dict[str, Any]:
    missing = real_llm.missing_hf_dependencies()
    metadata = synthetic_metadata()
    stage_summaries: list[dict[str, Any]] = []
    application_summaries: list[dict[str, Any]] = []
    if missing:
        support = real_llm.real_llm_execution_support_summary(metadata)
        report = {
            "schema": SCHEMA,
            "ok": False,
            "generated_at": utc_now(),
            "output_dir": str(output_dir),
            "stage_selective_weight_loading_ready": False,
            "model_id": metadata["model_id"],
            "execution_family": real_llm.execution_family_from_metadata(metadata),
            "missing_dependencies": missing,
            "stage_summaries": [],
            "model_execution_support": support,
            "diagnosis_codes": ["stage_selective_weight_loading_dependencies_missing"],
            "blockers": ["hf_dependencies_missing"],
            "safety": {
                "public_artifact_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "activation_public": False,
                "credentials_public": False,
                "private_kaggle_material_public": False,
            },
        }
    else:
        with tempfile.TemporaryDirectory(prefix="crowdtensor_stage_weights_") as tmp:
            root = Path(tmp)
            build_synthetic_weights(root)
            for stage_id in (0, 1):
                tensors, load_summary = real_llm._load_stage_selective_safetensors(  # noqa: SLF001
                    metadata,
                    stage_id=stage_id,
                    weight_root=root,
                )
                stage_summaries.append(load_summary)
                application_summaries.append(
                    real_llm._apply_stage_selective_tensors_to_model(  # noqa: SLF001
                        synthetic_stage_model(),
                        tensors,
                        metadata,
                        stage_id=stage_id,
                    )
                )
        support = real_llm.real_llm_execution_support_summary({
            **metadata,
            "stage_selective_weight_load_summaries": stage_summaries,
            "stage_selective_weight_application_summaries": application_summaries,
        })
        ready = bool(
            stage_summaries
            and all(stage.get("ready") and stage.get("loads_only_stage_weight_keys") for stage in stage_summaries)
            and application_summaries
            and all(stage.get("ready") and stage.get("loads_only_stage_weight_keys") for stage in application_summaries)
            and support.get("partial_weight_tensor_materialization_ready")
            and support.get("partial_weight_tensor_application_ready")
            and support.get("partial_weight_runtime_execution_ready") is False
        )
        report = {
            "schema": SCHEMA,
            "ok": ready,
            "generated_at": utc_now(),
            "output_dir": str(output_dir),
            "stage_selective_weight_loading_ready": ready,
            "model_id": metadata["model_id"],
            "execution_family": real_llm.execution_family_from_metadata(metadata),
            "partition_mode": metadata["partition_mode"],
            "stage_summaries": stage_summaries,
            "stage_application_summaries": application_summaries,
            "model_execution_support": support,
            "readiness_truth": {
                "stage_selective_weight_loading_is_not_7b_runtime": True,
                "stage_selective_weight_application_is_not_7b_runtime": True,
                "partial_weight_tensor_materialization_is_not_runtime_execution": True,
                "partial_weight_tensor_application_is_not_runtime_execution": True,
                "seven_b_eight_b_validated": False,
                "production_swarm_inference_claimed": False,
            },
            "diagnosis_codes": [
                "stage_selective_weight_loading_check_ready"
                if ready
                else "stage_selective_weight_loading_check_not_ready",
                "real_llm_stage_selective_weight_materialization_ready"
                if support.get("partial_weight_tensor_materialization_ready")
                else "real_llm_stage_selective_weight_materialization_not_ready",
                "real_llm_stage_selective_weight_application_ready"
                if support.get("partial_weight_tensor_application_ready")
                else "real_llm_stage_selective_weight_application_not_ready",
                "real_llm_partial_weight_runtime_execution_missing",
            ],
            "blockers": ["real_llm_partial_weight_runtime_execution_missing"],
            "safety": {
                "public_artifact_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "activation_public": False,
                "credentials_public": False,
                "private_kaggle_material_public": False,
            },
        }
    report["public_leak_paths"] = public_redaction_errors(report)
    if report["public_leak_paths"]:
        report["ok"] = False
        report["stage_selective_weight_loading_ready"] = False
        report["diagnosis_codes"].append("stage_selective_weight_loading_public_leak_detected")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate stage-selective safetensors loading.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    report = build_report(output_dir)
    write_json(output_dir / "stage_selective_weight_loading_check.json", report)
    print(json.dumps(report, sort_keys=True) if args.json else json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
