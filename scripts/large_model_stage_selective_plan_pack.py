#!/usr/bin/env python3
"""Build public-safe stage-selective large-model planning evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor import real_llm  # noqa: E402


SCHEMA = "large_model_stage_selective_plan_v1"
SUPPORT_BUNDLE_SCHEMA = "large_model_stage_selective_plan_support_bundle_v1"
DEFAULT_OUTPUT_DIR = "dist/large-model-stage-selective-plan"
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]
MODEL_CONFIGS = {
    "qwen/qwen2.5-7b-instruct": {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "vocab_size": 152064,
    },
    "qwen/qwen2.5-14b-instruct": {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "num_hidden_layers": 48,
        "hidden_size": 5120,
        "vocab_size": 152064,
    },
}
REDACTION_FRAGMENTS = (
    '"prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"activation_results":',
    '"hidden_state":',
    '"lease_token":',
    '"idempotency_key":',
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_ADMIN_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "operator.private.env",
    "miner.private.env",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def public_redaction_errors(payload: Any) -> list[str]:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return [fragment for fragment in REDACTION_FRAGMENTS if fragment in encoded]


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        rel = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        rel = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": rel, "present": path.is_file()}
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def synthetic_weight_map(*, family: str, layer_count: int, stage_count: int) -> dict[str, str]:
    shards = max(2, min(64, int(stage_count) * 2))
    def filename(index: int) -> str:
        shard = max(1, min(shards, (int(index) * shards // max(1, int(layer_count))) + 1))
        return f"model-{shard:05d}-of-{shards:05d}.safetensors"

    if family == real_llm.EXECUTION_FAMILY_LLAMA_LIKE:
        weight_map = {
            "model.embed_tokens.weight": "model-00001-of-%05d.safetensors" % shards,
            "model.norm.weight": f"model-{shards:05d}-of-{shards:05d}.safetensors",
            "lm_head.weight": f"model-{shards:05d}-of-{shards:05d}.safetensors",
        }
        for index in range(int(layer_count)):
            weight_map[f"model.layers.{index}.self_attn.q_proj.weight"] = filename(index)
            weight_map[f"model.layers.{index}.mlp.down_proj.weight"] = filename(index)
        return weight_map
    weight_map = {
        "transformer.wte.weight": "model-00001-of-%05d.safetensors" % shards,
        "transformer.wpe.weight": "model-00001-of-%05d.safetensors" % shards,
        "transformer.ln_f.weight": f"model-{shards:05d}-of-{shards:05d}.safetensors",
        "lm_head.weight": f"model-{shards:05d}-of-{shards:05d}.safetensors",
    }
    for index in range(int(layer_count)):
        weight_map[f"transformer.h.{index}.attn.c_attn.weight"] = filename(index)
        weight_map[f"transformer.h.{index}.mlp.c_fc.weight"] = filename(index)
    return weight_map


def model_metadata(model_id: str, *, stage_count: int) -> dict[str, Any]:
    key = str(model_id or "").strip().lower()
    config = dict(MODEL_CONFIGS.get(key) or {})
    if not config:
        config = {
            "model_type": "qwen2" if "qwen" in key else "",
            "architectures": ["Qwen2ForCausalLM"] if "qwen" in key else [],
            "num_hidden_layers": 0,
            "hidden_size": 0,
            "vocab_size": 0,
        }
    family = real_llm.execution_family_from_metadata({
        "model_id": model_id,
        **config,
    })
    layer_count = int(config.get("num_hidden_layers") or 0)
    metadata = {
        "model_id": model_id,
        **config,
        "stage_count": int(stage_count),
        "partition_mode": real_llm.PARTITION_MODE_STAGE_LOCAL,
    }
    metadata["weight_map"] = synthetic_weight_map(
        family=family,
        layer_count=layer_count,
        stage_count=int(stage_count),
    )
    return metadata


def model_plan(model_id: str, *, stage_count: int, kaggle_gpu_memory_gb: float) -> dict[str, Any]:
    metadata = model_metadata(model_id, stage_count=stage_count)
    two_stage_metadata = {**metadata, "stage_count": 2}
    two_stage_plan = real_llm.real_llm_n_stage_partition_plan(two_stage_metadata, stage_count=2)
    n_stage_plan = real_llm.real_llm_n_stage_partition_plan(metadata, stage_count=stage_count)
    support = real_llm.real_llm_execution_support_summary(metadata)
    max_stage_bytes = max(
        [int(stage.get("estimated_stage_weight_bytes_fp32") or 0) for stage in n_stage_plan.get("stage_plans", [])]
        or [0]
    )
    max_stage_gb_fp32 = round(max_stage_bytes / 1_000_000_000.0, 4)
    max_stage_gb_fp16 = round(max_stage_gb_fp32 / 2.0, 4)
    two_stage_max_bytes = max(
        [int(stage.get("estimated_stage_weight_bytes_fp32") or 0) for stage in two_stage_plan.get("stage_plans", [])]
        or [0]
    )
    two_stage_max_gb_fp16 = round((two_stage_max_bytes / 1_000_000_000.0) / 2.0, 4)
    # Dual-Kaggle feasibility is a two-stage question, independent of the
    # requested N-stage planning target. Keep the two-stage result visible so a
    # four-stage planning report does not incorrectly imply 7B needs >2 stages.
    fits_dual_kernel_target = bool(two_stage_plan.get("ready") and two_stage_max_gb_fp16 <= float(kaggle_gpu_memory_gb))
    overhead_guard_gb = float(kaggle_gpu_memory_gb) * 0.85
    dual_kernel_practical_fit = bool(two_stage_plan.get("ready") and two_stage_max_gb_fp16 <= overhead_guard_gb)
    requires_more_than_two_stages = bool(
        not dual_kernel_practical_fit
        and n_stage_plan.get("ready")
        and max_stage_gb_fp16 <= overhead_guard_gb
    )
    return {
        "schema": "large_model_stage_selective_model_plan_v1",
        "model_id": model_id,
        "execution_family": support.get("execution_family"),
        "parameter_count_estimate": support.get("parameter_count_estimate"),
        "estimated_weight_bytes_fp32": support.get("estimated_weight_bytes_fp32"),
        "target_stage_count": int(stage_count),
        "kaggle_gpu_memory_gb": float(kaggle_gpu_memory_gb),
        "two_stage_plan_ready": bool(two_stage_plan.get("ready")),
        "two_stage_max_stage_weight_gb_fp16_estimate": two_stage_max_gb_fp16,
        "two_stage_practical_fit_guard_gb": round(overhead_guard_gb, 4),
        "two_stage_practical_fit_with_overhead_guard": dual_kernel_practical_fit,
        "n_stage_plan_ready": bool(n_stage_plan.get("ready")),
        "n_stage_max_stage_weight_gb_fp32_estimate": max_stage_gb_fp32,
        "n_stage_max_stage_weight_gb_fp16_estimate": max_stage_gb_fp16,
        "dual_kaggle_kernel_fit_estimate": fits_dual_kernel_target,
        "requires_more_than_two_stages_estimate": requires_more_than_two_stages,
        "two_stage_partition_plan": two_stage_plan,
        "n_stage_partition_plan": n_stage_plan,
        "execution_support": support,
        "runtime_verified": False,
        "public_artifact_safe": True,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CrowdTensor Large-Model Stage-Selective Plan",
        "",
        f"Schema: `{report.get('schema')}`",
        f"OK: `{report.get('ok')}`",
        "",
        "## Models",
        "",
    ]
    for item in report.get("model_plans") or []:
        lines.extend([
            f"- `{item.get('model_id')}`: stages `{item.get('target_stage_count')}`, "
            f"params `{item.get('parameter_count_estimate')}`, "
            f"max fp16 stage GB `{item.get('n_stage_max_stage_weight_gb_fp16_estimate')}`, "
            f"dual-kernel fit `{item.get('dual_kaggle_kernel_fit_estimate')}`, "
            f"runtime verified `{item.get('runtime_verified')}`",
        ])
    lines.extend([
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "This is planning evidence only. It contains no prompts, generated text, token ids, activations, credentials, or leases.",
        "",
    ])
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [item.strip() for item in str(args.model_ids or "").split(",") if item.strip()]
    if not models:
        models = list(DEFAULT_MODELS)
    plans = [
        model_plan(model_id, stage_count=args.stage_count, kaggle_gpu_memory_gb=args.kaggle_gpu_memory_gb)
        for model_id in models
    ]
    codes: list[str] = []
    if all(plan.get("n_stage_plan_ready") for plan in plans):
        codes.append("large_model_n_stage_partition_plan_ready")
    if any(plan.get("requires_more_than_two_stages_estimate") for plan in plans):
        codes.append("large_model_more_than_two_stage_need_detected")
    if any(plan.get("dual_kaggle_kernel_fit_estimate") for plan in plans):
        codes.append("large_model_dual_kaggle_kernel_fit_estimated")
    if any(str(plan.get("model_id") or "").lower().find("14b") >= 0 for plan in plans):
        codes.append("large_model_14b_partition_plan_ready")
    if any(str(plan.get("model_id") or "").lower().find("7b") >= 0 for plan in plans):
        codes.append("large_model_7b_partition_plan_ready")
    report = {
        "schema": SCHEMA,
        "ok": bool(plans and all(plan.get("n_stage_plan_ready") for plan in plans)),
        "generated_at": utc_now(),
        "output_dir": str(output_dir),
        "target_stage_count": int(args.stage_count),
        "kaggle_gpu_memory_gb": float(args.kaggle_gpu_memory_gb),
        "model_plans": plans,
        "diagnosis_codes": sorted(set(codes)),
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "credentials_public": False,
            "lease_material_public": False,
        },
        "limitations": [
            "Planning evidence only; it does not execute 7B/14B inference.",
            "FP16 stage-size estimates exclude KV cache, framework overhead, allocator fragmentation, and temporary load buffers.",
            "Current live runtime remains the proven two-stage path until task scheduling and runtime forwarding are migrated to N-stage.",
        ],
    }
    redaction_errors = public_redaction_errors(report)
    if redaction_errors:
        report["ok"] = False
        report["diagnosis_codes"].append("large_model_stage_selective_plan_redaction_failed")
        report["redaction_errors"] = redaction_errors
    artifacts = {
        "summary_json": artifact_entry(
            output_dir / "large_model_stage_selective_plan.json",
            output_dir,
            kind="large_model_stage_selective_plan",
            schema=SCHEMA,
            ok=report.get("ok"),
        ),
        "summary_markdown": artifact_entry(
            output_dir / "large_model_stage_selective_plan.md",
            output_dir,
            kind="large_model_stage_selective_plan_markdown",
        ),
        "support_bundle_json": artifact_entry(
            output_dir / "support_bundle.json",
            output_dir,
            kind="large_model_stage_selective_plan_support_bundle",
            schema=SUPPORT_BUNDLE_SCHEMA,
            ok=report.get("ok"),
        ),
    }
    report["artifacts"] = artifacts
    support = {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "ok": report.get("ok"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "target_stage_count": report.get("target_stage_count"),
        "model_plans": report.get("model_plans"),
        "safety": report.get("safety"),
        "limitations": report.get("limitations"),
    }
    write_json(output_dir / "large_model_stage_selective_plan.json", report)
    (output_dir / "large_model_stage_selective_plan.md").write_text(render_markdown(report), encoding="utf-8")
    write_json(output_dir / "support_bundle.json", support)
    for artifact in artifacts.values():
        path = output_dir / str(artifact.get("path") or "")
        artifact["present"] = path.is_file()
    write_json(output_dir / "large_model_stage_selective_plan.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stage-selective large-model planning evidence.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-ids", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--stage-count", type=int, default=4)
    parser.add_argument("--kaggle-gpu-memory-gb", type=float, default=15.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.stage_count < 2 or args.stage_count > real_llm.MAX_PARTITION_STAGES:
        raise SystemExit(f"--stage-count must be between 2 and {real_llm.MAX_PARTITION_STAGES}")
    if args.kaggle_gpu_memory_gb <= 0:
        raise SystemExit("--kaggle-gpu-memory-gb must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(render_markdown(report))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
