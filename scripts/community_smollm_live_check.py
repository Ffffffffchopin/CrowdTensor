#!/usr/bin/env python3
"""Strict checker for the Community SmolLM2 real two-stage LoRA proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.smollm_training import LIVE_SCHEMA, MODEL_ID, MODEL_REVISION


def check(report_path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    errors: list[str] = []
    if value.get("schema") != LIVE_SCHEMA:
        errors.append("community_smollm_live_schema_invalid")
    required_true = (
        "ok",
        "real_open_model_weights",
        "distinct_worker_processes",
        "strictly_contiguous_atomic_steps",
        "all_stage_optimizer_steps_applied",
        "finite_loss_verified",
        "both_stage_adapters_updated",
        "clean_install_required",
        "public_artifact_safe",
    )
    for field in required_true:
        if value.get(field) is not True:
            errors.append("community_smollm_live_" + field + "_missing")
    if value.get("random_or_synthetic_weights_used") is not False:
        errors.append("community_smollm_live_synthetic_weights_not_rejected")
    if value.get("single_process_smoke") is not False:
        errors.append("community_smollm_live_single_process_smoke_invalid")
    if value.get("physical_multi_machine_verified") is not False:
        errors.append("community_smollm_live_physical_multi_machine_overclaim")
    if value.get("node_scope") != "Kaggle logical multi-node":
        errors.append("community_smollm_live_node_scope_invalid")
    if value.get("model_id") != MODEL_ID or value.get("model_revision") != MODEL_REVISION:
        errors.append("community_smollm_live_model_identity_invalid")
    if int(value.get("logical_miner_count") or 0) != 2 or int(value.get("logical_stage_count") or 0) != 2:
        errors.append("community_smollm_live_two_stage_coverage_missing")
    stages = value.get("stage_specs") if isinstance(value.get("stage_specs"), list) else []
    if stages != [
        {"stage_id": 0, "layer_start": 0, "layer_end": 15},
        {"stage_id": 1, "layer_start": 15, "layer_end": 30},
    ]:
        errors.append("community_smollm_live_stage_partition_invalid")
    steps = [int(item) for item in value.get("committed_step_ids") or []]
    if len(steps) < 2:
        errors.append("community_smollm_live_step_count_insufficient")
    checkpoints = value.get("stage_checkpoints") if isinstance(value.get("stage_checkpoints"), list) else []
    if len(checkpoints) != 2 or any(item.get("adapter_updated") is not True for item in checkpoints):
        errors.append("community_smollm_live_checkpoint_coverage_invalid")
    export = value.get("export") if isinstance(value.get("export"), dict) else {}
    if export.get("standard_peft_format") is not True or int(export.get("adapter_tensor_count") or 0) <= 0:
        errors.append("community_smollm_live_peft_export_invalid")
    reload = value.get("reload") if isinstance(value.get("reload"), dict) else {}
    if reload.get("adapter_reload_verified") is not True or reload.get("independent_process_reload") is not True:
        errors.append("community_smollm_live_reload_invalid")
    privacy = scan_public_value(value)
    if privacy["ok"] is not True:
        errors.append("community_smollm_live_public_safety_invalid")
    return {
        "schema": "crowdtensor_smollm_two_stage_lora_live_check_v1",
        "ok": not errors,
        "errors": sorted(set(errors)),
        "logical_miner_count": int(value.get("logical_miner_count") or 0),
        "step_count": len(steps),
        "public_safety": privacy,
        "public_artifact_safe": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(args.report)
    print(json.dumps(result, sort_keys=True) if args.json else f"ok={result['ok']} errors={len(result['errors'])}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
