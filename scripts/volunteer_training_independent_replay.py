#!/usr/bin/env python3
"""Independently reload and evaluate Volunteer Training Beta checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from crowdtensor.hf_lora_training import evaluate_adapter
from crowdtensor.training_contract import sha256_file, sha256_json
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator
from crowdtensor.volunteer_training_protocol import with_public_safety


SCHEMA = "crowdtensor_volunteer_training_independent_replay_v1"


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("volunteer replay request must be an object")
    return value


def run_replay(request_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    request = _read(request_path)
    required = {
        "campaign_dir",
        "base_model_path",
        "initial_adapter_path",
        "distributed_adapter_path",
        "centralized_adapter_path",
        "validation_dataset_path",
    }
    if not required.issubset(request):
        raise ValueError("volunteer replay request is incomplete")
    indexes = [int(item) for item in request.get("validation_sample_indexes") or []]
    if not indexes:
        raise ValueError("volunteer replay validation indexes are empty")
    lineage = VolunteerTrainingCoordinator(request["campaign_dir"]).checkpoint_lineage()
    evaluations: dict[str, dict[str, Any]] = {}
    for name, adapter_path in (
        ("initial", request["initial_adapter_path"]),
        ("distributed", request["distributed_adapter_path"]),
        ("centralized", request["centralized_adapter_path"]),
    ):
        evaluation = evaluate_adapter(
            base_model_path=request["base_model_path"],
            adapter_path=adapter_path,
            dataset_path=request["validation_dataset_path"],
            sample_indexes=indexes,
            batch_size=int(request.get("validation_batch_size") or 1),
        )
        adapter_file = Path(adapter_path) / "adapter_model.safetensors"
        evaluations[name] = {
            "mean_loss": float(evaluation["mean_loss"]),
            "logits_hash": evaluation["logits_hash"],
            "adapter_file_hash": sha256_file(adapter_file),
            "sample_count": int(evaluation["sample_count"]),
            "finite": math.isfinite(float(evaluation["mean_loss"])),
            "reload_verified": True,
        }
    expected_distributed_hash = str(request.get("expected_distributed_adapter_hash") or "")
    all_finite = all(item["finite"] for item in evaluations.values())
    required_checks = [
        lineage.get("ok") is True,
        int(lineage.get("adapter_version") or -1)
        == int(request.get("expected_adapter_version") or -2),
        evaluations["distributed"]["adapter_file_hash"] == expected_distributed_hash,
        all_finite,
    ]
    report = with_public_safety(
        {
            "schema": SCHEMA,
            "ok": all(required_checks),
            "independent_process_replay_verified": all(required_checks),
            "process_id_hash": "sha256:"
            + hashlib.sha256(str(os.getpid()).encode("ascii")).hexdigest(),
            "raw_process_id_public": False,
            "checkpoint_lineage": lineage,
            "evaluations": evaluations,
            "all_losses_finite": all_finite,
            "distributed_checkpoint_hash_matches_lineage_head": evaluations[
                "distributed"
            ]["adapter_file_hash"]
            == lineage.get("canonical_adapter_hash"),
            "initial_to_distributed_loss_change": evaluations["initial"]["mean_loss"]
            - evaluations["distributed"]["mean_loss"],
            "initial_to_centralized_loss_change": evaluations["initial"]["mean_loss"]
            - evaluations["centralized"]["mean_loss"],
            "quality_equivalence_claimed": False,
            "raw_text_public": False,
            "token_ids_public": False,
            "tensor_values_public": False,
        }
    )
    report["content_hash"] = sha256_json(report)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_replay(args.request, args.output)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
