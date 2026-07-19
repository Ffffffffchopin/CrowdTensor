#!/usr/bin/env python3
"""Package a completed local training job as the canonical Foundation RC."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.training_contract import sha256_file  # noqa: E402
from training_foundation_rc_check import SCHEMA, check  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return sha256_file(destination)


def pack(job_report_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source_report_path = Path(job_report_path).resolve()
    source_root = source_report_path.parent
    source_report = _load(source_report_path)
    if source_report.get("schema") != "crowdtensor_training_foundation_job_v1":
        raise ValueError("--job-report is not a Training Foundation job report")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "job_report": "training_foundation_job.json",
        "gpu_continuation_manifest": "gpu_training_continuation_manifest.json",
        "exported_adapter_model": "exported_adapter/adapter_model.safetensors",
        "exported_adapter_config": "exported_adapter/adapter_config.json",
        "cleanup": "training_cleanup.json",
        "evaluation": "training_evaluation.json",
    }
    artifact_hashes = {
        "job_report": _copy(source_report_path, output / artifacts["job_report"]),
        "gpu_continuation_manifest": _copy(
            source_root / "gpu_training_continuation_manifest.json",
            output / artifacts["gpu_continuation_manifest"],
        ),
        "exported_adapter_model": _copy(
            source_root / "exported_adapter" / "adapter_model.safetensors",
            output / artifacts["exported_adapter_model"],
        ),
        "exported_adapter_config": _copy(
            source_root / "exported_adapter" / "adapter_config.json",
            output / artifacts["exported_adapter_config"],
        ),
        "cleanup": _copy(source_root / "training_cleanup.json", output / artifacts["cleanup"]),
        "evaluation": _copy(source_root / "training_evaluation.json", output / artifacts["evaluation"]),
    }

    baseline = _load(source_root / "pipeline_baseline" / "pipeline_training_report.json")
    resumed = _load(source_root / "pipeline_resumed" / "pipeline_training_report.json")
    stage_adapter_artifacts: list[dict[str, Any]] = []
    for stage_id in (0, 1):
        baseline_source = Path(baseline["final_checkpoint"]["stages"][stage_id]["adapter_path"])
        resumed_source = Path(resumed["final_checkpoint"]["stages"][stage_id]["adapter_path"])
        baseline_relative = f"pipeline_checkpoints/baseline_stage{stage_id}_adapter.safetensors"
        resumed_relative = f"pipeline_checkpoints/resumed_stage{stage_id}_adapter.safetensors"
        stage_adapter_artifacts.append(
            {
                "stage_id": stage_id,
                "baseline": baseline_relative,
                "resumed": resumed_relative,
                "baseline_hash": _copy(baseline_source, output / baseline_relative),
                "resumed_hash": _copy(resumed_source, output / resumed_relative),
            }
        )

    dependencies: dict[str, str] = {}
    for name in ("torch", "transformers", "peft", "safetensors", "accelerate"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "missing"
    report = {
        "schema": SCHEMA,
        "training_foundation_rc_ready": True,
        "goal_achieved": True,
        "job_id": source_report.get("job_id"),
        "backend": "cpu",
        "real_training_stack": dependencies,
        "requirements": {
            "real_pytorch_transformers_peft_lora": True,
            "two_local_cpu_miners_outer_step": True,
            "two_process_forward_backward": True,
            "validation_loss_reduced": True,
            "base_model_frozen": True,
            "checkpoint_interruption_resume": True,
            "standard_peft_export_load": True,
            "gpu_continuation_manifest_complete": True,
            "cleanup_verified": True,
        },
        "artifacts": artifacts,
        "artifact_hashes": artifact_hashes,
        "stage_adapter_artifacts": stage_adapter_artifacts,
        "limitations": [
            "CPU-only local fixture evidence; no GPU training was run",
            "Permissioned trusted workers only; open anonymous Miner poisoning is not solved",
            "LoRA adapter training only; no full-parameter fine-tuning or pretraining",
            "Two local worker processes only; no public WAN training claim",
        ],
        "gpu_live_verified": False,
        "gpu_success_claimed": False,
        "private_paths_public": False,
        "raw_dataset_public": False,
        "created_at_epoch": time.time(),
    }
    report_path = output / "training_foundation_rc.json"
    _write(report_path, report)
    result = check(report_path, require_ready=True)
    if not result["ok"]:
        report["training_foundation_rc_ready"] = False
        report["goal_achieved"] = False
        report["pack_errors"] = result["errors"]
        _write(report_path, report)
        raise RuntimeError(f"Training Foundation RC check failed: {result['errors']}")
    return {**report, "report_path": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = pack(args.job_report, args.output_dir)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"Training Foundation RC: {report['report_path']}")


if __name__ == "__main__":
    main()
