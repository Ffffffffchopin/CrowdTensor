#!/usr/bin/env python3
"""Generate public-safe evidence for CUDA LoRA adapter-delta rejection contracts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from crowdtensor.training_contract import delta_manifest, tensor_specs, validate_adapter_delta  # noqa: E402
from crowdtensor.named_tensor_optimizer import save_tensors  # noqa: E402


SCHEMA = "crowdtensor_cuda_training_rejection_matrix_v1"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest(private: Path, name: str, tensors: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    path = save_tensors(tensors, private / f"{name}.safetensors")
    values: dict[str, Any] = {
        "delta_path": path,
        "job_id": "cuda-job",
        "round_id": "cuda-round",
        "result_id": f"result-{name}",
        "miner_id": "cuda-miner",
        "model_manifest_hash": "sha256:model",
        "base_model_hash": "sha256:base",
        "base_adapter_hash": "sha256:adapter",
        "base_model_version": 1,
        "adapter_version": 0,
        "dataset_shard_index": 0,
        "dataset_shard_hash": "sha256:shard0",
        "loss_start": 2.0,
        "loss_end": 1.0,
        "samples_seen": 4,
        "tokens_seen": 32,
    }
    values.update(overrides)
    return delta_manifest(**values)


def build(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    private = output / ".private-rejection-tensors"
    private.mkdir(parents=True, exist_ok=True)
    base = {"layer.lora_A.weight": torch.full((2, 3), 0.01, dtype=torch.float32)}
    expected = {
        "job_id": "cuda-job",
        "round_id": "cuda-round",
        "model_manifest_hash": "sha256:model",
        "base_model_hash": "sha256:base",
        "base_adapter_hash": "sha256:adapter",
        "base_model_version": 1,
        "adapter_version": 0,
        "dataset_shard_index": 0,
        "dataset_shard_hash": "sha256:shard0",
        "tensor_specs": tensor_specs(base),
    }
    checks: dict[str, dict[str, Any]] = {}
    try:
        valid = _manifest(private, "valid", base)
        cases = {
            "duplicate_result": validate_adapter_delta(valid, expected=expected, seen_result_ids=["result-valid"]),
            "wrong_shard": validate_adapter_delta(
                valid,
                expected={**expected, "dataset_shard_hash": "sha256:other"},
            ),
            "stale_adapter_version": validate_adapter_delta(
                valid,
                expected={**expected, "adapter_version": 1},
            ),
            "stale_model_version": validate_adapter_delta(
                valid,
                expected={**expected, "base_model_version": 2},
            ),
            "shape_mismatch": validate_adapter_delta(
                valid,
                expected={**expected, "tensor_specs": [{**tensor_specs(base)[0], "shape": [3, 2]}]},
            ),
            "dtype_mismatch": validate_adapter_delta(
                valid,
                expected={**expected, "tensor_specs": [{**tensor_specs(base)[0], "dtype": "float16"}]},
            ),
            "nan": validate_adapter_delta(
                _manifest(private, "nan", {"layer.lora_A.weight": torch.tensor([[float("nan")]])}),
                expected={**expected, "tensor_specs": tensor_specs({"layer.lora_A.weight": torch.tensor([[float("nan")]])})},
            ),
            "infinity": validate_adapter_delta(
                _manifest(private, "inf", {"layer.lora_A.weight": torch.tensor([[float("inf")]])}),
                expected={**expected, "tensor_specs": tensor_specs({"layer.lora_A.weight": torch.tensor([[float("inf")]])})},
            ),
            "excessive_norm": validate_adapter_delta(valid, expected=expected, max_delta_norm=0.001),
            "loss_spike": validate_adapter_delta(
                _manifest(private, "loss-spike", base, loss_start=1.0, loss_end=2.0),
                expected=expected,
                max_loss_increase=0.25,
            ),
        }
        expected_codes = {
            "duplicate_result": "duplicate_result",
            "wrong_shard": "dataset_shard_hash_mismatch",
            "stale_adapter_version": "adapter_version_mismatch",
            "stale_model_version": "base_model_version_mismatch",
            "shape_mismatch": "adapter_delta_shape_mismatch",
            "dtype_mismatch": "adapter_delta_dtype_mismatch",
            "nan": "adapter_delta_non_finite",
            "infinity": "adapter_delta_non_finite",
            "excessive_norm": "adapter_delta_norm_too_large",
            "loss_spike": "training_loss_spike",
        }
        for name, result in cases.items():
            checks[name] = {
                "accepted": result.get("accepted") is True,
                "code": result.get("code"),
                "expected_code": expected_codes[name],
                "rejected_as_expected": result.get("accepted") is False and result.get("code") == expected_codes[name],
            }
        report = {
            "schema": SCHEMA,
            "ok": all(item["rejected_as_expected"] for item in checks.values()),
            "checks": checks,
            "required_case_count": len(expected_codes),
            "verified_case_count": sum(item["rejected_as_expected"] for item in checks.values()),
            "tensor_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
    finally:
        shutil.rmtree(private, ignore_errors=True)
    report["private_tensors_removed"] = not private.exists()
    report["ok"] = bool(report["ok"] and report["private_tensors_removed"])
    _write(output / "training_cuda_rejection_matrix.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build(args.output_dir)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"training_cuda_rejection_matrix ok={report['ok']}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
