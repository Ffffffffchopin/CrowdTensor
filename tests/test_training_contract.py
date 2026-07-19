from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from crowdtensor.training_contract import delta_manifest, tensor_specs, validate_adapter_delta


class TrainingContractTests(unittest.TestCase):
    def _manifest(self, root: Path, tensors: dict[str, torch.Tensor], **overrides: object) -> dict:
        path = root / f"delta-{len(list(root.iterdir()))}.safetensors"
        save_file(tensors, str(path))
        values = {
            "delta_path": path,
            "job_id": "job-1",
            "round_id": "round-1",
            "result_id": "result-1",
            "miner_id": "miner-1",
            "model_manifest_hash": "sha256:model",
            "base_model_hash": "sha256:base",
            "base_adapter_hash": "sha256:adapter",
            "base_model_version": 1,
            "adapter_version": 0,
            "dataset_shard_index": 0,
            "dataset_shard_hash": "sha256:shard0",
            "loss_start": 2.0,
            "loss_end": 1.5,
            "samples_seen": 4,
            "tokens_seen": 32,
        }
        values.update(overrides)
        return delta_manifest(**values)

    def _expected(self, tensors: dict[str, torch.Tensor]) -> dict:
        return {
            "job_id": "job-1",
            "round_id": "round-1",
            "model_manifest_hash": "sha256:model",
            "base_model_hash": "sha256:base",
            "base_adapter_hash": "sha256:adapter",
            "base_model_version": 1,
            "adapter_version": 0,
            "dataset_shard_index": 0,
            "dataset_shard_hash": "sha256:shard0",
            "tensor_specs": tensor_specs(tensors),
        }

    def test_accepts_real_named_tensor_delta(self) -> None:
        tensors = {"layer.lora_A.weight": torch.ones(2, 3) * 0.01}
        with tempfile.TemporaryDirectory() as temp:
            manifest = self._manifest(Path(temp), tensors)
            result = validate_adapter_delta(manifest, expected=self._expected(tensors))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["tensor_count"], 1)

    def test_rejects_duplicate_version_shard_and_shape(self) -> None:
        tensors = {"layer.lora_A.weight": torch.ones(2, 3) * 0.01}
        with tempfile.TemporaryDirectory() as temp:
            manifest = self._manifest(Path(temp), tensors)
            duplicate = validate_adapter_delta(
                manifest,
                expected=self._expected(tensors),
                seen_result_ids=["result-1"],
            )
            stale = validate_adapter_delta(
                manifest,
                expected={**self._expected(tensors), "adapter_version": 2},
            )
            stale_model = validate_adapter_delta(
                manifest,
                expected={**self._expected(tensors), "base_model_version": 2},
            )
            wrong_shard = validate_adapter_delta(
                manifest,
                expected={**self._expected(tensors), "dataset_shard_hash": "sha256:other"},
            )
            shape = validate_adapter_delta(
                manifest,
                expected={**self._expected(tensors), "tensor_specs": [{
                    **tensor_specs(tensors)[0],
                    "shape": [3, 2],
                }]},
            )
        self.assertEqual(duplicate["code"], "duplicate_result")
        self.assertEqual(stale["code"], "adapter_version_mismatch")
        self.assertEqual(stale_model["code"], "base_model_version_mismatch")
        self.assertEqual(wrong_shard["code"], "dataset_shard_hash_mismatch")
        self.assertEqual(shape["code"], "adapter_delta_shape_mismatch")

    def test_rejects_non_finite_large_norm_and_loss_spike(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nan_tensors = {"layer.lora_A.weight": torch.tensor([[float("nan")]])}
            nan_manifest = self._manifest(root, nan_tensors, result_id="nan")
            non_finite = validate_adapter_delta(nan_manifest, expected=self._expected(nan_tensors))

            inf_tensors = {"layer.lora_A.weight": torch.tensor([[float("inf")]])}
            inf_manifest = self._manifest(root, inf_tensors, result_id="inf")
            infinite = validate_adapter_delta(inf_manifest, expected=self._expected(inf_tensors))

            large_tensors = {"layer.lora_A.weight": torch.ones(2, 3) * 10.0}
            large_manifest = self._manifest(root, large_tensors, result_id="large")
            large_norm = validate_adapter_delta(
                large_manifest,
                expected=self._expected(large_tensors),
                max_delta_norm=1.0,
            )

            spike_tensors = {"layer.lora_A.weight": torch.ones(2, 3) * 0.01}
            spike_manifest = self._manifest(
                root,
                spike_tensors,
                result_id="spike",
                loss_start=1.0,
                loss_end=2.0,
            )
            loss_spike = validate_adapter_delta(
                spike_manifest,
                expected=self._expected(spike_tensors),
                max_loss_increase=0.25,
            )
        self.assertEqual(non_finite["code"], "adapter_delta_non_finite")
        self.assertEqual(infinite["code"], "adapter_delta_non_finite")
        self.assertEqual(large_norm["code"], "adapter_delta_norm_too_large")
        self.assertEqual(loss_spike["code"], "training_loss_spike")


if __name__ == "__main__":
    unittest.main()
