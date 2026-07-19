from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from crowdtensor.named_tensor_optimizer import (
    apply_diloco_outer_step,
    compress_sign_with_error_feedback,
    decode_sign_transport,
    load_tensors,
)


class NamedTensorOptimizerTests(unittest.TestCase):
    def test_sign_transport_tracks_error_feedback(self) -> None:
        delta = {
            "a": torch.tensor([[1.0, -0.25], [0.0, 0.5]]),
            "b": torch.tensor([0.1, -0.2]),
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = compress_sign_with_error_feedback(
                delta,
                transport_path=root / "transport.safetensors",
                residual_path=root / "residual.safetensors",
            )
            decoded = decode_sign_transport(first)
            residual = load_tensors(root / "residual.safetensors")
        self.assertTrue(first["error_feedback"])
        self.assertGreater(first["compression_ratio"], 1.0)
        for name in delta:
            self.assertTrue(torch.allclose(decoded[name] + residual[name], delta[name]))

    def test_diloco_named_tensor_outer_step_advances_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = {"a": torch.zeros(2, 2), "b": torch.zeros(2)}
            delta0 = {"a": torch.ones(2, 2), "b": torch.ones(2) * 2}
            delta1 = {"a": torch.ones(2, 2) * 3, "b": torch.zeros(2)}
            save_file(base, str(root / "base.safetensors"))
            save_file(delta0, str(root / "delta0.safetensors"))
            save_file(delta1, str(root / "delta1.safetensors"))
            report = apply_diloco_outer_step(
                base_adapter_path=root / "base.safetensors",
                delta_paths=[root / "delta0.safetensors", root / "delta1.safetensors"],
                output_adapter_path=root / "global.safetensors",
                velocity_path=root / "velocity.safetensors",
                outer_step=3,
                adapter_version=7,
                outer_lr=0.5,
                momentum=0.0,
            )
            global_adapter = load_tensors(root / "global.safetensors")
        self.assertEqual(report["input_delta_count"], 2)
        self.assertEqual(report["outer_step_after"], 4)
        self.assertEqual(report["adapter_version_after"], 8)
        self.assertTrue(torch.equal(global_adapter["a"], torch.ones(2, 2)))
        self.assertTrue(torch.equal(global_adapter["b"], torch.ones(2) * 0.5))


if __name__ == "__main__":
    unittest.main()
