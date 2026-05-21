from __future__ import annotations

import math
import unittest

from crowdtensor.diloco import default_model
from crowdtensor.lora_mock import (
    adapter_loss,
    apply_adapter_update,
    lora_training_spec_for,
    run_lora_inner_loop,
    validate_adapter_delta,
)


class LoRAMockTests(unittest.TestCase):
    def test_lora_inner_loop_reduces_adapter_loss(self) -> None:
        model = default_model()
        spec = lora_training_spec_for("task-lora", "miner-lora", model)

        result = run_lora_inner_loop(spec, inner_steps=40)

        self.assertLess(result["adapter_loss_end"], result["adapter_loss_start"])
        self.assertEqual(len(result["adapter_delta"]["values"]), len(model["weights"]))
        self.assertEqual(result["adapter_delta"]["rank"], model["lora_adapter"]["rank"])

    def test_adapter_update_tracks_adapter_step(self) -> None:
        model = default_model()
        result = run_lora_inner_loop(
            lora_training_spec_for("task-lora", "miner-lora", model),
            inner_steps=20,
        )
        validation = validate_adapter_delta(model, result["adapter_delta"])

        updated = apply_adapter_update(model, validation["adapter_delta"])

        self.assertTrue(validation["accepted"])
        self.assertEqual(updated["adapter_step"], 1)
        self.assertNotEqual(updated["lora_adapter"]["values"], model["lora_adapter"]["values"])
        self.assertLess(adapter_loss(updated), adapter_loss(model))

    def test_rejects_invalid_adapter_delta(self) -> None:
        model = default_model()

        self.assertFalse(validate_adapter_delta(model, {"values": [1.0, 2.0]})["accepted"])
        self.assertFalse(validate_adapter_delta(model, {"values": [math.inf, 0.0, 0.0]})["accepted"])
        self.assertFalse(validate_adapter_delta(model, {"values": [100.0, 0.0, 0.0]})["accepted"])


if __name__ == "__main__":
    unittest.main()
