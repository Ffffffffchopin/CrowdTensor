from __future__ import annotations

import unittest

from crowdtensor.diloco import (
    DEFAULT_WEIGHTS,
    apply_outer_update,
    default_model,
    normalize_model,
    run_inner_loop,
    training_spec_for,
)


class DiLoCoTests(unittest.TestCase):
    def test_inner_loop_reduces_synthetic_loss(self) -> None:
        result = run_inner_loop(
            DEFAULT_WEIGHTS,
            task_id="task-a",
            miner_id="miner-a",
            model_version=0,
            inner_steps=40,
        )
        self.assertLess(result["inner_loss_end"], result["inner_loss_start"])
        self.assertEqual(len(result["local_delta"]), len(DEFAULT_WEIGHTS))
        self.assertEqual(result["samples_seen"], 40)

    def test_outer_update_tracks_optimizer_state(self) -> None:
        model = default_model()
        first = apply_outer_update(model, [0.1, -0.2, 0.05])
        second = apply_outer_update(first, [0.1, -0.2, 0.05])

        self.assertEqual(first["optimizer_step"], 1)
        self.assertEqual(second["optimizer_step"], 2)
        self.assertEqual(second["global_step"], 2)
        self.assertNotEqual(second["outer_velocity"], [0.1, -0.2, 0.05])
        self.assertNotEqual(second["weights"], model["weights"])

    def test_normalize_model_backfills_old_checkpoint_fields(self) -> None:
        old_model = {
            "version": 7,
            "global_step": 7,
            "weights": [1, 2, 3],
            "outer_lr": 0.25,
        }
        normalized = normalize_model(old_model)

        self.assertEqual(normalized["version"], 7)
        self.assertEqual(normalized["optimizer_step"], 7)
        self.assertEqual(normalized["outer_velocity"], [0.0, 0.0, 0.0])
        self.assertEqual(normalized["outer_momentum"], 0.9)
        self.assertEqual(normalized["outer_optimizer_type"], "diloco_momentum")
        self.assertEqual(normalized["outer_optimizer_contract"]["contract_version"], "outer_optimizer_contract_v1")
        self.assertEqual(normalized["outer_optimizer_contract"]["delta_format"], "dense_float")
        self.assertEqual(normalized["outer_optimizer_contract"]["optimizer_step"], 7)
        self.assertEqual(normalized["adapter_step"], 0)
        self.assertEqual(normalized["lora_adapter"]["values"], [0.0, 0.0, 0.0])

    def test_training_spec_controls_inner_loop_contract(self) -> None:
        spec = training_spec_for("task-a", "miner-a", 0)
        result = run_inner_loop(
            DEFAULT_WEIGHTS,
            task_id="task-a",
            miner_id="miner-a",
            model_version=0,
            inner_steps=40,
            training_spec=spec,
        )

        self.assertEqual(result["sample_offset"], spec["sample_offset"])
        self.assertEqual(result["local_delta_scale"], spec["local_delta_scale"])
        self.assertEqual(len(result["local_delta"]), len(DEFAULT_WEIGHTS))


if __name__ == "__main__":
    unittest.main()
