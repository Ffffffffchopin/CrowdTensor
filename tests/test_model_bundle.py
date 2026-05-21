from __future__ import annotations

import unittest

from crowdtensor.diloco import default_model
from crowdtensor.model_bundle import (
    BUNDLE_ID,
    MODEL_BUNDLE_SCHEMA_VERSION,
    apply_model_bundle_update,
    model_bundle_loss,
    model_bundle_training_spec_for,
    run_model_bundle_inner_loop,
    validate_model_bundle_delta,
)


class ModelBundleTests(unittest.TestCase):
    def test_inner_loop_emits_identity_bound_delta_that_reduces_local_loss(self) -> None:
        model = default_model()
        spec = model_bundle_training_spec_for("task-a", "miner-a", model["model_bundle"])

        result = run_model_bundle_inner_loop(spec, inner_steps=6)
        validation = validate_model_bundle_delta(model, result["bundle_delta"])

        self.assertEqual(result["schema_version"], MODEL_BUNDLE_SCHEMA_VERSION)
        self.assertEqual(result["bundle_delta"]["bundle_id"], BUNDLE_ID)
        self.assertLess(result["bundle_loss_end"], result["bundle_loss_start"])
        self.assertGreater(result["delta_norm"], 0.0)
        self.assertTrue(validation["accepted"])
        self.assertEqual(validation["code"], "ok")
        self.assertEqual(validation["bundle_id"], BUNDLE_ID)

    def test_apply_update_advances_bundle_without_touching_dense_global_step(self) -> None:
        model = default_model()
        spec = model_bundle_training_spec_for("task-b", "miner-b", model["model_bundle"])
        result = run_model_bundle_inner_loop(spec, inner_steps=6)

        updated = apply_model_bundle_update(model, result["bundle_delta"])

        self.assertEqual(updated["global_step"], 0)
        self.assertEqual(updated["model_bundle"]["version"], 1)
        self.assertEqual(updated["model_bundle"]["optimizer_step"], 1)
        self.assertNotEqual(
            updated["model_bundle"]["artifact_hash"],
            model["model_bundle"]["artifact_hash"],
        )
        self.assertLessEqual(
            model_bundle_loss(updated),
            model_bundle_loss(model) + 1.0,
        )

    def test_validation_rejects_wrong_artifact_identity(self) -> None:
        model = default_model()
        spec = model_bundle_training_spec_for("task-c", "miner-c", model["model_bundle"])
        result = run_model_bundle_inner_loop(spec, inner_steps=6)
        bad_delta = {**result["bundle_delta"], "artifact_hash": "sha256:" + "0" * 64}

        validation = validate_model_bundle_delta(model, bad_delta)

        self.assertFalse(validation["accepted"])
        self.assertEqual(validation["code"], "model_bundle_artifact_hash_mismatch")

    def test_validation_rejects_non_numeric_base_version(self) -> None:
        model = default_model()
        spec = model_bundle_training_spec_for("task-d", "miner-d", model["model_bundle"])
        result = run_model_bundle_inner_loop(spec, inner_steps=6)
        bad_delta = {**result["bundle_delta"], "base_bundle_version": "not-an-int"}

        validation = validate_model_bundle_delta(model, bad_delta)

        self.assertFalse(validation["accepted"])
        self.assertEqual(validation["code"], "model_bundle_version_not_numeric")


if __name__ == "__main__":
    unittest.main()
