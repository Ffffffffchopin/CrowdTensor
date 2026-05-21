from __future__ import annotations

import math
import unittest

from crowdtensor.micro_transformer import (
    TOKEN_IDS,
    analytic_gradient_for_batch,
    analytic_gradient_for_sample,
    apply_micro_transformer_update,
    default_micro_transformer_model,
    finite_difference_gradient,
    micro_transformer_loss,
    micro_transformer_training_spec_for,
    parameter_count,
    run_micro_transformer_inner_loop,
    validate_micro_transformer_delta,
)


class MicroTransformerTests(unittest.TestCase):
    def test_inner_loop_reduces_lm_loss_and_returns_stable_delta(self) -> None:
        model = default_micro_transformer_model()
        spec = micro_transformer_training_spec_for("task-a", "miner-a", model)

        result = run_micro_transformer_inner_loop(spec, inner_steps=6)

        self.assertLess(result["lm_loss_end"], result["lm_loss_start"])
        self.assertEqual(len(result["local_delta"]), parameter_count(model["config"]))
        self.assertEqual(result["samples_seen"], 6)
        self.assertEqual(result["vocab_size"], model["config"]["vocab_size"])
        self.assertEqual(result["gradient_mode"], "analytic")

    def test_analytic_gradient_matches_finite_difference_oracle(self) -> None:
        model = default_micro_transformer_model()
        weights = list(model["weights"])
        config = model["config"]

        analytic = analytic_gradient_for_sample(weights, config, TOKEN_IDS, 2)
        finite_difference = finite_difference_gradient(
            list(weights),
            config,
            TOKEN_IDS,
            2,
            eps=1e-5,
        )

        dot = sum(left * right for left, right in zip(analytic, finite_difference))
        analytic_norm = math.sqrt(sum(value * value for value in analytic))
        finite_norm = math.sqrt(sum(value * value for value in finite_difference))
        cosine = dot / (analytic_norm * finite_norm)
        max_abs_diff = max(abs(left - right) for left, right in zip(analytic, finite_difference))
        self.assertGreater(cosine, 0.999)
        self.assertLess(max_abs_diff, 1e-6)

    def test_analytic_batch_gradient_lowers_full_corpus_loss(self) -> None:
        model = default_micro_transformer_model()
        weights = list(model["weights"])
        config = model["config"]
        gradient = analytic_gradient_for_batch(weights, config, TOKEN_IDS)
        updated = [weight - 0.08 * grad for weight, grad in zip(weights, gradient)]

        self.assertLess(
            micro_transformer_loss({"micro_transformer": {**model, "weights": updated}}),
            micro_transformer_loss({"micro_transformer": model}),
        )

    def test_outer_update_tracks_micro_transformer_optimizer_state(self) -> None:
        root_model = {"micro_transformer": default_micro_transformer_model()}
        spec = micro_transformer_training_spec_for("task-b", "miner-b", root_model["micro_transformer"])
        result = run_micro_transformer_inner_loop(spec, inner_steps=5)

        updated = apply_micro_transformer_update(root_model, result["local_delta"])

        self.assertEqual(updated["micro_transformer"]["version"], 1)
        self.assertEqual(updated["micro_transformer"]["optimizer_step"], 1)
        self.assertNotEqual(updated["micro_transformer"]["weights"], root_model["micro_transformer"]["weights"])
        self.assertTrue(math.isfinite(micro_transformer_loss(updated)))

    def test_validate_micro_transformer_delta_rejects_invalid_values(self) -> None:
        root_model = {"micro_transformer": default_micro_transformer_model()}
        valid_delta = [0.0 for _ in root_model["micro_transformer"]["weights"]]

        self.assertTrue(validate_micro_transformer_delta(root_model, valid_delta)["accepted"])
        self.assertFalse(validate_micro_transformer_delta(root_model, [1.0, 2.0])["accepted"])
        self.assertFalse(validate_micro_transformer_delta(root_model, [math.inf for _ in valid_delta])["accepted"])
        self.assertFalse(validate_micro_transformer_delta(root_model, [100.0 for _ in valid_delta])["accepted"])


if __name__ == "__main__":
    unittest.main()
