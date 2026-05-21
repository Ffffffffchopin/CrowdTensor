from __future__ import annotations

import unittest

from crowdtensor.diloco import apply_outer_update, default_model
from crowdtensor.outer_optimizer import (
    CONTRACT_VERSION,
    DELTA_FORMAT_DENSE_FLOAT,
    DELTA_FORMAT_SIGN_COMPRESSED,
    OPTIMIZER_DILOCO_MOMENTUM,
    OPTIMIZER_DILOCO_NESTEROV,
    apply_outer_optimizer_update,
    compress_sign_delta,
    decode_delta_payload,
    normalize_outer_optimizer_contract,
    optimizer_claim_spec,
)


class OuterOptimizerTests(unittest.TestCase):
    def test_claim_spec_exposes_dense_momentum_contract(self) -> None:
        model = default_model()
        spec = optimizer_claim_spec(model)

        self.assertEqual(spec["contract_version"], CONTRACT_VERSION)
        self.assertEqual(spec["optimizer_type"], OPTIMIZER_DILOCO_MOMENTUM)
        self.assertEqual(spec["delta_format"], DELTA_FORMAT_DENSE_FLOAT)
        self.assertEqual(spec["optimizer_step"], 0)
        self.assertEqual(spec["weight_count"], len(model["weights"]))

    def test_optimizer_update_matches_existing_diloco_outer_update(self) -> None:
        model = default_model()
        delta = [0.1, -0.2, 0.05]

        optimized, summary = apply_outer_optimizer_update(model, delta)
        existing = apply_outer_update(model, delta)

        self.assertEqual(optimized["weights"], existing["weights"])
        self.assertEqual(optimized["outer_velocity"], existing["outer_velocity"])
        self.assertEqual(optimized["optimizer_step"], existing["optimizer_step"])
        self.assertEqual(summary["optimizer_step_before"], 0)
        self.assertEqual(summary["optimizer_step_after"], 1)
        self.assertGreater(summary["delta_norm"], 0.0)
        self.assertGreater(summary["velocity_norm"], 0.0)
        self.assertEqual(summary["outer_update_norm"], summary["velocity_norm"])

    def test_nesterov_optimizer_uses_lookahead_outer_update(self) -> None:
        model = default_model(outer_optimizer_type=OPTIMIZER_DILOCO_NESTEROV)
        model["outer_velocity"] = [0.05, -0.1, 0.0]
        delta = [0.1, -0.2, 0.05]

        optimized, summary = apply_outer_optimizer_update(model, delta)

        momentum = model["outer_momentum"]
        next_velocity = [
            momentum * old_velocity + update
            for old_velocity, update in zip(model["outer_velocity"], delta)
        ]
        outer_update = [
            update + momentum * update_velocity
            for update, update_velocity in zip(delta, next_velocity)
        ]
        expected_weights = [
            weight + model["outer_lr"] * update
            for weight, update in zip(model["weights"], outer_update)
        ]
        self.assertEqual(optimized["outer_optimizer_type"], OPTIMIZER_DILOCO_NESTEROV)
        self.assertEqual(optimized["outer_velocity"], next_velocity)
        self.assertEqual(optimized["weights"], expected_weights)
        self.assertEqual(summary["optimizer_type"], OPTIMIZER_DILOCO_NESTEROV)
        self.assertGreater(summary["outer_update_norm"], 0.0)
        self.assertNotEqual(summary["outer_update_norm"], summary["velocity_norm"])

    def test_unsupported_outer_optimizer_is_rejected(self) -> None:
        model = default_model()
        model["outer_optimizer_type"] = "broken"
        model["outer_optimizer_contract"] = {
            **model["outer_optimizer_contract"],
            "optimizer_type": "broken",
        }

        with self.assertRaises(ValueError):
            apply_outer_optimizer_update(model, [0.1, -0.2, 0.05])

    def test_normalize_backfills_legacy_contract(self) -> None:
        contract = normalize_outer_optimizer_contract({
            "global_step": 3,
            "optimizer_step": 3,
            "outer_lr": 0.25,
            "outer_momentum": 0.8,
            "weights": [1.0, 2.0],
        })

        self.assertEqual(contract["contract_version"], CONTRACT_VERSION)
        self.assertEqual(contract["optimizer_type"], OPTIMIZER_DILOCO_MOMENTUM)
        self.assertEqual(contract["delta_format"], DELTA_FORMAT_DENSE_FLOAT)
        self.assertEqual(contract["optimizer_step"], 3)
        self.assertEqual(contract["outer_lr"], 0.25)
        self.assertEqual(contract["outer_momentum"], 0.8)
        self.assertEqual(contract["weight_count"], 2)

    def test_sign_compressed_delta_decodes_to_scaled_signs(self) -> None:
        compressed = compress_sign_delta([0.2, -0.4, 0.0])
        decoded, metadata = decode_delta_payload(compressed_delta=compressed)

        self.assertEqual(compressed["format"], DELTA_FORMAT_SIGN_COMPRESSED)
        self.assertEqual(compressed["signs"], [1, -1, 0])
        self.assertEqual(len(decoded), 3)
        self.assertAlmostEqual(decoded[0], 0.2)
        self.assertAlmostEqual(decoded[1], -0.2)
        self.assertEqual(decoded[2], 0.0)
        self.assertEqual(metadata["delta_format"], DELTA_FORMAT_SIGN_COMPRESSED)
        self.assertGreater(metadata["compression_ratio_estimate"], 0.0)

    def test_sign_compressed_delta_rejects_invalid_payloads(self) -> None:
        cases = [
            {"format": "broken", "encoding": "ternary_signs_v1", "scale": 1.0, "signs": [1]},
            {"format": "sign_compressed", "encoding": "broken", "scale": 1.0, "signs": [1]},
            {"format": "sign_compressed", "encoding": "ternary_signs_v1", "scale": -1.0, "signs": [1]},
            {"format": "sign_compressed", "encoding": "ternary_signs_v1", "scale": 1.0, "signs": [2]},
            {"format": "sign_compressed", "encoding": "ternary_signs_v1", "scale": 1.0, "signs": "1"},
        ]

        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    decode_delta_payload(compressed_delta=payload)


if __name__ == "__main__":
    unittest.main()
