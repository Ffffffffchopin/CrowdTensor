from __future__ import annotations

import math
import unittest

from crowdtensor.diloco import default_model
from crowdtensor.validation import validate_local_delta


class ValidationTests(unittest.TestCase):
    def test_accepts_stable_delta(self) -> None:
        result = validate_local_delta(default_model(), [0.1, -0.2, 0.05])

        self.assertTrue(result["accepted"])
        self.assertEqual(result["code"], "ok")
        self.assertLess(result["delta_norm"], 5.0)
        self.assertIsNotNone(result["loss_delta"])

    def test_rejects_length_mismatch(self) -> None:
        result = validate_local_delta(default_model(), [0.1, 0.2])

        self.assertFalse(result["accepted"])
        self.assertEqual(result["code"], "delta_length_mismatch")

    def test_rejects_non_finite_delta(self) -> None:
        result = validate_local_delta(default_model(), [0.1, math.inf, 0.2])

        self.assertFalse(result["accepted"])
        self.assertEqual(result["code"], "delta_non_finite")

    def test_rejects_large_norm(self) -> None:
        result = validate_local_delta(default_model(), [100.0, 0.0, 0.0])

        self.assertFalse(result["accepted"])
        self.assertEqual(result["code"], "delta_norm_too_large")

    def test_rejects_loss_spike(self) -> None:
        result = validate_local_delta(default_model(), [-2.0, 2.0, -1.0])

        self.assertFalse(result["accepted"])
        self.assertEqual(result["code"], "loss_spike")
        self.assertGreater(result["loss_delta"], 1.0)


if __name__ == "__main__":
    unittest.main()
