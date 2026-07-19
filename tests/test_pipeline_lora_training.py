from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crowdtensor.pipeline_lora_training import compare_pipeline_runs, run_two_process_pipeline


class PipelineLoRATrainingTests(unittest.TestCase):
    def test_two_process_forward_backward_and_resume_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = run_two_process_pipeline(root / "baseline", total_steps=6)
            resumed = run_two_process_pipeline(
                root / "resumed",
                total_steps=6,
                interrupt_stage1_after_step=3,
            )
            comparison = compare_pipeline_runs(baseline, resumed)
        self.assertEqual(baseline["process_count"], 2)
        self.assertTrue(baseline["no_stage_loaded_full_model"])
        self.assertTrue(baseline["real_activation_transport"])
        self.assertTrue(baseline["real_backward_gradient_transport"])
        self.assertTrue(baseline["positive_lora_gradient_norms"])
        self.assertTrue(baseline["base_weights_frozen"])
        self.assertTrue(baseline["loss_reduced"])
        self.assertTrue(baseline["cleanup"]["all_worker_processes_stopped"])
        self.assertTrue(resumed["interruption"]["worker_restarted"])
        self.assertTrue(comparison["checkpoint_resume_verified"])


if __name__ == "__main__":
    unittest.main()
