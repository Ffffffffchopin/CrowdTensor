from __future__ import annotations

import torch

from crowdtensor.qwen15b_four_gpu_worker import compare_adapter_states, compare_losses


def test_resume_comparison_accepts_fixed_fp16_tolerance() -> None:
    baseline = [
        {
            "model.layers.0.self_attn.q_proj.lora_A.weight": torch.tensor(
                [1.0, 2.0], dtype=torch.float16
            )
        }
    ]
    resumed = [
        {
            "model.layers.0.self_attn.q_proj.lora_A.weight": torch.tensor(
                [1.001, 2.001], dtype=torch.float16
            )
        }
    ]
    report = compare_adapter_states(baseline, resumed, atol=0.005, rtol=0.005)
    assert report["verified"] is True
    assert report["exact_match"] is False
    assert report["maximum_absolute_difference"] <= 0.005


def test_resume_comparison_rejects_missing_or_divergent_stage_tensor() -> None:
    baseline = [{"model.layers.0.x.lora_A.weight": torch.tensor([1.0])}]
    assert compare_adapter_states(baseline, [])["verified"] is False
    assert compare_adapter_states(
        baseline,
        [{"model.layers.0.x.lora_A.weight": torch.tensor([2.0])}],
    )["verified"] is False
    assert compare_adapter_states(
        baseline,
        [{"model.layers.0.x.lora_B.weight": torch.tensor([1.0])}],
    )["verified"] is False


def test_loss_resume_comparison_is_shape_and_tolerance_bound() -> None:
    assert compare_losses([2.0, 1.5], [2.001, 1.499])["verified"] is True
    assert compare_losses([2.0], [2.0, 1.5])["verified"] is False
    assert compare_losses([2.0, 1.5], [3.0, 1.5])["verified"] is False
