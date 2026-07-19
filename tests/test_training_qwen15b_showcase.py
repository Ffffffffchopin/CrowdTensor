from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.training_qwen15b_showcase_check import check_report
from scripts.training_qwen15b_showcase_pack import build_showcase_report


ROOT = Path(__file__).resolve().parents[1]
LIVE_FIXTURE = ROOT / "dist/training-qwen15b-elastic-live-20260712-r2-repacked-achieved/training_qwen15b_elastic_live_probe.json"


def _live_fixture() -> dict:
    value = json.loads(LIVE_FIXTURE.read_text(encoding="utf-8"))
    value = copy.deepcopy(value)
    value.update(
        {
            "target_steps": 128,
            "replacement_generation_start_step": 64,
            "microbatches_per_step": 4,
            "learning_rate": 0.0005,
            "lora_rank": 4,
            "lora_alpha": 8,
            "training_budget": {
                "sequence_length": 128,
                "training_token_count": 65536,
            },
        }
    )
    value["source"].update(
        {
            "parameter_count": 1543714304,
            "dataset_id": "Salesforce/wikitext",
            "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "sequence_length": 128,
            "train_sequence_count": 512,
            "validation_sequence_count": 32,
            "train_token_hash": "sha256:" + "a" * 64,
            "validation_token_hash": "sha256:" + "b" * 64,
        }
    )
    value["evidence"].update(
        {
            "final_target_step_completed": True,
            "rendezvous_full_pipeline_verified": True,
            "exactly_once_optimizer_commits_verified": True,
        }
    )
    return value


def test_showcase_report_requires_large_budget_and_pinned_dataset() -> None:
    report = build_showcase_report(_live_fixture())
    assert report["showcase_ready"] is True
    assert report["evaluation"]["relative_validation_loss_improvement"] > 0.06
    assert check_report(report, require_ready=True)["ok"] is True

    blocked = _live_fixture()
    blocked["source"].pop("dataset_revision")
    blocked["training_budget"]["training_token_count"] = 1024
    report = build_showcase_report(blocked)
    assert report["showcase_ready"] is False
    assert "showcase_gate_pinned_public_dataset" in report["blockers"]
    assert "showcase_gate_large_bounded_training_budget" in report["blockers"]


def test_showcase_checker_rejects_unverified_validation_result() -> None:
    report = build_showcase_report(_live_fixture())
    report["evaluation"]["after_validation_loss"] = report["evaluation"]["before_validation_loss"]
    report["evaluation"]["validation_loss_reduced"] = False
    report["showcase_ready"] = False
    checked = check_report(report, require_ready=True)
    assert checked["ok"] is False
    assert "validation_loss_not_reduced" in checked["errors"]
    assert "showcase_not_ready" in checked["errors"]
