from __future__ import annotations

import json
from pathlib import Path

from scripts.volunteer_training_kaggle_contribution_probe import (
    KERNEL_REPORT,
    _kernel_source,
    _progress_delta,
    _snapshot_summary,
    build_private_packages,
)


def _invite() -> dict[str, str]:
    return {
        "schema": "crowdtensor_volunteer_training_invite_v1",
        "campaign_id": "founding-test",
        "coordinator_url": "https://coordinator.example",
        "invite_token": "private-invite-value",
    }


def test_private_kernel_runs_one_real_cuda_join_without_literal_invite() -> None:
    raw = json.dumps(_invite()).encode()
    source = _kernel_source(
        invite_bytes=raw,
        source_commit="a" * 40,
        cell_id="private-cell-id",
        contributor_index=1,
    )

    assert "private-invite-value" not in source
    assert "HTTPVolunteerTransport.from_invite" in source
    assert "VolunteerTrainingCell" in source
    assert "max_work_units=1" in source
    assert 'device="cuda:0"' in source
    assert KERNEL_REPORT in source
    assert "real_transformers_peft_lora" in source
    assert "private_invite_deleted" in source
    assert "raw_error_message_public" in source
    assert "traceback.extract_tb" in source


def test_private_packages_are_gpu_private_and_unique(tmp_path: Path) -> None:
    packages = build_private_packages(
        tmp_path,
        owner="authorized-owner",
        invite_bytes=json.dumps(_invite()).encode(),
        source_commit="b" * 40,
        contributor_count=2,
    )

    assert len(packages) == 2
    metadata = [
        json.loads((item["directory"] / "kernel-metadata.json").read_text())
        for item in packages
    ]
    assert len({item["id"] for item in metadata}) == 2
    assert all(item["is_private"] == "true" for item in metadata)
    assert all(item["enable_gpu"] == "true" for item in metadata)
    assert all(item["machine_shape"] == "NvidiaTeslaT4" for item in metadata)
    assert all(item["enable_internet"] == "true" for item in metadata)
    assert all(item["dataset_sources"] == [] for item in metadata)


def test_snapshot_summary_and_progress_delta_are_aggregate_only() -> None:
    before = _snapshot_summary(
        {
            "campaign": {
                "campaign_id": "founding-test",
                "campaign_manifest_hash": "sha256:" + "1" * 64,
                "model_id": "model",
                "dataset_id": "dataset",
            },
            "progress": {
                "lifecycle": "running",
                "adapter_version": 0,
                "completed_rounds": 0,
                "target_rounds": 100,
                "accepted_update_count": 0,
                "accepted_token_count": 0,
                "active_contributor_count": 0,
                "queued_work_count": 2,
            },
        }
    )
    after = {**before, "adapter_version": 1, "completed_rounds": 1}
    after["accepted_update_count"] = 2
    after["accepted_token_count"] = 32

    assert _progress_delta(before, after) == {
        "adapter_version": 1,
        "completed_rounds": 1,
        "accepted_update_count": 2,
        "accepted_token_count": 32,
    }
    assert "cell_id" not in before
    assert "invite" not in before
