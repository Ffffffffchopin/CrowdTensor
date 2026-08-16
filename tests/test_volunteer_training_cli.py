from __future__ import annotations

import json

import pytest

from crowdtensor import cli
from crowdtensor.hf_lora_training import create_local_training_fixture
from crowdtensor.volunteer_training_cell import VolunteerTrainingCell
from crowdtensor.volunteer_training_cli import parse_args, run
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator


class OfflineTransport:
    def __getattr__(self, _name):
        raise RuntimeError("offline")


def test_commons_campaign_cli_has_bounded_reviewed_defaults() -> None:
    args = parse_args(
        [
            "campaign",
            "import-commons",
            "./campaign",
            "--model-dir",
            "./model",
            "--train-data-pack",
            "./train-pack",
            "--evaluation-data-pack",
            "./heldout-pack",
            "--attest-model-source",
        ]
    )
    assert args.target_rounds == 100
    assert args.work_shards == 4
    assert args.minimum_quorum == 4
    assert args.local_steps == 1
    assert args.attest_model_source is True


def test_campaign_cli_exposes_operator_trusted_evaluation_import() -> None:
    args = parse_args(
        ["campaign", "import-evaluation", "./campaign", "./evaluation.json"]
    )
    assert args.campaign_action == "import-evaluation"
    assert args.campaign_dir == "./campaign"
    assert args.report == "./evaluation.json"


def test_top_level_cli_dispatches_volunteer_contract(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["volunteer", "contract", "--json"])
    assert raised.value.code == 0
    value = json.loads(capsys.readouterr().out)
    assert value["binary_safetensors_submission"] is True
    assert value["raw_tensor_json_transport"] is False


def test_cell_pause_resume_status_and_cleanup_are_public_safe(tmp_path) -> None:
    cell = VolunteerTrainingCell(OfflineTransport(), tmp_path / "cell", cell_id="private-cell")
    paused = cell.pause()
    assert paused["state"] == "paused"
    assert cell.local_status()["cell_id_hash"].startswith("sha256:")
    assert cell.resume()["state"] == "ready"
    cleanup = cell.cleanup()
    assert cleanup["live_resources_left_running"] is False
    serialized = (tmp_path / "cell" / "status.json").read_text(encoding="utf-8")
    assert "private-cell" not in serialized


def test_join_rejects_world_readable_private_invite(tmp_path, capsys) -> None:
    invite = tmp_path / "invite.json"
    invite.write_text(
        json.dumps(
            {
                "schema": "crowdtensor_volunteer_training_invite_v1",
                "coordinator_url": "http://127.0.0.1:1",
                "invite_token": "super-secret-token-value",
            }
        ),
        encoding="utf-8",
    )
    invite.chmod(0o644)
    code = run(["join", str(invite), "--dry-run", "--json"])
    value = json.loads(capsys.readouterr().out)
    assert code == 2
    assert value["error"] == "volunteer_invite_file_permissions_too_open"
    serialized = json.dumps(value)
    assert "super-secret-token-value" not in serialized
    assert '"invite_token"' not in serialized


def test_campaign_lifecycle_commands_do_not_require_release_dir(tmp_path, capsys) -> None:
    fixture = create_local_training_fixture(tmp_path / "fixture", row_count=4, local_steps=1)
    root = tmp_path / "campaign"
    VolunteerTrainingCoordinator.create_from_fixture(
        root, fixture, target_rounds=1, lease_seconds=60
    )

    assert run(["campaign", "validate", str(root), "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["ok"] is True
    assert value["errors"] == []

    assert (
        run(
            [
                "serve",
                str(root),
                "--host",
                "127.0.0.1",
                "--port",
                "8791",
                "--prepare-only",
                "--json",
            ]
        )
        == 0
    )
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["state"] == "prepared"
    assert prepared["public_release_download"] is False
