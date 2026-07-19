from __future__ import annotations

import json

import pytest

from crowdtensor import cli
from crowdtensor.volunteer_training_cell import VolunteerTrainingCell
from crowdtensor.volunteer_training_cli import run


class OfflineTransport:
    def __getattr__(self, _name):
        raise RuntimeError("offline")


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
