from __future__ import annotations

import json
import hashlib
import socket
import stat
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from crowdtensor.backends.volunteer_session import (
    prepare_volunteer_session,
)
from crowdtensor.core import inspect_workspace
from crowdtensor.hf_lora_training import create_local_training_fixture
from crowdtensor.volunteer_training_api import PUBLIC_RELEASE_ARTIFACT_NAMES
from crowdtensor.version import __version__
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator


def _campaign(tmp_path, *, target_rounds: int = 1) -> VolunteerTrainingCoordinator:
    fixture = create_local_training_fixture(
        tmp_path / "fixture", row_count=8, local_steps=1
    )
    return VolunteerTrainingCoordinator.create_from_fixture(
        tmp_path / "campaign",
        fixture,
        target_rounds=target_rounds,
        lease_seconds=120,
    )


def _free_port() -> int:
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _release_dir(root) -> None:
    root.mkdir()
    manifest_names = PUBLIC_RELEASE_ARTIFACT_NAMES - {"SHA256SUMS", "release.json"}
    artifacts = []
    for name in sorted(manifest_names):
        path = root / name
        path.write_text(name + "\n", encoding="utf-8")
        artifacts.append(
            {
                "name": name,
                "byte_count": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (root / "release.json").write_text(
        json.dumps(
            {
                "schema": "crowdtensor_release_v1",
                "version": __version__,
                "commit": "test",
                "artifacts": artifacts,
                "public_artifact_safe": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            + "  "
            + name
            + "\n"
            for name in sorted(PUBLIC_RELEASE_ARTIFACT_NAMES - {"SHA256SUMS"})
        ),
        encoding="utf-8",
    )


def test_prepared_session_exposes_concurrent_v2_claims_without_owner_ids(
    tmp_path,
) -> None:
    coordinator = _campaign(tmp_path)
    prepared = prepare_volunteer_session(
        tmp_path / "operator", campaign_dir=coordinator.root
    )
    client = TestClient(prepared.app)
    health = client.get("/v1/volunteer/health").json()
    assert health["v2_session_controller"] is True
    assert health["concurrent_elastic_work"] is True

    headers = {
        "Authorization": "Bearer " + coordinator.private_invite()["invite_token"]
    }
    for cell_id in ("contributor-a", "contributor-b"):
        response = client.post(
            "/v1/volunteer/work/claim",
            headers=headers,
            json={"cell_id": cell_id, "capability": {"device": "cpu"}},
        )
        assert response.status_code == 200
        assert response.json()["state"] == "leased"

    session = client.get("/v1/volunteer/session").json()
    assert session["active_work_count"] == 2
    assert session["concurrent_elastic_work_supported"] is True
    serialized = json.dumps(session, sort_keys=True)
    assert "contributor-a" not in serialized
    assert "contributor-b" not in serialized
    assert "contributor_id_hash" not in serialized


def test_top_level_run_prepares_new_v2_operator_workspace(tmp_path, capsys) -> None:
    coordinator = _campaign(tmp_path)
    release = tmp_path / "release"
    _release_dir(release)
    from crowdtensor.cli import main

    with pytest.raises(SystemExit) as result:
        main(
            [
                "train",
                "run",
                str(tmp_path / "operator"),
                "--campaign-dir",
                str(coordinator.root),
                "--release-dir",
                str(release),
                "--prepare-only",
                "--json",
            ]
        )
    assert result.value.code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["command_ok"] is True
    assert report["state"] == "prepared"
    assert report["controller_owned_by_session_user"] is True
    assert report["concurrent_elastic_work_supported"] is True
    assert report["coordinator_restart_recovery_performed"] is False
    assert report["coordinator_restart_recovery_verified"] is False
    assert report["public_release_download"] is True
    assert inspect_workspace(tmp_path / "operator")["lifecycle_state"] == "ready"
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)
    assert stat.S_IMODE(coordinator.invite_path.stat().st_mode) == 0o600


def test_ordinary_join_runs_real_cpu_peft_over_user_owned_session(
    tmp_path, capsys
) -> None:
    coordinator = _campaign(tmp_path)
    port = _free_port()
    prepared = prepare_volunteer_session(
        tmp_path / "operator",
        campaign_dir=coordinator.root,
        host="127.0.0.1",
        port=port,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            prepared.app,
            host="127.0.0.1",
            port=port,
            log_level="error",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                if httpx.get(endpoint + "/v1/volunteer/health", timeout=0.5).is_success:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.02)
        else:
            raise AssertionError("temporary Volunteer Session did not start")

        from crowdtensor.cli import main

        pairing = coordinator.create_pairing_code(
            invite_token=coordinator.private_invite()["invite_token"],
            mode="agent",
            ttl_seconds=120,
        )
        pairing_code = pairing["pairing_code"]
        contributor = tmp_path / "contributor"
        with pytest.raises(SystemExit) as blocked_preflight:
            main(
                [
                    "train",
                    "join",
                    str(contributor),
                    "--coordinator-url",
                    endpoint,
                    "--code",
                    pairing_code,
                    "--device",
                    "cpu",
                    "--max-download-gib",
                    "0.000001",
                    "--dry-run",
                    "--json",
                ]
            )
        assert blocked_preflight.value.code == 1
        blocked_report = json.loads(capsys.readouterr().out)
        assert blocked_report["state"] == "preflight_blocked"
        assert blocked_report["resource_ready"] is False
        assert blocked_report["blockers"] == ["campaign_download_exceeds_limit"]
        with pytest.raises(SystemExit) as preflight:
            main(
                [
                    "train",
                    "join",
                    str(contributor),
                    "--coordinator-url",
                    endpoint,
                    "--code",
                    pairing_code,
                    "--device",
                    "cpu",
                    "--dry-run",
                    "--json",
                ]
            )
        assert preflight.value.code == 0
        preflight_report = json.loads(capsys.readouterr().out)
        assert preflight_report["state"] == "preflight_ready"
        assert preflight_report["resource_ready"] is True
        assert preflight_report["resource_preflight"]["requirements_known"] is True
        assert preflight_report["work_claimed"] is False
        enrollment = contributor / ".crowdtensor/contributor/.private/agent_enrollment.json"
        assert not enrollment.exists()

        with pytest.raises(SystemExit) as result:
            main(
                [
                    "train",
                    "join",
                    str(contributor),
                    "--coordinator-url",
                    endpoint,
                    "--code",
                    pairing_code,
                    "--device",
                    "cpu",
                    "--max-local-steps",
                    "1",
                    "--max-work-units",
                    "1",
                    "--poll-interval-seconds",
                    "0.01",
                    "--timeout-seconds",
                    "30",
                    "--json",
                ]
            )
        assert result.value.code == 0
        report = json.loads(capsys.readouterr().out)
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert not thread.is_alive()
    assert report["command_ok"] is True
    assert report["real_peft_work_completed"] is True
    assert report["completed_work_units"] == 1
    assert report["v2_session_controller_verified"] is True
    assert report["concurrent_elastic_work_verified"] is True
    assert report["workspace_state"] == "ready"
    assert report["local_status_page_enabled"] is True
    assert report["local_status_endpoint"].startswith("http://127.0.0.1:")
    assert report["graceful_signal_stop"] is True
    assert inspect_workspace(tmp_path / "contributor")["last_action"] == "join"
    assert enrollment.is_file()
    assert stat.S_IMODE(enrollment.stat().st_mode) == 0o600
    serialized = json.dumps(report, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert coordinator.private_invite()["invite_token"] not in serialized
    assert "cell-" not in serialized
    assert stat.S_IMODE(coordinator.invite_path.stat().st_mode) == 0o600
