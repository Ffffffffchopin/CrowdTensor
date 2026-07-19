import io
import json
import socket
import stat
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from crowdtensor.elastic_training_client import ElasticTrainingHTTPClient
from crowdtensor.elastic_training_runtime import (
    ElasticTrainingRuntime,
    build_qwen_stage_checkpoint_archive,
    install_elastic_training_routes,
    restore_qwen_stage_checkpoint_archive,
    validate_qwen_stage_checkpoint_archive,
)
from crowdtensor.qwen15b_training import (
    MODEL_ID,
    MODEL_REVISION,
    QWEN_STAGE_CHECKPOINT_SCHEMA,
    canonical_stage_specs,
    stable_hash,
)


def _sha(value: bytes) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(value).hexdigest()


def _checkpoint_archive(
    root: Path,
    *,
    stage_id: int,
    step: int,
    microbatches: int = 4,
    variant: str = "",
) -> bytes:
    spec = canonical_stage_specs()[stage_id]
    prefix = f"stage{stage_id}"
    values = {
        f"{prefix}_adapter.safetensors": f"adapter:{stage_id}:{step}:{variant}".encode(),
        f"{prefix}_optimizer.pt": f"optimizer:{stage_id}:{step}:{variant}".encode(),
        f"{prefix}_grad_scaler.pt": f"scaler:{stage_id}:{step}:{variant}".encode(),
        f"{prefix}_rng.pt": f"rng:{stage_id}:{step}:{variant}".encode(),
    }
    stage_root = root / f"stage-{stage_id}-step-{step}-{variant or 'default'}"
    stage_root.mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        (stage_root / name).write_bytes(value)
    manifest = {
        "schema": QWEN_STAGE_CHECKPOINT_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "stage_id": stage_id,
        "layer_start": spec.layer_start,
        "layer_end": spec.layer_end,
        "global_step": step,
        "optimizer_step": step,
        "dataset_cursor": step * microbatches,
        "device": spec.device,
        "adapter_file": f"{prefix}_adapter.safetensors",
        "adapter_file_hash": _sha(values[f"{prefix}_adapter.safetensors"]),
        "adapter_tensor_hash": _sha(f"adapter-tensors:{stage_id}:{step}:{variant}".encode()),
        "adapter_tensor_count": 2,
        "optimizer_file": f"{prefix}_optimizer.pt",
        "optimizer_file_hash": _sha(values[f"{prefix}_optimizer.pt"]),
        "grad_scaler_file": f"{prefix}_grad_scaler.pt",
        "grad_scaler_file_hash": _sha(values[f"{prefix}_grad_scaler.pt"]),
        "grad_scaler_state_present": True,
        "rng_file": f"{prefix}_rng.pt",
        "rng_file_hash": _sha(values[f"{prefix}_rng.pt"]),
        "rng_state_present": True,
        "tensor_values_public": False,
        "token_ids_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    manifest["content_hash"] = stable_hash(manifest)
    (stage_root / f"{prefix}_checkpoint.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    archive, _report = build_qwen_stage_checkpoint_archive(
        stage_root, stage_id=stage_id
    )
    return archive


def _register_pair(runtime: ElasticTrainingRuntime, *, prefix: str):
    first = runtime.register_miner(
        miner_id_hash=_sha(f"{prefix}:kernel-a".encode()),
        registration_nonce=f"{prefix}:nonce-a",
        supported_stage_ids=[0, 1],
        slot_count=2,
    )
    second = runtime.register_miner(
        miner_id_hash=_sha(f"{prefix}:kernel-b".encode()),
        registration_nonce=f"{prefix}:nonce-b",
        supported_stage_ids=[2, 3],
        slot_count=2,
    )
    return first, second


def _all_assignments(runtime: ElasticTrainingRuntime, sessions):
    values = []
    for session in sessions:
        response = runtime.assignments(
            session_id=session["session_id"],
            session_token=session["session_token"],
        )
        values.extend((session, item) for item in response["assignments"])
    return sorted(values, key=lambda value: value[1]["stage_id"])


def _submit_epoch(
    runtime: ElasticTrainingRuntime,
    sessions,
    root: Path,
    *,
    expected_step: int,
):
    assignments = _all_assignments(runtime, sessions)
    assert [item["stage_id"] for _session, item in assignments] == [0, 1, 2, 3]
    archives = {}
    results = []
    for session, assignment in assignments:
        assert assignment["target_step"] == expected_step
        archive = _checkpoint_archive(
            root, stage_id=assignment["stage_id"], step=expected_step
        )
        archives[assignment["stage_id"]] = archive
        result = runtime.submit_checkpoint(
            session_id=session["session_id"],
            session_token=session["session_token"],
            epoch_id=assignment["epoch_id"],
            stage_id=assignment["stage_id"],
            assignment_token=assignment["assignment_token"],
            archive=archive,
        )
        results.append(result)
    assert results[-1]["global_commit_created"] is True
    assert results[-1]["committed_step"] == expected_step
    return assignments, archives, results


def test_checkpoint_archive_validates_and_restores_without_path_traversal(tmp_path) -> None:
    archive = _checkpoint_archive(tmp_path, stage_id=2, step=4)
    report = validate_qwen_stage_checkpoint_archive(
        archive,
        expected_stage_id=2,
        expected_step=4,
        expected_dataset_cursor=16,
    )
    assert report["optimizer_state_present"] is True
    assert report["grad_scaler_state_present"] is True
    assert report["rng_state_present"] is True

    restored = tmp_path / "restored"
    restore = restore_qwen_stage_checkpoint_archive(
        archive,
        restored,
        expected_stage_id=2,
        expected_step=4,
        expected_dataset_cursor=16,
    )
    assert restore["archive_hash"] == report["archive_hash"]
    assert (restored / "stage2_checkpoint.json").is_file()
    assert len(list(restored.iterdir())) == 5

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("../stage2_checkpoint.json", b"{}")
        for index in range(4):
            bundle.writestr(f"component-{index}", b"x")
    with pytest.raises(ValueError, match="archive_path_invalid"):
        validate_qwen_stage_checkpoint_archive(stream.getvalue())


def test_elastic_full_offline_new_miners_resume_exactly_once(tmp_path) -> None:
    runtime = ElasticTrainingRuntime(
        tmp_path / "elastic.sqlite3",
        run_id="local-full-offline",
        target_steps=3,
        lease_seconds=30,
    )
    assert stat.S_IMODE((tmp_path / "elastic.sqlite3").stat().st_mode) == 0o600
    assert stat.S_IMODE(runtime.blob_dir.stat().st_mode) == 0o700
    assert runtime.public_status()["runtime_state"] == "paused_waiting_for_miners"
    old = _register_pair(runtime, prefix="old")
    status = runtime.public_status()
    assert status["runtime_state"] == "running"
    assert status["stage_coverage_complete"] is True

    _assignments1, _archives1, _results1 = _submit_epoch(
        runtime, old, tmp_path, expected_step=1
    )
    assignments2, archives2, results2 = _submit_epoch(
        runtime, old, tmp_path, expected_step=2
    )
    final_session, final_assignment = assignments2[-1]
    duplicate = runtime.submit_checkpoint(
        session_id=final_session["session_id"],
        session_token=final_session["session_token"],
        epoch_id=final_assignment["epoch_id"],
        stage_id=final_assignment["stage_id"],
        assignment_token=final_assignment["assignment_token"],
        archive=archives2[final_assignment["stage_id"]],
    )
    assert duplicate["idempotent"] is True
    assert duplicate["global_commit_created"] is False
    assert duplicate["committed_step"] == 2
    assert results2[-1]["global_commit_created"] is True

    # Step 3 has already been assigned to the old pair.  One speculative stage
    # candidate is submitted, then every old Miner disappears.
    aborted_assignments = _all_assignments(runtime, old)
    old_stage0_session, old_stage0 = aborted_assignments[0]
    aborted_archive = _checkpoint_archive(
        tmp_path, stage_id=0, step=3, variant="aborted"
    )
    runtime.submit_checkpoint(
        session_id=old_stage0_session["session_id"],
        session_token=old_stage0_session["session_token"],
        epoch_id=old_stage0["epoch_id"],
        stage_id=0,
        assignment_token=old_stage0["assignment_token"],
        archive=aborted_archive,
    )
    blob_files = list(runtime.blob_dir.glob("*/*.zip"))
    assert blob_files
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in blob_files)
    assert all(stat.S_IMODE(path.parent.stat().st_mode) == 0o700 for path in blob_files)
    runtime.mark_offline(
        session_id=old[0]["session_id"], session_token=old[0]["session_token"]
    )
    runtime.mark_offline(
        session_id=old[1]["session_id"], session_token=old[1]["session_token"]
    )
    paused = runtime.public_status()
    assert paused["committed_step"] == 2
    assert paused["runtime_state"] == "paused_waiting_for_miners"
    assert paused["zero_live_miners"] is True
    assert any(
        epoch["epoch_id"] == old_stage0["epoch_id"]
        and epoch["state"] == "aborted"
        for epoch in paused["epochs"]
    )

    old_stage1_session, old_stage1 = aborted_assignments[1]
    with pytest.raises(ValueError, match="assignment_stale"):
        runtime.submit_checkpoint(
            session_id=old_stage1_session["session_id"],
            session_token=old_stage1_session["session_token"],
            epoch_id=old_stage1["epoch_id"],
            stage_id=1,
            assignment_token=old_stage1["assignment_token"],
            archive=_checkpoint_archive(tmp_path, stage_id=1, step=3),
        )

    # Reopen the Coordinator state to prove pause/checkpoint durability before
    # entirely new Miner sessions join.
    runtime = ElasticTrainingRuntime(
        tmp_path / "elastic.sqlite3",
        run_id="local-full-offline",
        target_steps=3,
        lease_seconds=30,
    )
    assert runtime.public_status()["committed_step"] == 2
    new = _register_pair(runtime, prefix="new")
    resumed_assignments = _all_assignments(runtime, new)
    assert {item["base_step"] for _session, item in resumed_assignments} == {2}
    assert {item["target_step"] for _session, item in resumed_assignments} == {3}
    assert all(item["restore_required"] for _session, item in resumed_assignments)

    for session, assignment in resumed_assignments:
        archive, report = runtime.download_committed_checkpoint(
            session_id=session["session_id"],
            session_token=session["session_token"],
            epoch_id=assignment["epoch_id"],
            stage_id=assignment["stage_id"],
            assignment_token=assignment["assignment_token"],
        )
        assert archive == archives2[assignment["stage_id"]]
        assert report["global_step"] == 2

    assignments3, archives3, _results3 = _submit_epoch(
        runtime, new, tmp_path, expected_step=3
    )
    last_session, last_assignment = assignments3[-1]
    again = runtime.submit_checkpoint(
        session_id=last_session["session_id"],
        session_token=last_session["session_token"],
        epoch_id=last_assignment["epoch_id"],
        stage_id=last_assignment["stage_id"],
        assignment_token=last_assignment["assignment_token"],
        archive=archives3[last_assignment["stage_id"]],
    )
    assert again["idempotent"] is True

    achieved = runtime.public_status()
    assert achieved["runtime_state"] == "completed"
    assert achieved["committed_step"] == 3
    assert achieved["committed_steps"] == [1, 2, 3]
    assert achieved["optimizer_commit_count"] == 3
    assert achieved["committed_steps_contiguous"] is True
    old_sessions = {
        item["miner_session_hash"]
        for item in achieved["miners"]
        if item["miner_id_hash"] in {_sha(b"old:kernel-a"), _sha(b"old:kernel-b")}
    }
    new_sessions = {
        item["miner_session_hash"]
        for item in achieved["miners"]
        if item["miner_id_hash"] in {_sha(b"new:kernel-a"), _sha(b"new:kernel-b")}
    }
    assert len(old_sessions) == len(new_sessions) == 2
    assert old_sessions.isdisjoint(new_sessions)
    assert any(event["operation"] == "training_paused" for event in achieved["events"])
    assert sum(
        event["operation"] == "training_auto_woke" for event in achieved["events"]
    ) >= 2
    cleanup = runtime.cleanup_uncommitted_blobs()
    assert cleanup["uncommitted_blob_count_removed"] == 1
    assert cleanup["committed_blob_count_retained"] == 12


def test_lease_expiry_aborts_epoch_and_fences_old_session(tmp_path) -> None:
    now = [1000.0]
    runtime = ElasticTrainingRuntime(
        tmp_path / "lease.sqlite3",
        run_id="lease-expiry",
        target_steps=2,
        lease_seconds=5,
        clock=lambda: now[0],
    )
    old = _register_pair(runtime, prefix="lease-old")
    assignment = _all_assignments(runtime, old)[0][1]
    now[0] += 6
    paused = runtime.tick()
    assert paused["runtime_state"] == "paused_waiting_for_miners"
    assert paused["zero_live_miners"] is True
    assert paused["epochs"][0]["state"] == "aborted"
    with pytest.raises(ValueError, match="session_stale"):
        runtime.heartbeat(
            session_id=old[0]["session_id"], session_token=old[0]["session_token"]
        )
    with pytest.raises(ValueError, match="assignment_stale"):
        runtime.submit_checkpoint(
            session_id=old[0]["session_id"],
            session_token=old[0]["session_token"],
            epoch_id=assignment["epoch_id"],
            stage_id=assignment["stage_id"],
            assignment_token=assignment["assignment_token"],
            archive=_checkpoint_archive(tmp_path, stage_id=0, step=1),
        )


def test_concurrent_duplicate_final_submission_commits_optimizer_step_once(tmp_path) -> None:
    runtime = ElasticTrainingRuntime(
        tmp_path / "concurrent.sqlite3", run_id="concurrent", target_steps=1
    )
    sessions = _register_pair(runtime, prefix="concurrent")
    assignments = _all_assignments(runtime, sessions)
    for session, assignment in assignments[:3]:
        runtime.submit_checkpoint(
            session_id=session["session_id"],
            session_token=session["session_token"],
            epoch_id=assignment["epoch_id"],
            stage_id=assignment["stage_id"],
            assignment_token=assignment["assignment_token"],
            archive=_checkpoint_archive(
                tmp_path, stage_id=assignment["stage_id"], step=1
            ),
        )
    session, assignment = assignments[3]
    archive = _checkpoint_archive(tmp_path, stage_id=3, step=1)

    def submit():
        return runtime.submit_checkpoint(
            session_id=session["session_id"],
            session_token=session["session_token"],
            epoch_id=assignment["epoch_id"],
            stage_id=3,
            assignment_token=assignment["assignment_token"],
            archive=archive,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _value: submit(), range(2)))
    assert sum(item["global_commit_created"] is True for item in results) == 1
    assert sum(item["idempotent"] is True for item in results) == 1
    status = runtime.public_status()
    assert status["optimizer_commit_count"] == 1
    assert status["committed_steps"] == [1]


def test_http_routes_upload_and_download_central_checkpoint(tmp_path) -> None:
    runtime = ElasticTrainingRuntime(
        tmp_path / "http.sqlite3", run_id="http-live", target_steps=2
    )
    app = FastAPI()

    def authorize(value: str | None) -> None:
        if value != "coordinator-secret":
            raise HTTPException(status_code=401, detail="unauthorized")

    install_elastic_training_routes(app, runtime=runtime, authorize=authorize)
    client = TestClient(app)
    assert client.get("/elastic-training/status").status_code == 401
    common = {"x-crowdtensor-miner-token": "coordinator-secret"}
    invalid_inline = client.post(
        "/elastic-training/tensors/inline",
        headers=common,
        json={"envelope": {"chunk_count": 1}, "chunk_b64": "not-base64!"},
    )
    assert invalid_inline.status_code == 400
    assert invalid_inline.json()["detail"] == "elastic_inline_tensor_payload_invalid"

    sessions = []
    for role, stages in (("a", [0, 1]), ("b", [2, 3])):
        response = client.post(
            "/elastic-training/miners/register",
            headers=common,
            json={
                "run_id": "http-live",
                "miner_id_hash": _sha(f"http-{role}".encode()),
                "registration_nonce": f"nonce-{role}",
                "supported_stage_ids": stages,
                "slot_count": 2,
                "accelerator": "cuda",
            },
        )
        assert response.status_code == 200
        sessions.append(response.json())

    assignments = []
    for session in sessions:
        response = client.get(
            f"/elastic-training/miners/{session['session_id']}/assignments",
            headers={
                **common,
                "x-crowdtensor-elastic-session-token": session["session_token"],
            },
        )
        assert response.status_code == 200
        assignments.extend((session, item) for item in response.json()["assignments"])
    archives = {}
    for session, assignment in sorted(assignments, key=lambda value: value[1]["stage_id"]):
        stage_id = assignment["stage_id"]
        archive = _checkpoint_archive(tmp_path, stage_id=stage_id, step=1)
        archives[stage_id] = archive
        response = client.post(
            f"/elastic-training/checkpoints/{assignment['epoch_id']}/{stage_id}",
            headers={
                **common,
                "content-type": "application/vnd.crowdtensor.stage-checkpoint+zip",
                "x-crowdtensor-elastic-session-id": session["session_id"],
                "x-crowdtensor-elastic-session-token": session["session_token"],
                "x-crowdtensor-elastic-assignment-token": assignment[
                    "assignment_token"
                ],
            },
            content=archive,
        )
        assert response.status_code == 200
    assert response.json()["global_commit_created"] is True

    refreshed = client.get(
        f"/elastic-training/miners/{sessions[0]['session_id']}/assignments",
        headers={
            **common,
            "x-crowdtensor-elastic-session-token": sessions[0]["session_token"],
        },
    ).json()
    stage0 = next(item for item in refreshed["assignments"] if item["stage_id"] == 0)
    download = client.get(
        f"/elastic-training/checkpoints/{stage0['epoch_id']}/0",
        headers={
            **common,
            "x-crowdtensor-elastic-session-id": sessions[0]["session_id"],
            "x-crowdtensor-elastic-session-token": sessions[0]["session_token"],
            "x-crowdtensor-elastic-assignment-token": stage0["assignment_token"],
        },
    )
    assert download.status_code == 200
    assert download.content == archives[0]
    assert download.headers["x-crowdtensor-global-step"] == "1"

    public = client.get("/elastic-training/status", headers=common).json()
    encoded = json.dumps(public, sort_keys=True)
    assert public["committed_step"] == 1
    for session in sessions:
        assert session["session_token"] not in encoded
        assert session["session_id"] not in encoded
    for _session, assignment in assignments:
        assert assignment["assignment_token"] not in encoded


def test_real_http_miner_client_rejoins_with_fresh_checkpoint_directory(tmp_path) -> None:
    import uvicorn

    runtime = ElasticTrainingRuntime(
        tmp_path / "client.sqlite3",
        run_id="client-live",
        target_steps=2,
        lease_seconds=5,
    )
    app = FastAPI()

    def authorize(value: str | None) -> None:
        if value != "private-coordinator-token":
            raise HTTPException(status_code=401, detail="unauthorized")

    install_elastic_training_routes(app, runtime=runtime, authorize=authorize)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        port = int(handle.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started

    def miner(name, stages):
        return ElasticTrainingHTTPClient(
            coordinator_url=f"http://127.0.0.1:{port}",
            coordinator_token="private-coordinator-token",
            run_id="client-live",
            miner_id_hash=_sha(name.encode()),
            registration_nonce=f"nonce:{name}",
            supported_stage_ids=stages,
            slot_count=2,
            heartbeat_interval_seconds=0.1,
        )

    old_clients = [miner("old-a", [0, 1]), miner("old-b", [2, 3])]
    try:
        for client in old_clients:
            client.register()
            client.start_heartbeat()
        assignment_sets = [
            client.wait_for_assignments(
                expected_stage_ids=client.supported_stage_ids,
                expected_base_step=0,
                timeout=5,
            )["assignments"]
            for client in old_clients
        ]
        for client, assignments in zip(old_clients, assignment_sets, strict=True):
            checkpoint_dir = tmp_path / f"old-{client.supported_stage_ids[0]}"
            for assignment in assignments:
                stage_id = assignment["stage_id"]
                archive = _checkpoint_archive(tmp_path, stage_id=stage_id, step=1)
                restore_qwen_stage_checkpoint_archive(
                    archive,
                    checkpoint_dir,
                    expected_stage_id=stage_id,
                    expected_step=1,
                    expected_dataset_cursor=4,
                )
            for assignment in assignments:
                client.submit_checkpoint(assignment, checkpoint_dir=checkpoint_dir)
        assert runtime.public_status()["committed_step"] == 1
        for client in old_clients:
            client.offline()
        assert runtime.public_status()["runtime_state"] == "paused_waiting_for_miners"

        new_clients = [miner("new-a", [0, 1]), miner("new-b", [2, 3])]
        for client in new_clients:
            client.register()
            client.start_heartbeat()
        for client in new_clients:
            response = client.wait_for_assignments(
                expected_stage_ids=client.supported_stage_ids,
                expected_base_step=1,
                timeout=5,
            )
            fresh = tmp_path / f"new-{client.supported_stage_ids[0]}"
            assert not fresh.exists()
            for assignment in response["assignments"]:
                restored = client.download_checkpoint(
                    assignment, checkpoint_dir=fresh
                )
                assert restored["global_step"] == 1
            assert len(list(fresh.glob("stage*_checkpoint.json"))) == 2
            assert client.public_report()["checkpoint_download_count"] == 2
        for client in new_clients:
            client.offline()
    finally:
        for client in old_clients:
            client.stop_heartbeat()
        for client in locals().get("new_clients", []):
            client.stop_heartbeat()
        server.should_exit = True
        thread.join(timeout=10)
    assert not thread.is_alive()
