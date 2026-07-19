from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from crowdtensor import cli
from crowdtensor.elastic_checkpoint_storage import S3CheckpointBlobStore
from crowdtensor.elastic_training_beta import (
    ElasticTrainingBetaController,
    create_elastic_training_beta_app,
)
from crowdtensor.elastic_training_runtime import (
    ElasticTrainingRuntime,
    build_qwen_stage_checkpoint_archive,
    sign_checkpoint_submission,
)
from crowdtensor.elastic_training_miner import parse_training_join_args
from crowdtensor.qwen15b_training import (
    MODEL_ID,
    MODEL_REVISION,
    QWEN_LORA_TARGET_MODULES,
    QWEN_STAGE_CHECKPOINT_SCHEMA,
    canonical_stage_specs,
    stable_hash,
)


def _sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _inputs(root: Path) -> tuple[Path, Path]:
    config = root / "config.json"
    tokenized = root / "tokenized.json"
    config.write_text(
        json.dumps({"model_type": "qwen2", "num_hidden_layers": 28}),
        encoding="utf-8",
    )
    tokenized.write_text(
        json.dumps(
            {
                "schema": "crowdtensor_qwen15b_tokenized_private_v1",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "sequence_length": 64,
                "train": [[index] * 64 for index in range(32)],
                "validation": [[index] * 64 for index in range(4)],
            }
        ),
        encoding="utf-8",
    )
    return config, tokenized


def _secure_archive(
    root: Path,
    *,
    stage_id: int,
    step: int = 1,
    non_finite: bool = False,
) -> bytes:
    import torch
    from safetensors.torch import save_file

    spec = canonical_stage_specs()[stage_id]
    stage_root = root / f"secure-stage-{stage_id}-{step}-{int(non_finite)}"
    stage_root.mkdir(parents=True)
    prefix = f"stage{stage_id}"
    adapter = {}
    for layer in range(spec.layer_start, spec.layer_end):
        for target in QWEN_LORA_TARGET_MODULES:
            owner = "self_attn" if target in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
            for side in ("A", "B"):
                name = f"model.layers.{layer}.{owner}.{target}.lora_{side}.weight"
                adapter[name] = torch.tensor(
                    [[float("nan") if non_finite and not adapter else float(layer + 1)]],
                    dtype=torch.float32,
                )
    adapter_path = stage_root / f"{prefix}_adapter.safetensors"
    save_file(adapter, str(adapter_path))
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter.items()):
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(len(raw).to_bytes(8, "little") + raw)
    optimizer_path = stage_root / f"{prefix}_optimizer.pt"
    scaler_path = stage_root / f"{prefix}_grad_scaler.pt"
    rng_path = stage_root / f"{prefix}_rng.pt"
    torch.save({"state": {}, "param_groups": []}, optimizer_path)
    torch.save({}, scaler_path)
    torch.save({"cpu": torch.random.get_rng_state()}, rng_path)
    manifest = {
        "schema": QWEN_STAGE_CHECKPOINT_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "stage_id": stage_id,
        "layer_start": spec.layer_start,
        "layer_end": spec.layer_end,
        "global_step": step,
        "optimizer_step": step,
        "dataset_cursor": step * 4,
        "device": spec.device,
        "adapter_file": adapter_path.name,
        "adapter_file_hash": _sha(adapter_path.read_bytes()),
        "adapter_tensor_hash": "sha256:" + digest.hexdigest(),
        "adapter_tensor_count": len(adapter),
        "optimizer_file": optimizer_path.name,
        "optimizer_file_hash": _sha(optimizer_path.read_bytes()),
        "grad_scaler_file": scaler_path.name,
        "grad_scaler_file_hash": _sha(scaler_path.read_bytes()),
        "grad_scaler_state_present": True,
        "rng_file": rng_path.name,
        "rng_file_hash": _sha(rng_path.read_bytes()),
        "rng_state_present": True,
        "tensor_values_public": False,
        "token_ids_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    manifest["content_hash"] = stable_hash(manifest)
    (stage_root / f"{prefix}_checkpoint.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    archive, _ = build_qwen_stage_checkpoint_archive(stage_root, stage_id=stage_id)
    return archive


def _register_all_stage_pair(runtime: ElasticTrainingRuntime):
    sessions = [
        runtime.register_miner(
            miner_id_hash=_sha(f"miner-{index}".encode()),
            registration_nonce=f"nonce-{index}",
            supported_stage_ids=[0, 1, 2, 3],
            slot_count=2,
        )
        for index in range(2)
    ]
    assignments = [
        runtime.assignments(
            session_id=session["session_id"],
            session_token=session["session_token"],
        )["assignments"]
        for session in sessions
    ]
    return sessions, assignments


def test_all_stage_miners_receive_executable_contiguous_groups(tmp_path) -> None:
    runtime = ElasticTrainingRuntime(tmp_path / "state.sqlite3", run_id="groups")
    _sessions, assignments = _register_all_stage_pair(runtime)
    groups = {tuple(sorted(item["stage_id"] for item in values)) for values in assignments}
    assert groups == {(0, 1), (2, 3)}
    status = runtime.public_status()
    assert status["topology_aware_stage_groups"] == [[0, 1], [2, 3]]
    assert status["active_stage_ids"] == [0, 1, 2, 3]
    assert status["missing_stage_ids"] == []
    assert status["pause_reason"] == ""


def test_secure_runtime_rejects_bad_signature_and_non_finite_adapter(tmp_path) -> None:
    runtime = ElasticTrainingRuntime(
        tmp_path / "secure.sqlite3",
        run_id="secure",
        target_steps=1,
        require_checkpoint_signatures=True,
        validate_checkpoint_tensors=True,
        max_rejected_submissions_per_session=2,
    )
    sessions, assignment_sets = _register_all_stage_pair(runtime)
    assignment_by_stage = {
        item["stage_id"]: (session, item)
        for session, values in zip(sessions, assignment_sets, strict=True)
        for item in values
    }
    session, assignment = assignment_by_stage[0]
    archive = _secure_archive(tmp_path, stage_id=0)
    with pytest.raises(ValueError, match="signature_invalid"):
        runtime.submit_checkpoint(
            session_id=session["session_id"],
            session_token=session["session_token"],
            epoch_id=assignment["epoch_id"],
            stage_id=0,
            assignment_token=assignment["assignment_token"],
            archive=archive,
            checkpoint_signature="bad",
        )
    bad = _secure_archive(tmp_path, stage_id=0, non_finite=True)
    signature = sign_checkpoint_submission(
        session_token=session["session_token"],
        run_id="secure",
        session_id=session["session_id"],
        epoch_id=assignment["epoch_id"],
        stage_id=0,
        assignment_token=assignment["assignment_token"],
        archive_hash=_sha(bad),
    )
    with pytest.raises(ValueError, match="tensor_non_finite"):
        runtime.submit_checkpoint(
            session_id=session["session_id"],
            session_token=session["session_token"],
            epoch_id=assignment["epoch_id"],
            stage_id=0,
            assignment_token=assignment["assignment_token"],
            archive=bad,
            checkpoint_signature=signature,
        )
    public = runtime.public_status()
    rejected = next(
        item for item in public["miners"] if item["miner_session_hash"] == _sha(session["session_id"].encode())
    )
    assert rejected["rejected_submission_count"] == 2
    assert rejected["quarantined"] is True
    assert public["runtime_state"] == "paused_waiting_for_miners"
    assert any(event["operation"] == "checkpoint_submission_rejected" for event in public["events"])


class _MissingObject(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class _Body:
    def __init__(self, value: bytes) -> None:
        self.value = value

    def read(self) -> bytes:
        return self.value


class _FakeS3:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}

    def head_object(self, *, Bucket, Key):
        try:
            value, metadata = self.values[(Bucket, Key)]
        except KeyError as exc:
            raise _MissingObject() from exc
        return {"ContentLength": len(value), "Metadata": metadata}

    def put_object(self, *, Bucket, Key, Body, ContentType, Metadata):
        self.values[(Bucket, Key)] = (bytes(Body), dict(Metadata))

    def get_object(self, *, Bucket, Key):
        try:
            value, _metadata = self.values[(Bucket, Key)]
        except KeyError as exc:
            raise _MissingObject() from exc
        return {"Body": _Body(value)}

    def delete_object(self, *, Bucket, Key):
        self.values.pop((Bucket, Key), None)

    def list_objects_v2(self, *, Bucket, Prefix, **_kwargs):
        return {
            "Contents": [
                {"Key": key}
                for bucket, key in self.values
                if bucket == Bucket and key.startswith(Prefix)
            ],
            "IsTruncated": False,
        }


def test_s3_minio_store_is_content_addressed_and_public_safe() -> None:
    client = _FakeS3()
    store = S3CheckpointBlobStore(bucket="private", prefix="jobs/one", client=client)
    value = b"checkpoint"
    archive_hash = _sha(value)
    assert store.put(archive_hash, value)["created"] is True
    assert store.put(archive_hash, value)["created"] is False
    assert store.get(archive_hash) == value
    assert list(store.list_hashes()) == [archive_hash]
    public = store.public_report()
    assert public["s3_compatible"] is True
    assert public["minio_compatible"] is True
    assert public["credential_values_public"] is False
    assert store.delete(archive_hash) is True


def test_product_controller_and_authenticated_service_survive_restart(tmp_path) -> None:
    config, tokenized = _inputs(tmp_path)
    job = tmp_path / "job"
    created = ElasticTrainingBetaController.create(
        job,
        config_path=config,
        tokenized_payload_path=tokenized,
    )
    status = created.status()
    assert status["overall_state"] == "waiting_for_miners"
    assert status["committed_step"] == 0
    assert status["online_miner_count"] == 0
    assert status["missing_stage_ids"] == [0, 1, 2, 3]
    assert status["pause_reason"] == "incomplete_stage_coverage"
    assert status["ordinary_user_create_status_cancel_export_ready"] is True
    credentials = created.credentials()
    app = create_elastic_training_beta_app(
        created,
        owner_token=credentials["owner_token"],
        miner_token=credentials["miner_token"],
    )
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/elastic-training/bootstrap").status_code == 401
    bootstrap = client.get(
        "/elastic-training/bootstrap",
        headers={"x-crowdtensor-miner-token": credentials["miner_token"]},
    )
    assert bootstrap.status_code == 200
    assert bootstrap.json()["token_ids_public"] is False
    assert client.get(
        f"/v1/training/jobs/{created.job_id}",
        headers={"x-crowdtensor-training-token": credentials["owner_token"]},
    ).status_code == 200

    restarted = ElasticTrainingBetaController(job)
    assert restarted.job_id == created.job_id
    assert restarted.status()["runtime"]["committed_steps"] == []
    cancelled = restarted.cancel()
    assert cancelled["overall_state"] == "cancelled"
    assert restarted.cancel()["runtime"]["runtime_state"] == "cancelled"
    restarted_client = TestClient(
        create_elastic_training_beta_app(
            restarted,
            owner_token=credentials["owner_token"],
            miner_token=credentials["miner_token"],
        )
    )
    cleanup_response = restarted_client.post(
        f"/v1/training/jobs/{created.job_id}/cleanup",
        headers={"x-crowdtensor-training-token": credentials["owner_token"]},
    )
    assert cleanup_response.status_code == 200
    cleanup = cleanup_response.json()
    assert cleanup["ok"] is True
    assert cleanup["overall_state"] == "cleaned"
    assert cleanup["active_miner_leases_revoked"] is True
    assert restarted.cleanup() == cleanup
    public_text = (job / "elastic_training_status.json").read_text(encoding="utf-8")
    assert credentials["owner_token"] not in public_text
    assert credentials["miner_token"] not in public_text


def test_public_http_old_miners_leave_service_restarts_and_new_miners_resume(tmp_path) -> None:
    config, tokenized = _inputs(tmp_path)
    job = tmp_path / "restart-job"
    controller = ElasticTrainingBetaController.create(
        job,
        config_path=config,
        tokenized_payload_path=tokenized,
    )
    credentials = controller.credentials()

    def app_client(current: ElasticTrainingBetaController) -> TestClient:
        return TestClient(
            create_elastic_training_beta_app(
                current,
                owner_token=credentials["owner_token"],
                miner_token=credentials["miner_token"],
            )
        )

    miner_headers = {"x-crowdtensor-miner-token": credentials["miner_token"]}

    def register_pair(client: TestClient, generation: str):
        sessions = []
        for index in range(2):
            response = client.post(
                "/elastic-training/miners/register",
                headers=miner_headers,
                json={
                    "run_id": controller.run_id,
                    "miner_id_hash": _sha(f"{generation}-{index}".encode()),
                    "registration_nonce": f"{generation}-nonce-{index}",
                    "supported_stage_ids": [0, 1, 2, 3],
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
                    **miner_headers,
                    "x-crowdtensor-elastic-session-token": session["session_token"],
                },
            )
            assert response.status_code == 200
            assignments.extend((session, item) for item in response.json()["assignments"])
        assert {tuple(sorted(item["stage_id"] for owner, item in assignments if owner is session)) for session in sessions} == {(0, 1), (2, 3)}
        return sessions, sorted(assignments, key=lambda value: value[1]["stage_id"])

    def submit_step(client: TestClient, assignments, step: int) -> None:
        responses = []
        for session, assignment in assignments:
            archive = _secure_archive(tmp_path, stage_id=assignment["stage_id"], step=step)
            signature = sign_checkpoint_submission(
                session_token=session["session_token"],
                run_id=controller.run_id,
                session_id=session["session_id"],
                epoch_id=assignment["epoch_id"],
                stage_id=assignment["stage_id"],
                assignment_token=assignment["assignment_token"],
                archive_hash=_sha(archive),
            )
            response = client.post(
                f"/elastic-training/checkpoints/{assignment['epoch_id']}/{assignment['stage_id']}",
                headers={
                    **miner_headers,
                    "content-type": "application/vnd.crowdtensor.stage-checkpoint+zip",
                    "x-crowdtensor-elastic-session-id": session["session_id"],
                    "x-crowdtensor-elastic-session-token": session["session_token"],
                    "x-crowdtensor-elastic-assignment-token": assignment["assignment_token"],
                    "x-crowdtensor-checkpoint-signature": signature,
                },
                content=archive,
            )
            assert response.status_code == 200, response.text
            responses.append(response.json())
        assert responses[-1]["global_commit_created"] is True

    first_client = app_client(controller)
    old_sessions, old_assignments = register_pair(first_client, "old")
    submit_step(first_client, old_assignments, 1)
    for session in old_sessions:
        response = first_client.post(
            f"/elastic-training/miners/{session['session_id']}/offline",
            headers={
                **miner_headers,
                "x-crowdtensor-elastic-session-token": session["session_token"],
            },
        )
        assert response.status_code == 200
    assert controller.status()["overall_state"] == "waiting_for_miners"

    restarted = ElasticTrainingBetaController(job)
    controller = restarted
    second_client = app_client(restarted)
    new_sessions, new_assignments = register_pair(second_client, "new")
    assert {item["base_step"] for _session, item in new_assignments} == {1}
    for session, assignment in new_assignments:
        restored = second_client.get(
            f"/elastic-training/checkpoints/{assignment['epoch_id']}/{assignment['stage_id']}",
            headers={
                **miner_headers,
                "x-crowdtensor-elastic-session-id": session["session_id"],
                "x-crowdtensor-elastic-session-token": session["session_token"],
                "x-crowdtensor-elastic-assignment-token": assignment["assignment_token"],
            },
        )
        assert restored.status_code == 200
        assert restored.headers["x-crowdtensor-global-step"] == "1"
    submit_step(second_client, new_assignments, 2)
    final = restarted.status()
    assert final["runtime"]["committed_steps"] == [1, 2]
    assert final["runtime"]["committed_steps_contiguous"] is True
    assert {item["miner_id_hash"] for item in final["runtime"]["miners"] if item["state"] == "online"} == {
        _sha(b"new-0"),
        _sha(b"new-1"),
    }


def test_public_cli_create_status_cancel_and_miner_join_parse(tmp_path, capsys) -> None:
    config, tokenized = _inputs(tmp_path)
    job = tmp_path / "cli-job"
    with pytest.raises(SystemExit) as created_exit:
        cli.main(
            [
                "train",
                "create",
                str(job),
                "--config",
                str(config),
                "--tokenized-payload",
                str(tokenized),
                "--json",
            ]
        )
    assert created_exit.value.code == 0
    created = json.loads(capsys.readouterr().out)
    assert created["overall_state"] == "waiting_for_miners"
    assert created["command_ok"] is True
    assert created["miner_join_training_ready"] is True

    with pytest.raises(SystemExit) as status_exit:
        cli.main(["train", "status", str(job), "--json"])
    assert status_exit.value.code == 0
    status = json.loads(capsys.readouterr().out)
    assert status["global_step"] == 0
    assert status["committed_step"] == 0
    assert status["online_miner_count"] == 0
    assert status["missing_stage_ids"] == [0, 1, 2, 3]
    assert status["pause_reason"] == "incomplete_stage_coverage"
    assert status["checkpoint_signature_verification_ready"] is True

    invite_path = tmp_path / "miner-invite.json"
    with pytest.raises(SystemExit) as invite_exit:
        cli.main(
            [
                "train",
                "invite",
                str(job),
                "--coordinator",
                "https://coordinator.invalid",
                "--output-file",
                str(invite_path),
                "--json",
            ]
        )
    assert invite_exit.value.code == 0
    invite_report = json.loads(capsys.readouterr().out)
    private_invite = json.loads(invite_path.read_text(encoding="utf-8"))
    assert invite_report["invite_file_written"] is True
    assert private_invite["miner_token"] not in json.dumps(invite_report)

    join = parse_training_join_args(
        [
            "--training",
            "--invite",
            str(invite_path),
            "--role",
            "auto",
            "--max-steps",
            "1",
        ]
    )
    assert join.training is True
    assert join.role == "auto"
    assert join.max_steps == 1

    heterogeneous_join = parse_training_join_args(
        [
            "--training",
            "--coordinator",
            "https://coordinator.invalid",
            "--device-policy",
            "cuda",
            "--cuda-device",
            "1",
            "--max-stages",
            "1",
            "--hf-token-env",
            "PRIVATE_HF_TOKEN",
        ]
    )
    assert heterogeneous_join.device_policy == "cuda"
    assert heterogeneous_join.cuda_device == [1]
    assert heterogeneous_join.max_stages == 1
    assert heterogeneous_join.hf_token_env == "PRIVATE_HF_TOKEN"

    with pytest.raises(SystemExit) as cancel_exit:
        cli.main(["train", "cancel", str(job), "--json"])
    assert cancel_exit.value.code == 0
    cancelled = json.loads(capsys.readouterr().out)
    assert cancelled["overall_state"] == "cancelled"

    exported = job / "exported_adapter"
    exported.mkdir()
    (exported / "adapter_model.safetensors").write_bytes(b"retained-adapter")
    (exported / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as cleanup_exit:
        cli.main(["train", "cleanup", str(job), "--json"])
    assert cleanup_exit.value.code == 0
    cleanup = json.loads(capsys.readouterr().out)
    assert cleanup["schema"] == "crowdtensor_elastic_training_beta_cleanup_v1"
    assert cleanup["overall_state"] == "cleaned"
    assert cleanup["exported_adapter_preserved"] is True
    assert (exported / "adapter_model.safetensors").read_bytes() == b"retained-adapter"
    with pytest.raises(SystemExit) as repeated_cleanup_exit:
        cli.main(["train", "cleanup", str(job), "--json"])
    assert repeated_cleanup_exit.value.code == 0
    assert json.loads(capsys.readouterr().out) == cleanup
