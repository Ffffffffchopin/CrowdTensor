import base64
import json
import py_compile
import time

import torch

from crowdtensor.elastic_training_client import ElasticTrainingHTTPClient
from crowdtensor.heterogeneous_tensor_transport import (
    ChunkedTensorStore,
    encode_tensor_message,
)
from crowdtensor.heterogeneous_training_miner import _recoverable_epoch_error
from scripts import training_heterogeneous_beta_worker_entry as worker_entry
from scripts.training_heterogeneous_beta_kaggle_package import build_package
from scripts.training_heterogeneous_beta_live_probe import (
    _committed_assignments,
    collect_public_tensor_metadata,
    transport_contract_probe,
)
from tests.test_heterogeneous_qwen_training import tiny_source


def test_private_kaggle_packages_render_single_gpu_and_cpu_roles(tmp_path) -> None:
    reports = []
    for role in ("gpu_a", "gpu_b", "cpu"):
        report = build_package(
            tmp_path / role,
            owner="fixture-owner",
            slug=f"fixture-{role}",
            role=role,
            coordinator_url="https://private.invalid",
            coordinator_token="private-fixture-token",
            wait_timeout_seconds=600.0,
        )
        reports.append(report)
        package = report["package_dir"]
        metadata = json.loads(
            (tmp_path / role / "private-kernel" / "kernel-metadata.json").read_text(
                encoding="utf-8"
            )
        )
        py_compile.compile(
            str(tmp_path / role / "private-kernel" / "kernel.py"), doraise=True
        )
        assert metadata["is_private"] == "true"
        assert metadata["enable_gpu"] == (
            "true" if role.startswith("gpu") else "false"
        )
        assert "private-fixture-token" not in json.dumps(
            {key: value for key, value in report.items() if key != "package_dir"}
        )
        assert package.endswith("private-kernel")
        assert report["operation_timeout_seconds"] == 600.0

    assert reports[0]["replacement_process_included"] is True
    assert reports[0]["single_gpu_process_count"] == 2
    assert reports[1]["single_gpu_process_count"] == 2
    assert reports[2]["pure_cpu_process_count"] == 1


def test_recovery_package_preserves_identity_nonces_without_public_leak(
    tmp_path,
) -> None:
    identity_nonces = {
        "gpu_stable_b0": "fixture-stable-b0",
        "gpu_stable_b1": "fixture-stable-b1",
    }
    report = build_package(
        tmp_path / "gpu-recovery",
        owner="fixture-owner",
        slug="fixture-gpu-recovery",
        role="gpu_b",
        coordinator_url="https://private.invalid",
        coordinator_token="private-fixture-token",
        wait_timeout_seconds=3600.0,
        operation_timeout_seconds=300.0,
        recovery_mode=True,
        identity_nonces=identity_nonces,
    )
    rendered = (
        tmp_path / "gpu-recovery" / "private-kernel" / "kernel.py"
    ).read_text(encoding="utf-8")
    public = {key: value for key, value in report.items() if key != "package_dir"}

    assert report["recovery_mode"] is True
    assert report["operation_timeout_seconds"] == 300.0
    assert "RECOVERY_MODE = True" in rendered
    assert all(value in rendered for value in identity_nonces.values())
    assert all(value not in json.dumps(public) for value in identity_nonces.values())
    assert "private-fixture-token" not in json.dumps(public)


def test_live_transport_contract_exercises_chunk_retry_dedup_and_fencing(
    tmp_path,
) -> None:
    report = transport_contract_probe(tmp_path / "transport")

    assert report["ok"] is True
    assert report["format"] == "safetensors"
    assert report["pickle_deserialization_allowed"] is False
    assert report["chunking_verified"] is True
    assert report["finite_retry_verified"] is True
    assert report["idempotent_delivery_verified"] is True
    assert report["stale_generation_rejected"] is True


def test_remote_worker_entry_preserves_full_public_training_evidence(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest, config, _model = tiny_source(source)
    tokenized = {
        "schema": "crowdtensor_heterogeneous_tokenized_private_v1",
        "training_manifest_hash": manifest["content_hash"],
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "sequence_length": manifest["training"]["sequence_length"],
        "train": [[1, 2, 3, 4, 5, 6]],
        "validation": [[1, 2, 3, 4, 5, 6]],
    }
    private = tmp_path / "private.json"
    private.write_text(
        json.dumps(
            {
                "coordinator_url": "https://private.invalid",
                "coordinator_token": "private-token",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        worker_entry,
        "_request_json",
        lambda *_args, **_kwargs: {
            "schema": worker_entry.BOOTSTRAP_SCHEMA,
            "run_id": "fixture-run",
            "training_manifest": manifest,
            "config": config,
            "tokenized_payload": tokenized,
            "config_hash": worker_entry.stable_hash(config),
            "tokenized_payload_hash": worker_entry.stable_hash(tokenized),
        },
    )
    worker_arguments = {}

    def fake_miner(**kwargs):
        worker_arguments.update(kwargs)
        return {
            "schema": "crowdtensor_heterogeneous_training_miner_v1",
            "ok": True,
            "miner_id_hash": kwargs["miner_id_hash"],
            "device_policy": kwargs["device_policy"],
            "steps": [{"target_step": 1, "stages": []}],
            "steps_completed": 1,
            "stage_process_ready": [{"resumed": False}],
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }

    monkeypatch.setattr(worker_entry, "run_heterogeneous_miner", fake_miner)

    report = worker_entry.run_worker(
        private_configuration_path=private,
        output_path=tmp_path / "worker.json",
        private_root=tmp_path / "worker-private",
        deployment_role="gpu_old",
        identity_nonce="fixture-nonce",
        device_policy="cuda",
        max_steps=3,
        wait_timeout=60.0,
        operation_timeout=45.0,
    )

    assert report["ok"] is True
    assert report["deployment_role"] == "gpu_old"
    assert report["steps"] == [{"target_step": 1, "stages": []}]
    assert report["single_stage_limit"] is True
    assert report["operation_timeout_seconds"] == 45.0
    assert worker_arguments["wait_timeout"] == 60.0
    assert worker_arguments["operation_timeout"] == 45.0
    assert report["visible_cuda_device_count_expected"] == 1
    assert "private-token" not in json.dumps(report)


def test_heartbeat_loop_recovers_after_transient_transport_failures(monkeypatch) -> None:
    client = ElasticTrainingHTTPClient(
        coordinator_url="https://private.invalid",
        coordinator_token="private-token",
        run_id="fixture-run",
        miner_id_hash="sha256:" + "1" * 64,
        registration_nonce="fixture-nonce",
        supported_stage_ids=[0],
        slot_count=1,
        retry_attempts=1,
        heartbeat_interval_seconds=0.01,
    )
    client._session_id = "fixture-session"
    client._session_token = "fixture-session-token"
    calls = 0

    def transient(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise RuntimeError("transient")
        return {"ok": True}

    monkeypatch.setattr(client, "_json_request", transient)
    client.start_heartbeat()
    deadline = time.monotonic() + 2.0
    while calls < 5 and time.monotonic() < deadline:
        time.sleep(0.01)
    client.stop_heartbeat()

    report = client.public_report()
    assert calls >= 5
    assert report["heartbeat_failed"] is False
    assert report["heartbeat_failure_count"] == 0
    assert report["heartbeat_recovery_count"] >= 1


def test_client_switches_to_persistent_http_only_after_baseline_step(
    monkeypatch,
) -> None:
    client = ElasticTrainingHTTPClient(
        coordinator_url="https://private.invalid",
        coordinator_token="private-token",
        run_id="fixture-run",
        miner_id_hash="sha256:" + "1" * 64,
        registration_nonce="fixture-nonce",
        supported_stage_ids=[0],
        slot_count=1,
        retry_attempts=1,
        persistent_http_after_step=2,
    )

    class LegacyResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"legacy"

    monkeypatch.setattr(
        "crowdtensor.elastic_training_client.urllib.request.urlopen",
        lambda *_args, **_kwargs: LegacyResponse(),
    )
    persistent_calls = []

    def persistent(**kwargs):
        persistent_calls.append(kwargs["path"])
        return b"persistent", {}

    monkeypatch.setattr(client, "_persistent_request", persistent)
    client.set_current_step(2)
    legacy, _ = client._request("/baseline", retry_attempts=1)
    client.set_current_step(3)
    optimized, _ = client._request("/candidate", retry_attempts=1)
    report = client.public_report()

    assert legacy == b"legacy"
    assert optimized == b"persistent"
    assert persistent_calls == ["/candidate"]
    assert report["legacy_http_request_count"] == 1
    assert report["persistent_http_request_count"] == 1
    assert report["connection_pool_reuse_enabled"] is True


def test_client_isolates_large_payloads_from_persistent_connection_pool(
    monkeypatch,
) -> None:
    client = ElasticTrainingHTTPClient(
        coordinator_url="https://private.invalid",
        coordinator_token="private-token",
        run_id="fixture-run",
        miner_id_hash="sha256:" + "1" * 64,
        registration_nonce="fixture-nonce",
        supported_stage_ids=[0],
        slot_count=1,
        retry_attempts=1,
        persistent_http_after_step=2,
        persistent_http_max_body_bytes=4,
    )

    class LegacyResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"legacy"

    monkeypatch.setattr(
        "crowdtensor.elastic_training_client.urllib.request.urlopen",
        lambda *_args, **_kwargs: LegacyResponse(),
    )
    persistent_payloads = []

    def persistent(**kwargs):
        persistent_payloads.append(kwargs["payload"])
        return b"persistent", {}

    monkeypatch.setattr(client, "_persistent_request", persistent)
    client.set_current_step(3)
    small, _ = client._request("/small", body=b"1234", retry_attempts=1)
    large, _ = client._request("/checkpoint", body=b"12345", retry_attempts=1)
    report = client.public_report()

    assert small == b"persistent"
    assert large == b"legacy"
    assert persistent_payloads == [b"1234"]
    assert report["persistent_http_request_count"] == 1
    assert report["legacy_http_request_count"] == 1
    assert report["large_payload_connection_isolation_count"] == 1
    assert report["persistent_http_max_body_bytes"] == 4
    assert report["large_payload_connection_isolation_enabled"] is True


def test_client_uses_single_request_inline_tensor_transport_after_switch(
    monkeypatch,
) -> None:
    client = ElasticTrainingHTTPClient(
        coordinator_url="https://private.invalid",
        coordinator_token="private-token",
        run_id="fixture-run",
        miner_id_hash="sha256:" + "2" * 64,
        registration_nonce="fixture-nonce",
        supported_stage_ids=[0, 1],
        slot_count=2,
        retry_attempts=1,
        persistent_http_after_step=2,
    )
    assignment = {
        "target_step": 3,
        "stage_id": 0,
        "placement_generation": 1,
        "assignment_token_hash": "sha256:" + "3" * 64,
        "assignment_token": "private-assignment",
    }
    requests = []

    def request(path, *, assignment, method="GET", json_value=None, body=None):
        requests.append((path, method, json_value, body))
        return json.dumps({"complete": True}).encode(), {}

    monkeypatch.setattr(client, "_assignment_request", request)
    client.set_current_step(3)
    sent = client.send_tensors(
        assignment,
        {"hidden": torch.ones((1, 4), dtype=torch.float32)},
        target_stage_id=1,
        direction="forward_activation",
        microbatch_id=0,
        manifest_hash="sha256:" + "4" * 64,
    )

    assert [item[0] for item in requests] == [
        "/elastic-training/tensors/inline"
    ]
    assert base64.b64decode(requests[0][2]["chunk_b64"], validate=True)
    assert sent["inline_transport_used"] is True
    assert client.public_report()["inline_tensor_message_upload_count"] == 1

    receive_assignment = {
        **assignment,
        "stage_id": 1,
    }
    expected = {"hidden": torch.arange(4, dtype=torch.float32).reshape(1, 4)}
    envelope, chunks = encode_tensor_message(
        expected,
        job_id="fixture-run",
        manifest_hash="sha256:" + "4" * 64,
        global_step=3,
        microbatch_id=0,
        source_stage_id=0,
        target_stage_id=1,
        direction="forward_activation",
        placement_generation=1,
        assignment_token_hash=assignment["assignment_token_hash"],
    )

    def download(path, **_kwargs):
        assert path.startswith("/elastic-training/tensors/inline?")
        return (
            json.dumps(
                {
                    "found": True,
                    "status": {"complete": True},
                    "envelope": envelope,
                    "inline_payload": True,
                    "chunk_b64": base64.b64encode(chunks[0]).decode("ascii"),
                }
            ).encode(),
            {},
        )

    monkeypatch.setattr(client, "_assignment_request", download)
    received, received_report = client.receive_tensors(
        receive_assignment,
        source_stage_id=0,
        direction="forward_activation",
        microbatch_id=0,
        timeout=1.0,
        target_device="cpu",
    )
    assert torch.equal(received["hidden"], expected["hidden"])
    assert received_report["inline_transport_used"] is True
    assert client.public_report()["inline_tensor_message_download_count"] == 1


def test_stale_or_aborted_epoch_errors_are_recoverable() -> None:
    for code in (
        "elastic_barrier_epoch_aborted",
        "elastic_barrier_wait_timeout",
        "elastic_stage_assignment_stale",
        "elastic_stage_placement_generation_stale",
        "elastic_tensor_receive_timeout",
        "elastic_tensor_global_step_stale",
        "heterogeneous_tensor_stale_generation",
    ):
        assert _recoverable_epoch_error(RuntimeError(code)) == code
    assert _recoverable_epoch_error(RuntimeError("cuda_oom")) == ""


def test_public_tensor_metadata_survives_worker_replacement(tmp_path) -> None:
    root = tmp_path / "payloads"
    envelope, chunks = encode_tensor_message(
        {"activation": torch.arange(64, dtype=torch.float32).reshape(1, 4, 16)},
        job_id="fixture-job",
        manifest_hash="sha256:" + "1" * 64,
        global_step=4,
        microbatch_id=0,
        source_stage_id=1,
        target_stage_id=2,
        direction="forward_activation",
        placement_generation=5,
        assignment_token_hash="sha256:" + "2" * 64,
        chunk_bytes=128,
    )
    store = ChunkedTensorStore(root, max_chunk_bytes=128)
    store.begin(envelope, expected_generation=5)
    for index, chunk in enumerate(chunks):
        store.put_chunk(
            envelope["message_id"], index, chunk, expected_generation=5
        )

    retained = collect_public_tensor_metadata(root)
    message = retained["messages"][0]
    assert retained["message_count"] == 1
    assert message["global_step"] == 4
    assert message["placement_generation"] == 5
    assert message["chunk_hashes_verified"] is True
    assert message["payload_hash_verified"] is True
    assert message["tensor_values_public"] is False
    assert envelope["assignment_token_hash"] not in json.dumps(retained)


def test_committed_assignment_evidence_keeps_stage_zero_scores() -> None:
    miner = "sha256:" + "3" * 64
    status = {
        "epochs": [{"epoch_id": 7, "target_step": 4, "state": "committed"}],
        "assignments": [
            {
                "epoch_id": 7,
                "stage_id": 0,
                "miner_id_hash": miner,
                "miner_session_hash": miner,
                "device_id": "cuda:0",
                "device_type": "cuda",
                "placement_generation": 5,
            }
        ],
    }
    snapshots = [
        {
            "placement_generation": 5,
            "placement_plan": {
                "content_hash": "sha256:" + "4" * 64,
                "assignments": [
                    {
                        "stage_id": 0,
                        "miner_id_hash": miner,
                        "device_id": "cuda:0",
                        "device_type": "cuda",
                        "available_after_reserve_bytes": 16 * 1024**3,
                        "compute_latency_ms": 12.0,
                        "compute_latency_measured": True,
                        "incoming_transfer_latency_ms": 0.0,
                        "incremental_score": 12.0,
                        "selection_reason": "measured-profile",
                        "resource_estimate": {
                            "estimated_peak_bytes": 8 * 1024**3
                        },
                    }
                ],
            },
        }
    ]

    assignments, generation = _committed_assignments(
        status, target_step=4, snapshots=snapshots
    )

    assert generation == 5
    assert assignments[0]["stage_id"] == 0
    assert assignments[0]["resource_fit_verified"] is True
    assert assignments[0]["selection_reason"] == "measured-profile"
