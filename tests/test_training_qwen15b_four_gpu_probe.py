from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

import scripts.training_qwen15b_four_gpu_probe as probe
from crowdtensor.qwen15b_training import (
    MODEL_ID,
    MODEL_REVISION,
    sha256_bytes,
    sha256_file,
    stable_hash,
)
from scripts.training_qwen15b_four_gpu_probe import (
    ALLOCATION_AMENDMENT_SCHEMA,
    BETA_ALLOCATION_AUTHORIZATION_SCHEMA,
    RestartableQwenCoordinator,
    UNBOUNDED_ALLOCATION_AMENDMENT_SCHEMA,
    allocation_budget_summary,
    build_beta_benchmark,
    evaluate_live_evidence,
    finish_attempt,
    inspect_adapter_bundle,
    inspect_checkpoint_bundle,
    preflight_accounts,
    reserve_attempt,
)


def test_attempt_ledger_hard_caps_dual_kernel_allocations_at_two(tmp_path) -> None:
    ledger = tmp_path / "attempts.json"
    assert reserve_attempt(ledger, limit=2) == 1
    finish_attempt(ledger, attempt=1, outcome="blocked")
    assert reserve_attempt(ledger, limit=2) == 2
    with pytest.raises(RuntimeError, match="attempt_limit_reached"):
        reserve_attempt(ledger, limit=2)
    with pytest.raises(RuntimeError, match="not_authorized"):
        reserve_attempt(ledger, limit=3)
    with pytest.raises(ValueError, match="positive"):
        reserve_attempt(tmp_path / "other.json", limit=0)


def test_explicit_amendment_preserves_two_attempts_and_allows_exactly_one_more(
    tmp_path,
) -> None:
    ledger = tmp_path / "attempts.json"
    assert reserve_attempt(ledger, limit=2) == 1
    finish_attempt(ledger, attempt=1, outcome="blocked")
    assert reserve_attempt(ledger, limit=2) == 2
    finish_attempt(ledger, attempt=2, outcome="blocked")
    value = json.loads(ledger.read_text(encoding="utf-8"))
    prior = copy.deepcopy(value["qwen15b_four_gpu_attempts"])
    value["allocation_budget_amendment"] = {
        "schema": ALLOCATION_AMENDMENT_SCHEMA,
        "authorized": True,
        "authorized_at": "2026-07-12T06:00:00Z",
        "authorization_hash": "sha256:" + "a" * 64,
        "authorization_text_public": False,
        "same_authorized_account_only": True,
        "topology": "kaggle-2x-t4x2",
        "original_attempt_limit": 2,
        "additional_attempts": 1,
        "revised_attempt_limit": 3,
        "allocation_timeout_seconds": 1800,
        "prior_attempt_count": 2,
        "prior_attempts_hash": probe.stable_hash(prior),
    }
    ledger.write_text(json.dumps(value), encoding="utf-8")
    summary = allocation_budget_summary(value)
    assert summary["amendment_valid"] is True
    assert summary["prior_attempts_preserved"] is True
    assert reserve_attempt(ledger, limit=3) == 3
    after = json.loads(ledger.read_text(encoding="utf-8"))
    assert after["qwen15b_four_gpu_attempts"][:2] == prior
    with pytest.raises(RuntimeError, match="attempt_limit_reached"):
        reserve_attempt(ledger, limit=3)


def test_unbounded_amendment_allows_one_audited_attempt_per_invocation(tmp_path) -> None:
    ledger = tmp_path / "attempts.json"
    for attempt in range(1, 4):
        limit = 2 if attempt <= 2 else 3
        if attempt == 3:
            value = json.loads(ledger.read_text(encoding="utf-8"))
            prior = copy.deepcopy(value["qwen15b_four_gpu_attempts"])
            value["allocation_budget_amendment"] = {
                "schema": ALLOCATION_AMENDMENT_SCHEMA,
                "authorized": True,
                "authorized_at": "2026-07-12T06:00:00Z",
                "authorization_hash": "sha256:" + "a" * 64,
                "authorization_text_public": False,
                "same_authorized_account_only": True,
                "topology": "kaggle-2x-t4x2",
                "original_attempt_limit": 2,
                "additional_attempts": 1,
                "revised_attempt_limit": 3,
                "allocation_timeout_seconds": 1800,
                "prior_attempt_count": 2,
                "prior_attempts_hash": probe.stable_hash(prior),
            }
            ledger.write_text(json.dumps(value), encoding="utf-8")
        assert reserve_attempt(ledger, limit=limit) == attempt
        finish_attempt(ledger, attempt=attempt, outcome="blocked")
    value = json.loads(ledger.read_text(encoding="utf-8"))
    prior = copy.deepcopy(value["qwen15b_four_gpu_attempts"])
    value["allocation_budget_amendment"] = {
        "schema": UNBOUNDED_ALLOCATION_AMENDMENT_SCHEMA,
        "authorized": True,
        "authorized_at": "2026-07-12T09:00:00Z",
        "authorization_hash": "sha256:" + "b" * 64,
        "authorization_text_public": False,
        "same_authorized_account_only": True,
        "topology": "kaggle-2x-t4x2",
        "total_attempt_limit_unbounded": True,
        "one_attempt_per_probe_invocation": True,
        "automatic_retry_loop": False,
        "allocation_timeout_seconds": 1800,
        "prior_attempt_count": 3,
        "prior_attempts_hash": probe.stable_hash(prior),
    }
    ledger.write_text(json.dumps(value), encoding="utf-8")
    summary = allocation_budget_summary(value)
    assert summary["amendment_valid"] is True
    assert summary["total_attempt_limit_unbounded"] is True
    assert summary["effective_attempt_limit"] is None
    assert reserve_attempt(ledger, limit=4) == 4
    finish_attempt(ledger, attempt=4, outcome="blocked")
    assert reserve_attempt(ledger, limit=5) == 5


def test_beta_goal_authorization_allows_exactly_three_attempts(tmp_path) -> None:
    ledger = tmp_path / "beta-attempts.json"
    ledger.write_text(
        json.dumps(
            {
                "beta_goal_allocation_authorization": {
                    "schema": BETA_ALLOCATION_AUTHORIZATION_SCHEMA,
                    "authorized": True,
                    "authorized_at": "2026-07-12T12:00:00Z",
                    "authorization_hash": "sha256:" + "c" * 64,
                    "authorization_text_public": False,
                    "same_authorized_account_only": True,
                    "topology": "kaggle-2x-t4x2",
                    "goal_attempt_limit": 3,
                    "one_attempt_per_probe_invocation": True,
                    "automatic_retry_loop": False,
                    "allocation_timeout_seconds": 1800,
                }
            }
        ),
        encoding="utf-8",
    )
    summary = allocation_budget_summary(json.loads(ledger.read_text(encoding="utf-8")))
    assert summary["beta_goal_authorization"] is True
    assert summary["effective_attempt_limit"] == 3
    for attempt in range(1, 4):
        assert reserve_attempt(ledger, limit=3) == attempt
        finish_attempt(ledger, attempt=attempt, outcome="blocked")
    with pytest.raises(RuntimeError, match="attempt_limit_reached"):
        reserve_attempt(ledger, limit=3)


def test_restartable_coordinator_recovers_private_rendezvous_state(tmp_path) -> None:
    import base64
    import socket

    from crowdtensor.qwen15b_four_gpu_runtime import QwenHTTPTransport, sha256_bytes

    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        port = int(handle.getsockname()[1])
    runtime = RestartableQwenCoordinator(
        private_root=tmp_path / "private",
        port=port,
        run_id="beta-restart-run",
        token="private-token",
    )
    runtime.start()
    try:
        transport = QwenHTTPTransport(
            coordinator_url=f"http://127.0.0.1:{port}",
            token="private-token",
            run_id="beta-restart-run",
            retry_attempts=3,
            retry_base_seconds=0.01,
        )
        for role, stage_ids, pids in (
            ("kernel_a", [0, 1], [10, 11]),
            ("kernel_b", [2, 3], [12, 13]),
        ):
            transport.register(
                role=role,
                ready=[
                    {
                        "stage_id": stage,
                        "pid": pid,
                        "device": f"cuda:{index}",
                        "cuda_live": True,
                        "cuda_device_name_hash": f"sha256:gpu-{stage}",
                    }
                    for index, (stage, pid) in enumerate(zip(stage_ids, pids))
                ],
            )
        raw = b"private-payload"
        assert runtime.rendezvous is not None
        runtime.rendezvous.put_payload(
            {
                "run_id": "beta-restart-run",
                "run_kind": "resumed",
                "kind": "activation",
                "step": 3,
                "microbatch": 0,
                "producer_role": "kernel_a",
                "payload_b64": base64.b64encode(raw).decode("ascii"),
                "payload_hash": sha256_bytes(raw),
                "tensor_count": 1,
            }
        )
        restart = runtime.restart(after_step=4, downtime_seconds=0.5)
        assert restart["verified"] is True
        for role, stage_ids, pids in (
            ("kernel_a", [0, 1], [20, 21]),
            ("kernel_b", [2, 3], [22, 23]),
        ):
            transport.register(
                role=role,
                ready=[
                    {
                        "stage_id": stage,
                        "pid": pid,
                        "device": f"cuda:{index}",
                        "cuda_live": True,
                        "cuda_device_name_hash": f"sha256:gpu-{stage}",
                    }
                    for index, (stage, pid) in enumerate(zip(stage_ids, pids))
                ],
            )
        status = transport.status()
        assert status["coordinator_restart_verified"] is True
        assert status["post_restart_registered_roles"] == ["kernel_a", "kernel_b"]
        assert status["payloads"][0]["payload_hash"] == sha256_bytes(raw)
    finally:
        assert runtime.stop() is True


def test_beta_benchmark_records_latency_network_memory_and_recovery() -> None:
    path = Path(
        "dist/training-qwen15b-four-gpu-live-20260712-r5-attempt5-fp32-stable-fp16-boundary/"
        "training_qwen15b_four_gpu_live_probe.json"
    )
    live = json.loads(path.read_text(encoding="utf-8"))
    rendezvous = copy.deepcopy(live["rendezvous"])
    rendezvous["coordinator_restarts"] = [{"duration_seconds": 1.25}]
    optimizer_at = min(
        float(item["at"])
        for item in rendezvous["events"]
        if item.get("operation") == "optimizer_step"
    )
    benchmark = build_beta_benchmark(
        workers=live["worker_reports"],
        rendezvous=rendezvous,
        attempt_started_epoch=optimizer_at - 10,
        attempt_elapsed_seconds=600,
        evidence=live["evidence"],
    )
    assert benchmark["benchmark_complete"] is True
    assert benchmark["step_latency_count"] == 16
    assert benchmark["private_network_payload_count"] == 129
    assert benchmark["private_network_bytes"] > 0
    assert benchmark["peak_gpu_allocated_bytes"] > 0
    assert benchmark["coordinator_recovery_seconds"] == 1.25


def test_account_preflight_is_read_only_and_requires_two_free_slots(monkeypatch) -> None:
    commands = []

    monkeypatch.setattr(
        probe,
        "fetch_accelerator_quota",
        lambda _env: {
            "ok": True,
            "quota_refresh_time": "2026-07-18T00:00:00Z",
            "gpu_quota": {
                "effective_remaining_after_reserved_seconds": 7200,
                "quota_exhausted_by_used": False,
                "reserved_exceeds_remaining": False,
            },
        },
    )
    monkeypatch.setattr(probe, "authenticated_owner", lambda _env: "same-owner")

    def run(command, **_kwargs):
        commands.append(command)
        if command[1:3] == ["kernels", "list"]:
            return {"ok": True, "output_tail": "ref title\nsame-owner/old old\n"}
        return {"ok": True, "output_tail": 'Kernel same-owner/old has status "COMPLETE"'}

    monkeypatch.setattr(probe, "run_command", run)
    public, candidates = preflight_accounts(
        [{"label": "private", "env": {"KAGGLE_USERNAME": "same-owner", "KAGGLE_KEY": "secret"}}]
    )
    assert public[0]["two_gpu_slots_read_only_preflight"] is True
    assert len(candidates) == 1
    assert all(command[1:3] in (["kernels", "list"], ["kernels", "status"]) for command in commands)
    assert not any("push" in command or "delete" in command for command in commands)
    assert "secret" not in json.dumps(public)


def _checkpoint_archive(path: Path) -> dict:
    with zipfile.ZipFile(path, "w") as archive:
        for run_kind in ("baseline", "resumed"):
            for stage in (0, 1):
                root = f"{run_kind}/checkpoints"
                files = {}
                for label in ("adapter", "optimizer", "grad_scaler", "rng"):
                    filename = f"stage{stage}_{label}.bin"
                    payload = f"{run_kind}-{stage}-{label}".encode()
                    archive.writestr(f"{root}/{filename}", payload)
                    files[label] = (filename, sha256_bytes(payload))
                manifest = {
                    "schema": "crowdtensor_qwen15b_stage_checkpoint_v1",
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "stage_id": stage,
                    "layer_start": stage,
                    "layer_end": stage + 1,
                    "global_step": 8,
                    "dataset_cursor": 32,
                    "adapter_file": files["adapter"][0],
                    "adapter_file_hash": files["adapter"][1],
                    "optimizer_file": files["optimizer"][0],
                    "optimizer_file_hash": files["optimizer"][1],
                    "grad_scaler_file": files["grad_scaler"][0],
                    "grad_scaler_file_hash": files["grad_scaler"][1],
                    "rng_file": files["rng"][0],
                    "rng_file_hash": files["rng"][1],
                }
                manifest["content_hash"] = stable_hash(manifest)
                archive.writestr(
                    f"{root}/stage{stage}_checkpoint.json",
                    json.dumps(manifest),
                )
    return {"file_hash": sha256_file(path)}


def test_checkpoint_archive_verifies_every_private_file_before_cleanup(tmp_path) -> None:
    archive = tmp_path / "checkpoints.zip"
    worker = _checkpoint_archive(archive)
    result = inspect_checkpoint_bundle(archive, worker)
    assert result["verified"] is True
    assert result["checkpoint_manifest_count"] == 4
    assert result["all_checkpoint_files_hash_verified"] is True
    with zipfile.ZipFile(archive, "a") as value:
        value.writestr("baseline/checkpoints/stage0_adapter.bin", b"tampered")
    assert inspect_checkpoint_bundle(archive, {"file_hash": sha256_file(archive)})["verified"] is False


def test_standard_peft_archive_requires_pinned_model_and_revision(tmp_path) -> None:
    import torch
    from safetensors.torch import save

    archive = tmp_path / "adapter.zip"
    adapter_bytes = save(
        {
            f"base_model.model.model.layers.{index}.self_attn.q_proj.lora_A.weight": (
                torch.ones(1, 1) * (index + 1)
            )
            for index in range(28)
        }
    )
    with zipfile.ZipFile(archive, "w") as value:
        value.writestr("adapter_model.safetensors", adapter_bytes)
        value.writestr(
            "adapter_config.json",
            json.dumps({"base_model_name_or_path": MODEL_ID, "revision": MODEL_REVISION}),
        )
    result = inspect_adapter_bundle(archive, {"file_hash": sha256_file(archive)})
    assert result["verified"] is True
    assert result["safetensors_header_verified"] is True
    assert result["layer_indexes"] == list(range(28))
    with zipfile.ZipFile(archive, "w") as value:
        value.writestr("adapter_model.safetensors", adapter_bytes)
        value.writestr(
            "adapter_config.json",
            json.dumps({"base_model_name_or_path": "toy", "revision": MODEL_REVISION}),
        )
    assert inspect_adapter_bundle(archive, {"file_hash": sha256_file(archive)})["verified"] is False


def _worker(role: str) -> dict:
    stages = [0, 1] if role == "kernel_a" else [2, 3]
    ready = [
        {
            "stage_id": stage,
            "device": f"cuda:{stage % 2}",
            "pid": 100 + stage,
        }
        for stage in stages
    ]
    events = [
        {
            "run_kind": "baseline",
            "step": 0,
            "stage_id": stage,
            "started_ns": 100 + stage,
            "ended_ns": 200 - stage,
        }
        for stage in stages
    ]
    runs = {
        run_kind: {
            "steps_completed": 8,
            "events": events if run_kind == "baseline" else [],
            "loss_reduced": True if role == "kernel_b" else None,
        }
        for run_kind in ("baseline", "resumed")
    }
    return {
        "ok": True,
        "role": role,
        "worker": {
            "stage_ready": {"baseline": ready, "resumed": ready},
            "runs": runs,
            "resume_adapter_equivalence": {"verified": True},
            "resume_loss_equivalence": {"verified": role == "kernel_b"},
            "controlled_restart_verified": role == "kernel_b",
            "evaluation": {"evaluation_verified": role == "kernel_b"},
            "export": {"standard_peft_format": role == "kernel_b"},
        },
    }


def test_live_acceptance_rejects_any_missing_four_gpu_training_proof() -> None:
    workers = [_worker("kernel_a"), _worker("kernel_b")]
    rendezvous = {
        "payloads": [
            *[
                {"kind": "activation", "payload_hash": f"sha256:a{i}"}
                for i in range(64)
            ],
            *[
                {"kind": "gradient", "payload_hash": f"sha256:g{i}"}
                for i in range(64)
            ],
            {"kind": "stage_adapter", "payload_hash": "sha256:s"},
        ]
    }
    checkpoints = [{"verified": True}, {"verified": True}]
    adapter = {"verified": True}
    result = evaluate_live_evidence(
        workers=workers,
        rendezvous=rendezvous,
        checkpoint_bundles=checkpoints,
        adapter_bundle=adapter,
        max_running=2,
    )
    assert result["verified"] is True
    workers[1]["worker"]["controlled_restart_verified"] = False
    assert evaluate_live_evidence(
        workers=workers,
        rendezvous=rendezvous,
        checkpoint_bundles=checkpoints,
        adapter_bundle=adapter,
        max_running=2,
    )["verified"] is False
    workers[1]["worker"]["controlled_restart_verified"] = True
    workers[1]["worker"]["stage_ready"]["baseline"][0]["device"] = "cpu"
    assert evaluate_live_evidence(
        workers=workers,
        rendezvous=rendezvous,
        checkpoint_bundles=checkpoints,
        adapter_bundle=adapter,
        max_running=2,
    )["verified"] is False
