from __future__ import annotations

import base64
import io
import json
import py_compile
import zipfile
from copy import deepcopy

from crowdtensor.heterogeneous_training_manifest import (
    qwen25_7b_lora_tpu_manifest,
    validate_training_manifest,
)
from scripts import training_heterogeneous_beta_worker_entry as worker_entry
from scripts.training_heterogeneous_beta_kaggle_package import _bundle_archive_b64
from scripts.training_heterogeneous_tpu_beta_kaggle_package import build_package
from scripts.training_heterogeneous_tpu_stage_diagnostic_package import (
    build_package as build_diagnostic_package,
)


def tiny_manifest() -> dict:
    manifest = deepcopy(qwen25_7b_lora_tpu_manifest())
    manifest.pop("content_hash")
    manifest["model"].update(
        num_hidden_layers=5,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        weight_bytes=10_000,
        parameter_count=5_000,
    )
    for stage_id, stage in enumerate(manifest["stages"]):
        stage.update(
            stage_id=stage_id,
            layer_start=stage_id,
            layer_end=stage_id + 1,
            layer_count=1,
            owns_embedding=stage_id == 0,
            owns_norm=stage_id == 4,
            owns_lm_head=stage_id == 4,
            estimated_parameter_count=1_000,
            estimated_weight_bytes=2_000,
            estimated_compute_units=1_000.0,
        )
        if stage_id == 2:
            stage["allowed_device_types"] = ["jax_tpu"]
            stage["preferred_device_type"] = "jax_tpu"
        elif stage_id == 4:
            stage["allowed_device_types"] = ["cpu"]
            stage["preferred_device_type"] = "cpu"
        else:
            stage["allowed_device_types"] = ["cpu", "cuda"]
            stage["preferred_device_type"] = "cuda"
    return validate_training_manifest(manifest)


def tiny_config() -> dict:
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 5,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "vocab_size": 32,
    }


def test_tpu_package_is_private_v5e8_and_contains_jax_training_runtime(tmp_path) -> None:
    report = build_package(
        tmp_path / "tpu",
        owner="fixture-owner",
        slug="fixture-training-tpu",
        coordinator_url="https://private.invalid",
        coordinator_token="private-fixture-token",
        hf_token="private-hf-token",
        wait_timeout_seconds=1200.0,
        operation_timeout_seconds=600.0,
        old_identity_nonce="private-old-nonce",
        replacement_identity_nonce="private-replacement-nonce",
    )
    package = tmp_path / "tpu" / "private-kernel"
    metadata = json.loads(
        (package / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    rendered = (package / "kernel.py").read_text(encoding="utf-8")
    py_compile.compile(str(package / "kernel.py"), doraise=True)
    public = {key: value for key, value in report.items() if key != "package_dir"}

    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "false"
    assert metadata["enable_tpu"] == "true"
    assert metadata["machine_shape"] == "tpuV5e8"
    assert report["expected_tpu_device_count"] == 8
    assert report["logical_restart_process_count"] == 2
    assert '"--device-policy", "jax_tpu"' in rendered
    assert "tpu_probe_process_released" in rendered
    assert 'sys.executable,\n            "-c"' in rendered
    assert "    import jax\n    devices =" not in rendered
    assert "REPLACEMENT_AFTER_STEPS = 3" in rendered
    assert 'launch("tpu_old", OLD_IDENTITY_NONCE, REPLACEMENT_AFTER_STEPS)' in rendered
    assert 'launch("tpu_replacement", REPLACEMENT_IDENTITY_NONCE, 0)' in rendered
    assert "private-fixture-token" not in json.dumps(public)
    assert "private-hf-token" not in json.dumps(public)
    assert "private-old-nonce" not in json.dumps(public)

    with zipfile.ZipFile(
        io.BytesIO(base64.b64decode(_bundle_archive_b64())), "r"
    ) as archive:
        assert "crowdtensor/heterogeneous_jax_qwen_training.py" in archive.namelist()


def test_tpu_stage_diagnostic_package_is_private_incremental_and_non_acceptance(
    tmp_path,
) -> None:
    report = build_diagnostic_package(
        tmp_path / "diagnostic",
        owner="fixture-owner",
        slug="fixture-tpu-stage-diagnostic",
        hf_token="private-hf-token",
        diagnostic_nonce="private-diagnostic-nonce",
    )
    package = tmp_path / "diagnostic" / "private-kernel"
    metadata = json.loads(
        (package / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    rendered = (package / "kernel.py").read_text(encoding="utf-8")
    py_compile.compile(str(package / "kernel.py"), doraise=True)
    public = {key: value for key, value in report.items() if key != "package_dir"}

    assert metadata["is_private"] == "true"
    assert metadata["enable_tpu"] == "true"
    assert metadata["machine_shape"] == "tpuV5e8"
    assert report["stage_id"] == 2
    assert report["diagnostic_only"] is True
    assert report["full_training_gate_evidence"] is False
    assert "progress_callback=shard_progress" in rendered
    assert 'update_phase(report, "forward_compiling")' in rendered
    assert '"synthetic_boundary_tensors_used": True' in rendered
    assert "private-hf-token" not in json.dumps(public)
    assert "private-diagnostic-nonce" not in json.dumps(public)


def test_remote_worker_entry_exposes_tpu_policy_without_private_values(
    tmp_path, monkeypatch
) -> None:
    manifest = tiny_manifest()
    config = tiny_config()
    tokenized = {
        "schema": "crowdtensor_heterogeneous_tokenized_private_v1",
        "training_manifest_hash": manifest["content_hash"],
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "sequence_length": manifest["training"]["sequence_length"],
        "train": [[1] * 8 for _ in range(6)],
        "validation": [[1] * 8],
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
    captured = {}

    def fake_miner(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "steps_completed": 3,
            "capability": {"tpu_groups": [{"device_count": 8}]},
            "credential_values_public": False,
            "public_artifact_safe": True,
        }

    monkeypatch.setattr(worker_entry, "run_heterogeneous_miner", fake_miner)
    report = worker_entry.run_worker(
        private_configuration_path=private,
        output_path=tmp_path / "worker.json",
        private_root=tmp_path / "worker-private",
        deployment_role="tpu_old",
        identity_nonce="private-nonce",
        device_policy="jax_tpu",
        max_steps=3,
        wait_timeout=60.0,
        operation_timeout=45.0,
    )

    assert report["ok"] is True
    assert report["jax_tpu_resource_group_expected"] is True
    assert report["visible_cuda_device_count_expected"] == 0
    assert captured["device_policy"] == "jax_tpu"
    assert captured["cuda_devices"] is None
    assert "private-token" not in json.dumps(report)
    assert "private-nonce" not in json.dumps(report)
