import concurrent.futures
import hashlib
import json
from pathlib import Path
import socket
import sys
import threading
import time

import uvicorn

from crowdtensor.heterogeneous_training_beta import (
    HeterogeneousTrainingBetaController,
    create_heterogeneous_training_beta_app,
)
from crowdtensor.heterogeneous_training_manifest import (
    qwen25_7b_lora_tpu_manifest,
    stable_hash,
)
from crowdtensor.heterogeneous_training_miner import run_heterogeneous_miner
from scripts.training_heterogeneous_export_reload_probe import (
    verify_exported_forward,
)
from tests.test_heterogeneous_qwen_training import tiny_source


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Service:
    def __init__(self, controller, owner_token, miner_token):
        self.port = free_port()
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_heterogeneous_training_beta_app(
                    controller,
                    owner_token=owner_token,
                    miner_token=miner_token,
                ),
                host="127.0.0.1",
                port=self.port,
                log_level="error",
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.server.started:
                return f"http://127.0.0.1:{self.port}"
            time.sleep(0.05)
        raise TimeoutError("test service did not start")

    def __exit__(self, *_args):
        self.server.should_exit = True
        self.thread.join(timeout=10)


def identity(name: str) -> str:
    return "sha256:" + hashlib.sha256(name.encode()).hexdigest()


def test_tpu_product_status_and_bootstrap_expose_public_mesh_contract(tmp_path) -> None:
    manifest = qwen25_7b_lora_tpu_manifest()
    manifest_path = tmp_path / "manifest.json"
    config_path = tmp_path / "config.json"
    tokenized_path = tmp_path / "tokenized.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 28,
                "hidden_size": 3584,
            }
        ),
        encoding="utf-8",
    )
    tokenized_path.write_text(
        json.dumps(
            {
                "schema": "crowdtensor_heterogeneous_tokenized_private_v1",
                "training_manifest_hash": manifest["content_hash"],
                "model_id": manifest["model"]["model_id"],
                "model_revision": manifest["model"]["model_revision"],
                "sequence_length": 8,
                "train": [[1] * 8 for _ in range(6)],
                "validation": [[2] * 8],
            }
        ),
        encoding="utf-8",
    )
    controller = HeterogeneousTrainingBetaController.create(
        tmp_path / "job",
        manifest_path=manifest_path,
        config_path=config_path,
        tokenized_payload_path=tokenized_path,
        enable_jax_tpu=True,
    )

    status = controller.status()
    bootstrap = controller.bootstrap()

    assert status["topology"] == "manifest-driven-cpu-cuda-jax-tpu-stages"
    assert status["jax_tpu_trainable_stage_ready"] is True
    assert status["required_device_types"] == ["cpu", "cuda", "jax_tpu"]
    assert bootstrap["jax_tpu_trainable_stages_supported"] is True
    assert bootstrap["training_manifest"]["stages"][2]["preferred_device_type"] == "jax_tpu"
    assert status["credential_values_public"] is False
    controller.cleanup()


def test_product_cpu_miners_pause_replace_resume_export_and_cleanup(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest, config, _full_model = tiny_source(source)
    manifest_path = tmp_path / "manifest.json"
    config_path = tmp_path / "config.json"
    tokenized_path = tmp_path / "tokenized.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    tokenized = {
        "schema": "crowdtensor_heterogeneous_tokenized_private_v1",
        "training_manifest_hash": manifest["content_hash"],
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "dataset_id": manifest["dataset"]["dataset_id"],
        "dataset_revision": manifest["dataset"]["dataset_revision"],
        "sequence_length": manifest["training"]["sequence_length"],
        "train": [
            [1, 7, 11, 3, 9, 2],
            [4, 5, 8, 13, 6, 10],
            [2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2],
        ],
        "validation": [[1, 2, 3, 4, 5, 6]],
    }
    tokenized_path.write_text(json.dumps(tokenized), encoding="utf-8")
    job = tmp_path / "job"
    controller = HeterogeneousTrainingBetaController.create(
        job,
        manifest_path=manifest_path,
        config_path=config_path,
        tokenized_payload_path=tokenized_path,
        lease_seconds=30.0,
    )
    credentials = controller.credentials()
    package_parent = Path(__file__).resolve().parent.parent
    monkeypatch.setattr(
        sys,
        "path",
        [
            value
            for value in sys.path
            if value and Path(value).resolve() != package_parent
        ],
    )
    monkeypatch.chdir(tmp_path)

    def run(name: str, generation: str, max_steps: int, url: str):
        return run_heterogeneous_miner(
            coordinator_url=url,
            coordinator_token=credentials["miner_token"],
            run_id=controller.run_id,
            miner_id_hash=identity(name),
            registration_nonce=f"{generation}:{name}",
            training_manifest=manifest,
            config=config,
            tokenized_payload=tokenized,
            private_root=tmp_path / generation / name,
            device_policy="cpu",
            max_stage_count=1,
            max_steps_per_session=max_steps,
            wait_timeout=120.0,
            heartbeat_interval_seconds=1.0,
            attached_model_root=source,
            run_microbenchmark=False,
        )

    with Service(
        controller, credentials["owner_token"], credentials["miner_token"]
    ) as url:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            old = [
                executor.submit(run, f"old-{index}", "old", 1, url)
                for index in range(2)
            ]
            old_reports = [future.result(timeout=180) for future in old]
        paused = controller.status()
        assert paused["committed_step"] == 1
        assert paused["online_miner_count"] == 0
        assert paused["overall_state"] == "waiting_for_miners"

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            replacements = [
                executor.submit(run, f"new-{index}", "new", 0, url)
                for index in range(2)
            ]
            replacement_reports = [
                future.result(timeout=180) for future in replacements
            ]
        private_export_configuration = tmp_path / "private-export-configuration.json"
        private_export_configuration.write_text(
            json.dumps(
                {
                    "coordinator_url": url,
                    "coordinator_token": credentials["miner_token"],
                    "hf_token": "",
                }
            ),
            encoding="utf-8",
        )
        export_reload = verify_exported_forward(
            private_configuration_path=private_export_configuration,
            private_root=tmp_path / "export-reload-private",
            output_path=tmp_path / "export-reload.json",
            wait_timeout=120.0,
            source_root=source,
        )

    final = controller.status()
    assert final["overall_state"] == "completed"
    assert final["committed_step"] == 2
    assert final["runtime"]["committed_steps"] == [1, 2]
    assert final["runtime"]["committed_steps_contiguous"] is True
    assert final["placement_generation"] >= 2
    assert all(report["ok"] for report in old_reports + replacement_reports)
    assert all(report["steps_completed"] == 1 for report in old_reports)
    assert all(report["central_checkpoint_restore_count"] == 1 for report in replacement_reports)
    assert all(report["positive_lora_gradient_norms"] for report in old_reports + replacement_reports)
    assert all(
        report["optimizer_and_scheduler_steps_applied"]
        for report in old_reports + replacement_reports
    )
    old_ids = {report["miner_id_hash"] for report in old_reports}
    new_ids = {report["miner_id_hash"] for report in replacement_reports}
    assert old_ids.isdisjoint(new_ids)
    assert export_reload["adapter_reload_verified"] is True
    assert export_reload["forward_inference_verified"] is True
    assert export_reload["finite_logits_verified"] is True
    assert len(export_reload["stage_reports"]) == 2

    exported = controller.export()
    assert exported["ok"] is True
    assert exported["global_step"] == 2
    assert exported["adapter_tensor_count"] == 28
    assert (job / "exported_adapter/adapter_model.safetensors").is_file()
    assert (job / "exported_adapter/adapter_config.json").is_file()
    cleanup = controller.cleanup()
    assert cleanup["ok"] is True
    assert cleanup["tensor_transport_cleanup"]["all_messages_removed"] is True
    assert cleanup["live_resources_left_running"] is False


def test_stable_miner_survives_peer_abort_and_replacement(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest, config, _full_model = tiny_source(source)
    manifest_path = tmp_path / "manifest.json"
    config_path = tmp_path / "config.json"
    tokenized_path = tmp_path / "tokenized.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    tokenized = {
        "schema": "crowdtensor_heterogeneous_tokenized_private_v1",
        "training_manifest_hash": manifest["content_hash"],
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "sequence_length": manifest["training"]["sequence_length"],
        "train": [
            [1, 7, 11, 3, 9, 2],
            [4, 5, 8, 13, 6, 10],
        ],
        "validation": [[1, 2, 3, 4, 5, 6]],
    }
    tokenized_path.write_text(json.dumps(tokenized), encoding="utf-8")
    controller = HeterogeneousTrainingBetaController.create(
        tmp_path / "job",
        manifest_path=manifest_path,
        config_path=config_path,
        tokenized_payload_path=tokenized_path,
        lease_seconds=30.0,
    )
    credentials = controller.credentials()

    def run(name: str, max_steps: int, url: str):
        return run_heterogeneous_miner(
            coordinator_url=url,
            coordinator_token=credentials["miner_token"],
            run_id=controller.run_id,
            miner_id_hash=identity(name),
            registration_nonce=name,
            training_manifest=manifest,
            config=config,
            tokenized_payload=tokenized,
            private_root=tmp_path / name,
            device_policy="cpu",
            max_stage_count=1,
            max_steps_per_session=max_steps,
            wait_timeout=120.0,
            heartbeat_interval_seconds=1.0,
            attached_model_root=source,
            run_microbenchmark=False,
        )

    with Service(
        controller, credentials["owner_token"], credentials["miner_token"]
    ) as url:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            old = executor.submit(run, "old", 1, url)
            stable = executor.submit(run, "stable", 0, url)
            old_report = old.result(timeout=180)
            paused = controller.status()
            assert paused["committed_step"] == 1
            assert paused["overall_state"] == "waiting_for_miners"
            replacement = executor.submit(run, "replacement", 0, url)
            replacement_report = replacement.result(timeout=180)
            stable_report = stable.result(timeout=180)

    final = controller.status()
    assert old_report["steps_completed"] == 1
    assert old_report["blockers"] == []
    assert stable_report["ok"] is True, stable_report["blockers"]
    assert stable_report["blockers"] == []
    assert stable_report["steps_completed"] == 2
    assert replacement_report["ok"] is True
    assert replacement_report["steps_completed"] == 1
    assert replacement_report["central_checkpoint_restore_count"] == 1
    assert final["runtime"]["committed_steps"] == [1, 2]
    assert final["placement_generation"] in {2, 3}
    assert final["placement_plan"]["rebalance_reason"] != "checkpoint_recovery"
