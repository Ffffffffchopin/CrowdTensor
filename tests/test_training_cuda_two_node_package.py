from __future__ import annotations

import importlib.util
import json
import py_compile
import zipfile
from pathlib import Path

from scripts.training_cuda_two_node_package import build_package


def _fixture(root) -> None:
    (root / "base_model").mkdir(parents=True)
    (root / "base_model" / "config.json").write_text("{}", encoding="utf-8")
    (root / "initial_adapter").mkdir()
    (root / "initial_adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    (root / "initial_adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (root / "private_dataset.jsonl").write_text("{}\n", encoding="utf-8")


def test_two_node_package_embeds_private_fixture_and_coordinator_inputs(tmp_path) -> None:
    fixture = tmp_path / "fixture"
    _fixture(fixture)
    report = build_package(
        tmp_path / "package",
        owner="gpu-owner",
        slug="ct-cuda-node-stage0",
        role="stage0",
        fixture_dir=fixture,
        coordinator_url="https://private.example.invalid",
        coordinator_token="private-token-value",
        run_id="run-1",
    )
    package = tmp_path / "package" / "private-kernel"
    source = (package / "kernel.py").read_text(encoding="utf-8")
    metadata = json.loads((package / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["role"] == "stage0"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert metadata["enable_tpu"] == "false"
    assert "private-token-value" not in source
    assert "private.example.invalid" not in source
    assert "run_cross_node_stage" in source
    assert "def run_embedded_single_kernel_gate():" in source
    assert source.index(
        'report["embedded_single_kernel_gate"] = run_embedded_single_kernel_gate()'
    ) < source.index("pipeline = run_cross_node_stage(")
    assert "run_remote_lora_miner" in source
    assert "pip\", \"uninstall\", \"-y\", \"torchao" in source
    assert "def public_blocker(exc):" in source
    assert "def checkpoint_bundle():" in source
    assert "training_cuda_two_node_stage0_checkpoint_bundle.zip" in source
    assert '"pipeline": PRIVATE_ROOT / "pipeline" / "checkpoint"' in source
    assert '"miner": PRIVATE_ROOT / "miner" / "checkpoint"' in source
    assert "contains_pipeline_and_miner_checkpoints" in source
    assert 're.sub(r"https?://[^\\s]+", "<private-url>"' in source
    assert source.index('importlib.metadata.version("torchao")') < source.index(
        "from crowdtensor.cuda_training_worker import"
    )
    assert "if __name__ == '__main__':" in source
    py_compile.compile(str(package / "kernel.py"), doraise=True)


def test_generated_two_node_checkpoint_bundle_excludes_fixture_and_source(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    fixture = tmp_path / "fixture"
    _fixture(fixture)
    report = build_package(
        tmp_path / "package",
        owner="gpu-owner",
        slug="ct-cuda-node-checkpoint-test",
        role="stage1",
        fixture_dir=fixture,
        coordinator_url="https://private.example.invalid",
        coordinator_token="private-token-value",
        run_id="run-1",
    )
    kernel_path = Path(report["package_dir"]) / "kernel.py"
    spec = importlib.util.spec_from_file_location("generated_two_node_checkpoint_kernel", kernel_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for label in ("pipeline", "miner"):
        root = module.PRIVATE_ROOT / label / "checkpoint"
        root.mkdir(parents=True, exist_ok=True)
        (root / "checkpoint.json").write_text("{}", encoding="utf-8")
        (root / "adapter.safetensors").write_bytes(label.encode("ascii"))
    (module.BUNDLE_ROOT / "fixture" / "private_dataset.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (module.BUNDLE_ROOT / "fixture" / "private_dataset.jsonl").write_text("private", encoding="utf-8")
    summary = module.checkpoint_bundle()
    assert summary["present"] is True
    assert summary["contains_pipeline_and_miner_checkpoints"] is True
    assert summary["file_count"] == 4
    assert summary["file_hash"].startswith("sha256:")
    with zipfile.ZipFile(module.CHECKPOINT_BUNDLE_PATH) as archive:
        names = set(archive.namelist())
    assert names == {
        "miner/adapter.safetensors",
        "miner/checkpoint.json",
        "pipeline/adapter.safetensors",
        "pipeline/checkpoint.json",
    }
    encoded = json.dumps(summary, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert "/root/" not in encoded


def test_stage0_checkpoint_bundle_contains_embedded_single_gate_checkpoints(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    fixture = tmp_path / "fixture"
    _fixture(fixture)
    report = build_package(
        tmp_path / "package",
        owner="gpu-owner",
        slug="ct-cuda-node-stage0-checkpoint-test",
        role="stage0",
        fixture_dir=fixture,
        coordinator_url="https://private.example.invalid",
        coordinator_token="private-token-value",
        run_id="run-1",
    )
    kernel_path = Path(report["package_dir"]) / "kernel.py"
    spec = importlib.util.spec_from_file_location(
        "generated_two_node_stage0_checkpoint_kernel", kernel_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    roots = {
        "pipeline": module.PRIVATE_ROOT / "pipeline" / "checkpoint",
        "miner": module.PRIVATE_ROOT / "miner" / "checkpoint",
        "single_baseline": (
            module.PRIVATE_ROOT / "embedded-single-kernel" / "baseline" / "checkpoint"
        ),
        "single_resumed": (
            module.PRIVATE_ROOT / "embedded-single-kernel" / "resumed" / "checkpoint"
        ),
    }
    for label, root in roots.items():
        root.mkdir(parents=True, exist_ok=True)
        (root / "checkpoint.json").write_text(json.dumps({"label": label}), encoding="utf-8")
    summary = module.checkpoint_bundle()
    assert summary["present"] is True
    assert summary["contains_pipeline_and_miner_checkpoints"] is True
    assert summary["contains_baseline_and_resumed_checkpoints"] is True
    assert summary["file_count"] == 4
    with zipfile.ZipFile(module.CHECKPOINT_BUNDLE_PATH) as archive:
        assert set(archive.namelist()) == {
            f"{label}/checkpoint.json" for label in roots
        }
