from __future__ import annotations

import importlib.util
import json
import py_compile
import zipfile
from pathlib import Path

from scripts.training_cuda_single_kernel_package import build_package


def test_single_kernel_package_is_private_bounded_and_spawn_safe(tmp_path) -> None:
    report = build_package(
        tmp_path / "package",
        owner="gpu-owner",
        slug="ct-cuda-gate-test",
        total_steps=4,
        interrupt_after_step=2,
    )
    package = tmp_path / "package" / "private-kernel"
    metadata = json.loads((package / "kernel-metadata.json").read_text(encoding="utf-8"))
    source = (package / "kernel.py").read_text(encoding="utf-8")
    assert report["ok"] is True
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert metadata["enable_tpu"] == "false"
    assert "if __name__ == '__main__':" in source
    assert "run_two_cuda_process_pipeline" in source
    assert "total_steps=4" in source
    assert "interrupt_stage1_after_step=2" in source
    assert "def checkpoint_bundle():" in source
    assert "training_cuda_single_kernel_checkpoint_bundle.zip" in source
    assert '"baseline": PRIVATE_ROOT / "baseline" / "checkpoint"' in source
    assert '"resumed": PRIVATE_ROOT / "resumed" / "checkpoint"' in source
    assert "contains_baseline_and_resumed_checkpoints" in source
    assert "def public_blocker(exc):" in source
    assert "KAGGLE_KEY" not in source
    py_compile.compile(str(package / "kernel.py"), doraise=True)


def test_generated_single_kernel_checkpoint_bundle_is_scoped_and_hashed(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    report = build_package(
        tmp_path / "package",
        owner="gpu-owner",
        slug="ct-cuda-checkpoint-test",
        total_steps=4,
        interrupt_after_step=2,
    )
    kernel_path = Path(report["package_dir"]) / "kernel.py"
    spec = importlib.util.spec_from_file_location("generated_single_checkpoint_kernel", kernel_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for run in ("baseline", "resumed"):
        root = module.PRIVATE_ROOT / run / "checkpoint"
        root.mkdir(parents=True, exist_ok=True)
        (root / "stage0_checkpoint.json").write_text("{}", encoding="utf-8")
        (root / "stage0_adapter.safetensors").write_bytes(run.encode("ascii"))
    (module.SOURCE_ROOT / "must-not-be-bundled.py").write_text("secret", encoding="utf-8")
    summary = module.checkpoint_bundle()
    assert summary["present"] is True
    assert summary["contains_baseline_and_resumed_checkpoints"] is True
    assert summary["file_count"] == 4
    assert summary["file_hash"].startswith("sha256:")
    with zipfile.ZipFile(module.CHECKPOINT_BUNDLE_PATH) as archive:
        names = set(archive.namelist())
    assert names == {
        "baseline/stage0_adapter.safetensors",
        "baseline/stage0_checkpoint.json",
        "resumed/stage0_adapter.safetensors",
        "resumed/stage0_checkpoint.json",
    }
    encoded = json.dumps(summary, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert "/root/" not in encoded
