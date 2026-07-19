from pathlib import Path

from scripts.community_kaggle_live_package import KAGGLE_RUNTIME_REQUIREMENTS
from scripts.mistral_kaggle_live_package import build_packages


def test_mistral_package_installs_both_wheels_and_replaces_gpu_worker(tmp_path) -> None:
    built = build_packages(
        tmp_path / "private",
        owner="authorized-owner",
        coordinator_url="https://private-route.example",
        miner_token="private-miner-token-value",
        unique_suffix="unit-test",
    )
    report = built["report"]
    assert report["ok"] is True
    assert report["model_adapter_id"] == "mistral_lora_v1"
    assert report["target_steps"] == 8
    assert report["checkpoint_steps"] == [4, 8]
    assert report["providers"] == ["kaggle_cpu", "kaggle_cuda"]
    assert "private-miner-token-value" not in str(report)
    assert "private-route.example" not in str(report)
    gpu = (tmp_path / "private" / "stage0" / "kernel.py").read_text()
    cpu = (tmp_path / "private" / "stage1" / "kernel.py").read_text()
    assert "/v1/community-live/wheel" in gpu
    assert "/v1/community-live/adapter-wheel" in gpu
    assert "mistral_lora_v1" in gpu
    assert "entry_point_plugin" in gpu
    assert "limits = [4, 0]" in gpu
    assert "limits = [4, 0]" not in cpu
    assert "--target" in gpu and "--no-deps" in gpu
    assert all("==" in item for item in KAGGLE_RUNTIME_REQUIREMENTS)
    metadata = Path(tmp_path / "private" / "stage0" / "kernel-metadata.json")
    assert '"enable_gpu": "true"' in metadata.read_text()
