from pathlib import Path

from scripts.community_kaggle_live_package import KAGGLE_RUNTIME_REQUIREMENTS, build_packages


def test_package_is_private_clean_wheel_and_public_report_redacts_inputs(tmp_path) -> None:
    built = build_packages(
        tmp_path / "private",
        owner="authorized-owner",
        coordinator_url="https://private-route.example",
        miner_token="private-miner-token-value",
        unique_suffix="unit-test",
    )
    report = built["report"]
    assert report["ok"] is True
    assert report["providers"] == ["kaggle_cpu", "kaggle_cuda"]
    assert report["logical_node_count"] == 2
    assert report["workspace_import_used"] is False
    assert "private-miner-token-value" not in str(report)
    assert "private-route.example" not in str(report)
    gpu = (tmp_path / "private" / "stage0" / "kernel.py").read_text()
    cpu = (tmp_path / "private" / "stage1" / "kernel.py").read_text()
    assert 'pathlib.Path("/kaggle/temp")' in gpu
    assert "ct-community-site" in gpu
    assert "--target" in gpu
    assert "--no-deps" in gpu
    assert "model_stack_import_verified" in gpu
    assert "runtime_requirements_exact_pins_verified" in gpu
    assert all("==" in item for item in KAGGLE_RUNTIME_REQUIREMENTS)
    lock = tuple(
        line.strip()
        for line in Path("requirements/community-kaggle-runtime.lock").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    assert lock == KAGGLE_RUNTIME_REQUIREMENTS
    assert "env=env" in gpu
    assert "limits = [30, 0]" in gpu
    assert "limits = [30, 0]" not in cpu
