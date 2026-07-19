from __future__ import annotations

from scripts.community_kaggle_wheel_smoke_probe import _source


def test_wheel_smoke_uses_isolated_pip_target_install() -> None:
    source = _source("https://coordinator.invalid", "private-token")

    assert '"--target",str(install_root)' in source
    assert 'pathlib.Path("/kaggle/temp")' in source
    assert 'env["PYTHONPATH"]=str(install_root)' in source
    assert '"-m","crowdtensor.community_cli"' in source
    assert '"fresh_install_kind":"pip_target"' in source
    assert '"fresh_install_root_per_kernel":True' in source
    assert '"installed_package_under_install_root"' in source
    assert '"model_stack_import_verified":True' in source
    assert '"runtime_requirements_exact_pins_verified"' in source
    assert "transformers==5.9.0" in source
    assert "import torch,transformers,peft,safetensors,fastapi" in source
    assert "-m\",\"venv" not in source
