from scripts.model_adapter_plugin_smoke import run_smoke


def test_plugin_smoke_fails_publicly_for_invalid_wheels(tmp_path) -> None:
    core = tmp_path / "crowdtensord-0-py3-none-any.whl"
    plugin = tmp_path / "crowdtensor_mistral_adapter-0-py3-none-any.whl"
    core.write_bytes(b"not-a-wheel")
    plugin.write_bytes(b"not-a-wheel")
    report = run_smoke(
        core_wheel=core,
        adapter_wheel=plugin,
        output_dir=tmp_path / "out",
    )
    assert report["ok"] is False
    assert report["blockers"][0].startswith("model_adapter_plugin_smoke_failed:")
    assert report["workspace_import_used"] is False
    assert report["private_paths_public"] is False
    assert report["public_safety"]["ok"] is True
