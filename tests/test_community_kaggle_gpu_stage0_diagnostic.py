from scripts.community_kaggle_gpu_stage0_diagnostic import _source


def test_gpu_stage0_diagnostic_is_bounded_clean_install_and_public_safe() -> None:
    source = _source("https://private.invalid", "private-token")
    assert '"full_live_gate":False' in source
    assert 'pathlib.Path("/kaggle/temp")' in source
    assert '"--target",str(install_root)' in source
    assert "adapter.load_model" in source
    assert "adapter.apply_lora" in source
    assert "optimizer.step()" in source
    assert "transformers==5.9.0" in source
    assert "accelerate==1.13.0" in source
    assert '"error_message_hash"' in source
    assert '"traceback_frames_public"' in source
    assert "item.filename" in source


def test_gpu_diagnostic_can_include_two_process_dual_stage_proof() -> None:
    source = _source(
        "https://private.invalid", "private-token", include_dual_stage=True
    )
    assert "include_dual_stage=True" in source
    assert "crowdtensor.community_smollm_runner" in source
    assert '"--steps","2"' in source
    assert '"devices":["cuda","cuda"]' in source
    assert '"dual_stage_output_summary_public"' in source
    assert "stderr=subprocess.STDOUT" in source
