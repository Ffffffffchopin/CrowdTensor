from crowdtensor.model_adapter import stable_hash
from scripts.model_ecosystem_beta_rc_check import REQUIRED_ARTIFACTS, check_report


HASH = "sha256:" + "a" * 64


def valid_report():
    value = {
        "schema": "crowdtensor_model_ecosystem_beta_rc_v1",
        "model_adapter_ecosystem_ready": True,
        "entry_point_plugin_contract_ready": True,
        "mistral_adapter_ready": True,
        "mistral_real_heterogeneous_training_verified": True,
        "dual_wheel_clean_install_verified": True,
        "checkpoint_replacement_verified": True,
        "peft_export_reload_verified": True,
        "regression_gate_verified": True,
        "cleanup_verified": True,
        "goal_achieved": True,
        "supported_model_families": ["mistral", "qwen2", "smollm2"],
        "plugin_smoke_summary": {
            "ok": True,
            "adapter_id": "mistral_lora_v1",
            "family": "mistral",
            "model_id": "Locutusque/TinyMistral-248M-v2",
            "model_revision": "0f57b17cb317bb322c7c1466b669c681f80c058f",
            "registration_kind": "entry_point_plugin",
            "entry_point_group": "crowdtensor.model_adapters.v1",
            "conformance_verified": True,
            "partition_verified": True,
            "isolated_venv": True,
            "wheel_install_no_deps": True,
            "workspace_import_used": False,
        },
        "live_summary": {
            "strict_check_ok": True,
            "mistral_live_verified": True,
            "live_run_performed": True,
            "committed_step_ids": list(range(1, 9)),
            "accepted_providers": ["kaggle_cpu", "kaggle_cuda"],
            "gpu_worker_replacement_verified": True,
            "restored_checkpoint_step": 4,
            "optimizer_state_restored": True,
            "adapter_tensor_count": 168,
            "adapter_reload_verified": True,
            "cleanup_verified": True,
            "public_safety_verified": True,
        },
        "quality_summary": {
            "ok": True,
            "passed": 40,
            "failed": 0,
            "py_compile_ok": True,
            "plugin_registry_tests_included": True,
            "mistral_architecture_tests_included": True,
            "live_report_checker_tests_included": True,
        },
        "cleanup": {
            "all_remote_kernels_deleted": True,
            "live_resources_left_running": False,
            "private_runtime_removed": True,
            "community_maturity_ledger_modified": False,
        },
        "artifacts": {
            name: {"relative_path": f"evidence/{name}", "sha256": HASH}
            for name in REQUIRED_ARTIFACTS
        },
        "unsupported_claims": {
            "arbitrary_architecture_support_verified": False,
            "full_parameter_training_verified": False,
            "mistral_7b_live_verified": False,
            "physical_multi_machine_verified": False,
            "production_sla_verified": False,
        },
        "public_artifact_safe": True,
    }
    value["content_hash"] = stable_hash(value)
    return value


def test_model_ecosystem_rc_checker_accepts_complete_evidence() -> None:
    result = check_report(valid_report())
    assert result["ok"] is True
    assert result["goal_achieved"] is True


def test_model_ecosystem_rc_checker_rejects_live_or_overclaim_gap() -> None:
    value = valid_report()
    value["live_summary"]["mistral_live_verified"] = False
    value["unsupported_claims"]["mistral_7b_live_verified"] = True
    value["content_hash"] = stable_hash(
        {key: item for key, item in value.items() if key != "content_hash"}
    )
    result = check_report(value)
    assert "model_ecosystem_rc_live_summary_invalid" in result["errors"]
    assert "model_ecosystem_rc_overclaim_invalid" in result["errors"]
