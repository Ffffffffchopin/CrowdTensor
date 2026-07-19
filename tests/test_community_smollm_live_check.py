import json

from scripts.community_smollm_live_check import check
from crowdtensor.smollm_training import LIVE_SCHEMA, MODEL_ID, MODEL_REVISION


def valid_report() -> dict:
    return {
        "schema": LIVE_SCHEMA,
        "ok": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "real_open_model_weights": True,
        "random_or_synthetic_weights_used": False,
        "node_scope": "Kaggle logical multi-node",
        "logical_miner_count": 2,
        "logical_stage_count": 2,
        "distinct_worker_processes": True,
        "physical_multi_machine_verified": False,
        "single_process_smoke": False,
        "stage_specs": [
            {"stage_id": 0, "layer_start": 0, "layer_end": 15},
            {"stage_id": 1, "layer_start": 15, "layer_end": 30},
        ],
        "committed_step_ids": [1, 2],
        "strictly_contiguous_atomic_steps": True,
        "all_stage_optimizer_steps_applied": True,
        "finite_loss_verified": True,
        "both_stage_adapters_updated": True,
        "stage_checkpoints": [
            {"stage_id": 0, "checkpoint_hash": "sha256:" + "a" * 64, "checkpoint_tensor_count": 10, "adapter_updated": True},
            {"stage_id": 1, "checkpoint_hash": "sha256:" + "b" * 64, "checkpoint_tensor_count": 10, "adapter_updated": True},
        ],
        "export": {"standard_peft_format": True, "adapter_tensor_count": 20},
        "reload": {"adapter_reload_verified": True, "independent_process_reload": True},
        "clean_install_required": True,
        "workspace_import_used": False,
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def test_checker_accepts_complete_real_two_stage_contract(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(valid_report()), encoding="utf-8")
    assert check(path)["ok"] is True


def test_checker_rejects_single_process_synthetic_or_overclaimed_evidence(tmp_path) -> None:
    value = valid_report()
    value["single_process_smoke"] = True
    value["random_or_synthetic_weights_used"] = True
    value["physical_multi_machine_verified"] = True
    value["both_stage_adapters_updated"] = False
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    result = check(path)
    assert result["ok"] is False
    assert len(result["errors"]) >= 4


def test_checker_rejects_sensitive_values(tmp_path) -> None:
    value = valid_report()
    value["token"] = "this-is-a-private-secret-token"
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    result = check(path)
    assert result["ok"] is False
    assert "community_smollm_live_public_safety_invalid" in result["errors"]
