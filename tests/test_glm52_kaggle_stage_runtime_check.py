from __future__ import annotations

from scripts import glm52_kaggle_stage_runtime_check as check


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _report(**overrides) -> dict:
    report = {
        "schema": check.STAGE_SCHEMA,
        "ok": True,
        "public_artifact_safe": True,
        "model_id": check.MODEL_ID,
        "compatible_weight_repo": check.COMPATIBLE_WEIGHT_REPO,
        "provider": "kaggle_jax_tpu",
        "stage_id": 4,
        "stage_layer_range": [28, 35],
        "coordinator_request_id_hash": _hash("b"),
        "stage_execution_verified": True,
        "stage_output_hash": _hash("a"),
        "weight_tensor_values_loaded": True,
        "weight_value_byte_count": 16,
        "weight_value_sha256": _hash("c"),
        "weight_tensor_values_public": False,
        "live_run_performed": True,
        "fallback_model_used": False,
        "queue_only_evidence": False,
        "metadata_only": False,
        "stage_smoke_only": False,
        "activation_public": False,
        "kv_cache_public": False,
        "safety": {
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }
    report.update(overrides)
    return report


def test_accepts_verified_glm52_stage_runtime_report() -> None:
    report = _report()

    assert check.validate_report(report, require_verified=True) == []
    assert check.stage_runtime_verified(report) is True


def test_rejects_tpu_stage_smoke_as_runtime_proof() -> None:
    report = _report(schema="glm52_awq_tpu_stage_smoke_v1", stage_smoke_only=True)

    errors = check.validate_report(report, require_verified=True)

    assert "stage_smoke_is_not_runtime_proof" in errors
    assert "schema_mismatch" in errors


def test_rejects_missing_live_run_and_request_hash() -> None:
    report = _report(live_run_performed=False, coordinator_request_id_hash="")

    errors = check.validate_report(report, require_verified=True)

    assert "live_run_not_performed" in errors
    assert "coordinator_request_hash_missing" in errors


def test_rejects_stage_runtime_without_weight_value_load_evidence() -> None:
    report = _report(weight_tensor_values_loaded=False, weight_value_sha256="", weight_value_byte_count=0)

    errors = check.validate_report(report, require_verified=True)

    assert "stage_weight_values_not_loaded" in errors
    assert check.stage_runtime_verified(report) is False


def test_rejects_missing_schema() -> None:
    report = _report()
    report.pop("schema")

    errors = check.validate_report(report, require_verified=True)

    assert "schema_mismatch" in errors


def test_rejects_wrong_provider_and_non_glm_model() -> None:
    report = _report(provider="colab_cuda", model_id="Qwen/Qwen2.5-32B-Instruct")

    errors = check.validate_report(report, require_verified=True)

    assert "provider_not_required:colab_cuda" in errors
    assert "model_id_not_glm52" in errors


def test_rejects_public_sensitive_fields() -> None:
    report = _report(raw_prompt="secret prompt", activation_public=True)

    errors = check.validate_report(report, require_verified=True)

    assert any(error.startswith("public_redaction_scan_failed:") for error in errors)
    assert "stage_flag_not_false:activation_public" in errors
