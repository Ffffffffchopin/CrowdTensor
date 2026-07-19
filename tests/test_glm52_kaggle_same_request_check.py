from __future__ import annotations

from scripts import glm52_kaggle_same_request_check as check


def _safety() -> dict:
    return {
        "public_artifact_safe": True,
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
    }


def _stage(provider: str, stage_id: int) -> dict:
    return {
        "stage_id": stage_id,
        "provider": provider,
        "model_id": check.MODEL_ID,
        "coordinator_request_id_hash": "sha256:" + "b" * 64,
        "stage_execution_verified": True,
        "stage_decode_verified": True,
        "stage_output_hash": "sha256:" + str(stage_id) * 64,
        "weight_tensor_values_loaded": True,
        "weight_value_byte_count": 16,
        "weight_value_sha256": "sha256:" + "w" * 64,
        "weight_tensor_values_public": False,
        "live_run_performed": True,
        "activation_public": False,
        "kv_cache_public": False,
    }


def _report(**overrides) -> dict:
    report = {
        "schema": check.PROOF_SCHEMA,
        "ok": True,
        "public_artifact_safe": True,
        "model": {"model_id": check.MODEL_ID, "fallback_model_used": False},
        "glm52_kaggle_same_request_verified": True,
        "live_run_performed": True,
        "success": {
            "same_request_decode_verified": True,
            "generated_token_count": 1,
            "generated_token_hash": "sha256:" + "a" * 64,
            "accepted_providers": sorted(check.REQUIRED_PROVIDERS),
        },
        "same_request": {
            "coordinator_request_verified": True,
            "coordinator_request_id_hash": "sha256:" + "b" * 64,
            "model_id": check.MODEL_ID,
        },
        "stage_reports": [
            _stage("kaggle_cuda", 0),
            _stage("kaggle_jax_tpu", 1),
            _stage("kaggle_cpu", 2),
        ],
        "cleanup": {
            "temporary_kaggle_kernels_deleted": True,
            "temporary_private_packages_removed": True,
            "live_resources_left_running": False,
            "public_artifact_safe": True,
        },
        "fallback_model_used": False,
        "queue_only_evidence": False,
        "metadata_only": False,
        "stage_smoke_only": False,
        "safety": _safety(),
    }
    report.update(overrides)
    return report


def test_checker_accepts_full_same_request_proof() -> None:
    report = _report()

    assert check.validate_report(report, require_verified=True) == []
    assert check.same_request_verified(report) is True


def test_checker_rejects_missing_required_provider() -> None:
    report = _report(
        success={
            "same_request_decode_verified": True,
            "generated_token_count": 1,
            "generated_token_hash": "sha256:" + "a" * 64,
            "accepted_providers": ["kaggle_cuda", "kaggle_cpu"],
        }
    )

    errors = check.validate_report(report, require_verified=True)

    assert "required_provider_missing:kaggle_jax_tpu" in errors


def test_checker_rejects_non_glm_fallback_success() -> None:
    report = _report(model={"model_id": "Qwen/Qwen2.5-32B-Instruct", "fallback_model_used": True})

    errors = check.validate_report(report, require_verified=True)

    assert "model_id_not_glm52" in errors
    assert "fallback_model_used" in errors


def test_checker_rejects_stage_smoke_or_queue_overclaim() -> None:
    report = _report(queue_only_evidence=True, stage_smoke_only=True)

    errors = check.validate_report(report, require_verified=True)

    assert "queue_evidence_overclaim" in errors
    assert "stage_smoke_only_overclaim" in errors


def test_checker_rejects_stage_without_weight_value_evidence() -> None:
    stages = [_stage("kaggle_cuda", 0), _stage("kaggle_jax_tpu", 1), _stage("kaggle_cpu", 2)]
    stages[1]["weight_tensor_values_loaded"] = False
    stages[1]["weight_value_sha256"] = ""
    report = _report(stage_reports=stages)

    errors = check.validate_report(report, require_verified=True)

    assert "stage_weight_values_not_loaded:kaggle_jax_tpu" in errors
    assert "stage_execution_missing:kaggle_jax_tpu" in errors


def test_checker_rejects_stage_without_decode_evidence() -> None:
    stages = [_stage("kaggle_cuda", 0), _stage("kaggle_jax_tpu", 1), _stage("kaggle_cpu", 2)]
    stages[1]["stage_decode_verified"] = False
    report = _report(stage_reports=stages)

    errors = check.validate_report(report, require_verified=True)

    assert "stage_decode_missing:kaggle_jax_tpu" in errors
    assert "stage_execution_missing:kaggle_jax_tpu" in errors


def test_require_verified_rejects_blocker_report() -> None:
    report = {
        "schema": check.PROOF_SCHEMA,
        "ok": True,
        "public_artifact_safe": True,
        "glm52_kaggle_same_request_verified": False,
        "safety": _safety(),
        "blockers": ["glm52_same_request_report_missing"],
    }

    errors = check.validate_report(report, require_verified=True)

    assert "same_request_not_verified" in errors
