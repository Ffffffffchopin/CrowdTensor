from __future__ import annotations

from scripts import glm52_awq_tpu_stage_smoke_check as check


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


def _watch() -> dict:
    return {
        "schema": check.WATCH_SCHEMA,
        "ref": "tpuowner/ct-glm52-awq-tpu-stage-smoke-0704-r1",
        "last_status": "KernelWorkerStatus.QUEUED",
        "observations": [{"attempt": 1, "status": "KernelWorkerStatus.QUEUED", "ok": True}],
        "stage_runtime_adapter_smoke_ready": False,
        "notebook_output_verified": False,
        "public_artifact_safe": True,
        "credentials_public": False,
        "signed_output_url_public": False,
        "safety": _safety(),
    }


def _verified_watch() -> dict:
    report = _watch()
    report.update(
        {
            "last_status": "KernelWorkerStatus.COMPLETE",
            "notebook_output_verified": True,
            "stage_runtime_adapter_smoke_ready": True,
            "tpu_runtime_ready": True,
            "stage_smoke_output": {
                "path": "notebook-output/glm52_awq_tpu_stage_smoke.json",
                "present": True,
                "sha256": "sha256:" + "0" * 64,
            },
            "stage_smoke_check": {
                "schema": check.SCHEMA,
                "ok": True,
                "error_count": 0,
                "errors": [],
                "stage_runtime_adapter_smoke_ready": True,
            },
        }
    )
    return report


def _smoke(**overrides) -> dict:
    report = {
        "schema": check.SMOKE_SCHEMA,
        "ok": True,
        "public_artifact_safe": True,
        "model_repo": check.MODEL_REPO,
        "base_model_id": check.BASE_MODEL_ID,
        "quantization": "AWQ-INT4",
        "stage_id": 4,
        "stage_count": 12,
        "stage_layer_range": [28, 35],
        "tpu_runtime_ready": True,
        "jax_tpu_device_count": 8,
        "jax_shape_smoke_ready": True,
        "glm52_awq_stage_header_ready": True,
        "assigned_weight_key_count": 21675,
        "assigned_weight_file_count": 8,
        "header_file_count": 8,
        "present_stage_key_count": 21675,
        "missing_stage_key_count": 0,
        "dtype_counts": {"BF16": 5477, "I32": 10794, "I64": 5397},
        "stage_family_hits": {"awq_quantized_tensors": True, "attention": True, "mlp_or_moe": True},
        "total_selected_tensor_storage_gb": 40.524259,
        "weight_tensor_values_loaded": False,
        "weight_tensor_values_public": False,
        "safetensors_header_payload_public": False,
        "stage_runtime_adapter_verified": False,
        "same_request_route_verified": False,
        "same_request_decode_verified": False,
        "safety": _safety(),
    }
    report.update(overrides)
    return report


def test_queued_watch_valid_as_blocker_but_not_ready() -> None:
    report = _watch()

    assert check.validate_report(report) == []
    assert "stage_smoke_not_ready" in check.validate_report(report, require_ready=True)


def test_verified_watch_valid_when_output_check_passed() -> None:
    report = _verified_watch()

    assert check.validate_report(report, require_ready=True) == []
    assert check._ready(report) is True


def test_completed_tpu_stage_smoke_ready() -> None:
    report = _smoke()

    assert check.validate_report(report, require_ready=True) == []
    assert check._ready(report) is True


def test_accepts_kaggle_tpu_stage_smoke_runtime_field_names() -> None:
    report = _smoke(
        schema=check.KAGGLE_SMOKE_SCHEMA,
        quantization=None,
        tpu_runtime_ready=None,
        jax_shape_smoke_ready=None,
        jax_tpu_shape_smoke_ready=True,
        header_file_count=None,
        stage_family_hits=None,
        diagnosis_codes=["glm52_awq_stage_header_ready_on_kaggle_tpu"],
        safetensors_header_payload_public=None,
    )

    assert check.validate_report(report, require_ready=True) == []
    assert check._ready(report) is True


def test_checker_rejects_same_request_overclaim() -> None:
    report = _smoke(same_request_decode_verified=True)

    errors = check.validate_report(report, require_ready=True)

    assert "same_request_decode_overclaim" in errors


def test_checker_rejects_public_weight_values() -> None:
    report = _smoke(weight_tensor_values_public=True)

    errors = check.validate_report(report, require_ready=True)

    assert "weight_tensor_values_public_unsafe" in errors
