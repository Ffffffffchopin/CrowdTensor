from __future__ import annotations

import tempfile
from pathlib import Path

from scripts import glm52_awq_tpu_stage_smoke_check as smoke_check
from scripts import glm52_kaggle_tpu_stage_smoke_watch as watch


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_stage_smoke_watch_"))


def _smoke_report() -> dict:
    return {
        "schema": smoke_check.SMOKE_SCHEMA,
        "ok": True,
        "public_artifact_safe": True,
        "model_repo": smoke_check.MODEL_REPO,
        "base_model_id": smoke_check.BASE_MODEL_ID,
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
        "weight_tensor_values_loaded": False,
        "weight_tensor_values_public": False,
        "safetensors_header_payload_public": False,
        "same_request_route_verified": False,
        "same_request_decode_verified": False,
        "safety": {
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
        },
    }


def test_watch_records_queued_status_without_claiming_ready(monkeypatch) -> None:
    base = _tmp_dir()

    def fake_run(command, *, env, timeout):
        assert command[:3] == ["kaggle", "kernels", "status"]
        return {
            "ok": True,
            "returncode": 0,
            "duration_seconds": 0.01,
            "command_public": command,
            "output_tail": f'{watch.DEFAULT_REF} has status "KernelWorkerStatus.QUEUED"',
        }

    monkeypatch.setattr(watch, "run_command", fake_run)
    report = watch.build_report(
        watch.parse_args([
            "--output-dir",
            str(base),
            "--token-section",
            "",
            "--status-polls",
            "1",
        ])
    )

    assert report["last_status"] == "KernelWorkerStatus.QUEUED"
    assert report["stage_runtime_adapter_smoke_ready"] is False
    assert "glm52_awq_tpu_stage_smoke_scheduler_queued" in report["blockers"]
    assert smoke_check.validate_report(report) == []


def test_watch_downloads_and_validates_completed_output(monkeypatch) -> None:
    base = _tmp_dir()
    watch_report = base / "watch.json"
    watch.write_json(
        watch_report,
        {
            "schema": watch.SCHEMA,
            "observations": [{"attempt": 1, "status": "KernelWorkerStatus.QUEUED", "ok": True}],
            "last_status": "KernelWorkerStatus.QUEUED",
            "blockers": [
                "glm52_awq_tpu_stage_smoke_not_ready",
                "glm52_awq_tpu_stage_smoke_scheduler_queued",
            ],
            "public_artifact_safe": True,
        },
    )

    def fake_run(command, *, env, timeout):
        if command[:3] == ["kaggle", "kernels", "status"]:
            return {
                "ok": True,
                "returncode": 0,
                "duration_seconds": 0.01,
                "command_public": command,
                "output_tail": f'{watch.DEFAULT_REF} has status "KernelWorkerStatus.COMPLETE"',
            }
        assert command[:3] == ["kaggle", "kernels", "output"]
        output_dir = Path(command[command.index("-p") + 1])
        watch.write_json(output_dir / watch.OUTPUT_REPORT_NAME, _smoke_report())
        return {
            "ok": True,
            "returncode": 0,
            "duration_seconds": 0.01,
            "command_public": command,
            "output_tail": "downloaded",
        }

    monkeypatch.setattr(watch, "run_command", fake_run)
    report = watch.build_report(
        watch.parse_args([
            "--output-dir",
            str(base),
            "--watch-report",
            str(watch_report),
            "--token-section",
            "",
            "--status-polls",
            "1",
        ])
    )

    assert report["last_status"] == "KernelWorkerStatus.COMPLETE"
    assert report["notebook_output_verified"] is True
    assert report["stage_runtime_adapter_smoke_ready"] is True
    assert report["stage_smoke_check"]["ok"] is True
    assert report["stage_smoke_summary"]["jax_tpu_device_count"] == 8
    assert "glm52_awq_tpu_stage_smoke_not_ready" not in report["blockers"]
    assert "glm52_awq_tpu_stage_smoke_scheduler_queued" not in report["blockers"]
    assert smoke_check.validate_report(report, require_ready=True) == []
