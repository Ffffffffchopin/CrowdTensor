from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts import glm52_kaggle_stage_runtime_check as runtime_check
from scripts import glm52_mcp_tpu_stage_runtime_watch as watch
from scripts import glm52_mcp_tpu_stage_runtime_watch_check as check


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_mcp_tpu_watch_"))


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _stage_report() -> dict:
    return {
        "schema": runtime_check.STAGE_SCHEMA,
        "ok": True,
        "public_artifact_safe": True,
        "model_id": runtime_check.MODEL_ID,
        "compatible_weight_repo": runtime_check.COMPATIBLE_WEIGHT_REPO,
        "provider": "kaggle_jax_tpu",
        "stage_id": 1,
        "stage_layer_range": [26, 52],
        "coordinator_request_id_hash": _hash("a"),
        "stage_execution_verified": True,
        "stage_decode_verified": False,
        "stage_output_hash": _hash("b"),
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


def test_queued_watch_is_public_safe_blocker_not_ready() -> None:
    out = _tmp_dir()
    args = watch.parse_args([
        "--output-dir",
        str(out),
        "--status-polls",
        "1",
        "--token-section",
        "",
    ])

    def fake_runner(command, env, timeout):
        return {
            "ok": True,
            "returncode": 0,
            "duration_seconds": 0.1,
            "command_public": command,
            "output_tail": 'owner/slug has status "KernelWorkerStatus.QUEUED"',
        }

    report = watch.build_report(args, runner=fake_runner)

    assert report["last_status_class"] == "queued"
    assert report["stage_runtime_report_verified"] is False
    assert "glm52_mcp_tpu_stage_runtime_scheduler_queued" in report["blockers"]
    assert check.validate_report(report) == []
    assert "stage_runtime_report_not_verified" in check.validate_report(report, require_ready=True)


def test_complete_watch_downloads_and_checks_stage_runtime_report() -> None:
    out = _tmp_dir()
    args = watch.parse_args([
        "--output-dir",
        str(out),
        "--status-polls",
        "1",
        "--token-section",
        "",
    ])

    def fake_runner(command, env, timeout):
        joined = " ".join(command)
        if "output" in joined:
            output_dir = Path(command[command.index("-p") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / watch.OUTPUT_REPORT_NAME).write_text(
                json.dumps(_stage_report(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return {
                "ok": True,
                "returncode": 0,
                "duration_seconds": 0.1,
                "command_public": command,
                "output_tail": "downloaded",
            }
        return {
            "ok": True,
            "returncode": 0,
            "duration_seconds": 0.1,
            "command_public": command,
            "output_tail": 'owner/slug has status "KernelWorkerStatus.COMPLETE"',
        }

    report = watch.build_report(args, runner=fake_runner)

    assert report["last_status_class"] == "complete"
    assert report["stage_runtime_report_verified"] is True
    assert report["stage_runtime_summary"]["provider"] == "kaggle_jax_tpu"
    assert report["stage_runtime_summary"]["stage_decode_verified"] is False
    assert check.validate_report(report, require_ready=True) == []


def test_checker_rejects_queued_ready_overclaim() -> None:
    report = {
        "schema": watch.SCHEMA,
        "ref": "owner/slug",
        "observations": [{"status": "KernelWorkerStatus.QUEUED"}],
        "last_status": "KernelWorkerStatus.QUEUED",
        "last_status_class": "queued",
        "stage_runtime_report_verified": True,
        "same_request_decode_verified": False,
        "public_artifact_safe": True,
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

    errors = check.validate_report(report)

    assert "queued_or_running_overclaims_ready" in errors
    assert "ready_without_stage_report" in errors
