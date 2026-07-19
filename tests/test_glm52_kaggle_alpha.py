from __future__ import annotations

import json
import io
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from crowdtensor import glm52_kaggle_alpha as alpha
from crowdtensor import cli
from scripts import glm52_kaggle_alpha_check as check
from scripts import glm52_kaggle_alpha_pack as pack
from scripts import glm52_kaggle_alpha_service_smoke_check as smoke_check
from scripts import glm52_kaggle_alpha_service_smoke_probe as smoke_probe
from scripts import glm52_kaggle_same_request_live_probe as live


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_alpha_"))


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _service_report(out: Path, **config_kwargs) -> dict:
    config = alpha.AlphaConfig(output_dir=out, **config_kwargs)
    return alpha.build_service_report(config, host="127.0.0.1", port=8789, run=False)


def _live_report(*, tokens: int = 8, providers: list[str] | None = None, schema: str | None = None) -> dict:
    providers = providers or list(live.REQUIRED_PROVIDERS)
    stage_count = len(providers)
    expected = stage_count * tokens
    return {
        "schema": schema or live.SCHEMA,
        "generated_at": alpha.utc_now(),
        "ok": True,
        "mode": "live",
        "model_id": live.same_request_probe.MODEL_ID,
        "compatible_weight_repo": live.same_request_probe.COMPATIBLE_WEIGHT_REPO,
        "target_generated_token_count": tokens,
        "expected_stage_task_count": expected,
        "coordinator_url_public": False,
        "coordinator_token_public": False,
        "stage_count": stage_count,
        "stage_order": list(range(stage_count)),
        "same_request_decode_verified": True,
        "generated_token_count": tokens,
        "generated_token_hashes": [_hash(str(index % 10)) for index in range(tokens)],
        "accepted_providers": providers,
        "stage_runtime_reports_collected": stage_count,
        "stage_runtime_reports_verified": stage_count,
        "coordinator_stage_reports_collected": expected,
        "worker_stage_decode_reports_collected": stage_count,
        "worker_stage_decode_task_count": expected,
        "full_stage_count_verified": True,
        "runtime_tuning": {
            "full_prefix_prefill_length": 1,
            "full_prefix_dsa_mask_topk": 1,
            "full_prefix_top_k": 1,
        },
        "coordinator_status": {
            "ready": True,
            "generated_token_count": tokens,
            "generated_token_hashes": [_hash(str(index % 10)) for index in range(tokens)],
            "completed_task_count": expected,
            "pending_count": 0,
            "elapsed_seconds": 12.5,
        },
        "cleanup_status": {
            "temporary_kaggle_kernels_deleted": True,
            "temporary_private_packages_removed": True,
            "live_resources_left_running": False,
            "public_artifact_safe": True,
        },
        "same_request_check": {"ok": True, "error_count": 0, "errors": []},
        "blockers": [],
        "safety": live.same_request_probe.safety_flags(),
        "public_artifact_safe": True,
    }


def _gpu_quota_report() -> dict:
    return {
        "schema": "kaggle_gpu_token_weekly_quota_probe_v1",
        "public_artifact_safe": True,
        "private_kernel_payloads_removed": True,
        "summary": {
            "account_count": 2,
            "auth_ok_count": 2,
            "gpu_submission_accepted_count": 0,
            "weekly_gpu_quota_exhausted_count": 2,
        },
        "accounts": [
            {
                "label": "a",
                "owner": "owner-a",
                "auth_ok": True,
                "quota_class": "weekly_gpu_quota_exhausted",
                "push_accepted": False,
                "weekly_gpu_quota_exhausted": True,
                "weekly_gpu_quota_exhausted_by_api": True,
                "gpu_reserved_exceeds_remaining_by_api": True,
                "accelerator_quota": {
                    "quota_refresh_time": "2026-07-11T00:00:00",
                    "gpu_quota": {
                        "effective_remaining_after_reserved_seconds": 0.0,
                    },
                },
            },
            {
                "label": "b",
                "owner": "owner-b",
                "auth_ok": True,
                "quota_class": "weekly_gpu_quota_exhausted",
                "push_accepted": False,
                "weekly_gpu_quota_exhausted": True,
                "weekly_gpu_quota_exhausted_by_api": True,
                "gpu_reserved_exceeds_remaining_by_api": True,
                "accelerator_quota": {
                    "quota_refresh_time": "2026-07-11T00:00:00",
                    "gpu_quota": {
                        "effective_remaining_after_reserved_seconds": 0.0,
                    },
                },
            },
        ],
    }


def _future_gpu_quota_report() -> dict:
    report = _gpu_quota_report()
    for account in report["accounts"]:
        account["accelerator_quota"]["quota_refresh_time"] = "2999-01-01T00:00:00+00:00"
    return report


def test_alpha_http_service_exposes_health_status_and_routes_generate() -> None:
    out = _tmp_dir()
    config = alpha.AlphaConfig(output_dir=out)
    calls: list[dict] = []

    def fake_generate(_config: alpha.AlphaConfig, payload: dict) -> dict:
        calls.append(payload)
        return {
            "schema": alpha.GENERATE_SCHEMA,
            "ok": True,
            "request_id_hash": _hash("r"),
            "prompt_hash": alpha.sha_text(payload.get("prompt") or ""),
            "target_generated_token_count": int(payload.get("max_new_tokens") or 2),
            "generated_token_count": int(payload.get("max_new_tokens") or 2),
            "generated_token_hashes": [_hash("a"), _hash("b")],
            "same_request_decode_verified": True,
            "accepted_providers": list(live.REQUIRED_PROVIDERS),
            "live_report_path": str(out / "live.json"),
            "cleanup_status": {
                "temporary_kaggle_kernels_deleted": True,
                "temporary_private_packages_removed": True,
                "live_resources_left_running": False,
            },
            "public_artifact_safe": True,
        }

    server = alpha.AlphaHTTPServer(host="127.0.0.1", port=0, config=config, generate_fn=fake_generate)
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
            health = json.loads(response.read().decode("utf-8"))
        assert health["ok"] is True
        request = urllib.request.Request(
            f"{base}/generate",
            data=json.dumps({"prompt": "private prompt", "max_new_tokens": 2}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            generated = json.loads(response.read().decode("utf-8"))
        assert generated["ok"] is True
        assert generated["generated_token_count"] == 2
        with urllib.request.urlopen(f"{base}/status", timeout=5) as response:
            status = json.loads(response.read().decode("utf-8"))
        assert status["phase"] == "decode_completed"
        assert status["latest_request"]["prompt_hash"].startswith("sha256:")
        assert calls and calls[0]["prompt"] == "private prompt"
        assert "private prompt" not in json.dumps(status, sort_keys=True)
    finally:
        server.stop()


def test_alpha_service_report_declares_generate_quota_blocker_contract() -> None:
    report = _service_report(_tmp_dir())

    assert report["cli_generate_command_available"] is True
    assert report["cli_generate_artifact_recovery_supported"] is True
    assert report["cli_serve_default_matches_deploy"] is True
    assert report["cli_status_default_matches_deploy"] is True
    assert report["cli_cleanup_default_matches_deploy"] is True
    assert report["generate_uses_current_gpu_quota_blocker"] is True
    assert report["status_loads_existing_alpha_artifacts"] is True
    assert report["status_exposes_resume_private_inputs"] is True
    assert report["cleanup_route_ready"] is True
    assert report["routes"]["cleanup"] == "POST /cleanup"
    assert report["generate_validates_request_schema"] is True
    assert report["requested_model"] == alpha.COMPATIBLE_WEIGHT_REPO
    assert report["model_request_supported"] is True
    assert report["accelerators"] == ["cpu", "gpu", "tpu"]
    assert report["accelerator_request_complete"] is True
    assert report["accelerator_request"]["missing_required"] == []
    assert report["accelerator_request"]["unsupported"] == []
    assert report["hf_token_env_supported"] is True
    resume_private_inputs = report["resume_private_inputs"]
    assert resume_private_inputs["schema"] == alpha.RESUME_PRIVATE_INPUTS_SCHEMA
    assert resume_private_inputs["required_for_live_resume"] is True
    assert resume_private_inputs["resume_command_omits_private_credentials"] is True
    assert resume_private_inputs["kaggle_credentials_required"] is True
    assert resume_private_inputs["kaggle_credential_values_public"] is False
    assert resume_private_inputs["kaggle_token_file_paths_public"] is False
    assert resume_private_inputs["hf_env_values_public"] is False
    assert resume_private_inputs["public_artifact_safe"] is True
    assert report["hf_token_public"] is False
    assert report["hf_token_env_count"] == 2
    assert all(str(item).startswith("sha256:") for item in report["hf_token_env_name_hashes"])
    assert report["kaggle_runtime_blocker_classification_ready"] is True
    assert "kaggle_kernel_wait_timeout" in report["kaggle_runtime_blocker_classes"]
    assert "kaggle_kernel_output_empty_response" in report["kaggle_runtime_blocker_classes"]


def test_alpha_service_report_marks_hf_env_configured_without_public_secret(monkeypatch) -> None:
    out = _tmp_dir()
    monkeypatch.setenv("HF_TOKEN", "hf_alpha_test_secret_value")

    report = _service_report(out, hf_token_env="HF_TOKEN")
    encoded = json.dumps(report, sort_keys=True)

    assert report["hf_token_env_supported"] is True
    assert report["hf_token_env_configured"] is True
    assert report["hf_token_env_configured_count"] == 1
    assert report["hf_token_env_count"] == 1
    assert report["hf_token_public"] is False
    assert "hf_alpha_test_secret_value" not in encoded
    assert "HF_TOKEN" not in encoded
    assert report["public_artifact_safe"] is True


def test_alpha_cli_model_and_accelerator_request_flow_into_service_config() -> None:
    out = _tmp_dir()
    args = cli.parse_args([
        "deploy",
        "glm52-kaggle",
        "--output-dir",
        str(out),
        "--model",
        alpha.COMPATIBLE_WEIGHT_REPO,
        "--accelerators",
        "tpu,cpu,gpu",
        "--hf-token-env",
        "CUSTOM_HF_TOKEN_ENV",
    ])

    cli._validate_glm52_alpha_model_accelerators(args)
    config = cli._glm52_alpha_config_from_args(args)
    report = alpha.build_service_report(config, host="127.0.0.1", port=8789, run=False)

    assert config.requested_model == alpha.COMPATIBLE_WEIGHT_REPO
    assert config.accelerators == ("cpu", "gpu", "tpu")
    assert report["requested_model"] == alpha.COMPATIBLE_WEIGHT_REPO
    assert report["model_request_supported"] is True
    assert report["accelerators"] == ["cpu", "gpu", "tpu"]
    assert report["required_accelerators"] == ["cpu", "gpu", "tpu"]
    assert report["accelerator_request_complete"] is True
    assert config.hf_token_env == "CUSTOM_HF_TOKEN_ENV"
    live_command = cli._glm52_alpha_live_command(args, out / "live")
    assert live_command[live_command.index("--hf-token-env") + 1] == "CUSTOM_HF_TOKEN_ENV"


def test_alpha_cli_rejects_non_glm_model_request() -> None:
    with pytest.raises(SystemExit, match="GLM 5.2 compatible source"):
        cli.parse_args(["deploy", "glm52-kaggle", "--model", "other/model"])


def test_alpha_cli_rejects_missing_or_unsupported_accelerator_request() -> None:
    with pytest.raises(SystemExit, match="missing tpu"):
        cli.parse_args(["deploy", "glm52-kaggle", "--accelerators", "cpu,gpu"])
    with pytest.raises(SystemExit, match="unsupported fpga"):
        cli.parse_args(["deploy", "glm52-kaggle", "--accelerators", "cpu,gpu,tpu,fpga"])


def test_alpha_cli_generate_posts_public_safe_request_without_leaking_prompt_or_url() -> None:
    out = _tmp_dir()
    observed: dict[str, str] = {}
    service_url = "http://127.0.0.1:8789/private-path?token=secret"

    def fake_opener(request, timeout):
        observed["url"] = request.full_url
        observed["body"] = request.data.decode("utf-8")

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "schema": alpha.GENERATE_SCHEMA,
                        "ok": True,
                        "generated_token_count": 8,
                        "generated_token_hashes": [_hash("a") for _ in range(8)],
                        "same_request_decode_verified": True,
                        "accepted_providers": list(live.REQUIRED_PROVIDERS),
                        "public_artifact_safe": True,
                    }
                ).encode("utf-8")

        return Response()

    args = cli.parse_args([
        "generate",
        "--target",
        "glm52-kaggle",
        "--prompt-text",
        "private prompt",
        "--coordinator-url",
        service_url,
        "--output-dir",
        str(out),
        "--max-new-tokens",
        "8",
    ])

    report = cli.build_glm52_kaggle_alpha_generate(args, opener=fake_opener)

    assert report["ok"] is True
    assert report["target"] == "glm52-kaggle"
    assert report["http_status"] == 200
    assert observed["url"] == f"{service_url}/generate"
    assert json.loads(observed["body"])["prompt"] == "private prompt"
    encoded_report = json.dumps(report, sort_keys=True)
    assert "private prompt" not in encoded_report
    assert service_url not in encoded_report
    assert "secret" not in encoded_report
    assert (out / "glm52_kaggle_alpha_generate_cli.json").is_file()


def test_alpha_cli_generate_captures_http_error_body_public_safely() -> None:
    out = _tmp_dir()
    response = {
        "schema": alpha.GENERATE_SCHEMA,
        "ok": False,
        "generated_token_count": 0,
        "generated_token_hashes": [],
        "same_request_decode_verified": False,
        "accepted_providers": [],
        "blockers": ["kaggle_gpu_quota_unavailable"],
        "public_artifact_safe": True,
    }

    def fake_opener(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            {},
            io.BytesIO(json.dumps(response).encode("utf-8")),
        )

    args = cli.parse_args([
        "generate",
        "glm52-kaggle",
        "--prompt-text",
        "private prompt",
        "--output-dir",
        str(out),
    ])

    report = cli.build_glm52_kaggle_alpha_generate(args, opener=fake_opener)

    assert report["ok"] is False
    assert report["http_status"] == 503
    assert "kaggle_gpu_quota_unavailable" in report["diagnosis_codes"]
    assert report["response"]["blockers"] == ["kaggle_gpu_quota_unavailable"]
    assert "private prompt" not in json.dumps(report, sort_keys=True)


def test_alpha_cli_generate_reads_existing_blocker_artifact_when_service_unreachable() -> None:
    out = _tmp_dir()
    service = _write(out / "glm52_kaggle_alpha_service.json", _service_report(out))
    quota = _write(out / "quota.json", _future_gpu_quota_report())
    args = pack.parse_args([
        "--output-dir",
        str(out),
        "--service-report",
        str(service),
        "--gpu-quota-report",
        str(quota),
        "--min-tokens",
        "8",
    ])
    alpha_report = pack.build_report(args)
    _write(out / "glm52_kaggle_alpha.json", alpha_report)

    def fake_opener(_request, timeout):
        raise urllib.error.URLError("connection refused")

    generate_args = cli.parse_args([
        "generate",
        "glm52-kaggle",
        "--prompt-text",
        "private prompt",
        "--output-dir",
        str(out),
    ])

    report = cli.build_glm52_kaggle_alpha_generate(generate_args, opener=fake_opener)

    assert report["ok"] is False
    assert report["http_status"] == 0
    assert report["artifact_recovery"]["present"] is True
    assert report["artifact_recovery"]["alpha_report_present"] is True
    assert report["artifact_recovery"]["phase"] == "blocked_gpu_quota"
    assert report["artifact_recovery"]["resume_private_inputs"]["schema"] == alpha.RESUME_PRIVATE_INPUTS_SCHEMA
    assert report["resume_private_inputs"]["resume_command_omits_private_credentials"] is True
    assert "crowdtensor deploy glm52-kaggle --run-live" in report["next_resume_command"]
    assert "kaggle_gpu_quota_unavailable" in report["diagnosis_codes"]
    assert "URLError" in report["diagnosis_codes"]
    assert "private prompt" not in json.dumps(report, sort_keys=True)


def test_alpha_http_generate_rejects_malformed_json_public_safely() -> None:
    out = _tmp_dir()

    def fail_generate(_config: alpha.AlphaConfig, _payload: dict) -> dict:
        raise AssertionError("generate should not run for malformed request JSON")

    server = alpha.AlphaHTTPServer(
        host="127.0.0.1",
        port=0,
        config=alpha.AlphaConfig(output_dir=out),
        generate_fn=fail_generate,
    )
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        request = urllib.request.Request(
            f"{base}/generate",
            data=b"{not-json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=5)
            raise AssertionError("malformed JSON should return HTTP 400")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            generated = json.loads(exc.read().decode("utf-8"))
        assert generated["ok"] is False
        assert generated["validation_error"] == "malformed_json"
        assert "glm52_alpha_generate_request_malformed_json" in generated["blockers"]
        with urllib.request.urlopen(f"{base}/status", timeout=5) as response:
            status = json.loads(response.read().decode("utf-8"))
        assert status["phase"] == "generate_request_invalid"
        assert status["latest_request"]["validation_error"] == "malformed_json"
        assert "not-json" not in json.dumps(status, sort_keys=True)
    finally:
        server.stop()


def test_alpha_http_generate_rejects_missing_prompt_public_safely() -> None:
    out = _tmp_dir()

    def fail_generate(_config: alpha.AlphaConfig, _payload: dict) -> dict:
        raise AssertionError("generate should not run without prompt")

    server = alpha.AlphaHTTPServer(
        host="127.0.0.1",
        port=0,
        config=alpha.AlphaConfig(output_dir=out),
        generate_fn=fail_generate,
    )
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        request = urllib.request.Request(
            f"{base}/generate",
            data=json.dumps({"max_new_tokens": 2}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=5)
            raise AssertionError("missing prompt should return HTTP 400")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            generated = json.loads(exc.read().decode("utf-8"))
        assert generated["validation_error"] == "prompt_required"
        assert "glm52_alpha_generate_request_prompt_required" in generated["blockers"]
        with urllib.request.urlopen(f"{base}/status", timeout=5) as response:
            status = json.loads(response.read().decode("utf-8"))
        assert status["phase"] == "generate_request_invalid"
        assert status["latest_request"]["validation_error"] == "prompt_required"
        assert status["latest_request"]["prompt_hash"].startswith("sha256:")
    finally:
        server.stop()


def test_alpha_http_status_loads_existing_quota_blocker_before_generate() -> None:
    out = _tmp_dir()
    service = _write(out / "glm52_kaggle_alpha_service.json", _service_report(out))
    quota = _write(out / "quota.json", _future_gpu_quota_report())
    args = pack.parse_args([
        "--output-dir",
        str(out),
        "--service-report",
        str(service),
        "--gpu-quota-report",
        str(quota),
        "--min-tokens",
        "8",
    ])
    report = pack.build_report(args)
    _write(out / "glm52_kaggle_alpha.json", report)

    server = alpha.AlphaHTTPServer(host="127.0.0.1", port=0, config=alpha.AlphaConfig(output_dir=out))
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        with urllib.request.urlopen(f"{base}/status", timeout=5) as response:
            status = json.loads(response.read().decode("utf-8"))
        assert status["phase"] == "blocked_gpu_quota"
        assert status["alpha_report_present"] is True
        assert status["glm52_kaggle_alpha_ready"] is False
        assert status["external_resource_blockers"]["next_quota_refresh_time"] == "2999-01-01T00:00:00+00:00"
        assert status["cleanup_status"]["temporary_kaggle_kernels_deleted"] is True
        assert status["next_resume_command"].startswith("crowdtensor deploy glm52-kaggle --run-live")
        assert status["phase_status"]["overall_state"] == "blocked"
        assert status["resume_private_inputs"]["schema"] == alpha.RESUME_PRIVATE_INPUTS_SCHEMA
        assert status["resume_private_inputs"]["resume_command_omits_private_credentials"] is True
    finally:
        server.stop()


def test_alpha_http_cleanup_route_returns_existing_cleanup_proof() -> None:
    out = _tmp_dir()
    service = _write(out / "glm52_kaggle_alpha_service.json", _service_report(out))
    quota = _write(out / "quota.json", _future_gpu_quota_report())
    args = pack.parse_args([
        "--output-dir",
        str(out),
        "--service-report",
        str(service),
        "--gpu-quota-report",
        str(quota),
        "--min-tokens",
        "8",
    ])
    report = pack.build_report(args)
    _write(out / "glm52_kaggle_alpha.json", report)

    server = alpha.AlphaHTTPServer(host="127.0.0.1", port=0, config=alpha.AlphaConfig(output_dir=out))
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        request = urllib.request.Request(
            f"{base}/cleanup",
            data=json.dumps({}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            cleanup = json.loads(response.read().decode("utf-8"))
        assert cleanup["ok"] is True
        assert cleanup["temporary_kaggle_kernels_deleted"] is True
        assert cleanup["temporary_private_packages_removed"] is True
        assert cleanup["live_resources_left_running"] is False
        assert cleanup["cleanup_status"]["cleanup_mode"] == "gpu_quota_preflight_skipped_live"
        with urllib.request.urlopen(f"{base}/status", timeout=5) as response:
            status = json.loads(response.read().decode("utf-8"))
        assert status["phase"] == "cleanup_completed"
        assert status["cleanup_status"]["temporary_kaggle_kernels_deleted"] is True
    finally:
        server.stop()


def test_generate_with_live_probe_forwards_request_timeout(monkeypatch) -> None:
    out = _tmp_dir()
    config = alpha.AlphaConfig(output_dir=out, wait_seconds=7200.0, coordinator_task_timeout_seconds=7200.0)
    observed: dict[str, float] = {}

    def fake_run_live(args, *, runner):
        observed["wait_seconds"] = float(args.wait_seconds)
        observed["coordinator_task_timeout_seconds"] = float(args.coordinator_task_timeout_seconds)
        observed["max_new_tokens"] = int(args.max_new_tokens)
        return _live_report(tokens=2)

    monkeypatch.setattr(alpha.live_probe, "run_live", fake_run_live)

    response = alpha.generate_with_live_probe(
        config,
        {"prompt": "private prompt", "max_new_tokens": 2, "timeout": 42},
    )

    assert response["ok"] is True
    assert response["timeout_seconds"] == 42.0
    assert observed["wait_seconds"] == 42.0
    assert observed["coordinator_task_timeout_seconds"] == 42.0
    assert observed["max_new_tokens"] == 2
    assert "private prompt" not in json.dumps(response, sort_keys=True)


def test_generate_short_circuits_current_gpu_quota_blocker_without_live_probe(monkeypatch) -> None:
    out = _tmp_dir()
    service = _write(out / "glm52_kaggle_alpha_service.json", _service_report(out))
    quota = _write(out / "quota.json", _future_gpu_quota_report())
    args = pack.parse_args([
        "--output-dir",
        str(out),
        "--service-report",
        str(service),
        "--gpu-quota-report",
        str(quota),
        "--min-tokens",
        "8",
    ])
    report = pack.build_report(args)
    _write(out / "glm52_kaggle_alpha.json", report)

    def fail_run_live(_args, *, runner):
        raise AssertionError("live probe should not run while current GPU quota blocker is valid")

    monkeypatch.setattr(alpha.live_probe, "run_live", fail_run_live)

    response = alpha.generate_with_live_probe(
        alpha.AlphaConfig(output_dir=out),
        {"prompt": "private prompt", "max_new_tokens": 8, "timeout": 30},
    )

    assert response["ok"] is False
    assert response["generated_token_count"] == 0
    assert "kaggle_gpu_quota_unavailable" in response["blockers"]
    assert "glm52_alpha_request_blocked_by_current_gpu_quota_preflight" in response["blockers"]
    assert response["external_resource_blockers"]["next_quota_refresh_time"] == "2999-01-01T00:00:00+00:00"
    assert response["cleanup_status"]["temporary_kaggle_kernels_deleted"] is True
    assert response["next_resume_command_redacts_credentials"] is True
    assert response["resume_private_inputs"]["schema"] == alpha.RESUME_PRIVATE_INPUTS_SCHEMA
    assert response["resume_private_inputs"]["resume_command_omits_private_credentials"] is True
    assert response["phase_status"]["overall_state"] == "blocked"
    assert "private prompt" not in json.dumps(response, sort_keys=True)


def test_alpha_http_service_exposes_quota_blocked_generate_status(monkeypatch) -> None:
    out = _tmp_dir()
    service = _write(out / "glm52_kaggle_alpha_service.json", _service_report(out))
    quota = _write(out / "quota.json", _future_gpu_quota_report())
    args = pack.parse_args([
        "--output-dir",
        str(out),
        "--service-report",
        str(service),
        "--gpu-quota-report",
        str(quota),
        "--min-tokens",
        "8",
    ])
    report = pack.build_report(args)
    _write(out / "glm52_kaggle_alpha.json", report)

    def fail_run_live(_args, *, runner):
        raise AssertionError("live probe should not run while current GPU quota blocker is valid")

    monkeypatch.setattr(alpha.live_probe, "run_live", fail_run_live)
    server = alpha.AlphaHTTPServer(host="127.0.0.1", port=0, config=alpha.AlphaConfig(output_dir=out))
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        request = urllib.request.Request(
            f"{base}/generate",
            data=json.dumps({"prompt": "private prompt", "max_new_tokens": 8}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=5)
            raise AssertionError("quota blocked generate should return HTTP 503")
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
            generated = json.loads(exc.read().decode("utf-8"))
        assert "kaggle_gpu_quota_unavailable" in generated["blockers"]
        with urllib.request.urlopen(f"{base}/status", timeout=5) as response:
            status = json.loads(response.read().decode("utf-8"))
        assert status["phase"] == "decode_blocked"
        assert "kaggle_gpu_quota_unavailable" in status["blockers"]
        assert status["external_resource_blockers"]["next_quota_refresh_time"] == "2999-01-01T00:00:00+00:00"
        assert status["cleanup_status"]["temporary_private_packages_removed"] is True
        assert status["next_resume_command"].startswith("crowdtensor deploy glm52-kaggle --run-live")
        assert generated["resume_private_inputs"]["schema"] == alpha.RESUME_PRIVATE_INPUTS_SCHEMA
        assert generated["resume_private_inputs"]["resume_command_omits_private_credentials"] is True
        assert status["resume_private_inputs"]["schema"] == alpha.RESUME_PRIVATE_INPUTS_SCHEMA
        assert status["resume_private_inputs"]["resume_command_omits_private_credentials"] is True
        assert status["phase_status"]["overall_state"] == "blocked"
        assert "private prompt" not in json.dumps(status, sort_keys=True)
    finally:
        server.stop()


def test_alpha_service_smoke_probe_verifies_http_routes_against_quota_blocker() -> None:
    out = _tmp_dir()
    service = _write(out / "glm52_kaggle_alpha_service.json", _service_report(out))
    quota = _write(out / "quota.json", _future_gpu_quota_report())
    args = pack.parse_args([
        "--output-dir",
        str(out),
        "--service-report",
        str(service),
        "--gpu-quota-report",
        str(quota),
        "--min-tokens",
        "8",
    ])
    alpha_report = pack.build_report(args)
    _write(out / "glm52_kaggle_alpha.json", alpha_report)

    smoke_args = smoke_probe.parse_args([
        "--output-dir",
        str(out / "smoke"),
        "--alpha-output-dir",
        str(out),
        "--max-new-tokens",
        "8",
    ])
    report = smoke_probe.build_report(smoke_args)
    encoded = json.dumps(report, sort_keys=True)

    assert report["service_http_smoke_verified"] is True
    assert report["health"]["ok"] is True
    assert report["status"]["phase"] == "blocked_gpu_quota"
    assert report["status"]["resume_private_inputs_verified"] is True
    assert report["generate"]["http_status"] == 503
    assert report["generate"]["quota_blocker_verified"] is True
    assert report["generate"]["resume_private_inputs_verified"] is True
    assert report["status_resume_private_inputs_verified"] is True
    assert report["generate_resume_private_inputs_verified"] is True
    assert report["cleanup_route_verified"] is True
    assert report["cleanup"]["http_status"] == 200
    assert report["cleanup"]["temporary_kaggle_kernels_deleted"] is True
    assert report["cleanup"]["temporary_private_packages_removed"] is True
    assert report["cleanup"]["live_resources_left_running"] is False
    assert "kaggle_gpu_quota_unavailable" in report["generate"]["blockers"]
    assert report["completion_boundary"]["service_smoke_is_not_live_success"] is True
    assert "CrowdTensor GLM 5.2 Alpha service smoke" not in encoded
    assert smoke_check.validate_report(report, require_verified=True) == []


def test_alpha_pack_accepts_multitoken_service_live_cleanup_evidence() -> None:
    out = _tmp_dir()
    service = _write(out / "service.json", _service_report(out))
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is True
    assert report["generated_token_count"] == 8
    assert report["cleanup_verified"] is True
    assert report["runtime_tuning"]["full_prefix_prefill_length"] == 1
    assert report["benchmark"]["runtime_tuning"]["full_prefix_top_k"] == 1
    assert check.validate_report(report, require_ready=True) == []


def test_alpha_pack_imports_service_smoke_summary() -> None:
    out = _tmp_dir()
    service = _write(out / "service.json", _service_report(out))
    smoke_report = _write(
        out / "smoke.json",
        {
            "schema": smoke_probe.SCHEMA,
            "ok": True,
            "service_http_smoke_verified": True,
            "public_artifact_safe": True,
            "health": {"http_status": 200, "ok": True, "public_artifact_safe": True},
            "status": {
                "http_status": 200,
                "phase": "blocked_gpu_quota",
                "resume_private_inputs_verified": True,
                "public_artifact_safe": True,
            },
            "generate": {
                "http_status": 503,
                "attempted": True,
                "quota_blocker_verified": True,
                "resume_private_inputs_verified": True,
                "successful_generate_verified": False,
                "same_request_decode_verified": False,
                "target_generated_token_count": 8,
                "generated_token_count": 0,
                "accepted_providers": [],
                "blockers": ["kaggle_gpu_quota_unavailable"],
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "public_artifact_safe": True,
            },
            "generate_route_reaches_service": True,
            "generate_route_quota_blocker_verified": True,
            "generate_route_success_verified": False,
            "status_resume_private_inputs_verified": True,
            "generate_resume_private_inputs_verified": True,
            "cleanup": {
                "http_status": 200,
                "ok": True,
                "temporary_kaggle_kernels_deleted": True,
                "temporary_private_packages_removed": True,
                "live_resources_left_running": False,
                "public_artifact_safe": True,
            },
            "cleanup_route_verified": True,
            "completion_boundary": {
                "service_smoke_is_not_live_success": True,
                "quota_blocker_generate_is_not_multitoken_success": True,
                "strict_alpha_ready_still_requires_live_report": True,
            },
        },
    )
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--service-smoke-report",
        str(smoke_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["service_smoke_summary"]["present"] is True
    assert report["service_smoke_summary"]["service_http_smoke_verified"] is True
    assert report["service_smoke_summary"]["generate_route_quota_blocker_verified"] is True
    assert report["service_smoke_summary"]["status_resume_private_inputs_verified"] is True
    assert report["service_smoke_summary"]["generate_resume_private_inputs_verified"] is True
    assert report["service_smoke_summary"]["cleanup_route_verified"] is True
    assert report["service_smoke_summary"]["cleanup_temporary_kaggle_kernels_deleted"] is True
    assert check.validate_report(report) == []


def test_alpha_check_rejects_invalid_imported_service_smoke_summary() -> None:
    out = _tmp_dir()
    service = _write(out / "service.json", _service_report(out))
    smoke_report = _write(
        out / "bad-smoke.json",
        {
            "schema": smoke_probe.SCHEMA,
            "ok": False,
            "service_http_smoke_verified": False,
            "public_artifact_safe": True,
            "health": {"http_status": 200, "ok": True, "public_artifact_safe": True},
            "status": {"http_status": 200, "phase": "blocked_gpu_quota", "public_artifact_safe": True},
            "generate": {
                "http_status": 0,
                "attempted": False,
                "quota_blocker_verified": False,
                "successful_generate_verified": False,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "public_artifact_safe": True,
            },
            "generate_route_reaches_service": False,
            "cleanup": {
                "http_status": 0,
                "ok": False,
                "temporary_kaggle_kernels_deleted": False,
                "temporary_private_packages_removed": False,
                "live_resources_left_running": True,
                "public_artifact_safe": True,
            },
            "cleanup_route_verified": False,
            "completion_boundary": {
                "service_smoke_is_not_live_success": True,
                "quota_blocker_generate_is_not_multitoken_success": True,
                "strict_alpha_ready_still_requires_live_report": True,
            },
        },
    )
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--service-smoke-report",
        str(smoke_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)
    errors = check.validate_report(report)

    assert "service_smoke_check_failed" in errors
    assert "service_http_smoke_not_verified" in errors
    assert "service_smoke_generate_route_not_reached" in errors
    assert "service_smoke_status_resume_private_inputs_missing" in errors
    assert "service_smoke_cleanup_route_not_verified" in errors


def test_alpha_pack_imports_generate_cli_artifact_recovery_summary() -> None:
    out = _tmp_dir()
    service = _write(out / "service.json", _service_report(out))
    generate_cli_report = _write(
        out / "generate-cli.json",
        {
            "schema": "glm52_kaggle_alpha_cli_v1",
            "command": "generate",
            "target": "glm52-kaggle",
            "ok": False,
            "http_status": 0,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "response": {},
            "artifact_recovery": {
                "present": True,
                "phase": "blocked_gpu_quota",
                "blockers": ["kaggle_gpu_quota_unavailable"],
                "next_resume_command": "crowdtensor deploy glm52-kaggle --run-live",
                "resume_private_inputs": {
                    "schema": alpha.RESUME_PRIVATE_INPUTS_SCHEMA,
                    "resume_command_omits_private_credentials": True,
                    "public_artifact_safe": True,
                },
            },
            "next_resume_command": "crowdtensor deploy glm52-kaggle --run-live",
            "resume_private_inputs": {
                "schema": alpha.RESUME_PRIVATE_INPUTS_SCHEMA,
                "resume_command_omits_private_credentials": True,
                "public_artifact_safe": True,
            },
            "diagnosis_codes": ["URLError", "kaggle_gpu_quota_unavailable"],
            "public_artifact_safe": True,
        },
    )
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--generate-cli-report",
        str(generate_cli_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["generate_cli_summary"]["present"] is True
    assert report["generate_cli_summary"]["generate_cli_check_ok"] is True
    assert report["generate_cli_summary"]["artifact_recovery_present"] is True
    assert report["generate_cli_summary"]["artifact_recovery_resume_private_inputs_verified"] is True
    assert check.validate_report(report) == []


def test_alpha_check_rejects_invalid_imported_generate_cli_summary() -> None:
    out = _tmp_dir()
    service = _write(out / "service.json", _service_report(out))
    generate_cli_report = _write(
        out / "bad-generate-cli.json",
        {
            "schema": "glm52_kaggle_alpha_cli_v1",
            "command": "generate",
            "target": "glm52-kaggle",
            "ok": False,
            "http_status": 0,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "response": {},
            "artifact_recovery": {
                "present": True,
                "phase": "blocked_gpu_quota",
                "blockers": ["kaggle_gpu_quota_unavailable"],
            },
            "diagnosis_codes": ["URLError"],
            "public_artifact_safe": True,
        },
    )
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--generate-cli-report",
        str(generate_cli_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)
    errors = check.validate_report(report)

    assert "generate_cli_check_failed" in errors
    assert "generate_cli_resume_command_missing" in errors
    assert "generate_cli_resume_private_inputs_missing" in errors


def test_alpha_pack_main_writes_benchmark_artifact() -> None:
    out = _tmp_dir()
    service = _write(out / "service.json", _service_report(out))
    live_report = _write(out / "live.json", _live_report(tokens=8))

    rc = pack.main([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    assert rc == 0
    summary = json.loads((out / "alpha" / "glm52_kaggle_alpha.json").read_text(encoding="utf-8"))
    benchmark = json.loads((out / "alpha" / "glm52_kaggle_alpha_benchmark.json").read_text(encoding="utf-8"))
    assert summary["artifacts"]["benchmark_json"]["present"] is True
    assert summary["artifacts"]["benchmark_json"]["schema"] == pack.BENCHMARK_SCHEMA
    assert benchmark["schema"] == pack.BENCHMARK_SCHEMA
    assert benchmark["deploy_time_seconds"] == 12.5
    assert benchmark["stage_count"] == 3
    assert benchmark["provider_coverage"] == list(live.REQUIRED_PROVIDERS)
    assert benchmark["tokens_generated"] == 8
    assert benchmark["cleanup_status"]["temporary_kaggle_kernels_deleted"] is True
    assert benchmark["public_artifact_safe"] is True


def test_alpha_pack_requires_generate_timeout_contract_for_ready() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload.pop("generate_request_fields", None)
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_generate_accepts_timeout_missing" in report["blockers"]
    assert "generate_accepts_timeout_missing" in check.validate_report(report, require_ready=True)


def test_alpha_pack_requires_generate_current_quota_blocker_contract_for_ready() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload.pop("generate_uses_current_gpu_quota_blocker", None)
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_generate_current_gpu_quota_blocker_missing" in report["blockers"]
    assert "generate_current_gpu_quota_blocker_missing" in check.validate_report(report, require_ready=True)


def test_alpha_pack_requires_status_existing_artifact_load_contract_for_ready() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload.pop("status_loads_existing_alpha_artifacts", None)
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_status_existing_artifact_load_missing" in report["blockers"]
    assert "status_existing_artifact_load_missing" in check.validate_report(report, require_ready=True)


def test_alpha_pack_requires_status_resume_private_inputs_contract_for_ready() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload.pop("status_exposes_resume_private_inputs", None)
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_status_resume_private_inputs_missing" in report["blockers"]
    assert "status_resume_private_inputs_missing" in check.validate_report(report, require_ready=True)


def test_alpha_pack_requires_http_cleanup_route_contract_for_ready() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload["routes"].pop("cleanup", None)
    service_payload.pop("cleanup_route_ready", None)
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_cleanup_route_ready_missing" in report["blockers"]
    assert "cleanup_route_ready_missing" in check.validate_report(report, require_ready=True)


def test_alpha_pack_requires_cli_generate_command_contract_for_ready() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload.pop("cli_generate_command_available", None)
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_cli_generate_command_missing" in report["blockers"]
    assert "cli_generate_command_missing" in check.validate_report(report, require_ready=True)


def test_alpha_pack_requires_cli_generate_artifact_recovery_contract_for_ready() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload.pop("cli_generate_artifact_recovery_supported", None)
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_cli_generate_artifact_recovery_missing" in report["blockers"]
    assert "cli_generate_artifact_recovery_missing" in check.validate_report(report, require_ready=True)


def test_alpha_pack_requires_cli_default_output_dir_contract_for_ready() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload.pop("cli_cleanup_default_matches_deploy", None)
    service_payload.pop("cli_serve_default_matches_deploy", None)
    service_payload.pop("cli_status_default_matches_deploy", None)
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_cli_cleanup_default_mismatch" in report["blockers"]
    assert "glm52_alpha_cli_serve_default_mismatch" in report["blockers"]
    assert "glm52_alpha_cli_status_default_mismatch" in report["blockers"]
    errors = check.validate_report(report, require_ready=True)
    assert "cli_cleanup_default_mismatch" in errors
    assert "cli_serve_default_mismatch" in errors
    assert "cli_status_default_mismatch" in errors


def test_alpha_pack_requires_resume_private_inputs_contract_for_ready() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload.pop("resume_private_inputs", None)
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_resume_private_inputs_missing" in report["blockers"]
    errors = check.validate_report(report, require_ready=True)
    assert "resume_private_inputs_schema_mismatch" in errors


def test_alpha_check_rejects_public_resume_private_inputs() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload["resume_private_inputs"]["kaggle_credential_values_public"] = True
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    errors = check.validate_report(report, require_ready=True)
    assert "resume_private_inputs_kaggle_values_public" in errors


def test_alpha_pack_requires_generate_request_validation_contract_for_ready() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload.pop("generate_validates_request_schema", None)
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_generate_request_validation_missing" in report["blockers"]
    assert "generate_request_validation_missing" in check.validate_report(report, require_ready=True)


def test_alpha_pack_requires_kaggle_runtime_blocker_classification_for_ready() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload.pop("kaggle_runtime_blocker_classification_ready", None)
    service_payload.pop("kaggle_runtime_blocker_classes", None)
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_kaggle_runtime_blocker_classification_missing" in report["blockers"]
    assert "kaggle_runtime_blocker_classification_missing" in check.validate_report(report, require_ready=True)


def test_alpha_pack_requires_hf_token_env_contract_for_ready() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload.pop("hf_token_env_supported", None)
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_hf_token_env_contract_missing" in report["blockers"]
    assert "hf_token_env_contract_missing" in check.validate_report(report, require_ready=True)


def test_alpha_pack_rejects_public_hf_token_signal() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload["hf_token_public"] = True
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_hf_token_public" in report["blockers"]
    assert "hf_token_public" in check.validate_report(report, require_ready=True)


def test_alpha_pack_requires_supported_model_request_for_ready() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload["requested_model"] = "other/model"
    service_payload["model_request_supported"] = False
    service_payload["service_api_ready"] = False
    service_payload["ok"] = False
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_model_request_not_supported" in report["blockers"]
    assert "model_request_not_supported" in check.validate_report(report, require_ready=True)


def test_alpha_pack_requires_complete_accelerator_request_for_ready() -> None:
    out = _tmp_dir()
    service_payload = _service_report(out)
    service_payload["accelerators"] = ["cpu", "gpu"]
    service_payload["accelerator_request_complete"] = False
    service_payload["accelerator_request"] = {
        "requested": ["cpu", "gpu"],
        "required": ["cpu", "gpu", "tpu"],
        "missing_required": ["tpu"],
        "unsupported": [],
        "all_required_present": False,
        "supported": True,
        "complete": False,
    }
    service_payload["service_api_ready"] = False
    service_payload["ok"] = False
    service = _write(out / "service.json", service_payload)
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_accelerator_request_incomplete" in report["blockers"]
    errors = check.validate_report(report, require_ready=True)
    assert "accelerator_request_incomplete" in errors
    assert "accelerator_missing:tpu" in errors


def test_alpha_pack_writes_checker_passing_blocker_without_live_report() -> None:
    out = _tmp_dir()
    service = _write(out / "service.json", _service_report(out))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_multitoken_live_not_verified" in report["blockers"]
    assert check.validate_report(report) == []
    assert "glm52_kaggle_alpha_not_ready" in check.validate_report(report, require_ready=True)


def test_alpha_pack_blocker_uses_service_runtime_tuning_resume_command() -> None:
    out = _tmp_dir()
    package_report = "dist/glm52-kaggle-stage-worker-package-20260707-alpha-r8/example.json"
    service = _write(
        out / "service.json",
        _service_report(
            out,
            stage_worker_package_report=package_report,
            stage_push_parallelism=7,
            full_prefix_prefill_length=1,
            full_prefix_dsa_mask_topk=1,
            full_prefix_top_k=1,
            cpu_group_stage_attempt_seconds=2.5,
        ),
    )
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    resume = report["blocker_report"]["next_resume_command"]
    assert report["next_resume_command"] == resume
    assert report["next_resume_command_redacts_credentials"] is True
    assert report["runtime_tuning"]["full_prefix_prefill_length"] == 1
    assert report["benchmark"]["runtime_tuning"]["full_prefix_top_k"] == 1
    assert report["blocker_report"]["next_resume_command_redacts_credentials"] is True
    assert package_report in resume
    assert "--run-live" in resume
    assert "--gpu-quota-preflight" in resume
    assert "--output-dir" in resume
    assert "--model cyankiwi/GLM-5.2-AWQ-INT4" in resume
    assert "--accelerators cpu,gpu,tpu" in resume
    assert str(out / "alpha") in resume
    assert "--stage-push-parallelism 7" in resume
    assert "--full-prefix-prefill-length 1" in resume
    assert "--cpu-group-stage-attempt-seconds 2.5" in resume
    assert check.validate_report(report) == []


def test_alpha_check_rejects_blocker_without_top_level_resume_command() -> None:
    out = _tmp_dir()
    service = _write(out / "service.json", _service_report(out))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--min-tokens",
        "8",
    ])
    report = pack.build_report(args)

    report.pop("next_resume_command", None)
    report.pop("next_resume_command_redacts_credentials", None)

    errors = check.validate_report(report)
    assert "next_resume_command_missing" in errors
    assert "next_resume_command_redaction_missing" in errors


def test_alpha_pack_imports_gpu_quota_reports_as_external_blocker() -> None:
    out = _tmp_dir()
    service = _write(out / "service.json", _service_report(out))
    quota = _write(out / "quota.json", _gpu_quota_report())
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--gpu-quota-report",
        str(quota),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "kaggle_gpu_quota_unavailable" in report["blockers"]
    assert "glm52_alpha_cleanup_not_verified" not in report["blockers"]
    assert report["cleanup_verified"] is True
    assert report["gpu_quota_summary"]["all_auth_ok_accounts_gpu_quota_exhausted"] is True
    assert report["gpu_quota_summary"]["cleanup_verified"] is True
    assert report["gpu_quota_summary"]["next_quota_refresh_time"] == "2026-07-11T00:00:00"
    assert report["blocker_report"]["external_resource_blockers"]["kaggle_gpu_quota_unavailable"] is True
    phase_status = report["phase_status"]
    assert phase_status["schema"] == pack.PHASE_STATUS_SCHEMA
    assert phase_status["overall_state"] == "blocked"
    assert "gpu_quota_preflight" in phase_status["blocked_phase_names"]
    assert "kernel_push" in phase_status["blocked_phase_names"]
    assert "cleanup_completed" in phase_status["completed_phase_names"]
    assert check.validate_report(report) == []


def test_alpha_check_rejects_missing_phase_status() -> None:
    out = _tmp_dir()
    service = _write(out / "service.json", _service_report(out))
    live_report = _write(out / "live.json", _live_report(tokens=8))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])
    report = pack.build_report(args)
    report.pop("phase_status", None)

    errors = check.validate_report(report)

    assert "phase_status_schema_mismatch" in errors
    assert "phase_status_missing:configuration_check" in errors


def test_alpha_config_forwards_runtime_tuning_to_live_args() -> None:
    out = _tmp_dir()
    config = alpha.AlphaConfig(
        output_dir=out,
        full_prefix_prefill_length=1,
        full_prefix_dsa_mask_topk=1,
        full_prefix_executed_expert_count=2,
        full_prefix_top_k=1,
        full_prefix_row_block_size=512,
        full_prefix_max_tensor_bytes=33554432,
        full_prefix_max_block_bytes=16777216,
        cpu_group_stage_attempt_seconds=2.5,
        cpu_group_stage_poll_seconds=0.5,
    )

    args = alpha.build_live_args(config, request_output_dir=out / "request", max_new_tokens=8)

    assert args.full_prefix_prefill_length == 1
    assert args.full_prefix_dsa_mask_topk == 1
    assert args.full_prefix_executed_expert_count == 2
    assert args.full_prefix_top_k == 1
    assert args.full_prefix_row_block_size == 512
    assert args.full_prefix_max_tensor_bytes == 33554432
    assert args.full_prefix_max_block_bytes == 16777216
    assert args.cpu_group_stage_attempt_seconds == 2.5
    assert args.cpu_group_stage_poll_seconds == 0.5


def test_alpha_deploy_live_command_forwards_runtime_tuning() -> None:
    out = _tmp_dir()
    args = cli.parse_args(
        [
            "deploy",
            "glm52-kaggle",
            "--output-dir",
            str(out),
            "--full-prefix-prefill-length",
            "1",
            "--full-prefix-dsa-mask-topk",
            "1",
            "--full-prefix-executed-expert-count",
            "2",
            "--full-prefix-top-k",
            "1",
            "--full-prefix-row-block-size",
            "512",
            "--full-prefix-max-tensor-bytes",
            "33554432",
            "--full-prefix-max-block-bytes",
            "16777216",
            "--cpu-group-stage-attempt-seconds",
            "2.5",
            "--cpu-group-stage-poll-seconds",
            "0.5",
        ]
    )

    command = cli._glm52_alpha_live_command(args, out / "live")

    assert command[command.index("--full-prefix-prefill-length") + 1] == "1"
    assert command[command.index("--full-prefix-dsa-mask-topk") + 1] == "1"
    assert command[command.index("--full-prefix-executed-expert-count") + 1] == "2"
    assert command[command.index("--full-prefix-top-k") + 1] == "1"
    assert command[command.index("--full-prefix-row-block-size") + 1] == "512"
    assert command[command.index("--full-prefix-max-tensor-bytes") + 1] == "33554432"
    assert command[command.index("--full-prefix-max-block-bytes") + 1] == "16777216"
    assert command[command.index("--cpu-group-stage-attempt-seconds") + 1] == "2.5"
    assert command[command.index("--cpu-group-stage-poll-seconds") + 1] == "0.5"


def test_alpha_deploy_passes_gpu_quota_reports_to_pack() -> None:
    out = _tmp_dir()
    quota_report = _write(out / "quota.json", _gpu_quota_report())
    observed_commands: list[list[str]] = []

    def fake_runner(command, **kwargs):
        observed_commands.append(command)
        payload = {
            "schema": pack.SCHEMA,
            "ok": False,
            "glm52_kaggle_alpha_ready": False,
            "model_id": alpha.MODEL_ID,
            "compatible_weight_repo": alpha.COMPATIBLE_WEIGHT_REPO,
            "service_api_ready": True,
            "generate_routes_to_same_request_live_probe": True,
            "generated_token_count": 0,
            "accepted_providers": [],
            "cleanup_verified": False,
            "blockers": ["kaggle_gpu_quota_unavailable"],
            "public_artifact_safe": True,
        }
        return subprocess.CompletedProcess(command, 2, stdout=json.dumps(payload), stderr="")

    args = cli.parse_args([
        "deploy",
        "glm52-kaggle",
        "--output-dir",
        str(out / "deploy"),
        "--gpu-quota-report",
        str(quota_report),
    ])

    cli.build_glm52_kaggle_alpha_deploy(args, runner=fake_runner)

    pack_command = observed_commands[0]
    assert "--gpu-quota-report" in pack_command
    assert pack_command[pack_command.index("--gpu-quota-report") + 1] == str(quota_report)


def test_alpha_gpu_quota_preflight_command_uses_cuda_raw_token_map() -> None:
    out = _tmp_dir()
    args = cli.parse_args([
        "deploy",
        "glm52-kaggle",
        "--output-dir",
        str(out),
        "--provider-raw-token-file-map",
        "kaggle_cuda=~/.config/crowdtensor/kaggle-gpu-token.md",
        "--provider-raw-token-username-map",
        "kaggle_cuda=gpuowner",
        "--gpu-quota-preflight",
    ])

    command = cli._glm52_alpha_gpu_quota_preflight_command(args, out / "quota")

    assert command[command.index("--raw-token-file") + 1] == "~/.config/crowdtensor/kaggle-gpu-token.md"
    assert command[command.index("--raw-token-username") + 1] == "gpuowner"


def test_alpha_deploy_preflight_skips_live_when_all_gpu_quota_exhausted() -> None:
    out = _tmp_dir()
    observed_commands: list[list[str]] = []

    def fake_runner(command, **kwargs):
        observed_commands.append(command)
        joined = " ".join(command)
        if "kaggle_gpu_token_weekly_quota_probe.py" in joined:
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(_gpu_quota_report()), stderr="")
        payload = {
            "schema": pack.SCHEMA,
            "ok": False,
            "glm52_kaggle_alpha_ready": False,
            "model_id": alpha.MODEL_ID,
            "compatible_weight_repo": alpha.COMPATIBLE_WEIGHT_REPO,
            "service_api_ready": True,
            "generate_routes_to_same_request_live_probe": True,
            "generated_token_count": 0,
            "accepted_providers": [],
            "cleanup_verified": False,
            "blockers": ["kaggle_gpu_quota_unavailable"],
            "public_artifact_safe": True,
        }
        return subprocess.CompletedProcess(command, 2, stdout=json.dumps(payload), stderr="")

    args = cli.parse_args([
        "deploy",
        "glm52-kaggle",
        "--output-dir",
        str(out / "deploy"),
        "--run-live",
        "--gpu-quota-preflight",
    ])

    summary = cli.build_glm52_kaggle_alpha_deploy(args, runner=fake_runner)

    assert summary["gpu_quota_preflight_performed"] is True
    assert summary["live_skipped_by_gpu_quota_preflight"] is True
    assert any("kaggle_gpu_token_weekly_quota_probe.py" in " ".join(command) for command in observed_commands)
    assert not any("glm52_kaggle_same_request_live_probe.py" in " ".join(command) for command in observed_commands)
    pack_command = observed_commands[-1]
    assert "--gpu-quota-report" in pack_command
    assert summary["steps"][1]["skipped"] is True


def test_alpha_status_reads_deploy_blocker_artifact_and_quota_preflight() -> None:
    out = _tmp_dir()
    deploy_dir = out / "deploy"
    service = _write(deploy_dir / "glm52_kaggle_alpha_service.json", _service_report(deploy_dir))
    quota = _write(deploy_dir / "gpu-quota-preflight" / "kaggle_gpu_token_weekly_quota_probe.json", _gpu_quota_report())
    args = pack.parse_args([
        "--output-dir",
        str(deploy_dir),
        "--service-report",
        str(service),
        "--gpu-quota-report",
        str(quota),
        "--min-tokens",
        "8",
    ])
    alpha_report = pack.build_report(args)
    _write(deploy_dir / "glm52_kaggle_alpha.json", alpha_report)
    _write(deploy_dir / "glm52_kaggle_alpha_cli_summary.json", {"live_skipped_by_gpu_quota_preflight": True})

    status_args = cli.parse_args(["status", "glm52-kaggle", "--output-dir", str(deploy_dir)])
    status = cli.build_glm52_kaggle_alpha_status(status_args)

    assert status["ok"] is True
    assert status["status_report_present"] is False
    assert status["alpha_report_present"] is True
    assert status["gpu_quota_preflight_report_present"] is True
    assert status["phase"] == "blocked_gpu_quota"
    assert status["glm52_kaggle_alpha_ready"] is False
    assert status["cleanup_verified"] is True
    assert status["live_skipped_by_gpu_quota_preflight"] is True
    assert status["gpu_quota_status"]["all_auth_ok_accounts_gpu_quota_exhausted"] is True
    assert status["gpu_quota_status"]["next_quota_refresh_time"] == "2026-07-11T00:00:00"
    assert "kaggle_gpu_quota_unavailable" in status["blockers"]
    assert "crowdtensor deploy glm52-kaggle --run-live" in status["next_resume_command"]
    assert status["resume_private_inputs"]["schema"] == alpha.RESUME_PRIVATE_INPUTS_SCHEMA
    assert status["resume_private_inputs"]["resume_command_omits_private_credentials"] is True


def test_alpha_status_reads_imported_quota_summary_without_local_preflight_report() -> None:
    out = _tmp_dir()
    deploy_dir = out / "deploy"
    service = _write(deploy_dir / "glm52_kaggle_alpha_service.json", _service_report(deploy_dir))
    quota = _write(out / "external-quota.json", _gpu_quota_report())
    args = pack.parse_args([
        "--output-dir",
        str(deploy_dir),
        "--service-report",
        str(service),
        "--gpu-quota-report",
        str(quota),
        "--min-tokens",
        "8",
    ])
    alpha_report = pack.build_report(args)
    _write(deploy_dir / "glm52_kaggle_alpha.json", alpha_report)

    status_args = cli.parse_args(["status", "glm52-kaggle", "--output-dir", str(deploy_dir)])
    status = cli.build_glm52_kaggle_alpha_status(status_args)

    assert status["ok"] is True
    assert status["gpu_quota_preflight_report_present"] is False
    assert status["gpu_quota_status"]["present"] is True
    assert status["gpu_quota_status"]["source"] == "alpha_gpu_quota_summary"
    assert status["gpu_quota_status"]["all_auth_ok_accounts_gpu_quota_exhausted"] is True
    assert status["phase"] == "blocked_gpu_quota"
    assert status["phase_status"]["overall_state"] == "blocked"
    assert "gpu_quota_preflight" in status["phase_status"]["blocked_phase_names"]
    assert status["resume_private_inputs"]["schema"] == alpha.RESUME_PRIVATE_INPUTS_SCHEMA
    assert status["resume_private_inputs"]["resume_command_omits_private_credentials"] is True


def test_alpha_cleanup_reads_deploy_blocker_artifact_and_quota_preflight() -> None:
    out = _tmp_dir()
    deploy_dir = out / "deploy"
    service = _write(deploy_dir / "glm52_kaggle_alpha_service.json", _service_report(deploy_dir))
    quota = _write(deploy_dir / "gpu-quota-preflight" / "kaggle_gpu_token_weekly_quota_probe.json", _gpu_quota_report())
    args = pack.parse_args([
        "--output-dir",
        str(deploy_dir),
        "--service-report",
        str(service),
        "--gpu-quota-report",
        str(quota),
        "--min-tokens",
        "8",
    ])
    alpha_report = pack.build_report(args)
    _write(deploy_dir / "glm52_kaggle_alpha.json", alpha_report)
    _write(deploy_dir / "glm52_kaggle_alpha_cli_summary.json", {"live_skipped_by_gpu_quota_preflight": True})

    cleanup_args = cli.parse_args(["cleanup", "glm52-kaggle", "--output-dir", str(deploy_dir)])
    cleanup = cli.build_glm52_kaggle_alpha_cleanup(cleanup_args)

    assert cleanup["ok"] is True
    assert cleanup["cleanup_evidence_source"] == "alpha_report"
    assert cleanup["alpha_report_present"] is True
    assert cleanup["gpu_quota_preflight_report_present"] is True
    assert cleanup["live_skipped_by_gpu_quota_preflight"] is True
    assert cleanup["alpha_cleanup_verified"] is True
    assert cleanup["quota_preflight_cleanup_verified"] is True
    assert cleanup["temporary_kaggle_kernels_deleted"] is True
    assert cleanup["temporary_private_packages_removed"] is True
    assert cleanup["live_resources_left_running"] is False
    assert cleanup["cleanup_status"]["cleanup_mode"] == "gpu_quota_preflight_skipped_live"
    assert (deploy_dir / "glm52_kaggle_alpha_cleanup.json").is_file()


def test_alpha_status_default_output_dir_matches_deploy_default() -> None:
    args = cli.parse_args(["status"])

    assert args.output_dir == "dist/glm52-kaggle-alpha"


def test_alpha_serve_default_output_dir_matches_deploy_default() -> None:
    args = cli.parse_args(["serve", "glm52-kaggle"])

    assert args.output_dir == "dist/glm52-kaggle-alpha"


def test_alpha_serve_explicit_output_dir_is_preserved() -> None:
    args = cli.parse_args(["serve", "glm52-kaggle", "--output-dir", "dist/custom-alpha-service"])

    assert args.output_dir == "dist/custom-alpha-service"


def test_product_serve_default_output_dir_is_unchanged() -> None:
    args = cli.parse_args(["serve"])

    assert args.output_dir == "dist/glm52-kaggle-alpha-service"


def test_alpha_cleanup_default_output_dir_matches_deploy_default() -> None:
    args = cli.parse_args(["cleanup"])

    assert args.output_dir == "dist/glm52-kaggle-alpha"


def test_alpha_pack_rejects_old_single_token_live_report_as_ready() -> None:
    out = _tmp_dir()
    service = _write(out / "service.json", _service_report(out))
    live_report = _write(out / "live.json", _live_report(tokens=1))
    args = pack.parse_args([
        "--output-dir",
        str(out / "alpha"),
        "--service-report",
        str(service),
        "--live-report",
        str(live_report),
        "--min-tokens",
        "8",
    ])

    report = pack.build_report(args)

    assert report["glm52_kaggle_alpha_ready"] is False
    assert "glm52_alpha_old_single_token_live_report" in report["blockers"]
    assert "generated_token_count_below_minimum" in check.validate_report(report, require_ready=True)
