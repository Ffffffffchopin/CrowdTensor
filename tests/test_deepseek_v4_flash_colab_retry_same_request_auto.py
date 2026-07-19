from pathlib import Path
import subprocess
from types import SimpleNamespace

from scripts import deepseek_v4_flash_colab_retry_same_request_auto as auto


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(tmp_path / "auto"),
        retry_output_dir=str(tmp_path / "retry"),
        same_request_output_dir=str(tmp_path / "same"),
        source_resolver_report="source.json",
        kaggle_owner="owner",
        runtime_tarball_path="runtime.tar.gz",
        runtime_tarball_sha256="sha256:test",
        colab_accelerators="T4",
        colab_authusers="0,1",
        colab_session="ct-colab-cuda-gpu",
        colab_config="/tmp/sessions.json",
        colab_token_cache="/tmp/token.json",
        colab_retry_attempts=2,
        colab_retry_sleep_seconds=0,
        colab_retry_attempt_timeout_seconds=30,
        cleanup_before_gpu=True,
        cleanup_other_gpu=False,
        colab_reacquire_before_same_request=False,
        colab_max_attempts=1,
        colab_background_launch_timeout_seconds=180,
        colab_background_timeout_seconds=7200,
        colab_background_poll_interval_seconds=30,
        colab_keepalive_seconds=7200,
        kernel_timeout_seconds=7200,
        kaggle_status_timeout_seconds=7500,
        kaggle_status_poll_interval=60,
        run_timeout_seconds=2400,
        max_new_tokens=1,
        context_length=64,
        retry_timeout_seconds=600,
        same_request_timeout_seconds=9000,
    )


def test_build_report_does_not_claim_success_when_retry_not_ready(tmp_path):
    args = _args(tmp_path)
    retry = {
        "schema": "colab_cuda_reacquire_retry_probe_v1",
        "ok": False,
        "colab_cuda_reacquire_ready": False,
        "attempts_completed": 2,
        "blockers": ["colab_gpu_assignment_http_503"],
        "public_artifact_safe": True,
    }

    report = auto.build_report(args=args, retry_report=retry, same_request_report={}, steps=[], started=0)

    assert report["ok"] is False
    assert report["retry_ready"] is False
    assert report["same_request_started"] is False
    assert report["failure_stage"] == "colab_cuda_reacquire_not_ready"
    assert "colab_cuda_reacquire_not_ready" in report["blockers"]


def test_build_report_requires_all_providers_and_token(tmp_path):
    args = _args(tmp_path)
    retry = {
        "schema": "colab_cuda_reacquire_retry_probe_v1",
        "ok": True,
        "colab_cuda_reacquire_ready": True,
        "attempts_completed": 1,
        "accelerator": "T4",
        "authuser": "0",
        "blockers": [],
        "public_artifact_safe": True,
    }
    same = {
        "schema": "deepseek_v4_flash_quantized_same_request_probe_v1",
        "ok": True,
        "same_request_decode_verified": True,
        "generated_token_count": 1,
        "accepted_providers": ["kaggle_cuda", "colab_cuda", "cpu"],
        "blockers": [],
        "public_artifact_safe": True,
    }

    report = auto.build_report(args=args, retry_report=retry, same_request_report=same, steps=[], started=0)

    assert report["ok"] is True
    assert report["same_request_decode_verified"] is True
    assert report["generated_token_count"] == 1


def test_same_request_command_uses_successful_retry_target(tmp_path):
    args = _args(tmp_path)
    retry = {"accelerator": "T4", "authuser": "1"}

    command = auto.build_same_request_command(args, tmp_path / "same", retry)

    assert command[command.index("--colab-accelerators") + 1] == "T4"
    assert command[command.index("--colab-authusers") + 1] == "1"
    assert "--colab-reacquire-before" not in command


def test_main_skips_same_request_when_retry_not_ready(monkeypatch, tmp_path):
    args = _args(tmp_path)
    retry_dir = Path(args.retry_output_dir)
    retry_dir.mkdir(parents=True)
    auto.write_json(
        retry_dir / "colab_cuda_reacquire_retry_probe.json",
        {
            "schema": "colab_cuda_reacquire_retry_probe_v1",
            "ok": False,
            "colab_cuda_reacquire_ready": False,
            "attempts_completed": 1,
            "blockers": ["colab_gpu_assignment_http_503"],
            "public_artifact_safe": True,
        },
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    monkeypatch.setattr(auto.subprocess, "run", fake_run)

    rc = auto.main([
        "--output-dir", args.output_dir,
        "--retry-output-dir", args.retry_output_dir,
        "--same-request-output-dir", args.same_request_output_dir,
        "--source-resolver-report", args.source_resolver_report,
        "--kaggle-owner", args.kaggle_owner,
        "--runtime-tarball-path", args.runtime_tarball_path,
        "--runtime-tarball-sha256", args.runtime_tarball_sha256,
        "--colab-retry-attempts", "1",
        "--colab-retry-sleep-seconds", "0",
    ])

    report = auto.load_json(Path(args.output_dir) / "deepseek_v4_flash_colab_retry_same_request_auto.json")
    assert rc == 0
    assert len(calls) == 1
    assert report["same_request_started"] is False
