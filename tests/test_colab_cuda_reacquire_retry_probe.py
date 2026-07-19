import importlib.util
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab_cuda_reacquire_retry_probe.py"
spec = importlib.util.spec_from_file_location("colab_cuda_reacquire_retry_probe", MODULE_PATH)
probe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(probe)


def test_build_report_records_success_after_retry(tmp_path):
    attempts = [
        {
            "attempt_index": 1,
            "accelerator_requested": "T4",
            "authuser": "0",
            "ok": False,
            "report_path": str(tmp_path / "attempt-01" / "colab_cuda_session_probe.json"),
            "blockers": ["colab_gpu_assignment_http_503"],
        },
        {
            "attempt_index": 2,
            "accelerator_requested": "T4",
            "authuser": "1",
            "ok": True,
            "report_path": str(tmp_path / "attempt-02" / "colab_cuda_session_probe.json"),
            "accelerator": "T4",
            "endpoint_hash": "endpoint",
            "runtime_proxy_host_hash": "host",
            "blockers": [],
        },
    ]

    report = probe.build_report(
        output_dir=tmp_path,
        session_name="ct-colab-cuda-gpu",
        accelerators=["T4"],
        authusers=["0", "1"],
        attempts_requested=3,
        sleep_seconds=0,
        attempt_summaries=attempts,
        started=time.time(),
    )

    assert report["ok"] is True
    assert report["colab_cuda_reacquire_ready"] is True
    assert report["successful_attempt_index"] == 2
    assert report["accelerator"] == "T4"
    assert report["authuser"] == "1"
    assert report["endpoint_hash"] == "endpoint"
    assert report["runtime_proxy_token_public"] is False


def test_build_report_records_all_failure_blockers(tmp_path):
    attempts = [
        {
            "attempt_index": 1,
            "accelerator_requested": "T4",
            "authuser": "0",
            "ok": False,
            "report_path": str(tmp_path / "attempt-01" / "colab_cuda_session_probe.json"),
            "blockers": ["colab_gpu_assignment_http_503"],
        },
        {
            "attempt_index": 2,
            "accelerator_requested": "L4",
            "authuser": "1",
            "ok": False,
            "report_path": str(tmp_path / "attempt-02" / "colab_cuda_session_probe.json"),
            "blockers": ["colab_gpu_assignment_http_400"],
        },
    ]

    report = probe.build_report(
        output_dir=tmp_path,
        session_name="ct-colab-cuda-gpu",
        accelerators=["T4", "L4"],
        authusers=["0", "1"],
        attempts_requested=2,
        sleep_seconds=0,
        attempt_summaries=attempts,
        started=time.time(),
    )

    assert report["ok"] is False
    assert report["colab_cuda_reacquire_ready"] is False
    assert "colab_gpu_assignment_http_503" in report["blockers"]
    assert "colab_gpu_assignment_http_400" in report["blockers"]
    assert report["public_artifact_safe"] is True


def test_summarize_attempt_keeps_sensitive_values_private(tmp_path):
    report_path = tmp_path / "colab_cuda_session_probe.json"
    report = {
        "schema": "colab_cuda_session_probe_v1",
        "ok": False,
        "colab_cuda_session_allocated": False,
        "public_artifact_safe": True,
        "oauth_token_public": False,
        "runtime_proxy_token_public": False,
        "runtime_proxy_url_public": False,
        "endpoint_public": False,
        "http_status": 503,
        "error_type": "PublicHTTPError",
        "diagnosis_codes": ["colab_gpu_assignment_resource_unavailable"],
    }

    summary = probe.summarize_attempt(1, "T4", report_path, report, 1)

    assert summary["ok"] is False
    assert summary["runtime_proxy_token_public"] is False
    assert summary["runtime_proxy_url_public"] is False
    assert summary["endpoint_public"] is False
    assert "colab_gpu_assignment_http_503" in summary["blockers"]


def test_run_session_probe_forwards_cleanup_before_gpu(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = [str(part) for part in command]
        report_path = tmp_path / "attempt" / "colab_cuda_session_probe.json"
        probe.write_json(
            report_path,
            {
                "schema": "colab_cuda_session_probe_v1",
                "ok": False,
                "colab_cuda_session_allocated": False,
                "public_artifact_safe": True,
                "oauth_token_public": False,
                "runtime_proxy_token_public": False,
                "runtime_proxy_url_public": False,
                "endpoint_public": False,
            },
        )
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    args = SimpleNamespace(
        session_name="ct-colab-cuda-gpu",
        token_cache="/tmp/token.json",
        state_path="/tmp/sessions.json",
        cleanup_other_gpu=True,
        cleanup_before_gpu=True,
        attempt_timeout_seconds=10,
    )

    probe.run_session_probe(args, attempt_index=1, accelerator="T4", authuser="2", attempt_dir=tmp_path / "attempt")

    assert "--cleanup-other-gpu" in captured["command"]
    assert "--cleanup-before-gpu" in captured["command"]
    assert captured["command"][captured["command"].index("--authuser") + 1] == "2"
    assert captured["command"][captured["command"].index("--accelerator") + 1] == "T4"


def test_parse_args_accepts_authuser_and_accelerator_rotation(tmp_path):
    args = probe.parse_args([
        "--output-dir",
        str(tmp_path),
        "--authusers",
        "0,1",
        "--accelerators",
        "T4,L4",
    ])

    assert args.authuser_list == ["0", "1"]
    assert args.accelerator_list == ["T4", "L4"]
