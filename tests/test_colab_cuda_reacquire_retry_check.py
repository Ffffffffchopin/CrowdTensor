import importlib.util
from pathlib import Path


CHECK_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab_cuda_reacquire_retry_check.py"
spec = importlib.util.spec_from_file_location("colab_cuda_reacquire_retry_check", CHECK_PATH)
check = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(check)


def valid_success_report():
    return {
        "schema": "colab_cuda_reacquire_retry_probe_v1",
        "ok": True,
        "colab_cuda_reacquire_ready": True,
        "public_artifact_safe": True,
        "oauth_token_public": False,
        "runtime_proxy_token_public": False,
        "runtime_proxy_url_public": False,
        "endpoint_public": False,
        "credentials_public": False,
        "private_runtime_state_public": False,
        "attempts_requested": 3,
        "attempts_completed": 2,
        "successful_attempt_index": 2,
        "successful_report_path": "dist/example/attempt-02-t4/colab_cuda_session_probe.json",
        "accelerator": "T4",
        "authuser": "1",
        "endpoint_hash": "endpoint",
        "runtime_proxy_host_hash": "host",
        "blockers": [],
        "attempts": [
            {
                "attempt_index": 1,
                "accelerator_requested": "T4",
                "authuser": "0",
                "ok": False,
                "public_artifact_safe": True,
                "runtime_proxy_token_public": False,
                "runtime_proxy_url_public": False,
                "endpoint_public": False,
                "report_path": "dist/example/attempt-01-t4/colab_cuda_session_probe.json",
                "blockers": ["colab_gpu_assignment_http_503"],
            },
            {
                "attempt_index": 2,
                "accelerator_requested": "T4",
                "authuser": "1",
                "ok": True,
                "public_artifact_safe": True,
                "runtime_proxy_token_public": False,
                "runtime_proxy_url_public": False,
                "endpoint_public": False,
                "report_path": "dist/example/attempt-02-t4/colab_cuda_session_probe.json",
            },
        ],
    }


def valid_failure_report():
    report = valid_success_report()
    report.update(
        {
            "ok": False,
            "colab_cuda_reacquire_ready": False,
            "attempts_completed": 1,
            "successful_attempt_index": 0,
            "successful_report_path": "",
            "accelerator": "",
            "authuser": "",
            "endpoint_hash": "",
            "runtime_proxy_host_hash": "",
            "blockers": ["colab_gpu_assignment_http_503"],
            "attempts": [report["attempts"][0]],
        }
    )
    return report


def test_success_report_passes_with_require_ready():
    assert check.validate(valid_success_report(), require_ready=True) == []


def test_failure_report_passes_without_require_ready():
    assert check.validate(valid_failure_report()) == []


def test_failure_report_fails_with_require_ready():
    errors = check.validate(valid_failure_report(), require_ready=True)
    assert "Colab CUDA reacquire was not verified" in errors


def test_rejects_public_proxy_url():
    report = valid_success_report()
    report["attempts"][1]["runtime_proxy_url_public"] = True
    errors = check.validate(report)
    assert "attempt 1 exposes proxy URL" in errors


def test_rejects_missing_blocker_on_not_ready():
    report = valid_failure_report()
    report["blockers"] = []
    assert "not-ready report missing blockers" in check.validate(report)
