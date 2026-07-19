import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab_cuda_runtime_check.py"
spec = importlib.util.spec_from_file_location("colab_cuda_runtime_check", MODULE_PATH)
check = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(check)


def valid_report():
    return {
        "schema": "colab_cuda_runtime_probe_v1",
        "ok": True,
        "colab_cuda_runtime_ready": True,
        "runtime_proxy_connected": True,
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_matmul_ready": True,
        "torch_version": "2.0.0",
        "cuda_version": "12.1",
        "devices": [
            {
                "index": 0,
                "name_hash": "abc123",
                "name_public": False,
                "total_memory_mb": 15360,
                "major": 7,
                "minor": 5,
            }
        ],
        "endpoint_hash": "endpoint",
        "runtime_proxy_host_hash": "proxy",
        "runtime_proxy_token_public": False,
        "runtime_proxy_url_public": False,
        "endpoint_public": False,
        "credentials_public": False,
        "private_runtime_state_public": False,
        "public_artifact_safe": True,
    }


def test_valid_report_passes():
    assert check.validate(valid_report()) == []


def test_rejects_public_runtime_proxy_url():
    report = valid_report()
    report["runtime_proxy_url_public"] = True
    assert "runtime_proxy_url_public must be false" in check.validate(report)


def test_rejects_ready_without_cuda_device():
    report = valid_report()
    report["cuda_device_count"] = 0
    errors = check.validate(report)
    assert "no CUDA device was reported" in errors


def test_rejects_public_device_name():
    report = valid_report()
    report["devices"][0]["name_public"] = True
    errors = check.validate(report)
    assert "device 0 name must not be public" in errors


def test_not_ready_requires_blocker():
    report = valid_report()
    report["ok"] = False
    report["colab_cuda_runtime_ready"] = False
    report["blockers"] = []
    errors = check.validate(report)
    assert "not-ready report missing blockers" in errors


def test_valid_report_allows_session_manager_metadata():
    report = valid_report()
    report["session_manager"] = {
        "ok": True,
        "attempt_count": 2,
        "attempts": [
            {"attempt": 0, "event": "force_reacquire_before", "ok": True},
            {"attempt": 1, "ok": True, "stale_detected": False},
        ],
        "public_artifact_safe": True,
    }
    assert check.validate(report) == []
