import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab_tpu_runtime_stability_check.py"
spec = importlib.util.spec_from_file_location("colab_tpu_runtime_stability_check", MODULE_PATH)
check = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(check)


def valid_report():
    return {
        "schema": "colab_tpu_runtime_stability_probe_v1",
        "public_artifact_safe": True,
        "runtime_proxy_token_public": False,
        "runtime_proxy_url_public": False,
        "endpoint_public": False,
        "ok": True,
        "colab_tpu_runtime_stably_acquired": True,
        "runtime_proxy_connected": True,
        "rounds_requested": 3,
        "rounds_completed": 3,
        "rounds_ready": 3,
        "observed_device_count_max": 1,
        "endpoint_hash": "abc123",
        "runtime_proxy_host_hash": "def456",
        "observations": [
            {"matmul_ready": True, "device_count": 1, "matmul_dtype": "bfloat16"},
            {"matmul_ready": True, "device_count": 1, "matmul_dtype": "bfloat16"},
            {"matmul_ready": True, "device_count": 1, "matmul_dtype": "bfloat16"},
        ],
    }


def test_valid_report_passes():
    assert check.validate(valid_report()) == []


def test_rejects_public_runtime_proxy_token():
    report = valid_report()
    report["runtime_proxy_token_public"] = True
    assert "runtime proxy token must not be public" in check.validate(report)


def test_rejects_incomplete_rounds():
    report = valid_report()
    report["rounds_ready"] = 2
    errors = check.validate(report)
    assert "not all requested rounds were TPU-ready" in errors


def test_rejects_non_tpu_observation():
    report = valid_report()
    report["observations"][1]["device_count"] = 0
    errors = check.validate(report)
    assert "round 1 has no TPU device" in errors
