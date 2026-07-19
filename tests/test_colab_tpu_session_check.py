import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab_tpu_session_check.py"
spec = importlib.util.spec_from_file_location("colab_tpu_session_check", MODULE_PATH)
check = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(check)


def valid_report():
    return {
        "schema": "colab_tpu_session_probe_v1",
        "ok": True,
        "colab_tpu_session_allocated": True,
        "public_artifact_safe": True,
        "oauth_token_public": False,
        "runtime_proxy_token_public": False,
        "runtime_proxy_url_public": False,
        "endpoint_public": False,
        "endpoint_hash": "abc123",
        "runtime_proxy_host_hash": "def456",
        "accelerator": "V5E1",
    }


def test_valid_session_report_passes():
    assert check.validate(valid_report()) == []


def test_rejects_public_proxy_url_flag():
    report = valid_report()
    report["runtime_proxy_url_public"] = True
    assert "runtime_proxy_url_public must be false" in check.validate(report)


def test_rejects_failed_allocation():
    report = valid_report()
    report["ok"] = False
    errors = check.validate(report)
    assert "session allocation did not succeed" in errors


def test_rejects_unexpected_accelerator():
    report = valid_report()
    report["accelerator"] = "TPU_V5E8"
    assert "unexpected accelerator" in check.validate(report)
