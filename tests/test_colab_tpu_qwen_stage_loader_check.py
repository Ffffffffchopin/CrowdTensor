import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab_tpu_qwen_stage_loader_check.py"
spec = importlib.util.spec_from_file_location("colab_tpu_qwen_stage_loader_check", MODULE_PATH)
check = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(check)

PROBE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab_tpu_qwen_stage_loader_probe.py"
probe_spec = importlib.util.spec_from_file_location("colab_tpu_qwen_stage_loader_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(probe_spec)
assert probe_spec.loader is not None
probe_spec.loader.exec_module(probe)


def valid_report():
    return {
        "schema": "colab_tpu_qwen_stage_loader_probe_v1",
        "ok": True,
        "colab_qwen_stage_loader_ready": True,
        "public_artifact_safe": True,
        "runtime_proxy_token_public": False,
        "runtime_proxy_url_public": False,
        "endpoint_public": False,
        "endpoint_hash": "abc123",
        "runtime_proxy_host_hash": "def456",
        "executed_layer_count": 4,
        "missing_stage_key_count": 0,
        "tpu_device_count": 1,
    }


def test_ready_report_passes():
    assert check.validate(valid_report(), require_ready=True) == []


def test_failed_report_passes_without_ready_requirement():
    report = valid_report()
    report["ok"] = False
    report["colab_qwen_stage_loader_ready"] = False

    assert check.validate(report, require_ready=False) == []


def test_rejects_failed_report_when_ready_required():
    report = valid_report()
    report["ok"] = False

    assert "loader probe did not succeed" in check.validate(report, require_ready=True)


def test_rejects_full_72b_overclaim():
    report = valid_report()
    report["gpu_tpu_cpu_72b_same_request_verified"] = True

    assert "stage loader report must not claim full 72B same-request success" in check.validate(report, require_ready=False)


def test_rejects_public_runtime_proxy_url_flag():
    report = valid_report()
    report["runtime_proxy_url_public"] = True

    assert "runtime_proxy_url_public must be false" in check.validate(report)


def test_extract_runtime_report_prefers_nested_report_wrapper():
    outputs = [
        {
            "text": "\n".join(
                [
                    "progress",
                    '{"schema": "kaggle_tpu_32b_stage_owned_loader_probe_v1", '
                    '"report": {"ok": true, "full_stage_owned_tpu_loader_ready": true, '
                    '"executed_layer_count": 4, "tpu_device_count": 1}}',
                ]
            )
        }
    ]

    report = probe.extract_runtime_report(outputs)

    assert report["ok"] is True
    assert report["full_stage_owned_tpu_loader_ready"] is True
    assert report["executed_layer_count"] == 4
    assert "report" not in report
