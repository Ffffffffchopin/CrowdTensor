from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest

from scripts import deepseek_v4_flash_jax_stage_adapter_smoke as smoke
from scripts import deepseek_v4_flash_jax_stage_adapter_smoke_check as check


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_dsv4_jax_smoke_"))


def test_numpy_reference_records_deepseek_stage_components_without_overclaiming_jax() -> None:
    report = smoke.build_report(smoke.parse_args(["--output-dir", str(_tmp_dir())]))

    assert report["ok"] is False
    assert report["deepseek_v4_flash_jax_stage_adapter_smoke_ready"] is False
    assert report["numpy_reference"]["ok"] is True
    assert report["jax_runtime_execution_requested"] is False
    assert report["jax_runtime_execution_ready"] is False
    assert "jax_execution_not_requested" in report["blockers"]
    components = report["numpy_reference"]["components_exercised"]
    for component in [
        "manifold_hyper_connections",
        "mla_shared_kv_attention",
        "grouped_output_projection",
        "attention_sink",
        "topk_moe_router",
        "routed_experts",
        "shared_experts",
        "hca_compressor_shape_metadata",
    ]:
        assert components[component] is True
    assert check.validate_report(report) == []


def test_run_jax_records_current_environment_without_overclaiming_tpu() -> None:
    report = smoke.build_report(smoke.parse_args(["--output-dir", str(_tmp_dir()), "--run-jax"]))

    assert report["jax_runtime_execution_requested"] is True
    if report["jax_runtime_execution_ready"] is True:
        assert report["deepseek_v4_jax_stage_forward_ready"] is True
        assert report["deepseek_v4_jax_tpu_stage_forward_ready"] is False
        assert report["tpu_runtime_ready"] is False
        assert report["blockers"] == []
    else:
        assert "jax_missing" in report["blockers"]
        assert report["deepseek_v4_jax_stage_forward_ready"] is False
    assert check.validate_report(report) == []


def test_checker_accepts_cpu_jax_ready_without_tpu_overclaim() -> None:
    report = smoke.build_report(smoke.parse_args(["--output-dir", str(_tmp_dir())]))
    ready = copy.deepcopy(report)
    ready["ok"] = True
    ready["deepseek_v4_flash_jax_stage_adapter_smoke_ready"] = True
    ready["jax_runtime_execution_requested"] = True
    ready["jax_runtime_execution_ready"] = True
    ready["deepseek_v4_jax_stage_forward_ready"] = True
    ready["deepseek_v4_jax_tpu_stage_forward_ready"] = False
    ready["blockers"] = []
    ready["jax_result"] = {
        "ok": True,
        "jax_runtime_execution_ready": True,
        "tpu_runtime_ready": False,
        "jax_device_count": 1,
        "jax_tpu_device_count": 0,
        "jax_devices_public": [{"platform": "cpu", "device_kind": "cpu"}],
        "output_summary": ready["numpy_reference"]["output_summary"],
        "stage_local_kv_cache_metadata": ready["numpy_reference"]["stage_local_kv_cache_metadata"],
        "blockers": [],
    }

    assert check.validate_report(ready) == []


def test_checker_rejects_jax_stage_forward_overclaim() -> None:
    report = smoke.build_report(smoke.parse_args(["--output-dir", str(_tmp_dir())]))
    bad = copy.deepcopy(report)
    bad["deepseek_v4_jax_stage_forward_ready"] = True

    assert "jax_stage_forward_overclaimed" in check.validate_report(bad)


def test_checker_rejects_tpu_stage_forward_overclaim() -> None:
    report = smoke.build_report(smoke.parse_args(["--output-dir", str(_tmp_dir())]))
    bad = copy.deepcopy(report)
    bad["deepseek_v4_jax_tpu_stage_forward_ready"] = True

    assert "tpu_stage_forward_overclaimed" in check.validate_report(bad)


def test_require_tpu_requires_run_jax() -> None:
    with pytest.raises(SystemExit):
        smoke.parse_args(["--require-tpu"])
