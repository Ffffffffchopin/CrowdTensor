from __future__ import annotations

import copy
import tempfile
from pathlib import Path

from scripts import deepseek_v4_flash_torch_stage_adapter_smoke as smoke
from scripts import deepseek_v4_flash_torch_stage_adapter_smoke_check as check


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_dsv4_torch_smoke_"))


def test_torch_reference_smoke_exercises_deepseek_components_without_overclaiming_tpu() -> None:
    report = smoke.build_report(smoke.parse_args(["--output-dir", str(_tmp_dir())]))

    assert report["ok"] is True
    assert report["deepseek_v4_flash_torch_stage_adapter_smoke_ready"] is True
    assert report["jax_tpu_translation_ready"] is False
    assert report["real_deepseek_weights_loaded"] is False
    components = report["reference_stage"]["real_deepseek_v4_components_exercised"]
    for name in [
        "manifold_hyper_connections",
        "compressed_attention",
        "mla_shared_kv_attention",
        "grouped_output_projection",
        "moe_router",
        "routed_experts",
        "shared_experts",
        "stage_local_kv_cache_shape",
    ]:
        assert components[name] is True
    assert check.validate_report(report) == []


def test_checker_rejects_non_finite_reference_summary() -> None:
    report = smoke.build_report(smoke.parse_args(["--output-dir", str(_tmp_dir())]))
    bad = copy.deepcopy(report)
    bad["reference_stage"]["output_summary"]["mean"] = float("nan")

    assert "non_finite_output_summary:mean" in check.validate_report(bad)


def test_checker_rejects_jax_tpu_overclaim_from_torch_smoke() -> None:
    report = smoke.build_report(smoke.parse_args(["--output-dir", str(_tmp_dir())]))
    bad = copy.deepcopy(report)
    bad["jax_tpu_translation_ready"] = True

    assert "jax_tpu_translation_overclaimed" in check.validate_report(bad)
