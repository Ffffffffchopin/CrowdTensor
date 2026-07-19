from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts import glm52_kaggle_stage_runtime_plan as plan
from scripts import glm52_kaggle_stage_runtime_plan_check as check


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_stage_plan_"))


def _source_report() -> dict:
    return {
        "schema": "glm52_model_source_resolver_v1",
        "ok": True,
        "glm52_source_resolver_ready": True,
        "model": {"model_id": plan.MODEL_ID, "num_hidden_layers": 78},
        "stage_adapter_plan": {
            "schema": "glm52_kaggle_stage_adapter_plan_v1",
            "stage_count": 3,
            "stage_backends": plan.REQUIRED_PROVIDERS,
            "stage_plans": [
                {
                    "stage_id": 0,
                    "backend": "kaggle_cuda",
                    "layer_range": [0, 26],
                    "assigned_key_count": 18063,
                    "assigned_file_count": 95,
                    "key_digest": "sha256:" + "a" * 64,
                    "metadata_only": True,
                },
                {
                    "stage_id": 1,
                    "backend": "kaggle_jax_tpu",
                    "layer_range": [26, 52],
                    "assigned_key_count": 20367,
                    "assigned_file_count": 98,
                    "key_digest": "sha256:" + "b" * 64,
                    "metadata_only": True,
                },
                {
                    "stage_id": 2,
                    "backend": "kaggle_cpu",
                    "layer_range": [52, 78],
                    "assigned_key_count": 20364,
                    "assigned_file_count": 100,
                    "key_digest": "sha256:" + "c" * 64,
                    "metadata_only": True,
                },
            ],
        },
    }


def _awq_header() -> dict:
    return {
        "schema": "glm52_awq_stage_header_probe_v1",
        "model_repo": plan.COMPATIBLE_WEIGHT_REPO,
        "base_model_id": plan.MODEL_ID,
        "stage_id": 4,
        "stage_count": 12,
        "stage_layer_range": [28, 35],
        "assigned_weight_key_count": 21675,
        "total_selected_tensor_storage_gb": 40.524259,
        "public_artifact_safe": True,
    }


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_plan_builds_three_required_provider_stage_specs() -> None:
    report = plan.build_report(
        plan.parse_args([
            "--source-report",
            "",
            "--awq-stage-header-report",
            "",
        ])
    )

    providers = {spec["provider"] for spec in report["stage_specs"]}
    assert providers == set(plan.REQUIRED_PROVIDERS)
    assert report["stage_runtime_adapter_verified"] is False
    assert "glm52_stage_runtime_live_reports_missing" in report["blockers"]
    assert check.validate_report(report) == []


def test_plan_uses_source_ranges_and_tpu_awq_header_subset() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    awq = _write(base / "awq.json", _awq_header())

    report = plan.build_report(
        plan.parse_args([
            "--source-report",
            str(source),
            "--awq-stage-header-report",
            str(awq),
        ])
    )

    assert [spec["stage_layer_range"] for spec in report["stage_specs"]] == [[0, 26], [26, 52], [52, 78]]
    tpu = next(spec for spec in report["stage_specs"] if spec["provider"] == "kaggle_jax_tpu")
    assert tpu["awq_header_probe_subset"]["awq_header_layer_range"] == [28, 35]
    assert tpu["expected_stage_report_schema"] == "glm52_kaggle_stage_runtime_report_v1"
    assert check.validate_report(report) == []


def test_plan_accepts_cli_stage_spec_override_with_multiple_cpu_stages() -> None:
    report = plan.build_report(
        plan.parse_args([
            "--source-report",
            "",
            "--awq-stage-header-report",
            "",
            "--stage-spec",
            "0:kaggle_cuda:0:20",
            "--stage-spec",
            "1:kaggle_jax_tpu:20:52",
            "--stage-spec",
            "2:kaggle_cpu:52:60",
            "--stage-spec",
            "3:kaggle_cpu:60:69",
            "--stage-spec",
            "4:kaggle_cpu:69:78",
        ])
    )

    assert [spec["stage_layer_range"] for spec in report["stage_specs"]] == [
        [0, 20],
        [20, 52],
        [52, 60],
        [60, 69],
        [69, 78],
    ]
    assert report["stage_topology"]["source"] == "cli_stage_spec"
    assert report["stage_topology"]["provider_counts"]["kaggle_cpu"] == 3
    assert report["stage_topology"]["contiguous_full_layer_coverage"] is True
    assert [spec["stage_count"] for spec in report["stage_specs"]] == [5, 5, 5, 5, 5]
    assert check.validate_report(report) == []


def test_checker_rejects_noncontiguous_stage_topology() -> None:
    report = plan.build_report(
        plan.parse_args([
            "--stage-spec",
            "0:kaggle_cuda:0:20",
            "--stage-spec",
            "1:kaggle_jax_tpu:22:52",
            "--stage-spec",
            "2:kaggle_cpu:52:78",
        ])
    )

    errors = check.validate_report(report)

    assert "topology_layer_coverage_not_contiguous" in errors
    assert "glm52_stage_runtime_layer_coverage_not_contiguous" in report["blockers"]


def test_checker_rejects_verified_plan_with_blockers() -> None:
    report = plan.build_report(plan.parse_args([]))
    report["stage_runtime_adapter_verified"] = True
    report["same_request_route_verified"] = True

    errors = check.validate_report(report, require_verified=True)

    assert "verified_plan_has_blockers" in errors
    assert any(error.startswith("verified_plan_stage_not_verified:") for error in errors)


def test_cli_writes_plan_report() -> None:
    out = _tmp_dir()
    code = plan.main(["--output-dir", str(out)])

    assert code == 0
    payload = json.loads((out / "glm52_kaggle_stage_runtime_plan.json").read_text(encoding="utf-8"))
    assert payload["schema"] == plan.SCHEMA
    assert check.validate_report(payload) == []
