from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts import glm52_kaggle_same_request_check as check
from scripts import glm52_kaggle_same_request_probe as probe


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_same_request_probe_"))


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _stage(provider: str, stage_id: int, *, request_hash: str = "") -> dict:
    return {
        "schema": probe.STAGE_SCHEMA,
        "provider": provider,
        "stage_id": stage_id,
        "model_id": probe.MODEL_ID,
        "compatible_weight_repo": probe.COMPATIBLE_WEIGHT_REPO,
        "coordinator_request_id_hash": request_hash or _hash("b"),
        "stage_layer_range": [stage_id * 2, stage_id * 2 + 1],
        "stage_execution_verified": True,
        "stage_decode_verified": True,
        "stage_output_hash": _hash(str(stage_id)),
        "weight_tensor_values_loaded": True,
        "weight_value_byte_count": 16,
        "weight_value_sha256": _hash("w"),
        "weight_tensor_values_public": False,
        "live_run_performed": True,
        "public_artifact_safe": True,
    }


def _coordinator(*, request_hash: str = "") -> dict:
    return {
        "schema": "glm52_kaggle_coordinator_decode_v1",
        "model_id": probe.MODEL_ID,
        "coordinator_request_id_hash": request_hash or _hash("b"),
        "generated_token_count": 1,
        "generated_token_hash": _hash("a"),
        "live_run_performed": True,
        "public_artifact_safe": True,
    }


def _cleanup() -> dict:
    return {
        "temporary_kaggle_kernels_deleted": True,
        "temporary_private_packages_removed": True,
        "live_resources_left_running": False,
        "public_artifact_safe": True,
    }


def test_preflight_writes_public_safe_not_started_report() -> None:
    out = _tmp_dir()
    args = probe.parse_args(["--mode", "preflight", "--output-dir", str(out)])

    report = probe.build_report(args)
    probe.write_json(out / "glm52_kaggle_same_request_probe.json", report)

    assert report["same_request_decode_verified"] is False
    assert report["public_artifact_safe"] is True
    assert "glm52_same_request_live_run_not_started" in report["blockers"]
    assert "glm52_stage_reports_missing" in report["blockers"]
    assert "same_request_not_verified" in check.validate_report(report, require_verified=True)


def test_assemble_accepts_three_provider_same_request_proof() -> None:
    out = _tmp_dir()
    request_hash = _hash("c")
    args = probe.parse_args(["--mode", "assemble", "--output-dir", str(out)])
    stages = [
        _stage("kaggle_cuda", 0, request_hash=request_hash),
        _stage("kaggle_jax_tpu", 1, request_hash=request_hash),
        _stage("kaggle_cpu", 2, request_hash=request_hash),
    ]

    report = probe.build_report(
        args,
        stage_reports=stages,
        coordinator_report=_coordinator(request_hash=request_hash),
        cleanup_report=_cleanup(),
    )

    assert report["same_request_decode_verified"] is True
    assert report["accepted_providers"] == probe.REQUIRED_PROVIDERS
    assert check.validate_report(report, require_verified=True) == []


def test_assemble_rejects_tpu_stage_smoke_as_same_request_stage() -> None:
    args = probe.parse_args(["--mode", "assemble"])
    stages = [
        _stage("kaggle_cuda", 0),
        {
            **_stage("kaggle_jax_tpu", 1),
            "schema": "glm52_awq_tpu_stage_smoke_v1",
            "stage_smoke_only": True,
        },
        _stage("kaggle_cpu", 2),
    ]

    report = probe.build_report(
        args,
        stage_reports=stages,
        coordinator_report=_coordinator(),
        cleanup_report=_cleanup(),
    )

    assert report["same_request_decode_verified"] is False
    assert "stage_report_is_stage_smoke_only" in report["blockers"]
    assert "same_request_provider_missing:kaggle_jax_tpu" in report["blockers"]


def test_assemble_rejects_stage_without_weight_value_evidence() -> None:
    args = probe.parse_args(["--mode", "assemble"])
    stages = [
        _stage("kaggle_cuda", 0),
        {**_stage("kaggle_jax_tpu", 1), "weight_tensor_values_loaded": False, "weight_value_sha256": ""},
        _stage("kaggle_cpu", 2),
    ]

    report = probe.build_report(
        args,
        stage_reports=stages,
        coordinator_report=_coordinator(),
        cleanup_report=_cleanup(),
    )

    assert report["same_request_decode_verified"] is False
    assert "stage_weight_values_not_loaded" in report["blockers"]
    assert "same_request_provider_missing:kaggle_jax_tpu" in report["blockers"]


def test_assemble_rejects_stage_value_op_without_decode_evidence() -> None:
    args = probe.parse_args(["--mode", "assemble"])
    stages = [
        _stage("kaggle_cuda", 0),
        {**_stage("kaggle_jax_tpu", 1), "stage_decode_verified": False},
        _stage("kaggle_cpu", 2),
    ]

    report = probe.build_report(
        args,
        stage_reports=stages,
        coordinator_report=_coordinator(),
        cleanup_report=_cleanup(),
    )

    assert report["same_request_decode_verified"] is False
    assert "stage_decode_not_verified" in report["blockers"]
    assert "same_request_provider_missing:kaggle_jax_tpu" in report["blockers"]


def test_assemble_rejects_missing_cleanup_and_request_hash_mismatch() -> None:
    args = probe.parse_args(["--mode", "assemble"])
    stages = [
        _stage("kaggle_cuda", 0, request_hash=_hash("c")),
        _stage("kaggle_jax_tpu", 1, request_hash=_hash("d")),
        _stage("kaggle_cpu", 2, request_hash=_hash("c")),
    ]

    report = probe.build_report(
        args,
        stage_reports=stages,
        coordinator_report=_coordinator(request_hash=_hash("c")),
        cleanup_report={},
    )

    assert report["same_request_decode_verified"] is False
    assert "glm52_same_request_hash_not_unique" in report["blockers"]
    assert "stage_coordinator_request_hash_mismatch" in report["blockers"]
    assert "cleanup_kernel_delete_missing" in report["blockers"]
    assert "cleanup_private_package_removal_missing" in report["blockers"]
