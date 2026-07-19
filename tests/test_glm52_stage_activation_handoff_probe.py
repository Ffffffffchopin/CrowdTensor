from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts import glm52_stage_activation_handoff_check as check
from scripts import glm52_stage_activation_handoff_probe as probe


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_handoff_"))


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _stage(provider: str, stage_id: int, request_hash: str) -> dict:
    return {
        "schema": "glm52_kaggle_stage_runtime_report_v1",
        "ok": True,
        "model_id": probe.MODEL_ID,
        "provider": provider,
        "stage_id": stage_id,
        "stage_layer_range": [stage_id * 26, (stage_id + 1) * 26],
        "coordinator_request_id_hash": request_hash,
        "stage_output_hash": _hash(str(stage_id)),
        "stage_execution_verified": True,
        "stage_decode_verified": False,
        "same_request_route_verified": False,
        "stage_runtime_kind": "glm52_full_prefix_stage_decode_host_adapter_with_provider_op",
        "stage_owned_weight_values_loaded": True,
        "weight_tensor_values_loaded": True,
        "weight_tensor_values_public": False,
        "weight_value_byte_count": 16,
        "weight_value_sha256": _hash("w"),
        "live_run_performed": True,
        "stage_smoke_only": False,
        "public_artifact_safe": True,
        "blockers": ["glm52_stage_decode_not_verified"],
    }


def test_activation_handoff_probe_verifies_three_provider_hash_chain() -> None:
    base = _tmp_dir()
    request_hash = _hash("r")
    stage_paths = [
        _write(base / "cuda.json", _stage("kaggle_cuda", 0, request_hash)),
        _write(base / "tpu.json", _stage("kaggle_jax_tpu", 1, request_hash)),
        _write(base / "cpu.json", _stage("kaggle_cpu", 2, request_hash)),
    ]
    argv = ["--output-dir", str(base / "handoff")]
    for path in stage_paths:
        argv.extend(["--stage-report", str(path)])

    report = probe.build_report(probe.parse_args(argv))

    assert report["stage_activation_handoff_runtime_verified"] is True
    assert report["same_request_decode_verified"] is False
    assert report["generated_token_verified"] is False
    assert set(report["stage_runtime_provider_coverage"]) == set(probe.REQUIRED_PROVIDERS)
    assert report["handoff_count"] == 2
    assert all(item["handoff_verified"] is True for item in report["activation_handoffs"])
    assert check.validate_report(report, require_verified=True) == []


def test_activation_handoff_probe_fails_on_request_hash_split() -> None:
    base = _tmp_dir()
    stage_paths = [
        _write(base / "cuda.json", _stage("kaggle_cuda", 0, _hash("r"))),
        _write(base / "tpu.json", _stage("kaggle_jax_tpu", 1, _hash("s"))),
        _write(base / "cpu.json", _stage("kaggle_cpu", 2, _hash("r"))),
    ]
    argv = ["--output-dir", str(base / "handoff")]
    for path in stage_paths:
        argv.extend(["--stage-report", str(path)])

    report = probe.build_report(probe.parse_args(argv))

    assert report["stage_activation_handoff_runtime_verified"] is False
    assert "glm52_stage_activation_handoff_request_hash_not_unique" in report["blockers"]
    assert "stage_activation_handoff_not_verified" in check.validate_report(report, require_verified=True)
