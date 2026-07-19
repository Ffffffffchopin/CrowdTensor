from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts import glm52_awq_stage_value_probe as probe
from scripts import glm52_awq_stage_value_probe_check as check


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_stage_value_"))


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _report(**overrides) -> dict:
    report = {
        "schema": probe.SCHEMA,
        "ok": True,
        "glm52_awq_stage_value_probe_ready": True,
        "model_repo": check.MODEL_REPO,
        "base_model_id": check.BASE_MODEL_ID,
        "model_type": "glm_moe_dsa",
        "quantization": "AWQ-INT4",
        "stage_id": 4,
        "stage_count": 12,
        "stage_layer_range": [28, 35],
        "assigned_weight_key_count": 10,
        "assigned_weight_file_count": 1,
        "header_file_count": 1,
        "selected_tensor": {
            "key_digest": _hash("a"),
            "filename": "model-00001-of-00083.safetensors",
            "dtype": "I32",
            "shape_digest": _hash("b"),
            "rank": 1,
            "tensor_nbytes": 16,
            "data_offsets_digest": _hash("c"),
        },
        "weight_value_byte_count": 16,
        "weight_value_sha256": _hash("d"),
        "weight_tensor_values_loaded": True,
        "weight_tensor_values_public": False,
        "safetensors_header_payload_public": False,
        "stage_runtime_adapter_verified": False,
        "same_request_route_verified": False,
        "same_request_decode_verified": False,
        "stage_smoke_only": True,
        "blockers": [],
        "public_artifact_safe": True,
    }
    report.update(overrides)
    return report


def test_checker_accepts_ready_public_safe_value_probe() -> None:
    report = _report()

    assert check.validate_report(report, require_ready=True) == []


def test_checker_rejects_same_request_overclaim() -> None:
    report = _report(same_request_decode_verified=True, stage_runtime_adapter_verified=True)

    errors = check.validate_report(report, require_ready=True)

    assert "same_request_decode_overclaim" in errors
    assert "stage_runtime_adapter_overclaim" in errors


def test_build_report_reads_one_stage_owned_tensor_value_without_publishing(monkeypatch) -> None:
    out = _tmp_dir()
    tensor_bytes = b"0123456789abcdef"
    header_payload = {
        "model.layers.28.self_attn.q_proj.qzeros": {
            "dtype": "I32",
            "shape": [4],
            "data_offsets": [0, len(tensor_bytes)],
        }
    }
    header_bytes = json.dumps(header_payload).encode("utf-8")
    prefix = len(header_bytes).to_bytes(8, "little")

    def fake_fetch_json(repo, filename, *, timeout_seconds):
        if filename == "config.json":
            return {"model_type": "glm_moe_dsa", "num_hidden_layers": 78}
        return {
            "metadata": {"total_size": 1000},
            "weight_map": {"model.layers.28.self_attn.q_proj.qzeros": "model-00001-of-00083.safetensors"},
        }

    def fake_read_range(repo, filename, start, end, *, timeout_seconds, max_bytes):
        if start == 0 and end == 7:
            return prefix
        if start == 8:
            return header_bytes
        return tensor_bytes

    monkeypatch.setattr(probe.header_probe, "fetch_hf_json", fake_fetch_json)
    monkeypatch.setattr(probe.header_probe, "read_hf_range", fake_read_range)
    report = probe.build_report(
        probe.parse_args([
            "--output-dir",
            str(out),
            "--stage-id",
            "4",
            "--stage-count",
            "12",
            "--max-tensor-bytes",
            "64",
        ])
    )

    assert report["glm52_awq_stage_value_probe_ready"] is True
    assert report["weight_value_sha256"] == probe.sha_bytes(tensor_bytes)
    assert report["weight_tensor_values_public"] is False
    assert "0123456789abcdef" not in json.dumps(report)
    assert check.validate_report(report, require_ready=True) == []
