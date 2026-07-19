#!/usr/bin/env python3
"""Build a public-safe GLM 5.2 stage activation handoff proof.

The probe consumes real stage runtime reports and verifies that adjacent
Kaggle CPU/GPU/TPU stages produced private activation hashes under one bound
Coordinator request id. This is a handoff-runtime contract over live stage
outputs, not a full same-request decode success artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "glm52_stage_activation_handoff_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-stage-activation-handoff-probe"
MODEL_ID = "zai-org/GLM-5.2"
REQUIRED_PROVIDERS = ["kaggle_cuda", "kaggle_jax_tpu", "kaggle_cpu"]
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "KAGGLE_API_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "token=",
    "runtime_proxy",
    "jupyter-proxy",
    '"prompt":',
    '"raw_prompt":',
    '"generated_text":',
    '"raw_generated_text":',
    '"generated_token_ids":',
    '"input_ids":',
    '"activation":',
    '"activations":',
    '"hidden_state":',
    '"hidden_states":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
    '"weight_tensor_values":',
    '"safetensors_header_payload":',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    loaded = json.loads(p.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 71


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def _model_id(report: dict[str, Any]) -> str:
    return str(report.get("model_id") or _dict(report.get("model")).get("model_id") or "")


def _request_hash(report: dict[str, Any]) -> str:
    same = _dict(report.get("same_request"))
    return str(
        report.get("coordinator_request_id_hash")
        or report.get("request_id_hash")
        or same.get("coordinator_request_id_hash")
        or same.get("request_id_hash")
        or ""
    )


def _output_hash(report: dict[str, Any]) -> str:
    return str(
        report.get("stage_output_hash")
        or report.get("output_hash")
        or report.get("activation_handoff_hash")
        or _dict(report.get("stage")).get("stage_output_hash")
        or ""
    )


def _weight_hash(report: dict[str, Any]) -> str:
    return str(
        report.get("weight_value_sha256")
        or report.get("stage_weight_value_hash")
        or report.get("weight_value_hash")
        or ""
    )


def normalize_stage(report: dict[str, Any], *, ordinal: int) -> dict[str, Any]:
    layer_range = [_int(item, -1) for item in _list(report.get("stage_layer_range"))[:2]]
    if len(layer_range) != 2:
        layer_range = []
    output_hash = _output_hash(report)
    request_hash = _request_hash(report)
    weight_hash = _weight_hash(report)
    provider = str(report.get("provider") or report.get("backend") or report.get("stage_provider") or "")
    blockers = [str(item) for item in _list(report.get("blockers")) if item]
    weight_values_loaded = bool(
        (
            report.get("stage_owned_weight_values_loaded") is True
            or report.get("weight_tensor_values_loaded") is True
        )
        and _int(report.get("weight_value_byte_count") or report.get("stage_weight_value_byte_count")) > 0
        and _hash_ok(weight_hash)
        and report.get("weight_tensor_values_public") is not True
    )
    stage_ready = bool(
        report
        and report.get("ok") is True
        and _model_id(report) == MODEL_ID
        and provider in REQUIRED_PROVIDERS
        and report.get("stage_execution_verified") is True
        and report.get("live_run_performed") is True
        and report.get("stage_smoke_only") is not True
        and report.get("public_artifact_safe") is True
        and _hash_ok(output_hash)
        and _hash_ok(request_hash)
        and weight_values_loaded
        and len(layer_range) == 2
        and 0 <= layer_range[0] < layer_range[1]
    )
    return {
        "source_schema": str(report.get("schema") or ""),
        "stage_id": _int(report.get("stage_id"), ordinal),
        "provider": provider,
        "model_id": _model_id(report),
        "stage_layer_range": layer_range,
        "coordinator_request_id_hash": request_hash if _hash_ok(request_hash) else "",
        "stage_output_hash": output_hash if _hash_ok(output_hash) else "",
        "weight_value_hash_present": _hash_ok(weight_hash),
        "weight_value_byte_count": _int(report.get("weight_value_byte_count") or report.get("stage_weight_value_byte_count")),
        "weight_tensor_values_loaded": weight_values_loaded,
        "stage_execution_verified": report.get("stage_execution_verified") is True,
        "stage_decode_verified": report.get("stage_decode_verified") is True,
        "same_request_route_verified": report.get("same_request_route_verified") is True,
        "live_run_performed": report.get("live_run_performed") is True,
        "stage_smoke_only": report.get("stage_smoke_only") is True,
        "stage_runtime_kind": str(report.get("stage_runtime_kind") or ""),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "stage_handoff_endpoint_ready": stage_ready,
        "activation_payload_public": False,
        "hidden_state_public": False,
        "kv_cache_public": False,
        "blockers": sorted(set(blockers)),
    }


def build_handoffs(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    handoffs: list[dict[str, Any]] = []
    for left, right in zip(stages, stages[1:]):
        left_range = _list(left.get("stage_layer_range"))
        right_range = _list(right.get("stage_layer_range"))
        boundary = _int(left_range[1], -1) if len(left_range) == 2 else -1
        contiguous = bool(len(left_range) == 2 and len(right_range) == 2 and _int(left_range[1]) == _int(right_range[0]))
        same_request_hash = str(left.get("coordinator_request_id_hash") or "") == str(right.get("coordinator_request_id_hash") or "") and _hash_ok(left.get("coordinator_request_id_hash"))
        producer_hash = str(left.get("stage_output_hash") or "")
        consumer_anchor_hash = str(right.get("stage_output_hash") or "")
        handoff_verified = bool(
            contiguous
            and same_request_hash
            and left.get("stage_handoff_endpoint_ready") is True
            and right.get("stage_handoff_endpoint_ready") is True
            and _hash_ok(producer_hash)
            and _hash_ok(consumer_anchor_hash)
        )
        handoff_contract = {
            "request_hash": left.get("coordinator_request_id_hash"),
            "from_stage_id": left.get("stage_id"),
            "to_stage_id": right.get("stage_id"),
            "from_provider": left.get("provider"),
            "to_provider": right.get("provider"),
            "layer_boundary": boundary,
            "producer_output_hash": producer_hash,
            "consumer_anchor_hash": consumer_anchor_hash,
        }
        handoffs.append(
            {
                "from_stage_id": left.get("stage_id"),
                "to_stage_id": right.get("stage_id"),
                "from_provider": left.get("provider"),
                "to_provider": right.get("provider"),
                "layer_boundary": boundary,
                "same_request_hash_verified": same_request_hash,
                "contiguous_layer_boundary": contiguous,
                "producer_output_hash_present": _hash_ok(producer_hash),
                "consumer_anchor_hash_present": _hash_ok(consumer_anchor_hash),
                "handoff_contract_hash": sha_payload(handoff_contract),
                "activation_payload_public": False,
                "handoff_verified": handoff_verified,
            }
        )
    return handoffs


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    raw_reports = [load_json(path) for path in args.stage_report]
    stages = [normalize_stage(report, ordinal=index) for index, report in enumerate(raw_reports) if report]
    stages.sort(key=lambda stage: (_int(_list(stage.get("stage_layer_range"))[0] if _list(stage.get("stage_layer_range")) else -1), _int(stage.get("stage_id"))))
    handoffs = build_handoffs(stages)
    request_hashes = {str(stage.get("coordinator_request_id_hash") or "") for stage in stages if _hash_ok(stage.get("coordinator_request_id_hash"))}
    provider_coverage = sorted({str(stage.get("provider") or "") for stage in stages if stage.get("stage_handoff_endpoint_ready") is True})
    model_ids = {str(stage.get("model_id") or "") for stage in stages}
    blockers: set[str] = set()
    if len(stages) < int(args.min_stage_count):
        blockers.add("glm52_stage_activation_handoff_stage_count_below_minimum")
    if model_ids != {MODEL_ID}:
        blockers.add("glm52_stage_activation_handoff_model_mismatch")
    if len(request_hashes) != 1:
        blockers.add("glm52_stage_activation_handoff_request_hash_not_unique")
    if not set(REQUIRED_PROVIDERS).issubset(set(provider_coverage)):
        blockers.add("glm52_stage_activation_handoff_provider_coverage_incomplete")
    for stage in stages:
        if stage.get("stage_handoff_endpoint_ready") is not True:
            blockers.add(f"glm52_stage_activation_handoff_endpoint_not_ready:{stage.get('provider') or stage.get('stage_id')}")
        if stage.get("stage_decode_verified") is not True:
            blockers.add("glm52_stage_decode_not_verified")
        blockers.update(str(item) for item in _list(stage.get("blockers")) if item)
    for handoff in handoffs:
        if handoff.get("handoff_verified") is not True:
            blockers.add(f"glm52_stage_activation_handoff_not_verified:{handoff.get('from_stage_id')}->{handoff.get('to_stage_id')}")

    handoff_verified = bool(
        len(handoffs) >= max(1, int(args.min_stage_count) - 1)
        and all(item.get("handoff_verified") is True for item in handoffs)
        and set(REQUIRED_PROVIDERS).issubset(set(provider_coverage))
        and len(request_hashes) == 1
        and model_ids == {MODEL_ID}
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "glm52_stage_activation_handoff_probe_ready": True,
        "stage_activation_handoff_runtime_verified": handoff_verified,
        "stage_activation_handoff_contract_verified": handoff_verified,
        "same_request_decode_verified": False,
        "stage_decode_verified": all(stage.get("stage_decode_verified") is True for stage in stages) if stages else False,
        "generated_token_verified": False,
        "model_id": MODEL_ID,
        "stage_count": len(stages),
        "handoff_count": len(handoffs),
        "required_provider_coverage": REQUIRED_PROVIDERS,
        "stage_runtime_provider_coverage": provider_coverage,
        "coordinator_request_id_hash_present": len(request_hashes) == 1,
        "coordinator_request_id_hash": sorted(request_hashes)[0] if len(request_hashes) == 1 else "",
        "stages": stages,
        "activation_handoffs": handoffs,
        "blockers": [] if handoff_verified else sorted(blockers),
        "completion_boundary": {
            "activation_handoff_evidence_is_not_same_request_decode": True,
            "activation_handoff_evidence_is_not_generated_token": True,
            "activation_handoff_evidence_is_not_stage_decode": True,
            "requires_coordinator_same_request_decode": True,
            "requires_stage_local_kv_cache": True,
            "requires_lm_head_token_selection": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["stage_activation_handoff_runtime_verified"] = False
        report["stage_activation_handoff_contract_verified"] = False
        report["blockers"] = sorted(set(_list(report.get("blockers")) + ["public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage-report", action="append", default=[])
    parser.add_argument("--min-stage-count", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_stage_activation_handoff_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Stage activation handoff verified: {report.get('stage_activation_handoff_runtime_verified')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
