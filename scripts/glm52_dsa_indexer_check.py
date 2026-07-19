#!/usr/bin/env python3
"""Validate GLM 5.2 DSA indexer probe reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_dsa_indexer_probe as probe  # noqa: E402


SCHEMA = "glm52_dsa_indexer_check_v1"


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 71


def validate_report(report: dict[str, Any], *, require_verified: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != probe.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = probe.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))
    if report.get("model_id") != probe.MODEL_ID:
        errors.append("model_id_mismatch")
    if report.get("model_type") != "glm_moe_dsa":
        errors.append("model_type_not_glm_moe_dsa")
    if report.get("layer_indexer_type") != "full":
        errors.append("layer_indexer_type_not_full")
    for key in ["dsa_indexer_verified", "dsa_topk_verified"]:
        if report.get(key) is not True:
            errors.append(f"{key}_missing")
    for key in ["indexer_cache_updated", "attention_output_verified", "stage_decode_verified"]:
        if report.get(key) is not False:
            errors.append(f"{key}_overclaim")
    seq_len = _int(report.get("sequence_length"))
    heads = _int(report.get("index_n_heads"))
    head_dim = _int(report.get("index_head_dim"))
    effective_topk = _int(report.get("effective_topk"))
    expected_shapes = {
        "hidden_norm_shape": [seq_len, _int(report.get("hidden_size"))],
        "q_resid_shape": [seq_len, _int(report.get("q_lora_rank"))],
        "indexer_query_shape": [seq_len, heads, head_dim],
        "indexer_key_shape": [seq_len, head_dim],
        "head_weights_shape": [seq_len, heads],
        "index_score_shape": [seq_len, seq_len],
        "topk_indices_shape": [seq_len, effective_topk],
    }
    for key, expected in expected_shapes.items():
        if _list(report.get(key)) != expected:
            errors.append(f"shape_mismatch:{key}")
    for key in [
        "hidden_norm_hash",
        "q_resid_hash",
        "indexer_query_hash",
        "indexer_key_hash",
        "head_weights_hash",
        "index_score_hash",
        "topk_indices_hash",
    ]:
        if not _hash_ok(report.get(key)):
            errors.append(f"hash_missing:{key}")
    for key in ["wq_b_weight_shape", "wk_weight_shape", "weights_proj_shape"]:
        if len(_list(report.get(key))) != 2:
            errors.append(f"weight_shape_invalid:{key}")
    if require_verified and report.get("dsa_indexer_verified") is not True:
        errors.append("dsa_indexer_verification_required")
    boundary = report.get("completion_boundary") if isinstance(report.get("completion_boundary"), dict) else {}
    for key in [
        "dsa_indexer_small_sequence_is_not_full_prefill",
        "dsa_indexer_is_not_kv_cache_decode",
        "dsa_indexer_is_not_attention_output",
        "dsa_indexer_is_not_transformer_block",
        "dsa_indexer_is_not_stage_decode",
        "requires_stage_decode_verified",
    ]:
        if boundary.get(key) is not True:
            errors.append(f"completion_boundary_missing:{key}")
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    for key in [
        "credentials_public",
        "cookies_public",
        "signed_url_public",
        "raw_prompt_public",
        "raw_generated_text_public",
        "generated_token_ids_public",
        "activation_public",
        "hidden_state_public",
        "logits_public",
        "kv_cache_public",
        "weight_tensor_values_public",
        "safetensors_header_payload_public",
    ]:
        if safety.get(key) is not False:
            errors.append(f"safety_flag_not_false:{key}")
    return sorted(set(errors))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-verified", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = load_json(Path(args.report))
    errors = validate_report(report, require_verified=bool(args.require_verified))
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "dsa_indexer_verified": report.get("dsa_indexer_verified") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_dsa_indexer_check: ok={result['ok']} "
            f"errors={len(errors)} dsa_indexer={result['dsa_indexer_verified']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
