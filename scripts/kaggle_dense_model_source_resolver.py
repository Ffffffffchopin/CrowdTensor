#!/usr/bin/env python3
"""Resolve public-safe Kaggle attached model sources for dense Qwen runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "kaggle_dense_model_source_resolver_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-dense-model-source-resolver"
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Cookie:",
    "Set-Cookie",
    "kaggle-cookies",
    "kaggle-web-storage-state",
    "operator.private.env",
    "miner.private.env",
    '"prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"activation":',
    '"activations":',
    '"hidden_state":',
    '"hidden_states":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
    '"lease_token":',
    '"idempotency_key":',
)

DEFAULT_CANDIDATES = (
    "72b|Qwen/Qwen2.5-72B-Instruct|qwen-lm|qwen2.5|Transformers|72b-instruct|1|Other (specified in description)|145424099850",
    "32b|Qwen/Qwen2.5-32B-Instruct|qwen-lm|qwen2.5|Transformers|32b-instruct|1|Apache 2.0|65539410984",
    "14b|Qwen/Qwen2.5-14B-Instruct|qwen-lm|qwen2.5|Transformers|14b-instruct|1|Apache 2.0|29551688871",
    "7b|Qwen/Qwen2.5-7B-Instruct|qwen-lm|qwen2.5|Transformers|7b-instruct|1|Apache 2.0|15242807035",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if path.is_file():
        entry["sha256"] = sha256_file(path)
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def fetch_hf_json(model_repo: str, filename: str, *, timeout_seconds: float = 90.0) -> dict[str, Any]:
    url = f"https://huggingface.co/{model_repo}/resolve/main/{filename}"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        loaded = json.load(response)
    return loaded if isinstance(loaded, dict) else {}


def parse_candidate(raw: str) -> dict[str, Any]:
    parts = [part.strip() for part in str(raw or "").split("|")]
    if len(parts) != 9 or not all(parts[:7]):
        raise SystemExit(
            "--candidate must be parameter_class|hf_repo|owner|model|framework|instance|version|license|bytes"
        )
    return {
        "parameter_class": parts[0],
        "hf_repo": parts[1],
        "owner_slug": parts[2],
        "model_slug": parts[3],
        "framework": parts[4],
        "instance_slug": parts[5],
        "version_number": _int(parts[6], 1),
        "license_name": parts[7],
        "total_uncompressed_bytes": _int(parts[8]),
    }


def parameter_class_value(value: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", str(value or "").lower())
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)", str(value or ""))
    return float(match.group(1)) if match else 0.0


def attached_runtime_path(candidate: dict[str, Any], *, input_root: str) -> str:
    return (
        f"{input_root.rstrip('/')}/"
        f"models/"
        f"{candidate['owner_slug']}/"
        f"{candidate['model_slug']}/"
        f"{str(candidate['framework']).lower()}/"
        f"{candidate['instance_slug']}/"
        f"{candidate['version_number']}"
    )


def legacy_attached_runtime_path(candidate: dict[str, Any], *, input_root: str) -> str:
    return (
        f"{input_root.rstrip('/')}/"
        f"{candidate['model_slug']}/"
        f"{str(candidate['framework']).lower()}/"
        f"{candidate['instance_slug']}/"
        f"{candidate['version_number']}"
    )


def kernel_model_source_ref(candidate: dict[str, Any]) -> str:
    return (
        f"{candidate['owner_slug']}/{candidate['model_slug']}/"
        f"{candidate['framework']}/{candidate['instance_slug']}/{candidate['version_number']}"
    )


def is_full_precision_dense(candidate: dict[str, Any], config: dict[str, Any]) -> bool:
    combined = " ".join(
        [
            str(candidate.get("framework") or ""),
            str(candidate.get("instance_slug") or ""),
            str(candidate.get("hf_repo") or ""),
            json.dumps(_dict(config.get("quantization_config")), sort_keys=True),
            json.dumps(_dict(config.get("compression_config")), sort_keys=True),
        ]
    ).lower()
    blocked_terms = ("awq", "gptq", "4bit", "8bit", "int4", "int8", "fp8", "bnb", "gguf", "quant")
    return str(candidate.get("framework")) == "Transformers" and not any(term in combined for term in blocked_terms)


def inspect_attached_path(path: Path) -> dict[str, Any]:
    config = path / "config.json"
    index = path / "model.safetensors.index.json"
    tokenizer_candidates = [
        path / "tokenizer.json",
        path / "tokenizer.model",
        path / "tokenizer_config.json",
    ]
    safetensor_files = sorted(item.name for item in path.glob("*.safetensors")) if path.is_dir() else []
    return {
        "path_present": path.is_dir(),
        "config_json_present": config.is_file(),
        "weight_index_present": index.is_file(),
        "tokenizer_file_present": any(item.is_file() for item in tokenizer_candidates),
        "safetensors_file_count": len(safetensor_files),
        "safetensors_file_digest": stable_hash(safetensor_files),
        "local_metadata_read_ready": bool(config.is_file() and index.is_file()),
        "weight_tensor_values_public": False,
    }


def summarize_hf_metadata(candidate: dict[str, Any], *, timeout_seconds: float, fetch: bool) -> dict[str, Any]:
    if not fetch:
        return {
            "attempted": False,
            "metadata_only": True,
            "config_ready": False,
            "weight_index_ready": False,
            "weight_tensor_downloaded": False,
            "weight_tensor_values_public": False,
        }
    try:
        config = fetch_hf_json(str(candidate["hf_repo"]), "config.json", timeout_seconds=timeout_seconds)
        index = fetch_hf_json(str(candidate["hf_repo"]), "model.safetensors.index.json", timeout_seconds=timeout_seconds)
        weight_map = _dict(index.get("weight_map"))
        files = sorted({Path(str(value)).name for value in weight_map.values() if str(value or "").strip()})
        return {
            "attempted": True,
            "metadata_only": True,
            "config_ready": bool(config),
            "weight_index_ready": bool(weight_map),
            "model_type": str(config.get("model_type") or ""),
            "architectures": list(config.get("architectures") or []),
            "num_hidden_layers": _int(config.get("num_hidden_layers") or config.get("n_layer")),
            "hidden_size": _int(config.get("hidden_size") or config.get("n_embd")),
            "intermediate_size": _int(config.get("intermediate_size")),
            "num_attention_heads": _int(config.get("num_attention_heads") or config.get("n_head")),
            "num_key_value_heads": _int(config.get("num_key_value_heads")),
            "vocab_size": _int(config.get("vocab_size")),
            "torch_dtype": str(config.get("torch_dtype") or ""),
            "weight_key_count": len(weight_map),
            "weight_file_count": len(files),
            "weight_file_digest": stable_hash(files),
            "total_size_bytes": _int(_dict(index.get("metadata")).get("total_size")),
            "full_precision_dense_metadata": is_full_precision_dense(candidate, config),
            "weight_tensor_downloaded": False,
            "weight_tensor_values_public": False,
        }
    except Exception as exc:
        return {
            "attempted": True,
            "metadata_only": True,
            "config_ready": False,
            "weight_index_ready": False,
            "error_type": type(exc).__name__,
            "error_digest": stable_hash(str(exc)),
            "weight_tensor_downloaded": False,
            "weight_tensor_values_public": False,
        }


def build_candidate_summary(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    primary_path = Path(attached_runtime_path(candidate, input_root=args.kaggle_input_root))
    legacy_path = Path(legacy_attached_runtime_path(candidate, input_root=args.kaggle_input_root))
    local = inspect_attached_path(primary_path)
    legacy_local = inspect_attached_path(legacy_path)
    selected_path = primary_path if local["path_present"] or not legacy_local["path_present"] else legacy_path
    selected_local = local if selected_path == primary_path else legacy_local
    hf = summarize_hf_metadata(candidate, timeout_seconds=args.hf_timeout_seconds, fetch=args.fetch_hf_metadata)
    metadata_ready = bool(selected_local["local_metadata_read_ready"] or (hf.get("config_ready") and hf.get("weight_index_ready")))
    full_precision_dense = bool(
        (hf.get("full_precision_dense_metadata") is True)
        or (
            not hf.get("attempted")
            and str(candidate.get("framework")) == "Transformers"
            and not any(term in str(candidate.get("instance_slug") or "").lower() for term in ("awq", "gptq", "4bit", "8bit", "fp8", "bnb"))
        )
    )
    return {
        "schema": "kaggle_dense_model_source_candidate_v1",
        "parameter_class": candidate["parameter_class"],
        "parameter_class_value_b": parameter_class_value(str(candidate["parameter_class"])),
        "hf_repo": candidate["hf_repo"],
        "owner_slug": candidate["owner_slug"],
        "model_slug": candidate["model_slug"],
        "framework": candidate["framework"],
        "instance_slug": candidate["instance_slug"],
        "version_number": candidate["version_number"],
        "license_name": candidate["license_name"],
        "total_uncompressed_bytes": candidate["total_uncompressed_bytes"],
        "total_uncompressed_gb": round(_int(candidate["total_uncompressed_bytes"]) / 1024 / 1024 / 1024, 6),
        "kaggle_model_url": f"https://www.kaggle.com/models/{candidate['owner_slug']}/{candidate['model_slug']}/{candidate['framework']}/{candidate['instance_slug']}",
        "kaggle_kernel_model_source": kernel_model_source_ref(candidate),
        "attached_runtime_path": str(primary_path),
        "legacy_attached_runtime_path": str(legacy_path),
        "resolved_attached_runtime_path": str(selected_path),
        "attach_expected": True,
        "attach_path_present": bool(selected_local["path_present"]),
        "attach_can_avoid_runtime_download": True,
        "kaggle_model_attach_used_in_current_environment": bool(selected_local["path_present"]),
        "local_attached_path": selected_local,
        "primary_local_attached_path": local,
        "legacy_local_attached_path": legacy_local,
        "hf_metadata_fallback": hf,
        "metadata_ready": metadata_ready,
        "full_precision_dense_candidate": full_precision_dense,
        "safetensors_expected": True,
        "runtime_disk_download_required": False if local["path_present"] else None,
        "weight_tensor_values_public": False,
        "public_artifact_safe": True,
        "blockers": [] if metadata_ready and full_precision_dense else [
            *([] if metadata_ready else ["dense_model_source_metadata_not_verified"]),
            *([] if full_precision_dense else ["candidate_not_full_precision_dense_transformers"]),
        ],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [build_candidate_summary(parse_candidate(raw), args) for raw in (args.candidate or list(DEFAULT_CANDIDATES))]
    dense_candidates = [item for item in candidates if item.get("full_precision_dense_candidate") is True]
    ready = bool(candidates and all(item.get("metadata_ready") is True for item in candidates) and all(item.get("full_precision_dense_candidate") is True for item in candidates))
    largest = max(dense_candidates, key=lambda item: float(item.get("parameter_class_value_b") or 0), default={})
    attached_present = [item for item in candidates if item.get("attach_path_present") is True]
    blockers = sorted({blocker for item in candidates for blocker in (item.get("blockers") or []) if blocker})
    if not attached_present:
        blockers.append("kaggle_model_attach_paths_not_present_in_current_runtime")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "kaggle_dense_model_source_resolver_ready": ready,
        "fetch_hf_metadata": bool(args.fetch_hf_metadata),
        "kaggle_input_root": args.kaggle_input_root,
        "candidate_count": len(candidates),
        "dense_candidate_count": len(dense_candidates),
        "largest_dense_attach_candidate": {
            "parameter_class": largest.get("parameter_class", ""),
            "hf_repo": largest.get("hf_repo", ""),
            "kaggle_kernel_model_source": largest.get("kaggle_kernel_model_source", ""),
            "attached_runtime_path": largest.get("attached_runtime_path", ""),
        },
        "kaggle_model_attach_available": bool(candidates),
        "kaggle_model_attach_used": bool(attached_present),
        "kaggle_model_attach_current_runtime_count": len(attached_present),
        "kernel_model_sources": [item.get("kaggle_kernel_model_source") for item in candidates],
        "candidates": candidates,
        "blockers": blockers,
        "diagnosis_codes": [
            "kaggle_dense_model_source_resolver_ready" if ready else "kaggle_dense_model_source_resolver_blocked",
            "kaggle_model_attach_expected",
            "kaggle_model_attach_paths_present" if attached_present else "kaggle_model_attach_paths_not_present_in_current_runtime",
            "hf_metadata_fallback_used" if args.fetch_hf_metadata else "hf_metadata_fallback_not_used",
        ],
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "past_key_values_public": False,
            "credentials_public": False,
            "cookies_public": False,
            "private_runtime_state_public": False,
            "weight_tensor_values_public": False,
        },
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["kaggle_dense_model_source_resolver_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"].append("public_redaction_scan_failed")
        report["diagnosis_codes"].append("public_redaction_scan_failed")
        report["redaction_errors"] = leaks
    summary_path = output_dir / "kaggle_dense_model_source_resolver.json"
    write_json(summary_path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
    }
    write_json(summary_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve dense Qwen Kaggle model attach sources.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--kaggle-input-root", default="/kaggle/input")
    parser.add_argument("--fetch-hf-metadata", action="store_true")
    parser.add_argument("--hf-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.hf_timeout_seconds < 1 or args.hf_timeout_seconds > 600:
        raise SystemExit("--hf-timeout-seconds must be between 1 and 600")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'kaggle_dense_model_source_resolver.json'}")
        print(f"Ready: {report.get('kaggle_dense_model_source_resolver_ready')}")
        print(f"Kaggle attach used in current runtime: {report.get('kaggle_model_attach_used')}")
        if report.get("blockers"):
            print("Blockers: " + ", ".join(str(item) for item in report.get("blockers") or []))
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
