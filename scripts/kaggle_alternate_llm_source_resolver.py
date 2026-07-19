#!/usr/bin/env python3
"""Resolve public-safe Kaggle source metadata for alternate large LLM candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "kaggle_alternate_llm_source_resolver_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-alternate-llm-source-resolver"
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    '"prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"activation":',
    '"hidden_state":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
)

# priority|parameter_class|hf_repo|owner|model|framework|instance|version|license|bytes|architecture_class|total_params_b|active_params_b|notes
DEFAULT_CANDIDATES = (
    "1|405b|meta-llama/Meta-Llama-3.1-405B|metaresearch|llama-3.1|Transformers|405b|1|Llama 3.1 Community License|820171909694|dense|405|405|license_agreement_required",
    "1|405b-instruct|meta-llama/Meta-Llama-3.1-405B-Instruct|metaresearch|llama-3.1|Transformers|405b-instruct|1|Llama 3.1 Community License|820171929529|dense|405|405|license_agreement_required",
    "2|235b-a22b|Qwen/Qwen3-235B-A22B|qwen-lm|qwen-3|Transformers|235b-a22b|1|Apache 2.0|470211106181|moe|235|22|qwen3_moe_adapter_required",
    "3|671b-v3|deepseek-ai/DeepSeek-V3|deepseek-ai|deepseek-v3|Transformers|deepseek-v3|2|Other (specified in description)|688603933219|moe|671|37|deepseek_mla_moe_adapter_required",
    "3|671b-r1|deepseek-ai/DeepSeek-R1|deepseek-ai|deepseek-r1|Transformers|deepseek-r1|2|MIT|688604358477|moe|671|37|deepseek_mla_moe_adapter_required",
    "4|80b-a3b|Qwen/Qwen3-Next-80B-A3B-Instruct|qwen-lm|qwen3-next-80b|Transformers|qwen3-next-80b-a3b-instruct|1|Apache 2.0|162682272799|hybrid_moe|80|3|qwen3_next_hybrid_adapter_required",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha_payload(value: Any) -> str:
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


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
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
    if len(parts) != 14 or not all(parts[:10]):
        raise SystemExit(
            "--candidate must be priority|parameter_class|hf_repo|owner|model|framework|instance|version|license|bytes|architecture_class|total_params_b|active_params_b|notes"
        )
    return {
        "priority": _int(parts[0], 999),
        "parameter_class": parts[1],
        "hf_repo": parts[2],
        "owner_slug": parts[3],
        "model_slug": parts[4],
        "framework": parts[5],
        "instance_slug": parts[6],
        "version_number": _int(parts[7], 1),
        "license_name": parts[8],
        "total_uncompressed_bytes": _int(parts[9]),
        "architecture_class": parts[10],
        "total_params_b": _float(parts[11]),
        "active_params_b": _float(parts[12]),
        "notes": [item for item in re.split(r"[,;]", parts[13]) if item],
    }


def kernel_model_source_ref(candidate: dict[str, Any]) -> str:
    return (
        f"{candidate['owner_slug']}/{candidate['model_slug']}/"
        f"{candidate['framework']}/{candidate['instance_slug']}/{candidate['version_number']}"
    )


def kaggle_model_url(candidate: dict[str, Any]) -> str:
    return (
        f"https://www.kaggle.com/models/{candidate['owner_slug']}/{candidate['model_slug']}/"
        f"{candidate['framework']}/{candidate['instance_slug']}"
    )


def expected_paths(candidate: dict[str, Any], *, input_root: str) -> list[str]:
    root = input_root.rstrip("/")
    owner = candidate["owner_slug"]
    model = candidate["model_slug"]
    framework = str(candidate["framework"]).lower()
    instance = candidate["instance_slug"]
    version = str(candidate["version_number"])
    values = [
        f"{root}/models/{owner}/{model}/{framework}/{instance}/{version}",
        f"{root}/models/{owner}/{model}/{framework}/{instance}",
        f"{root}/{owner}/{model}/{framework}/{instance}/{version}",
        f"{root}/{owner}/{model}/{framework}/{instance}",
        f"{root}/{model}/{framework}/{instance}/{version}",
        f"{root}/{model}/{framework}/{instance}",
    ]
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def inspect_path(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "path_present": path.is_dir(),
        "config_json_present": (path / "config.json").is_file(),
        "weight_index_present": (path / "model.safetensors.index.json").is_file(),
        "tokenizer_file_present": any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")),
        "safetensors_file_count": len(list(path.glob("*.safetensors"))) if path.is_dir() else 0,
    }


def precision_class(candidate: dict[str, Any], config: dict[str, Any]) -> str:
    combined = " ".join(
        [
            str(candidate.get("framework") or ""),
            str(candidate.get("instance_slug") or ""),
            json.dumps(_dict(config.get("quantization_config")), sort_keys=True),
            json.dumps(_dict(config.get("compression_config")), sort_keys=True),
        ]
    ).lower()
    if "gguf" in combined:
        return "gguf_quantized"
    if "fp8" in combined:
        return "fp8"
    if any(term in combined for term in ("awq", "gptq", "4bit", "8bit", "int4", "int8", "bnb", "quant")):
        return "quantized"
    return "full_precision_or_bf16"


def summarize_hf(candidate: dict[str, Any], *, fetch: bool, timeout_seconds: float) -> dict[str, Any]:
    if not fetch:
        return {"attempted": False, "metadata_only": True}
    try:
        config = fetch_hf_json(candidate["hf_repo"], "config.json", timeout_seconds=timeout_seconds)
    except Exception as exc:
        return {
            "attempted": True,
            "metadata_only": True,
            "config_ready": False,
            "weight_index_ready": False,
            "error_type": type(exc).__name__,
            "error_digest": sha_payload(str(exc)),
            "weight_tensor_downloaded": False,
        }
    index: dict[str, Any] = {}
    index_error = ""
    try:
        index = fetch_hf_json(candidate["hf_repo"], "model.safetensors.index.json", timeout_seconds=timeout_seconds)
    except Exception as exc:
        index_error = type(exc).__name__
    weight_map = _dict(index.get("weight_map"))
    files = sorted({Path(str(value)).name for value in weight_map.values() if str(value or "").strip()})
    return {
        "attempted": True,
        "metadata_only": True,
        "config_ready": bool(config),
        "weight_index_ready": bool(weight_map),
        "weight_index_error_type": index_error,
        "model_type": str(config.get("model_type") or ""),
        "architectures": list(config.get("architectures") or []),
        "num_hidden_layers": _int(config.get("num_hidden_layers") or config.get("n_layer")),
        "hidden_size": _int(config.get("hidden_size") or config.get("n_embd")),
        "intermediate_size": _int(config.get("intermediate_size")),
        "num_attention_heads": _int(config.get("num_attention_heads") or config.get("n_head")),
        "num_key_value_heads": _int(config.get("num_key_value_heads")),
        "num_experts": _int(config.get("num_experts") or config.get("n_routed_experts") or config.get("n_shared_experts")),
        "num_experts_per_tok": _int(config.get("num_experts_per_tok") or config.get("num_experts_per_token")),
        "vocab_size": _int(config.get("vocab_size")),
        "torch_dtype": str(config.get("torch_dtype") or ""),
        "precision_class": precision_class(candidate, config),
        "weight_key_count": len(weight_map),
        "weight_file_count": len(files),
        "weight_file_digest": sha_payload(files),
        "total_size_bytes": _int(_dict(index.get("metadata")).get("total_size")),
        "weight_tensor_downloaded": False,
        "weight_tensor_values_public": False,
    }


def build_candidate(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    paths = expected_paths(candidate, input_root=args.kaggle_input_root)
    path_probes = [inspect_path(Path(path)) for path in paths]
    attach_present = any(item.get("path_present") for item in path_probes)
    local_metadata_ready = any(
        item.get("path_present") and item.get("config_json_present") and item.get("weight_index_present")
        for item in path_probes
    )
    hf = summarize_hf(candidate, fetch=bool(args.fetch_hf_metadata), timeout_seconds=float(args.hf_timeout_seconds))
    metadata_ready = bool(local_metadata_ready or (hf.get("config_ready") and hf.get("weight_index_ready")))
    precision = str(hf.get("precision_class") or "unknown_without_hf_metadata")
    architecture_class = str(candidate.get("architecture_class") or "")
    dense_full_precision = bool(architecture_class == "dense" and precision == "full_precision_or_bf16")
    moe = architecture_class in {"moe", "hybrid_moe"}
    license_requires_agreement = "agreement" in " ".join(str(item) for item in candidate.get("notes") or []).lower()
    adapter_status = "existing_llama_like_adapter_possible" if dense_full_precision else "adapter_required"
    if "qwen3" in str(hf.get("model_type") or "").lower() or "qwen3" in candidate["hf_repo"].lower():
        adapter_status = "qwen3_moe_or_hybrid_adapter_required"
    if "deepseek" in candidate["hf_repo"].lower():
        adapter_status = "deepseek_mla_moe_adapter_required"
    blockers: list[str] = []
    if not metadata_ready:
        blockers.append("model_source_metadata_not_verified")
    if license_requires_agreement:
        blockers.append("kaggle_model_license_agreement_required")
    if not attach_present:
        blockers.append("kaggle_model_attach_not_verified_in_current_runtime")
    if moe:
        blockers.append(adapter_status)
    if precision not in {"full_precision_or_bf16", "unknown_without_hf_metadata"}:
        blockers.append("candidate_not_full_precision_bf16")
    return {
        "schema": "kaggle_alternate_llm_source_candidate_v1",
        **candidate,
        "candidate_id": candidate["parameter_class"],
        "kaggle_ref": kernel_model_source_ref(candidate),
        "kaggle_owner": candidate["owner_slug"],
        "kaggle_model": candidate["model_slug"],
        "kaggle_framework": candidate["framework"],
        "kaggle_instance": candidate["instance_slug"],
        "kaggle_version": candidate["version_number"],
        "kaggle_model_url": kaggle_model_url(candidate),
        "kaggle_kernel_model_source": kernel_model_source_ref(candidate),
        "expected_attached_paths": paths,
        "local_attach_path_probes": path_probes,
        "attach_path_present_in_current_runtime": attach_present,
        "attach_can_avoid_runtime_download": True,
        "metadata_ready": metadata_ready,
        "local_metadata_ready": local_metadata_ready,
        "hf_metadata_fallback": hf,
        "parameter_count_b": float(candidate.get("total_params_b") or 0),
        "active_parameter_count_b": float(candidate.get("active_params_b") or 0),
        "total_size_bytes": int(hf.get("total_size_bytes") or candidate.get("total_uncompressed_bytes") or 0),
        "model_type": str(hf.get("model_type") or ""),
        "num_hidden_layers": int(hf.get("num_hidden_layers") or 0),
        "architecture_class": architecture_class,
        "precision_class": precision,
        "dense_full_precision_candidate": dense_full_precision,
        "moe_candidate": moe,
        "license_agreement_required": license_requires_agreement,
        "adapter_status": adapter_status,
        "runtime_disk_download_required": False if attach_present else None,
        "weight_tensor_values_public": False,
        "public_artifact_safe": True,
        "blockers": sorted(set(blockers)),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [build_candidate(parse_candidate(raw), args) for raw in (args.candidate or list(DEFAULT_CANDIDATES))]
    candidates = sorted(candidates, key=lambda item: (int(item.get("priority") or 999), -float(item.get("total_params_b") or 0)))
    metadata_ready = [item for item in candidates if item.get("metadata_ready") is True]
    stage_loader_candidates = [
        item for item in candidates
        if item.get("metadata_ready") is True
        and item.get("precision_class") == "full_precision_or_bf16"
        and not item.get("license_agreement_required")
    ]
    largest_dense = max(
        (item for item in candidates if item.get("dense_full_precision_candidate") is True),
        key=lambda item: float(item.get("total_params_b") or 0),
        default={},
    )
    largest_moe = max(
        (item for item in candidates if item.get("moe_candidate") is True and item.get("precision_class") == "full_precision_or_bf16"),
        key=lambda item: float(item.get("total_params_b") or 0),
        default={},
    )
    blockers = sorted({str(blocker) for item in candidates for blocker in (item.get("blockers") or []) if blocker})
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(metadata_ready),
        "kaggle_alternate_llm_source_resolver_ready": bool(metadata_ready),
        "fetch_hf_metadata": bool(args.fetch_hf_metadata),
        "kaggle_input_root": args.kaggle_input_root,
        "candidate_count": len(candidates),
        "metadata_ready_candidate_count": len(metadata_ready),
        "stage_loader_candidate_count": len(stage_loader_candidates),
        "kernel_model_sources": [item.get("kaggle_kernel_model_source") for item in candidates],
        "model_source_refs": [item.get("kaggle_kernel_model_source") for item in candidates],
        "largest_dense_full_precision_candidate": {
            "parameter_class": largest_dense.get("parameter_class", ""),
            "model_source": largest_dense.get("kaggle_kernel_model_source", ""),
            "total_params_b": largest_dense.get("total_params_b", 0),
            "license_agreement_required": largest_dense.get("license_agreement_required", False),
        },
        "largest_moe_full_precision_candidate": {
            "parameter_class": largest_moe.get("parameter_class", ""),
            "model_source": largest_moe.get("kaggle_kernel_model_source", ""),
            "total_params_b": largest_moe.get("total_params_b", 0),
            "active_params_b": largest_moe.get("active_params_b", 0),
        },
        "recommended_stage_loader_candidate": stage_loader_candidates[0] if stage_loader_candidates else {},
        "candidates": candidates,
        "blockers": blockers,
        "diagnosis_codes": [
            "kaggle_alternate_llm_source_resolver_ready" if metadata_ready else "kaggle_alternate_llm_source_resolver_blocked",
            "alternate_large_llm_candidates_scanned",
            "kaggle_model_attach_paths_present" if any(item.get("attach_path_present_in_current_runtime") for item in candidates) else "kaggle_model_attach_paths_not_present_in_current_runtime",
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
        report["kaggle_alternate_llm_source_resolver_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"].append("public_redaction_scan_failed")
        report["redaction_errors"] = leaks
    path = output_dir / "kaggle_alternate_llm_source_resolver.json"
    write_json(path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
    }
    write_json(path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        print(f"Report: {Path(args.output_dir) / 'kaggle_alternate_llm_source_resolver.json'}")
        print(f"Ready: {report.get('kaggle_alternate_llm_source_resolver_ready')}")
        if report.get("blockers"):
            print("Blockers: " + ", ".join(str(item) for item in report.get("blockers") or []))
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
