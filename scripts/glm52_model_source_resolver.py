#!/usr/bin/env python3
"""Resolve public-safe GLM 5.2 source and Kaggle deployment metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "glm52_model_source_resolver_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-model-source-resolver"
MODEL_ID = "zai-org/GLM-5.2"
COMPATIBLE_MODEL_IDS = {"zai-org/GLM-5.2"}
REQUIRED_STAGE_BACKENDS = ["kaggle_cuda", "kaggle_jax_tpu", "kaggle_cpu"]
DEFAULT_CANDIDATES = (
    {
        "candidate_id": "official-full-safetensors",
        "repo": "zai-org/GLM-5.2",
        "source_kind": "official_hf",
        "format": "safetensors",
        "quantization": "bf16_or_native",
        "priority": 1,
    },
    {
        "candidate_id": "awq-int4-safetensors",
        "repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "source_kind": "hf_quantized",
        "format": "safetensors",
        "quantization": "AWQ-INT4",
        "priority": 2,
    },
    {
        "candidate_id": "unsloth-gguf",
        "repo": "unsloth/GLM-5.2-GGUF",
        "source_kind": "hf_quantized",
        "format": "gguf",
        "quantization": "GGUF",
        "priority": 3,
    },
    {
        "candidate_id": "sokann-2244bpw-gguf",
        "repo": "sokann/GLM-5.2-GGUF-2.244bpw",
        "source_kind": "hf_quantized",
        "format": "gguf",
        "quantization": "2.244bpw_GGUF",
        "priority": 4,
    },
)
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "token=",
    "runtime_proxy",
    '"prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"activation":',
    '"hidden_state":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


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


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def fetch_json_url(url: str, *, timeout_seconds: float) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "crowdtensor-glm52-source-resolver/1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.load(response)


def fetch_text_url(url: str, *, timeout_seconds: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "crowdtensor-glm52-source-resolver/1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8", errors="replace")


def hf_model_api(repo: str, *, timeout_seconds: float) -> dict[str, Any]:
    loaded = fetch_json_url(f"https://huggingface.co/api/models/{repo}", timeout_seconds=timeout_seconds)
    return loaded if isinstance(loaded, dict) else {}


def hf_tree_api(repo: str, *, timeout_seconds: float) -> list[dict[str, Any]]:
    loaded = fetch_json_url(
        f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1&expand=true",
        timeout_seconds=timeout_seconds,
    )
    return [item for item in loaded if isinstance(item, dict)] if isinstance(loaded, list) else []


def hf_raw_json(repo: str, filename: str, *, timeout_seconds: float) -> dict[str, Any]:
    quoted = urllib.parse.quote(filename)
    loaded = fetch_json_url(f"https://huggingface.co/{repo}/resolve/main/{quoted}", timeout_seconds=timeout_seconds)
    return loaded if isinstance(loaded, dict) else {}


def hf_readme(repo: str, *, timeout_seconds: float) -> str:
    try:
        return fetch_text_url(f"https://huggingface.co/{repo}/raw/main/README.md", timeout_seconds=timeout_seconds)
    except Exception:
        return ""


def file_size(item: dict[str, Any]) -> int:
    lfs = _dict(item.get("lfs"))
    return _int(lfs.get("size") or item.get("size"))


def weight_files_from_meta_and_tree(meta: dict[str, Any], tree: list[dict[str, Any]], *, suffixes: tuple[str, ...]) -> list[dict[str, Any]]:
    tree_by_path = {str(item.get("path") or ""): item for item in tree if item.get("type") == "file"}
    names: list[str] = []
    for sibling in _list(meta.get("siblings")):
        path = str(_dict(sibling).get("rfilename") or "")
        if path.endswith(suffixes):
            names.append(path)
    if not names:
        names = [path for path in tree_by_path if path.endswith(suffixes)]
    files: list[dict[str, Any]] = []
    for path in sorted(set(names)):
        size = file_size(tree_by_path.get(path, {}))
        files.append(
            {
                "path": path,
                "size_bytes": size,
                "size_gb": round(size / 1_000_000_000, 6) if size else 0,
                "size_source": "hf_tree_expand" if size else "not_resolved_from_hf_tree_page",
            }
        )
    return files


def infer_license(meta: dict[str, Any]) -> str:
    for tag in _list(meta.get("tags")):
        text = str(tag)
        if text.startswith("license:"):
            return text.split(":", 1)[1]
    return str(meta.get("license") or "")


def build_stage_plan(config: dict[str, Any], index: dict[str, Any], *, stage_backends: list[str]) -> dict[str, Any]:
    layer_count = _int(config.get("num_hidden_layers"))
    stage_count = max(1, len(stage_backends))
    base = layer_count // stage_count if stage_count else layer_count
    remainder = layer_count % stage_count if stage_count else 0
    ranges: list[list[int]] = []
    start = 0
    for stage_id in range(stage_count):
        width = base + (1 if stage_id < remainder else 0)
        end = start + width
        ranges.append([start, end])
        start = end
    weight_map = _dict(index.get("weight_map"))
    stage_plans = []
    assigned_total = 0
    for stage_id, layer_range in enumerate(ranges):
        prefixes = []
        if stage_id == 0:
            prefixes.append("model.embed_tokens.")
        prefixes.extend(f"model.layers.{layer_id}." for layer_id in range(layer_range[0], layer_range[1]))
        if stage_id == stage_count - 1:
            prefixes.extend(["model.norm.", "lm_head."])
        keys = [key for key in weight_map if any(str(key).startswith(prefix) for prefix in prefixes)]
        assigned_total += len(keys)
        stage_plans.append(
            {
                "stage_id": stage_id,
                "backend": stage_backends[stage_id],
                "layer_range": layer_range,
                "assigned_key_count": len(keys),
                "assigned_file_count": len({str(weight_map.get(key) or "") for key in keys if weight_map.get(key)}),
                "key_digest": sha_payload(sorted(keys)[:200]),
                "metadata_only": True,
                "stage_runtime_adapter_verified": False,
            }
        )
    return {
        "schema": "glm52_kaggle_stage_adapter_plan_v1",
        "model_id": MODEL_ID,
        "stage_backends": stage_backends,
        "stage_count": stage_count,
        "layer_count": layer_count,
        "stage_plans": stage_plans,
        "assigned_key_count_total": assigned_total,
        "weight_key_count": len(weight_map),
        "metadata_only": True,
        "stage_runtime_adapter_verified": False,
        "same_request_route_verified": False,
        "public_artifact_safe": True,
    }


def build_candidate(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    repo = str(candidate["repo"])
    suffixes = (".safetensors",) if candidate["format"] == "safetensors" else (".gguf",)
    try:
        meta = hf_model_api(repo, timeout_seconds=float(args.hf_timeout_seconds))
        tree = hf_tree_api(repo, timeout_seconds=float(args.hf_timeout_seconds))
        index: dict[str, Any] = {}
        if candidate["format"] == "safetensors":
            try:
                index = hf_raw_json(repo, "model.safetensors.index.json", timeout_seconds=float(args.hf_timeout_seconds))
            except Exception:
                index = {}
        files = weight_files_from_meta_and_tree(meta, tree, suffixes=suffixes)
        metadata_ready = bool(meta)
        api_error = {}
    except Exception as exc:
        meta = {}
        tree = []
        index = {}
        files = []
        metadata_ready = False
        api_error = {"error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))}

    index_total_size = _int(_dict(index.get("metadata")).get("total_size"))
    known_total_size = index_total_size or sum(_int(item.get("size_bytes")) for item in files)
    resolved_file_sizes = sum(1 for item in files if _int(item.get("size_bytes")) > 0)
    siblings_count = len(_list(meta.get("siblings")))
    blockers: list[str] = []
    if not metadata_ready:
        blockers.append("hf_model_metadata_unavailable")
    if bool(meta.get("private")):
        blockers.append("hf_model_private")
    if bool(meta.get("gated")) and str(meta.get("gated")).lower() != "false":
        blockers.append("hf_model_gated")
    if not files:
        blockers.append("weight_files_not_resolved")
    if not index_total_size and resolved_file_sizes < len(files):
        blockers.append("complete_weight_size_not_resolved")
    if known_total_size > int(args.runtime_disk_budget_gb * 1_000_000_000):
        blockers.append("candidate_exceeds_runtime_disk_budget")
    if known_total_size > int(args.single_kaggle_account_memory_budget_gb * 1_000_000_000):
        blockers.append("candidate_exceeds_single_account_memory_budget")
    if candidate["format"] == "gguf":
        blockers.append("gguf_runtime_adapter_not_verified_for_glm52")
    if candidate["format"] == "safetensors" and candidate["source_kind"] == "hf_quantized":
        blockers.append("quantized_transformers_runtime_adapter_not_verified_for_glm52")

    return {
        "schema": "glm52_source_candidate_v1",
        "candidate_id": str(candidate["candidate_id"]),
        "priority": int(candidate["priority"]),
        "repo": repo,
        "source_kind": str(candidate["source_kind"]),
        "format": str(candidate["format"]),
        "quantization": str(candidate["quantization"]),
        "base_model_verified": repo == MODEL_ID or f"base_model:{MODEL_ID}" in {str(tag) for tag in _list(meta.get("tags"))},
        "hf_metadata_ready": metadata_ready,
        "hf_private": meta.get("private") is True,
        "hf_gated": bool(meta.get("gated")) and str(meta.get("gated")).lower() != "false",
        "library_name": str(meta.get("library_name") or ""),
        "pipeline_tag": str(meta.get("pipeline_tag") or ""),
        "license": infer_license(meta),
        "downloads": meta.get("downloads"),
        "likes": meta.get("likes"),
        "tags": [str(tag) for tag in _list(meta.get("tags"))[:32]],
        "sibling_count": siblings_count,
        "weight_file_count": len(files),
        "resolved_weight_file_size_count": resolved_file_sizes,
        "known_total_size_bytes": known_total_size,
        "known_total_size_gb": round(known_total_size / 1_000_000_000, 6) if known_total_size else 0,
        "index_total_size_bytes": index_total_size,
        "files_sample": files[:24],
        "hf_api_error": api_error,
        "public_artifact_safe": True,
        "blockers": sorted(set(blockers)),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {}
    index: dict[str, Any] = {}
    readme = ""
    full_source_error: dict[str, Any] = {}
    try:
        config = hf_raw_json(MODEL_ID, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
        index = hf_raw_json(MODEL_ID, "model.safetensors.index.json", timeout_seconds=float(args.hf_timeout_seconds))
        readme = hf_readme(MODEL_ID, timeout_seconds=float(args.hf_timeout_seconds))
    except Exception as exc:
        full_source_error = {"error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))}

    candidates = [build_candidate(dict(item), args) for item in DEFAULT_CANDIDATES]
    candidates = sorted(candidates, key=lambda item: int(item.get("priority") or 999))
    ready_candidates = [
        item for item in candidates
        if item.get("hf_metadata_ready") is True and item.get("hf_private") is not True and item.get("hf_gated") is not True
    ]
    deploy_candidates = [
        item for item in ready_candidates
        if "candidate_exceeds_runtime_disk_budget" not in _list(item.get("blockers"))
        and "complete_weight_size_not_resolved" not in _list(item.get("blockers"))
    ]
    quantized_ready = [item for item in ready_candidates if item.get("source_kind") == "hf_quantized"]
    recommended_pool = deploy_candidates or quantized_ready or ready_candidates
    recommended = min(
        recommended_pool,
        key=lambda item: (
            item.get("source_kind") != "hf_quantized",
            item.get("format") != "safetensors",
            _int(item.get("known_total_size_bytes")) <= 0,
            _int(item.get("known_total_size_bytes")) or 10**18,
            int(item.get("priority") or 999),
        ),
        default={},
    )
    stage_backends = [item.strip() for item in str(args.stage_backends).split(",") if item.strip()]
    stage_plan = build_stage_plan(config, index, stage_backends=stage_backends)
    full_weight_size = _int(_dict(index.get("metadata")).get("total_size"))
    source_ready = bool(config and index and ready_candidates)
    blockers = sorted({str(blocker) for item in candidates for blocker in _list(item.get("blockers")) if blocker})
    if not source_ready:
        blockers.append("glm52_official_source_metadata_not_ready")
    if full_weight_size > int(args.runtime_disk_budget_gb * 1_000_000_000):
        blockers.append("glm52_full_weights_exceed_kaggle_runtime_disk_budget")
    if stage_plan.get("stage_runtime_adapter_verified") is not True:
        blockers.append("glm52_stage_runtime_adapter_not_verified")
    if stage_plan.get("same_request_route_verified") is not True:
        blockers.append("glm52_same_request_route_not_verified")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": source_ready,
        "glm52_source_resolver_ready": source_ready,
        "model": {
            "model_id": MODEL_ID,
            "compatible_model_ids": sorted(COMPATIBLE_MODEL_IDS),
            "architecture_class": "moe",
            "model_type": str(config.get("model_type") or ""),
            "architectures": [str(item) for item in _list(config.get("architectures"))],
            "num_hidden_layers": _int(config.get("num_hidden_layers")),
            "hidden_size": _int(config.get("hidden_size")),
            "num_attention_heads": _int(config.get("num_attention_heads")),
            "num_key_value_heads": _int(config.get("num_key_value_heads")),
            "intermediate_size": _int(config.get("intermediate_size")),
            "n_routed_experts": _int(config.get("n_routed_experts")),
            "moe_intermediate_size": _int(config.get("moe_intermediate_size")),
            "official_weight_key_count": len(_dict(index.get("weight_map"))),
            "official_weight_total_size_bytes": full_weight_size,
            "official_weight_total_size_gb": round(full_weight_size / 1_000_000_000, 6) if full_weight_size else 0,
            "readme_digest": sha_payload(readme[:12000]),
            "readme_public_excerpt_included": False,
            "full_source_error": full_source_error,
        },
        "candidate_count": len(candidates),
        "ready_candidate_count": len(ready_candidates),
        "recommended_deployment_candidate": {
            "candidate_id": str(recommended.get("candidate_id") or ""),
            "repo": str(recommended.get("repo") or ""),
            "format": str(recommended.get("format") or ""),
            "quantization": str(recommended.get("quantization") or ""),
            "known_total_size_gb": recommended.get("known_total_size_gb", 0),
            "weight_file_count": recommended.get("weight_file_count", 0),
            "blockers": [str(item) for item in _list(recommended.get("blockers"))],
        },
        "candidates": candidates,
        "stage_adapter_plan": stage_plan,
        "kaggle_attach_plan": {
            "schema": "glm52_kaggle_attach_plan_v1",
            "kaggle_models_source_verified": False,
            "hf_source_verified": source_ready,
            "preferred_strategy": "attach_kaggle_model_or_dataset_if_available_else_stage_selective_hf_range_loader",
            "full_runtime_download_supported": full_weight_size <= int(args.runtime_disk_budget_gb * 1_000_000_000) if full_weight_size else False,
            "blockers": [
                "kaggle_models_glm52_source_not_verified",
                "full_runtime_download_exceeds_budget" if full_weight_size > int(args.runtime_disk_budget_gb * 1_000_000_000) else "",
            ],
            "public_artifact_safe": True,
        },
        "blockers": sorted({item for item in blockers if item}),
        "diagnosis_codes": [
            "glm52_public_hf_source_ready" if source_ready else "glm52_public_hf_source_not_ready",
            "glm52_full_weight_size_resolved" if full_weight_size else "glm52_full_weight_size_not_resolved",
            "glm52_kaggle_same_request_not_verified_by_source_resolver",
        ],
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
        },
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["glm52_source_resolver_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["redaction_errors"] = leaks
        report["blockers"].append("public_redaction_scan_failed")
    path = output_dir / "glm52_model_source_resolver.json"
    write_json(path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
    }
    write_json(path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--runtime-disk-budget-gb", type=float, default=120.0)
    parser.add_argument("--single-kaggle-account-memory-budget-gb", type=float, default=96.0)
    parser.add_argument("--stage-backends", default="kaggle_cuda,kaggle_jax_tpu,kaggle_cpu")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.hf_timeout_seconds < 1 or args.hf_timeout_seconds > 600:
        raise SystemExit("--hf-timeout-seconds must be between 1 and 600")
    if args.runtime_disk_budget_gb <= 0:
        raise SystemExit("--runtime-disk-budget-gb must be positive")
    if args.single_kaggle_account_memory_budget_gb <= 0:
        raise SystemExit("--single-kaggle-account-memory-budget-gb must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'glm52_model_source_resolver.json'}")
        print(f"Ready: {report.get('glm52_source_resolver_ready')}")
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
