#!/usr/bin/env python3
"""Resolve public-safe DeepSeek-V4-Flash quantized source/runtime metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "deepseek_v4_flash_quantized_source_resolver_v1"
DEFAULT_OUTPUT_DIR = "dist/deepseek-v4-flash-quantized-source-resolver"
MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
TOTAL_PARAMS_B = 284.0
ACTIVE_PARAMS_B = 13.0
DEFAULT_CANDIDATES = (
    {
        "candidate_id": "iq1-s-xl-gguf",
        "repo": "teamblobfish/DeepSeek-V4-Flash-GGUF",
        "quant": "IQ1_S-XL",
        "path_prefix": "IQ1_S-XL/",
        "runtime_backend": "llama_cpp_v4_fork",
        "runtime_fork": "cchuter/llama.cpp@feat/v4-port-cuda",
        "expected_min_size_gb": 55.0,
        "priority": 1,
    },
    {
        "candidate_id": "iq1-m-gguf",
        "repo": "teamblobfish/DeepSeek-V4-Flash-GGUF",
        "quant": "IQ1_M",
        "path_prefix": "IQ1_M/",
        "runtime_backend": "llama_cpp_v4_fork",
        "runtime_fork": "cchuter/llama.cpp@feat/v4-port-cuda",
        "expected_min_size_gb": 58.0,
        "priority": 2,
    },
    {
        "candidate_id": "q2-k-single-gguf",
        "repo": "Preyazz/DeepSeek-V4-Flash-GGUF",
        "quant": "Q2_K",
        "path_prefix": "",
        "runtime_backend": "llama_cpp_v4_fork",
        "runtime_fork": "nisparks/llama.cpp@wip/deepseek-v4-support",
        "expected_min_size_gb": 95.0,
        "priority": 3,
    },
    {
        "candidate_id": "native-fp4-fp8-gguf",
        "repo": "nsparks/DeepSeek-V4-Flash-FP4-FP8-GGUF",
        "quant": "F8_E4M3_MXFP4",
        "file_contains": "fp4-fp8-native",
        "path_prefix": "",
        "runtime_backend": "llama_cpp_v4_fork",
        "runtime_fork": "nisparks/llama.cpp@wip/deepseek-v4-support",
        "expected_min_size_gb": 145.0,
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
    '"prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"activation":',
    '"hidden_state":',
    '"kv_cache":',
    '"past_key_values":',
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


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def fetch_json_url(url: str, *, timeout_seconds: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        return json.load(response)


def fetch_text_url(url: str, *, timeout_seconds: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
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


def hf_readme(repo: str, *, timeout_seconds: float) -> str:
    try:
        return fetch_text_url(f"https://huggingface.co/{repo}/raw/main/README.md", timeout_seconds=timeout_seconds)
    except Exception:
        return ""


def file_size(item: dict[str, Any]) -> int:
    lfs = _dict(item.get("lfs"))
    try:
        return int(lfs.get("size") or item.get("size") or 0)
    except (TypeError, ValueError):
        return 0


def candidate_files(tree: list[dict[str, Any]], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = str(candidate.get("path_prefix") or "")
    quant = str(candidate.get("file_contains") or candidate.get("quant") or "").lower()
    values: list[dict[str, Any]] = []
    for item in tree:
        path = str(item.get("path") or "")
        if item.get("type") != "file" or not path.lower().endswith(".gguf"):
            continue
        if prefix and not path.startswith(prefix):
            continue
        if quant and quant not in path.lower():
            continue
        values.append({
            "path": path,
            "size_bytes": file_size(item),
            "size_gb": round(file_size(item) / 1_000_000_000, 6),
            "lfs_oid_hash": sha_payload(_dict(item.get("lfs")).get("oid") or item.get("oid") or ""),
        })
    return sorted(values, key=lambda item: str(item.get("path") or ""))


def runtime_notes(readme: str) -> dict[str, Any]:
    lower = readme.lower()
    upstream_blocked = any(
        phrase in lower
        for phrase in (
            "stock upstream `llama.cpp` cannot load",
            "stock upstream llama.cpp cannot load",
            "not yet in stable llama.cpp",
            "these quants don't load on upstream",
        )
    )
    wip = "wip" in lower or "active wip" in lower or "work in progress" in lower
    cuda_fp8_gate = "__cuda_arch__ >= 890" in lower or "sm_120" in lower or "cuda testers wanted" in lower
    return {
        "requires_v4_aware_llama_cpp_fork": upstream_blocked,
        "runtime_branch_wip": wip,
        "t4_cuda_runtime_validated": False if cuda_fp8_gate or upstream_blocked else None,
        "readme_digest": sha_payload(readme[:12000]),
        "readme_public_excerpt_included": False,
    }


def build_candidate(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    repo = str(candidate["repo"])
    try:
        meta = hf_model_api(repo, timeout_seconds=float(args.hf_timeout_seconds))
        tree = hf_tree_api(repo, timeout_seconds=float(args.hf_timeout_seconds))
        readme = hf_readme(repo, timeout_seconds=float(args.hf_timeout_seconds))
        metadata_ready = bool(meta and tree)
        api_error = {}
    except Exception as exc:
        meta = {}
        tree = []
        readme = ""
        metadata_ready = False
        api_error = {"error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))}
    files = candidate_files(tree, candidate)
    total_size_bytes = sum(int(item.get("size_bytes") or 0) for item in files)
    notes = runtime_notes(readme)
    blockers: list[str] = []
    if not metadata_ready:
        blockers.append("hf_quantized_source_metadata_unavailable")
    if not files:
        blockers.append("gguf_quantized_files_not_resolved")
    if notes.get("requires_v4_aware_llama_cpp_fork"):
        blockers.append("stock_llama_cpp_cannot_load_deepseek_v4_flash")
    if notes.get("runtime_branch_wip"):
        blockers.append("deepseek_v4_flash_llama_cpp_runtime_wip")
    if notes.get("t4_cuda_runtime_validated") is False:
        blockers.append("t4_cuda_runtime_not_validated_for_deepseek_v4_flash")
    if total_size_bytes > int(args.runtime_download_budget_gb * 1_000_000_000):
        blockers.append("candidate_exceeds_runtime_download_budget")
    if total_size_bytes > int(args.single_t4x2_fit_budget_gb * 1_000_000_000):
        blockers.append("candidate_exceeds_single_t4x2_memory_budget")
    return {
        "schema": "deepseek_v4_flash_quantized_candidate_v1",
        "candidate_id": candidate["candidate_id"],
        "priority": int(candidate["priority"]),
        "repo": repo,
        "model_id": MODEL_ID,
        "architecture_class": "moe",
        "total_params_b": TOTAL_PARAMS_B,
        "active_params_b": ACTIVE_PARAMS_B,
        "quant": candidate["quant"],
        "format": "gguf",
        "runtime_backend": candidate["runtime_backend"],
        "runtime_fork": candidate["runtime_fork"],
        "hf_metadata_ready": metadata_ready,
        "hf_private": meta.get("private") is True,
        "hf_gated": bool(meta.get("gated")) and str(meta.get("gated")).lower() != "false",
        "license": meta.get("license") or next((str(tag).split("license:", 1)[1] for tag in _list(meta.get("tags")) if str(tag).startswith("license:")), ""),
        "downloads": meta.get("downloads"),
        "tags": [str(tag) for tag in _list(meta.get("tags"))[:24]],
        "files": files,
        "split_file_count": len(files),
        "total_size_bytes": total_size_bytes,
        "total_size_gb": round(total_size_bytes / 1_000_000_000, 6),
        "runtime_download_budget_gb": float(args.runtime_download_budget_gb),
        "single_t4x2_fit_budget_gb": float(args.single_t4x2_fit_budget_gb),
        "runtime_notes": notes,
        "hf_api_error": api_error,
        "public_artifact_safe": True,
        "blockers": sorted(set(blockers)),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [build_candidate(dict(item), args) for item in DEFAULT_CANDIDATES]
    candidates = sorted(candidates, key=lambda item: int(item.get("priority") or 999))
    ready = [item for item in candidates if item.get("hf_metadata_ready") is True and item.get("files")]
    smallest = min(ready, key=lambda item: float(item.get("total_size_gb") or 1e9), default={})
    recommended = next(
        (
            item for item in candidates
            if item.get("hf_metadata_ready") is True
            and item.get("files")
            and "candidate_exceeds_runtime_download_budget" not in item.get("blockers", [])
        ),
        smallest,
    )
    blockers = sorted({str(blocker) for item in candidates for blocker in _list(item.get("blockers")) if blocker})
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(ready),
        "deepseek_v4_flash_quantized_source_resolver_ready": bool(ready),
        "model": {
            "model_id": MODEL_ID,
            "architecture_class": "moe",
            "total_params_b": TOTAL_PARAMS_B,
            "active_params_b": ACTIVE_PARAMS_B,
            "quantized_goal": True,
        },
        "candidate_count": len(candidates),
        "ready_candidate_count": len(ready),
        "smallest_ready_candidate": {
            "candidate_id": smallest.get("candidate_id", ""),
            "repo": smallest.get("repo", ""),
            "quant": smallest.get("quant", ""),
            "total_size_gb": smallest.get("total_size_gb", 0),
            "split_file_count": smallest.get("split_file_count", 0),
        },
        "recommended_live_probe_candidate": {
            "candidate_id": recommended.get("candidate_id", ""),
            "repo": recommended.get("repo", ""),
            "quant": recommended.get("quant", ""),
            "runtime_backend": recommended.get("runtime_backend", ""),
            "runtime_fork": recommended.get("runtime_fork", ""),
            "total_size_gb": recommended.get("total_size_gb", 0),
            "split_file_count": recommended.get("split_file_count", 0),
            "files": recommended.get("files", []),
            "blockers": recommended.get("blockers", []),
        },
        "candidates": candidates,
        "blockers": blockers,
        "diagnosis_codes": [
            "deepseek_v4_flash_quantized_sources_ready" if ready else "deepseek_v4_flash_quantized_sources_not_ready",
            "deepseek_v4_flash_quantized_runtime_requires_v4_aware_llama_cpp_fork",
            "deepseek_v4_flash_quantized_same_request_decode_not_verified_by_resolver",
        ],
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
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
        report["deepseek_v4_flash_quantized_source_resolver_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["redaction_errors"] = leaks
        report["blockers"].append("public_redaction_scan_failed")
    path = output_dir / "deepseek_v4_flash_quantized_source_resolver.json"
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
    parser.add_argument("--runtime-download-budget-gb", type=float, default=80.0)
    parser.add_argument("--single-t4x2-fit-budget-gb", type=float, default=28.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.hf_timeout_seconds < 1 or args.hf_timeout_seconds > 600:
        raise SystemExit("--hf-timeout-seconds must be between 1 and 600")
    if args.runtime_download_budget_gb <= 0:
        raise SystemExit("--runtime-download-budget-gb must be positive")
    if args.single_t4x2_fit_budget_gb <= 0:
        raise SystemExit("--single-t4x2-fit-budget-gb must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'deepseek_v4_flash_quantized_source_resolver.json'}")
        print(f"Ready: {report.get('deepseek_v4_flash_quantized_source_resolver_ready')}")
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
