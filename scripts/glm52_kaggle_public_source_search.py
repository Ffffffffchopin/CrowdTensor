#!/usr/bin/env python3
"""Search public Kaggle Models/Datasets for GLM 5.2 attachable sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_gpu_token_weekly_quota_probe as token_probe  # noqa: E402


SCHEMA = "glm52_kaggle_public_source_search_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-kaggle-public-source-search"
MODEL_ID = "zai-org/GLM-5.2"
DEFAULT_QUERIES = (
    "GLM-5.2",
    "GLM 5.2",
    "zai-org GLM-5.2",
    "cyankiwi GLM-5.2",
    "GLM-5",
    "GLM 5",
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
    "kaggle-cookies",
    "kaggle-web-storage-state",
    "token=",
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


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def text_has_glm52(text: str) -> bool:
    normalized = normalize(text)
    return "glm52" in normalized or "glm520" in normalized


def text_has_weight_signal(text: str) -> bool:
    normalized = normalize(text)
    signals = (
        "safetensors",
        "transformers",
        "pytorch",
        "awq",
        "gguf",
        "modelweights",
        "modelweight",
        "llm",
        "causallm",
    )
    return any(signal in normalized for signal in signals)


def text_has_dataset_weight_signal(text: str) -> bool:
    normalized = normalize(text)
    signals = (
        "safetensors",
        "modelweights",
        "modelweight",
        "weights",
        "checkpoint",
        "pytorchmodel",
        "transformersmodel",
        "awq",
        "gguf",
        "gptq",
    )
    return any(signal in normalized for signal in signals)


def safe_get(obj: Any, name: str, default: Any = "") -> Any:
    return getattr(obj, name, default)


def parse_instance(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            loaded = {}
    elif isinstance(raw, dict):
        loaded = raw
    else:
        loaded = {}
        for name in dir(raw):
            if name.startswith("_"):
                continue
            value = getattr(raw, name, None)
            if isinstance(value, (str, int, float, bool, type(None), list, dict)):
                loaded[name] = value
    if not isinstance(loaded, dict):
        loaded = {}
    total_bytes = _int(loaded.get("totalUncompressedBytes"))
    framework = str(loaded.get("framework") or "")
    slug = str(loaded.get("slug") or "")
    version = _int(loaded.get("versionNumber"), 0)
    url = str(loaded.get("url") or "")
    download_url = str(loaded.get("downloadUrl") or "")
    license_name = str(loaded.get("licenseName") or "")
    text = " ".join([framework, slug, url, download_url, license_name])
    return {
        "slug": slug,
        "framework": framework,
        "framework_public_label": framework.replace("MODEL_FRAMEWORK_", "").replace("_", " ").title(),
        "version_number": version,
        "total_uncompressed_bytes": total_bytes,
        "total_uncompressed_gb": round(total_bytes / 1_000_000_000, 6) if total_bytes else 0,
        "license_name": license_name,
        "url": url,
        "download_url_public_path_digest": sha_payload(download_url) if download_url else "",
        "glm52_text_match": text_has_glm52(text),
        "weight_source_signal": text_has_weight_signal(text) or total_bytes > 1_000_000_000,
    }


def summarize_model(obj: Any, *, query: str) -> dict[str, Any]:
    instances = [parse_instance(item) for item in _list(safe_get(obj, "instances", []))]
    ref = str(safe_get(obj, "ref", ""))
    title = str(safe_get(obj, "title", ""))
    subtitle = str(safe_get(obj, "subtitle", ""))
    slug = str(safe_get(obj, "slug", ""))
    author = str(safe_get(obj, "author", ""))
    url = str(safe_get(obj, "url", ""))
    text = " ".join([ref, title, subtitle, slug, author, url])
    glm52 = text_has_glm52(text) or any(item.get("glm52_text_match") for item in instances)
    weight_signal = text_has_weight_signal(text) or any(item.get("weight_source_signal") for item in instances)
    attachable = bool(glm52 and weight_signal and instances)
    return {
        "schema": "glm52_kaggle_model_search_result_v1",
        "query": query,
        "ref": ref,
        "title": title,
        "subtitle": subtitle[:240],
        "author": author,
        "url": url,
        "is_private": safe_get(obj, "is_private", None) is True,
        "vote_count": safe_get(obj, "vote_count", None),
        "instance_count": len(instances),
        "instances": instances[:12],
        "description_digest": sha_payload(str(safe_get(obj, "description", ""))[:12000]),
        "description_public_excerpt_included": False,
        "glm52_text_match": glm52,
        "weight_source_signal": weight_signal,
        "attachable_glm52_weight_source_candidate": attachable,
        "kaggle_kernel_model_sources": [
            f"{ref}/{item['framework_public_label'].replace(' ', '')}/{item['slug']}/{item['version_number']}"
            for item in instances
            if ref and item.get("slug") and item.get("version_number")
        ],
        "public_artifact_safe": True,
    }


def summarize_dataset(obj: Any, *, query: str) -> dict[str, Any]:
    ref = str(safe_get(obj, "ref", ""))
    title = str(safe_get(obj, "title", ""))
    subtitle = str(safe_get(obj, "subtitle", ""))
    owner = str(safe_get(obj, "owner_ref", ""))
    url = str(safe_get(obj, "url", ""))
    total_bytes = _int(safe_get(obj, "total_bytes", 0))
    license_name = str(safe_get(obj, "license_name", ""))
    text = " ".join([ref, title, subtitle, owner, url, license_name])
    glm52 = text_has_glm52(text)
    weight_signal = text_has_dataset_weight_signal(text)
    return {
        "schema": "glm52_kaggle_dataset_search_result_v1",
        "query": query,
        "ref": ref,
        "title": title,
        "subtitle": subtitle[:240],
        "owner_ref": owner,
        "url": url,
        "license_name": license_name,
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 1_000_000_000, 6) if total_bytes else 0,
        "download_count": safe_get(obj, "download_count", None),
        "vote_count": safe_get(obj, "vote_count", None),
        "description_digest": sha_payload(str(safe_get(obj, "description", ""))[:12000]),
        "description_public_excerpt_included": False,
        "glm52_text_match": glm52,
        "weight_source_signal": weight_signal,
        "attachable_glm52_weight_source_candidate": bool(glm52 and weight_signal),
        "public_artifact_safe": True,
    }


def build_env(args: argparse.Namespace) -> tuple[dict[str, str] | None, tempfile.TemporaryDirectory[str] | None]:
    if not args.token_section:
        return None, None
    sections = {section["label"]: section for section in token_probe.parse_token_sections(Path(args.token_file))}
    if args.token_section not in sections:
        raise SystemExit(f"token section not found: {args.token_section}")
    temp_dir = tempfile.TemporaryDirectory(prefix="ct_glm52_kaggle_source_search_")
    env = token_probe.clean_env(sections[args.token_section]["env"], config_dir=Path(temp_dir.name))
    return env, temp_dir


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = [item.strip() for item in (args.query or list(DEFAULT_QUERIES)) if item.strip()]
    env, temp_dir = build_env(args)
    original_env = os.environ.copy()
    if env is not None:
        os.environ.clear()
        os.environ.update(env)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        steps: list[dict[str, Any]] = []
        models: list[dict[str, Any]] = []
        datasets: list[dict[str, Any]] = []
        for query in queries:
            model_error: dict[str, Any] = {}
            dataset_error: dict[str, Any] = {}
            try:
                raw_models = [item for item in (api.model_list(search=query, page_size=int(args.page_size)) or []) if item]
                models.extend(summarize_model(item, query=query) for item in raw_models)
            except Exception as exc:  # pragma: no cover - live Kaggle API branch
                raw_models = []
                model_error = {"error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))}
            try:
                raw_datasets = [item for item in (api.dataset_list(search=query, page=1) or []) if item]
                datasets.extend(summarize_dataset(item, query=query) for item in raw_datasets[: int(args.page_size)])
            except Exception as exc:  # pragma: no cover - live Kaggle API branch
                raw_datasets = []
                dataset_error = {"error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))}
            steps.append(
                {
                    "query": query,
                    "model_result_count": len(raw_models),
                    "dataset_result_count": len(raw_datasets),
                    "model_error": model_error,
                    "dataset_error": dataset_error,
                }
            )
    finally:
        if env is not None:
            os.environ.clear()
            os.environ.update(original_env)
        if temp_dir is not None:
            temp_dir.cleanup()

    model_candidates = [item for item in models if item.get("attachable_glm52_weight_source_candidate") is True]
    dataset_candidates = [item for item in datasets if item.get("attachable_glm52_weight_source_candidate") is True]
    exact_ready = bool(model_candidates or dataset_candidates)
    blockers = []
    if not model_candidates:
        blockers.append("kaggle_models_glm52_weight_source_not_found")
    if not dataset_candidates:
        blockers.append("kaggle_datasets_glm52_weight_source_not_found")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "glm52_kaggle_public_source_search_ready": True,
        "token_section_label": str(args.token_section or ""),
        "query_count": len(queries),
        "queries": queries,
        "model_result_count": len(models),
        "dataset_result_count": len(datasets),
        "compatible_model_source_count": len(model_candidates),
        "compatible_dataset_source_count": len(dataset_candidates),
        "kaggle_models_glm52_source_verified": bool(model_candidates),
        "kaggle_datasets_glm52_source_verified": bool(dataset_candidates),
        "kaggle_attach_source_verified": exact_ready,
        "recommended_kaggle_kernel_model_sources": [
            source
            for item in model_candidates
            for source in _list(item.get("kaggle_kernel_model_sources"))
        ][:12],
        "model_results": models[: int(args.max_public_results)],
        "dataset_results": datasets[: int(args.max_public_results)],
        "compatible_model_candidates": model_candidates[:12],
        "compatible_dataset_candidates": dataset_candidates[:12],
        "steps": steps,
        "blockers": blockers,
        "diagnosis_codes": [
            "kaggle_glm52_attach_source_found" if exact_ready else "kaggle_glm52_attach_source_not_found",
            "kaggle_public_model_search_completed",
            "kaggle_public_dataset_search_completed",
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
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"].append("public_redaction_scan_failed")
        report["redaction_errors"] = leaks
    path = output_dir / "glm52_kaggle_public_source_search.json"
    write_json(path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
    }
    write_json(path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--max-public-results", type=int, default=64)
    parser.add_argument("--token-file", default="~/.config/crowdtensor/kaggle-tokens.md")
    parser.add_argument("--token-section", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.page_size < 1 or args.page_size > 200:
        raise SystemExit("--page-size must be between 1 and 200")
    if args.max_public_results < 1 or args.max_public_results > 500:
        raise SystemExit("--max-public-results must be between 1 and 500")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'glm52_kaggle_public_source_search.json'}")
        print(f"Kaggle attach source verified: {report.get('kaggle_attach_source_verified')}")
        if report.get("blockers"):
            print("Blockers: " + ", ".join(str(item) for item in report.get("blockers") or []))
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
