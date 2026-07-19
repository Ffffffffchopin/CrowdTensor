#!/usr/bin/env python3
"""Check Kaggle CUDA + Colab CUDA + CPU same-request decode reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "kaggle_32b_full_heterogeneous_probe_v1"
REQUIRED_PROVIDERS = {"kaggle_cuda", "colab_cuda", "cpu"}
PRIVATE_KEYS = {
    "token",
    "proxy_token",
    "runtime_proxy_token",
    "url",
    "runtime_proxy_url",
    "endpoint",
    "credentials",
    "oauth_token",
    "input_ids",
    "hidden_b64",
    "next_token_id_private",
}


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit("report must be a JSON object")
    return loaded


def walk_private_keys(value: Any, *, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in PRIVATE_KEYS or lowered.endswith("_private"):
                if child not in (False, "", None):
                    hits.append(f"{path}.{key}".strip("."))
            hits.extend(walk_private_keys(child, path=f"{path}.{key}".strip(".")))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(walk_private_keys(child, path=f"{path}[{index}]"))
    return hits


def check_report(report: dict[str, Any], *, require_ready: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    if report.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    accepted = set(str(item) for item in report.get("accepted_providers") or [])
    if require_ready and not REQUIRED_PROVIDERS.issubset(accepted):
        errors.append("required_providers_missing")
    provider_counts = report.get("provider_stage_counts") if isinstance(report.get("provider_stage_counts"), dict) else {}
    for provider in REQUIRED_PROVIDERS:
        if require_ready and int(provider_counts.get(provider) or 0) < 1:
            errors.append(f"{provider}_stage_count_missing")
    coordinator = report.get("coordinator") if isinstance(report.get("coordinator"), dict) else {}
    if require_ready and int(report.get("generated_token_count") or 0) < 1:
        errors.append("generated_token_missing")
    if require_ready and int(coordinator.get("generated_token_count") or 0) < 1:
        errors.append("coordinator_generated_token_missing")
    if require_ready and not coordinator.get("generated_token_hashes"):
        errors.append("generated_token_hash_missing")
    if not coordinator.get("activation_hashes"):
        errors.append("activation_hash_missing")
    stage_counts = report.get("stage_task_counts") if isinstance(report.get("stage_task_counts"), dict) else {}
    stage_count = int((report.get("model") or {}).get("stage_count") or 0) if isinstance(report.get("model"), dict) else 0
    if stage_count < 3:
        errors.append("stage_count_too_small")
    for stage_id in range(stage_count):
        if int(stage_counts.get(f"stage{stage_id}") or 0) < int(report.get("max_new_tokens") or 1):
            if require_ready:
                errors.append("stage_task_count_incomplete")
            break
    lifecycle = report.get("kaggle_lifecycle") if isinstance(report.get("kaggle_lifecycle"), dict) else {}
    if lifecycle.get("kernels_deleted") is not True:
        errors.append("kaggle_cleanup_not_verified")
    if lifecycle.get("private_packages_removed") is not True:
        errors.append("private_package_cleanup_not_verified")
    if report.get("quantization") != "none":
        errors.append("quantization_not_none")
    if (report.get("safety") or {}).get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    private_hits = walk_private_keys(report)
    if private_hits:
        errors.append("private_material_public")
    if require_ready and report.get("kaggle_colab_gpu_cpu_same_request_verified") is not True:
        errors.append("kaggle_colab_gpu_cpu_same_request_not_verified")
    ok = not errors
    return {
        "ok": ok,
        "errors": sorted(set(errors)),
        "accepted_providers": sorted(accepted),
        "generated_token_count": int(report.get("generated_token_count") or 0),
        "stage_count": stage_count,
        "private_hit_count": len(private_hits),
        "private_hit_examples": private_hits[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--no-require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check_report(load_json(Path(args.report)), require_ready=not args.no_require_ready)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"kaggle_colab_gpu_cpu_heterogeneous_check: ok={result['ok']} errors={','.join(result['errors']) or 'none'}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
