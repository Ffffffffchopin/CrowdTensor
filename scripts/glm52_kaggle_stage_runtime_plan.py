#!/usr/bin/env python3
"""Build a public-safe GLM 5.2 Kaggle stage runtime adapter plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "glm52_kaggle_stage_runtime_plan_v1"
MODEL_ID = "zai-org/GLM-5.2"
COMPATIBLE_WEIGHT_REPO = "cyankiwi/GLM-5.2-AWQ-INT4"
REQUIRED_PROVIDERS = ["kaggle_cuda", "kaggle_jax_tpu", "kaggle_cpu"]
STAGE_REPORT_SCHEMA = "glm52_kaggle_stage_runtime_report_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-kaggle-stage-runtime-plan"
DEFAULT_LAYER_COUNT = 78
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


def sha_text(value: Any) -> str:
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()


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


def safety_flags() -> dict[str, bool]:
    return {
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
    }


def _source_stage_plans(source_report: dict[str, Any]) -> list[dict[str, Any]]:
    stage_plan = _dict(source_report.get("stage_adapter_plan"))
    plans = [item for item in _list(stage_plan.get("stage_plans")) if isinstance(item, dict)]
    if plans:
        return plans
    model = _dict(source_report.get("model"))
    layer_count = max(1, _int(model.get("num_hidden_layers"), 78))
    spans = [(0, layer_count // 3), (layer_count // 3, 2 * layer_count // 3), (2 * layer_count // 3, layer_count)]
    return [
        {"stage_id": index, "backend": provider, "layer_range": list(span), "metadata_only": True}
        for index, (provider, span) in enumerate(zip(REQUIRED_PROVIDERS, spans, strict=True))
    ]


def parse_stage_specs(values: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for raw_value in values or []:
        raw = str(raw_value or "").strip()
        if not raw:
            continue
        parts = [part.strip() for part in raw.split(":")]
        if len(parts) != 4:
            raise SystemExit("--stage-spec must use stage_id:provider:start:end")
        stage_id_raw, provider, start_raw, end_raw = parts
        if provider not in REQUIRED_PROVIDERS:
            raise SystemExit(f"--stage-spec provider must be one of {','.join(REQUIRED_PROVIDERS)}")
        stage_id = int(stage_id_raw)
        start = int(start_raw)
        end = int(end_raw)
        if stage_id < 0 or end <= start or start < 0:
            raise SystemExit("--stage-spec must have non-negative stage_id/start and end > start")
        specs.append({
            "stage_id": stage_id,
            "backend": provider,
            "layer_range": [start, end],
            "metadata_only": True,
        })
    return specs


def _selected_stage_plans(source_report: dict[str, Any], args: argparse.Namespace | None) -> list[dict[str, Any]]:
    overrides = parse_stage_specs(getattr(args, "stage_spec", []) or []) if args is not None else []
    return overrides or _source_stage_plans(source_report)


def build_stage_specs(source_report: dict[str, Any], awq_header_report: dict[str, Any], args: argparse.Namespace | None = None) -> list[dict[str, Any]]:
    header_layer_range = _list(awq_header_report.get("stage_layer_range"))
    header_subset = {
        "awq_header_probe_present": bool(awq_header_report),
        "awq_header_stage_id": _int(awq_header_report.get("stage_id")),
        "awq_header_stage_count": _int(awq_header_report.get("stage_count")),
        "awq_header_layer_range": header_layer_range,
        "awq_header_assigned_weight_key_count": _int(awq_header_report.get("assigned_weight_key_count")),
        "awq_header_selected_tensor_storage_gb": awq_header_report.get("total_selected_tensor_storage_gb", 0),
    }
    specs: list[dict[str, Any]] = []
    for ordinal, plan in enumerate(_selected_stage_plans(source_report, args)):
        provider = str(plan.get("backend") or plan.get("provider") or REQUIRED_PROVIDERS[min(ordinal, 2)])
        layer_range = _list(plan.get("layer_range"))
        if len(layer_range) != 2:
            layer_range = [ordinal * 26, (ordinal + 1) * 26]
        blockers = [
            "glm52_stage_runtime_live_report_missing",
            f"glm52_{provider}_stage_runtime_not_verified",
        ]
        if provider == "kaggle_cuda":
            runtime_adapter = "torch_or_transformers_awq_stage_selective_cuda_worker"
            blockers.append("glm52_awq_cuda_memory_budget_not_verified")
        elif provider == "kaggle_jax_tpu":
            runtime_adapter = "jax_awq_stage_selective_tpu_worker"
            blockers.append("glm52_awq_tpu_stage_smoke_not_ready")
        else:
            runtime_adapter = "torch_or_transformers_awq_stage_selective_cpu_worker"
            blockers.append("glm52_awq_cpu_runtime_not_verified")
        spec = {
            "schema": "glm52_kaggle_stage_runtime_spec_v1",
            "stage_id": _int(plan.get("stage_id"), ordinal),
            "provider": provider,
            "model_id": MODEL_ID,
            "compatible_weight_repo": COMPATIBLE_WEIGHT_REPO,
            "stage_layer_range": [int(layer_range[0]), int(layer_range[1])],
            "assigned_key_count": _int(plan.get("assigned_key_count")),
            "assigned_file_count": _int(plan.get("assigned_file_count")),
            "key_digest": str(plan.get("key_digest") or ""),
            "runtime_adapter": runtime_adapter,
            "expected_stage_report_schema": STAGE_REPORT_SCHEMA,
            "stage_runtime_adapter_verified": False,
            "same_request_route_verified": False,
            "live_run_performed": False,
            "public_artifact_safe": True,
            "blockers": sorted(set(blockers)),
        }
        if provider == "kaggle_jax_tpu":
            spec["awq_header_probe_subset"] = header_subset
        specs.append(spec)
    return specs


def build_launcher_contract(stage_specs: list[dict[str, Any]]) -> dict[str, Any]:
    env_vars = [
        "CT_GLM52_STAGE_ID",
        "CT_GLM52_PROVIDER",
        "CT_GLM52_LAYER_START",
        "CT_GLM52_LAYER_END",
        "CT_GLM52_COORDINATOR_REQUEST_HASH",
        "CT_GLM52_COMPATIBLE_WEIGHT_REPO",
        "CT_GLM52_STAGE_REPORT_PATH",
    ]
    return {
        "schema": "glm52_kaggle_stage_runtime_launcher_contract_v1",
        "private_kernel_required": True,
        "expected_stage_report_schema": STAGE_REPORT_SCHEMA,
        "required_env_vars": env_vars,
        "required_stage_report_fields": [
            "schema",
            "model_id",
            "provider",
            "stage_id",
            "stage_layer_range",
            "coordinator_request_id_hash",
            "stage_execution_verified",
            "stage_output_hash",
            "live_run_performed",
            "public_artifact_safe",
        ],
        "provider_launchers": [
            {
                "provider": str(spec.get("provider")),
                "stage_id": _int(spec.get("stage_id")),
                "private_package_dir_template": f"private-kaggle-glm52-stage-{_int(spec.get('stage_id'))}-{spec.get('provider')}",
                "post_run_check_command": (
                    "python scripts/glm52_kaggle_stage_runtime_check.py "
                    f"--report <{spec.get('provider')}-stage-report.json> --require-verified --json"
                ),
            }
            for spec in stage_specs
        ],
        "public_artifact_safe": True,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    source_report = load_json(args.source_report)
    awq_header_report = load_json(args.awq_stage_header_report)
    stage_specs = build_stage_specs(source_report, awq_header_report, args)
    stage_count = len(stage_specs)
    for spec in stage_specs:
        spec["stage_count"] = stage_count
    providers = [str(spec.get("provider")) for spec in stage_specs]
    blockers: set[str] = {
        "glm52_stage_runtime_live_reports_missing",
        "glm52_same_request_route_not_verified",
    }
    for spec in stage_specs:
        blockers.update(str(item) for item in _list(spec.get("blockers")) if item)
    if set(REQUIRED_PROVIDERS) - set(providers):
        blockers.add("glm52_stage_runtime_required_provider_missing")
    stage_ids = [_int(spec.get("stage_id"), -1) for spec in stage_specs]
    if len(stage_ids) != len(set(stage_ids)):
        blockers.add("glm52_stage_runtime_duplicate_stage_id")
    ranges = sorted((_int(spec.get("stage_layer_range", [0, 0])[0]), _int(spec.get("stage_layer_range", [0, 0])[1])) for spec in stage_specs)
    expected_layer_count = _int(getattr(args, "expected_layer_count", DEFAULT_LAYER_COUNT), DEFAULT_LAYER_COUNT)
    contiguous = bool(ranges and ranges[0][0] == 0 and ranges[-1][1] == expected_layer_count and all(left[1] == right[0] for left, right in zip(ranges, ranges[1:])))
    if not contiguous:
        blockers.add("glm52_stage_runtime_layer_coverage_not_contiguous")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "glm52_stage_runtime_plan_ready": True,
        "stage_runtime_adapter_verified": False,
        "same_request_route_verified": False,
        "model": {
            "model_id": MODEL_ID,
            "compatible_weight_repo": args.compatible_weight_repo,
            "fallback_model_allowed_for_success": False,
        },
        "source": {
            "source_schema": str(source_report.get("schema") or ""),
            "source_ready": source_report.get("glm52_source_resolver_ready") is True,
            "stage_adapter_plan_present": bool(_dict(source_report.get("stage_adapter_plan"))),
        },
        "stage_specs": stage_specs,
        "stage_topology": {
            "schema": "glm52_kaggle_stage_topology_v1",
            "source": "cli_stage_spec" if getattr(args, "stage_spec", []) else "source_report_or_default",
            "stage_count": stage_count,
            "required_providers": REQUIRED_PROVIDERS,
            "provider_counts": {provider: providers.count(provider) for provider in sorted(set(providers))},
            "expected_layer_count": expected_layer_count,
            "contiguous_full_layer_coverage": contiguous,
            "layer_ranges": [list(item) for item in ranges],
            "public_artifact_safe": True,
        },
        "launcher_contract": build_launcher_contract(stage_specs),
        "completion_boundary": {
            "plan_is_not_runtime_success": True,
            "stage_runtime_report_required": True,
            "same_request_probe_required": True,
            "queue_or_stage_smoke_is_not_success": True,
        },
        "blockers": sorted(blockers),
        "next_commands": [
            "python scripts/glm52_kaggle_stage_runtime_check.py --report <kaggle-cuda-stage.json> --require-verified --json",
            "python scripts/glm52_kaggle_stage_runtime_check.py --report <kaggle-jax-tpu-stage.json> --require-verified --json",
            "python scripts/glm52_kaggle_stage_runtime_check.py --report <kaggle-cpu-stage.json> --require-verified --json",
            "python scripts/glm52_kaggle_same_request_probe.py --mode assemble --stage-report <kaggle-cuda-stage.json> --stage-report <kaggle-jax-tpu-stage.json> --stage-report <kaggle-cpu-stage.json> --coordinator-report <coordinator.json> --cleanup-report <cleanup.json>",
        ],
        "safety": safety_flags(),
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set(_list(report.get("blockers")) + ["public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-report", default="")
    parser.add_argument("--awq-stage-header-report", default="")
    parser.add_argument("--compatible-weight-repo", default=COMPATIBLE_WEIGHT_REPO)
    parser.add_argument("--stage-spec", action="append", default=[], help="Override topology as stage_id:provider:start:end; repeatable.")
    parser.add_argument("--expected-layer-count", type=int, default=DEFAULT_LAYER_COUNT)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_kaggle_stage_runtime_plan.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Stage runtime plan ready: {report.get('glm52_stage_runtime_plan_ready')}")
        print(f"Stage runtime verified: {report.get('stage_runtime_adapter_verified')}")
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
