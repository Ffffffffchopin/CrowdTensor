#!/usr/bin/env python3
"""Build a public-safe blocker artifact for larger Kaggle/Colab GPU+CPU attempts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "kaggle_colab_gpu_cpu_large_model_blocker_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: str) -> dict[str, Any]:
    if not path:
        return {}
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def attach_stage_owned_preflight_ready(report: dict[str, Any]) -> bool:
    runtime = _dict(report.get("runtime_report"))
    stage_plan = _dict(runtime.get("stage_plan"))
    return bool(
        report.get("stage_owned_preflight_verified") is True
        or runtime.get("stage_owned_preflight_verified") is True
        or stage_plan.get("stage_owned_preflight_verified") is True
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    source = load_json(args.source_resolver_report)
    attach_reports = [load_json(path) for path in args.attach_probe_report]
    stage_loader_reports = [load_json(path) for path in args.stage_loader_report]
    stage_plan_attach_reports = [
        report for report in attach_reports
        if attach_stage_owned_preflight_ready(report)
    ]
    source_candidates = [
        item for item in _list(source.get("candidates"))
        if isinstance(item, dict) and str(item.get("parameter_class") or "") == str(args.parameter_class or "")
    ]
    source_candidate = source_candidates[0] if source_candidates else {}
    architecture_class = args.architecture_class or str(source_candidate.get("architecture_class") or "dense")
    source_precision = str(source_candidate.get("precision_class") or "")
    if source_precision == "full_precision_or_bf16":
        model_precision = "bf16_or_fp16_stage_runtime"
    elif source_precision:
        model_precision = source_precision
    else:
        model_precision = "unknown"
    active_parameter_count_b = (
        float(args.active_parameter_count_b)
        if args.active_parameter_count_b is not None
        else float(source_candidate.get("active_params_b") or args.parameter_count_b)
    )
    model = {
        "repo": args.model_repo,
        "parameter_count_b": float(args.parameter_count_b),
        "parameter_class": args.parameter_class or f"{args.parameter_count_b:g}b",
        "architecture_class": architecture_class,
        "moe_total_parameter_count_b": float(args.parameter_count_b) if architecture_class in {"moe", "hybrid_moe"} else 0.0,
        "moe_active_parameter_count_b": active_parameter_count_b if architecture_class in {"moe", "hybrid_moe"} else 0.0,
        "quantization": "none",
        "precision": model_precision,
        "stage_count": int(args.stage_count),
        "expected_layer_count": int(args.expected_layer_count),
        "stage_ranges": json.loads(args.stage_ranges_json),
        "full_layer_coverage_verified": False,
    }
    blockers = [str(item) for item in args.blocker]
    if not blockers:
        blockers = ["larger_model_not_executed"]
    if any(report.get("kaggle_model_attach_probe_ready") is False for report in attach_reports):
        blockers.append("kaggle_attach_path_missing_in_runtime")
    if attach_reports and not any(attach_stage_owned_preflight_ready(report) for report in attach_reports):
        blockers.append("stage_owned_preflight_not_verified")
    for report in stage_loader_reports:
        if report.get("ok") is not True:
            blockers.append("stage_loader_not_verified")
    if architecture_class in {"moe", "hybrid_moe"}:
        blockers.append("moe_same_request_runtime_adapter_not_verified")
    if source_candidate.get("license_agreement_required") is True:
        blockers.append("kaggle_model_license_agreement_required")
    return {
        "schema": SCHEMA,
        "ok": False,
        "generated_at": utc_now(),
        "model": model,
        "quantization": "none",
        "accepted_providers": ["kaggle_cuda", "colab_cuda", "cpu"],
        "provider_stage_counts": {"kaggle_cuda": 0, "colab_cuda": 0, "cpu": 0, "web_tpu": 0},
        "generated_token_count": 0,
        "coordinator": {"generated_token_count": 0, "activation_hashes": [], "generated_token_hashes": []},
        "stage_task_counts": {f"stage{index}": 0 for index in range(int(args.stage_count))},
        "stage_owned_preflight_verified": bool(stage_plan_attach_reports),
        "kaggle_colab_gpu_cpu_same_request_verified": False,
        "same_request_decode_verified": False,
        "blockers": sorted(set(blockers)),
        "diagnosis_codes": [
            "kaggle_colab_gpu_cpu_larger_model_attempt_blocked",
            "alternate_llm_source_metadata_ready" if source.get("ok") is True else "alternate_llm_source_metadata_not_ready",
            "alternate_llm_stage_plan_evidence_present" if stage_plan_attach_reports else "alternate_llm_stage_plan_evidence_missing",
            "alternate_llm_stage_loader_evidence_present" if stage_loader_reports else "alternate_llm_stage_loader_evidence_missing",
        ],
        "failure_stage": args.failure_stage,
        "kaggle_lifecycle": {
            "requested_topology": args.requested_topology,
            "actual_gpu_push_count": 0,
            "actual_colab_gpu_runtime_count": 0,
            "actual_cpu_push_count": 0,
            "kernels_deleted": True,
            "private_packages_removed": True,
        },
        "source_evidence": {
            "source_resolver_report": args.source_resolver_report,
            "source_resolver_ok": source.get("ok") is True,
            "model_source_refs": source.get("model_source_refs") or source.get("kernel_model_sources") or [],
            "source_candidate": source_candidate,
            "largest_dense_attach_candidate": source.get("largest_dense_attach_candidate") if isinstance(source.get("largest_dense_attach_candidate"), dict) else {},
            "largest_dense_full_precision_candidate": source.get("largest_dense_full_precision_candidate") if isinstance(source.get("largest_dense_full_precision_candidate"), dict) else {},
            "largest_moe_full_precision_candidate": source.get("largest_moe_full_precision_candidate") if isinstance(source.get("largest_moe_full_precision_candidate"), dict) else {},
            "attach_probe_reports": [
                {
                    "path": path,
                    "ok": report.get("ok") is True,
                    "kaggle_model_attach_probe_ready": report.get("kaggle_model_attach_probe_ready") is True,
                    "stage_owned_preflight_verified": attach_stage_owned_preflight_ready(report),
                    "resolved_attached_path": report.get("resolved_attached_path") or _dict(report.get("runtime_report")).get("resolved_attached_path"),
                    "blocker_codes": report.get("blocker_codes") or [],
                    "model_source": report.get("model_source"),
                    "runtime_model_type": _dict(report.get("runtime_report")).get("model_type"),
                    "runtime_num_hidden_layers": _dict(report.get("runtime_report")).get("num_hidden_layers"),
                    "stage_plan": {
                        "stage_count": _dict(_dict(report.get("runtime_report")).get("stage_plan")).get("stage_count"),
                        "assigned_key_count_total": _dict(_dict(report.get("runtime_report")).get("stage_plan")).get("assigned_key_count_total"),
                        "present_key_count_total": _dict(_dict(report.get("runtime_report")).get("stage_plan")).get("present_key_count_total"),
                        "assigned_file_count_total": _dict(_dict(report.get("runtime_report")).get("stage_plan")).get("assigned_file_count_total"),
                        "total_planned_logical_tensor_gb": _dict(_dict(report.get("runtime_report")).get("stage_plan")).get("total_planned_logical_tensor_gb"),
                        "max_stage_planned_logical_tensor_gb": _dict(_dict(report.get("runtime_report")).get("stage_plan")).get("max_stage_planned_logical_tensor_gb"),
                        "stage_owned_preflight_verified": _dict(_dict(report.get("runtime_report")).get("stage_plan")).get("stage_owned_preflight_verified") is True,
                    },
                }
                for path, report in zip(args.attach_probe_report, attach_reports)
            ],
            "stage_loader_reports": [
                {
                    "path": path,
                    "ok": report.get("ok") is True,
                    "schema": report.get("schema"),
                    "public_artifact_safe": report.get("public_artifact_safe") is True
                    or _dict(report.get("safety")).get("public_artifact_safe") is True,
                }
                for path, report in zip(args.stage_loader_report, stage_loader_reports)
            ],
            "kaggle_model_observation": _dict(source.get("largest_moe_full_precision_candidate")),
        },
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
            "private_kaggle_payload_public": False,
            "weight_tensor_values_public": False,
        },
        "public_artifact_safe": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--parameter-class", default="")
    parser.add_argument("--parameter-count-b", type=float, required=True)
    parser.add_argument("--active-parameter-count-b", type=float, default=None)
    parser.add_argument("--architecture-class", default="")
    parser.add_argument("--expected-layer-count", type=int, required=True)
    parser.add_argument("--stage-count", type=int, required=True)
    parser.add_argument("--stage-ranges-json", required=True)
    parser.add_argument("--requested-topology", required=True)
    parser.add_argument("--source-resolver-report", required=True)
    parser.add_argument("--attach-probe-report", action="append", default=[])
    parser.add_argument("--stage-loader-report", action="append", default=[])
    parser.add_argument("--blocker", action="append", default=[])
    parser.add_argument("--failure-stage", default="model_source_or_adapter_unavailable")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(args)
    output_dir = Path(args.output_dir)
    output_path = output_dir / "kaggle_colab_gpu_cpu_large_model_blocker.json"
    write_json(output_path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
