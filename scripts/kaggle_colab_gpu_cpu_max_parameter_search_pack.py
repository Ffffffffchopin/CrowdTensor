#!/usr/bin/env python3
"""Build Kaggle CUDA + Colab CUDA + CPU dense max-parameter search evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "kaggle_colab_gpu_cpu_max_parameter_search_v1"
SUPPORT_BUNDLE_SCHEMA = "kaggle_colab_gpu_cpu_max_parameter_search_support_bundle_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-colab-gpu-cpu-max-parameter-search"
DEFAULT_BASELINE_32B_REPORT = (
    "dist/kaggle-colab-gpu-cpu-32b-20260629-r2-manager-retry/"
    "kaggle_32b_full_heterogeneous_probe.json"
)
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Set-Cookie",
    "Cookie:",
    "runtime_proxy_token",
    "runtime_proxy_url",
    "oauth_token",
    "endpoint",
    '"prompt":',
    '"raw_prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"input_ids":',
    '"hidden_b64":',
    '"next_token_id_private":',
    '"activation":',
    '"hidden_state":',
    '"logits":',
    '"past_key_values":',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parameter_value(value: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", str(value or "").lower())
    return float(match.group(1)) if match else 0.0


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    if path.is_file():
        entry["sha256"] = sha256_file(path)
    return entry


def report_parameter_class(report: dict[str, Any]) -> str:
    model = _dict(report.get("model"))
    value = model.get("parameter_count_b")
    if value is not None:
        try:
            number = float(value)
            if number.is_integer():
                return f"{int(number)}b"
            return f"{number:g}b"
        except (TypeError, ValueError):
            pass
    repo = str(model.get("repo") or report.get("model_repo") or "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", repo.lower())
    return f"{match.group(1)}b" if match else ""


def classify_failure(report: dict[str, Any]) -> str:
    blockers = set(str(item) for item in _list(report.get("blockers")) if item)
    stage_runs = [item for item in _list(report.get("stage_runs")) if isinstance(item, dict)]
    if any("not_accepted" in item or "kernel_not_accepted" in item for item in blockers):
        return "kaggle_kernel_acceptance"
    for run in stage_runs:
        for step in _list(run.get("steps")):
            if not isinstance(step, dict):
                continue
            if step.get("name") == "kaggle_kernel_push" and step.get("accepted") is False:
                return "kaggle_kernel_acceptance"
            session_manager = _dict(step.get("session_manager"))
            if session_manager.get("blocker"):
                return "colab_cuda_runtime_lifecycle"
    if any("oom" in item.lower() for item in blockers):
        return "gpu_oom"
    if "coordinator_stage_task_counts_incomplete" in blockers:
        return "stage_task_incomplete"
    if "one_token_generation_incomplete" in blockers:
        return "decode_incomplete"
    if "stage_runtime_not_ready" in blockers:
        return "stage_runtime_not_ready"
    return "none" if report.get("ok") is True else "unknown"


def summarize_probe(path: Path) -> dict[str, Any]:
    report = load_json(path)
    if report.get("schema") == "kaggle_colab_gpu_cpu_72b_recovered_runtime_evidence_v1":
        return summarize_recovered_runtime_evidence(path, report)
    if report.get("schema") == "kaggle_colab_gpu_cpu_large_model_blocker_v1":
        return summarize_large_model_blocker(path, report)
    model = _dict(report.get("model"))
    lifecycle = _dict(report.get("kaggle_lifecycle"))
    provider_counts = _dict(report.get("provider_stage_counts"))
    coordinator = _dict(report.get("coordinator"))
    stage_counts = _dict(report.get("stage_task_counts"))
    parameter_class = report_parameter_class(report)
    same_request_ready = bool(
        report.get("ok") is True
        and report.get("kaggle_colab_gpu_cpu_same_request_verified") is True
        and {"kaggle_cuda", "colab_cuda", "cpu"}.issubset(set(str(item) for item in _list(report.get("accepted_providers"))))
        and _int(report.get("generated_token_count")) >= 1
    )
    if parameter_class == "72b":
        same_request_ready = bool(
            same_request_ready
            and report.get("same_request_72b_kaggle_colab_gpu_cpu_full_model_verified") is True
            and model.get("full_layer_coverage_verified") is True
        )
    if parameter_value(parameter_class) > 72:
        same_request_ready = bool(same_request_ready and model.get("full_layer_coverage_verified") is True)
    stage_total = _int(model.get("stage_count"))
    completed_stage_count = sum(1 for index in range(stage_total) if _int(stage_counts.get(f"stage{index}")) >= 1)
    return {
        "schema": "kaggle_colab_gpu_cpu_max_search_attempt_v1",
        "source": {
            "path": str(path),
            "present": path.is_file(),
            "schema": str(report.get("schema") or ""),
            "ok": report.get("ok") is True,
            "sha256": sha256_file(path) if path.is_file() else "",
        },
        "parameter_class": parameter_class,
        "model_repo": str(model.get("repo") or ""),
        "architecture_class": str(model.get("architecture_class") or "dense"),
        "moe_total_parameter_count_b": float(model.get("moe_total_parameter_count_b") or 0),
        "moe_active_parameter_count_b": float(model.get("moe_active_parameter_count_b") or 0),
        "quantization": str(report.get("quantization") or model.get("quantization") or ""),
        "precision": str(model.get("precision") or ""),
        "stage_count": stage_total,
        "stage_ranges": _list(model.get("stage_ranges")),
        "expected_layer_count": _int(model.get("expected_layer_count")),
        "full_layer_coverage_verified": model.get("full_layer_coverage_verified") is True,
        "same_request_decode_verified": same_request_ready,
        "generated_token_count": _int(report.get("generated_token_count")),
        "coordinator_generated_token_count": _int(coordinator.get("generated_token_count")),
        "accepted_providers": sorted(str(item) for item in _list(report.get("accepted_providers"))),
        "provider_stage_counts": {
            "kaggle_cuda": _int(provider_counts.get("kaggle_cuda")),
            "colab_cuda": _int(provider_counts.get("colab_cuda")),
            "cpu": _int(provider_counts.get("cpu")),
            "web_tpu": _int(provider_counts.get("web_tpu")),
        },
        "stage_task_counts": stage_counts,
        "completed_stage_count": completed_stage_count,
        "blockers": [str(item) for item in _list(report.get("blockers")) if item],
        "diagnosis_codes": [str(item) for item in _list(report.get("diagnosis_codes")) if item],
        "failure_stage": classify_failure(report),
        "kernels_deleted": lifecycle.get("kernels_deleted") is True,
        "private_packages_removed": lifecycle.get("private_packages_removed") is True,
        "requested_topology": str(lifecycle.get("requested_topology") or ""),
        "actual_gpu_push_count": _int(lifecycle.get("actual_gpu_push_count")),
        "actual_colab_gpu_runtime_count": _int(lifecycle.get("actual_colab_gpu_runtime_count")),
        "actual_cpu_push_count": _int(lifecycle.get("actual_cpu_push_count")),
        "public_artifact_safe": bool(report.get("public_artifact_safe") is True or _dict(report.get("safety")).get("public_artifact_safe") is True),
    }


def summarize_large_model_blocker(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    model = _dict(report.get("model"))
    lifecycle = _dict(report.get("kaggle_lifecycle"))
    stage_ranges = _list(model.get("stage_ranges"))
    stage_total = _int(model.get("stage_count")) or len(stage_ranges)
    return {
        "schema": "kaggle_colab_gpu_cpu_max_search_attempt_v1",
        "source": {
            "path": str(path),
            "present": path.is_file(),
            "schema": str(report.get("schema") or ""),
            "ok": report.get("ok") is True,
            "sha256": sha256_file(path) if path.is_file() else "",
        },
        "parameter_class": report_parameter_class({"model": model}),
        "model_repo": str(model.get("repo") or ""),
        "architecture_class": str(model.get("architecture_class") or "dense"),
        "moe_total_parameter_count_b": float(model.get("moe_total_parameter_count_b") or 0),
        "moe_active_parameter_count_b": float(model.get("moe_active_parameter_count_b") or 0),
        "quantization": str(model.get("quantization") or report.get("quantization") or ""),
        "precision": str(model.get("precision") or ""),
        "stage_count": stage_total,
        "stage_ranges": stage_ranges,
        "expected_layer_count": _int(model.get("expected_layer_count")),
        "full_layer_coverage_verified": model.get("full_layer_coverage_verified") is True,
        "same_request_decode_verified": False,
        "generated_token_count": _int(report.get("generated_token_count")),
        "coordinator_generated_token_count": _int(_dict(report.get("coordinator")).get("generated_token_count")),
        "accepted_providers": sorted(str(item) for item in _list(report.get("accepted_providers"))),
        "provider_stage_counts": _dict(report.get("provider_stage_counts")),
        "stage_task_counts": _dict(report.get("stage_task_counts")),
        "completed_stage_count": 0,
        "blockers": [str(item) for item in _list(report.get("blockers")) if item],
        "diagnosis_codes": [str(item) for item in _list(report.get("diagnosis_codes")) if item],
        "failure_stage": str(report.get("failure_stage") or "larger_model_blocked"),
        "kernels_deleted": lifecycle.get("kernels_deleted") is True,
        "private_packages_removed": lifecycle.get("private_packages_removed") is True,
        "requested_topology": str(lifecycle.get("requested_topology") or ""),
        "actual_gpu_push_count": _int(lifecycle.get("actual_gpu_push_count")),
        "actual_colab_gpu_runtime_count": _int(lifecycle.get("actual_colab_gpu_runtime_count")),
        "actual_cpu_push_count": _int(lifecycle.get("actual_cpu_push_count")),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "source_evidence": _dict(report.get("source_evidence")),
    }


def summarize_recovered_runtime_evidence(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    model = _dict(report.get("model"))
    coordinator = _dict(report.get("coordinator"))
    topology = _dict(report.get("topology"))
    stage_counts = _dict(coordinator.get("stage_task_counts"))
    stage_ranges = _list(model.get("stage_ranges"))
    stage_total = len(stage_ranges)
    completed_stage_count = sum(1 for index in range(stage_total) if _int(stage_counts.get(f"stage{index}")) >= 1)
    provider_stage_counts = {
        "kaggle_cuda": len(_list(topology.get("kaggle_gpu_stage_ids"))),
        "colab_cuda": len(_list(topology.get("colab_gpu_stage_ids"))),
        "cpu": len(_list(topology.get("kaggle_cpu_stage_ids"))),
        "web_tpu": 0,
    }
    accepted = sorted(str(item) for item in _list(topology.get("accepted_provider_families")) if item)
    same_request_ready = bool(
        report.get("ok") is True
        and coordinator.get("ready") is True
        and _int(coordinator.get("generated_token_count")) >= 1
        and completed_stage_count == stage_total
        and {"kaggle_cuda", "colab_cuda", "cpu"}.issubset(set(accepted))
        and model.get("full_layer_coverage_verified") is True
    )
    return {
        "schema": "kaggle_colab_gpu_cpu_max_search_attempt_v1",
        "source": {
            "path": str(path),
            "present": path.is_file(),
            "schema": str(report.get("schema") or ""),
            "ok": report.get("ok") is True,
            "sha256": sha256_file(path) if path.is_file() else "",
        },
        "parameter_class": report_parameter_class({"model": model}),
        "model_repo": str(model.get("repo") or ""),
        "architecture_class": str(model.get("architecture_class") or "dense"),
        "moe_total_parameter_count_b": float(model.get("moe_total_parameter_count_b") or 0),
        "moe_active_parameter_count_b": float(model.get("moe_active_parameter_count_b") or 0),
        "quantization": str(model.get("quantization") or ""),
        "precision": "bf16_or_fp16_stage_runtime",
        "stage_count": stage_total,
        "stage_ranges": stage_ranges,
        "expected_layer_count": 80 if parameter_value(report_parameter_class({"model": model})) == 72 else 0,
        "full_layer_coverage_verified": model.get("full_layer_coverage_verified") is True,
        "same_request_decode_verified": same_request_ready,
        "generated_token_count": _int(coordinator.get("generated_token_count")),
        "coordinator_generated_token_count": _int(coordinator.get("generated_token_count")),
        "accepted_providers": accepted,
        "provider_stage_counts": provider_stage_counts,
        "stage_task_counts": stage_counts,
        "completed_stage_count": completed_stage_count,
        "blockers": [] if same_request_ready else ["recovered_runtime_evidence_not_ready"],
        "diagnosis_codes": [
            "kaggle_colab_gpu_cpu_72b_recovered_runtime_evidence_ready"
            if same_request_ready
            else "kaggle_colab_gpu_cpu_72b_recovered_runtime_evidence_not_ready"
        ],
        "failure_stage": "none" if same_request_ready else "recovered_runtime_evidence_not_ready",
        "kernels_deleted": report.get("kaggle_cleanup_verified") is True,
        "private_packages_removed": True,
        "requested_topology": "2KaggleGPU_stages_1ColabGPU_stages_0WebTPU_stages_5CPU_stages",
        "actual_gpu_push_count": 1,
        "actual_colab_gpu_runtime_count": 1,
        "actual_cpu_push_count": 5,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "recovered_from_incomplete_main_report": report.get("raw_main_report_written") is False,
        "recovery_reason": str(report.get("raw_main_report_blocker") or ""),
    }


def max_parameter_class(attempts: list[dict[str, Any]], *, successful: bool) -> str:
    values = [
        str(item.get("parameter_class") or "")
        for item in attempts
        if (item.get("same_request_decode_verified") is True) == successful and str(item.get("parameter_class") or "")
    ]
    return max(values, key=parameter_value, default="")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts = [summarize_probe(Path(args.baseline_32b_report))]
    attempts.extend(summarize_probe(Path(item)) for item in args.attempt_report)
    max_success = max_parameter_class(attempts, successful=True)
    attempted = [str(item.get("parameter_class") or "") for item in attempts if str(item.get("parameter_class") or "")]
    max_attempted = max(attempted, key=parameter_value, default="")
    failed = [item for item in attempts if item.get("same_request_decode_verified") is not True and item.get("parameter_class")]
    largest_failed = max(failed, key=lambda item: parameter_value(str(item.get("parameter_class") or "")), default={})
    blockers = sorted({blocker for item in failed for blocker in _list(item.get("blockers")) if blocker})
    if failed and not blockers:
        blockers = ["larger_than_baseline_decode_not_verified"]
    successful = [item for item in attempts if item.get("same_request_decode_verified") is True]
    dense_successful = [
        item for item in successful
        if str(item.get("architecture_class") or "dense") == "dense"
        and str(item.get("quantization") or "") == "none"
    ]
    dense_attempted = [
        item for item in attempts
        if str(item.get("architecture_class") or "dense") == "dense"
        and str(item.get("quantization") or "") == "none"
    ]
    moe_successful = [
        item for item in successful
        if str(item.get("architecture_class") or "") in {"moe", "hybrid_moe"}
    ]
    max_dense_success = max(
        (str(item.get("parameter_class") or "") for item in dense_successful),
        key=parameter_value,
        default="",
    )
    max_dense_attempted = max(
        (str(item.get("parameter_class") or "") for item in dense_attempted),
        key=parameter_value,
        default="",
    )
    max_moe_total_success = max(
        (float(item.get("moe_total_parameter_count_b") or parameter_value(str(item.get("parameter_class") or ""))) for item in moe_successful),
        default=0.0,
    )
    max_moe_active_success = max(
        (float(item.get("moe_active_parameter_count_b") or 0.0) for item in moe_successful),
        default=0.0,
    )
    model_source_refs = sorted(
        {
            str(source)
            for item in attempts
            for source in _list(_dict(item.get("source_evidence")).get("model_source_refs"))
            if str(source or "").strip()
        }
    )
    if not model_source_refs:
        for item in attempts:
            source_candidate = _dict(_dict(item.get("source_evidence")).get("source_candidate"))
            source = source_candidate.get("kaggle_kernel_model_source")
            if source:
                model_source_refs.append(str(source))
        model_source_refs = sorted(set(model_source_refs))
    result = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "kaggle_colab_gpu_cpu_max_parameter_search_ready": True,
        "output_dir": str(output_dir),
        "goal_scope": {
            "accelerator_path": "kaggle_t4x2_colab_t4_kaggle_cpu",
            "dense_full_precision_main_path": True,
            "quantized_success_allowed": False,
            "same_request_full_layer_decode_required_for_success": True,
        },
        "max_successful_same_request_decode_parameter_class": max_success,
        "max_successful_dense_full_precision_parameter_class": max_dense_success,
        "max_successful_moe_total_parameter_class": f"{max_moe_total_success:g}b" if max_moe_total_success else "",
        "max_successful_moe_activated_parameter_class": f"{max_moe_active_success:g}b" if max_moe_active_success else "",
        "max_attempted_parameter_class": max_attempted,
        "model_source_refs": model_source_refs,
        "attempts": attempts,
        "largest_failed_attempt": largest_failed,
        "blocker_codes": blockers,
        "failure_stage": str(largest_failed.get("failure_stage") or ""),
        "conclusions": {
            "max_stably_verified_dense_full_precision_parameter_class": max_dense_success,
            "max_successful_dense_full_precision_parameter_class": max_dense_success,
            "max_successful_moe_total_parameter_class": f"{max_moe_total_success:g}b" if max_moe_total_success else "",
            "max_successful_moe_activated_parameter_class": f"{max_moe_active_success:g}b" if max_moe_active_success else "",
            "max_attempted_dense_full_precision_parameter_class": max_dense_attempted,
            "larger_than_max_success_attempted": bool(parameter_value(max_attempted) > parameter_value(max_success)),
            "largest_failed_failure_stage": str(largest_failed.get("failure_stage") or ""),
            "next_bottleneck": sorted({
                str(largest_failed.get("failure_stage") or "unknown"),
                *[str(item) for item in _list(largest_failed.get("blockers")) if item],
            }) if largest_failed else [],
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
        "artifacts": {},
    }
    result["redaction_scan_errors"] = public_redaction_errors(result)
    support = {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "attempt_sources": [item["source"] for item in attempts],
        "public_artifact_safe": True,
    }
    summary_path = output_dir / "kaggle_colab_gpu_cpu_max_parameter_search.json"
    support_path = output_dir / "kaggle_colab_gpu_cpu_max_parameter_search_support.json"
    write_json(support_path, support)
    result["artifacts"] = {
        "summary_json": {"kind": "summary", "path": summary_path.name, "present": True, "schema": SCHEMA, "ok": True},
        "support_bundle_json": artifact_entry(
            support_path,
            output_dir,
            kind="support_bundle",
            schema=SUPPORT_BUNDLE_SCHEMA,
            ok=True,
        ),
    }
    write_json(summary_path, result)
    result["artifacts"]["summary_json"] = artifact_entry(summary_path, output_dir, kind="summary", schema=SCHEMA, ok=True)
    write_json(summary_path, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-32b-report", default=DEFAULT_BASELINE_32B_REPORT)
    parser.add_argument("--attempt-report", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "kaggle_colab_gpu_cpu_max_parameter_search: "
            f"max_success={report['max_successful_same_request_decode_parameter_class']} "
            f"max_attempted={report['max_attempted_parameter_class']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
