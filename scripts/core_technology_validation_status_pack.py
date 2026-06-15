#!/usr/bin/env python3
"""Summarize core large-LLM validation status from retained evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA = "core_technology_validation_status_v1"
SUPPORT_BUNDLE_SCHEMA = "core_technology_validation_status_support_bundle_v1"
DEFAULT_OUTPUT_DIR = "dist/core-technology-validation-status"
DEFAULT_SMALL_GPU_REPORT = (
    "dist/gpt2-xl-small-tier-kaggle-logfix-20260614172932/"
    "public_swarm_gpu_inference_beta_kaggle_auto.json"
)
DEFAULT_SEVEN_B_BLOCKER_REPORT = (
    "dist/large-model-kaggle-validation-t4x2-rpc-small-telemetry-inplace-20260613/"
    "large_model_kaggle_validation_run_normalized.json"
)
DEFAULT_LLAMA_LIKE_LOCAL_REPORT = (
    "dist/real-llm-llama-like-local-smoke-20260615/"
    "real_llm_sharded_evidence.json"
)
DEFAULT_STAGE_SELECTIVE_WEIGHT_REPORT = (
    "dist/stage-selective-weight-loading-check/"
    "stage_selective_weight_loading_check.json"
)

REDACTION_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "CROWDTENSOR_ADMIN_TOKEN",
    "SOURCE_TARBALL_B64",
    "MINER_ENV_TEXT",
    '"generated_text":',
    '"generated_token_ids":',
    '"activation_results":',
    '"hidden_state":',
    '"lease_token":',
    '"idempotency_key":',
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def try_load(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        return {}, {"ok": False, "reason": "missing", "path": str(path)}
    try:
        return load_json(path), {"ok": True, "path": str(path)}
    except json.JSONDecodeError as exc:
        return {}, {
            "ok": False,
            "reason": "invalid_json",
            "path": str(path),
            "line": exc.lineno,
            "column": exc.colno,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        rel = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        rel = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": rel, "present": path.is_file()}
    if path.is_file():
        entry["sha256"] = sha256_file(path)
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return [fragment for fragment in REDACTION_FRAGMENTS if fragment in encoded]


def summarize_small_gpu(report: dict[str, Any], meta: dict[str, Any], path: Path) -> dict[str, Any]:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    generation = (
        (report.get("payload_summaries") or {})
        .get("real_llm_internet_beta", {})
        .get("generation", {})
        if isinstance(report.get("payload_summaries"), dict)
        else {}
    )
    support = report.get("model_execution_support") if isinstance(report.get("model_execution_support"), dict) else {}
    codes = set(report.get("diagnosis_codes") or [])
    ready = bool(
        meta.get("ok")
        and report.get("ok") is True
        and "public_swarm_gpu_beta_kaggle_auto_ready" in codes
        and "external_runtime_verified" in codes
        and "kaggle_kernels_deleted" in codes
        and int(generation.get("generated_token_count") or 0) > 0
    )
    return {
        "ready": ready,
        "report_path": str(path),
        "report_sha256": sha256_file(path) if path.is_file() else "",
        "schema": report.get("schema", ""),
        "model_id": beta.get("model_id") or "",
        "parameter_count_estimate": support.get("parameter_count_estimate"),
        "backend": beta.get("backend") or "",
        "partition_mode": beta.get("partition_mode") or "",
        "stage_count": beta.get("stage_count"),
        "generated_token_count": int(generation.get("generated_token_count") or 0),
        "generated_text_redacted": generation.get("generated_text_redacted") is True,
        "decoded_tokens_match": "decoded_tokens_match" in codes,
        "kaggle_kernels_deleted": "kaggle_kernels_deleted" in codes,
        "external_runtime_verified": "external_runtime_verified" in codes,
        "large_model_sharded_execution_ready": bool(support.get("large_model_sharded_execution_ready")),
        "partial_weight_loading_plan_ready": bool(support.get("partial_weight_loading_plan_ready")),
        "true_partial_weight_loading_ready": bool(support.get("true_partial_weight_loading_ready")),
        "partial_weight_runtime_execution_ready": bool(support.get("partial_weight_runtime_execution_ready")),
        "large_model_blockers": support.get("large_model_blockers") or [],
        "diagnosis_codes": sorted(codes),
    }


def summarize_seven_b_blocker(report: dict[str, Any], meta: dict[str, Any], path: Path) -> dict[str, Any]:
    validation = report.get("validation") if isinstance(report.get("validation"), dict) else {}
    hardware = report.get("hardware") if isinstance(report.get("hardware"), dict) else {}
    pressure = report.get("resource_pressure_summary") if isinstance(report.get("resource_pressure_summary"), dict) else {}
    codes = set(report.get("diagnosis_codes") or [])
    blocked = bool(
        meta.get("ok")
        and report.get("ok") is False
        and validation.get("real_7b_runtime_verified") is False
        and "large_model_kaggle_gpu_hardware_verified" in codes
    )
    return {
        "blocked": blocked,
        "report_path": str(path),
        "report_sha256": sha256_file(path) if path.is_file() else "",
        "schema": report.get("schema", ""),
        "kaggle_gpu_verified": hardware.get("kaggle_gpu_verified") is True,
        "gpu_count": hardware.get("gpu_count"),
        "gpu_names": hardware.get("gpu_names") or [],
        "real_7b_runtime_verified": validation.get("real_7b_runtime_verified") is True,
        "real_runtime_verified": validation.get("real_runtime_verified") is True,
        "gpu_runtime_verified": validation.get("gpu_runtime_verified") is True,
        "sharded_path_verified": validation.get("sharded_path_verified") is True,
        "multi_worker_sharded_path_verified": validation.get("multi_worker_sharded_path_verified") is True,
        "core_validation_ready": validation.get("core_validation_ready") is True,
        "container_memory_pressure_not_vram": "large_model_kaggle_container_memory_pressure_not_vram" in codes,
        "cgroup_memory_pressure": bool(pressure.get("cgroup_memory_pressure")),
        "gpu_memory_low_pressure": bool(pressure.get("gpu_memory_low_pressure")),
        "cgroup_memory_peak_ratio": pressure.get("cgroup_memory_peak_ratio"),
        "gpu_memory_used_peak_ratio": pressure.get("gpu_memory_used_peak_ratio"),
        "blockers": report.get("blockers") or [],
        "diagnosis_codes": sorted(codes),
    }


def summarize_llama_like_local(report: dict[str, Any], meta: dict[str, Any], path: Path) -> dict[str, Any]:
    artifact = report.get("artifact") if isinstance(report.get("artifact"), dict) else {}
    generation = report.get("generation") if isinstance(report.get("generation"), dict) else {}
    codes = set(report.get("diagnosis_codes") or [])
    model_id = str(artifact.get("model_id") or (report.get("session") or {}).get("model_id") or "")
    ready = bool(
        meta.get("ok")
        and report.get("ok") is True
        and "real_llm_sharded_ready" in codes
        and "stage_local_partition_ready" in codes
        and "decoded_tokens_match" in codes
        and model_id
        and model_id != "gpt2-xl"
    )
    return {
        "ready": ready,
        "report_path": str(path),
        "report_sha256": sha256_file(path) if path.is_file() else "",
        "schema": report.get("schema", ""),
        "model_id": model_id,
        "backend": artifact.get("backend") or "",
        "partition_mode": artifact.get("partition_mode") or "",
        "stage_local_partition_ready": "stage_local_partition_ready" in codes,
        "decoded_tokens_match": "decoded_tokens_match" in codes,
        "generated_token_count": int(generation.get("generated_token_count") or 0),
        "large_model_validation": False,
        "diagnosis_codes": sorted(codes),
    }


def summarize_stage_selective_weight_loading(report: dict[str, Any], meta: dict[str, Any], path: Path) -> dict[str, Any]:
    support = report.get("model_execution_support") if isinstance(report.get("model_execution_support"), dict) else {}
    stage_rows = report.get("stage_summaries") if isinstance(report.get("stage_summaries"), list) else []
    codes = set(report.get("diagnosis_codes") or [])
    ready = bool(
        meta.get("ok")
        and report.get("ok") is True
        and report.get("stage_selective_weight_loading_ready") is True
        and support.get("partial_weight_tensor_materialization_ready") is True
        and all(
            isinstance(row, dict)
            and row.get("ready") is True
            and row.get("loads_only_stage_weight_keys") is True
            and row.get("cross_stage_weight_keys_loaded") is False
            for row in stage_rows
        )
    )
    return {
        "ready": ready,
        "report_path": str(path),
        "report_sha256": sha256_file(path) if path.is_file() else "",
        "schema": report.get("schema", ""),
        "model_id": report.get("model_id", ""),
        "execution_family": report.get("execution_family", ""),
        "stage_count": len(stage_rows),
        "stage_selective_weight_loading_ready": report.get("stage_selective_weight_loading_ready") is True,
        "partial_weight_tensor_materialization_ready": bool(support.get("partial_weight_tensor_materialization_ready")),
        "true_partial_weight_loading_ready": bool(support.get("true_partial_weight_loading_ready")),
        "partial_weight_runtime_execution_ready": bool(support.get("partial_weight_runtime_execution_ready")),
        "loaded_weight_key_count_total": sum(
            int(row.get("loaded_weight_key_count") or 0)
            for row in stage_rows
            if isinstance(row, dict)
        ),
        "loaded_tensor_bytes_total": sum(
            int(row.get("loaded_tensor_bytes") or 0)
            for row in stage_rows
            if isinstance(row, dict)
        ),
        "loads_only_stage_weight_keys": bool(
            stage_rows
            and all(isinstance(row, dict) and row.get("loads_only_stage_weight_keys") is True for row in stage_rows)
        ),
        "large_model_validation": False,
        "runtime_execution_validation": False,
        "diagnosis_codes": sorted(codes),
        "blockers": report.get("blockers") or [],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    small_path = Path(args.small_gpu_report)
    seven_path = Path(args.seven_b_blocker_report)
    llama_local_path = Path(args.llama_like_local_report)
    stage_selective_path = Path(args.stage_selective_weight_report)
    small_report, small_meta = try_load(small_path)
    seven_report, seven_meta = try_load(seven_path)
    llama_local_report, llama_local_meta = try_load(llama_local_path)
    stage_selective_report, stage_selective_meta = try_load(stage_selective_path)
    small = summarize_small_gpu(small_report, small_meta, small_path)
    seven = summarize_seven_b_blocker(seven_report, seven_meta, seven_path)
    llama_local = summarize_llama_like_local(llama_local_report, llama_local_meta, llama_local_path)
    stage_selective = summarize_stage_selective_weight_loading(
        stage_selective_report,
        stage_selective_meta,
        stage_selective_path,
    )
    input_leaks = {
        "small_gpu_report": public_redaction_errors(small_report) if small_report else [],
        "seven_b_blocker_report": public_redaction_errors(seven_report) if seven_report else [],
        "llama_like_local_report": public_redaction_errors(llama_local_report) if llama_local_report else [],
        "stage_selective_weight_report": public_redaction_errors(stage_selective_report) if stage_selective_report else [],
    }

    core_ready = bool(
        small["ready"]
        and seven["real_7b_runtime_verified"]
        and seven["sharded_path_verified"]
        and seven["gpu_runtime_verified"]
    )
    diagnosis_codes = [
        "core_small_tier_kaggle_gpu_validated" if small["ready"] else "core_small_tier_kaggle_gpu_not_validated",
        "core_7b_8b_kaggle_validation_ready" if core_ready else "core_7b_8b_kaggle_validation_not_ready",
        "core_technology_validation_ready" if core_ready else "core_technology_validation_incomplete",
    ]
    if seven.get("kaggle_gpu_verified"):
        diagnosis_codes.append("core_7b_8b_kaggle_gpu_hardware_verified")
    if seven.get("container_memory_pressure_not_vram"):
        diagnosis_codes.append("core_7b_8b_kaggle_container_memory_pressure_not_vram")
    if small.get("large_model_blockers") and not stage_selective.get("ready"):
        diagnosis_codes.extend(str(code) for code in small["large_model_blockers"])
    if stage_selective.get("ready"):
        diagnosis_codes.append("core_stage_selective_weight_materialization_validated")

    blockers: list[str] = []
    if not seven.get("real_7b_runtime_verified"):
        blockers.append("core_7b_8b_real_runtime_not_verified")
    if not seven.get("sharded_path_verified"):
        blockers.append("core_7b_8b_sharded_path_not_verified")
    if seven.get("container_memory_pressure_not_vram"):
        blockers.append("kaggle_single_container_memory_pressure")
    if (
        "real_llm_true_partial_weight_loading_missing" in small.get("large_model_blockers", [])
        and not stage_selective.get("ready")
    ):
        blockers.append("real_llm_true_partial_weight_loading_missing")
    if small.get("partial_weight_loading_plan_ready") and not small.get("partial_weight_runtime_execution_ready"):
        blockers.append("real_llm_partial_weight_runtime_execution_missing")
    if stage_selective.get("ready") and not stage_selective.get("partial_weight_runtime_execution_ready"):
        blockers.append("real_llm_partial_weight_runtime_execution_missing")

    report = {
        "schema": SCHEMA,
        "ok": core_ready,
        "generated_at": utc_now(),
        "output_dir": str(output_dir),
        "core_validation_ready": core_ready,
        "small_tier_gpu_validated": bool(small["ready"]),
        "seven_b_eight_b_validated": bool(seven["real_7b_runtime_verified"]),
        "largest_successful_tier": "small" if small["ready"] else "",
        "small_tier_evidence": small,
        "llama_like_local_evidence": llama_local,
        "stage_selective_weight_loading_evidence": stage_selective,
        "seven_b_eight_b_blocker_evidence": seven,
        "blockers": sorted(set(blockers)),
        "diagnosis_codes": sorted(set(diagnosis_codes)),
        "readiness_truth": {
            "do_not_treat_core_layer_complete": not core_ready,
            "small_tier_success_is_not_7b_8b_completion": bool(small["ready"] and not seven["real_7b_runtime_verified"]),
            "partial_weight_plan_is_not_runtime_execution": bool(
                small.get("partial_weight_loading_plan_ready")
                and not small.get("partial_weight_runtime_execution_ready")
            ),
            "stage_selective_weight_loading_is_not_7b_8b_completion": bool(stage_selective.get("ready")),
            "stage_selective_weight_loading_is_not_runtime_execution": bool(
                stage_selective.get("ready")
                and not stage_selective.get("partial_weight_runtime_execution_ready")
            ),
            "thirteen_b_validated": False,
            "production_swarm_inference_claimed": False,
        },
        "input_public_leak_paths": {
            key: value for key, value in input_leaks.items() if value
        },
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "credentials_public": False,
            "private_kaggle_material_public": False,
        },
    }
    leaks = public_redaction_errors(report)
    for key, value in input_leaks.items():
        leaks.extend([f"{key}:{fragment}" for fragment in value])
    report["public_leak_paths"] = leaks
    if leaks:
        report["ok"] = False
        report["core_validation_ready"] = False
        report["diagnosis_codes"].append("core_validation_status_public_leak_detected")
    artifacts = {
        "summary_json": artifact_entry(
            output_dir / "core_technology_validation_status.json",
            output_dir,
            kind="core_technology_validation_status",
            schema=SCHEMA,
            ok=report["ok"],
        ),
        "summary_markdown": artifact_entry(
            output_dir / "core_technology_validation_status.md",
            output_dir,
            kind="core_technology_validation_status_markdown",
        ),
        "support_bundle_json": artifact_entry(
            output_dir / "support_bundle.json",
            output_dir,
            kind="core_technology_validation_status_support_bundle",
            schema=SUPPORT_BUNDLE_SCHEMA,
        ),
    }
    report["artifacts"] = artifacts
    markdown = render_markdown(report)
    (output_dir / "core_technology_validation_status.md").write_text(markdown, encoding="utf-8")
    support = build_support_bundle(report)
    write_json(output_dir / "support_bundle.json", support)
    write_json(output_dir / "core_technology_validation_status.json", report)
    for entry in artifacts.values():
        path = output_dir / str(entry["path"])
        entry["present"] = path.is_file()
        if path.is_file():
            entry["sha256"] = sha256_file(path)
    write_json(output_dir / "core_technology_validation_status.json", report)
    return report


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_support_bundle(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "ok": report.get("ok"),
        "core_validation_ready": report.get("core_validation_ready"),
        "small_tier_gpu_validated": report.get("small_tier_gpu_validated"),
        "stage_selective_weight_loading_ready": (
            report.get("stage_selective_weight_loading_evidence") or {}
        ).get("ready")
        if isinstance(report.get("stage_selective_weight_loading_evidence"), dict)
        else False,
        "seven_b_eight_b_validated": report.get("seven_b_eight_b_validated"),
        "blockers": report.get("blockers"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "safety": report.get("safety"),
        "public_leak_paths": report.get("public_leak_paths"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    small = report.get("small_tier_evidence") if isinstance(report.get("small_tier_evidence"), dict) else {}
    llama_local = report.get("llama_like_local_evidence") if isinstance(report.get("llama_like_local_evidence"), dict) else {}
    stage_selective = report.get("stage_selective_weight_loading_evidence") if isinstance(report.get("stage_selective_weight_loading_evidence"), dict) else {}
    seven = report.get("seven_b_eight_b_blocker_evidence") if isinstance(report.get("seven_b_eight_b_blocker_evidence"), dict) else {}
    lines = [
        "# CrowdTensor Core Technology Validation Status",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- core validation ready: `{report.get('core_validation_ready')}`",
        f"- small-tier Kaggle GPU validated: `{report.get('small_tier_gpu_validated')}`",
        f"- 7B/8B validated: `{report.get('seven_b_eight_b_validated')}`",
        f"- largest successful tier: `{report.get('largest_successful_tier')}`",
        f"- blockers: `{', '.join(report.get('blockers') or [])}`",
        "",
        "## Small-Tier Evidence",
        "",
        f"- report: `{small.get('report_path', '')}`",
        f"- model: `{small.get('model_id', '')}`",
        f"- backend: `{small.get('backend', '')}`",
        f"- generated tokens: `{small.get('generated_token_count', 0)}`",
        f"- cleanup: `{small.get('kaggle_kernels_deleted')}`",
        "",
        "## Llama-Like Local Stage Runtime Evidence",
        "",
        f"- report: `{llama_local.get('report_path', '')}`",
        f"- model: `{llama_local.get('model_id', '')}`",
        f"- backend: `{llama_local.get('backend', '')}`",
        f"- stage-local partition ready: `{llama_local.get('stage_local_partition_ready')}`",
        f"- decoded tokens match: `{llama_local.get('decoded_tokens_match')}`",
        f"- large model validation: `{llama_local.get('large_model_validation')}`",
        "",
        "## Stage-Selective Weight Loading Evidence",
        "",
        f"- report: `{stage_selective.get('report_path', '')}`",
        f"- ready: `{stage_selective.get('ready')}`",
        f"- loaded weight keys: `{stage_selective.get('loaded_weight_key_count_total', 0)}`",
        f"- loads only stage keys: `{stage_selective.get('loads_only_stage_weight_keys')}`",
        f"- runtime execution validation: `{stage_selective.get('runtime_execution_validation')}`",
        f"- large model validation: `{stage_selective.get('large_model_validation')}`",
        "",
        "## 7B/8B Evidence",
        "",
        f"- report: `{seven.get('report_path', '')}`",
        f"- Kaggle GPU verified: `{seven.get('kaggle_gpu_verified')}`",
        f"- GPUs: `{', '.join(seven.get('gpu_names') or [])}`",
        f"- real 7B runtime verified: `{seven.get('real_7b_runtime_verified')}`",
        f"- sharded path verified: `{seven.get('sharded_path_verified')}`",
        f"- container memory pressure not VRAM: `{seven.get('container_memory_pressure_not_vram')}`",
        "",
        "This status is public-safe evidence. It does not include raw prompts, generated text, generated token ids, activations, credentials, private Kaggle material, leases, or idempotency material.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build core technology validation status from retained evidence.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--small-gpu-report", default=DEFAULT_SMALL_GPU_REPORT)
    parser.add_argument("--seven-b-blocker-report", default=DEFAULT_SEVEN_B_BLOCKER_REPORT)
    parser.add_argument("--llama-like-local-report", default=DEFAULT_LLAMA_LIKE_LOCAL_REPORT)
    parser.add_argument("--stage-selective-weight-report", default=DEFAULT_STAGE_SELECTIVE_WEIGHT_REPORT)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(render_markdown(report))


if __name__ == "__main__":
    main()
