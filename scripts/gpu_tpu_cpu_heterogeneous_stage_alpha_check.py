#!/usr/bin/env python3
"""Validate GPU+TPU+CPU heterogeneous stage-inference Alpha evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gpu_tpu_cpu_heterogeneous_stage_alpha_pack as pack  # noqa: E402


SCHEMA = "gpu_tpu_cpu_heterogeneous_stage_alpha_check_v1"


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    for field in [
        "gpu_tpu_cpu_heterogeneous_stage_alpha_ready",
        "backend_evidence_imported",
        "gpu_backend_evidence_ready",
        "tpu_backend_evidence_ready",
        "cpu_backend_evidence_ready",
        "logical_stage_contract_ready",
        "local_three_stage_real_model_e2e_ready",
        "small_medium_real_model_end_to_end_ready",
        "gpu_tpu_cpu_32b_feasibility_report_ready",
        "next_rc_boundary_ready",
        "public_artifact_safe",
    ]:
        if report.get(field) is not True:
            errors.append(f"{field}_missing")
    for field in [
        "same_request_live_heterogeneous_verified",
        "live_tpu_stage_miner_integrated",
    ]:
        if report.get(field) is not False:
            errors.append(f"{field}_must_remain_false_for_alpha")
    if report.get("execution_mode") not in pack.EXECUTION_MODES:
        errors.append("execution_mode_invalid")

    boundaries = _dict(report.get("boundaries"))
    for name in pack.BOUNDARIES:
        if boundaries.get(name) is not True:
            errors.append(f"boundary_missing:{name}")

    gpu = _dict(report.get("gpu_backend"))
    if gpu.get("gpu_backend_evidence_ready") is not True:
        errors.append("gpu_backend_not_ready")
    if not (gpu.get("full_precision_32b_gpu_cpu_ready") is True or gpu.get("quantized_32b_gpu_upper_bound_ready") is True):
        errors.append("gpu_32b_evidence_missing")
    if gpu.get("coordinator_direct_management_ready") is not True:
        errors.append("gpu_coordinator_direct_management_missing")
    if gpu.get("redaction_ready") is not True:
        errors.append("gpu_redaction_missing")

    tpu = _dict(report.get("tpu_backend"))
    if tpu.get("real_model_tpu_inference_ready") is not True:
        errors.append("tpu_real_model_not_ready")
    if tpu.get("small_medium_real_model_ready") is not True:
        errors.append("tpu_small_medium_model_not_ready")
    if int(tpu.get("tpu_device_count") or 0) < 1:
        errors.append("tpu_device_count_missing")
    if int(tpu.get("generated_token_count") or 0) < 1:
        errors.append("tpu_generated_token_missing")
    if tpu.get("redaction_ready") is not True:
        errors.append("tpu_redaction_missing")
    public_flags = _dict(tpu.get("public_flags"))
    for name, value in public_flags.items():
        if value is not False:
            errors.append(f"tpu_public_flag_mismatch:{name}")

    cpu = _dict(report.get("cpu_backend"))
    if cpu.get("cpu_backend_evidence_ready") is not True:
        errors.append("cpu_backend_not_ready")
    if not (cpu.get("local_cpu_real_llm_sharded_ready") is True or cpu.get("retained_32b_cpu_stage_ready") is True):
        errors.append("cpu_stage_evidence_missing")

    contract = _dict(report.get("stage_contract_smoke"))
    if contract.get("schema") != pack.STAGE_CONTRACT_SCHEMA:
        errors.append("stage_contract_schema_mismatch")
    if contract.get("contract_smoke_ready") is not True:
        errors.append("stage_contract_not_ready")
    if contract.get("same_request_live_heterogeneous_verified") is not False:
        errors.append("stage_contract_overclaims_live_request")
    if contract.get("live_tpu_stage_miner_integrated") is not False:
        errors.append("stage_contract_overclaims_tpu_miner")
    if contract.get("stage_count") != 3:
        errors.append("stage_contract_stage_count_mismatch")
    stage_backends = {str(item.get("backend") or "") for item in _list(contract.get("stage_plan")) if isinstance(item, dict)}
    if stage_backends != {"cuda", "jax_tpu", "cpu"}:
        errors.append("stage_contract_backend_set_mismatch")
    for item in _list(contract.get("handoffs")):
        if isinstance(item, dict):
            if item.get("activation_payload_public") is not False:
                errors.append("stage_contract_activation_public")
            if not str(item.get("activation_hash") or "").startswith("sha256:"):
                errors.append("stage_contract_activation_hash_missing")

    local_e2e = _dict(report.get("local_three_stage_real_model_e2e"))
    if local_e2e.get("schema") != pack.LOCAL_THREE_STAGE_SCHEMA:
        errors.append("local_e2e_schema_mismatch")
    if local_e2e.get("three_stage_real_model_e2e_ready") is not True:
        errors.append("local_e2e_not_ready")
    if local_e2e.get("real_hf_model_loaded") is not True:
        errors.append("local_e2e_model_not_loaded")
    if local_e2e.get("real_model_forward_executed") is not True:
        errors.append("local_e2e_forward_not_executed")
    if local_e2e.get("small_medium_model_e2e_ready") is not True:
        errors.append("local_e2e_small_medium_not_ready")
    if local_e2e.get("baseline_match") is not True:
        errors.append("local_e2e_baseline_mismatch")
    if int(local_e2e.get("generated_token_count") or 0) < 1:
        errors.append("local_e2e_generated_token_missing")
    if local_e2e.get("stage_count") != 3:
        errors.append("local_e2e_stage_count_mismatch")
    local_backends = {str(item.get("target_backend_family") or "") for item in _list(local_e2e.get("stage_plan")) if isinstance(item, dict)}
    if local_backends != {"gpu", "tpu", "cpu"}:
        errors.append("local_e2e_target_backend_set_mismatch")
    for item in _list(local_e2e.get("activation_handoffs")):
        if isinstance(item, dict):
            if item.get("activation_payload_public") is not False:
                errors.append("local_e2e_activation_public")
            if not str(item.get("activation_hash") or "").startswith("sha256:"):
                errors.append("local_e2e_activation_hash_missing")
    for field in [
        "raw_prompt_public",
        "raw_generated_text_public",
        "generated_token_ids_public",
        "activation_public",
        "logits_public",
    ]:
        if local_e2e.get(field) is not False:
            errors.append(f"local_e2e_public_flag_mismatch:{field}")

    bridge_probe = _dict(report.get("torch_jax_torch_bridge_probe"))
    if bridge_probe.get("schema") != pack.TORCH_JAX_BRIDGE_SCHEMA:
        errors.append("bridge_probe_schema_mismatch")
    if bridge_probe.get("mode") not in {"run", "fixture", "skip"}:
        errors.append("bridge_probe_mode_invalid")
    if bridge_probe.get("bridge_ready") is True:
        if bridge_probe.get("baseline_match") is not True:
            errors.append("bridge_probe_baseline_mismatch")
        if int(bridge_probe.get("generated_token_count") or 0) < 1:
            errors.append("bridge_probe_generated_token_missing")
        if bridge_probe.get("torch_stage0_ready") is not True:
            errors.append("bridge_probe_torch_stage0_missing")
        if bridge_probe.get("jax_stage1_ready") is not True:
            errors.append("bridge_probe_jax_stage1_missing")
        if bridge_probe.get("torch_stage2_ready") is not True:
            errors.append("bridge_probe_torch_stage2_missing")
    for field in [
        "activation_public",
        "generated_token_ids_public",
        "logits_public",
    ]:
        if bridge_probe.get(field) is not False:
            errors.append(f"bridge_probe_public_flag_mismatch:{field}")

    feasibility = _dict(report.get("heterogeneous_32b_feasibility"))
    if feasibility.get("schema") != pack.FEASIBILITY_SCHEMA:
        errors.append("feasibility_schema_mismatch")
    if feasibility.get("gpu_tpu_cpu_32b_feasibility_report_ready") is not True:
        errors.append("feasibility_report_not_ready")
    if feasibility.get("same_request_live_heterogeneous_verified") is not False:
        errors.append("feasibility_overclaims_same_request")
    if feasibility.get("gpu_tpu_cpu_32b_same_request_feasible_now") is not False:
        errors.append("feasibility_overclaims_32b_now")
    if feasibility.get("tpu_32b_runtime_adapter_ready") is not False:
        errors.append("feasibility_overclaims_tpu_32b_adapter")
    if feasibility.get("ready_for_bounded_rc") is not True:
        errors.append("feasibility_rc_not_ready")
    if feasibility.get("local_three_stage_real_model_e2e_ready") is not True:
        errors.append("feasibility_local_e2e_missing")
    if feasibility.get("verdict") != "ready_for_bounded_rc_not_yet_live_verified":
        errors.append("feasibility_verdict_mismatch")
    next_rc = _dict(feasibility.get("next_rc_boundary"))
    if next_rc.get("next_rc_boundary_ready") is not True:
        errors.append("next_rc_boundary_not_ready")
    if not next_rc.get("success_criteria"):
        errors.append("next_rc_success_criteria_missing")
    required_items = {str(item.get("item") or "") for item in _list(feasibility.get("required_adapter_work")) if isinstance(item, dict)}
    for name in [
        "jax_tpu_llama_like_stage_runtime",
        "safetensors_or_maxtext_checkpoint_bridge",
        "cuda_to_jax_activation_wire_format",
        "stage_local_kv_cache_format_boundary",
        "coordinator_backend_capability_routing",
    ]:
        if name not in required_items:
            errors.append(f"required_adapter_work_missing:{name}")

    safety = _dict(report.get("safety"))
    for field in [
        "raw_prompt_public",
        "raw_generated_text_public",
        "generated_token_ids_public",
        "activation_public",
        "hidden_state_public",
        "logits_public",
        "kv_cache_public",
        "past_key_values_public",
        "credentials_public",
        "cookies_public",
        "lease_material_public",
        "idempotency_material_public",
        "private_runtime_state_public",
    ]:
        if safety.get(field) is not False:
            errors.append(f"safety_flag_mismatch:{field}")
    if safety.get("public_artifact_safe") is not True:
        errors.append("safety_public_artifact_safe_mismatch")
    if safety.get("report_public_leak_paths"):
        errors.append("report_public_leak_paths_present")
    leaks = pack.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    diagnosis = set(report.get("diagnosis_codes") or [])
    for code in [
        "gpu_tpu_cpu_heterogeneous_stage_alpha_ready",
        "backend_evidence_imported",
        "gpu_backend_evidence_ready",
        "tpu_backend_evidence_ready",
        "cpu_backend_evidence_ready",
        "logical_stage_contract_ready",
        "local_three_stage_real_model_e2e_ready",
        "small_medium_real_model_alpha_path_ready",
        "same_request_live_heterogeneous_not_verified",
        "tpu_stage_miner_not_integrated",
        "gpu_tpu_cpu_32b_feasibility_report_ready",
        "next_rc_boundary_ready",
        "gpu_tpu_cpu_public_artifact_redaction_ready",
    ]:
        if code not in diagnosis:
            errors.append(f"diagnosis_missing:{code}")

    artifacts = _dict(report.get("artifacts"))
    for name in [
        "summary_json",
        "summary_markdown",
        "support_bundle_json",
        "stage_contract_smoke_json",
        "local_three_stage_real_model_e2e_json",
        "heterogeneous_32b_feasibility_json",
    ]:
        if _dict(artifacts.get(name)).get("present") is not True:
            errors.append(f"artifact_missing:{name}")
    return errors


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    if args.report:
        report = load_json(Path(args.report))
    else:
        pack_args = pack.parse_args([
            "--output-dir",
            args.output_dir,
            "--execution-mode",
            args.execution_mode,
            "--tpu-real-llm-report",
            args.tpu_real_llm_report,
            "--gpu-full-32b-report",
            args.gpu_full_32b_report,
            "--gpu-awq-32b-report",
            args.gpu_awq_32b_report,
            "--cpu-real-llm-report",
            args.cpu_real_llm_report,
            "--small-medium-min-parameter-count",
            str(args.small_medium_min_parameter_count),
            "--local-e2e-mode",
            args.local_e2e_mode,
            "--local-e2e-model-id",
            args.local_e2e_model_id,
            "--bridge-mode",
            args.bridge_mode,
            "--bridge-model-id",
            args.bridge_model_id,
            "--target-max-new-tokens",
            str(args.target_max_new_tokens),
            "--context-length",
            str(args.context_length),
        ])
        report = pack.build_report(pack_args)
    errors = validate_report(report)
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "report_schema": report.get("schema"),
        "output_dir": report.get("output_dir") or args.output_dir,
        "report_path": args.report or str(Path(args.output_dir) / "gpu_tpu_cpu_heterogeneous_stage_alpha.json"),
        "gpu_tpu_cpu_heterogeneous_stage_alpha_ready": report.get("gpu_tpu_cpu_heterogeneous_stage_alpha_ready") is True,
        "small_medium_real_model_end_to_end_ready": report.get("small_medium_real_model_end_to_end_ready") is True,
        "same_request_live_heterogeneous_verified": report.get("same_request_live_heterogeneous_verified") is True,
        "local_three_stage_real_model_e2e_ready": report.get("local_three_stage_real_model_e2e_ready") is True,
        "gpu_tpu_cpu_32b_feasibility_report_ready": report.get("gpu_tpu_cpu_32b_feasibility_report_ready") is True,
        "next_rc_boundary_ready": report.get("next_rc_boundary_ready") is True,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "execution_mode": report.get("execution_mode"),
        "errors": errors,
        "diagnosis_codes": ["gpu_tpu_cpu_heterogeneous_stage_alpha_check_ready"] if not errors else ["gpu_tpu_cpu_heterogeneous_stage_alpha_check_failed"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GPU+TPU+CPU heterogeneous stage inference Alpha evidence.")
    parser.add_argument("--report", default="")
    parser.add_argument("--output-dir", default=pack.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execution-mode", choices=pack.EXECUTION_MODES, default="fixture")
    parser.add_argument("--tpu-real-llm-report", default=pack.DEFAULT_TPU_REAL_LLM_REPORT)
    parser.add_argument("--gpu-full-32b-report", default=pack.DEFAULT_GPU_FULL_32B_REPORT)
    parser.add_argument("--gpu-awq-32b-report", default=pack.DEFAULT_GPU_AWQ_32B_REPORT)
    parser.add_argument("--cpu-real-llm-report", default=pack.DEFAULT_CPU_REAL_LLM_REPORT)
    parser.add_argument("--small-medium-min-parameter-count", type=int, default=100_000_000)
    parser.add_argument("--local-e2e-mode", choices=["run", "fixture", "skip"], default="fixture")
    parser.add_argument("--local-e2e-model-id", default="gpt2")
    parser.add_argument("--bridge-mode", choices=["run", "fixture", "skip"], default="fixture")
    parser.add_argument("--bridge-model-id", default="hf-internal-testing/tiny-random-gpt2")
    parser.add_argument("--target-max-new-tokens", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.report and not Path(args.report).is_file():
        raise SystemExit("--report must point to an existing JSON file")
    if args.small_medium_min_parameter_count < 1:
        raise SystemExit("--small-medium-min-parameter-count must be positive")
    if args.target_max_new_tokens < 1 or args.target_max_new_tokens > 16:
        raise SystemExit("--target-max-new-tokens must be between 1 and 16")
    if args.context_length < 1 or args.context_length > 4096:
        raise SystemExit("--context-length must be between 1 and 4096")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"GPU+TPU+CPU heterogeneous stage Alpha check ready: {result.get('ok')}")
        if result.get("errors"):
            print("errors: " + ", ".join(result.get("errors") or []))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
