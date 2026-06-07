#!/usr/bin/env python3
"""CI-safe checks for the Public Swarm v0.2 Usable Inference Trial."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import public_swarm_trial_pack as pack  # noqa: E402


SCHEMA = "public_swarm_trial_check_v1"
COMMON_REQUIRED_CODES = {
    "public_swarm_trial_ready",
    "public_swarm_trial_user_path_ready",
    "operator_preview_import_ready",
    "support_bundle_ready",
    "cpu_fallback_ready",
    "read_only_workload",
    "not_production",
    "not_p2p",
    "not_libp2p",
    "not_dht",
    "not_nat_traversal",
    "not_gpu_pooling_marketplace",
    "not_large_model_serving",
}
LOCAL_REQUIRED_CODES = COMMON_REQUIRED_CODES | {
    "swarm_trial_local_loopback_ready",
    "serve_join_generate_trial_ready",
    "stage0_join_ready",
    "stage1_join_ready",
    "generate_ready",
    "generated_token_count_ready",
    "private_artifacts_cleaned",
}
PACKAGE_REQUIRED_CODES = COMMON_REQUIRED_CODES | {
    "public_swarm_trial_package_ready",
    "miner_join_pack_ready",
    "private_artifacts_local_only",
}
LIVE_REQUIRED_CODES = COMMON_REQUIRED_CODES | {
    "public_swarm_trial_live_kaggle_ready",
    "external_runtime_verified",
    "kaggle_kernels_deleted",
    "private_artifacts_cleaned",
    "token_rotation_required",
}
EVIDENCE_REQUIRED_CODES = COMMON_REQUIRED_CODES | {
    "public_swarm_trial_evidence_import_ready",
    "operator_preview_retained_evidence_ready",
    "gpu_generation_evidence_import_ready",
    "token_rotation_required",
}
SECRET_FRAGMENTS = [
    "operator-secret",
    "admin-secret",
    "stage0-secret",
    "stage1-secret",
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "Bearer ",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def completed(payload: dict[str, Any], *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


def option_value(command: list[str], option: str, default: str = "") -> str:
    if option not in command:
        return default
    index = command.index(option) + 1
    return command[index] if index < len(command) else default


def mode_from_command(command: list[str], script_name: str, default: str) -> str:
    for index, part in enumerate(command):
        if part.endswith(script_name) and index + 1 < len(command):
            return command[index + 1]
    return default


def fake_product_beta_payload(mode: str) -> dict[str, Any]:
    codes = [
        "public_swarm_product_beta_ready",
        "public_swarm_product_beta_user_path_ready",
        "support_bundle_ready",
        "p2p_lite_route_ready",
        "p2p_lite_discovery_ready",
        "cpu_fallback_ready",
        "local_cpu_inference_ready",
        "read_only_workload",
        "not_production",
        "not_p2p",
        "not_large_model_serving",
    ]
    if mode == "package":
        codes.extend(["public_swarm_product_beta_package_ready", "miner_join_pack_ready", "private_artifacts_local_only"])
    else:
        codes.extend([
            "serve_ready",
            "stage0_join_ready",
            "stage1_join_ready",
            "generate_ready",
            "serve_join_generate_loop_ready",
            "remote_generate_session_ready",
            "public_swarm_generate_ready",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "private_artifacts_cleaned",
        ])
    return {
        "schema": pack.PRODUCT_BETA_SCHEMA,
        "ok": True,
        "mode": mode,
        "product_beta": {
            "ready": True,
            "cpu_fallback_ready": True,
            "workload_type": pack.WORKLOAD_TYPE,
        },
        "generation": {"generated_token_count": 4},
        "diagnosis_codes": codes,
        "payload_summaries": {
            "serve_join_generate": {
                "generated_token_count": 4,
                "diagnosis_codes": ["public_swarm_generate_ready"],
            }
        },
    }


def fake_operator_preview_payload(mode: str) -> dict[str, Any]:
    codes = [
        "public_swarm_operator_preview_ready",
        "operator_preview_user_path_ready",
        "cpu_fallback_ready",
        "live_preview_ready",
        "release_readiness_ready",
        "support_bundle_ready",
        "read_only_workload",
        "not_production",
        "not_p2p",
        "not_large_model_serving",
    ]
    preview = {
        "ready": True,
        "cpu_fallback_ready": True,
        "live_preview_ready": True,
        "release_readiness_ready": True,
        "support_bundle_ready": True,
        "user_path_ready": True,
        "external_runtime_verified": False,
        "external_runtime_blocked": False,
    }
    if mode == "package":
        codes.extend(["operator_preview_package_ready", "miner_join_pack_ready", "private_artifacts_local_only"])
        preview["package_ready"] = True
    elif mode == "evidence-import":
        codes.extend([
            "operator_preview_evidence_import_ready",
            "operator_preview_retained_evidence_ready",
            "operator_preview_retained_evidence_user_path_ready",
            "external_runtime_verified",
            "kaggle_kernels_deleted",
            "token_rotation_required",
        ])
        preview.update({"retained_evidence_ready": True, "external_runtime_verified": True, "token_rotation_required": True})
    elif mode == "live-kaggle":
        codes.extend([
            "operator_preview_live_kaggle_ready",
            "serve_join_generate_ready",
            "external_runtime_verified",
            "kaggle_kernels_deleted",
            "private_artifacts_cleaned",
            "token_rotation_required",
        ])
        preview.update({"serve_join_generate_ready": True, "external_runtime_verified": True, "token_rotation_required": True})
    else:
        codes.extend(["operator_preview_local_smoke_ready", "serve_join_generate_ready", "private_artifacts_cleaned"])
        preview["serve_join_generate_ready"] = True
    return {
        "schema": pack.OPERATOR_PREVIEW_SCHEMA,
        "ok": True,
        "mode": mode,
        "operator_preview": preview,
        "diagnosis_codes": codes,
    }


def fake_gpu_generation_payload(mode: str) -> dict[str, Any]:
    codes = [
        "gpu_sharded_generation_ready",
        "multi_token_generation_ready",
        "gpu_generation_evidence_import_ready",
        "activation_transport_ready",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "read_only_workload",
        "not_production",
        "not_p2p",
    ]
    if mode == "kaggle-auto":
        codes.extend(["gpu_multi_machine_generation_ready", "external_gpu_runtime_verified", "external_runtime_verified", "kaggle_kernels_deleted", "token_rotation_required"])
    else:
        codes.extend(["gpu_loopback_generation_ready", "external_gpu_runtime_verified", "token_rotation_required"])
    return {
        "schema": pack.GPU_GENERATION_SCHEMA,
        "ok": True,
        "mode": mode,
        "generation": {
            "generated_token_count": 16,
            "multi_token_generation_ready": True,
            "raw_generated_text_public": False,
        },
        "diagnosis_codes": codes,
    }


def fake_gpu_report(path: Path) -> None:
    write_json(path, fake_gpu_generation_payload("kaggle-auto"))


def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    output_dir = Path(option_value(command, "--output-dir", tempfile.mkdtemp(prefix="crowdtensor_swarm_trial_step_")))
    output_dir.mkdir(parents=True, exist_ok=True)
    joined = " ".join(command)
    if "public_swarm_product_beta_pack.py" in joined:
        mode = mode_from_command(command, "public_swarm_product_beta_pack.py", "local-loopback")
        payload = fake_product_beta_payload(mode)
        write_json(output_dir / "public_swarm_product_beta.json", payload)
        return completed(payload)
    if "public_swarm_operator_preview_pack.py" in joined:
        mode = mode_from_command(command, "public_swarm_operator_preview_pack.py", "local-smoke")
        payload = fake_operator_preview_payload(mode)
        write_json(output_dir / "public_swarm_operator_preview.json", payload)
        return completed(payload)
    if "gpu_sharded_generation_beta_pack.py" in joined:
        mode = mode_from_command(command, "gpu_sharded_generation_beta_pack.py", "evidence-import")
        payload = fake_gpu_generation_payload(mode)
        write_json(output_dir / f"gpu_sharded_generation_beta_{mode.replace('-', '_')}.json", payload)
        return completed(payload)
    raise AssertionError(command)


def validate_payload(payload: dict[str, Any], *, mode: str, required_codes: set[str]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if payload.get("ok") is not True:
        errors.append("swarm_trial_not_ready")
    if payload.get("mode") != mode:
        errors.append("mode_mismatch")
    trial = payload.get("trial") if isinstance(payload.get("trial"), dict) else {}
    for key in ["ready", "user_path_ready", "support_bundle_ready", "cpu_fallback_ready", "operator_preview_ready"]:
        if trial.get(key) is not True:
            errors.append(f"trial_missing:{key}")
    if mode == pack.MODE_LOCAL_LOOPBACK:
        for key in ["serve_join_generate_trial_ready", "stage0_join_ready", "stage1_join_ready", "generate_ready", "generated_token_count_ready"]:
            if trial.get(key) is not True:
                errors.append(f"trial_missing:{key}")
    codes = set(payload.get("diagnosis_codes") or [])
    for code in sorted(required_codes - codes):
        errors.append(f"missing_code:{code}")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    for key in ["coordinator_backed_task_execution", "p2p_lite_discovery_only", "not_libp2p", "not_dht", "not_nat_traversal", "not_gpu_pooling_marketplace", "not_large_model_serving"]:
        if safety.get(key) is not True:
            errors.append(f"safety_missing:{key}")
    if safety.get("read_only_workload") != pack.WORKLOAD_TYPE:
        errors.append("safety_workload_mismatch")
    output_request = payload.get("output_request") if isinstance(payload.get("output_request"), dict) else {}
    if output_request.get("include_output") is not False:
        errors.append("output_request_include_output_mismatch")
    if output_request.get("raw_prompt_public") is not False:
        errors.append("output_request_raw_prompt_public_mismatch")
    if output_request.get("raw_generated_text_public") is not False:
        errors.append("output_request_raw_generated_text_public_mismatch")
    if output_request.get("generated_token_ids_public") is not False:
        errors.append("output_request_generated_token_ids_public_mismatch")
    if output_request.get("public_artifact_safe") is not True:
        errors.append("output_request_public_artifact_safe_mismatch")
    answer_scope = payload.get("answer_scope") if isinstance(payload.get("answer_scope"), dict) else {}
    if answer_scope.get("scope_state") != "no-local-answer":
        errors.append("answer_scope_state_mismatch")
    if answer_scope.get("visible_in_terminal") is not False:
        errors.append("answer_scope_visible_in_terminal_mismatch")
    if answer_scope.get("terminal_only") is not False:
        errors.append("answer_scope_terminal_only_mismatch")
    if answer_scope.get("saved_json_display") != "hash-only":
        errors.append("answer_scope_saved_json_display_mismatch")
    if answer_scope.get("saved_markdown_display") != "hash-only":
        errors.append("answer_scope_saved_markdown_display_mismatch")
    if answer_scope.get("public_artifact_safe") is not True:
        errors.append("answer_scope_public_artifact_safe_mismatch")
    shareable = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
    if shareable.get("saved_artifacts_public_safe") is not True:
        errors.append("shareable_saved_artifacts_public_safe_mismatch")
    if shareable.get("raw_prompt_public") is not False:
        errors.append("shareable_raw_prompt_public_mismatch")
    if shareable.get("raw_generated_text_public") is not False:
        errors.append("shareable_raw_generated_text_public_mismatch")
    if shareable.get("generated_token_ids_public") is not False:
        errors.append("shareable_generated_token_ids_public_mismatch")
    if shareable.get("answer_scope_state") != "no-local-answer":
        errors.append("shareable_answer_scope_state_mismatch")
    if shareable.get("local_answer_terminal_only") is not False:
        errors.append("shareable_local_answer_terminal_only_mismatch")
    if shareable.get("public_artifact_safe") is not True:
        errors.append("shareable_public_artifact_safe_mismatch")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    for name in ["public_swarm_trial_json", "public_swarm_trial_markdown", "support_bundle_json"]:
        artifact = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
        if artifact.get("present") is not True:
            errors.append(f"missing_artifact:{name}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    return errors


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_swarm_trial_check_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_report = output_dir / "gpu_report.json"
    fake_gpu_report(gpu_report)
    product_report = output_dir / "product_beta.json"
    write_json(product_report, fake_product_beta_payload("local-loopback"))
    operator_report = output_dir / "operator_preview.json"
    write_json(operator_report, fake_operator_preview_payload("evidence-import"))
    stage0_report = output_dir / "stage0_live_preview.json"
    stage1_report = output_dir / "stage1_live_preview.json"
    write_json(stage0_report, fake_operator_preview_payload("live-kaggle"))
    write_json(stage1_report, fake_operator_preview_payload("live-kaggle"))
    release_report = output_dir / "release_readiness.json"
    write_json(release_report, {"schema": "release_readiness_v1", "ok": True, "diagnosis_codes": ["release_ready"]})
    argv = [
        args.mode,
        "--output-dir",
        str(output_dir / "swarm-trial"),
        "--base-port",
        str(args.base_port),
        "--port",
        str(args.port),
        "--release-base-port",
        str(args.base_port + 10),
        "--backend",
        args.backend,
        "--gpu-report",
        str(gpu_report),
        "--product-beta-report",
        str(product_report),
        "--operator-preview-report",
        str(operator_report),
        "--live-stage0-report",
        str(stage0_report),
        "--live-stage1-report",
        str(stage1_report),
        "--release-readiness-report",
        str(release_report),
        "--kaggle-owner",
        "operator",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.timeout_seconds),
        "--cpu-timeout-seconds",
        str(args.timeout_seconds),
        "--release-timeout-seconds",
        str(args.timeout_seconds),
        "--allow-dirty-release",
        "--json",
    ]
    parsed = pack.parse_args(argv)
    report = pack.build_report(parsed, runner=fake_runner)
    required = {
        pack.MODE_LOCAL_LOOPBACK: LOCAL_REQUIRED_CODES,
        pack.MODE_PACKAGE: PACKAGE_REQUIRED_CODES,
        pack.MODE_LIVE_KAGGLE: LIVE_REQUIRED_CODES,
        pack.MODE_EVIDENCE_IMPORT: EVIDENCE_REQUIRED_CODES,
    }[args.mode]
    errors = validate_payload(report, mode=args.mode, required_codes=required)
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "mode": args.mode,
        "backend": args.backend,
        "output_dir": str(output_dir),
        "errors": errors,
        "public_swarm_trial_schema": report.get("schema"),
        "public_swarm_trial_ok": report.get("ok"),
        "diagnosis_codes": ["public_swarm_trial_check_ready"] if not errors else ["public_swarm_trial_check_failed"],
        "artifacts": {
            "public_swarm_trial_json": str(output_dir / "swarm-trial" / "public_swarm_trial.json"),
            "public_swarm_trial_markdown": str(output_dir / "swarm-trial" / "public_swarm_trial.md"),
            "support_bundle_json": str(output_dir / "swarm-trial" / "support_bundle.json"),
        },
    }
    write_json(output_dir / "public_swarm_trial_check.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Public Swarm v0.2 Usable Inference Trial.")
    parser.add_argument("--mode", choices=pack.MODES, default=pack.MODE_LOCAL_LOOPBACK)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--base-port", type=int, default=9380)
    parser.add_argument("--port", type=int, default=9381)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.base_port < 1 or args.port < 1:
        raise SystemExit("--base-port and --port must be positive")
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = run_check(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"Public Swarm Trial check ok={report.get('ok')} errors={report.get('errors')}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
