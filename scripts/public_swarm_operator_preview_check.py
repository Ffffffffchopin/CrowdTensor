#!/usr/bin/env python3
"""CI-safe checks for Public Swarm v0.1 Operator Preview."""

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

import public_swarm_operator_preview_pack as pack  # noqa: E402


SCHEMA = "public_swarm_operator_preview_check_v1"
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
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    "real_llm_sharded_result",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
    "Bearer ",
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


def fake_gpu_report(path: Path) -> None:
    write_json(path, {
        "schema": pack.GPU_GENERATION_SCHEMA,
        "ok": True,
        "diagnosis_codes": ["gpu_sharded_generation_ready", "multi_token_generation_ready"],
        "generated_token_count": 16,
        "generated_text_hash": "sha256:synthetic",
        "raw_generated_text_public": False,
    })


def fake_developer_payload(mode: str = "local") -> dict[str, Any]:
    codes = [
        "developer_preview_ready",
        "public_swarm_developer_preview_ready",
        "product_beta_ready",
        "support_bundle_ready",
        "cpu_fallback_ready",
        "local_cpu_inference_ready",
        "read_only_workload",
        "not_production",
        "not_p2p",
        "not_large_model_serving",
    ]
    if mode == "package":
        codes.extend(["developer_preview_package_ready", "miner_join_pack_ready", "private_artifacts_local_only"])
    else:
        codes.extend(["developer_preview_local_ready", "local_two_stage_generation_ready", "serve_join_generate_ready"])
    return {
        "schema": pack.DEVELOPER_PREVIEW_SCHEMA,
        "ok": True,
        "mode": mode,
        "developer_preview": {
            "ready": True,
            "product_beta_ready": True,
            "support_bundle_ready": True,
            "cpu_fallback_ready": True,
            "workload_type": pack.WORKLOAD_TYPE,
        },
        "diagnosis_codes": codes,
    }


def fake_live_payload(*, mode: str = "local-smoke", stage: str = "stage0", fresh: bool = False) -> dict[str, Any]:
    codes = [
        "public_swarm_live_preview_rc_ready",
        "public_swarm_live_preview_local_smoke_ready" if mode == "local-smoke" else "public_swarm_live_preview_evidence_import_ready",
        "developer_preview_ready",
        "public_swarm_developer_preview_ready",
        "support_bundle_ready",
        "cpu_fallback_ready",
        "read_only_workload",
        "not_production",
        "not_p2p",
        "not_large_model_serving",
        "external_runtime_verified",
        "kaggle_kernels_deleted",
        "private_artifacts_cleaned",
        "token_rotation_required",
    ]
    if mode == "live-kaggle":
        codes.append("public_swarm_live_preview_live_kaggle_ready")
    if stage == "stage0":
        codes.append("live_stage0_requeue_ready")
    if stage == "stage1":
        codes.append("live_stage1_requeue_ready")
    if stage == "both":
        codes.extend(["live_stage0_requeue_ready", "live_stage1_requeue_ready"])
    return {
        "schema": pack.LIVE_PREVIEW_SCHEMA,
        "ok": True,
        "mode": mode,
        "diagnosis_codes": codes,
        "live_preview": {
            "ready": True,
            "external_runtime_verified": True,
            "fresh_live_kaggle_run": fresh,
            "stage0_live_requeue_ready": stage in {"stage0", "both"},
            "stage1_live_requeue_ready": stage in {"stage1", "both"},
            "kaggle_kernels_deleted": True,
            "private_artifacts_cleaned": True,
            "token_rotation_required": True,
        },
    }


def fake_release_payload() -> dict[str, Any]:
    return {
        "schema": pack.RELEASE_READINESS_SCHEMA,
        "ok": True,
        "release_status": {
            "ready": True,
            "status": "ready",
            "diagnosis_codes": ["release_ready"],
            "warnings": [],
        },
        "diagnosis_codes": ["release_ready"],
    }


def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    joined = " ".join(command)
    output_dir = Path(option_value(command, "--output-dir", tempfile.mkdtemp(prefix="crowdtensor_operator_check_step_")))
    output_dir.mkdir(parents=True, exist_ok=True)
    if "public_swarm_developer_preview_pack.py" in joined:
        mode = command[2] if len(command) > 2 else "local"
        payload = fake_developer_payload(mode)
        write_json(output_dir / "public_swarm_developer_preview.json", payload)
        return completed(payload)
    if "public_swarm_live_preview_rc_pack.py" in joined:
        mode = command[2] if len(command) > 2 else "local-smoke"
        if mode == "live-kaggle" and "--force-live-failure" in command:
            return completed({
                "schema": pack.LIVE_PREVIEW_SCHEMA,
                "ok": False,
                "mode": mode,
                "diagnosis_codes": ["hf_dependencies_missing"],
            }, returncode=1)
        payload = fake_live_payload(mode=mode, stage="stage0", fresh=mode == "live-kaggle")
        write_json(output_dir / "public_swarm_live_preview_rc.json", payload)
        return completed(payload)
    if "release_readiness_pack.py" in joined:
        payload = fake_release_payload()
        write_json(output_dir / "release_readiness.json", payload)
        return completed(payload)
    raise AssertionError(command)


def validate_payload(payload: dict[str, Any], *, mode: str) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if payload.get("ok") is not True:
        errors.append("operator_preview_not_ready")
    if payload.get("mode") != mode:
        errors.append("mode_mismatch")
    preview = payload.get("operator_preview") if isinstance(payload.get("operator_preview"), dict) else {}
    for key in [
        "ready",
        "cpu_fallback_ready",
        "live_preview_ready",
        "release_readiness_ready",
        "support_bundle_ready",
    ]:
        if preview.get(key) is not True:
            errors.append(f"operator_preview_missing:{key}")
    if preview.get("developer_preview_ready") is not True and preview.get("developer_preview_degraded") is not True:
        errors.append("operator_preview_missing:developer_preview_ready_or_degraded")
    if preview.get("user_path_ready") is not True:
        errors.append("operator_preview_missing:user_path_ready")
    if (
        preview.get("serve_join_generate_ready") is not True
        and preview.get("package_ready") is not True
        and preview.get("cpu_fallback_user_path_ready") is not True
        and preview.get("retained_evidence_user_path_ready") is not True
    ):
        errors.append("operator_preview_missing:user_path")
    codes = set(payload.get("diagnosis_codes") or [])
    required = {
        "public_swarm_operator_preview_ready",
        "operator_preview_user_path_ready",
        "cpu_fallback_ready",
        "live_preview_ready",
        "support_bundle_ready",
        "release_readiness_ready",
        "read_only_workload",
        "not_production",
        "not_p2p",
        "not_large_model_serving",
    }
    if mode == pack.MODE_LOCAL_SMOKE:
        required.add("operator_preview_local_smoke_ready")
    if mode == pack.MODE_PACKAGE:
        required.add("operator_preview_package_ready")
        required.add("miner_join_pack_ready")
        required.add("private_artifacts_local_only")
    else:
        required.add("serve_join_generate_ready")
    if mode == pack.MODE_LIVE_KAGGLE:
        required.add("operator_preview_live_kaggle_ready")
    if mode == pack.MODE_EVIDENCE_IMPORT:
        required.add("operator_preview_evidence_import_ready")
    for code in sorted(required - codes):
        errors.append(f"missing_code:{code}")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    for key in ["coordinator_backed_task_execution", "operator_preview_user_path", "not_libp2p", "not_dht", "not_nat_traversal", "not_large_model_serving"]:
        if safety.get(key) is not True:
            errors.append(f"safety_missing:{key}")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    for name in ["public_swarm_operator_preview_json", "public_swarm_operator_preview_markdown", "support_bundle_json"]:
        artifact = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
        if artifact.get("present") is not True:
            errors.append(f"missing_artifact:{name}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    return errors


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_operator_preview_check_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_report = output_dir / "gpu_report.json"
    fake_gpu_report(gpu_report)
    release_report = output_dir / "release_readiness.json"
    write_json(release_report, fake_release_payload())
    developer_report = output_dir / "developer_preview_import.json"
    write_json(developer_report, fake_developer_payload("local"))
    stage0_report = output_dir / "stage0_live_preview.json"
    stage1_report = output_dir / "stage1_live_preview.json"
    write_json(stage0_report, fake_live_payload(mode="live-kaggle", stage="stage0", fresh=False))
    write_json(stage1_report, fake_live_payload(mode="live-kaggle", stage="stage1", fresh=False))

    argv = [
        args.mode,
        "--output-dir",
        str(output_dir / "operator-preview"),
        "--base-port",
        str(args.base_port),
        "--port",
        str(args.port),
        "--release-base-port",
        str(args.base_port + 10),
        "--gpu-report",
        str(gpu_report),
        "--developer-preview-report",
        str(developer_report),
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
        "--release-timeout-seconds",
        str(args.timeout_seconds),
        "--allow-dirty-release",
        "--json",
    ]
    parsed = pack.parse_args(argv)
    report = pack.build_report(parsed, runner=fake_runner)
    errors = validate_payload(report, mode=args.mode)
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "operator_preview_ok": report.get("ok"),
        "operator_preview_schema": report.get("schema"),
        "diagnosis_codes": ["public_swarm_operator_preview_check_ready"] if not errors else ["public_swarm_operator_preview_check_failed"],
        "errors": errors,
        "artifacts": {
            "public_swarm_operator_preview_json": str(output_dir / "operator-preview" / "public_swarm_operator_preview.json"),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Public Swarm v0.1 Operator Preview.")
    parser.add_argument("--mode", choices=[pack.MODE_LOCAL_SMOKE, pack.MODE_PACKAGE, pack.MODE_LIVE_KAGGLE, pack.MODE_EVIDENCE_IMPORT], default=pack.MODE_LOCAL_SMOKE)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--base-port", type=int, default=9370)
    parser.add_argument("--port", type=int, default=9371)
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
        print(f"Public Swarm Operator Preview check ok={report.get('ok')} errors={report.get('errors')}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
