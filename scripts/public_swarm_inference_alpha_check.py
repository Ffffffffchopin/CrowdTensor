#!/usr/bin/env python3
"""CI-safe contract check for Public Swarm Inference Alpha."""

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

from scripts import public_swarm_inference_alpha_pack as pack  # noqa: E402


SCHEMA = "public_swarm_inference_alpha_check_v1"
SECRET_FRAGMENTS = [
    "operator-secret",
    "admin-secret",
    "stage0-secret",
    "stage1-secret",
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    "real_llm_sharded_result",
    "generated_text",
    "generated_token_ids",
]


def completed(payload: dict[str, Any], *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


def option_value(command: list[str], option: str, default: str = "") -> str:
    if option not in command:
        return default
    index = command.index(option) + 1
    return command[index] if index < len(command) else default


def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    joined = " ".join(command)
    if "real_llm_internet_alpha_pack.py" in joined and option_value(command, "--mode") == "local-generated":
        output_dir = Path(option_value(command, "--output-dir"))
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "real_llm_internet_alpha_v1",
            "ok": True,
            "mode": "local-generated",
            "diagnosis_codes": [
                "real_llm_internet_alpha_ready",
                "real_llm_stage_requeue_ready",
                "stage0_requeue_ready",
                "stage1_requeue_ready",
                "stage_requeue_ready",
                "real_llm_live_rc_ready",
                "remote_real_llm_sharded_ready",
                "real_llm_artifact_ready",
                "activation_transport_ready",
                "baseline_match",
                "decoded_tokens_match",
                "distinct_stage_miners",
                "stage_assignment_valid",
            ],
            "runtime_classification": {
                "local_generated_stage_upload_standins": True,
                "external_runtime_verified": False,
                "stage_requeue_verified": True,
            },
            "payload_summaries": {
                "stage0_requeue": {"diagnosis_codes": ["stage_requeue_ready"], "ok": True},
                "stage1_requeue": {"diagnosis_codes": ["stage_requeue_ready"], "ok": True},
            },
        }
        (output_dir / "real_llm_internet_alpha.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return completed(payload)
    if "swarm_inference_beta_pack.py" in joined and " live " in f" {joined} ":
        output_dir = Path(option_value(command, "--output-dir"))
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_mode = option_value(command, "--failure-mode", "none")
        target = "stage1" if failure_mode == "kill-stage1-after-claim" else "stage0"
        requeue_enabled = failure_mode != "none"
        payload = {
            "schema": "swarm_inference_beta_v1",
            "ok": True,
            "mode": "live",
            "live_mode": "kaggle-auto",
            "coordinator_url": "http://24.199.118.54:9220",
            "diagnosis_codes": [
                "swarm_inference_beta_live_ready",
                "swarm_inference_beta_ready",
                "two_machine_swarm_inference_ready",
                "real_llm_internet_beta_ready",
                "real_llm_internet_alpha_ready",
                "external_runtime_verified",
                "kaggle_kernels_deleted",
                "decoded_tokens_match",
                "distinct_stage_miners",
                "stage_assignment_valid",
                "token_rotation_required",
                "swarm_inference_beta_live_private_artifacts_cleaned",
            ] + ([
                "external_stage_requeue_ready",
                f"live_{target}_requeue_ready",
            ] if requeue_enabled else []),
            "runtime_classification": {
                "kaggle_auto": True,
                "external_runtime_verified": True,
                "stage_requeue_verified": requeue_enabled,
            },
            "workload": {
                "workload_type": "real_llm_sharded_infer",
                "stage_mode": "split",
                "request_count": 2,
                "hf_model_id": "sshleifer/tiny-gpt2",
            },
            "kaggle_lifecycle": {"kernels_deleted": True, "cleanup_required": True},
            "live_requeue_summary": {
                "enabled": requeue_enabled,
                "failure_mode": failure_mode,
                "target_stage": target,
                "claim_observed": requeue_enabled,
                "victim_kernel_deleted": requeue_enabled,
                "lease_expired": requeue_enabled,
                "rescued_result": requeue_enabled,
                "victim_result_accepted": False,
            },
            "real_llm_internet_beta_summary": {
                "ok": True,
                "diagnosis_codes": ["real_llm_internet_beta_ready"] + (["external_stage_requeue_ready"] if requeue_enabled else []),
                "live_requeue_summary": {
                    "enabled": requeue_enabled,
                    "failure_mode": failure_mode,
                    "target_stage": target,
                    "claim_observed": requeue_enabled,
                    "victim_kernel_deleted": requeue_enabled,
                    "lease_expired": requeue_enabled,
                    "rescued_result": requeue_enabled,
                    "victim_result_accepted": False,
                },
            },
        }
        (output_dir / "swarm_inference_beta_live.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        (output_dir / "support_bundle.json").write_text('{"schema":"support_bundle_v1","ok":true}\n', encoding="utf-8")
        return completed(payload)
    raise AssertionError(command)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Public Swarm Inference Alpha without external side effects.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--mode", choices=[pack.MODE_LOCAL_GENERATED, pack.MODE_LIVE_KAGGLE], default=pack.MODE_LIVE_KAGGLE)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_public_swarm_alpha_check_"))
    report = pack.build_report(pack.parse_args([
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        "24.199.118.54",
        "--port",
        "9220",
        "--base-port",
        "9221",
        "--request-count",
        str(args.request_count),
        "--failure-mode",
        "kill-stage0-after-claim",
        "--kaggle-owner",
        "xuyuhaosuyi",
    ]), runner=fake_runner)
    required = {
        "public_swarm_inference_alpha_ready",
        "public_swarm_session_ready",
        "local_stage_requeue_ready",
        "stage_requeue_ready",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
    }
    if args.mode == pack.MODE_LIVE_KAGGLE:
        required.update({
            "public_swarm_live_kaggle_ready",
            "swarm_inference_beta_live_ready",
            "external_runtime_verified",
            "kaggle_kernels_deleted",
            "token_rotation_required",
        })
    codes = set(report.get("diagnosis_codes") or [])
    missing = sorted(required - codes)
    serialized = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in serialized]
    ok = bool(report.get("ok") and not missing and not leaks)
    return {
        "schema": SCHEMA,
        "ok": ok,
        "output_dir": str(output_dir),
        "report_schema": report.get("schema"),
        "report_ok": report.get("ok"),
        "missing_codes": missing,
        "sensitive_leaks": leaks,
        "diagnosis_codes": sorted(codes | ({"public_swarm_inference_alpha_check_ready"} if ok else {"public_swarm_inference_alpha_check_failed"})),
        "artifacts": {
            "public_swarm_inference_alpha_json": pack.artifact_entry(
                output_dir / "public_swarm_inference_alpha.json",
                output_dir,
                kind="public_swarm_inference_alpha",
                schema=pack.SCHEMA,
                ok=report.get("ok"),
            ),
        },
        "limitations": [
            "CI-safe fake-runner check; it does not start a Coordinator, Miner, or Kaggle resource.",
            "Use crowdtensor swarm-session --mode live-kaggle for the side-effectful public proof.",
        ],
    }


def print_human(report: dict[str, Any]) -> None:
    print("CrowdTensor Public Swarm Inference Alpha check")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")


def main() -> None:
    args = parse_args()
    report = build_check(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human(report)
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
