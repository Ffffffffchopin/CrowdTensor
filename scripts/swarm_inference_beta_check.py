#!/usr/bin/env python3
"""CI-safe contract check for the user-facing Swarm Inference Beta wrapper."""

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

from scripts import swarm_inference_beta_pack as pack  # noqa: E402


SCHEMA = "swarm_inference_beta_check_v1"
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
    if "real_llm_internet_beta_pack.py" in joined:
        output_dir = Path(option_value(command, "--output-dir"))
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_mode = option_value(command, "--failure-mode", "none")
        requeue_codes = []
        if failure_mode != "none":
            target = "stage1" if failure_mode == "kill-stage1-after-claim" else "stage0"
            requeue_codes = [
                "external_stage_requeue_ready",
                f"live_{target}_requeue_ready",
                "live_requeue_victim_claim_observed",
                "live_requeue_victim_kernel_deleted",
                "live_requeue_lease_timeout_observed",
                "live_requeue_rescue_result_accepted",
            ]
        payload = {
            "schema": "real_llm_internet_beta_v1",
            "ok": True,
            "mode": "kaggle-auto",
            "coordinator_url": "http://24.199.118.54:9210",
            "diagnosis_codes": [
                "real_llm_internet_beta_ready",
                "real_llm_internet_alpha_ready",
                "external_runtime_verified",
                "kaggle_real_llm_stage0_seen",
                "kaggle_real_llm_stage1_seen",
                "decoded_tokens_match",
                "distinct_stage_miners",
                "stage_assignment_valid",
                "kaggle_kernels_deleted",
                "token_rotation_required",
                *requeue_codes,
            ],
            "runtime_classification": {
                "kaggle_auto": True,
                "external_runtime_verified": True,
                "kaggle_notebook_verified": True,
                "stage_requeue_verified": failure_mode != "none",
            },
            "workload": {
                "workload_type": "real_llm_sharded_infer",
                "stage_mode": "split",
                "request_count": 2,
                "hf_model_id": "sshleifer/tiny-gpt2",
                "require_distinct_stage_miners": True,
            },
            "kaggle_lifecycle": {
                "kernels_deleted": True,
                "cleanup_required": True,
                "cleanup_skipped": False,
                "pushed_refs": {"stage0": "owner/stage0", "stage1": "owner/stage1"},
                "stage_refs": {"stage0": "owner/stage0", "stage1": "owner/stage1"},
            },
            "live_requeue_summary": {
                "enabled": failure_mode != "none",
                "failure_mode": failure_mode,
                "target_stage": "stage1" if failure_mode == "kill-stage1-after-claim" else ("stage0" if failure_mode != "none" else ""),
                "claim_observed": failure_mode != "none",
                "victim_kernel_deleted": failure_mode != "none",
                "lease_expired": failure_mode != "none",
                "rescued_result": failure_mode != "none",
                "victim_result_accepted": False,
            },
            "artifacts": {
                "kept_json": {
                    "kind": "kept",
                    "path": "real_llm_internet_beta.json",
                    "present": True,
                },
                "deleted_runtime_state": {
                    "kind": "transient",
                    "path": "alpha-package/package-live-rc/real_llm_live_rc.json",
                    "present": True,
                },
            },
        }
        deleted_path = output_dir / "alpha-package" / "package-live-rc" / "real_llm_live_rc.json"
        deleted_path.parent.mkdir(parents=True, exist_ok=True)
        deleted_path.write_text("{}", encoding="utf-8")
        (output_dir / "real_llm_internet_beta.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        (output_dir / "real_llm_internet_beta.md").write_text("# Real Internet Beta\n", encoding="utf-8")
        return completed(payload)
    if "support_bundle.py" in joined:
        json_out = Path(option_value(command, "--json-out"))
        markdown_out = Path(option_value(command, "--markdown-out"))
        json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "support_bundle_v1",
            "doctor": {"ok": True},
            "release_gate": {"ok": True},
            "reports": {"remote": {"ok": True, "present": True}},
        }
        json_out.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        markdown_out.write_text("# Support Bundle\n", encoding="utf-8")
        return completed(payload)
    if "remote_real_llm_sharded_beta_pack.py" in joined:
        output_dir = Path(option_value(command, "--output-dir"))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "remote_real_llm_sharded_beta.json").write_text("{}", encoding="utf-8")
        (output_dir / "remote_real_llm_sharded_beta.md").write_text("# Remote Real LLM\n", encoding="utf-8")
        return completed({
            "schema": "remote_real_llm_sharded_beta_v1",
            "ok": True,
            "mode": "remote-existing",
            "diagnosis_codes": [
                "remote_real_llm_sharded_ready",
                "remote_real_llm_sharded_existing_ready",
                "real_llm_sharded_ready",
                "real_llm_artifact_ready",
                "activation_transport_ready",
                "baseline_match",
                "decoded_tokens_match",
                "distinct_stage_miners",
                "stage_assignment_valid",
            ],
            "payload_summaries": {
                "remote_existing_real_llm_sharded_inference": {
                    "schema": "real_llm_sharded_evidence_v1",
                    "ok": True,
                    "diagnosis_codes": [
                        "real_llm_sharded_ready",
                        "decoded_tokens_match",
                        "distinct_stage_miners",
                        "stage_assignment_valid",
                    ],
                    "session": {
                        "session_id": "session-test",
                        "stage_count": 2,
                        "request_count": 2,
                        "model_id": "sshleifer/tiny-gpt2",
                        "artifact_hash": "sha256:artifact",
                    },
                    "artifact": {
                        "schema": "real_llm_artifact_v1",
                        "backend": "hf_transformers_cpu",
                        "model_id": "sshleifer/tiny-gpt2",
                        "artifact_hash": "sha256:artifact",
                    },
                    "stage_assignment": {
                        "stage0_miner_id": "swarm-beta-stage0",
                        "stage1_miner_id": "swarm-beta-stage1",
                        "distinct_stage_miners": True,
                        "stage_assignment_valid": True,
                    },
                    "safety": {
                        "read_only": True,
                        "redaction_ok": True,
                        "raw_activation_redacted": True,
                    },
                },
            },
            "artifacts": {},
        })
    if "remote_home_compute_demo_pack.py" in joined and " collect " in f" {joined} ":
        output_dir = Path(option_value(command, "--output-dir"))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "remote_home_compute_collect.json").write_text("{}", encoding="utf-8")
        (output_dir / "support_bundle.json").write_text("{}", encoding="utf-8")
        return completed({
            "schema": "remote_home_compute_collect_v1",
            "ok": True,
            "workload_kind": "real-llm-sharded",
            "diagnosis_codes": ["remote_home_compute_collect_ready", "remote_real_llm_sharded_ready"],
            "status_summary": {"ready": True, "matched_capabilities": ["stage_0_completed", "stage_1_completed"]},
            "evidence_summary": {"schema": "remote_real_llm_sharded_beta_v1", "ok": True},
            "support_bundle_summary": {"schema": "support_bundle_v1", "ok": True},
        })
    raise AssertionError(command)


def write_external_beta(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "real_llm_internet_beta_v1",
        "ok": True,
        "mode": "kaggle-auto",
        "diagnosis_codes": [
            "real_llm_internet_beta_ready",
            "external_runtime_verified",
            "kaggle_kernels_deleted",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
        ],
        "runtime_classification": {"external_runtime_verified": True, "kaggle_auto": True},
        "kaggle_lifecycle": {"kernels_deleted": True},
    }, sort_keys=True), encoding="utf-8")


def output_scope_errors(label: str, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    output_request = payload.get("output_request") if isinstance(payload.get("output_request"), dict) else {}
    if output_request.get("include_output") is not False:
        errors.append(f"{label}:output_request_include_output_mismatch")
    if output_request.get("raw_prompt_public") is not False:
        errors.append(f"{label}:output_request_raw_prompt_public_mismatch")
    if output_request.get("raw_generated_text_public") is not False:
        errors.append(f"{label}:output_request_raw_generated_text_public_mismatch")
    if output_request.get("generated_token_ids_public") is not False:
        errors.append(f"{label}:output_request_generated_token_ids_public_mismatch")
    if output_request.get("local_output_display_only") is not False:
        errors.append(f"{label}:output_request_local_output_display_only_mismatch")
    if output_request.get("public_artifact_safe") is not True:
        errors.append(f"{label}:output_request_public_artifact_safe_mismatch")
    answer_scope = payload.get("answer_scope") if isinstance(payload.get("answer_scope"), dict) else {}
    if answer_scope.get("scope_state") != "no-local-answer":
        errors.append(f"{label}:answer_scope_state_mismatch")
    if answer_scope.get("visible_in_terminal") is not False:
        errors.append(f"{label}:answer_scope_visible_in_terminal_mismatch")
    if answer_scope.get("terminal_only") is not False:
        errors.append(f"{label}:answer_scope_terminal_only_mismatch")
    if answer_scope.get("saved_json_display") != "hash-only":
        errors.append(f"{label}:answer_scope_saved_json_display_mismatch")
    if answer_scope.get("saved_markdown_display") != "hash-only":
        errors.append(f"{label}:answer_scope_saved_markdown_display_mismatch")
    if answer_scope.get("raw_prompt_public") is not False:
        errors.append(f"{label}:answer_scope_raw_prompt_public_mismatch")
    if answer_scope.get("raw_generated_text_public") is not False:
        errors.append(f"{label}:answer_scope_raw_generated_text_public_mismatch")
    if answer_scope.get("generated_token_ids_public") is not False:
        errors.append(f"{label}:answer_scope_generated_token_ids_public_mismatch")
    if answer_scope.get("public_artifact_safe") is not True:
        errors.append(f"{label}:answer_scope_public_artifact_safe_mismatch")
    shareable = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
    if shareable.get("saved_artifacts_public_safe") is not True:
        errors.append(f"{label}:shareable_saved_artifacts_public_safe_mismatch")
    if shareable.get("raw_prompt_public") is not False:
        errors.append(f"{label}:shareable_raw_prompt_public_mismatch")
    if shareable.get("raw_generated_text_public") is not False:
        errors.append(f"{label}:shareable_raw_generated_text_public_mismatch")
    if shareable.get("generated_token_ids_public") is not False:
        errors.append(f"{label}:shareable_generated_token_ids_public_mismatch")
    if shareable.get("local_output_display_only") is not False:
        errors.append(f"{label}:shareable_local_output_display_only_mismatch")
    if shareable.get("answer_scope_state") != "no-local-answer":
        errors.append(f"{label}:shareable_answer_scope_state_mismatch")
    if shareable.get("local_answer_terminal_only") is not False:
        errors.append(f"{label}:shareable_local_answer_terminal_only_mismatch")
    if shareable.get("public_artifact_safe") is not True:
        errors.append(f"{label}:shareable_public_artifact_safe_mismatch")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Swarm Inference Beta without external side effects.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--coordinator-url", default="http://127.0.0.1:9200")
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_swarm_inference_beta_check_"))
    external_report = output_dir / "real_llm_internet_beta.json"
    write_external_beta(external_report)

    prepare = pack.build_report(pack.parse_args([
        "prepare",
        "--output-dir",
        str(output_dir),
        "--coordinator-url",
        args.coordinator_url,
        "--request-count",
        str(args.request_count),
        "--observer-token",
        "operator-secret",
        "--admin-token",
        "admin-secret",
    ]))
    verify = pack.build_report(pack.parse_args([
        "verify",
        "--output-dir",
        str(output_dir),
        "--coordinator-url",
        args.coordinator_url,
        "--request-count",
        str(args.request_count),
        "--observer-token",
        "operator-secret",
        "--admin-token",
        "admin-secret",
        "--real-internet-beta-report",
        str(external_report),
    ]), runner=fake_runner)
    collect = pack.build_report(pack.parse_args([
        "collect",
        "--output-dir",
        str(output_dir),
        "--coordinator-url",
        args.coordinator_url,
        "--request-count",
        str(args.request_count),
        "--observer-token",
        "operator-secret",
        "--admin-token",
        "admin-secret",
    ]), runner=fake_runner)
    live = pack.build_report(pack.parse_args([
        "live",
        "--output-dir",
        str(output_dir / "live"),
        "--public-host",
        "24.199.118.54",
        "--port",
        "9210",
        "--base-port",
        "9211",
        "--request-count",
        str(args.request_count),
        "--kaggle-owner",
        "xuyuhaosuyi",
    ]), runner=fake_runner)
    clean = pack.build_report(pack.parse_args(["clean", "--output-dir", str(output_dir)]))

    required = {
        "swarm_inference_beta_ready",
        "real_llm_split_route_ready",
        "two_machine_swarm_inference_ready",
        "external_beta_evidence_imported",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
    }
    live_required = {
        "swarm_inference_beta_ready",
        "swarm_inference_beta_live_ready",
        "two_machine_swarm_inference_ready",
        "real_llm_internet_beta_ready",
        "real_llm_internet_alpha_ready",
        "external_runtime_verified",
        "kaggle_kernels_deleted",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "token_rotation_required",
    }
    verify_codes = set(verify.get("diagnosis_codes") or [])
    live_codes = set(live.get("diagnosis_codes") or [])
    missing = sorted((required - verify_codes) | (live_required - live_codes))
    serialized = json.dumps({"prepare": prepare, "verify": verify, "collect": collect, "live": live, "clean": clean}, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in serialized]
    scope_errors = (
        output_scope_errors("prepare", prepare)
        + output_scope_errors("verify", verify)
        + output_scope_errors("collect", collect)
        + output_scope_errors("live", live)
        + output_scope_errors("clean", clean)
    )
    ok = bool(
        prepare.get("ok")
        and verify.get("ok")
        and collect.get("ok")
        and live.get("ok")
        and clean.get("ok")
        and not missing
        and not leaks
        and not scope_errors
    )
    return {
        "schema": SCHEMA,
        "ok": ok,
        "output_dir": str(output_dir),
        "prepare_ok": prepare.get("ok"),
        "verify_ok": verify.get("ok"),
        "collect_ok": collect.get("ok"),
        "live_ok": live.get("ok"),
        "clean_ok": clean.get("ok"),
        "missing_codes": missing,
        "sensitive_leaks": leaks,
        "scope_errors": scope_errors,
        "diagnosis_codes": sorted((verify_codes | live_codes) | {"swarm_inference_beta_check_ready"} if ok else (verify_codes | live_codes) | {"swarm_inference_beta_check_failed"}),
        "artifacts": {
            "swarm_inference_beta_prepare_json": pack.artifact_entry(
                output_dir / "swarm_inference_beta_prepare.json",
                output_dir,
                kind="swarm_inference_beta_prepare",
                schema=pack.SCHEMA,
                ok=prepare.get("ok"),
            ),
            "swarm_inference_beta_verify_json": pack.artifact_entry(
                output_dir / "swarm_inference_beta_verify.json",
                output_dir,
                kind="swarm_inference_beta_verify",
                schema=pack.SCHEMA,
                ok=verify.get("ok"),
            ),
            "swarm_inference_beta_live_json": pack.artifact_entry(
                output_dir / "live" / "swarm_inference_beta_live.json",
                output_dir,
                kind="swarm_inference_beta_live",
                schema=pack.SCHEMA,
                ok=live.get("ok"),
            ),
        },
        "limitations": [
            "CI-safe fake-runner check; does not start a Coordinator, Miner, or Kaggle resource.",
            "Use crowdtensor swarm-infer-beta live for the side-effectful public Kaggle auto proof.",
        ],
    }


def print_human(report: dict[str, Any]) -> None:
    print("CrowdTensor Swarm Inference Beta check")
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
