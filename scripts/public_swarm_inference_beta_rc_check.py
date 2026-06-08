#!/usr/bin/env python3
"""Acceptance checks for Public Swarm Inference Beta RC."""

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

import public_swarm_inference_beta_rc_pack as pack  # noqa: E402


SCHEMA = "public_swarm_inference_beta_rc_check_v1"
SECRET_FRAGMENTS = [
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
]
COMMON_REQUIRED_CODES = {
    "public_swarm_inference_beta_rc_ready",
    "public_swarm_product_beta_ready",
    "p2p_lite_route_ready",
    "p2p_lite_discovery_ready",
    "cpu_fallback_ready",
    "local_cpu_inference_ready",
    "read_only_workload",
    "not_production",
}
LOCAL_REQUIRED_CODES = COMMON_REQUIRED_CODES | {
    "serve_join_generate_loop_ready",
    "remote_generate_session_ready",
    "public_swarm_generate_ready",
}
PACKAGE_REQUIRED_CODES = COMMON_REQUIRED_CODES | {
    "public_swarm_beta_rc_package_ready",
    "miner_join_pack_ready",
    "private_artifacts_local_only",
}
EXTERNAL_REQUIRED_CODES = COMMON_REQUIRED_CODES | {
    "serve_join_generate_loop_ready",
    "remote_generate_session_ready",
    "public_swarm_generate_ready",
    "external_runtime_verified",
    "remote_real_llm_sharded_existing_ready",
}


def completed(payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["cmd"], 0, stdout=json.dumps(payload) + "\n", stderr="")


def write(path: Path, payload: str = "{}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def output_scope_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    output_request = payload.get("output_request") if isinstance(payload.get("output_request"), dict) else {}
    prompt_scope = payload.get("prompt_scope") if isinstance(payload.get("prompt_scope"), dict) else {}
    answer_scope = payload.get("answer_scope") if isinstance(payload.get("answer_scope"), dict) else {}
    shareable = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
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
    if prompt_scope.get("source") not in {"prompt-text", "prompt-texts", "prompt-texts-file"}:
        errors.append("prompt_scope_source_mismatch")
    if not isinstance(prompt_scope.get("prompt_count"), int) or prompt_scope.get("prompt_count") < 1:
        errors.append("prompt_scope_count_mismatch")
    inline_prompt_text = prompt_scope.get("source") in {"prompt-text", "prompt-texts"}
    if prompt_scope.get("inline_prompt_text") is not inline_prompt_text:
        errors.append("prompt_scope_inline_prompt_text_mismatch")
    if prompt_scope.get("terminal_next_commands_local_private") is not inline_prompt_text:
        errors.append("prompt_scope_terminal_next_commands_mismatch")
    if prompt_scope.get("terminal_logs_local_private") is not inline_prompt_text:
        errors.append("prompt_scope_terminal_logs_mismatch")
    if prompt_scope.get("saved_artifacts_prompt_placeholders") is not True:
        errors.append("prompt_scope_saved_placeholders_mismatch")
    if prompt_scope.get("saved_artifacts_public_safe") is not True:
        errors.append("prompt_scope_saved_artifacts_public_safe_mismatch")
    if prompt_scope.get("prefer_prompt_file_or_stdin_for_shareable_logs") is not (prompt_scope.get("source") in {"prompt-text", "prompt-texts", "prompt-texts-file"}):
        errors.append("prompt_scope_shareable_log_guidance_mismatch")
    if prompt_scope.get("prompt_file_path_public") is not False:
        errors.append("prompt_scope_file_path_public_mismatch")
    if prompt_scope.get("raw_prompt_public") is not False:
        errors.append("prompt_scope_raw_prompt_public_mismatch")
    if prompt_scope.get("public_artifact_safe") is not True:
        errors.append("prompt_scope_public_artifact_safe_mismatch")
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
    if answer_scope.get("raw_prompt_public") is not False:
        errors.append("answer_scope_raw_prompt_public_mismatch")
    if answer_scope.get("raw_generated_text_public") is not False:
        errors.append("answer_scope_raw_generated_text_public_mismatch")
    if answer_scope.get("generated_token_ids_public") is not False:
        errors.append("answer_scope_generated_token_ids_public_mismatch")
    if answer_scope.get("public_artifact_safe") is not True:
        errors.append("answer_scope_public_artifact_safe_mismatch")
    if shareable.get("saved_artifacts_public_safe") is not True:
        errors.append("shareable_saved_artifacts_mismatch")
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
    return errors


def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    joined = " ".join(command)
    if "public_swarm_inference_beta_pack.py" in joined and "product-beta" in command:
        output_dir = Path(command[command.index("--output-dir") + 1])
        write(output_dir / "public_swarm_inference_beta.json")
        return completed({
            "schema": "public_swarm_inference_beta_v1",
            "ok": True,
            "mode": "product-beta",
            "diagnosis_codes": [
                "public_swarm_inference_beta_ready",
                "public_swarm_product_beta_ready",
                "public_swarm_product_rc_ready",
                "coordinator_product_surface_ready",
                "session_protocol_ready",
                "p2p_lite_discovery_ready",
                "gpu_generation_evidence_import_ready",
                "cpu_inference_beta_ready",
                "cpu_fallback_ready",
                "local_cpu_inference_ready",
                "read_only_workload",
                "not_production",
            ],
            "payload_summaries": {
                "public_swarm_product_rc": {
                    "schema": "public_swarm_product_rc_v1",
                    "ok": True,
                    "diagnosis_codes": ["public_swarm_product_rc_ready"],
                    "product_surface": {
                        "serve_ok": True,
                        "join_stage0_ok": True,
                        "join_stage1_ok": True,
                        "generate_ok": True,
                    },
                    "p2p_lite": {"ok": True},
                }
            },
            "beta": {"ready": True},
        })
    if "p2p_lite_discovery_check.py" in joined:
        return completed({
            "schema": "p2p_lite_discovery_check_v1",
            "ok": True,
            "diagnosis_codes": ["p2p_lite_discovery_ready"],
            "cpu_route": {"ok": True},
            "cuda_route": {"ok": True},
        })
    if "cpu_inference_beta_pack.py" in joined:
        output_dir = Path(command[command.index("--output-dir") + 1])
        write(output_dir / "cpu_inference_beta.json")
        write(output_dir / "cpu_inference_beta.md", "# cpu\n")
        return completed({
            "schema": "cpu_inference_beta_v1",
            "ok": True,
            "mode": "local",
            "workload": "all",
            "diagnosis_codes": ["cpu_inference_beta_ready", "local_cpu_inference_ready"],
            "steps": [{"name": "home_infer", "ok": True}],
            "safety": {"cpu_only_default": True, "summary_excludes_raw_inference_payloads": True},
        })
    if "public_swarm_inference_beta_pack.py" in joined and "prepare" in command:
        output_dir = Path(command[command.index("--output-dir") + 1])
        write(output_dir / "public_swarm_inference_beta_prepare.json")
        return completed({
            "schema": "public_swarm_inference_beta_v1",
            "ok": True,
            "mode": "prepare",
            "diagnosis_codes": [
                "public_swarm_beta_prepare_ready",
                "two_stage_join_pack_ready",
                "stage0_join_pack_ready",
                "stage1_join_pack_ready",
                "miner_registry_hashed",
            ],
        })
    if "crowdtensor.cli generate" in joined:
        return completed({
            "schema": "public_swarm_product_cli_v1",
            "ok": True,
            "mode": "generate",
            "diagnosis_codes": ["public_swarm_generate_ready"],
            "session": {"session_id": "session-1", "workload_type": "real_llm_sharded_infer"},
            "generation": {
                "generated_token_count": 2,
                "generated_text_hash": "sha256:synthetic",
                "multi_token_generation_ready": True,
            },
        })
    if "remote_real_llm_sharded_beta_pack.py" in joined:
        output_dir = Path(command[command.index("--output-dir") + 1])
        write(output_dir / "remote_real_llm_sharded_beta.json")
        return completed({
            "schema": "remote_real_llm_sharded_beta_v1",
            "ok": True,
            "mode": "remote-existing",
            "diagnosis_codes": [
                "remote_real_llm_sharded_ready",
                "remote_real_llm_sharded_existing_ready",
                "real_llm_sharded_ready",
                "activation_transport_ready",
                "decoded_tokens_match",
                "distinct_stage_miners",
                "stage_assignment_valid",
            ],
        })
    raise AssertionError(command)


def fake_generate_loop(_args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    generate_path = output_dir / "serve-join-generate" / "generate.json"
    write(
        generate_path,
        json.dumps(
            {
                "schema": "public_swarm_product_cli_v1",
                "ok": True,
                "mode": "generate",
                "diagnosis_codes": ["public_swarm_generate_ready"],
                "generation": {
                    "generated_token_count": 2,
                    "generated_text_hash": "sha256:synthetic",
                    "multi_token_generation_ready": True,
                    "raw_generated_text_public": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return {
        "ok": True,
        "step": {
            "name": "serve_join_generate_loop",
            "ok": True,
            "returncode": 0,
            "duration_seconds": 0.01,
            "payload_schema": "public_swarm_product_cli_v1",
            "payload_ok": True,
        },
        "coordinator_url": "http://127.0.0.1:9310",
        "generation": {
            "ok": True,
            "schema": "public_swarm_product_cli_v1",
            "session": {"session_id": "session-1", "workload_type": "real_llm_sharded_infer"},
            "generation": {
                "generated_token_count": 2,
                "generated_text_hash": "sha256:synthetic",
                "multi_token_generation_ready": True,
                "raw_generated_text_public": False,
            },
        },
        "diagnosis_codes": [
            "serve_join_generate_loop_ready",
            "remote_generate_session_ready",
            "public_swarm_generate_ready",
        ],
        "processes": {
            "stage0": {"returncode": 0},
            "stage1": {"returncode": 0},
        },
        "artifacts": {
            "generate_json": pack.artifact_entry(
                generate_path,
                output_dir,
                kind="public_swarm_generate",
                schema="public_swarm_product_cli_v1",
                ok=True,
            )
        },
    }


def validate_payload(payload: dict[str, Any], *, mode: str, required_codes: set[str]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if payload.get("ok") is not True:
        errors.append("rc_not_ready")
    if payload.get("mode") != mode:
        errors.append("mode_mismatch")
    codes = set(payload.get("diagnosis_codes") or [])
    for code in sorted(required_codes - codes):
        errors.append(f"missing_code:{code}")
    rc = payload.get("rc") if isinstance(payload.get("rc"), dict) else {}
    if rc.get("ready") is not True:
        errors.append("rc_summary_not_ready")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    for key in ["coordinator_backed_task_execution", "p2p_lite_discovery_only", "not_libp2p", "not_dht", "not_nat_traversal"]:
        if safety.get(key) is not True:
            errors.append(f"safety_missing:{key}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    rc_json = artifacts.get("public_swarm_inference_beta_rc_json") if isinstance(artifacts.get("public_swarm_inference_beta_rc_json"), dict) else {}
    if rc_json.get("present") is not True:
        errors.append("missing_rc_json_artifact")
    errors.extend(output_scope_errors(payload))
    return errors


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_public_swarm_beta_rc_check_"))
    if args.mode == "local-loopback":
        report_args = pack.parse_args([
            "local-loopback",
            "--output-dir",
            str(output_dir / "rc"),
            "--base-port",
            str(args.base_port),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--prompt-texts",
            "private beta rc check one,private beta rc check two",
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--json",
        ])
        original_generate_loop = pack.run_product_generate_loop
        try:
            pack.run_product_generate_loop = fake_generate_loop
            payload = pack.build_report(report_args, runner=fake_runner)
        finally:
            pack.run_product_generate_loop = original_generate_loop
        required = LOCAL_REQUIRED_CODES
    elif args.mode == "package":
        report_args = pack.parse_args([
            "package",
            "--output-dir",
            str(output_dir / "rc"),
            "--base-port",
            str(args.base_port),
            "--target",
            args.target,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--json",
        ])
        payload = pack.build_report(report_args, runner=fake_runner)
        required = set(PACKAGE_REQUIRED_CODES)
        if args.target == "kaggle":
            required.add("kaggle_remote_miner_package_ready")
    else:
        report_args = pack.parse_args([
            "external-existing",
            "--output-dir",
            str(output_dir / "rc"),
            "--coordinator-url",
            "http://127.0.0.1:9999",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--prompt-texts",
            "private beta rc check one,private beta rc check two",
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--json",
        ])
        payload = pack.build_report(report_args, runner=fake_runner)
        required = EXTERNAL_REQUIRED_CODES
    errors = validate_payload(payload, mode=args.mode, required_codes=required)
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "mode": args.mode,
        "target": args.target,
        "output_dir": str(output_dir),
        "errors": errors,
        "rc_schema": payload.get("schema"),
        "rc_ok": payload.get("ok"),
        "diagnosis_codes": ["public_swarm_inference_beta_rc_check_ready"] if not errors else ["public_swarm_inference_beta_rc_check_blocked"],
        "artifacts": {
            "public_swarm_inference_beta_rc_json": str(output_dir / "rc" / "public_swarm_inference_beta_rc.json")
        },
    }
    write(output_dir / "public_swarm_inference_beta_rc_check.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Public Swarm Inference Beta RC.")
    parser.add_argument("--mode", choices=["local-loopback", "package", "external-existing"], default="local-loopback")
    parser.add_argument("--target", choices=["local", "kaggle"], default="local")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--base-port", type=int, default=9310)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.base_port < 1:
        raise SystemExit("--base-port must be positive")
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    return args


def main() -> None:
    args = parse_args()
    result = run_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"Public Swarm Inference Beta RC check ready: {result.get('ok')}")
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
