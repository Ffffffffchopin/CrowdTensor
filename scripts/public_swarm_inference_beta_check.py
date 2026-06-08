#!/usr/bin/env python3
"""Acceptance checks for the Public Swarm Inference Beta wrapper."""

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

import public_swarm_inference_beta_pack as pack  # noqa: E402
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
    "real_llm_sharded_result",
    '"generated_text":',
    '"generated_token_ids":',
]
LOCAL_REQUIRED_CODES = {
    "public_swarm_inference_beta_ready",
    "two_stage_split_inference_ready",
    "decoded_tokens_match",
    "distinct_stage_miners",
    "stage_assignment_valid",
    "local_loopback_ready",
    "read_only_workload",
    "not_production",
    "not_p2p",
}
IMPORT_REQUIRED_CODES = {
    "public_swarm_inference_beta_ready",
    "public_swarm_beta_evidence_import_ready",
    "external_live_evidence_imported",
    "stage0_live_requeue_evidence_ready",
    "stage1_live_requeue_evidence_ready",
    "two_stage_split_inference_ready",
    "decoded_tokens_match",
    "distinct_stage_miners",
    "stage_assignment_valid",
    "read_only_workload",
    "not_production",
    "not_p2p",
}
PRODUCT_REQUIRED_CODES = {
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
}


def run_json(command: list[str], *, timeout: float) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    for line in reversed([line.strip() for line in completed.stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError(f"command emitted no JSON: {' '.join(command)}\nstdout:\n{completed.stdout}")


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
    if prompt_scope.get("source") not in {"prompt-text", "prompt-texts", "prompt-texts-file", "inherited-or-fixed-evidence"}:
        errors.append("prompt_scope_source_mismatch")
    if not isinstance(prompt_scope.get("prompt_count"), int) or prompt_scope.get("prompt_count") < 0:
        errors.append("prompt_scope_count_mismatch")
    inline_prompt_text = prompt_scope.get("source") in {"prompt-text", "prompt-texts"}
    has_prompt_source = prompt_scope.get("source") in {"prompt-text", "prompt-texts", "prompt-texts-file"}
    if has_prompt_source and prompt_scope.get("prompt_count") < 1:
        errors.append("prompt_scope_prompt_texts_count_mismatch")
    if not has_prompt_source and prompt_scope.get("prompt_count") != 0:
        errors.append("prompt_scope_inherited_count_mismatch")
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
    if prompt_scope.get("prefer_prompt_file_or_stdin_for_shareable_logs") is not has_prompt_source:
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


def validate_payload(payload: dict[str, Any], *, mode: str, required_codes: set[str]) -> None:
    if payload.get("schema") != "public_swarm_inference_beta_v1" or payload.get("ok") is not True:
        raise SystemExit(f"unexpected Public Swarm Inference Beta report: {json.dumps(payload, sort_keys=True)}")
    if payload.get("mode") != mode:
        raise SystemExit(f"unexpected mode: {payload.get('mode')}")
    codes = set(payload.get("diagnosis_codes") or [])
    missing = sorted(required_codes - codes)
    if missing:
        raise SystemExit(f"missing readiness diagnosis: {missing}")
    beta = payload.get("beta") if isinstance(payload.get("beta"), dict) else {}
    if beta.get("ready") is not True:
        raise SystemExit(f"beta readiness summary is not ready: {beta}")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    if mode == "product-beta":
        if safety.get("coordinator_backed_task_execution") is not True or safety.get("p2p_lite_discovery_only") is not True:
            raise SystemExit(f"unexpected product Beta safety summary: {safety}")
    elif safety.get("cpu_only") is not True or safety.get("read_only_workload") != "real_llm_sharded_infer":
        raise SystemExit(f"unexpected safety summary: {safety}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            raise SystemExit(f"Public Swarm Inference Beta report leaked sensitive fragment: {fragment}")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    beta_json = artifacts.get("public_swarm_inference_beta_json") if isinstance(artifacts.get("public_swarm_inference_beta_json"), dict) else {}
    if beta_json.get("present") is not True:
        raise SystemExit("Public Swarm Inference Beta JSON artifact was not reported present")
    scope_errors = output_scope_errors(payload)
    if scope_errors:
        raise SystemExit(f"Public Swarm Inference Beta output scope mismatch: {scope_errors}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Public Swarm Inference Beta acceptance checks.")
    parser.add_argument("--mode", choices=["local-loopback", "evidence-import", "product-beta"], default="local-loopback")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--base-port", type=int, default=9290)
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--cpu-request-count", type=int, default=1)
    parser.add_argument("--external-llm-request-count", type=int, default=1)
    parser.add_argument("--scenario-id", default="route-baseline")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--alpha-rc-report", default="dist/public-swarm-inference-alpha-rc/public_swarm_inference_alpha_rc.json")
    parser.add_argument(
        "--stage0-report",
        default=(
            "dist/public-swarm-inference-alpha-live-stage0-requeue-20260527165830/"
            "public_swarm_inference_alpha.json"
        ),
    )
    parser.add_argument(
        "--stage1-report",
        default=(
            "dist/public-swarm-inference-alpha-live-stage1-requeue-20260527170600/"
            "public_swarm_inference_alpha.json"
        ),
    )
    parser.add_argument("--summary-report", default="dist/public-swarm-inference-alpha-live-requeue-summary.json")
    parser.add_argument("--allow-missing-live-evidence", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def completed(payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["cmd"], 0, stdout=json.dumps(payload) + "\n", stderr="")


def fake_product_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    joined = " ".join(command)
    if "public_swarm_product_rc_pack.py" in joined:
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "public_swarm_product_rc.json").write_text("{}", encoding="utf-8")
        gpu_dir = output_dir / "gpu-generation-import"
        gpu_dir.mkdir(parents=True, exist_ok=True)
        (gpu_dir / "gpu_sharded_generation_beta_evidence_import.json").write_text("{}", encoding="utf-8")
        return completed({
            "schema": "public_swarm_product_rc_v1",
            "ok": True,
            "diagnosis_codes": [
                "public_swarm_product_rc_ready",
                "coordinator_product_surface_ready",
                "session_protocol_ready",
                "p2p_lite_discovery_ready",
                "gpu_generation_evidence_import_ready",
            ],
            "product_surface": {
                "serve": {"ok": True},
                "join_stage0": {"ok": True},
                "join_stage1": {"ok": True},
                "generate": {"ok": True},
                "peer_check": {"ok": True},
            },
            "session_protocol": {"ok": True, "schema": "session_protocol_check_v1", "route_usable": True},
            "p2p_lite": {"ok": True, "schema": "p2p_lite_discovery_check_v1", "cpu_route_ok": True, "cuda_route_ok": True},
            "gpu_generation_import": {
                "ok": True,
                "schema": "gpu_sharded_generation_beta_v1",
                "mode": "evidence-import",
                "generated_text_hash": "sha256:synthetic",
                "raw_generated_text_public": False,
            },
        })
    if "cpu_inference_beta_pack.py" in joined:
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "cpu_inference_beta.json").write_text("{}", encoding="utf-8")
        (output_dir / "cpu_inference_beta.md").write_text("# cpu\n", encoding="utf-8")
        return completed({
            "schema": "cpu_inference_beta_v1",
            "ok": True,
            "mode": "local",
            "workload": "all",
            "diagnosis_codes": ["cpu_inference_beta_ready", "local_cpu_inference_ready"],
            "steps": [{"name": "home_infer", "ok": True}, {"name": "llm_infer_mock", "ok": True}],
            "safety": {
                "cpu_only_default": True,
                "summary_excludes_raw_inference_payloads": True,
            },
        })
    raise AssertionError(command)


def main() -> None:
    args = parse_args()
    if args.base_port < 1:
        raise SystemExit("--base-port must be positive")
    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_public_swarm_beta_check_"))
    if args.mode == "product-beta":
        output_dir.mkdir(parents=True, exist_ok=True)
        gpu_report = output_dir / "synthetic_gpu_product_beta.json"
        gpu_report.write_text(json.dumps({"schema": "public_swarm_gpu_inference_beta_v1", "ok": True}), encoding="utf-8")
        payload = pack.build_report(pack.parse_args([
            "product-beta",
            "--output-dir",
            str(output_dir / "product-beta"),
            "--base-port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--cpu-request-count",
            str(args.cpu_request_count),
            "--external-llm-request-count",
            str(args.external_llm_request_count),
            "--scenario-id",
            args.scenario_id,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--gpu-report",
            str(gpu_report),
            "--prompt-texts",
            "private beta prompt one,private beta prompt two",
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--json",
        ]), runner=fake_product_runner)
        validate_payload(payload, mode=args.mode, required_codes=PRODUCT_REQUIRED_CODES)
        print(json.dumps({
            "schema": "public_swarm_inference_beta_check_v1",
            "ok": True,
            "mode": args.mode,
            "output_dir": str(output_dir),
            "diagnosis_codes": payload.get("diagnosis_codes") or [],
        }, sort_keys=True))
        return

    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_inference_beta_pack.py"),
        args.mode,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    if args.mode == "local-loopback":
        command.extend([
            "--base-port",
            str(args.base_port),
            "--hf-model-id",
            args.hf_model_id,
        ])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
    else:
        command.extend([
            "--alpha-rc-report",
            args.alpha_rc_report,
            "--stage0-report",
            args.stage0_report,
            "--stage1-report",
            args.stage1_report,
            "--summary-report",
            args.summary_report,
        ])
        if args.allow_missing_live_evidence:
            command.append("--allow-missing-live-evidence")
    payload = run_json(command, timeout=float(args.timeout_seconds) + 180.0)
    validate_payload(
        payload,
        mode=args.mode,
        required_codes=LOCAL_REQUIRED_CODES if args.mode == "local-loopback" else IMPORT_REQUIRED_CODES,
    )
    print(json.dumps({
        "schema": "public_swarm_inference_beta_check_v1",
        "ok": True,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "diagnosis_codes": payload.get("diagnosis_codes") or [],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
