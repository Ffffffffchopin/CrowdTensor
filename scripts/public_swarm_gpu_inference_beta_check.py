#!/usr/bin/env python3
"""Acceptance checks for the optional CUDA Public Swarm Inference Beta pack."""

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

import public_swarm_gpu_inference_beta_pack as gpu_pack  # noqa: E402

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
SMOKE_REQUIRED_CODES = {
    "public_swarm_gpu_beta_smoke_ready",
    "gpu_runtime_smoke_ready",
    "read_only_workload",
    "not_production",
    "not_p2p",
}
LOCAL_REQUIRED_CODES = {
    "public_swarm_gpu_beta_ready",
    "two_stage_split_inference_ready",
    "gpu_runtime_ready",
    "cuda_runtime_available",
    "hf_transformers_cuda_ready",
    "gpu_stage0_ready",
    "gpu_stage1_ready",
    "stage_local_partition_ready",
    "stage0_partition_loaded",
    "stage1_partition_loaded",
    "partition_parameter_split_valid",
    "decoded_tokens_match",
    "distinct_stage_miners",
    "stage_assignment_valid",
    "read_only_workload",
    "not_production",
    "not_p2p",
}
PACKAGE_REQUIRED_CODES = {
    "public_swarm_gpu_beta_package_ready",
    "kaggle_gpu_package_ready",
    "kaggle_gpu_requires_private_kernels",
    "read_only_workload",
    "not_production",
    "not_p2p",
}
KAGGLE_AUTO_REQUIRED_CODES = {
    "public_swarm_gpu_beta_ready",
    "public_swarm_gpu_beta_kaggle_auto_ready",
    "external_gpu_runtime_verified",
    "gpu_runtime_ready",
    "cuda_runtime_available",
    "hf_transformers_cuda_ready",
    "gpu_stage0_ready",
    "gpu_stage1_ready",
    "stage_local_partition_ready",
    "stage0_partition_loaded",
    "stage1_partition_loaded",
    "partition_parameter_split_valid",
    "decoded_tokens_match",
    "distinct_stage_miners",
    "stage_assignment_valid",
    "kaggle_kernels_deleted",
    "token_rotation_required",
    "read_only_workload",
    "not_production",
    "not_p2p",
}


def completed(payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["cmd"], 0, stdout=json.dumps(payload) + "\n", stderr="")


def option_value(command: list[str], option: str, default: str = "") -> str:
    if option not in command:
        return default
    index = command.index(option) + 1
    if index >= len(command):
        return default
    return command[index]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fake_gpu_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    del kwargs
    joined = " ".join(command)
    if "real_llm_internet_beta_pack.py" not in joined:
        raise AssertionError(command)
    output_dir = Path(option_value(command, "--output-dir"))
    beta_path = output_dir / "real_llm_internet_beta.json"
    remote_path = output_dir / "external-alpha" / "live-rc" / "remote-real-llm-runtime" / "remote_real_llm_sharded_beta.json"
    write_json(remote_path, {
        "schema": "remote_real_llm_sharded_beta_v1",
        "ok": True,
        "diagnosis_codes": ["remote_real_llm_sharded_ready", "hf_transformers_cuda_ready"],
    })
    payload = {
        "schema": "real_llm_internet_beta_v1",
        "ok": True,
        "mode": "kaggle-auto",
        "workload": {
            "workload_type": "real_llm_sharded_infer",
            "real_llm_backend": "hf_transformers_cuda",
            "real_llm_partition_mode": "stage_local",
            "hf_model_id": "sshleifer/tiny-gpt2",
        },
        "runtime_classification": {
            "kaggle_auto": True,
            "external_runtime_verified": True,
        },
        "kaggle_lifecycle": {
            "owner": option_value(command, "--kaggle-owner", "xuyuhaosuyi"),
            "kernel_slug_prefix": option_value(command, "--kernel-slug-prefix", "crowdtensor-public-swarm-gpu-beta-check"),
            "expected_push_count": 2,
            "pushed_refs": {
                "stage0": "xuyuhaosuyi/crowdtensor-public-swarm-gpu-beta-check-stage0",
                "stage1": "xuyuhaosuyi/crowdtensor-public-swarm-gpu-beta-check-stage1",
            },
            "kernels_deleted": True,
            "cleanup_required": True,
        },
        "diagnosis_codes": [
            "real_llm_internet_beta_ready",
            "external_runtime_verified",
            "cuda_runtime_available",
            "hf_transformers_cuda_ready",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "stage_local_partition_ready",
            "stage0_partition_loaded",
            "stage1_partition_loaded",
            "partition_parameter_split_valid",
            "kaggle_kernels_deleted",
            "token_rotation_required",
        ],
    }
    write_json(beta_path, payload)
    return completed(payload)


def run_json(command: list[str], *, timeout: float, runner=subprocess.run) -> dict[str, Any]:
    completed = runner(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
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


def validate_payload(payload: dict[str, Any], *, mode: str, required_codes: set[str]) -> None:
    if payload.get("schema") != "public_swarm_gpu_inference_beta_v1" or payload.get("ok") is not True:
        raise SystemExit(f"unexpected Public Swarm GPU Inference Beta report: {json.dumps(payload, sort_keys=True)}")
    if payload.get("mode") != mode:
        raise SystemExit(f"unexpected mode: {payload.get('mode')}")
    codes = set(payload.get("diagnosis_codes") or [])
    missing = sorted(required_codes - codes)
    if missing:
        raise SystemExit(f"missing readiness diagnosis: {missing}")
    beta = payload.get("beta") if isinstance(payload.get("beta"), dict) else {}
    if beta.get("ready") is not True or beta.get("backend") != "hf_transformers_cuda":
        raise SystemExit(f"GPU beta readiness summary is not ready: {beta}")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    if safety.get("read_only_workload") != "real_llm_sharded_infer" or safety.get("not_production") is not True:
        raise SystemExit(f"unexpected safety summary: {safety}")
    output_request = payload.get("output_request") if isinstance(payload.get("output_request"), dict) else {}
    if output_request.get("include_output") is not False:
        raise SystemExit(f"unexpected output_request.include_output: {output_request}")
    if output_request.get("raw_prompt_public") is not False:
        raise SystemExit(f"unexpected output_request.raw_prompt_public: {output_request}")
    if output_request.get("raw_generated_text_public") is not False:
        raise SystemExit(f"unexpected output_request.raw_generated_text_public: {output_request}")
    if output_request.get("generated_token_ids_public") is not False:
        raise SystemExit(f"unexpected output_request.generated_token_ids_public: {output_request}")
    if output_request.get("public_artifact_safe") is not True:
        raise SystemExit(f"unexpected output_request.public_artifact_safe: {output_request}")
    prompt_scope = payload.get("prompt_scope") if isinstance(payload.get("prompt_scope"), dict) else {}
    expected_prompt_source = "prompt-texts" if mode in {"local-loopback", "verify"} else "none"
    expected_inline = mode in {"local-loopback", "verify"}
    if prompt_scope.get("source") != expected_prompt_source:
        raise SystemExit(f"unexpected prompt_scope.source: {prompt_scope}")
    prompt_count = prompt_scope.get("prompt_count")
    if expected_inline:
        if not isinstance(prompt_count, int) or prompt_count < 1:
            raise SystemExit(f"unexpected prompt_scope.prompt_count: {prompt_scope}")
    elif prompt_count != 0:
        raise SystemExit(f"unexpected prompt_scope.prompt_count: {prompt_scope}")
    for key in ["inline_prompt_text", "terminal_next_commands_local_private", "terminal_logs_local_private", "prefer_prompt_file_or_stdin_for_shareable_logs"]:
        if prompt_scope.get(key) is not expected_inline:
            raise SystemExit(f"unexpected prompt_scope.{key}: {prompt_scope}")
    for key in ["saved_artifacts_prompt_placeholders", "saved_artifacts_public_safe", "public_artifact_safe"]:
        if prompt_scope.get(key) is not True:
            raise SystemExit(f"unexpected prompt_scope.{key}: {prompt_scope}")
    for key in ["prompt_file_path_public", "raw_prompt_public"]:
        if prompt_scope.get(key) is not False:
            raise SystemExit(f"unexpected prompt_scope.{key}: {prompt_scope}")
    answer_scope = payload.get("answer_scope") if isinstance(payload.get("answer_scope"), dict) else {}
    if answer_scope.get("scope_state") != "no-local-answer":
        raise SystemExit(f"unexpected answer_scope.scope_state: {answer_scope}")
    if answer_scope.get("visible_in_terminal") is not False or answer_scope.get("terminal_only") is not False:
        raise SystemExit(f"unexpected answer_scope terminal visibility: {answer_scope}")
    if answer_scope.get("saved_json_display") != "hash-only" or answer_scope.get("saved_markdown_display") != "hash-only":
        raise SystemExit(f"unexpected answer_scope saved display: {answer_scope}")
    if answer_scope.get("public_artifact_safe") is not True:
        raise SystemExit(f"unexpected answer_scope.public_artifact_safe: {answer_scope}")
    shareable = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
    if shareable.get("saved_artifacts_public_safe") is not True:
        raise SystemExit(f"unexpected shareable saved artifact safety: {shareable}")
    if shareable.get("raw_prompt_public") is not False:
        raise SystemExit(f"unexpected shareable raw_prompt_public: {shareable}")
    if shareable.get("raw_generated_text_public") is not False:
        raise SystemExit(f"unexpected shareable raw_generated_text_public: {shareable}")
    if shareable.get("generated_token_ids_public") is not False:
        raise SystemExit(f"unexpected shareable generated_token_ids_public: {shareable}")
    if shareable.get("answer_scope_state") != "no-local-answer":
        raise SystemExit(f"unexpected shareable answer_scope_state: {shareable}")
    if shareable.get("local_answer_terminal_only") is not False:
        raise SystemExit(f"unexpected shareable local_answer_terminal_only: {shareable}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            raise SystemExit(f"Public Swarm GPU Inference Beta report leaked sensitive fragment: {fragment}")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    gpu_json = artifacts.get("public_swarm_gpu_inference_beta_json") if isinstance(artifacts.get("public_swarm_gpu_inference_beta_json"), dict) else {}
    if gpu_json.get("present") is not True:
        raise SystemExit("Public Swarm GPU Inference Beta JSON artifact was not reported present")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Public Swarm GPU Inference Beta acceptance checks.")
    parser.add_argument("--mode", choices=["local-smoke", "local-loopback", "kaggle-package", "kaggle-auto", "evidence-import"], default="local-smoke")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--base-port", type=int, default=9390)
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--gpu-report", default="")
    parser.add_argument("--kaggle-owner", default="xuyuhaosuyi")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_public_swarm_gpu_beta_check_"))
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_gpu_inference_beta_pack.py"),
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
    elif args.mode == "evidence-import":
        command.extend(["--gpu-report", args.gpu_report])
    elif args.mode == "kaggle-auto":
        command.extend([
            "--kaggle-owner",
            args.kaggle_owner,
            "--kernel-slug-prefix",
            "crowdtensor-public-swarm-gpu-beta-check",
        ])
    if args.mode == "kaggle-auto":
        pack_args = gpu_pack.parse_args(command[2:])
        payload = gpu_pack.build_report(pack_args, runner=fake_gpu_runner)
    else:
        payload = run_json(
            command,
            timeout=float(args.timeout_seconds) + 240.0,
            runner=subprocess.run,
        )
    required = {
        "local-smoke": SMOKE_REQUIRED_CODES,
        "local-loopback": LOCAL_REQUIRED_CODES,
        "kaggle-package": PACKAGE_REQUIRED_CODES,
        "kaggle-auto": KAGGLE_AUTO_REQUIRED_CODES,
        "evidence-import": {"public_swarm_gpu_beta_evidence_import_ready", "external_gpu_runtime_verified"},
    }[args.mode]
    validate_payload(payload, mode=args.mode, required_codes=required)
    summary = {
        "schema": "public_swarm_gpu_inference_beta_check_v1",
        "ok": True,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "diagnosis_codes": payload.get("diagnosis_codes") or [],
    }
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(f"Public Swarm GPU Inference Beta check passed: mode={args.mode} output_dir={output_dir}")


if __name__ == "__main__":
    main()
