#!/usr/bin/env python3
"""CI-safe checks for the GPU sharded multi-token generation Beta wrapper."""

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

import gpu_sharded_generation_beta_pack as pack  # noqa: E402


SCHEMA = "gpu_sharded_generation_beta_check_v1"
REQUIRED_CODES = {
    "gpu_sharded_generation_ready",
    "multi_token_generation_ready",
    "public_swarm_gpu_beta_ready",
    "hf_transformers_cuda_ready",
    "stage0_partition_loaded",
    "stage1_partition_loaded",
    "partition_parameter_split_valid",
    "stage_local_partition_ready",
    "decoded_tokens_match",
    "distinct_stage_miners",
    "stage_assignment_valid",
}
SECRET_FRAGMENTS = {
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
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def completed(payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["cmd"], 0, stdout=json.dumps(payload) + "\n", stderr="")


def option_value(command: list[str], option: str, default: str = "") -> str:
    if option not in command:
        return default
    index = command.index(option) + 1
    if index >= len(command):
        return default
    return command[index]


def synthetic_public_gpu_report(max_new_tokens: int, *, mode: str = "local-loopback") -> dict[str, Any]:
    codes = sorted(REQUIRED_CODES | {
        "read_only_workload",
        "not_production",
        "not_p2p",
        "gpu_loopback_generation_ready" if mode != "kaggle-auto" else "gpu_multi_machine_generation_ready",
    })
    if mode == "kaggle-auto":
        codes.extend([
            "public_swarm_gpu_beta_kaggle_auto_ready",
            "external_gpu_runtime_verified",
            "kaggle_kernels_deleted",
        ])
    return {
        "schema": "public_swarm_gpu_inference_beta_v1",
        "ok": True,
        "mode": mode,
        "diagnosis_codes": codes,
        "payload_summaries": {
            "real_llm_internet_beta": {
                "schema": "real_llm_internet_beta_v1",
                "ok": True,
                "generation": {
                    "max_new_tokens": max_new_tokens,
                    "generated_token_count": max_new_tokens,
                    "generated_text_hash": "sha256:synthetic-generation",
                    "generated_text_redacted": True,
                    "multi_token_generation_ready": True,
                },
                "external_alpha": {
                    "schema": "real_llm_internet_alpha_v1",
                    "ok": True,
                    "generation": {
                        "max_new_tokens": max_new_tokens,
                        "generated_token_count": max_new_tokens,
                        "generated_text_hash": "sha256:synthetic-generation",
                        "generated_text_redacted": True,
                        "multi_token_generation_ready": True,
                    },
                },
            },
        },
    }


def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    if "public_swarm_gpu_inference_beta_pack.py" not in " ".join(command):
        raise AssertionError(command)
    max_new_tokens = int(option_value(command, "--max-new-tokens", "4"))
    script_index = next(
        (index for index, item in enumerate(command) if item.endswith("public_swarm_gpu_inference_beta_pack.py")),
        -1,
    )
    mode = command[script_index + 1] if script_index >= 0 and script_index + 1 < len(command) else "local-loopback"
    return completed(synthetic_public_gpu_report(max_new_tokens, mode=mode))


def validate_report(report: dict[str, Any], *, max_new_tokens: int) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ready")
    generation = report.get("generation") if isinstance(report.get("generation"), dict) else {}
    if generation.get("generated_token_count") != max_new_tokens:
        errors.append("generated_token_count_mismatch")
    if generation.get("multi_token_generation_ready") is not True:
        errors.append("multi_token_generation_not_ready")
    codes = set(report.get("diagnosis_codes") or [])
    missing = sorted(REQUIRED_CODES - codes)
    if missing:
        errors.append(f"missing_codes:{','.join(missing)}")
    encoded = json.dumps(report, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    if pack.has_raw_generation_payload(report):
        errors.append("raw_generation_payload_present")
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    own_json = artifacts.get("gpu_sharded_generation_beta_json") if isinstance(artifacts.get("gpu_sharded_generation_beta_json"), dict) else {}
    if own_json.get("present") is not True:
        errors.append("summary_artifact_missing")
    return errors


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="crowdtensor_gpu_generation_check_")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.gpu_report).resolve() if args.gpu_report else output_dir / "synthetic_public_gpu_report.json"
    if args.gpu_report:
        report_args = [
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(source),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        generated_source = False
    else:
        write_json(source, synthetic_public_gpu_report(args.max_new_tokens))
        report_args = [
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(source),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        generated_source = True
    imported = pack.build_report(pack.parse_args(report_args))
    errors = validate_report(imported, max_new_tokens=args.max_new_tokens)

    loopback_report = None
    loopback_errors: list[str] = []
    if args.include_wrapper_check:
        loopback_report = pack.build_report(pack.parse_args([
            "local-loopback",
            "--output-dir",
            str(output_dir / "wrapper"),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]), runner=fake_runner)
        loopback_errors = validate_report(loopback_report, max_new_tokens=args.max_new_tokens)

    ok = not errors and not loopback_errors
    result = {
        "schema": SCHEMA,
        "ok": ok,
        "output_dir": str(output_dir),
        "max_new_tokens": args.max_new_tokens,
        "generated_source": generated_source,
        "checks": {
            "evidence_import": {
                "ok": not errors,
                "errors": errors,
                "report_schema": imported.get("schema"),
                "generated_token_count": (imported.get("generation") or {}).get("generated_token_count"),
            },
            "wrapper": {
                "ok": not loopback_errors,
                "enabled": bool(args.include_wrapper_check),
                "errors": loopback_errors,
                "report_schema": loopback_report.get("schema") if loopback_report else None,
            },
        },
        "diagnosis_codes": [
            "gpu_sharded_generation_beta_check_ready" if ok else "gpu_sharded_generation_beta_check_blocked",
        ],
        "artifacts": {
            "source_report": str(source),
            "imported_report": str(output_dir / "gpu_sharded_generation_beta_evidence_import.json"),
        },
    }
    write_json(output_dir / "gpu_sharded_generation_beta_check.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CI-safe GPU sharded generation Beta checks.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--gpu-report", default="")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--include-wrapper-check", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    return args


def main() -> None:
    result = run_check(parse_args())
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
