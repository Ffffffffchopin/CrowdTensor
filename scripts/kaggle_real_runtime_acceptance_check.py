#!/usr/bin/env python3
"""CI-safe artifact check for Kaggle real runtime acceptance preparation."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "kaggle_real_runtime_acceptance_check_v1"
MODEL_BUNDLE_KIND = "model-bundle"
MICRO_LLM_SHARDED_KIND = "micro-llm-sharded"
WORKLOAD_CHOICES = [MODEL_BUNDLE_KIND, MICRO_LLM_SHARDED_KIND]
SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "inference_results",
    "output_text",
    "Bearer ",
)


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("command emitted no JSON object")


def run_json(command: list[str], *, timeout: float) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json_from_stdout(completed.stdout)


def parse_private_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        parsed = shlex.split(raw_value)
        values[key] = parsed[0] if parsed else ""
    return values


def assert_no_sensitive_output(payload: dict[str, Any], *, extra_secrets: list[str] | None = None) -> None:
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in [*SECRET_FRAGMENTS, *(extra_secrets or [])]:
        if fragment and fragment in encoded:
            raise SystemExit(f"Kaggle real runtime acceptance check leaked sensitive fragment: {fragment}")


def artifact_path(report: dict[str, Any], output_dir: Path, name: str) -> Path:
    artifact = (report.get("artifacts") or {}).get(name) or {}
    path = artifact.get("path")
    if not path:
        raise SystemExit(f"missing artifact path for {name}: {artifact}")
    return output_dir / path


def validate_prepare(payload: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    workload_arg = getattr(args, "workload", MODEL_BUNDLE_KIND)
    stage_mode_arg = getattr(args, "stage_mode", "both")
    decode_steps_arg = getattr(args, "decode_steps", 4)
    if payload.get("schema") != "kaggle_real_runtime_acceptance_v1" or payload.get("ok") is not True:
        raise SystemExit(f"unexpected prepare payload: {json.dumps(payload, sort_keys=True)}")
    if payload.get("mode") != "prepare":
        raise SystemExit(f"unexpected prepare mode: {payload.get('mode')}")
    expected_url = f"http://{args.public_host}:{args.port}"
    if payload.get("coordinator_url") != expected_url:
        raise SystemExit(f"unexpected Coordinator URL: {payload.get('coordinator_url')} != {expected_url}")
    codes = payload.get("diagnosis_codes") or []
    if "kaggle_artifacts_ready" not in codes:
        raise SystemExit(f"prepare missing kaggle_artifacts_ready: {codes}")
    workload = payload.get("workload") if isinstance(payload.get("workload"), dict) else {}
    if workload and workload.get("kind") != workload_arg:
        raise SystemExit(f"unexpected workload kind: {workload.get('kind')} != {workload_arg}")
    if workload_arg == MICRO_LLM_SHARDED_KIND:
        if workload.get("workload_type") != "micro_llm_sharded_infer":
            raise SystemExit(f"unexpected micro workload type: {workload}")
        if workload.get("stage_mode") != stage_mode_arg:
            raise SystemExit(f"unexpected stage_mode: {workload.get('stage_mode')} != {stage_mode_arg}")
        if workload.get("decode_steps") != decode_steps_arg:
            raise SystemExit(f"unexpected decode_steps: {workload.get('decode_steps')} != {decode_steps_arg}")
    safety = payload.get("safety") or {}
    for key in [
        "temporary_http",
        "temporary_http_boundary_confirmed",
        "token_rotation_required",
        "public_http_not_production",
        "operator_env_excluded_from_kaggle",
        "cpu_only_workload",
        "not_production",
        "not_p2p",
    ]:
        if safety.get(key) is not True:
            raise SystemExit(f"safety flag {key} must be true: {safety}")

    required_artifacts = ["operator_private_env", "miner_registry", "coordinator_launch_script", "operator_commands"]
    if workload_arg == MICRO_LLM_SHARDED_KIND and stage_mode_arg == "split":
        required_artifacts.extend([
            "kaggle_upload_stage0_miner_env",
            "kaggle_upload_stage0_miner_script",
            "kaggle_upload_stage0_runbook",
            "kaggle_upload_stage1_miner_env",
            "kaggle_upload_stage1_miner_script",
            "kaggle_upload_stage1_runbook",
        ])
    else:
        required_artifacts.extend([
            "miner_private_env",
            "kaggle_upload_miner_env",
            "kaggle_upload_miner_script",
            "kaggle_upload_runbook",
        ])
    for name in required_artifacts:
        artifact = (payload.get("artifacts") or {}).get(name) or {}
        if artifact.get("present") is not True:
            raise SystemExit(f"missing required artifact {name}: {artifact}")

    uploads = [output_dir / "kaggle-upload"]
    if workload_arg == MICRO_LLM_SHARDED_KIND and stage_mode_arg == "split":
        uploads = [output_dir / "kaggle-upload-stage0", output_dir / "kaggle-upload-stage1"]
    for upload in uploads:
        if (upload / "operator.private.env").exists():
            raise SystemExit(f"operator.private.env must not be copied into {upload.name}")
        runbook_text = (upload / "KAGGLE_RUN.md").read_text(encoding="utf-8")
        if "Do not upload `operator.private.env`" not in runbook_text:
            raise SystemExit("Kaggle runbook must warn against uploading operator.private.env")
    if workload_arg == MICRO_LLM_SHARDED_KIND and stage_mode_arg == "split":
        stage0_script = (output_dir / "kaggle-upload-stage0" / "kaggle_remote_miner.py").read_text(encoding="utf-8")
        stage1_script = (output_dir / "kaggle-upload-stage1" / "kaggle_remote_miner.py").read_text(encoding="utf-8")
        if "--micro-llm-stage-role" not in stage0_script or "stage0" not in stage0_script:
            raise SystemExit("stage0 Kaggle launcher must include --micro-llm-stage-role stage0")
        if "--micro-llm-stage-role" not in stage1_script or "stage1" not in stage1_script:
            raise SystemExit("stage1 Kaggle launcher must include --micro-llm-stage-role stage1")
        registry = json.loads((output_dir / "remote-home-compute" / "miner_registry.json").read_text(encoding="utf-8"))
        miner_ids = {item.get("miner_id") for item in registry.get("miners", []) if isinstance(item, dict)}
        if f"{args.miner_id}-stage0" not in miner_ids or f"{args.miner_id}-stage1" not in miner_ids:
            raise SystemExit(f"micro split registry must contain stage0 and stage1 miners: {sorted(miner_ids)}")
    launch_text = artifact_path(payload, output_dir, "coordinator_launch_script").read_text(encoding="utf-8")
    if "--host 0.0.0.0" not in launch_text or f"--port {args.port}" not in launch_text:
        raise SystemExit("Coordinator launch script must bind the requested public port")
    if "--miner-token-registry" not in launch_text or "sha256:" not in launch_text:
        raise SystemExit("Coordinator launch script must use hashed tokens and the miner registry")

    operator_env = parse_private_env(output_dir / "remote-home-compute" / "operator.private.env")
    miner_env = parse_private_env(output_dir / "remote-home-compute" / "miner.private.env")
    stage1_env = parse_private_env(output_dir / "remote-home-compute" / "miner.stage1.private.env")
    assert_no_sensitive_output(payload, extra_secrets=list(operator_env.values()) + list(miner_env.values()) + list(stage1_env.values()))
    return {
        "name": "kaggle_real_runtime_prepare",
        "ok": True,
        "schema": payload.get("schema"),
        "coordinator_url": payload.get("coordinator_url"),
        "workload": workload_arg,
        "diagnosis_codes": codes,
        "artifact_count": len(payload.get("artifacts") or {}),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="crowdtensor_kaggle_real_runtime_check_") as temp:
        output_dir = Path(temp) / "kaggle-real-runtime"
        payload = run_json(
            [
                sys.executable,
                str(ROOT / "scripts" / "kaggle_real_runtime_acceptance_pack.py"),
                "prepare",
                "--public-host",
                args.public_host,
                "--port",
                str(args.port),
                "--miner-id",
                args.miner_id,
                "--workload",
                args.workload,
                "--output-dir",
                str(output_dir),
                "--request-count",
                str(args.request_count),
                "--scenario-id",
                args.scenario_id,
                "--decode-steps",
                str(args.decode_steps),
                "--stage-mode",
                args.stage_mode,
                "--timeout-seconds",
                str(args.command_timeout),
                "--replace",
                "--json",
            ],
            timeout=args.command_timeout,
        )
        steps = [validate_prepare(payload, output_dir, args)]
    return {
        "schema": SCHEMA,
        "ok": all(step.get("ok") for step in steps),
        "steps": steps,
        "diagnosis_codes": sorted({"kaggle_artifacts_ready", "kaggle_real_runtime_prepare_ready"}),
        "safety": {
            "ci_safe_prepare_only": True,
            "real_kaggle_runtime_not_claimed_by_check": True,
            "token_redaction_checked": True,
            "temporary_http_boundary_checked": True,
            "operator_env_excluded_from_kaggle": True,
            "not_production": True,
            "not_p2p": True,
        },
        "limitations": [
            "This check validates generated real-runtime artifacts only.",
            "It does not claim that a Kaggle Notebook connected; that requires kaggle_real_runtime_ready from the verify action.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Kaggle real runtime acceptance preparation artifacts.")
    parser.add_argument("--public-host", default="24.199.118.54")
    parser.add_argument("--port", type=int, default=9180)
    parser.add_argument("--miner-id", default="kaggle-cpu-1")
    parser.add_argument("--workload", choices=WORKLOAD_CHOICES, default=MODEL_BUNDLE_KIND)
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--scenario-id", default="route-baseline")
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--stage-mode", choices=["both", "split"], default="both")
    parser.add_argument("--require-distinct-stage-miners", action="store_true")
    parser.add_argument("--command-timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.port < 1:
        raise SystemExit("--port must be positive")
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    if args.decode_steps < 1 or args.decode_steps > 4:
        raise SystemExit("--decode-steps must be between 1 and 4")
    if args.workload == MICRO_LLM_SHARDED_KIND and args.stage_mode == "split":
        args.require_distinct_stage_miners = True
    report = build_report(args)
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
