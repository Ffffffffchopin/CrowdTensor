#!/usr/bin/env python3
"""CI-safe validation for the core technology Inference RC contract."""

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

from crowdtensor import large_model_inference_rc as rc  # noqa: E402


REQUIRED_CODES = {
    "core_technology_inference_rc_ready",
    "large_model_runtime_adapter_interface_ready",
    "large_model_runtime_adapter_probe_ready",
    "large_model_device_profile_v2_ready",
    "large_model_partition_manifest_v2_ready",
    "large_model_runner_supervisor_ready",
    "large_model_benchmark_v2_ready",
    "large_model_correctness_summary_ready",
    "large_model_serving_hooks_v1_ready",
    "large_model_public_artifact_redaction_ready",
}


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def run_pack(output_dir: Path, *, mode: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "large_model_inference_rc_pack.py"),
            "--mode",
            mode,
            "--output-dir",
            str(output_dir),
            "--json",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
    )
    if completed.returncode != 0:
        raise SystemExit(f"pack failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    for line in reversed([line.strip() for line in completed.stdout.splitlines() if line.strip()]):
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    raise SystemExit(f"pack emitted no JSON\nstdout:\n{completed.stdout}")


def artifact_path(report: dict[str, Any], name: str) -> Path:
    output_dir = Path(str(report.get("output_dir") or "."))
    artifact = (report.get("artifacts") or {}).get(name) if isinstance(report.get("artifacts"), dict) else {}
    if not isinstance(artifact, dict):
        raise SystemExit(f"missing artifact entry: {name}")
    path = Path(str(artifact.get("path") or ""))
    if not path.is_absolute():
        path = output_dir / path
    if not path.is_file():
        raise SystemExit(f"artifact missing: {name} at {path}")
    return path


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema") != rc.RC_SCHEMA:
        raise SystemExit(f"unexpected schema: {report.get('schema')}")
    if report.get("ok") is not True:
        raise SystemExit(f"Inference RC report not ready: {report.get('diagnosis_codes')} blockers={report.get('blockers')}")
    codes = set(report.get("diagnosis_codes") or [])
    missing = sorted(REQUIRED_CODES - codes)
    if missing:
        raise SystemExit(f"missing required diagnosis codes: {missing}")
    adapter_interface = report.get("adapter_interface") if isinstance(report.get("adapter_interface"), dict) else {}
    if adapter_interface.get("schema") != rc.RUNTIME_ADAPTER_INTERFACE_SCHEMA:
        raise SystemExit("adapter interface schema missing")
    descriptors = adapter_interface.get("descriptors") if isinstance(adapter_interface.get("descriptors"), list) else []
    unsupported = [item for item in descriptors if isinstance(item, dict) and item.get("status") == "unsupported"]
    if len(unsupported) < len(rc.UNSUPPORTED_RUNTIMES):
        raise SystemExit("future runtime unsupported descriptors missing")
    runtime_probe = report.get("runtime_adapter_probe") if isinstance(report.get("runtime_adapter_probe"), dict) else {}
    if runtime_probe.get("schema") != rc.RUNTIME_ADAPTER_PROBE_SCHEMA:
        raise SystemExit("runtime probe schema missing")
    device_profile = report.get("device_profile") if isinstance(report.get("device_profile"), dict) else {}
    if device_profile.get("schema") != rc.DEVICE_PROFILE_SCHEMA:
        raise SystemExit("device profile v2 schema missing")
    partition = report.get("partition_manifest") if isinstance(report.get("partition_manifest"), dict) else {}
    if partition.get("schema") != rc.PARTITION_MANIFEST_SCHEMA:
        raise SystemExit("partition v2 schema missing")
    if not (partition.get("tensor_split_plan") or {}).get("tensor_split"):
        raise SystemExit("tensor split plan missing")
    runner = report.get("runner_result") if isinstance(report.get("runner_result"), dict) else {}
    if runner.get("schema") != rc.RUNNER_RESULT_SCHEMA:
        raise SystemExit("runner result schema missing")
    if runner.get("runner_mode") == "real" and runner.get("real_runtime_verified") is not True:
        raise SystemExit("real mode requires real runtime verification")
    benchmark = report.get("benchmark") if isinstance(report.get("benchmark"), dict) else {}
    if benchmark.get("schema") != rc.BENCHMARK_SCHEMA:
        raise SystemExit("benchmark v2 schema missing")
    correctness = report.get("correctness_summary") if isinstance(report.get("correctness_summary"), dict) else {}
    if correctness.get("schema") != rc.CORRECTNESS_SCHEMA:
        raise SystemExit("correctness summary schema missing")
    serving = report.get("serving_readiness_hooks") if isinstance(report.get("serving_readiness_hooks"), dict) else {}
    if serving.get("schema") != rc.SERVING_HOOKS_SCHEMA:
        raise SystemExit("serving hooks schema missing")
    for field in [
        "streaming_event_schema",
        "bounded_batch_request_schema",
        "cancellation_field",
        "timeout_field",
        "health_aware_route_metadata_schema",
    ]:
        if not serving.get(field):
            raise SystemExit(f"serving hook missing: {field}")
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    if safety.get("public_artifact_safe") is not True:
        raise SystemExit(f"public artifact safety failed: {safety}")
    errors = rc.public_redaction_errors(report)
    if errors:
        raise SystemExit(f"public report leaked sensitive fragments: {errors}")
    for name in [
        "summary_json",
        "summary_markdown",
        "support_bundle_json",
        "runtime_adapter_interface_json",
        "runtime_adapter_probe_json",
        "device_profile_json",
        "partition_manifest_v2_json",
        "runner_result_json",
        "benchmark_v2_json",
        "correctness_summary_json",
        "serving_hooks_json",
    ]:
        artifact_path(report, name)
    if report.get("real_runtime_verified") is not True:
        if report.get("real_7b_runtime_verified") is not False:
            raise SystemExit("fixture/plan RC must keep real_7b_runtime_verified=false")
        if "core_technology_real_7b_runtime_not_verified" not in codes:
            raise SystemExit("missing real runtime not-verified code")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate core technology Inference RC evidence.")
    parser.add_argument("--report", default="")
    parser.add_argument("--mode", choices=["plan", "fixture"], default="fixture")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report:
        report = load_json(Path(args.report))
    else:
        output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_inference_rc_check_"))
        report = run_pack(output_dir, mode=args.mode)
    validate_report(report)
    result = {
        "ok": True,
        "schema": "core_technology_inference_rc_check_v1",
        "report_schema": report.get("schema"),
        "mode": report.get("mode"),
        "real_runtime_verified": bool(report.get("real_runtime_verified")),
        "real_7b_runtime_verified": bool(report.get("real_7b_runtime_verified")),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "output_dir": report.get("output_dir"),
    }
    print(json.dumps(result, sort_keys=True) if args.json else json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
