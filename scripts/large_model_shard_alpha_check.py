#!/usr/bin/env python3
"""CI-safe validation for the Large-Model Shard Alpha contract."""

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

from crowdtensor import large_model_shard as lms  # noqa: E402


REQUIRED_CODES = {
    "large_model_runtime_adapter_ready",
    "large_model_partition_manifest_ready",
    "large_model_layer_range_placement_ready",
    "large_model_memory_budget_ready",
    "large_model_workload_contract_ready",
    "large_model_benchmark_harness_ready",
    "large_model_single_device_comparison_ready",
    "large_model_sharded_adapter_comparison_ready",
    "large_model_serving_hooks_ready",
    "large_model_public_artifact_redaction_ready",
    "large_model_core_alpha_boundary_ready",
}


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


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


def run_pack(output_dir: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "large_model_shard_alpha_pack.py"),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
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


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema") != lms.ALPHA_SCHEMA:
        raise SystemExit(f"unexpected schema: {report.get('schema')}")
    if report.get("ok") is not True:
        raise SystemExit(f"large model shard alpha report not ready: {report.get('diagnosis_codes')}")
    alpha = report.get("alpha") if isinstance(report.get("alpha"), dict) else {}
    if alpha.get("ready") is not True:
        raise SystemExit(f"alpha summary not ready: {alpha}")
    codes = set(report.get("diagnosis_codes") or [])
    missing = sorted(REQUIRED_CODES - codes)
    if missing:
        raise SystemExit(f"missing required diagnosis codes: {missing}")
    adapter = report.get("runtime_adapter") if isinstance(report.get("runtime_adapter"), dict) else {}
    if adapter.get("schema") != lms.RUNTIME_ADAPTER_SCHEMA:
        raise SystemExit("runtime adapter schema missing")
    if adapter.get("adapter_kind") != "llama_cpp_rpc":
        raise SystemExit(f"unexpected adapter kind: {adapter.get('adapter_kind')}")
    if adapter.get("not_public_rpc_safe") is not True or adapter.get("controlled_network_only") is not True:
        raise SystemExit("llama.cpp RPC adapter must remain controlled-network only")
    if not adapter.get("client_command_line") or "--rpc" not in str(adapter.get("client_command_line")):
        raise SystemExit("adapter must include llama.cpp client --rpc command")
    partition = report.get("partition_manifest") if isinstance(report.get("partition_manifest"), dict) else {}
    if partition.get("schema") != lms.PARTITION_MANIFEST_SCHEMA or partition.get("runnable") is not True:
        raise SystemExit(f"partition manifest not ready: {partition}")
    if int(partition.get("assigned_layer_count") or 0) <= 0:
        raise SystemExit("partition must assign layers")
    contract = report.get("workload_contract") if isinstance(report.get("workload_contract"), dict) else {}
    if contract.get("schema") != lms.WORKLOAD_CONTRACT_SCHEMA:
        raise SystemExit("workload contract schema missing")
    redaction = contract.get("redaction_policy") if isinstance(contract.get("redaction_policy"), dict) else {}
    for field in [
        "raw_prompt_public",
        "raw_generated_text_public",
        "generated_token_ids_public",
        "activation_public",
        "kv_cache_public",
        "credentials_public",
        "lease_material_public",
    ]:
        if redaction.get(field) is not False:
            raise SystemExit(f"redaction policy must set {field}=False: {redaction}")
    serving = contract.get("serving_readiness") if isinstance(contract.get("serving_readiness"), dict) else {}
    for field in [
        "streaming_event_schema",
        "bounded_batch_request_schema",
        "cancellation_field",
        "health_aware_route_metadata_schema",
    ]:
        if not serving.get(field):
            raise SystemExit(f"serving readiness hook missing: {field}")
    benchmark = report.get("benchmark") if isinstance(report.get("benchmark"), dict) else {}
    if benchmark.get("schema") != lms.BENCHMARK_SCHEMA:
        raise SystemExit("benchmark schema missing")
    if benchmark.get("real_runtime_verified") is True and benchmark.get("measurement_kind") != "real-runtime":
        raise SystemExit("real_runtime_verified requires measurement_kind=real-runtime")
    if benchmark.get("real_runtime_verified") is not True:
        if benchmark.get("fixture") is not True or benchmark.get("measurement_kind") != "fixture-planning-estimate":
            raise SystemExit("fixture benchmark must be explicitly marked")
    for path_name in [
        "summary_json",
        "summary_markdown",
        "runtime_adapter_json",
        "partition_manifest_json",
        "workload_contract_json",
        "benchmark_json",
        "support_bundle_json",
    ]:
        artifact_path(report, path_name)
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    if safety.get("public_artifact_safe") is not True:
        raise SystemExit(f"public artifact safety failed: {safety}")
    errors = lms.public_redaction_errors(report)
    if errors:
        raise SystemExit(f"public report leaked sensitive fragments: {errors}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Large-Model Shard Alpha output.")
    parser.add_argument("--report", default="", help="existing large_model_shard_alpha.json to validate")
    parser.add_argument("--output-dir", default="", help="directory for generated fixture check when --report is omitted")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report:
        report = load_json(Path(args.report))
    else:
        output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_large_model_alpha_check_"))
        report = run_pack(output_dir)
    validate_report(report)
    result = {
        "ok": True,
        "schema": "large_model_shard_alpha_check_v1",
        "report_schema": report.get("schema"),
        "real_runtime_verified": bool(report.get("real_runtime_verified")),
        "evidence_scope": report.get("evidence_scope"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "output_dir": report.get("output_dir"),
    }
    print(json.dumps(result, sort_keys=True) if args.json else json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
