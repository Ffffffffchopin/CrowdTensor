#!/usr/bin/env python3
"""CI-safe validation for the core technology Handoff RC contract."""

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

from crowdtensor import core_technology_handoff as handoff  # noqa: E402
from crowdtensor import large_model_inference_rc as inference_rc  # noqa: E402


REQUIRED_CODES = {
    "core_technology_handoff_rc_ready",
    "core_technology_stable_entrypoint_ready",
    "core_technology_inference_rc_imported",
    "core_technology_deployment_runbook_ready",
    "core_technology_next_layer_contract_ready",
    "core_technology_adapter_conformance_ready",
    "core_technology_test_gates_ready",
    "core_technology_public_artifact_redaction_ready",
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
            str(ROOT / "scripts" / "core_technology_handoff_pack.py"),
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
        timeout=120,
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
    if report.get("schema") != handoff.HANDOFF_SCHEMA:
        raise SystemExit(f"unexpected schema: {report.get('schema')}")
    if report.get("ok") is not True:
        raise SystemExit(f"Handoff RC report not ready: {report.get('diagnosis_codes')} blockers={report.get('blockers')}")
    codes = set(report.get("diagnosis_codes") or [])
    missing_codes = sorted(REQUIRED_CODES - codes)
    if missing_codes:
        raise SystemExit(f"missing required diagnosis codes: {missing_codes}")
    entrypoints = set(report.get("stable_entrypoints") or [])
    for entry in [
        "crowdtensor large-model-shard-rc",
        "crowdtensor core-tech-handoff",
        "scripts/core_technology_handoff_pack.py",
        "scripts/core_technology_handoff_check.py",
    ]:
        if entry not in entrypoints:
            raise SystemExit(f"missing stable entrypoint: {entry}")
    inference_report = report.get("inference_rc_report") if isinstance(report.get("inference_rc_report"), dict) else {}
    if inference_report.get("schema") != inference_rc.RC_SCHEMA or inference_report.get("ok") is not True:
        raise SystemExit("embedded Inference RC report missing or not ready")
    deployment = report.get("deployment_runbook") if isinstance(report.get("deployment_runbook"), dict) else {}
    if deployment.get("schema") != handoff.DEPLOYMENT_RUNBOOK_SCHEMA or deployment.get("ready") is not True:
        raise SystemExit("deployment runbook missing")
    for section in ["local_fixture", "local_real_runtime", "lan_vpn_two_worker_runtime", "import_retained_evidence", "troubleshooting", "cleanup"]:
        if not deployment.get(section):
            raise SystemExit(f"deployment runbook missing section: {section}")
    next_layer = report.get("next_layer_integration_contract") if isinstance(report.get("next_layer_integration_contract"), dict) else {}
    if next_layer.get("schema") != handoff.NEXT_LAYER_CONTRACT_SCHEMA or next_layer.get("ready") is not True:
        raise SystemExit("next-layer integration contract missing")
    for section in ["control_layer", "user_layer", "permissions_trust_billing_layer", "sample_control_request"]:
        if not next_layer.get(section):
            raise SystemExit(f"next-layer contract missing section: {section}")
    adapter = report.get("adapter_conformance") if isinstance(report.get("adapter_conformance"), dict) else {}
    if adapter.get("schema") != handoff.ADAPTER_CONFORMANCE_SCHEMA or adapter.get("ready") is not True:
        raise SystemExit("adapter conformance missing")
    future = set(adapter.get("future_runtime_backends") or [])
    if future != set(inference_rc.UNSUPPORTED_RUNTIMES):
        raise SystemExit(f"future runtime descriptors mismatch: {future}")
    tests = report.get("test_gate_summary") if isinstance(report.get("test_gate_summary"), dict) else {}
    if tests.get("schema") != handoff.TEST_GATE_SCHEMA or tests.get("ready") is not True:
        raise SystemExit("test gate summary missing")
    coverage = set(tests.get("coverage") or [])
    for item in [
        "CLI/API entry",
        "adapter interface",
        "deployment/runbook artifact generation",
        "aggregate handoff report",
        "backward compatibility",
    ]:
        if item not in coverage:
            raise SystemExit(f"test coverage item missing: {item}")
    answers = report.get("handoff_answers") if isinstance(report.get("handoff_answers"), dict) else {}
    for field in [
        "what_core_can_do",
        "control_layer_call",
        "user_layer_call",
        "permissions_trust_billing_dependencies",
        "external_runtime_future_work",
    ]:
        if not answers.get(field):
            raise SystemExit(f"handoff answer missing: {field}")
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    if safety.get("public_artifact_safe") is not True:
        raise SystemExit(f"public artifact safety failed: {safety}")
    errors = handoff.public_redaction_errors(report)
    if errors:
        raise SystemExit(f"public report leaked sensitive fragments: {errors}")
    for name in [
        "summary_json",
        "summary_markdown",
        "support_bundle_json",
        "inference_rc_json",
        "deployment_runbook_json",
        "next_layer_contract_json",
        "adapter_conformance_json",
        "test_gate_summary_json",
    ]:
        artifact_path(report, name)
    if report.get("real_runtime_verified") is not True:
        if report.get("real_7b_runtime_verified") is not False:
            raise SystemExit("fixture/plan Handoff RC must keep real_7b_runtime_verified=false")
        for code in ["core_technology_real_runtime_not_verified", "core_technology_handoff_fixture_or_import_ready"]:
            if code not in codes:
                raise SystemExit(f"missing non-real diagnosis code: {code}")
        blockers = set(report.get("blockers") or [])
        if "external_real_runtime_resources_required" not in blockers:
            raise SystemExit("missing external runtime blocker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate core technology Handoff RC evidence.")
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
        output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_handoff_rc_check_"))
        report = run_pack(output_dir, mode=args.mode)
    validate_report(report)
    result = {
        "ok": True,
        "schema": handoff.HANDOFF_CHECK_SCHEMA,
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
