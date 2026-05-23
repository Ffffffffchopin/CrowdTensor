#!/usr/bin/env python3
"""Deterministic acceptance check for the runtime capability matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import runtime_matrix  # noqa: E402


CPU_REQUIRED = {
    "diloco_train",
    "cpu_lora_mock",
    "micro_transformer_lm",
    "model_bundle_lm",
    "model_bundle_infer",
    "external_llm_infer_mock",
}

REQUIRED_TARGETS = {
    "cpu_baseline",
    "nvidia_cuda",
    "amd_rocm",
    "apple_metal",
    "browser",
    "remote_container",
    "external_llm_command",
    "external_llm_http",
}

OPERATOR_ACTIONS = {
    "run_now",
    "configure_optional_runtime",
    "future_adapter",
    "fix_blocker",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CrowdTensorD runtime capability matrix output.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8894)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--root", default=str(ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = runtime_matrix.build_matrix(root=Path(args.root))
    workloads = {row["name"]: row for row in matrix["workloads"]}
    missing = [name for name in sorted(CPU_REQUIRED) if workloads.get(name, {}).get("status") != "available"]
    if missing:
        raise SystemExit(f"CPU baseline workloads must be available: {missing}")
    for name in ["external_llm_infer_command", "external_llm_infer_http", "browser_probe"]:
        status = workloads.get(name, {}).get("status")
        if status not in {"available", "configured", "optional_missing", "blocked"}:
            raise SystemExit(f"unexpected status for {name}: {status}")
        if status == "blocked" and name != "external_llm_infer_command":
            raise SystemExit(f"{name} should be optional_missing when not configured")
    targets = {row["name"]: row for row in matrix.get("hardware_targets", [])}
    missing_targets = sorted(REQUIRED_TARGETS - set(targets))
    if missing_targets:
        raise SystemExit(f"missing hardware targets: {missing_targets}")
    if targets["cpu_baseline"].get("status") != "available" or not targets["cpu_baseline"].get("usable_now"):
        raise SystemExit(f"CPU baseline target must be usable: {targets['cpu_baseline']}")
    for name, target in targets.items():
        if target.get("status") not in {"available", "configured", "detected", "optional_missing", "blocked"}:
            raise SystemExit(f"unexpected hardware target status for {name}: {target}")
        if not isinstance(target.get("matched_capabilities"), list):
            raise SystemExit(f"hardware target must include matched_capabilities list: {target}")
        if not isinstance(target.get("missing_capabilities"), list):
            raise SystemExit(f"hardware target must include missing_capabilities list: {target}")
        if not isinstance(target.get("diagnosis_codes"), list):
            raise SystemExit(f"hardware target must include diagnosis_codes list: {target}")
        if target.get("operator_action") not in OPERATOR_ACTIONS:
            raise SystemExit(f"hardware target must include valid operator_action: {target}")
    routes = {row["name"]: row for row in matrix.get("recommended_routes", [])}
    for name, route_row in routes.items():
        if route_row.get("confidence") not in {"ready", "configured", "future", "blocked"}:
            raise SystemExit(f"unexpected route confidence for {name}: {route_row}")
        if not route_row.get("reason"):
            raise SystemExit(f"route must include reason: {route_row}")
        if not isinstance(route_row.get("matched_capabilities"), list):
            raise SystemExit(f"route must include matched_capabilities list: {route_row}")
        if not isinstance(route_row.get("missing_capabilities"), list):
            raise SystemExit(f"route must include missing_capabilities list: {route_row}")
        if not isinstance(route_row.get("diagnosis_codes"), list):
            raise SystemExit(f"route must include diagnosis_codes list: {route_row}")
        if route_row.get("operator_action") not in OPERATOR_ACTIONS:
            raise SystemExit(f"route must include valid operator_action: {route_row}")
    route = routes.get("local_cpu_model_bundle_infer")
    if (
        not route
        or route.get("status") not in {"available", "configured"}
        or not route.get("usable_now")
        or route.get("confidence") != "ready"
    ):
        raise SystemExit(f"home-compute route must be usable: {route}")
    diagnosis = matrix.get("diagnosis_summary") or {}
    if not isinstance(diagnosis.get("codes"), list):
        raise SystemExit(f"runtime matrix must include diagnosis_summary.codes: {diagnosis}")
    route_codes = sorted({
        str(code)
        for route_row in routes.values()
        for code in route_row.get("diagnosis_codes") or []
    })
    if sorted(diagnosis.get("codes") or []) != route_codes:
        raise SystemExit(f"diagnosis summary must aggregate route codes: {diagnosis}")
    hardware_diagnosis = matrix.get("hardware_diagnosis_summary") or {}
    if not isinstance(hardware_diagnosis.get("codes"), list):
        raise SystemExit(f"runtime matrix must include hardware_diagnosis_summary.codes: {hardware_diagnosis}")
    target_codes = sorted({
        str(code)
        for target in targets.values()
        for code in target.get("diagnosis_codes") or []
    })
    if sorted(hardware_diagnosis.get("codes") or []) != target_codes:
        raise SystemExit(f"hardware diagnosis summary must aggregate target codes: {hardware_diagnosis}")
    payload = json.dumps(matrix, sort_keys=True)
    for secret_fragment in ["local-runtime-key", "CROWDTENSOR_LLM_RUNTIME_API_KEY=", "Bearer "]:
        if secret_fragment in payload:
            raise SystemExit("runtime matrix leaked secret-like material")
    print(json.dumps({
        "ok": matrix["ok"],
        "available": matrix["summary"]["available"],
        "optional_missing": matrix["summary"]["optional_missing"],
        "blocked": matrix["summary"]["blocked"],
        "cpu_required": sorted(CPU_REQUIRED),
        "hardware_targets": sorted(REQUIRED_TARGETS),
        "diagnosis_codes": diagnosis["codes"],
        "hardware_diagnosis_codes": hardware_diagnosis["codes"],
        "route": route["name"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
