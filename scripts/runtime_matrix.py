#!/usr/bin/env python3
"""Report local CrowdTensorD runtime capability readiness."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


CPU_BASELINE_WORKLOADS = [
    ("diloco_train", "Dense toy DiLoCo training contract"),
    ("cpu_lora_mock", "CPU LoRA-style adapter contract"),
    ("micro_transformer_lm", "Tiny CPU micro Transformer LM contract"),
    ("model_bundle_lm", "Model bundle training-shaped contract"),
    ("model_bundle_infer", "Read-only Swarm Inference-shaped bundle probe"),
]

OPERATOR_ACTIONS = {
    "run_now",
    "configure_optional_runtime",
    "future_adapter",
    "fix_blocker",
}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _project_files_ok(root: Path) -> bool:
    required = [
        "coordinator.py",
        "miner_cli.py",
        "crowdtensor",
        "scripts/runtime_acceptance_pack.py",
    ]
    return all((root / relative).exists() for relative in required)


def _status(
    name: str,
    *,
    status: str,
    reason: str,
    next_command: str,
    optional: bool,
    cpu_only: bool,
    required_config: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "next_command": next_command,
        "required_config": required_config or [],
        "optional": bool(optional),
        "cpu_only": bool(cpu_only),
    }


def _command_executable_available(command: str) -> bool:
    try:
        args = shlex.split(command)
    except ValueError:
        return False
    if not args:
        return False
    executable = args[0]
    if "/" in executable:
        return Path(executable).exists()
    return shutil.which(executable) is not None


def executable_available(command: str) -> bool:
    return shutil.which(command) is not None


def host_profile(root: Path) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 11),
        "platform": platform.platform(),
        "os": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count() or 1,
        "root": str(root),
    }


def _target(
    name: str,
    *,
    status: str,
    reason: str,
    next_command: str,
    optional: bool,
    usable_now: bool,
    supported_workloads: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "next_command": next_command,
        "optional": bool(optional),
        "usable_now": bool(usable_now),
        "supported_workloads": supported_workloads or [],
    }


def build_hardware_targets(
    *,
    core_ok: bool,
    profile: dict[str, Any],
    playwright_ok: bool,
    browser_available: bool,
    command_configured: bool,
    command_available: bool,
    http_configured: bool,
    source_env: dict[str, str],
) -> list[dict[str, Any]]:
    cpu_workloads = [name for name, _label in CPU_BASELINE_WORKLOADS] + ["external_llm_infer_mock"]
    nvidia_detected = executable_available("nvidia-smi") or executable_available("nvcc")
    amd_detected = executable_available("rocminfo") or executable_available("rocm-smi")
    apple_detected = profile.get("os") == "Darwin" and str(profile.get("machine", "")).lower() in {"arm64", "aarch64"}
    container_detected = (
        Path("/.dockerenv").exists()
        or bool(source_env.get("KAGGLE_KERNEL_RUN_TYPE"))
        or bool(source_env.get("CROWDTENSOR_REMOTE_CONTAINER"))
    )
    return [
        _target(
            "cpu_baseline",
            status="available" if core_ok else "blocked",
            reason="CPU-only Coordinator/Miner workloads are ready" if core_ok else
            "Python >=3.11, fastapi, uvicorn, or project files are missing",
            next_command="python3 scripts/home_compute_demo.py --port 8909 --request-count 4 --json",
            optional=False,
            usable_now=core_ok,
            supported_workloads=cpu_workloads if core_ok else [],
        ),
        _target(
            "nvidia_cuda",
            status="detected" if nvidia_detected else "optional_missing",
            reason="NVIDIA tooling was detected, but no CUDA runtime adapter is implemented yet" if nvidia_detected else
            "Install NVIDIA drivers/tooling when future CUDA adapters are available",
            next_command="python3 scripts/runtime_matrix.py --json",
            optional=True,
            usable_now=False,
        ),
        _target(
            "amd_rocm",
            status="detected" if amd_detected else "optional_missing",
            reason="ROCm tooling was detected, but no AMD runtime adapter is implemented yet" if amd_detected else
            "Install ROCm tooling when future AMD adapters are available",
            next_command="python3 scripts/runtime_matrix.py --json",
            optional=True,
            usable_now=False,
        ),
        _target(
            "apple_metal",
            status="detected" if apple_detected else "optional_missing",
            reason="Apple Silicon was detected, but no Metal runtime adapter is implemented yet" if apple_detected else
            "Run on Apple Silicon when future Metal adapters are available",
            next_command="python3 scripts/runtime_matrix.py --json",
            optional=True,
            usable_now=False,
        ),
        _target(
            "browser",
            status="available" if playwright_ok and browser_available else "optional_missing",
            reason="Playwright and a Chromium-compatible browser were found" if playwright_ok and browser_available else
            "Install Playwright/browser extras to run browser-native checks",
            next_command="python3 scripts/browser_acceptance_pack.py --allow-skip --base-port 9310",
            optional=True,
            usable_now=bool(playwright_ok and browser_available),
            supported_workloads=["browser_probe"] if playwright_ok and browser_available else [],
        ),
        _target(
            "remote_container",
            status="detected" if container_detected else "optional_missing",
            reason="Container-like environment detected; controlled remote Miner demos can run with operator networking" if container_detected else
            "Set up a Linux container or remote host for controlled remote Miner demos",
            next_command="python3 scripts/remote_miner_readiness_check.py --port 8895",
            optional=True,
            usable_now=False,
        ),
        _target(
            "external_llm_command",
            status="configured" if command_configured and command_available else (
                "blocked" if command_configured else "optional_missing"
            ),
            reason="Command runtime is configured and executable" if command_configured and command_available else (
                "Configured command executable was not found" if command_configured else
                "Set CROWDTENSOR_LLM_RUNTIME_CMD to enable an operator-owned command runtime"
            ),
            next_command="crowdtensor-miner --llm-runtime-cmd /path/to/wrapper",
            optional=True,
            usable_now=bool(command_configured and command_available),
            supported_workloads=["external_llm_infer"] if command_configured and command_available else [],
        ),
        _target(
            "external_llm_http",
            status="configured" if http_configured else "optional_missing",
            reason="OpenAI-compatible HTTP endpoint is configured" if http_configured else
            "Set CROWDTENSOR_LLM_RUNTIME_URL to enable an operator-owned HTTP runtime",
            next_command="python3 scripts/external_llm_http_adapter_smoke.py --port 8907 --runtime-port 8908",
            optional=True,
            usable_now=bool(http_configured),
            supported_workloads=["external_llm_infer"] if http_configured else [],
        ),
    ]


def build_recommended_routes(hardware_targets: list[dict[str, Any]], workloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_by_name = {target["name"]: target for target in hardware_targets}
    workload_by_name = {workload["name"]: workload for workload in workloads}
    route_specs = [
        (
            "local_cpu_model_bundle_infer",
            "cpu_baseline",
            "model_bundle_infer",
            "python3 scripts/home_compute_demo.py --port 8909 --request-count 4 --json",
        ),
        (
            "local_cpu_acceptance",
            "cpu_baseline",
            "diloco_train",
            "python3 scripts/runtime_acceptance_pack.py --base-port 8910 --report /tmp/crowdtensor_acceptance.json",
        ),
        (
            "browser_probe",
            "browser",
            "browser_probe",
            "python3 scripts/browser_acceptance_pack.py --allow-skip --base-port 9310",
        ),
        (
            "external_llm_http_adapter",
            "external_llm_http",
            "external_llm_infer",
            "python3 scripts/external_llm_http_adapter_smoke.py --port 8907 --runtime-port 8908",
        ),
        (
            "external_llm_command_adapter",
            "external_llm_command",
            "external_llm_infer",
            "crowdtensor-miner --llm-runtime-cmd /path/to/wrapper",
        ),
    ]
    routes = []
    for name, target_name, workload_name, next_command in route_specs:
        target = target_by_name.get(target_name, {})
        workload = workload_by_name.get(workload_name, {})
        workload_ready = (
            workload.get("status") in {"available", "configured"}
            or workload_name in set(target.get("supported_workloads") or [])
        )
        status = target.get("status") or workload.get("status") or "optional_missing"
        if target.get("usable_now") and workload_ready:
            status = "available" if target.get("status") == "available" else "configured"
        usable_now = bool(target.get("usable_now") and workload_ready)
        matched = []
        missing = []
        if target.get("usable_now"):
            matched.append(f"target:{target_name}")
        else:
            missing.append(f"target:{target_name}")
        if workload_ready:
            matched.append(f"workload:{workload_name}")
        else:
            missing.append(f"workload:{workload_name}")
        for capability in target.get("supported_workloads") or []:
            if capability == workload_name:
                matched.append(f"supported_workload:{capability}")
        diagnosis_codes: list[str] = []
        if target_name == "cpu_baseline" and usable_now:
            matched.extend(["python_runtime", "cpu_only_contract"])
            diagnosis_codes.append("cpu_baseline_ready")
        elif target_name == "cpu_baseline":
            missing.extend(["python_runtime", "cpu_only_contract"])
            diagnosis_codes.append("cpu_baseline_blocked")
        if target_name in {"nvidia_cuda", "amd_rocm", "apple_metal"}:
            missing.append("runtime_adapter_not_implemented")
            if status == "detected":
                diagnosis_codes.append("accelerator_adapter_not_implemented")
        if target_name == "browser" and not usable_now:
            missing.append("browser_runtime")
            diagnosis_codes.append("browser_runtime_missing")
        if target_name == "remote_container":
            missing.append("operator_networking")
        if target_name == "external_llm_command" and not usable_now:
            missing.append("command_runtime")
            if status == "blocked":
                diagnosis_codes.append("external_llm_command_missing")
        if target_name == "external_llm_http" and not usable_now:
            missing.append("http_runtime_url")
        if target_name == "external_llm_http" and usable_now:
            diagnosis_codes.append("external_llm_http_configured")
        if usable_now and status == "available":
            confidence = "ready"
        elif usable_now and status == "configured":
            confidence = "configured"
        elif status in {"detected", "optional_missing"}:
            confidence = "future"
        else:
            confidence = "blocked"
        if usable_now:
            operator_action = "run_now"
        elif confidence == "blocked":
            operator_action = "fix_blocker"
        elif target_name in {"nvidia_cuda", "amd_rocm", "apple_metal"} and status == "detected":
            operator_action = "future_adapter"
        elif target_name == "remote_container":
            operator_action = "configure_optional_runtime"
        elif status in {"optional_missing", "detected"}:
            operator_action = "configure_optional_runtime"
        else:
            operator_action = "fix_blocker"
        reason = (
            f"{target_name} can run {workload_name}"
            if usable_now else
            f"{target_name} cannot run {workload_name}: "
            f"{target.get('reason') or workload.get('reason') or 'capability is missing'}"
        )
        routes.append({
            "name": name,
            "target": target_name,
            "workload": workload_name,
            "status": status,
            "usable_now": usable_now,
            "confidence": confidence,
            "reason": reason,
            "matched_capabilities": sorted(set(matched)),
            "missing_capabilities": sorted(set(missing)),
            "diagnosis_codes": sorted(set(diagnosis_codes)),
            "operator_action": operator_action,
            "next_command": next_command,
        })
    return routes


def diagnosis_summary(routes: list[dict[str, Any]]) -> dict[str, Any]:
    by_route: dict[str, list[str]] = {}
    all_codes: list[str] = []
    for route in routes:
        codes = [str(code) for code in route.get("diagnosis_codes") or [] if code]
        if codes:
            name = str(route.get("name") or "<unnamed>")
            by_route[name] = codes
            all_codes.extend(codes)
    return {
        "codes": sorted(set(all_codes)),
        "by_route": by_route,
    }


def build_matrix(
    *,
    root: Path = ROOT,
    env: dict[str, str] | None = None,
    browser_path: str = "",
    llm_runtime_cmd: str = "",
    llm_runtime_url: str = "",
    llm_runtime_api_key: str = "",
) -> dict[str, Any]:
    root = Path(root).resolve()
    source_env = dict(os.environ if env is None else env)
    profile = host_profile(root)
    fastapi_ok = module_available("fastapi")
    uvicorn_ok = module_available("uvicorn")
    project_ok = _project_files_ok(root)
    core_ok = bool(profile["python_ok"] and fastapi_ok and uvicorn_ok and project_ok)

    command_configured = bool(str(llm_runtime_cmd or source_env.get("CROWDTENSOR_LLM_RUNTIME_CMD", "")).strip())
    command_available = _command_executable_available(
        str(llm_runtime_cmd or source_env.get("CROWDTENSOR_LLM_RUNTIME_CMD", "")).strip()
    ) if command_configured else False
    http_configured = bool(str(llm_runtime_url or source_env.get("CROWDTENSOR_LLM_RUNTIME_URL", "")).strip())
    api_key_configured = bool(
        str(llm_runtime_api_key or source_env.get("CROWDTENSOR_LLM_RUNTIME_API_KEY", "")).strip()
    )
    detected_browser = str(browser_path or source_env.get("CROWDTENSOR_BROWSER", "")).strip()
    detected_browser = detected_browser or shutil.which("google-chrome") or shutil.which("chromium") or ""
    playwright_ok = module_available("playwright")

    workloads: list[dict[str, Any]] = []
    core_reason = "CPU baseline dependencies are available" if core_ok else (
        "Python >=3.11, fastapi, uvicorn, or project files are missing"
    )
    for workload, label in CPU_BASELINE_WORKLOADS:
        workloads.append(_status(
            workload,
            status="available" if core_ok else "blocked",
            reason=f"{label}: {core_reason}",
            next_command="python3 scripts/home_compute_demo.py --port 8909 --request-count 4 --json"
            if workload == "model_bundle_infer"
            else "python3 scripts/runtime_acceptance_pack.py --base-port 8910 --report /tmp/crowdtensor_acceptance.json",
            optional=False,
            cpu_only=True,
        ))
    workloads.append(_status(
        "external_llm_infer_mock",
        status="available" if core_ok else "blocked",
        reason="Deterministic mock external LLM adapter is available" if core_ok else core_reason,
        next_command="python3 scripts/external_llm_inference_smoke.py --port 8906 --request-count 3",
        optional=False,
        cpu_only=True,
    ))
    workloads.append(_status(
        "external_llm_infer_command",
        status="configured" if command_configured and command_available else (
            "blocked" if command_configured else "optional_missing"
        ),
        reason="Command adapter is configured" if command_configured and command_available else (
            "Configured command executable was not found" if command_configured else
            "Set CROWDTENSOR_LLM_RUNTIME_CMD or --llm-runtime-cmd to enable a command runtime"
        ),
        next_command="crowdtensor-miner --llm-runtime-cmd /path/to/wrapper",
        required_config=["CROWDTENSOR_LLM_RUNTIME_CMD"],
        optional=True,
        cpu_only=False,
    ))
    workloads.append(_status(
        "external_llm_infer_http",
        status="configured" if http_configured else "optional_missing",
        reason="OpenAI-compatible HTTP endpoint is configured" if http_configured else
        "Set CROWDTENSOR_LLM_RUNTIME_URL or --llm-runtime-url to enable an HTTP runtime",
        next_command="python3 scripts/external_llm_http_adapter_smoke.py --port 8907 --runtime-port 8908",
        required_config=["CROWDTENSOR_LLM_RUNTIME_URL"],
        optional=True,
        cpu_only=False,
    ))
    workloads.append(_status(
        "browser_probe",
        status="available" if playwright_ok and bool(detected_browser) else "optional_missing",
        reason="Playwright and a Chromium-compatible browser were found" if playwright_ok and detected_browser else
        "Install Playwright/browser extras to run browser-native checks",
        next_command="python3 scripts/browser_acceptance_pack.py --allow-skip --base-port 9310",
        required_config=["playwright", "chromium"],
        optional=True,
        cpu_only=False,
    ))

    available = [row["name"] for row in workloads if row["status"] in {"available", "configured"}]
    optional_missing = [row["name"] for row in workloads if row["status"] == "optional_missing"]
    blocked = [row["name"] for row in workloads if row["status"] == "blocked" and not row["optional"]]
    hardware_targets = build_hardware_targets(
        core_ok=core_ok,
        profile=profile,
        playwright_ok=playwright_ok,
        browser_available=bool(detected_browser),
        command_configured=command_configured,
        command_available=command_available,
        http_configured=http_configured,
        source_env=source_env,
    )
    recommended_routes = build_recommended_routes(hardware_targets, workloads)
    route_diagnosis = diagnosis_summary(recommended_routes)
    matrix = {
        "ok": not blocked,
        "host_profile": profile,
        "configured_runtimes": {
            "external_llm_command": {
                "configured": command_configured,
                "executable_available": command_available,
            },
            "external_llm_http": {
                "configured": http_configured,
                "api_key_configured": api_key_configured,
            },
            "browser": {
                "playwright_available": playwright_ok,
                "browser_available": bool(detected_browser),
            },
        },
        "hardware_targets": hardware_targets,
        "recommended_routes": recommended_routes,
        "workloads": workloads,
        "summary": {
            "available": len(available),
            "optional_missing": len(optional_missing),
            "blocked": len(blocked),
            "available_workloads": available,
            "optional_missing_workloads": optional_missing,
            "blocked_workloads": blocked,
        },
        "diagnosis_summary": route_diagnosis,
        "recommended_next_commands": [
            "python3 scripts/home_compute_demo.py --port 8909 --request-count 4 --json",
            "python3 scripts/runtime_acceptance_pack.py --base-port 8910 --report /tmp/crowdtensor_acceptance.json",
            "python3 scripts/external_llm_http_adapter_smoke.py --port 8907 --runtime-port 8908",
        ],
    }
    return matrix


def print_human(matrix: dict[str, Any]) -> None:
    profile = matrix["host_profile"]
    print("CrowdTensor Runtime Capability Matrix")
    print(
        f"  ok: {matrix['ok']} "
        f"python={profile['python']} os={profile['os']} "
        f"machine={profile['machine']} cpu_count={profile['cpu_count']}"
    )
    print("  workloads:")
    for workload in matrix["workloads"]:
        print(f"    - {workload['name']}: {workload['status']} ({workload['reason']})")
    print("  hardware targets:")
    for target in matrix.get("hardware_targets", []):
        print(f"    - {target['name']}: {target['status']} ({target['reason']})")
    print("  recommended routes:")
    for route in matrix.get("recommended_routes", []):
        print(
            f"    - {route['name']}: {route['status']} "
            f"confidence={route.get('confidence')} -> {route['next_command']}"
        )
    print("  recommended next commands:")
    for command in matrix["recommended_next_commands"]:
        print(f"    - {command}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report local CrowdTensorD runtime capability readiness.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--browser-path", default="")
    parser.add_argument("--llm-runtime-cmd", default="")
    parser.add_argument("--llm-runtime-url", default="")
    parser.add_argument("--llm-runtime-api-key", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    matrix = build_matrix(
        root=Path(args.root),
        browser_path=args.browser_path,
        llm_runtime_cmd=args.llm_runtime_cmd,
        llm_runtime_url=args.llm_runtime_url,
        llm_runtime_api_key=args.llm_runtime_api_key,
    )
    if args.json:
        print(json.dumps(matrix, sort_keys=True))
    else:
        print_human(matrix)
    raise SystemExit(0 if matrix["ok"] else 1)


if __name__ == "__main__":
    main()
