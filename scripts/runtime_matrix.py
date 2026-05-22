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
            next_command="python3 scripts/inference_session_demo.py --port 8904 --request-count 4 --json"
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
        "workloads": workloads,
        "summary": {
            "available": len(available),
            "optional_missing": len(optional_missing),
            "blocked": len(blocked),
            "available_workloads": available,
            "optional_missing_workloads": optional_missing,
            "blocked_workloads": blocked,
        },
        "recommended_next_commands": [
            "python3 scripts/inference_session_demo.py --port 8904 --request-count 4 --json",
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
