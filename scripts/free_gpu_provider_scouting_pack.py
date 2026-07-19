#!/usr/bin/env python3
"""Build a public-safe free GPU provider scouting report."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


SCHEMA = "free_gpu_provider_scouting_v1"


def run_command(command: list[str], *, timeout: float = 60.0) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {
            "ok": False,
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error_type": type(exc).__name__,
        }
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_tail": (completed.stdout or "")[-1000:],
        "stderr_tail": (completed.stderr or "")[-1000:],
    }


def sha16(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def command_present(name: str) -> bool:
    return shutil.which(name) is not None


def package_present(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def hf_status() -> dict[str, Any]:
    step = run_command(["hf", "auth", "whoami"], timeout=30) if command_present("hf") else {"ok": False}
    return {
        "cli_present": command_present("hf"),
        "logged_in": step.get("ok") is True,
        "error_type_public": "" if step.get("ok") else "not_logged_in_or_cli_unavailable",
        "credentials_public": False,
    }


def gcloud_status() -> dict[str, Any]:
    active = run_command(["gcloud", "config", "get-value", "account"], timeout=30) if command_present("gcloud") else {"ok": False}
    project = run_command(["gcloud", "config", "get-value", "project"], timeout=30) if command_present("gcloud") else {"ok": False}
    account = str(active.get("stdout_tail") or "").strip()
    project_value = str(project.get("stdout_tail") or "").strip()
    projects = run_command(["gcloud", "projects", "list", "--format=json"], timeout=60) if command_present("gcloud") else {"ok": False}
    project_count = 0
    if projects.get("ok"):
        try:
            loaded = json.loads(str(projects.get("stdout_tail") or "[]"))
        except json.JSONDecodeError:
            loaded = []
        if isinstance(loaded, list):
            project_count = len(loaded)
    service_probe = probe_gcloud_services() if project_count else {
        "attempted": False,
        "project_count": 0,
        "compute_enabled_project_count": 0,
        "tpu_enabled_project_count": 0,
    }
    return {
        "cli_present": command_present("gcloud"),
        "active_account_present": bool(account and account != "(unset)"),
        "active_account_hash": sha16(account) if account and account != "(unset)" else "",
        "default_project_set": bool(project_value and project_value != "(unset)"),
        "visible_project_count": project_count,
        "service_probe": service_probe,
        "credentials_public": False,
    }


def probe_gcloud_services() -> dict[str, Any]:
    projects = run_command(["gcloud", "projects", "list", "--format=json"], timeout=60)
    if not projects.get("ok"):
        return {
            "attempted": True,
            "ok": False,
            "project_count": 0,
            "compute_enabled_project_count": 0,
            "tpu_enabled_project_count": 0,
            "error_type": "projects_list_failed",
        }
    try:
        loaded = json.loads(str(projects.get("stdout_tail") or "[]"))
    except json.JSONDecodeError:
        loaded = []
    rows = loaded if isinstance(loaded, list) else []
    project_summaries: list[dict[str, Any]] = []
    compute_enabled = 0
    tpu_enabled = 0
    for item in rows[:10]:
        project = str(item.get("projectId") or "")
        services: dict[str, bool] = {}
        for service in ["compute.googleapis.com", "tpu.googleapis.com", "serviceusage.googleapis.com"]:
            step = run_command(
                [
                    "gcloud",
                    "services",
                    "list",
                    "--enabled",
                    "--project",
                    project,
                    "--filter",
                    f"config.name:{service}",
                    "--format=json",
                ],
                timeout=60,
            )
            enabled = False
            if step.get("ok"):
                try:
                    enabled = bool(json.loads(str(step.get("stdout_tail") or "[]")))
                except json.JSONDecodeError:
                    enabled = False
            services[service] = enabled
        if services.get("compute.googleapis.com"):
            compute_enabled += 1
        if services.get("tpu.googleapis.com"):
            tpu_enabled += 1
        project_summaries.append({
            "project_hash": sha16(project),
            "lifecycle_state": str(item.get("lifecycleState") or ""),
            "compute_api_enabled": services.get("compute.googleapis.com") is True,
            "tpu_api_enabled": services.get("tpu.googleapis.com") is True,
            "serviceusage_api_enabled": services.get("serviceusage.googleapis.com") is True,
        })
    return {
        "attempted": True,
        "ok": True,
        "project_count": len(rows),
        "compute_enabled_project_count": compute_enabled,
        "tpu_enabled_project_count": tpu_enabled,
        "projects": project_summaries,
        "project_ids_public": False,
    }


def aws_status() -> dict[str, Any]:
    has_credentials = False
    try:
        import botocore.session

        has_credentials = botocore.session.get_session().get_credentials() is not None
    except Exception:
        has_credentials = False
    return {
        "cli_present": command_present("aws"),
        "boto3_present": package_present("boto3"),
        "credentials_present": has_credentials,
        "credentials_public": False,
    }


def local_status() -> dict[str, Any]:
    return {
        "commands": {
            "lightning": command_present("lightning"),
            "modal": command_present("modal"),
            "aws": command_present("aws"),
            "paperspace": command_present("paperspace"),
            "gradient": command_present("gradient"),
            "hf": command_present("hf"),
            "gcloud": command_present("gcloud"),
        },
        "python_packages": {
            "modal": package_present("modal"),
            "lightning": package_present("lightning"),
            "boto3": package_present("boto3"),
            "huggingface_hub": package_present("huggingface_hub"),
        },
        "huggingface": hf_status(),
        "gcloud": gcloud_status(),
        "aws": aws_status(),
    }


def providers(status: dict[str, Any], latest_lightning_api_probe: dict[str, Any]) -> list[dict[str, Any]]:
    lightning_api_ready = latest_lightning_api_probe.get("api_auth_verified") is True
    lightning_zero_balance_gpu_safe = latest_lightning_api_probe.get("safe_to_attempt_zero_balance_gpu_start") is True
    return [
        {
            "provider": "lightning_ai_studio",
            "official_sources": [
                "https://lightning.ai/pricing/",
                "https://lightning.ai/docs/platform/overview/setups/academia/students",
                "https://lightning.ai/docs/platform/overview/faq/billing",
            ],
            "free_or_trial_signal": "15 free Lightning credits/month; docs also describe free Studio and free GPU-hour allocation depending on plan/status.",
            "automation_surface": "web Studio plus Python API/SDK. The local isolated SDK probe authenticates, but the browser cookie does not create a logged-in page session.",
            "local_ready_now": bool(lightning_zero_balance_gpu_safe),
            "worker_fit": "blocked_for_current_zero_balance_account",
            "deepseek_worker_fit": "technically good for a CUDA RPC worker if credits/free Studio are available; current account cannot safely start one.",
            "needs_user": ["Lightning credits/free Studio eligibility or another Lightning account with can_start_free_cloud_space=true"],
            "next_probe": "Only run a bounded start+nvidia-smi probe after API evidence shows positive balance or can_start_free_cloud_space=true.",
            "rank": 4 if lightning_api_ready else 1,
        },
        {
            "provider": "modal",
            "official_sources": ["https://modal.com/pricing", "https://modal.com/signup"],
            "free_or_trial_signal": "$30/month free credit on Starter; pricing page lists GPU concurrency.",
            "automation_surface": "Python SDK/serverless functions; local modal package/CLI is not installed.",
            "local_ready_now": bool(status["commands"]["modal"] or status["python_packages"]["modal"]),
            "worker_fit": "high_if_login_available",
            "deepseek_worker_fit": "best engineering fit for scriptable CUDA worker, but credits may be consumed quickly by long-lived RPC workers.",
            "needs_user": ["Modal account login/token"],
            "next_probe": "Install modal client, authenticate, run a short GPU function with nvidia-smi, then test persistent tunnel feasibility.",
            "rank": 2,
        },
        {
            "provider": "paperspace_gradient",
            "official_sources": [
                "https://www.paperspace.com/pricing",
                "https://www.paperspace.com/notebooks",
                "https://docs.digitalocean.com/products/paperspace/pricing/",
            ],
            "free_or_trial_signal": "Pricing/notebooks pages advertise a FREE GPU plan for notebooks; docs say free machines are limited and notebook-only.",
            "automation_surface": "mostly web notebook; local paperspace/gradient CLI is not installed.",
            "local_ready_now": bool(status["commands"]["paperspace"] or status["commands"]["gradient"]),
            "worker_fit": "medium_if_login_available",
            "deepseek_worker_fit": "promising as a Colab-like CUDA worker if free GPU is assignable and shell/network tunneling are available.",
            "needs_user": ["Paperspace/DigitalOcean login cookie or API key"],
            "next_probe": "After login, create a free GPU notebook, run nvidia-smi, and test bore reverse tunnel.",
            "rank": 3,
        },
        {
            "provider": "aws_sagemaker_studio_lab",
            "official_sources": ["https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html"],
            "free_or_trial_signal": "AWS describes Studio Lab as a free JupyterLab service; GPU session availability is capacity-limited.",
            "automation_surface": "web JupyterLab; local AWS credentials are not present.",
            "local_ready_now": bool(status["aws"]["credentials_present"]),
            "worker_fit": "medium_if_account_available",
            "deepseek_worker_fit": "possible notebook-style worker, but historically session-limited and less directly automatable than Modal.",
            "needs_user": ["Studio Lab account/login cookie"],
            "next_probe": "After login, start GPU runtime, run nvidia-smi, and verify outbound tunnel support.",
            "rank": 4,
        },
        {
            "provider": "huggingface_zerogpu",
            "official_sources": [
                "https://huggingface.co/docs/hub/en/spaces-zerogpu",
                "https://huggingface.co/docs/hub/en/spaces-gpus",
            ],
            "free_or_trial_signal": "ZeroGPU offers free GPU access for Spaces with daily quota; hosting ZeroGPU on personal accounts requires PRO.",
            "automation_surface": "HF Spaces/Gradio function calls; local HF CLI is present but not logged in.",
            "local_ready_now": bool(status["huggingface"]["logged_in"]),
            "worker_fit": "low_for_persistent_rpc",
            "deepseek_worker_fit": "not a good persistent llama.cpp RPC worker; useful for public demo functions or short probes.",
            "needs_user": ["HF login token; PRO if hosting own ZeroGPU Space"],
            "next_probe": "Login HF and test whether a Docker/Gradio Space can execute a short CUDA call within quota.",
            "rank": 5,
        },
        {
            "provider": "google_cloud_gpu_or_tpu_trial",
            "official_sources": [
                "https://cloud.google.com/free",
                "https://cloud.google.com/tpu/pricing",
                "https://sites.research.google/trc/",
            ],
            "free_or_trial_signal": "Google Cloud can provide free trial credits; TPU Research Cloud is application/grant based.",
            "automation_surface": "gcloud is installed and an account is active, but no default project is configured.",
            "local_ready_now": bool(status["gcloud"]["active_account_present"] and status["gcloud"]["default_project_set"]),
            "worker_fit": "high_if_quota_and_credits_available",
            "deepseek_worker_fit": "technically strong and stable if quota exists, but likely requires billing/trial credits and GPU/TPU quota requests.",
            "needs_user": ["select project", "confirm billing/free credits", "approve GPU or TPU quota usage"],
            "next_probe": "Set a project, inspect accelerator quotas, then run a tiny GPU/TPU VM test if free credits are available.",
            "rank": 6,
        },
    ]


def build_report() -> dict[str, Any]:
    status = local_status()
    lightning_probe = load_optional_json(Path("dist/lightning-gpu-provider-probe-20260630-r6-post-token-cookie-readonly/lightning_gpu_provider_probe.json"))
    lightning_api_probe = load_optional_json(Path("dist/lightning-api-gpu-provider-probe-20260630-r1-token-readonly/lightning_api_gpu_provider_probe.json"))
    candidates = providers(status, lightning_api_probe)
    if lightning_probe:
        for item in candidates:
            if item.get("provider") == "lightning_ai_studio":
                item["latest_probe"] = {
                    "schema": lightning_probe.get("schema"),
                    "path": "dist/lightning-gpu-provider-probe-20260630-r6-post-token-cookie-readonly/lightning_gpu_provider_probe.json",
                    "lightning_login_verified": lightning_probe.get("lightning_login_verified") is True,
                    "safe_to_attempt_free_gpu_start": lightning_probe.get("safe_to_attempt_free_gpu_start") is True,
                    "blockers": lightning_probe.get("blockers") if isinstance(lightning_probe.get("blockers"), list) else [],
                    "public_artifact_safe": lightning_probe.get("public_artifact_safe") is True,
                }
    if lightning_api_probe:
        for item in candidates:
            if item.get("provider") == "lightning_ai_studio":
                item["latest_api_probe"] = {
                    "schema": lightning_api_probe.get("schema"),
                    "path": "dist/lightning-api-gpu-provider-probe-20260630-r1-token-readonly/lightning_api_gpu_provider_probe.json",
                    "api_auth_verified": lightning_api_probe.get("api_auth_verified") is True,
                    "safe_to_attempt_zero_balance_gpu_start": lightning_api_probe.get("safe_to_attempt_zero_balance_gpu_start") is True,
                    "user_balance": (lightning_api_probe.get("user_balance") or {}).get("balance"),
                    "cheapest_enabled_gpu_cost_per_hour": (lightning_api_probe.get("default_accelerators") or {}).get("cheapest_enabled_gpu_cost_per_hour"),
                    "gpu_accelerator_count": (lightning_api_probe.get("default_accelerators") or {}).get("gpu_accelerator_count"),
                    "can_start_free_cloud_space": (lightning_api_probe.get("cloud_space_instances") or {}).get("can_start_free_cloud_space") is True,
                    "blockers": lightning_api_probe.get("blockers") if isinstance(lightning_api_probe.get("blockers"), list) else [],
                    "public_artifact_safe": lightning_api_probe.get("public_artifact_safe") is True,
                }
                item["local_ready_now"] = item["latest_api_probe"]["safe_to_attempt_zero_balance_gpu_start"]
    recommended = [
        item["provider"]
        for item in sorted(
            candidates,
            key=lambda x: (not bool(x.get("local_ready_now")), int(x["rank"])),
        )[:3]
    ]
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "objective": "Find a stable-enough free or free-credit GPU provider for CrowdTensor heterogeneous accelerator workers.",
        "local_status": status,
        "providers": candidates,
        "recommended_probe_order": recommended,
        "best_current_bets": {
            "first": "modal",
            "second": "paperspace_gradient",
            "third": "google_cloud_gpu_or_tpu_trial",
            "blocked_but_validated": "lightning_ai_studio",
            "reason": "Lightning API auth works but current account has 0 balance and cannot start a free CloudSpace. Modal/Paperspace/GCP remain the next plausible channels once credentials or free credits are supplied.",
        },
        "not_ready_without_user_login_or_credits": True,
        "public_artifact_safe": True,
        "credentials_public": False,
        "private_runtime_state_public": False,
    }


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    path = output_dir / "free_gpu_provider_scouting.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
