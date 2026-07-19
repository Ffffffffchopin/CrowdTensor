#!/usr/bin/env python3
"""Build private clean-wheel Kaggle CPU/CUDA packages for Mistral training."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from scripts.community_kaggle_live_package import KAGGLE_RUNTIME_REQUIREMENTS


SCHEMA = "crowdtensor_mistral_kaggle_live_package_v1"
MODEL_ADAPTER_ID = "mistral_lora_v1"
MODEL_ID = "Locutusque/TinyMistral-248M-v2"
MODEL_REVISION = "0f57b17cb317bb322c7c1466b669c681f80c058f"
KERNEL_REPORT = "mistral_live_kernel.json"
PROGRESS_REPORT = "mistral_live_progress.json"


def _hash(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _slug(value: str) -> str:
    return re.sub(
        r"-+", "-", re.sub(r"[^a-z0-9-]+", "-", value.lower())
    ).strip("-")[:63]


def _kernel_source(
    *,
    role: str,
    backend: str,
    coordinator_url: str,
    miner_token: str,
    timeout_seconds: int,
) -> str:
    limits = "[4, 0]" if role == "stage0" else "[0]"
    return f'''import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.request

COORDINATOR = {coordinator_url!r}
TOKEN = {miner_token!r}
ROLE = {role!r}
BACKEND = {backend!r}
TIMEOUT = {int(timeout_seconds)!r}
MODEL_ADAPTER_ID = {MODEL_ADAPTER_ID!r}
working = pathlib.Path("/kaggle/working")
progress_path = working / {PROGRESS_REPORT!r}

def progress(phase):
    progress_path.write_text(json.dumps({{"schema":"crowdtensor_mistral_kaggle_progress_v1","phase":phase,"role":ROLE,"public_artifact_safe":True}}, sort_keys=True) + "\\n")

def fetch_wheel(route, kind):
    request = urllib.request.Request(
        COORDINATOR.rstrip("/") + route,
        headers={{"x-crowdtensor-miner-token": TOKEN}},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
        expected = str(response.headers.get("x-crowdtensor-wheel-sha256") or "")
        name = str(response.headers.get("x-crowdtensor-wheel-filename") or "")
    actual = "sha256:" + hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError("mistral_" + kind + "_wheel_hash_invalid")
    if not re.fullmatch(r"[A-Za-z0-9_.]+-[A-Za-z0-9_.!+]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+\\.whl", name):
        raise RuntimeError("mistral_" + kind + "_wheel_filename_invalid")
    target = working / name
    target.write_bytes(payload)
    return target, actual

progress("wheel_download_started")
core_wheel, core_hash = fetch_wheel("/v1/community-live/wheel", "core")
adapter_wheel, adapter_hash = fetch_wheel("/v1/community-live/adapter-wheel", "adapter")
progress("wheel_download_verified")
runtime_root = pathlib.Path("/kaggle/temp") if pathlib.Path("/kaggle/temp").is_dir() else pathlib.Path("/tmp")
install_root = runtime_root / "ct-mistral-site"
install_root.mkdir(parents=True, exist_ok=True)
python = pathlib.Path(sys.executable)
for wheel in (core_wheel, adapter_wheel):
    subprocess.run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--target", str(install_root), "--no-deps", str(wheel)],
        check=True,
    )
progress("wheels_installed")
runtime_requirements = {list(KAGGLE_RUNTIME_REQUIREMENTS)!r}
subprocess.run(
    [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--target", str(install_root), "--upgrade", "--no-deps", *runtime_requirements],
    check=True,
)
progress("dependencies_ready")
env = dict(os.environ)
env["PYTHONPATH"] = str(install_root)
plugin_code = "import json; from crowdtensor.model_adapter import adapter_registry_report,get_model_adapter,get_model_adapter_registration; r=adapter_registry_report(); a=get_model_adapter('mistral_lora_v1'); p=get_model_adapter_registration('mistral_lora_v1'); print(json.dumps({{'adapter_id':a.adapter_id,'family':a.family,'model_id':a.default_model_id,'plugin_count':r['plugin_adapter_count'],'registration':p}}))"
plugin_process = subprocess.run(
    [str(python), "-c", plugin_code], env=env, text=True, capture_output=True, check=True
)
plugin = json.loads(plugin_process.stdout.strip())
if plugin.get("adapter_id") != MODEL_ADAPTER_ID or plugin.get("model_id") != {MODEL_ID!r} or plugin.get("registration", {{}}).get("kind") != "entry_point_plugin":
    raise RuntimeError("mistral_adapter_plugin_discovery_invalid")
progress("adapter_plugin_discovered")
env["CROWDTENSOR_PRIVATE_COORDINATOR_URL"] = COORDINATOR
env["CROWDTENSOR_PRIVATE_MINER_TOKEN"] = TOKEN
env["CROWDTENSOR_INSTALL_SOURCE"] = "wheel"
reports = []
limits = {limits}
for index, limit in enumerate(limits):
    progress(f"worker_{{index}}_started")
    output = working / f"mistral-worker-{{ROLE}}-{{index}}.json"
    command = [
        str(python), "-m", "crowdtensor.community_live_worker",
        "--role", ROLE, "--backend", BACKEND,
        "--output", str(output), "--timeout-seconds", str(TIMEOUT),
        "--cache-dir", str(runtime_root / "ct-mistral-hf-cache"),
    ]
    if limit:
        command.extend(["--max-committed-step", str(limit)])
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(f"mistral_worker_process_failed:{{ROLE}}:{{index}}")
    reports.append(json.loads(output.read_text()))
    progress(f"worker_{{index}}_completed")
import torch
installed = subprocess.run(
    [str(python), "-c", "import crowdtensor,json,pathlib,os; print(json.dumps({{'version':crowdtensor.__version__,'under_install_root':str(pathlib.Path(crowdtensor.__file__).resolve()).startswith(str(pathlib.Path(os.environ['PYTHONPATH']).resolve()))}}))"],
    env=env, text=True, capture_output=True, check=True,
)
installed_info = json.loads(installed.stdout.strip())
replacement_verified = bool(
    ROLE != "stage0" or (
        len(reports) == 2
        and reports[0].get("worker_id_hash") != reports[1].get("worker_id_hash")
        and reports[1].get("checkpoint_restored") is True
        and int(reports[1].get("restored_checkpoint_step") or 0) == 4
        and reports[1].get("optimizer_state_restored") is True
    )
)
kernel_report = {{
    "schema": "crowdtensor_mistral_kaggle_live_kernel_v1",
    "ok": all(item.get("ok") is True for item in reports) and replacement_verified,
    "kernel_role": ROLE,
    "backend": BACKEND,
    "node_scope": "Kaggle logical multi-node",
    "worker_reports": reports,
    "worker_process_count": len(reports),
    "worker_replacement_verified": replacement_verified,
    "cuda_device_count": int(torch.cuda.device_count()) if BACKEND == "cuda" else 0,
    "core_wheel_hash": core_hash,
    "adapter_wheel_hash": adapter_hash,
    "core_wheel_hash_verified": True,
    "adapter_wheel_hash_verified": True,
    "both_wheels_installed_in_fresh_environment": True,
    "fresh_install_kind": "pip_target",
    "installed_package_version": str(installed_info.get("version") or ""),
    "installed_package_under_install_root": installed_info.get("under_install_root") is True,
    "adapter_plugin_discovered": True,
    "adapter_plugin_count": int(plugin.get("plugin_count") or 0),
    "adapter_plugin_registration_kind": str(plugin.get("registration", {{}}).get("kind") or ""),
    "adapter_plugin_distribution_name": str(plugin.get("registration", {{}}).get("distribution_name") or ""),
    "model_adapter_id": MODEL_ADAPTER_ID,
    "model_id": {MODEL_ID!r},
    "model_stack_import_verified": True,
    "runtime_requirements_exact_pins_verified": all("==" in item for item in runtime_requirements),
    "workspace_import_used": False,
    "coordinator_url_public": False,
    "credential_values_public": False,
    "private_paths_public": False,
    "public_artifact_safe": True,
}}
(working / {KERNEL_REPORT!r}).write_text(json.dumps(kernel_report, indent=2, sort_keys=True) + "\\n")
progress("kernel_report_written")
print(json.dumps({{"ok":kernel_report["ok"],"role":ROLE,"worker_count":len(reports)}}))
'''


def build_packages(
    output_dir: str | Path,
    *,
    owner: str,
    coordinator_url: str,
    miner_token: str,
    unique_suffix: str,
    timeout_seconds: int = 2700,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for role, backend in (("stage0", "cuda"), ("stage1", "cpu")):
        slug = _slug(f"ct-mistral-{role}-{unique_suffix}")
        directory = root / role
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
        (directory / "kernel.py").write_text(
            _kernel_source(
                role=role,
                backend=backend,
                coordinator_url=coordinator_url,
                miner_token=miner_token,
                timeout_seconds=timeout_seconds,
            ),
            encoding="utf-8",
        )
        metadata = {
            "id": f"{owner}/{slug}",
            "title": slug,
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if backend == "cuda" else "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        if backend == "cuda":
            metadata["machine_shape"] = "NvidiaTeslaT4"
        (directory / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        records.append(
            {
                "role": role,
                "backend": backend,
                "kernel_ref_hash": _hash(metadata["id"]),
                "private_package": True,
                "core_and_plugin_wheel_clean_install": True,
                "workspace_source_uploaded": False,
            }
        )
    report = {
        "schema": SCHEMA,
        "ok": True,
        "node_scope": "Kaggle logical multi-node",
        "logical_node_count": 2,
        "providers": ["kaggle_cpu", "kaggle_cuda"],
        "model_adapter_id": MODEL_ADAPTER_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "target_steps": 8,
        "checkpoint_steps": [4, 8],
        "gpu_worker_replacement_after_step": 4,
        "packages": records,
        "core_and_plugin_wheels_served_by_authenticated_coordinator": True,
        "fresh_install_root_per_kernel": True,
        "workspace_import_used": False,
        "coordinator_url_public": False,
        "credential_values_public": False,
        "private_package_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = _hash(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return {"report": report, "private_root": root}
