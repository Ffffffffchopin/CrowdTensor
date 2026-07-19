#!/usr/bin/env python3
"""Build private clean-wheel Kaggle CPU/GPU logical-node packages."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any


SCHEMA = "crowdtensor_community_kaggle_live_package_v1"
KAGGLE_RUNTIME_REQUIREMENTS = (
    "fastapi==0.136.1",
    "httpx==0.28.1",
    "uvicorn==0.42.0",
    "peft==0.19.1",
    "transformers==5.9.0",
    "safetensors==0.7.0",
    "accelerate==1.13.0",
)


def _hash(value: str | bytes) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]+", "-", value.lower())).strip("-")[:63]


def _kernel_source(
    *,
    role: str,
    backend: str,
    coordinator_url: str,
    miner_token: str,
    timeout_seconds: int,
) -> str:
    replacement = role == "stage0"
    limits = "[30, 0]" if replacement else "[0]"
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
working = pathlib.Path("/kaggle/working")
progress_path = working / "community_live_progress.json"
def progress(phase):
    progress_path.write_text(json.dumps({{"schema": "crowdtensor_community_kaggle_progress_v1", "phase": phase, "role": ROLE if "ROLE" in globals() else {role!r}, "public_artifact_safe": True}}, sort_keys=True) + "\\n")
progress("wheel_download_started")
request = urllib.request.Request(
    COORDINATOR.rstrip("/") + "/v1/community-live/wheel",
    headers={{"x-crowdtensor-miner-token": TOKEN}},
)
with urllib.request.urlopen(request, timeout=180) as response:
    payload = response.read()
    expected = str(response.headers.get("x-crowdtensor-wheel-sha256") or "")
    wheel_name = str(response.headers.get("x-crowdtensor-wheel-filename") or "")
if "sha256:" + hashlib.sha256(payload).hexdigest() != expected:
    raise RuntimeError("community_wheel_download_hash_invalid")
if not re.fullmatch(r"[A-Za-z0-9_.]+-[A-Za-z0-9_.!+]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+\\.whl", wheel_name):
    raise RuntimeError("community_wheel_filename_invalid")
wheel = working / wheel_name
wheel.write_bytes(payload)
progress("wheel_download_verified")
runtime_root = pathlib.Path("/kaggle/temp") if pathlib.Path("/kaggle/temp").is_dir() else pathlib.Path("/tmp")
install_root = runtime_root / "ct-community-site"
install_root.mkdir(parents=True, exist_ok=True)
python = pathlib.Path(sys.executable)
runtime_requirements = {list(KAGGLE_RUNTIME_REQUIREMENTS)!r}
subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--target", str(install_root), "--no-deps", str(wheel)], check=True)
progress("wheel_installed")
subprocess.run(
    [
        str(python), "-m", "pip", "install", "--disable-pip-version-check",
        "--target", str(install_root), "--upgrade", "--no-deps",
        *runtime_requirements,
    ],
    check=True,
)
progress("dependencies_ready")
env = dict(os.environ)
env["PYTHONPATH"] = str(install_root)
subprocess.run(
    [str(python), "-c", "import torch,transformers,peft,safetensors,fastapi,accelerate; print('dependencies-ready')"],
    env=env,
    check=True,
)
env["CROWDTENSOR_PRIVATE_COORDINATOR_URL"] = COORDINATOR
env["CROWDTENSOR_PRIVATE_MINER_TOKEN"] = TOKEN
env["CROWDTENSOR_INSTALL_SOURCE"] = "wheel"
reports = []
limits = {limits}
for index, limit in enumerate(limits):
    progress(f"worker_{{index}}_started")
    output = working / f"worker-{{ROLE}}-{{index}}.json"
    command = [
        str(python), "-m", "crowdtensor.community_live_worker",
        "--role", ROLE, "--backend", BACKEND,
        "--output", str(output), "--timeout-seconds", str(TIMEOUT),
        "--cache-dir", str(runtime_root / "ct-community-hf-cache"),
    ]
    if limit:
        command.extend(["--max-committed-step", str(limit)])
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(f"community_worker_process_failed:{{ROLE}}:{{index}}")
    reports.append(json.loads(output.read_text()))
    progress(f"worker_{{index}}_completed")
second_model_live = {{}}
if ROLE == "stage0":
    progress("dual_gpu_second_model_started")
    second_output = working / "smollm-dual-gpu-live"
    completed = subprocess.run(
        [
            str(python), "-m", "crowdtensor.community_smollm_runner",
            "--output-dir", str(second_output), "--steps", "2",
            "--timeout-seconds", "1200",
        ],
        env=env,
        check=False,
    )
    second_report = second_output / "smollm_two_stage_lora_live.json"
    if completed.returncode != 0 or not second_report.is_file():
        raise RuntimeError("community_smollm_dual_gpu_live_failed")
    second_model_live = json.loads(second_report.read_text())
    progress("dual_gpu_second_model_completed")
installed = subprocess.run(
    [
        str(python), "-c",
        "import crowdtensor,json,pathlib,os; print(json.dumps({{'version':crowdtensor.__version__,'under_install_root':str(pathlib.Path(crowdtensor.__file__).resolve()).startswith(str(pathlib.Path(os.environ['PYTHONPATH']).resolve()))}}))",
    ],
    env=env,
    text=True,
    capture_output=True,
    check=True,
)
installed_info = json.loads(installed.stdout.strip())
kernel_report = {{
    "schema": "crowdtensor_community_kaggle_live_kernel_v1",
    "ok": all(item.get("ok") is True for item in reports),
    "kernel_role": ROLE,
    "backend": BACKEND,
    "node_scope": "Kaggle logical multi-node",
    "worker_reports": reports,
    "worker_process_count": len(reports),
    "worker_replacement_verified": bool(
        ROLE == "stage0" and len(reports) == 2
        and reports[0].get("worker_id_hash") != reports[1].get("worker_id_hash")
        and reports[1].get("checkpoint_restored") is True
        and int(reports[1].get("restored_checkpoint_step") or 0) >= 30
    ),
    "second_model_live": second_model_live,
    "second_model_dual_gpu_verified": bool(
        ROLE != "stage0" or (
            second_model_live.get("ok") is True
            and second_model_live.get("devices") == ["cuda", "cuda"]
            and second_model_live.get("logical_stage_count") == 2
        )
    ),
    "wheel_download_hash_verified": True,
    "wheel_installed_in_fresh_environment": True,
    "fresh_install_kind": "pip_target",
    "installed_package_version": str(installed_info.get("version") or ""),
    "installed_package_under_install_root": installed_info.get("under_install_root") is True,
    "model_stack_import_verified": True,
    "runtime_requirements_exact_pins_verified": all("==" in item for item in runtime_requirements),
    "workspace_import_used": False,
    "coordinator_url_public": False,
    "credential_values_public": False,
    "private_paths_public": False,
    "public_artifact_safe": True,
}}
(working / "community_live_kernel.json").write_text(json.dumps(kernel_report, indent=2, sort_keys=True) + "\\n")
progress("kernel_report_written")
print(json.dumps({{"ok": kernel_report["ok"], "role": ROLE, "worker_count": len(reports)}}))
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
    for role, backend in (
        ("stage0", "cuda"),
        ("stage1", "cpu"),
    ):
        slug = _slug(f"ct-community-{role}-{unique_suffix}")
        directory = root / role
        if directory.exists():
            shutil.rmtree(directory)
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
                "wheel_clean_install": True,
                "workspace_source_uploaded": False,
            }
        )
    report = {
        "schema": SCHEMA,
        "ok": True,
        "node_scope": "Kaggle logical multi-node",
        "logical_node_count": 2,
        "providers": ["kaggle_cpu", "kaggle_cuda"],
        "packages": records,
        "wheel_served_by_authenticated_coordinator": True,
        "fresh_install_root_per_kernel": True,
        "workspace_import_used": False,
        "coordinator_url_public": False,
        "credential_values_public": False,
        "private_package_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = _hash(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    )
    return {"report": report, "private_root": root}
