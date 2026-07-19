#!/usr/bin/env python3
"""Build the private Kaggle TPU v5e-8 worker used by the Training TPU Beta."""

from __future__ import annotations

import argparse
import base64
import json
import re
import secrets
import shutil
import textwrap
from pathlib import Path
from typing import Any

from scripts.training_heterogeneous_beta_kaggle_package import (
    _bundle_archive_b64,
    _safe_slug,
)


PACKAGE_SCHEMA = "crowdtensor_heterogeneous_training_tpu_beta_kaggle_package_v1"
KERNEL_SCHEMA = "crowdtensor_heterogeneous_training_tpu_beta_kaggle_kernel_v1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def render_kernel(
    *,
    bundle_archive_b64: str,
    private_configuration_b64: str,
    old_identity_nonce: str,
    replacement_identity_nonce: str,
    wait_timeout_seconds: float,
    operation_timeout_seconds: float,
    transport_optimization_after_step: int = -1,
    replacement_after_steps: int = 3,
) -> str:
    if int(replacement_after_steps) < 1:
        raise ValueError("heterogeneous_tpu_replacement_step_invalid")
    source = f'''from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


WORKING = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
REPORT_PATH = WORKING / "training_heterogeneous_tpu_beta_kernel.json"
PRIVATE_ROOT = WORKING / ".crowdtensor-heterogeneous-training-tpu"
BUNDLE_ROOT = PRIVATE_ROOT / "bundle"
PRIVATE_CONFIGURATION = PRIVATE_ROOT / "private_configuration.json"
BUNDLE_ARCHIVE_B64 = {bundle_archive_b64!r}
PRIVATE_CONFIGURATION_B64 = {private_configuration_b64!r}
OLD_IDENTITY_NONCE = {old_identity_nonce!r}
REPLACEMENT_IDENTITY_NONCE = {replacement_identity_nonce!r}
WAIT_TIMEOUT_SECONDS = {float(wait_timeout_seconds)!r}
OPERATION_TIMEOUT_SECONDS = {float(operation_timeout_seconds)!r}
TRANSPORT_OPTIMIZATION_AFTER_STEP = {int(transport_optimization_after_step)!r}
REPLACEMENT_AFTER_STEPS = {int(replacement_after_steps)!r}
KERNEL_SCHEMA = {KERNEL_SCHEMA!r}


def stable_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_report(value):
    REPORT_PATH.write_text(json.dumps(value, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def safe_blocker(exc):
    text = str(exc)
    if text.startswith(("heterogeneous_", "elastic_", "qwen15b_")):
        return re.sub(r"[^a-zA-Z0-9:_-]", "_", text[:180])
    return f"heterogeneous_tpu_kaggle_kernel_failed:{{type(exc).__name__}}"


def ensure_dependencies():
    required = {{"transformers": "5.9.0", "peft": "0.19.1", "safetensors": "0.7.0"}}
    installed = {{}}
    for name in required:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = ""
    if any(installed[name] != version for name, version in required.items()):
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
                "transformers==5.9.0", "peft==0.19.1", "safetensors==0.7.0",
                "accelerate>=1.2,<2", "fastapi>=0.115,<1", "httpx>=0.27,<1", "pydantic>=2.10,<3",
                "psutil>=5.9,<8",
            ],
            check=True,
            timeout=900,
        )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, jax; "
                "devices=[item for item in jax.devices() "
                "if str(getattr(item, 'platform', '')).lower() == 'tpu']; "
                "print(json.dumps({{'jax': str(getattr(jax, '__version__', '')), "
                "'jax_tpu_device_count': len(devices)}}))"
            ),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    payload = {{}}
    for line in reversed((probe.stdout or "").splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if probe.returncode != 0 or int(payload.get("jax_tpu_device_count") or 0) != 8:
        raise RuntimeError("heterogeneous_jax_tpu_device_count_invalid")
    return {{
        **{{name: importlib.metadata.version(name) for name in required}},
        "jax": str(payload.get("jax") or ""),
        "jax_tpu_device_count": int(payload["jax_tpu_device_count"]),
        "accelerator_type": "TPU v5e",
        "tpu_probe_process_released": True,
    }}


def request_status():
    private = json.loads(PRIVATE_CONFIGURATION.read_text(encoding="utf-8"))
    request = urllib.request.Request(
        private["coordinator_url"].rstrip("/") + "/elastic-training/status",
        headers={{
            "User-Agent": "crowdtensor-heterogeneous-tpu-kaggle/1",
            "x-crowdtensor-miner-token": private["coordinator_token"],
        }},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=60.0) as response:
        value = json.loads(response.read().decode("utf-8"))
    return value if isinstance(value, dict) else {{}}


def launch(label, identity_nonce, max_steps):
    output = WORKING / f"worker-{{label}}.json"
    log_path = PRIVATE_ROOT / f"worker-{{label}}.log"
    command = [
        sys.executable,
        str(BUNDLE_ROOT / "scripts" / "training_heterogeneous_beta_worker_entry.py"),
        "--private-configuration", str(PRIVATE_CONFIGURATION),
        "--output", str(output),
        "--private-root", str(PRIVATE_ROOT / "shared-worker-state"),
        "--deployment-role", label,
        "--identity-nonce", identity_nonce,
        "--device-policy", "jax_tpu",
        "--max-steps", str(int(max_steps)),
        "--wait-timeout", str(WAIT_TIMEOUT_SECONDS),
        "--operation-timeout", str(OPERATION_TIMEOUT_SECONDS),
        "--session-retries", "1",
        "--transport-optimization-after-step", str(TRANSPORT_OPTIMIZATION_AFTER_STEP),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True
    )
    process._ct_log = log
    return {{"label": label, "process": process, "output": output}}


def wait_worker(record, deadline):
    process = record["process"]
    try:
        returncode = process.wait(timeout=max(1.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30.0)
        returncode = process.returncode
    finally:
        process._ct_log.close()
    report = {{}}
    if record["output"].is_file():
        try:
            report = json.loads(record["output"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {{}}
    return {{"label": record["label"], "returncode": returncode, "report": report}}


def wait_for_pause(deadline):
    observations = []
    while time.monotonic() < deadline:
        try:
            status = request_status()
            observation = {{
                "committed_step": int(status.get("committed_step") or 0),
                "runtime_state": str(status.get("runtime_state") or ""),
                "missing_stage_ids": list(status.get("missing_stage_ids") or []),
                "placement_generation": int(status.get("placement_generation") or 0),
            }}
            observations.append(observation)
            if observation["committed_step"] >= REPLACEMENT_AFTER_STEPS and (
                2 in observation["missing_stage_ids"]
                or observation["runtime_state"] == "paused_waiting_for_miners"
            ):
                return {{"verified": True, "observations": observations[-20:]}}
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(2.0)
    return {{"verified": False, "observations": observations[-20:]}}


def stop_lingering(records):
    stopped = True
    for record in records:
        process = record["process"]
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30.0)
        stopped = stopped and process.poll() is not None
        try:
            process._ct_log.close()
        except Exception:
            pass
    return stopped


def main():
    started = time.time()
    runtime_nonce_hash = stable_hash([os.getpid(), time.time_ns(), "same-kernel-tpu-runtime"])
    report = {{
        "schema": KERNEL_SCHEMA,
        "ok": False,
        "kernel_role": "tpu",
        "worker_results": [],
        "pause_observation": {{}},
        "logical_tpu_restart_count": 0,
        "same_tpu_kernel_runtime_hash": runtime_nonce_hash,
        "dependency_versions": {{}},
        "private_runtime_removed": False,
        "blockers": [],
        "credential_values_public": False,
        "coordinator_url_public": False,
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }}
    records = []
    deadline = time.monotonic() + WAIT_TIMEOUT_SECONDS
    try:
        PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
        BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_ARCHIVE_B64)), "r") as archive:
            archive.extractall(BUNDLE_ROOT)
        PRIVATE_CONFIGURATION.write_bytes(base64.b64decode(PRIVATE_CONFIGURATION_B64))
        PRIVATE_CONFIGURATION.chmod(0o600)
        report["dependency_versions"] = ensure_dependencies()
        old = launch("tpu_old", OLD_IDENTITY_NONCE, REPLACEMENT_AFTER_STEPS)
        records.append(old)
        report["worker_results"].append(wait_worker(old, deadline))
        records.remove(old)
        report["pause_observation"] = wait_for_pause(
            min(deadline, time.monotonic() + 300.0)
        )
        replacement = launch("tpu_replacement", REPLACEMENT_IDENTITY_NONCE, 0)
        records.append(replacement)
        report["logical_tpu_restart_count"] = 1
        report["worker_results"].append(wait_worker(replacement, deadline))
        records.remove(replacement)
        workers = [item.get("report") or {{}} for item in report["worker_results"]]
        report["ok"] = bool(
            len(workers) == 2
            and all(item.get("ok") is True for item in workers)
            and all(item.get("returncode") == 0 for item in report["worker_results"])
            and report["pause_observation"].get("verified") is True
            and int((workers[0].get("capability") or {{}}).get("tpu_groups", [{{}}])[0].get("device_count") or 0) == 8
            and int(workers[0].get("steps_completed") or 0) == REPLACEMENT_AFTER_STEPS
            and int(workers[1].get("central_checkpoint_restore_count") or 0) >= 1
        )
        if not report["ok"]:
            report["blockers"].append("heterogeneous_tpu_kaggle_worker_acceptance_incomplete")
    except BaseException as exc:
        report["blockers"].append(safe_blocker(exc))
    finally:
        report["all_worker_processes_stopped"] = stop_lingering(records)
        shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)
        report["private_runtime_removed"] = not PRIVATE_ROOT.exists()
        report["elapsed_seconds"] = round(time.time() - started, 3)
        report["blockers"] = sorted(set(report["blockers"]))
        report["ok"] = bool(
            report["ok"]
            and report["all_worker_processes_stopped"]
            and report["private_runtime_removed"]
        )
        write_report(report)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
'''
    return textwrap.dedent(source)


def build_package(
    output_dir: str | Path,
    *,
    owner: str,
    slug: str,
    coordinator_url: str,
    coordinator_token: str,
    hf_token: str = "",
    wait_timeout_seconds: float = 21600.0,
    operation_timeout_seconds: float = 3600.0,
    old_identity_nonce: str | None = None,
    replacement_identity_nonce: str | None = None,
    transport_optimization_after_step: int = -1,
    replacement_after_steps: int = 3,
) -> dict[str, Any]:
    if not str(coordinator_url).strip() or not str(coordinator_token):
        raise ValueError("heterogeneous_tpu_kaggle_package_private_inputs_required")
    output = Path(output_dir).resolve()
    shutil.rmtree(output, ignore_errors=True)
    package = output / "private-kernel"
    package.mkdir(parents=True, exist_ok=True)
    safe_owner = _safe_slug(owner)
    safe_kernel_slug = _safe_slug(slug)
    kernel_ref = f"{safe_owner}/{safe_kernel_slug}"
    bounded_operation_timeout = min(
        float(wait_timeout_seconds), max(30.0, float(operation_timeout_seconds))
    )
    private_configuration_b64 = base64.b64encode(
        json.dumps(
            {
                "coordinator_url": str(coordinator_url).rstrip("/"),
                "coordinator_token": str(coordinator_token),
                "hf_token": str(hf_token),
            },
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii")
    (package / "kernel.py").write_text(
        render_kernel(
            bundle_archive_b64=_bundle_archive_b64(),
            private_configuration_b64=private_configuration_b64,
            old_identity_nonce=old_identity_nonce or secrets.token_urlsafe(24),
            replacement_identity_nonce=(
                replacement_identity_nonce or secrets.token_urlsafe(24)
            ),
            wait_timeout_seconds=wait_timeout_seconds,
            operation_timeout_seconds=bounded_operation_timeout,
            transport_optimization_after_step=int(
                transport_optimization_after_step
            ),
            replacement_after_steps=int(replacement_after_steps),
        ),
        encoding="utf-8",
    )
    metadata = {
        "id": kernel_ref,
        "title": safe_kernel_slug.replace("-", " ").title(),
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "true",
        "enable_internet": "true",
        "machine_shape": "tpuV5e8",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    _write_json(package / "kernel-metadata.json", metadata)
    report = {
        "schema": PACKAGE_SCHEMA,
        "ok": True,
        "role": "tpu",
        "kernel_ref": kernel_ref,
        "package_dir": str(package),
        "private_kernel": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "machine_shape": "tpuV5e8",
        "jax_tpu_resource_group_count": 1,
        "expected_tpu_device_count": 8,
        "logical_restart_process_count": 2,
        "operation_timeout_seconds": bounded_operation_timeout,
        "transport_optimization_after_step": int(
            transport_optimization_after_step
        ),
        "replacement_after_steps": int(replacement_after_steps),
        "private_coordinator_inputs_embedded": True,
        "credential_values_public": False,
        "coordinator_url_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    _write_json(
        output / "training_heterogeneous_tpu_beta_kaggle_package.json", report
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--coordinator-token", required=True)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--wait-timeout-seconds", type=float, default=21600.0)
    parser.add_argument("--operation-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--transport-optimization-after-step", type=int, default=-1)
    parser.add_argument("--replacement-after-steps", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_package(
        args.output_dir,
        owner=args.owner,
        slug=args.slug,
        coordinator_url=args.coordinator_url,
        coordinator_token=args.coordinator_token,
        hf_token=args.hf_token,
        wait_timeout_seconds=args.wait_timeout_seconds,
        operation_timeout_seconds=args.operation_timeout_seconds,
        transport_optimization_after_step=args.transport_optimization_after_step,
        replacement_after_steps=args.replacement_after_steps,
    )
    public = {key: value for key, value in report.items() if key != "package_dir"}
    print(json.dumps(public, sort_keys=True) if args.json else public)


if __name__ == "__main__":
    main()
