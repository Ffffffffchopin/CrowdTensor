#!/usr/bin/env python3
"""Build one private Kaggle Kernel for the heterogeneous Training Beta gate."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import secrets
import shutil
import textwrap
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCHEMA = "crowdtensor_heterogeneous_training_beta_kaggle_package_v1"
KERNEL_SCHEMA = "crowdtensor_heterogeneous_training_beta_kaggle_kernel_v1"
ROLES = {"gpu_a", "gpu_b", "cpu"}
SOURCE_FILES = [
    "crowdtensor/__init__.py",
    "crowdtensor/elastic_checkpoint_storage.py",
    "crowdtensor/elastic_training_runtime.py",
    "crowdtensor/elastic_training_client.py",
    "crowdtensor/heterogeneous_training_manifest.py",
    "crowdtensor/heterogeneous_training_scheduler.py",
    "crowdtensor/heterogeneous_tensor_transport.py",
    "crowdtensor/heterogeneous_training_checkpoint.py",
    "crowdtensor/heterogeneous_qwen_source.py",
    "crowdtensor/heterogeneous_qwen_training.py",
    "crowdtensor/heterogeneous_jax_qwen_training.py",
    "crowdtensor/heterogeneous_training_miner.py",
    "crowdtensor/qwen15b_training.py",
    "scripts/training_heterogeneous_export_reload_probe.py",
    "scripts/training_heterogeneous_beta_worker_entry.py",
]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", str(value).lower()).strip("-")
    return re.sub(r"-+", "-", slug)[:63].strip("-") or "ct-heterogeneous-training"


def _bundle_archive_b64() -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in SOURCE_FILES:
            archive.writestr(relative, (ROOT / relative).read_bytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def render_kernel(
    *,
    role: str,
    bundle_archive_b64: str,
    private_configuration_b64: str,
    identity_nonces: dict[str, str],
    wait_timeout_seconds: float,
    operation_timeout_seconds: float,
    recovery_mode: bool = False,
    transport_optimization_after_step: int = -1,
    replacement_after_steps: int = 3,
) -> str:
    if role not in ROLES:
        raise ValueError("heterogeneous_kaggle_package_role_invalid")
    if replacement_after_steps is not None and int(replacement_after_steps) < 0:
        raise ValueError("heterogeneous_kaggle_replacement_step_invalid")
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
ROLE = {role!r}
REPORT_PATH = WORKING / "training_heterogeneous_beta_kernel.json"
PRIVATE_ROOT = WORKING / f".crowdtensor-heterogeneous-training-{{ROLE}}"
BUNDLE_ROOT = PRIVATE_ROOT / "bundle"
PRIVATE_CONFIGURATION = PRIVATE_ROOT / "private_configuration.json"
BUNDLE_ARCHIVE_B64 = {bundle_archive_b64!r}
PRIVATE_CONFIGURATION_B64 = {private_configuration_b64!r}
IDENTITY_NONCES = {identity_nonces!r}
WAIT_TIMEOUT_SECONDS = {float(wait_timeout_seconds)!r}
OPERATION_TIMEOUT_SECONDS = {float(operation_timeout_seconds)!r}
RECOVERY_MODE = {bool(recovery_mode)!r}
TRANSPORT_OPTIMIZATION_AFTER_STEP = {int(transport_optimization_after_step)!r}
REPLACEMENT_AFTER_STEPS = {int(replacement_after_steps)!r}
KERNEL_SCHEMA = {KERNEL_SCHEMA!r}


def write_report(value):
    REPORT_PATH.write_text(json.dumps(value, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def safe_blocker(exc):
    text = str(exc)
    if text.startswith(("heterogeneous_", "elastic_", "qwen15b_")):
        return re.sub(r"[^a-zA-Z0-9:_-]", "_", text[:180])
    return f"heterogeneous_kaggle_kernel_failed:{{type(exc).__name__}}"


def ensure_dependencies():
    try:
        from packaging.version import Version
        torchao_version = importlib.metadata.version("torchao")
        if Version(torchao_version) < Version("0.16.0"):
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                check=True,
                timeout=300,
            )
    except importlib.metadata.PackageNotFoundError:
        pass
    required = {{
        "transformers": "5.9.0",
        "peft": "0.19.1",
        "safetensors": "0.7.0",
    }}
    installed = {{}}
    for name in required:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = ""
    if any(installed[name] != version for name, version in required.items()):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-cache-dir",
                "transformers==5.9.0",
                "peft==0.19.1",
                "safetensors==0.7.0",
                "accelerate>=1.2,<2",
                "fastapi>=0.115,<1",
                "httpx>=0.27,<1",
                "pydantic>=2.10,<3",
                "psutil>=5.9,<8",
            ],
            check=True,
            timeout=900,
        )
    return {{name: importlib.metadata.version(name) for name in required}}


def request_status():
    private = json.loads(PRIVATE_CONFIGURATION.read_text(encoding="utf-8"))
    request = urllib.request.Request(
        private["coordinator_url"].rstrip("/") + "/elastic-training/status",
        headers={{
            "User-Agent": "crowdtensor-heterogeneous-kaggle-kernel/1",
            "x-crowdtensor-miner-token": private["coordinator_token"],
        }},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=60.0) as response:
        value = json.loads(response.read().decode("utf-8"))
    return value if isinstance(value, dict) else {{}}


def launch(
    label,
    *,
    device_policy,
    physical_cuda_index=None,
    max_steps=0,
    session_retries=2,
):
    output = WORKING / f"worker-{{label}}.json"
    private_worker_root = PRIVATE_ROOT / f"worker-{{label}}"
    log_path = PRIVATE_ROOT / f"worker-{{label}}.log"
    command = [
        sys.executable,
        str(BUNDLE_ROOT / "scripts" / "training_heterogeneous_beta_worker_entry.py"),
        "--private-configuration",
        str(PRIVATE_CONFIGURATION),
        "--output",
        str(output),
        "--private-root",
        str(private_worker_root),
        "--deployment-role",
        label,
        "--identity-nonce",
        IDENTITY_NONCES[label],
        "--device-policy",
        device_policy,
        "--max-steps",
        str(int(max_steps)),
        "--wait-timeout",
        str(WAIT_TIMEOUT_SECONDS),
        "--operation-timeout",
        str(OPERATION_TIMEOUT_SECONDS),
        "--session-retries",
        str(int(session_retries)),
        "--transport-optimization-after-step",
        str(TRANSPORT_OPTIMIZATION_AFTER_STEP),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if physical_cuda_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(int(physical_cuda_index))
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._ct_log = log
    return {{
        "label": label,
        "process": process,
        "output": output,
        "physical_cuda_index": physical_cuda_index,
    }}


def wait_worker(record, *, deadline):
    process = record["process"]
    remaining = max(1.0, deadline - time.monotonic())
    try:
        returncode = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30.0)
        returncode = process.returncode
    finally:
        record["process"]._ct_log.close()
    report = {{}}
    if record["output"].is_file():
        try:
            report = json.loads(record["output"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {{}}
    return {{
        "label": record["label"],
        "physical_cuda_index": record["physical_cuda_index"],
        "returncode": returncode,
        "report": report,
    }}


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
                observation["missing_stage_ids"]
                or observation["runtime_state"] == "paused_waiting_for_miners"
            ):
                return {{
                    "verified": True,
                    "automatic_takeover_observed": False,
                    "observations": observations[-12:],
                }}
            if (
                observation["committed_step"] >= REPLACEMENT_AFTER_STEPS
                and observation["runtime_state"] == "running"
                and not observation["missing_stage_ids"]
                and observation["placement_generation"] >= 2
            ):
                return {{
                    "verified": False,
                    "automatic_takeover_observed": True,
                    "observations": observations[-12:],
                }}
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(2.0)
    return {{
        "verified": False,
        "automatic_takeover_observed": False,
        "observations": observations[-12:],
    }}


def run_export_reload(deadline):
    output = WORKING / "training_heterogeneous_export_reload_probe.json"
    log_path = PRIVATE_ROOT / "export-reload.log"
    command = [
        sys.executable,
        str(BUNDLE_ROOT / "scripts" / "training_heterogeneous_export_reload_probe.py"),
        "--private-configuration",
        str(PRIVATE_CONFIGURATION),
        "--private-root",
        str(PRIVATE_ROOT / "export-reload"),
        "--output",
        str(output),
        "--wait-timeout",
        str(max(60.0, deadline - time.monotonic())),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=max(60.0, deadline - time.monotonic()),
                check=False,
            )
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            returncode = None
    value = {{}}
    if output.is_file():
        try:
            value = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = {{}}
    return {{"returncode": returncode, "report": value}}


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
    report = {{
        "schema": KERNEL_SCHEMA,
        "ok": False,
        "kernel_role": ROLE,
        "worker_results": [],
        "pause_observation": {{}},
        "export_reload": {{}},
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
        if ROLE == "gpu_a" and RECOVERY_MODE:
            records.extend(
                [
                    launch(
                        "gpu_stable_a1",
                        device_policy="cuda",
                        physical_cuda_index=1,
                    ),
                    launch(
                        "gpu_replacement",
                        device_policy="cuda",
                        physical_cuda_index=0,
                    ),
                ]
            )
            for record in list(records):
                report["worker_results"].append(
                    wait_worker(record, deadline=deadline)
                )
                records.remove(record)
        elif ROLE == "gpu_a":
            stable = launch(
                "gpu_stable_a1",
                device_policy="cuda",
                physical_cuda_index=1,
            )
            old = launch(
                "gpu_old",
                device_policy="cuda",
                physical_cuda_index=0,
                max_steps=REPLACEMENT_AFTER_STEPS,
                session_retries=0,
            )
            records.extend([stable, old])
            old_result = wait_worker(old, deadline=deadline)
            report["worker_results"].append(old_result)
            records.remove(old)
            report["pause_observation"] = wait_for_pause(
                min(deadline, time.monotonic() + 180.0)
            )
            replacement = launch(
                "gpu_replacement",
                device_policy="cuda",
                physical_cuda_index=0,
            )
            records.append(replacement)
            for record in [stable, replacement]:
                report["worker_results"].append(
                    wait_worker(record, deadline=deadline)
                )
                records.remove(record)
        elif ROLE == "gpu_b":
            records.extend(
                [
                    launch(
                        "gpu_stable_b0",
                        device_policy="cuda",
                        physical_cuda_index=0,
                    ),
                    launch(
                        "gpu_stable_b1",
                        device_policy="cuda",
                        physical_cuda_index=1,
                    ),
                ]
            )
            for record in list(records):
                report["worker_results"].append(
                    wait_worker(record, deadline=deadline)
                )
                records.remove(record)
        else:
            if REPLACEMENT_AFTER_STEPS > 0:
                cpu = launch(
                    "cpu_old",
                    device_policy="cpu",
                    max_steps=REPLACEMENT_AFTER_STEPS,
                    session_retries=0,
                )
                records.append(cpu)
                report["worker_results"].append(wait_worker(cpu, deadline=deadline))
                records.remove(cpu)
                report["pause_observation"] = wait_for_pause(
                    min(deadline, time.monotonic() + 300.0)
                )
                replacement = launch("cpu_replacement", device_policy="cpu")
                records.append(replacement)
                report["worker_results"].append(
                    wait_worker(replacement, deadline=deadline)
                )
                records.remove(replacement)
            else:
                cpu = launch("cpu", device_policy="cpu")
                records.append(cpu)
                report["worker_results"].append(wait_worker(cpu, deadline=deadline))
                records.remove(cpu)
            report["export_reload"] = run_export_reload(deadline)
        reports = [item.get("report") or {{}} for item in report["worker_results"]]
        report["ok"] = bool(
            reports
            and all(item.get("ok") is True for item in reports)
            and all(item.get("returncode") == 0 for item in report["worker_results"])
            and (
                ROLE not in {{"gpu_a", "cpu"}}
                or RECOVERY_MODE
                or REPLACEMENT_AFTER_STEPS <= 0
                or report["pause_observation"].get("verified") is True
                or report["pause_observation"].get("automatic_takeover_observed")
                is True
            )
            and (
                ROLE != "cpu"
                or (
                    report["export_reload"].get("returncode") == 0
                    and (report["export_reload"].get("report") or {{}}).get("ok") is True
                )
            )
        )
        if not report["ok"]:
            report["blockers"].append("heterogeneous_kaggle_worker_acceptance_incomplete")
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
    role: str,
    coordinator_url: str,
    coordinator_token: str,
    hf_token: str = "",
    wait_timeout_seconds: float = 10800.0,
    operation_timeout_seconds: float = 1800.0,
    recovery_mode: bool = False,
    identity_nonces: dict[str, str] | None = None,
    transport_optimization_after_step: int = -1,
    replacement_after_steps: int | None = None,
) -> dict[str, Any]:
    if role not in ROLES:
        raise ValueError("heterogeneous_kaggle_package_role_invalid")
    if not str(coordinator_url).strip() or not str(coordinator_token):
        raise ValueError("heterogeneous_kaggle_package_private_inputs_required")
    output = Path(output_dir).resolve()
    shutil.rmtree(output, ignore_errors=True)
    package = output / "private-kernel"
    package.mkdir(parents=True, exist_ok=True)
    safe_owner = _safe_slug(owner)
    safe_kernel_slug = _safe_slug(slug)
    kernel_ref = f"{safe_owner}/{safe_kernel_slug}"
    labels = (
        ["gpu_stable_a1", "gpu_old", "gpu_replacement"]
        if role == "gpu_a"
        else ["gpu_stable_b0", "gpu_stable_b1"]
        if role == "gpu_b"
        else (
            ["cpu_old", "cpu_replacement"]
            if int(replacement_after_steps or 0) > 0
            else ["cpu"]
        )
    )
    resolved_identity_nonces = dict(identity_nonces or {})
    if set(resolved_identity_nonces) - set(labels):
        raise ValueError("heterogeneous_kaggle_package_identity_labels_invalid")
    for label in labels:
        resolved_identity_nonces.setdefault(label, secrets.token_urlsafe(24))
    bounded_operation_timeout_seconds = min(
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
            role=role,
            bundle_archive_b64=_bundle_archive_b64(),
            private_configuration_b64=private_configuration_b64,
            identity_nonces=resolved_identity_nonces,
            wait_timeout_seconds=wait_timeout_seconds,
            operation_timeout_seconds=bounded_operation_timeout_seconds,
            recovery_mode=recovery_mode,
            transport_optimization_after_step=int(
                transport_optimization_after_step
            ),
            replacement_after_steps=(
                int(replacement_after_steps)
                if replacement_after_steps is not None
                else 3
                if role == "gpu_a"
                else 0
            ),
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
        "enable_gpu": "true" if role.startswith("gpu") else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if role.startswith("gpu"):
        metadata["machine_shape"] = "NvidiaTeslaT4"
    _write_json(package / "kernel-metadata.json", metadata)
    report = {
        "schema": PACKAGE_SCHEMA,
        "ok": True,
        "role": role,
        "kernel_ref": kernel_ref,
        "package_dir": str(package),
        "private_kernel": True,
        "enable_gpu": role.startswith("gpu"),
        "single_gpu_process_count": 2 if role.startswith("gpu") else 0,
        "pure_cpu_process_count": (
            2
            if role == "cpu" and int(replacement_after_steps or 0) > 0
            else 1
            if role == "cpu"
            else 0
        ),
        "replacement_process_included": bool(
            role == "gpu_a"
            or (role == "cpu" and int(replacement_after_steps or 0) > 0)
        ),
        "recovery_mode": bool(recovery_mode),
        "transport_optimization_after_step": int(
            transport_optimization_after_step
        ),
        "replacement_after_steps": (
            int(replacement_after_steps)
            if replacement_after_steps is not None
            else 3
            if role == "gpu_a"
            else 0
        ),
        "operation_timeout_seconds": bounded_operation_timeout_seconds,
        "private_coordinator_inputs_embedded": True,
        "credential_values_public": False,
        "coordinator_url_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    _write_json(output / "training_heterogeneous_beta_kaggle_package.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--role", choices=sorted(ROLES), required=True)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--coordinator-token", required=True)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--wait-timeout-seconds", type=float, default=10800.0)
    parser.add_argument("--operation-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--transport-optimization-after-step", type=int, default=-1)
    parser.add_argument("--replacement-after-steps", type=int, default=-1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_package(
        args.output_dir,
        owner=args.owner,
        slug=args.slug,
        role=args.role,
        coordinator_url=args.coordinator_url,
        coordinator_token=args.coordinator_token,
        hf_token=args.hf_token,
        wait_timeout_seconds=args.wait_timeout_seconds,
        operation_timeout_seconds=args.operation_timeout_seconds,
        transport_optimization_after_step=args.transport_optimization_after_step,
        replacement_after_steps=(
            args.replacement_after_steps
            if args.replacement_after_steps >= 0
            else None
        ),
    )
    if args.json:
        public = {key: value for key, value in report.items() if key != "package_dir"}
        print(json.dumps(public, sort_keys=True))
    else:
        print(f"training_heterogeneous_beta_kaggle_package role={args.role} ok=True")


if __name__ == "__main__":
    main()
