#!/usr/bin/env python3
"""Run bounded real Volunteer Campaign contributions in private Kaggle GPUs."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import re
import secrets
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.model_adapter import stable_hash
from scripts.kaggle_gpu_token_weekly_quota_probe import fetch_accelerator_quota
from scripts.training_cuda_kaggle_common import (
    authenticated_owner,
    delete_succeeded_or_absent,
    extract_kernel_ref,
    kaggle_env,
    public_safety_errors,
    push_accepted,
    run_command,
    status_class,
)


SCHEMA = "crowdtensor_volunteer_kaggle_contribution_probe_v1"
KERNEL_SCHEMA = "crowdtensor_volunteer_kaggle_contributor_v1"
KERNEL_REPORT = "volunteer_kaggle_contributor.json"
KERNEL_PROGRESS = "volunteer_kaggle_contributor_progress.json"
RUNTIME_REQUIREMENTS = (
    "httpx==0.28.1",
    "peft==0.19.1",
    "transformers==5.9.0",
    "safetensors==0.7.0",
    "accelerate==1.13.0",
)


def _hash(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _write(path: Path, value: Any, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(mode)


def _safe_slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", str(value).lower()).strip("-")
    return re.sub(r"-+", "-", normalized)[:63].strip("-")


def _snapshot(coordinator_url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        str(coordinator_url).rstrip("/") + "/v1/volunteer/public-snapshot",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict) or value.get("ok") is not True:
        raise RuntimeError("volunteer_kaggle_public_snapshot_invalid")
    return value


def _snapshot_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    campaign = dict(snapshot.get("campaign") or {})
    progress = dict(snapshot.get("progress") or {})
    return {
        "campaign_id": str(campaign.get("campaign_id") or ""),
        "campaign_manifest_hash": str(campaign.get("campaign_manifest_hash") or ""),
        "model_id": str(campaign.get("model_id") or ""),
        "dataset_id": str(campaign.get("dataset_id") or ""),
        "lifecycle": str(progress.get("lifecycle") or ""),
        "adapter_version": int(progress.get("adapter_version") or 0),
        "completed_rounds": int(progress.get("completed_rounds") or 0),
        "target_rounds": int(progress.get("target_rounds") or 0),
        "accepted_update_count": int(progress.get("accepted_update_count") or 0),
        "accepted_token_count": int(progress.get("accepted_token_count") or 0),
        "active_contributor_count": int(progress.get("active_contributor_count") or 0),
        "queued_work_count": int(progress.get("queued_work_count") or 0),
    }


def _kernel_source(
    *,
    invite_bytes: bytes,
    source_commit: str,
    cell_id: str,
    contributor_index: int,
) -> str:
    invite_b64 = base64.b64encode(invite_bytes).decode("ascii")
    return f'''import base64
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import traceback

SCHEMA = {KERNEL_SCHEMA!r}
SOURCE_COMMIT = {source_commit!r}
CELL_ID = {cell_id!r}
CONTRIBUTOR_INDEX = {int(contributor_index)!r}
INVITE_B64 = {invite_b64!r}
working = pathlib.Path("/kaggle/working")
runtime_root = pathlib.Path("/kaggle/temp") if pathlib.Path("/kaggle/temp").is_dir() else pathlib.Path("/tmp")
progress_path = working / {KERNEL_PROGRESS!r}
report_path = working / {KERNEL_REPORT!r}
invite_path = runtime_root / "crowdtensor-private-invite.json"
workspace = runtime_root / f"crowdtensor-volunteer-cell-{{CONTRIBUTOR_INDEX}}"
repository = runtime_root / "crowdtensor-source"
install_root = runtime_root / "crowdtensor-runtime"

def progress(phase):
    progress_path.write_text(json.dumps({{
        "schema": "crowdtensor_volunteer_kaggle_contributor_progress_v1",
        "phase": phase,
        "contributor_index": CONTRIBUTOR_INDEX,
        "public_artifact_safe": True,
    }}, sort_keys=True) + "\\n", encoding="utf-8")

def public_result(value):
    last = dict(value.get("last_report") or {{}})
    submission = dict(last.get("submission") or {{}})
    return {{
        "cell_id_hash": str(value.get("cell_id_hash") or ""),
        "completed_in_run": int(value.get("completed_in_run") or 0),
        "last_state": str(value.get("last_state") or ""),
        "campaign_id": str(last.get("campaign_id") or ""),
        "campaign_manifest_hash": str(last.get("campaign_manifest_hash") or ""),
        "round_index": int(last.get("round_index") or 0),
        "adapter_version_before": int(last.get("adapter_version") or 0),
        "optimizer_steps": int(last.get("optimizer_steps") or 0),
        "samples_seen": int(last.get("samples_seen") or 0),
        "tokens_seen": int(last.get("tokens_seen") or 0),
        "loss_start": float(last.get("loss_start") or 0.0),
        "loss_end": float(last.get("loss_end") or 0.0),
        "real_pytorch_autograd": last.get("real_pytorch_autograd") is True,
        "real_transformers_peft_lora": last.get("real_transformers_peft_lora") is True,
        "base_weights_frozen": last.get("base_weights_frozen") is True,
        "lease_heartbeat_enabled": last.get("lease_heartbeat_enabled") is True,
        "submission_accepted": submission.get("accepted") is True,
        "round_completed": submission.get("round_completed") is True,
    }}

report = {{
    "schema": SCHEMA,
    "ok": False,
    "contributor_index": CONTRIBUTOR_INDEX,
    "source_commit": SOURCE_COMMIT,
    "accelerator": "cuda:0",
    "credential_values_public": False,
    "coordinator_url_public": False,
    "private_paths_public": False,
    "public_artifact_safe": True,
}}
try:
    import torch
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if not report["cuda_available"]:
        raise RuntimeError("volunteer_kaggle_cuda_unavailable")
    progress("source_checkout_started")
    shutil.rmtree(repository, ignore_errors=True)
    subprocess.run([
        "git", "clone", "--filter=blob:none", "--no-checkout",
        "https://github.com/Ffffffffchopin/CrowdTensor.git", str(repository),
    ], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([
        "git", "-C", str(repository), "checkout", "--detach", SOURCE_COMMIT,
    ], check=True, stdout=subprocess.DEVNULL)
    actual_commit = subprocess.run([
        "git", "-C", str(repository), "rev-parse", "HEAD",
    ], check=True, capture_output=True, text=True).stdout.strip()
    if actual_commit != SOURCE_COMMIT:
        raise RuntimeError("volunteer_kaggle_source_commit_mismatch")
    report["source_commit_verified"] = True
    progress("runtime_install_started")
    install_root.mkdir(parents=True, exist_ok=True)
    requirements = {list(RUNTIME_REQUIREMENTS)!r}
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--target", str(install_root), "--upgrade", "--no-deps", *requirements,
    ], check=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repository), str(install_root)])
    subprocess.run([
        sys.executable, "-c",
        "import httpx,torch,transformers,peft,safetensors,accelerate,crowdtensor; print(crowdtensor.__version__)",
    ], env=env, check=True, stdout=subprocess.DEVNULL)
    progress("private_invite_ready")
    invite_path.write_bytes(base64.b64decode(INVITE_B64))
    invite_path.chmod(0o600)
    progress("volunteer_join_started")
    report["ordinary_volunteer_cell_path"] = True
    sys.path.insert(0, str(install_root))
    sys.path.insert(0, str(repository))
    from crowdtensor.volunteer_training_cell import HTTPVolunteerTransport, VolunteerTrainingCell
    from crowdtensor.volunteer_training_protocol import VolunteerProtocolError
    try:
        transport = HTTPVolunteerTransport.from_invite(invite_path, timeout_seconds=180)
        cell = VolunteerTrainingCell(
            transport,
            workspace,
            cell_id=CELL_ID,
            device="cuda:0",
            max_local_steps=1,
            max_download_bytes=2 * 1024**3,
        )
        value = cell.run(max_work_units=1, poll_interval_seconds=2.0)
    except VolunteerProtocolError as cell_error:
        raise RuntimeError("volunteer_kaggle_protocol_error:" + cell_error.code) from cell_error
    except BaseException as cell_error:
        lowered = str(cell_error).lower()
        markers = (
            ("out of memory", "cuda_out_of_memory"),
            ("operation not permitted", "accelerator_operation_not_permitted"),
            ("device-side assert", "cuda_device_side_assert"),
            ("invalid argument", "accelerator_invalid_argument"),
            ("initialization error", "cuda_initialization_error"),
            ("driver shutting down", "cuda_driver_shutting_down"),
            ("forward compatibility", "cuda_forward_compatibility_error"),
            ("busy or unavailable", "cuda_device_busy_or_unavailable"),
            ("not yet implemented", "accelerator_operation_not_implemented"),
            ("torchao", "incompatible_optional_torchao"),
            ("torchvision::nms", "incompatible_torchvision_nms"),
            ("could not import module", "transformers_module_import_failed"),
            ("no module named", "runtime_dependency_missing"),
            ("safetensor", "safetensors_runtime_failed"),
        )
        category = next((code for marker, code in markers if marker in lowered), type(cell_error).__name__)
        frames = traceback.extract_tb(cell_error.__traceback__)
        report["cell_error_diagnostic"] = {{
            "error_class": type(cell_error).__name__,
            "error_category": category,
            "error_message_hash": "sha256:" + hashlib.sha256(str(cell_error).encode()).hexdigest(),
            "frames": [
                {{
                    "file": pathlib.Path(frame.filename).name,
                    "function": frame.name,
                    "line": int(frame.lineno),
                }}
                for frame in frames[-8:]
            ],
            "raw_error_message_public": False,
            "public_artifact_safe": True,
        }}
        raise RuntimeError("volunteer_kaggle_cell_runtime_error:" + category) from cell_error
    contribution = public_result(value)
    report["contribution"] = contribution
    report["runtime_requirements_exact_pins"] = all("==" in item for item in requirements)
    report["real_training_verified"] = bool(
        contribution["completed_in_run"] == 1
        and contribution["last_state"] == "submitted"
        and contribution["optimizer_steps"] == 1
        and contribution["tokens_seen"] > 0
        and contribution["real_pytorch_autograd"]
        and contribution["real_transformers_peft_lora"]
        and contribution["base_weights_frozen"]
        and contribution["submission_accepted"]
    )
    report["ok"] = report["real_training_verified"]
    progress("contribution_completed")
except BaseException as exc:
    report["error_class"] = type(exc).__name__
    report["error_code"] = str(exc) if str(exc).startswith("volunteer_kaggle_") else "volunteer_kaggle_runtime_failed"
    report["traceback_hash"] = "sha256:" + hashlib.sha256(traceback.format_exc().encode()).hexdigest()
finally:
    invite_path.unlink(missing_ok=True)
    shutil.rmtree(workspace, ignore_errors=True)
    report["private_invite_deleted"] = not invite_path.exists()
    report["workspace_removed"] = not workspace.exists()
    report["ok"] = bool(report.get("ok") and report["private_invite_deleted"] and report["workspace_removed"])
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    progress("finished")
print(json.dumps({{"ok": report["ok"], "contributor_index": CONTRIBUTOR_INDEX}}, sort_keys=True))
'''


def build_private_packages(
    destination: Path,
    *,
    owner: str,
    invite_bytes: bytes,
    source_commit: str,
    contributor_count: int,
) -> list[dict[str, Any]]:
    packages: list[dict[str, Any]] = []
    suffix = time.strftime("%m%d-%H%M", time.gmtime()) + "-" + secrets.token_hex(3)
    for index in range(contributor_count):
        directory = destination / f"contributor-{index + 1}"
        directory.mkdir(parents=True, exist_ok=True)
        cell_id = "kaggle-founding-" + secrets.token_urlsafe(18)
        slug = _safe_slug(f"ct-volunteer-founding-{index + 1}-{suffix}")
        (directory / "kernel.py").write_text(
            _kernel_source(
                invite_bytes=invite_bytes,
                source_commit=source_commit,
                cell_id=cell_id,
                contributor_index=index + 1,
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
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        _write(directory / "kernel-metadata.json", metadata, mode=0o600)
        packages.append(
            {
                "directory": directory,
                "fallback_ref": metadata["id"],
                "contributor_index": index + 1,
            }
        )
    return packages


def _collect_output(
    ref: str,
    *,
    env: dict[str, str],
    destination: Path,
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        shutil.rmtree(destination, ignore_errors=True)
        destination.mkdir(parents=True, exist_ok=True)
        result = run_command(
            ["kaggle", "kernels", "output", ref, "-p", str(destination)],
            env=env,
            timeout=min(120.0, max(30.0, deadline - time.monotonic())),
        )
        report_path = destination / KERNEL_REPORT
        progress_path = destination / KERNEL_PROGRESS
        if result.get("ok") and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            phase = ""
            if progress_path.is_file():
                phase = str(json.loads(progress_path.read_text(encoding="utf-8")).get("phase") or "")
            return {"found": True, "attempt_count": attempts, "phase": phase, "report": report}
        time.sleep(min(10.0, 2.0 * attempts))
    return {"found": False, "attempt_count": attempts, "phase": "", "report": {}}


def _progress_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    fields = (
        "adapter_version",
        "completed_rounds",
        "accepted_update_count",
        "accepted_token_count",
    )
    return {field: int(after[field]) - int(before[field]) for field in fields}


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    private = output / ".private-kaggle-contribution"
    private.mkdir(parents=True, exist_ok=True)
    private.chmod(0o700)
    invite_path = Path(args.invite_file).expanduser().resolve()
    source_commit = str(args.source_commit).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise RuntimeError("volunteer_kaggle_source_commit_invalid")
    if not invite_path.is_file() or invite_path.stat().st_mode & 0o077:
        raise RuntimeError("volunteer_kaggle_private_invite_invalid")
    invite_bytes = invite_path.read_bytes()
    invite = json.loads(invite_bytes)
    coordinator_url = str(invite.get("coordinator_url") or "").rstrip("/")
    if invite.get("schema") != "crowdtensor_volunteer_training_invite_v1":
        raise RuntimeError("volunteer_kaggle_private_invite_schema_invalid")
    if not coordinator_url.startswith("https://"):
        raise RuntimeError("volunteer_kaggle_https_coordinator_required")
    before = _snapshot_summary(_snapshot(coordinator_url))
    expected_campaign_id = str(invite.get("campaign_id") or "")
    if before["campaign_id"] != expected_campaign_id or before["lifecycle"] != "running":
        raise RuntimeError("volunteer_kaggle_campaign_not_running")

    refs: list[str] = []
    reports: list[dict[str, Any]] = []
    phases: list[str] = []
    quota_summary: dict[str, Any] = {}
    cleanup = {
        "all_remote_kernels_deleted": False,
        "private_package_removed": False,
        "live_resources_left_running": True,
    }
    report: dict[str, Any] = {}
    try:
        with kaggle_env(args.kaggle_raw_token_file, username_hint=args.kaggle_username) as env:
            owner = authenticated_owner(env)
            if not owner:
                raise RuntimeError("volunteer_kaggle_authentication_failed")
            quota = fetch_accelerator_quota(env)
            gpu = dict(quota.get("gpu_quota") or {})
            quota_summary = {
                "ok": quota.get("ok") is True,
                "quota_refresh_time": str(quota.get("quota_refresh_time") or ""),
                "effective_remaining_seconds": float(
                    gpu.get("effective_remaining_after_reserved_seconds") or 0.0
                ),
                "reserved_seconds": float(gpu.get("time_reserved_seconds") or 0.0),
            }
            if not quota_summary["ok"] or quota_summary["effective_remaining_seconds"] < 1800:
                raise RuntimeError("volunteer_kaggle_gpu_quota_unavailable")
            packages = build_private_packages(
                private / "packages",
                owner=owner,
                invite_bytes=invite_bytes,
                source_commit=source_commit,
                contributor_count=int(args.contributor_count),
            )

            def push(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
                step = run_command(
                    ["kaggle", "kernels", "push", "-p", str(item["directory"])],
                    env=env,
                    timeout=300,
                )
                ref = extract_kernel_ref(str(step.get("output_tail") or ""), str(item["fallback_ref"]))
                if not push_accepted(step):
                    raise RuntimeError("volunteer_kaggle_kernel_push_rejected")
                return ref, step

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=int(args.contributor_count)
            ) as executor:
                pushed = list(executor.map(push, packages))
            refs = [item[0] for item in pushed]
            deadline = time.monotonic() + float(args.timeout_seconds)
            terminal: dict[str, str] = {}
            while time.monotonic() < deadline:
                terminal = {}
                for ref in refs:
                    status = run_command(
                        ["kaggle", "kernels", "status", ref], env=env, timeout=30
                    )
                    terminal[_hash(ref)] = status_class(str(status.get("output_tail") or ""))
                if all(state in {"complete", "failed"} for state in terminal.values()):
                    break
                time.sleep(8)
            if len(terminal) != len(refs) or not all(
                state == "complete" for state in terminal.values()
            ):
                raise RuntimeError("volunteer_kaggle_kernel_terminal_failure_or_timeout")
            for index, ref in enumerate(refs, start=1):
                collected = _collect_output(
                    ref,
                    env=env,
                    destination=private / "outputs" / f"contributor-{index}",
                )
                phases.append(str(collected.get("phase") or ""))
                if not collected.get("found"):
                    raise RuntimeError("volunteer_kaggle_kernel_output_missing")
                reports.append(dict(collected["report"]))

        after = _snapshot_summary(_snapshot(coordinator_url))
        delta = _progress_delta(before, after)
        contributions = [dict(item.get("contribution") or {}) for item in reports]
        kernel_diagnostics = [
            {
                "contributor_index": int(item.get("contributor_index") or 0),
                "ok": item.get("ok") is True,
                "error_class": str(item.get("error_class") or ""),
                "error_code": str(item.get("error_code") or ""),
                "traceback_hash": str(item.get("traceback_hash") or ""),
                "cuda_available": item.get("cuda_available") is True,
                "cuda_device_count": int(item.get("cuda_device_count") or 0),
                "source_commit_verified": item.get("source_commit_verified") is True,
                "private_invite_deleted": item.get("private_invite_deleted") is True,
                "workspace_removed": item.get("workspace_removed") is True,
                "cell_error_diagnostic": dict(item.get("cell_error_diagnostic") or {}),
            }
            for item in reports
        ]
        cell_hashes = {str(item.get("cell_id_hash") or "") for item in contributions}
        live_verified = bool(
            len(reports) == int(args.contributor_count)
            and all(item.get("ok") is True for item in reports)
            and all(item.get("real_training_verified") is True for item in reports)
            and len(cell_hashes) == int(args.contributor_count)
            and "" not in cell_hashes
            and delta["accepted_update_count"] >= int(args.contributor_count)
            and delta["completed_rounds"] >= 1
            and delta["adapter_version"] >= 1
            and delta["accepted_token_count"] > 0
        )
        report = {
            "schema": SCHEMA,
            "ok": live_verified,
            "live_run_performed": True,
            "live_contribution_verified": live_verified,
            "node_scope": "Kaggle logical multi-node",
            "physical_multi_machine_verified": False,
            "source_commit": source_commit,
            "source_commit_verified": all(
                item.get("source_commit_verified") is True for item in reports
            ),
            "logical_kernel_count": len(reports),
            "distinct_cell_count": len(cell_hashes - {""}),
            "providers": ["kaggle_cuda"],
            "campaign_before": before,
            "campaign_after": after,
            "progress_delta": delta,
            "contributions": contributions,
            "kernel_diagnostics": kernel_diagnostics,
            "kernel_terminal_phases": phases,
            "gpu_quota_preflight": quota_summary,
            "credential_values_public": False,
            "kaggle_account_names_public": False,
            "kaggle_kernel_references_public": False,
            "coordinator_url_public": False,
            "private_paths_public": False,
            "raw_data_public": False,
            "tensor_values_public": False,
            "public_artifact_safe": True,
        }
        if not live_verified:
            report["blockers"] = ["volunteer_kaggle_live_contribution_not_verified"]
    except BaseException as exc:
        code = str(exc).splitlines()[0] if str(exc).startswith("volunteer_kaggle_") else "volunteer_kaggle_live_failed"
        report = {
            "schema": SCHEMA,
            "ok": False,
            "live_run_performed": bool(refs),
            "live_contribution_verified": False,
            "node_scope": "Kaggle logical multi-node",
            "physical_multi_machine_verified": False,
            "source_commit": source_commit,
            "campaign_before": before,
            "blockers": [code[:180]],
            "kernel_terminal_phases": phases,
            "gpu_quota_preflight": quota_summary,
            "credential_values_public": False,
            "kaggle_account_names_public": False,
            "kaggle_kernel_references_public": False,
            "coordinator_url_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
    finally:
        deleted: list[bool] = []
        try:
            with kaggle_env(args.kaggle_raw_token_file, username_hint=args.kaggle_username) as env:
                for ref in refs:
                    deleted.append(
                        delete_succeeded_or_absent(
                            run_command(
                                ["kaggle", "kernels", "delete", ref, "-y"],
                                env=env,
                                timeout=120,
                            )
                        )
                    )
        except BaseException:
            deleted.extend([False] * max(0, len(refs) - len(deleted)))
        cleanup["all_remote_kernels_deleted"] = len(deleted) == len(refs) and all(deleted)
        shutil.rmtree(private, ignore_errors=True)
        cleanup["private_package_removed"] = not private.exists()
        cleanup["live_resources_left_running"] = not cleanup["all_remote_kernels_deleted"]
        report["cleanup"] = cleanup
        report["cleanup_verified"] = bool(
            cleanup["all_remote_kernels_deleted"]
            and cleanup["private_package_removed"]
            and not cleanup["live_resources_left_running"]
        )
        safety = scan_public_value(report)
        helper_errors = public_safety_errors(report)
        report["public_safety"] = safety
        report["public_safety_helper_errors"] = helper_errors
        report["public_artifact_safe"] = bool(safety.get("ok") and not helper_errors)
        report["ok"] = bool(
            report.get("ok") and report["cleanup_verified"] and report["public_artifact_safe"]
        )
        report["content_hash"] = stable_hash(report)
        _write(output / "volunteer_training_kaggle_contribution_probe.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--invite-file", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--kaggle-raw-token-file", required=True)
    parser.add_argument("--kaggle-username", required=True)
    parser.add_argument("--contributor-count", type=int, choices=(2,), default=2)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_probe(args)
    print(json.dumps(report, sort_keys=True) if args.json else f"ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
