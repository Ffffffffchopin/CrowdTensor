#!/usr/bin/env python3
"""Prove full-offline Qwen training continuation with replacement Kaggle Miners."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import hmac
import json
import os
import secrets
import shutil
import socket
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from coordinator import create_app  # noqa: E402
from crowdtensor.elastic_training_runtime import (  # noqa: E402
    ElasticTrainingRuntime,
    install_elastic_training_routes,
)
from crowdtensor.qwen15b_four_gpu_runtime import (  # noqa: E402
    four_stage_overlap_summary,
)
from crowdtensor.qwen15b_training import (  # noqa: E402
    MODEL_ID,
    MODEL_PARAMETER_COUNT,
    MODEL_REVISION,
    fetch_bytes,
    sha256_bytes,
    sha256_file,
    stable_hash,
    _hf_url,
)
from crowdtensor.qwen15b_training_rendezvous import (  # noqa: E402
    Qwen15BTrainingRendezvous,
    install_qwen15b_training_routes,
)
from scripts.training_cuda_kaggle_common import (  # noqa: E402
    delete_succeeded_or_absent,
    extract_kernel_ref,
    public_safety_errors,
    push_accepted,
    run_command,
    safe_slug,
    status_class,
    utc_now,
)
from scripts.training_cuda_two_node_probe import (  # noqa: E402
    ensure_cloudflared,
    stop_process,
)
from scripts.training_qwen15b_four_gpu_package import build_package  # noqa: E402
from scripts.training_qwen15b_four_gpu_probe import (  # noqa: E402
    _credential_sections,
    _extract_adapter,
    _load,
    _preserve_public_kernel_log,
    inspect_adapter_bundle,
    preflight_accounts,
    start_verified_tunnel,
)


SCHEMA = "crowdtensor_qwen15b_elastic_live_probe_v1"
WORKER_REPORT = "training_qwen15b_four_gpu_worker.json"
OUTPUT_PATTERN = (
    r"training_qwen15b_(four_gpu_worker\.json|kernel_[ab]_checkpoint_bundle\.zip|"
    r"standard_peft_adapter\.zip)"
)
TERMINAL = {"complete", "failed"}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _get_json(url: str, *, token: str = "", timeout: float = 15.0) -> dict[str, Any]:
    headers = {"User-Agent": "crowdtensor-elastic-live-probe/1"}
    if token:
        headers["x-crowdtensor-miner-token"] = token
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("elastic_live_route_response_invalid")
    return value


def _wait_ready(url: str, *, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        try:
            if _get_json(f"{url}/ready", timeout=5.0).get("ok") is True:
                return
        except BaseException:
            pass
        time.sleep(0.25)
    raise TimeoutError("elastic_live_coordinator_readiness_timeout")


class ElasticQwenCoordinator:
    def __init__(
        self,
        *,
        private_root: Path,
        port: int,
        run_id: str,
        token: str,
        target_steps: int = 8,
        microbatches_per_step: int = 4,
        lease_seconds: float = 90.0,
        model_id: str = MODEL_ID,
        model_revision: str = MODEL_REVISION,
    ) -> None:
        self.private_root = private_root
        self.port = int(port)
        self.run_id = str(run_id)
        self.token = str(token)
        self.runtime = ElasticTrainingRuntime(
            private_root / "elastic-training.sqlite3",
            run_id=run_id,
            target_steps=target_steps,
            microbatches_per_step=microbatches_per_step,
            lease_seconds=lease_seconds,
            legacy_model_id=model_id,
            legacy_model_revision=model_revision,
        )
        self.rendezvous = Qwen15BTrainingRendezvous(
            run_id=run_id,
            state_path=private_root / "qwen-rendezvous-state.json",
        )
        self.server: Any = None
        self.thread: threading.Thread | None = None

    def authorize(self, value: str | None) -> None:
        from fastapi import HTTPException

        if value is None or not hmac.compare_digest(value, self.token):
            raise HTTPException(status_code=401, detail="unauthorized")

    def start(self) -> None:
        import uvicorn

        app = create_app(
            state_dir=self.private_root / "coordinator-state",
            lease_seconds=1800.0,
            backlog=0,
            task_lanes=[],
            miner_token=self.token,
        )
        install_qwen15b_training_routes(
            app, rendezvous=self.rendezvous, authorize=self.authorize
        )
        install_elastic_training_routes(
            app, runtime=self.runtime, authorize=self.authorize
        )
        app.state.qwen15b_training_rendezvous = self.rendezvous
        app.state.elastic_training_runtime = self.runtime
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self.server = uvicorn.Server(config)
        self.server.install_signal_handlers = lambda: None
        self.thread = threading.Thread(
            target=self.server.run,
            name="qwen15b-elastic-coordinator",
            daemon=True,
        )
        self.thread.start()
        _wait_ready(f"http://127.0.0.1:{self.port}")

    def stop(self) -> bool:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=20.0)
        return bool(self.thread is None or not self.thread.is_alive())


def _public_step(step: dict[str, Any], *, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "ok": step.get("ok") is True,
        "returncode": step.get("returncode"),
        "timed_out": step.get("timed_out") is True,
        "duration_seconds": float(step.get("duration_seconds") or 0),
        "output_hash": stable_hash({"output": str(step.get("output_tail") or "")}),
    }


def _inspect_checkpoint_bundle(
    path: Path,
    worker_bundle: dict[str, Any],
    *,
    expected_step: int,
    microbatches_per_step: int = 4,
    expected_model_id: str = MODEL_ID,
    expected_model_revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    report = {
        "present": path.is_file(),
        "worker_hash_match": False,
        "archive_safe": False,
        "checkpoint_manifest_count": 0,
        "expected_step": int(expected_step),
        "all_expected_steps_verified": False,
        "all_component_hashes_verified": False,
        "manifest_summaries": [],
        "checkpoint_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    if not path.is_file():
        report["verified"] = False
        return report
    file_hash = sha256_file(path)
    report["file_hash"] = file_hash
    report["byte_count"] = path.stat().st_size
    report["worker_hash_match"] = file_hash == worker_bundle.get("file_hash")
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        safe = len(names) == len(set(names)) and all(
            name
            and not name.startswith(("/", "\\"))
            and ".." not in Path(name).parts
            for name in names
        )
        manifests = [name for name in names if name.endswith("_checkpoint.json")]
        summaries = []
        components_ok = True
        for name in manifests:
            manifest = json.loads(archive.read(name))
            parent = Path(name).parent
            item_components_ok = True
            for file_key, hash_key in (
                ("adapter_file", "adapter_file_hash"),
                ("optimizer_file", "optimizer_file_hash"),
                ("grad_scaler_file", "grad_scaler_file_hash"),
                ("rng_file", "rng_file_hash"),
            ):
                member = str(parent / str(manifest.get(file_key) or ""))
                if (
                    member not in names
                    or sha256_bytes(archive.read(member)) != manifest.get(hash_key)
                ):
                    item_components_ok = False
            components_ok = components_ok and item_components_ok
            summaries.append(
                {
                    "stage_id": int(manifest.get("stage_id", -1)),
                    "global_step": int(manifest.get("global_step") or 0),
                    "optimizer_step": int(manifest.get("optimizer_step") or 0),
                    "dataset_cursor": int(manifest.get("dataset_cursor") or 0),
                    "model_id": str(manifest.get("model_id") or ""),
                    "model_revision": str(manifest.get("model_revision") or ""),
                    "checkpoint_content_hash": str(manifest.get("content_hash") or ""),
                    "component_hashes_verified": item_components_ok,
                }
            )
        report.update(
            {
                "archive_safe": safe,
                "checkpoint_manifest_count": len(manifests),
                "all_expected_steps_verified": bool(summaries)
                and all(
                    item["global_step"] == int(expected_step)
                    and item["optimizer_step"] == int(expected_step)
                    and item["dataset_cursor"]
                    == int(expected_step) * int(microbatches_per_step)
                    for item in summaries
                ),
                "all_component_hashes_verified": components_ok and bool(summaries),
                "manifest_summaries": sorted(
                    summaries, key=lambda item: item["stage_id"]
                ),
            }
        )
    report["verified"] = bool(
        report["worker_hash_match"]
        and report["archive_safe"]
        and report["checkpoint_manifest_count"] == 2
        and report["all_expected_steps_verified"]
        and report["all_component_hashes_verified"]
        and all(
            item["model_id"] == expected_model_id
            and item["model_revision"] == expected_model_revision
            for item in report["manifest_summaries"]
        )
    )
    return report


def _wait_pair(
    refs: list[str],
    *,
    env: dict[str, str],
    timeout: float,
    status_timeout: float,
    poll_interval: float,
) -> tuple[dict[str, str], list[dict[str, Any]], int]:
    deadline = time.monotonic() + float(timeout)
    observations = []
    maximum_running = 0
    while time.monotonic() < deadline:
        states = {}
        for ref in refs:
            step = run_command(
                ["kaggle", "kernels", "status", ref],
                env=env,
                timeout=min(float(status_timeout), max(1.0, deadline - time.monotonic())),
            )
            states[ref] = status_class(str(step.get("output_tail") or ""))
        running = sum(value == "running" for value in states.values())
        maximum_running = max(maximum_running, running)
        observations.append(
            {
                "observed_at": utc_now(),
                "running_count": running,
                "queued_count": sum(value == "queued" for value in states.values()),
                "complete_count": sum(value == "complete" for value in states.values()),
                "failed_count": sum(value == "failed" for value in states.values()),
            }
        )
        if all(value in TERMINAL for value in states.values()):
            return states, observations, maximum_running
        time.sleep(min(float(poll_interval), max(0.1, deadline - time.monotonic())))
    raise TimeoutError("elastic_dual_kernel_generation_timeout")


def _delete_refs(
    refs: list[str],
    *,
    env: dict[str, str],
    timeout: float,
    role_by_ref: dict[str, str],
) -> tuple[list[dict[str, Any]], bool]:
    reports = []
    for ref in sorted(set(refs)):
        step = run_command(
            ["kaggle", "kernels", "delete", ref, "-y"],
            env=env,
            timeout=float(timeout),
        )
        reports.append(
            {
                **_public_step(step, role=role_by_ref.get(ref, "")),
                "deleted_or_absent": delete_succeeded_or_absent(step),
                "kernel_ref_hash": stable_hash({"ref": ref}),
            }
        )
    return reports, bool(
        len(reports) == len(set(refs))
        and all(item["deleted_or_absent"] for item in reports)
    )


def _launch_generation(
    *,
    generation: str,
    expected_start_step: int,
    segment_end_step: int,
    owner: str,
    suffix: str,
    config: dict[str, Any],
    tokenized: Path,
    coordinator_url: str,
    coordinator_token: str,
    run_id: str,
    env: dict[str, str],
    private: Path,
    output: Path,
    allocation_timeout: float,
    push_timeout: float,
    status_timeout: float,
    output_timeout: float,
    delete_timeout: float,
    poll_interval: float,
    cleanup_registry: list[str],
    product_miner_mode: bool = False,
    product_role: str = "auto",
    max_steps_per_session: int = 0,
    require_worker_adapter: bool = True,
    parallel_pushes: bool = True,
    target_steps: int = 8,
    microbatches_per_step: int = 4,
    learning_rate: float = 5e-4,
    lora_rank: int = 4,
    lora_alpha: int = 8,
    kernel_timeout_seconds: int = 1800,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    parameter_count: int = MODEL_PARAMETER_COUNT,
    source_layout_path: Path | None = None,
    defer_evaluation: bool = False,
    adapter_destination_name: str = "training_qwen15b_standard_peft_adapter.zip",
) -> tuple[dict[str, Any], list[str]]:
    packages = []
    for role in ("kernel_a", "kernel_b"):
        generation_slug = safe_slug(generation)[:12]
        slug = safe_slug(f"ct-qel-{generation_slug}-{role}-{suffix}")
        ref = f"{safe_slug(owner)}/{slug}"
        packages.append(
            build_package(
                private / f"package-{generation}-{role}",
                owner=owner,
                slug=slug,
                role=role,
                config=config,
                tokenized_payload_path=tokenized,
                coordinator_url=coordinator_url,
                coordinator_token=coordinator_token,
                run_id=run_id,
                wait_timeout=min(900.0, allocation_timeout),
                elastic_mode=True,
                miner_id_hash=stable_hash(
                    {"generation": generation, "role": role, "kernel_ref": ref}
                ),
                registration_nonce=secrets.token_urlsafe(32),
                expected_start_step=expected_start_step,
                segment_end_step=segment_end_step,
                target_steps=target_steps,
                microbatch_count=microbatches_per_step,
                learning_rate=learning_rate,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                heartbeat_interval_seconds=5.0,
                product_miner_mode=product_miner_mode,
                product_role=product_role,
                max_steps_per_session=max_steps_per_session,
                model_id=model_id,
                model_revision=model_revision,
                parameter_count=parameter_count,
                source_layout_path=source_layout_path,
                defer_evaluation=defer_evaluation,
            )
        )
    started_monotonic = time.monotonic()
    started_epoch = time.time()

    def push(package: dict[str, Any]) -> dict[str, Any]:
        remaining = max(1.0, allocation_timeout - (time.monotonic() - started_monotonic))
        step = run_command(
            [
                "kaggle",
                "kernels",
                "push",
                "-p",
                str(package["package_dir"]),
                "-t",
                str(int(kernel_timeout_seconds)),
                "--accelerator",
                "NvidiaTeslaT4",
            ],
            env=env,
            timeout=min(push_timeout, remaining),
        )
        return {
            "role": package["role"],
            "step": step,
            "accepted": push_accepted(step),
            "ref": extract_kernel_ref(
                str(step.get("output_tail") or ""), str(package["kernel_ref"])
            ),
        }

    if parallel_pushes:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            pushes = list(executor.map(push, packages))
    else:
        pushes = [push(package) for package in packages]
    refs = [str(item["ref"]) for item in pushes if item["accepted"]]
    cleanup_registry.extend(refs)
    role_by_ref = {str(item["ref"]): str(item["role"]) for item in pushes}
    generation_report: dict[str, Any] = {
        "schema": (
            "crowdtensor_qwen15b_elastic_generation_v1"
            if model_id == MODEL_ID and model_revision == MODEL_REVISION
            else "crowdtensor_qwen_elastic_generation_v2"
        ),
        "generation": generation,
        "expected_start_step": int(expected_start_step),
        "segment_end_step": int(segment_end_step),
        "target_steps": int(target_steps),
        "microbatches_per_step": int(microbatches_per_step),
        "launched_at_epoch": started_epoch,
        "pushes": [
            {
                **_public_step(item["step"], role=item["role"]),
                "accepted": item["accepted"],
                "kernel_ref_hash": stable_hash({"ref": item["ref"]}),
            }
            for item in pushes
        ],
        "kernel_ref_hashes": [stable_hash({"ref": ref}) for ref in refs],
        "worker_reports": [],
        "checkpoint_bundles": [],
        "status_observations": [],
        "deletions": [],
        "all_kernels_deleted": False,
        "same_account": True,
        "public_artifact_safe": True,
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        "parameter_count": int(parameter_count),
    }
    if len(refs) != 2:
        deletion_reports, deleted = _delete_refs(
            refs,
            env=env,
            timeout=delete_timeout,
            role_by_ref=role_by_ref,
        )
        generation_report["deletions"] = deletion_reports
        generation_report["all_kernels_deleted"] = deleted
        _write(
            output / f"{generation}-launch-failed.json",
            generation_report,
        )
        raise RuntimeError("elastic_dual_kernel_push_incomplete")
    try:
        terminal, observations, maximum_running = _wait_pair(
            refs,
            env=env,
            timeout=max(1.0, allocation_timeout - (time.monotonic() - started_monotonic)),
            status_timeout=status_timeout,
            poll_interval=poll_interval,
        )
    except BaseException:
        _write(
            output / f"{generation}-wait-failed.json",
            generation_report,
        )
        raise
    generation_report["status_observations"] = observations
    generation_report["terminal_states"] = [terminal[ref] for ref in refs]
    generation_report["maximum_running_kernel_count"] = maximum_running
    adapter_report: dict[str, Any] = {}
    try:
        for ref in refs:
            role = role_by_ref[ref]
            private_output = private / f"output-{generation}-{role}"
            step = run_command(
                [
                    "kaggle",
                    "kernels",
                    "output",
                    ref,
                    "-p",
                    str(private_output),
                    "--force",
                    "--file-pattern",
                    OUTPUT_PATTERN,
                ],
                env=env,
                timeout=output_timeout,
            )
            generation_report.setdefault("outputs", []).append(
                _public_step(step, role=role)
            )
            generation_report.setdefault("kernel_logs", {})[
                role
            ] = _preserve_public_kernel_log(
                private_output,
                output / "logs" / f"{generation}-{role}.log",
            )
            artifact = _load(private_output / WORKER_REPORT)
            if artifact:
                generation_report["worker_reports"].append(artifact)
                _write(output / "workers" / f"{generation}-{role}.json", artifact)
            checkpoint_source = (
                private_output / f"training_qwen15b_{role}_checkpoint_bundle.zip"
            )
            checkpoint = _inspect_checkpoint_bundle(
                checkpoint_source,
                dict(artifact.get("checkpoint_bundle") or {}),
                expected_step=segment_end_step,
                microbatches_per_step=microbatches_per_step,
                expected_model_id=model_id,
                expected_model_revision=model_revision,
            )
            checkpoint["role"] = role
            generation_report["checkpoint_bundles"].append(checkpoint)
            checkpoint_source.unlink(missing_ok=True)
            checkpoint["private_archive_removed_after_validation"] = not checkpoint_source.exists()
            if role == "kernel_b" and segment_end_step == target_steps:
                adapter_source = (
                    private_output / "training_qwen15b_standard_peft_adapter.zip"
                )
                adapter_destination = (
                    output / str(adapter_destination_name)
                )
                if adapter_source.is_file():
                    shutil.move(str(adapter_source), adapter_destination)
                adapter_report = inspect_adapter_bundle(
                    adapter_destination,
                    dict(artifact.get("adapter_bundle") or {}),
                    expected_model_id=model_id,
                    expected_model_revision=model_revision,
                )
                if adapter_report.get("verified"):
                    _extract_adapter(adapter_destination, output / "exported_adapter")
    finally:
        deletion_reports, deleted = _delete_refs(
            refs,
            env=env,
            timeout=delete_timeout,
            role_by_ref=role_by_ref,
        )
        generation_report["deletions"] = deletion_reports
        generation_report["all_kernels_deleted"] = deleted
        generation_report["deleted_at_epoch"] = time.time()
        for package in packages:
            shutil.rmtree(Path(str(package["package_dir"])).parent, ignore_errors=True)
    generation_report["adapter_bundle"] = adapter_report
    generation_report["ok"] = bool(
        len(generation_report["worker_reports"]) == 2
        and all(item.get("ok") is True for item in generation_report["worker_reports"])
        and len(generation_report["checkpoint_bundles"]) == 2
        and all(item.get("verified") for item in generation_report["checkpoint_bundles"])
        and maximum_running == 2
        and generation_report["all_kernels_deleted"]
        and (
            segment_end_step < target_steps
            or not require_worker_adapter
            or adapter_report.get("verified") is True
        )
    )
    return generation_report, refs


def _evaluate(
    *,
    old: dict[str, Any],
    new: dict[str, Any],
    midpoint: dict[str, Any],
    final: dict[str, Any],
    rendezvous: dict[str, Any],
    pause_observations: list[dict[str, Any]],
    pause_seconds: float,
    target_steps: int = 8,
    replacement_step: int = 4,
    allow_deferred_evaluation: bool = False,
) -> dict[str, Any]:
    old_workers = [dict(item.get("worker") or {}) for item in old["worker_reports"]]
    new_workers = [dict(item.get("worker") or {}) for item in new["worker_reports"]]
    old_hashes = set(old.get("kernel_ref_hashes") or [])
    new_hashes = set(new.get("kernel_ref_hashes") or [])
    old_sessions = {
        item["miner_session_hash"]
        for item in midpoint.get("miners") or []
        if item.get("miner_id_hash")
        in {worker.get("elastic_client", {}).get("miner_id_hash") for worker in old_workers}
    }
    new_sessions = {
        item["miner_session_hash"]
        for item in final.get("miners") or []
        if item.get("miner_id_hash")
        in {worker.get("elastic_client", {}).get("miner_id_hash") for worker in new_workers}
    }
    old_steps = {
        worker.get("role"): [
            int(item.get("step") or 0)
            for item in (worker.get("runtime") or {}).get("step_reports") or []
        ]
        for worker in old_workers
    }
    new_steps = {
        worker.get("role"): [
            int(item.get("step") or 0)
            for item in (worker.get("runtime") or {}).get("step_reports") or []
        ]
        for worker in new_workers
    }
    replacement_checkpoint_events = {
        int(item.get("stage_id", -1)): str(item.get("archive_hash") or "")
        for item in final.get("events") or []
        if item.get("operation") == "stage_checkpoint_submitted"
        and int(item.get("target_step") or 0) == int(replacement_step)
    }
    restore_hashes = {
        int(item.get("stage_id", -1)): str(item.get("archive_hash") or "")
        for worker in new_workers
        for item in worker.get("central_checkpoint_restore") or []
    }
    epoch_by_id = {
        int(item["epoch_id"]): item for item in final.get("epochs") or []
    }
    replacement_assignments = [
        item
        for item in final.get("assignments") or []
        if int(epoch_by_id.get(int(item["epoch_id"]), {}).get("target_step") or 0)
        == int(replacement_step) + 1
        and epoch_by_id.get(int(item["epoch_id"]), {}).get("state") == "committed"
        and item.get("state") == "completed"
    ]
    new_miner_hashes = {
        worker.get("elastic_client", {}).get("miner_id_hash") for worker in new_workers
    }
    final_role_b = next(
        (worker for worker in new_workers if worker.get("role") == "kernel_b"), {}
    )
    expected_old_steps = list(range(1, int(replacement_step) + 1))
    expected_new_steps = list(range(int(replacement_step) + 1, int(target_steps) + 1))
    target_completed = bool(
        final.get("runtime_state") == "completed"
        and int(final.get("committed_step") or 0) == int(target_steps)
    )
    gates = {
        "old_generation_live_verified": old.get("ok") is True
        and all(worker.get("ok") is True for worker in old_workers),
        "old_generation_steps_1_to_4_verified": set(old_steps) == {"kernel_a", "kernel_b"}
        and all(value == expected_old_steps for value in old_steps.values()),
        "midpoint_step4_committed": int(midpoint.get("committed_step") or 0)
        == int(replacement_step),
        "all_old_miners_offline": midpoint.get("zero_live_miners") is True
        and midpoint.get("paused_waiting_for_miners") is True,
        "old_kernels_deleted_before_replacement": old.get("all_kernels_deleted") is True
        and float(old.get("deleted_at_epoch") or 0)
        <= float(new.get("launched_at_epoch") or 0),
        "bounded_no_miner_pause_verified": pause_seconds >= 5.0
        and len(pause_observations) >= 2
        and all(
            item.get("runtime_state") == "paused_waiting_for_miners"
            and int(item.get("committed_step") or 0) == int(replacement_step)
            and int(item.get("live_miner_count", -1)) == 0
            for item in pause_observations
        ),
        "new_generation_live_verified": new.get("ok") is True
        and all(worker.get("ok") is True for worker in new_workers),
        "new_generation_steps_5_to_8_verified": set(new_steps) == {"kernel_a", "kernel_b"}
        and all(value == expected_new_steps for value in new_steps.values()),
        "new_miners_restore_step4_verified": len(restore_hashes) == 4
        and restore_hashes == replacement_checkpoint_events
        and all(
            worker.get("central_checkpoint_restore_verified") is True
            and worker.get("fresh_checkpoint_directory_before_restore") is True
            and worker.get("old_kernel_local_checkpoint_dependency") is False
            for worker in new_workers
        ),
        "entirely_new_miner_sessions_verified": len(old_sessions) == 2
        and len(new_sessions) == 2
        and old_sessions.isdisjoint(new_sessions),
        "stage_reassignment_to_new_miners_verified": len(replacement_assignments) == 4
        and {int(item["stage_id"]) for item in replacement_assignments} == {0, 1, 2, 3}
        and all(
            item.get("miner_id_hash") in new_miner_hashes
            for item in replacement_assignments
        ),
        "exactly_once_optimizer_commits_verified": int(
            final.get("optimizer_commit_count") or 0
        )
        == int(target_steps)
        and final.get("committed_steps") == list(range(1, int(target_steps) + 1))
        and final.get("committed_steps_contiguous") is True,
        "automatic_pause_wake_verified": any(
            item.get("operation") == "training_paused" for item in final.get("events") or []
        )
        and sum(
            item.get("operation") == "training_auto_woke"
            for item in final.get("events") or []
        )
        >= 2,
        "central_checkpoint_independent_of_old_kernels": old.get(
            "all_kernels_deleted"
        )
        is True
        and len(restore_hashes) == 4,
        "final_step8_completed": target_completed,
        "final_peft_export_evaluation_verified": final_role_b.get("export", {}).get(
            "standard_peft_format"
        )
        is True
        and final_role_b.get("evaluation", {}).get("evaluation_verified") is True,
        "four_distinct_kernel_refs_verified": len(old_hashes) == 2
        and len(new_hashes) == 2
        and old_hashes.isdisjoint(new_hashes),
        "real_cuda_only_verified": all(
            artifact.get("kaggle_kernel") is True
            and artifact.get("cuda_available") is True
            and int(artifact.get("cuda_device_count") or 0) >= 2
            for artifact in [*old["worker_reports"], *new["worker_reports"]]
        ),
        "rendezvous_full_pipeline_verified": len(
            [
                item
                for item in rendezvous.get("events") or []
                if item.get("run_kind") == "elastic"
                and item.get("operation") == "optimizer_step"
            ]
        )
        == int(target_steps) * 4,
    }
    if allow_deferred_evaluation:
        gates["final_peft_export_evaluation_verified"] = bool(
            final_role_b.get("export", {}).get("standard_peft_format") is True
            and final_role_b.get("evaluation", {}).get(
                "evaluation_deferred_to_isolated_benchmark"
            )
            is True
        )
    return {
        **gates,
        "old_miner_session_hashes": sorted(old_sessions),
        "new_miner_session_hashes": sorted(new_sessions),
        "old_step_sequences": old_steps,
        "new_step_sequences": new_steps,
        "target_steps": int(target_steps),
        "replacement_step": int(replacement_step),
        "final_target_step_completed": target_completed,
        "replacement_checkpoint_restore_hashes_match": restore_hashes
        == replacement_checkpoint_events,
        "step4_checkpoint_restore_hashes_match": restore_hashes
        == replacement_checkpoint_events,
        "verified": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token-file", action="append", default=[])
    parser.add_argument("--raw-token-file", default="")
    parser.add_argument("--raw-token-username", default="")
    parser.add_argument(
        "--tokenized-payload",
        default="dist/training-qwen15b-dataset-20260712-r1/qwen15b_tokenized_private.json",
    )
    parser.add_argument(
        "--source-manifest",
        default="dist/training-qwen15b-source-20260712-r1/qwen15b_source_manifest.json",
    )
    parser.add_argument("--allocation-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--push-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--output-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0)
    parser.add_argument("--pause-delay-seconds", type=float, default=10.0)
    parser.add_argument("--tunnel-attempts", type=int, default=3)
    parser.add_argument("--route-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--target-steps", type=int, default=8)
    parser.add_argument("--replacement-step", type=int, default=4)
    parser.add_argument("--microbatches-per-step", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=1800)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not 5.0 <= args.pause_delay_seconds <= 120.0:
        parser.error("--pause-delay-seconds must be in [5, 120]")
    if not 1.0 <= args.allocation_timeout_seconds <= 14400.0:
        parser.error("--allocation-timeout-seconds must be in (0, 14400]")
    if not 2 <= args.target_steps <= 2048:
        parser.error("--target-steps must be in [2, 2048]")
    if not 1 <= args.replacement_step < args.target_steps:
        parser.error("--replacement-step must be in [1, target_steps)")
    if not 1 <= args.microbatches_per_step <= 16:
        parser.error("--microbatches-per-step must be in [1, 16]")
    if not 0.0 < args.learning_rate <= 0.01:
        parser.error("--learning-rate must be in (0, 0.01]")
    if not 1 <= args.lora_rank <= 128 or not 1 <= args.lora_alpha <= 512:
        parser.error("LoRA rank/alpha outside bounded range")
    if not 300 <= args.kernel_timeout_seconds <= 43200:
        parser.error("--kernel-timeout-seconds must be in [300, 43200]")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    private = output / ".private-runtime"
    private.mkdir(parents=True, exist_ok=True)
    private.chmod(0o700)
    report_path = output / "training_qwen15b_elastic_live_probe.json"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "elastic_volunteer_training_ready": False,
        "live_run_performed": not args.preflight_only,
        "requested_model": MODEL_ID,
        "requested_model_revision": MODEL_REVISION,
        "requested_topology": "two-sequential-same-account-kaggle-t4x2-pairs",
        "parameter_count": MODEL_PARAMETER_COUNT,
        "target_steps": int(args.target_steps),
        "old_generation_end_step": int(args.replacement_step),
        "replacement_generation_start_step": int(args.replacement_step),
        "microbatches_per_step": int(args.microbatches_per_step),
        "learning_rate": float(args.learning_rate),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "blockers": [],
        "started_at": utc_now(),
        "cleanup": {
            "all_four_kernels_deleted": False,
            "coordinator_stopped": False,
            "tunnel_stopped": False,
            "private_runtime_removed": False,
            "rendezvous_payloads_removed": False,
            "uncommitted_checkpoint_blobs_removed": False,
            "live_resources_left_running": True,
        },
        "mock_runtime_used": False,
        "cpu_fallback_used": False,
        "tiny_or_random_model_used": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_tensor_values_public": False,
        "adapter_tensor_values_public": False,
        "token_ids_public": False,
        "raw_training_text_public": False,
        "credentials_public": False,
        "credential_paths_public": False,
        "coordinator_url_public": False,
        "session_tokens_public": False,
        "assignment_tokens_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    coordinator: ElasticQwenCoordinator | None = None
    tunnel_process = None
    selected_env_values: dict[str, str] = {}
    all_refs: list[str] = []
    old_report: dict[str, Any] = {}
    new_report: dict[str, Any] = {}
    try:
        sections = _credential_sections(
            list(args.token_file or []),
            raw_token_file=str(args.raw_token_file),
            raw_token_username=str(args.raw_token_username),
        )
        if not sections:
            raise RuntimeError("elastic_private_kaggle_credentials_required")
        preflight, candidates = preflight_accounts(sections)
        report["account_preflight"] = preflight
        report["eligible_account_count"] = len(candidates)
        if args.preflight_only:
            report["blockers"].append("elastic_preflight_only_no_live_run")
            return_code = 1
            return return_code
        if not candidates:
            raise RuntimeError("elastic_same_account_t4x2_pair_unavailable")
        selected = candidates[0]
        selected_env_values = dict(selected["env_values"])
        owner = str(selected["owner"])
        report["selected_account"] = {
            "owner_hash": str(selected["owner_hash"]),
            "candidate_index": int(selected["index"]),
            "effective_remaining_seconds": float(selected["effective_remaining"]),
            "credential_values_public": False,
        }
        source_manifest_path = Path(args.source_manifest).resolve()
        source_manifest = _load(source_manifest_path)
        if not source_manifest_path.is_file():
            raise RuntimeError("elastic_qwen_source_manifest_missing")
        config_bytes = fetch_bytes(_hf_url(MODEL_ID, MODEL_REVISION, "config.json"))
        if sha256_bytes(config_bytes) != source_manifest.get("config_hash"):
            raise RuntimeError("elastic_qwen_config_hash_mismatch")
        config = json.loads(config_bytes)
        tokenized = Path(args.tokenized_payload).resolve()
        if not tokenized.is_file():
            raise RuntimeError("elastic_qwen_tokenized_payload_missing")
        tokenized_payload = _load(tokenized)
        train_rows = list(tokenized_payload.get("train") or [])
        validation_rows = list(tokenized_payload.get("validation") or [])
        required_train_rows = int(args.target_steps) * int(args.microbatches_per_step)
        if len(train_rows) < required_train_rows or len(validation_rows) < 4:
            raise RuntimeError("elastic_qwen_tokenized_payload_training_budget_incomplete")
        sequence_length = int(tokenized_payload.get("sequence_length") or 0)
        if sequence_length < 1:
            raise RuntimeError("elastic_qwen_tokenized_payload_sequence_length_invalid")
        report["source"] = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "parameter_count": MODEL_PARAMETER_COUNT,
            "source_manifest_hash": sha256_file(source_manifest_path),
            "tokenized_payload_hash": sha256_file(tokenized),
            "dataset_id": str(tokenized_payload.get("dataset_id") or ""),
            "dataset_revision": str(tokenized_payload.get("dataset_revision") or ""),
            "sequence_length": sequence_length,
            "train_sequence_count": len(train_rows),
            "validation_sequence_count": len(validation_rows),
            "train_token_hash": stable_hash(train_rows),
            "validation_token_hash": stable_hash(validation_rows),
            "raw_samples_public": False,
            "token_ids_public": False,
        }
        report["training_budget"] = {
            "optimizer_steps": int(args.target_steps),
            "microbatches_per_step": int(args.microbatches_per_step),
            "sequence_length": sequence_length,
            "training_sequence_count": required_train_rows,
            "training_token_count": required_train_rows * sequence_length,
            "replacement_after_step": int(args.replacement_step),
        }
        run_id = f"qwen15b-elastic-{int(time.time())}-{secrets.token_hex(3)}"
        coordinator_token = secrets.token_urlsafe(32)
        port = _free_port()
        coordinator = ElasticQwenCoordinator(
            private_root=private,
            port=port,
            run_id=run_id,
            token=coordinator_token,
            target_steps=int(args.target_steps),
            microbatches_per_step=int(args.microbatches_per_step),
        )
        coordinator.start()
        local_url = f"http://127.0.0.1:{port}"
        cloudflared = ensure_cloudflared(private)
        tunnel_process, tunnel_url, route = start_verified_tunnel(
            cloudflared,
            local_url,
            private,
            token=coordinator_token,
            run_id=run_id,
            attempts=int(args.tunnel_attempts),
            timeout=float(args.route_timeout_seconds),
        )
        route_elastic = _get_json(
            f"{tunnel_url}/elastic-training/status", token=coordinator_token
        )
        report["route_preflight"] = {
            **route,
            "elastic_status_verified": route_elastic.get("schema")
            == "crowdtensor_elastic_training_status_v1",
        }
        if report["route_preflight"]["elastic_status_verified"] is not True:
            raise RuntimeError("elastic_tunnel_status_route_invalid")
        suffix = f"{str(int(time.time()))[-8:]}-{secrets.token_hex(2)}"
        with tempfile.TemporaryDirectory(prefix="ct-qwen-elastic-account-") as config_dir:
            from scripts.kaggle_gpu_token_weekly_quota_probe import clean_env

            env = clean_env(selected_env_values, config_dir=Path(config_dir))
            old_report, old_refs = _launch_generation(
                generation="old",
                expected_start_step=0,
                segment_end_step=int(args.replacement_step),
                target_steps=int(args.target_steps),
                microbatches_per_step=int(args.microbatches_per_step),
                owner=owner,
                suffix=suffix,
                config=config,
                tokenized=tokenized,
                coordinator_url=tunnel_url,
                coordinator_token=coordinator_token,
                run_id=run_id,
                env=env,
                private=private,
                output=output,
                allocation_timeout=float(args.allocation_timeout_seconds),
                push_timeout=float(args.push_timeout_seconds),
                status_timeout=float(args.status_timeout_seconds),
                output_timeout=float(args.output_timeout_seconds),
                delete_timeout=float(args.delete_timeout_seconds),
                poll_interval=float(args.poll_interval_seconds),
                cleanup_registry=all_refs,
                learning_rate=float(args.learning_rate),
                lora_rank=int(args.lora_rank),
                lora_alpha=int(args.lora_alpha),
                kernel_timeout_seconds=int(args.kernel_timeout_seconds),
            )
            report["old_generation"] = old_report
            if old_report.get("ok") is not True:
                raise RuntimeError("elastic_old_generation_acceptance_incomplete")
            midpoint = coordinator.runtime.public_status()
            report["midpoint_status"] = midpoint
            if not (
                int(midpoint.get("committed_step") or 0)
                == int(args.replacement_step)
                and midpoint.get("zero_live_miners") is True
                and midpoint.get("paused_waiting_for_miners") is True
                and old_report.get("all_kernels_deleted") is True
            ):
                raise RuntimeError("elastic_midpoint_pause_not_verified")
            report["midpoint_checkpoint_retention"] = (
                coordinator.runtime.enforce_checkpoint_retention()
            )
            report["midpoint_consumed_payload_cleanup"] = (
                coordinator.rendezvous.cleanup()
            )
            pause_started = time.monotonic()
            pause_observations = []
            while time.monotonic() - pause_started < float(args.pause_delay_seconds):
                current = coordinator.runtime.public_status()
                pause_observations.append(
                    {
                        "observed_at": utc_now(),
                        "runtime_state": current.get("runtime_state"),
                        "committed_step": current.get("committed_step"),
                        "live_miner_count": current.get("live_miner_count"),
                    }
                )
                time.sleep(
                    min(1.0, max(0.05, float(args.pause_delay_seconds) - (time.monotonic() - pause_started)))
                )
            pause_elapsed = time.monotonic() - pause_started
            report["full_offline_pause"] = {
                "requested_seconds": float(args.pause_delay_seconds),
                "observed_seconds": pause_elapsed,
                "observations": pause_observations,
                "new_kernel_launched_during_pause": False,
            }
            new_report, new_refs = _launch_generation(
                generation="new",
                expected_start_step=int(args.replacement_step),
                segment_end_step=int(args.target_steps),
                target_steps=int(args.target_steps),
                microbatches_per_step=int(args.microbatches_per_step),
                owner=owner,
                suffix=suffix,
                config=config,
                tokenized=tokenized,
                coordinator_url=tunnel_url,
                coordinator_token=coordinator_token,
                run_id=run_id,
                env=env,
                private=private,
                output=output,
                allocation_timeout=float(args.allocation_timeout_seconds),
                push_timeout=float(args.push_timeout_seconds),
                status_timeout=float(args.status_timeout_seconds),
                output_timeout=float(args.output_timeout_seconds),
                delete_timeout=float(args.delete_timeout_seconds),
                poll_interval=float(args.poll_interval_seconds),
                cleanup_registry=all_refs,
                learning_rate=float(args.learning_rate),
                lora_rank=int(args.lora_rank),
                lora_alpha=int(args.lora_alpha),
                kernel_timeout_seconds=int(args.kernel_timeout_seconds),
            )
            report["new_generation"] = new_report
            final_status = coordinator.runtime.public_status()
            rendezvous = coordinator.rendezvous.public_status()
            report["final_status"] = final_status
            report["rendezvous"] = rendezvous
            report["overlap"] = four_stage_overlap_summary(
                rendezvous.get("events") or []
            )
            evidence = _evaluate(
                old=old_report,
                new=new_report,
                midpoint=midpoint,
                final=final_status,
                rendezvous=rendezvous,
                pause_observations=pause_observations,
                pause_seconds=pause_elapsed,
                target_steps=int(args.target_steps),
                replacement_step=int(args.replacement_step),
            )
            report["evidence"] = evidence
            report["elastic_volunteer_training_ready"] = evidence["verified"]
            report["ok"] = evidence["verified"]
            if not report["ok"]:
                report["blockers"].append("elastic_live_acceptance_incomplete")
    except BaseException as exc:
        report["blockers"].append(str(exc).split(":", 1)[0][:180] or type(exc).__name__)
        report["error_class"] = type(exc).__name__
    finally:
        generation_cleanup_already_verified = bool(
            all_refs
            and old_report.get("all_kernels_deleted") is True
            and (
                not new_report
                or new_report.get("all_kernels_deleted") is True
            )
        )
        if generation_cleanup_already_verified:
            report["final_cleanup_deletions"] = []
            report["cleanup"]["all_four_kernels_deleted"] = True
        elif selected_env_values and all_refs:
            from scripts.kaggle_gpu_token_weekly_quota_probe import clean_env

            with tempfile.TemporaryDirectory(prefix="ct-qwen-elastic-final-cleanup-") as config_dir:
                env = clean_env(selected_env_values, config_dir=Path(config_dir))
                role_by_ref = {ref: "" for ref in all_refs}
                final_deletes, all_deleted = _delete_refs(
                    all_refs,
                    env=env,
                    timeout=float(args.delete_timeout_seconds),
                    role_by_ref=role_by_ref,
                )
                report["final_cleanup_deletions"] = final_deletes
                report["cleanup"]["all_four_kernels_deleted"] = all_deleted
        else:
            report["cleanup"]["all_four_kernels_deleted"] = not all_refs
        if coordinator is not None:
            report["final_checkpoint_retention"] = (
                coordinator.runtime.enforce_checkpoint_retention()
            )
            blob_cleanup = coordinator.runtime.cleanup_uncommitted_blobs()
            report["checkpoint_blob_cleanup"] = blob_cleanup
            report["cleanup"]["uncommitted_checkpoint_blobs_removed"] = bool(
                blob_cleanup.get("ok") is True
            )
            rendezvous_cleanup = coordinator.rendezvous.cleanup()
            report["rendezvous_cleanup"] = rendezvous_cleanup
            report["cleanup"]["rendezvous_payloads_removed"] = bool(
                rendezvous_cleanup.get("private_payloads_removed") is True
            )
            report["cleanup"]["coordinator_stopped"] = coordinator.stop()
        else:
            report["cleanup"]["uncommitted_checkpoint_blobs_removed"] = True
            report["cleanup"]["rendezvous_payloads_removed"] = True
            report["cleanup"]["coordinator_stopped"] = True
        report["cleanup"]["tunnel_stopped"] = stop_process(tunnel_process)
        shutil.rmtree(private, ignore_errors=True)
        report["cleanup"]["private_runtime_removed"] = not private.exists()
        report["cleanup"]["live_resources_left_running"] = not all(
            report["cleanup"].get(key) is True
            for key in (
                "all_four_kernels_deleted",
                "coordinator_stopped",
                "tunnel_stopped",
                "private_runtime_removed",
            )
        )
        cleanup_ok = bool(
            not report["cleanup"]["live_resources_left_running"]
            and report["cleanup"]["rendezvous_payloads_removed"]
            and report["cleanup"]["uncommitted_checkpoint_blobs_removed"]
        )
        safety = public_safety_errors(report)
        report["public_artifact_safe"] = not safety
        if safety:
            report["safety_errors"] = safety
        report["blockers"] = sorted(set(report.get("blockers") or []))
        report["ok"] = bool(report.get("ok") and cleanup_ok and not safety)
        report["elastic_volunteer_training_ready"] = bool(
            report.get("elastic_volunteer_training_ready") and report["ok"]
        )
        report["finished_at"] = utc_now()
        _write(report_path, report)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(
                "training_qwen15b_elastic_live_probe "
                f"ok={report['ok']} blockers={','.join(report['blockers']) or 'none'}"
            )
    return 0 if report["ok"] else (1 if report["public_artifact_safe"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
