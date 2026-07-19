#!/usr/bin/env python3
"""Run one bounded same-account dual-Kaggle-Kernel Qwen 1.5B Training Alpha attempt."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from coordinator import create_app  # noqa: E402
from crowdtensor.qwen15b_four_gpu_runtime import four_stage_overlap_summary  # noqa: E402
from crowdtensor.qwen15b_training import (  # noqa: E402
    MODEL_ID,
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
from scripts.kaggle_gpu_token_weekly_quota_probe import (  # noqa: E402
    clean_env,
    fetch_accelerator_quota,
    parse_raw_token_file,
    parse_token_sections,
)
from scripts.training_cuda_kaggle_common import (  # noqa: E402
    authenticated_owner,
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
    start_tunnel,
    stop_process,
)
from scripts.training_qwen15b_four_gpu_package import build_package  # noqa: E402


SCHEMA = "crowdtensor_qwen15b_four_gpu_live_probe_v1"
ALLOCATION_AMENDMENT_SCHEMA = (
    "crowdtensor_qwen15b_four_gpu_allocation_budget_amendment_v1"
)
UNBOUNDED_ALLOCATION_AMENDMENT_SCHEMA = (
    "crowdtensor_qwen15b_four_gpu_unbounded_allocation_budget_amendment_v1"
)
BETA_ALLOCATION_AUTHORIZATION_SCHEMA = (
    "crowdtensor_qwen15b_beta_goal_allocation_authorization_v1"
)
ORIGINAL_ATTEMPT_LIMIT = 2
MAX_AUTHORIZED_ATTEMPT_LIMIT = 3
WORKER_REPORT = "training_qwen15b_four_gpu_worker.json"
OUTPUT_PATTERN = (
    r"training_qwen15b_(four_gpu_worker\.json|kernel_[ab]_checkpoint_bundle\.zip|"
    r"standard_peft_adapter\.zip)"
)
TERMINAL = {"complete", "failed"}
REF_RE = re.compile(r"\b([a-z0-9-]+/[a-z0-9-]+)\b")


class PreflightOnlyComplete(RuntimeError):
    pass


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def allocation_budget_summary(ledger: dict[str, Any]) -> dict[str, Any]:
    attempts = list(ledger.get("qwen15b_four_gpu_attempts") or [])
    amendment = dict(ledger.get("allocation_budget_amendment") or {})
    authorization_hash = str(amendment.get("authorization_hash") or "")
    prior_attempts = attempts[:ORIGINAL_ATTEMPT_LIMIT]
    bounded_amendment_valid = bool(
        amendment.get("schema") == ALLOCATION_AMENDMENT_SCHEMA
        and amendment.get("authorized") is True
        and amendment.get("authorization_text_public") is False
        and amendment.get("same_authorized_account_only") is True
        and amendment.get("topology") == "kaggle-2x-t4x2"
        and int(amendment.get("original_attempt_limit") or 0) == ORIGINAL_ATTEMPT_LIMIT
        and int(amendment.get("additional_attempts") or 0) == 1
        and int(amendment.get("revised_attempt_limit") or 0)
        == MAX_AUTHORIZED_ATTEMPT_LIMIT
        and int(amendment.get("allocation_timeout_seconds") or 0) == 1800
        and int(amendment.get("prior_attempt_count") or 0) == ORIGINAL_ATTEMPT_LIMIT
        and amendment.get("prior_attempts_hash") == stable_hash(prior_attempts)
        and len(prior_attempts) == ORIGINAL_ATTEMPT_LIMIT
        and all(int(item.get("attempt") or 0) == index for index, item in enumerate(prior_attempts, 1))
        and bool(str(amendment.get("authorized_at") or ""))
        and re.fullmatch(r"sha256:[0-9a-f]{64}", authorization_hash) is not None
    )
    unbounded_prior_count = int(amendment.get("prior_attempt_count") or 0)
    unbounded_prior_attempts = attempts[:unbounded_prior_count]
    unbounded_amendment_valid = bool(
        amendment.get("schema") == UNBOUNDED_ALLOCATION_AMENDMENT_SCHEMA
        and amendment.get("authorized") is True
        and amendment.get("authorization_text_public") is False
        and amendment.get("same_authorized_account_only") is True
        and amendment.get("topology") == "kaggle-2x-t4x2"
        and amendment.get("total_attempt_limit_unbounded") is True
        and amendment.get("one_attempt_per_probe_invocation") is True
        and amendment.get("automatic_retry_loop") is False
        and int(amendment.get("allocation_timeout_seconds") or 0) == 1800
        and unbounded_prior_count >= ORIGINAL_ATTEMPT_LIMIT
        and len(unbounded_prior_attempts) == unbounded_prior_count
        and amendment.get("prior_attempts_hash") == stable_hash(unbounded_prior_attempts)
        and all(
            int(item.get("attempt") or 0) == index
            for index, item in enumerate(unbounded_prior_attempts, 1)
        )
        and bool(str(amendment.get("authorized_at") or ""))
        and re.fullmatch(r"sha256:[0-9a-f]{64}", authorization_hash) is not None
    )
    beta_authorization = dict(ledger.get("beta_goal_allocation_authorization") or {})
    beta_authorization_hash = str(beta_authorization.get("authorization_hash") or "")
    beta_authorization_valid = bool(
        beta_authorization.get("schema") == BETA_ALLOCATION_AUTHORIZATION_SCHEMA
        and beta_authorization.get("authorized") is True
        and beta_authorization.get("authorization_text_public") is False
        and beta_authorization.get("same_authorized_account_only") is True
        and beta_authorization.get("topology") == "kaggle-2x-t4x2"
        and int(beta_authorization.get("goal_attempt_limit") or 0) == 3
        and beta_authorization.get("one_attempt_per_probe_invocation") is True
        and beta_authorization.get("automatic_retry_loop") is False
        and int(beta_authorization.get("allocation_timeout_seconds") or 0) == 1800
        and bool(str(beta_authorization.get("authorized_at") or ""))
        and re.fullmatch(r"sha256:[0-9a-f]{64}", beta_authorization_hash) is not None
    )
    amendment_valid = (
        bounded_amendment_valid or unbounded_amendment_valid or beta_authorization_valid
    )
    preserved_attempts = (
        unbounded_prior_attempts if unbounded_amendment_valid else prior_attempts
    )
    return {
        "schema": "crowdtensor_qwen15b_four_gpu_allocation_budget_summary_v1",
        "amendment_present": bool(amendment),
        "amendment_valid": amendment_valid,
        "original_attempt_limit": ORIGINAL_ATTEMPT_LIMIT,
        "effective_attempt_limit": (
            None
            if unbounded_amendment_valid
            else 3
            if beta_authorization_valid
            else MAX_AUTHORIZED_ATTEMPT_LIMIT
            if bounded_amendment_valid
            else ORIGINAL_ATTEMPT_LIMIT
        ),
        "total_attempt_limit_unbounded": unbounded_amendment_valid,
        "one_attempt_per_probe_invocation": unbounded_amendment_valid,
        "automatic_retry_loop": False,
        "additional_attempts_authorized": (
            None
            if unbounded_amendment_valid
            else 3
            if beta_authorization_valid
            else 1
            if bounded_amendment_valid
            else 0
        ),
        "prior_attempts_preserved": bool(
            preserved_attempts
            and amendment.get("prior_attempts_hash") == stable_hash(preserved_attempts)
        ),
        "same_authorized_account_only": amendment_valid,
        "beta_goal_authorization": beta_authorization_valid,
        "beta_goal_attempt_limit": 3 if beta_authorization_valid else 0,
        "allocation_timeout_seconds": 1800,
        "authorization_hash": (
            beta_authorization_hash
            if beta_authorization_valid
            else authorization_hash
            if amendment_valid
            else ""
        ),
        "authorization_text_public": False,
        "credential_values_public": False,
        "public_artifact_safe": True,
    }


def reserve_attempt(ledger_path: Path, *, limit: int = 2) -> int:
    if int(limit) < 1:
        raise ValueError("Qwen Alpha allocation attempt limit must be positive")
    ledger = _load(ledger_path)
    attempts = list(ledger.get("qwen15b_four_gpu_attempts") or [])
    budget = allocation_budget_summary(ledger)
    if budget.get("total_attempt_limit_unbounded") is not True and int(limit) > int(
        budget["effective_attempt_limit"]
    ):
        raise RuntimeError("qwen15b_four_gpu_allocation_attempt_limit_not_authorized")
    if len(attempts) >= int(limit):
        raise RuntimeError("qwen15b_four_gpu_allocation_attempt_limit_reached")
    attempt = len(attempts) + 1
    attempts.append(
        {
            "attempt": attempt,
            "started_at": utc_now(),
            "allocation_started": True,
            "completed": False,
        }
    )
    ledger.update(
        {
            "schema": "crowdtensor_qwen15b_four_gpu_allocation_ledger_v1",
            "qwen15b_four_gpu_attempts": attempts,
            "attempt_limit": int(limit),
            "allocation_budget_summary": budget,
        }
    )
    _write(ledger_path, ledger)
    return attempt


def finish_attempt(ledger_path: Path, *, attempt: int, outcome: str) -> None:
    ledger = _load(ledger_path)
    attempts = list(ledger.get("qwen15b_four_gpu_attempts") or [])
    for value in attempts:
        if int(value.get("attempt") or 0) == int(attempt):
            value.update({"completed": True, "finished_at": utc_now(), "outcome": str(outcome)})
    ledger["qwen15b_four_gpu_attempts"] = attempts
    _write(ledger_path, ledger)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _get_json(
    url: str,
    *,
    token: str = "",
    timeout: float = 15.0,
) -> dict[str, Any]:
    headers = {"User-Agent": "crowdtensor-qwen15b-route-preflight/1"}
    if token:
        headers["x-crowdtensor-miner-token"] = token
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("qwen15b_route_response_invalid")
    return value


def _wait_local_ready(url: str, *, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        try:
            if _get_json(f"{url}/ready", timeout=5.0).get("ok") is True:
                return
        except BaseException:
            pass
        time.sleep(0.5)
    raise TimeoutError("qwen15b_local_coordinator_readiness_timeout")


class RestartableQwenCoordinator:
    def __init__(
        self,
        *,
        private_root: Path,
        port: int,
        run_id: str,
        token: str,
    ) -> None:
        self.private_root = private_root
        self.port = int(port)
        self.run_id = str(run_id)
        self.token = str(token)
        self.state_path = private_root / "rendezvous-state.json"
        self.server: Any = None
        self.thread: threading.Thread | None = None
        self.rendezvous: Qwen15BTrainingRendezvous | None = None
        self._lock = threading.RLock()

    def _authorize(self, value: str | None) -> None:
        from fastapi import HTTPException

        if value is None or not hmac.compare_digest(value, self.token):
            raise HTTPException(status_code=401, detail="unauthorized")

    def start(self, *, recovered: bool = False) -> None:
        import uvicorn

        with self._lock:
            rendezvous = Qwen15BTrainingRendezvous(
                run_id=self.run_id,
                state_path=self.state_path,
            )
            if recovered:
                rendezvous.complete_coordinator_restart()
            app = create_app(
                state_dir=self.private_root / "coordinator-state",
                lease_seconds=1800.0,
                backlog=0,
                task_lanes=[],
                miner_token=self.token,
            )
            install_qwen15b_training_routes(
                app,
                rendezvous=rendezvous,
                authorize=self._authorize,
            )
            app.state.qwen15b_training_rendezvous = rendezvous
            config = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
                access_log=False,
            )
            server = uvicorn.Server(config)
            server.install_signal_handlers = lambda: None
            thread = threading.Thread(
                target=server.run,
                name="qwen15b-training-coordinator",
                daemon=True,
            )
            self.rendezvous = rendezvous
            self.server = server
            self.thread = thread
            thread.start()
        _wait_local_ready(f"http://127.0.0.1:{self.port}")

    def stop(self) -> bool:
        with self._lock:
            server = self.server
            thread = self.thread
            if server is not None:
                server.should_exit = True
        if thread is not None:
            thread.join(timeout=20.0)
        stopped = bool(thread is None or not thread.is_alive())
        with self._lock:
            self.server = None
            self.thread = None
        return stopped

    def restart(self, *, after_step: int, downtime_seconds: float) -> dict[str, Any]:
        with self._lock:
            if self.rendezvous is None:
                raise RuntimeError("qwen15b_coordinator_not_started")
            before = self.rendezvous.begin_coordinator_restart(after_step=after_step)
        if not self.stop():
            raise RuntimeError("qwen15b_coordinator_restart_stop_failed")
        time.sleep(float(downtime_seconds))
        self.start(recovered=True)
        with self._lock:
            assert self.rendezvous is not None
            status = self.rendezvous.public_status()
        restart = list(status.get("coordinator_restarts") or [])[-1]
        return {
            "schema": "crowdtensor_qwen15b_coordinator_restart_v1",
            "verified": bool(
                int(restart.get("after_step") or 0) == int(after_step)
                and float(restart.get("completed_at") or 0)
                > float(restart.get("started_at") or 0)
            ),
            "generation": int(restart.get("generation") or 0),
            "after_step": int(restart.get("after_step") or 0),
            "downtime_seconds_requested": float(downtime_seconds),
            "duration_seconds": float(restart.get("duration_seconds") or 0),
            "recovered_payload_count": int(before.get("recovered_payload_count") or 0),
            "recovered_event_count": int(before.get("recovered_event_count") or 0),
            "coordinator_url_public": False,
            "coordinator_token_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }


def _probe_remote_route(
    url: str,
    *,
    token: str,
    run_id: str,
) -> dict[str, Any]:
    result = {
        "ready": False,
        "auth_required": False,
        "status": False,
        "run_hash": False,
        "ok": False,
    }
    try:
        ready = _get_json(f"{url}/ready")
        result["ready"] = ready.get("ok") is True
        result["auth_required"] = (ready.get("auth") or {}).get("miner_required") is True
        status = _get_json(f"{url}/qwen15b-training/status", token=token)
        result["status"] = status.get("schema") == (
            "crowdtensor_qwen15b_four_gpu_rendezvous_v1"
        )
        result["run_hash"] = status.get("run_id_hash") == sha256_bytes(run_id.encode("utf-8"))
        result["ok"] = all(
            result[key] for key in ("ready", "auth_required", "status", "run_hash")
        )
    except BaseException as exc:
        result["error_class"] = type(exc).__name__
    return result


def start_verified_tunnel(
    binary: Path,
    local_url: str,
    private_dir: Path,
    *,
    token: str,
    run_id: str,
    attempts: int,
    timeout: float,
) -> tuple[Any, str, dict[str, Any]]:
    diagnostics = []
    for attempt in range(1, int(attempts) + 1):
        process = None
        try:
            process, tunnel_url, _log = start_tunnel(
                binary,
                local_url,
                private_dir,
                log_name=f"qwen15b-cloudflared-{attempt}.log",
            )
            deadline = time.monotonic() + float(timeout)
            consecutive = 0
            observations = 0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                observation = _probe_remote_route(
                    tunnel_url,
                    token=token,
                    run_id=run_id,
                )
                observations += 1
                consecutive = consecutive + 1 if observation.get("ok") else 0
                if consecutive >= 2:
                    return process, tunnel_url, {
                        "verified": True,
                        "tunnel_attempt": attempt,
                        "observation_count": observations,
                        "stable_success_count": consecutive,
                        "url_hash": sha256_bytes(tunnel_url.encode("utf-8")),
                        "url_public": False,
                        "credentials_public": False,
                    }
                time.sleep(2.0)
            diagnostics.append(
                {"attempt": attempt, "blocker": "authenticated_route_not_stable"}
            )
        except BaseException as exc:
            diagnostics.append({"attempt": attempt, "blocker": type(exc).__name__})
        stop_process(process)
    raise RuntimeError(
        "qwen15b_authenticated_tunnel_unavailable:"
        + stable_hash(diagnostics)
    )


def _credential_sections(
    token_files: list[str],
    *,
    raw_token_file: str,
    raw_token_username: str,
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for value in token_files:
        path = Path(value).expanduser()
        if path.is_file():
            sections.extend(parse_token_sections(path))
    if raw_token_file and Path(raw_token_file).expanduser().is_file():
        sections.append(
            parse_raw_token_file(
                Path(raw_token_file).expanduser(),
                username_hint=raw_token_username,
                label=raw_token_username or "dedicated-gpu-account",
            )
        )
    return sections


def _listed_refs(output: str, *, limit: int = 20) -> list[str]:
    refs = []
    for line in str(output or "").splitlines():
        match = re.match(r"\s*([a-z0-9-]+/[a-z0-9-]+)\s+", line.strip())
        if match and match.group(1) not in refs:
            refs.append(match.group(1))
            if len(refs) >= int(limit):
                break
    return refs


def preflight_accounts(sections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public: list[dict[str, Any]] = []
    private_candidates: list[dict[str, Any]] = []
    for index, section in enumerate(sections):
        with tempfile.TemporaryDirectory(prefix="ct-qwen15b-account-preflight-") as config_dir:
            env = clean_env(dict(section.get("env") or {}), config_dir=Path(config_dir))
            quota = fetch_accelerator_quota(env)
            owner = authenticated_owner(env)
            listing = run_command(
                ["kaggle", "kernels", "list", "--mine", "--page-size", "20", "--sort-by", "dateRun"],
                env=env,
                timeout=60.0,
            )
            refs = _listed_refs(str(listing.get("output_tail") or ""))
            active = 0
            statuses = {"running": 0, "queued": 0, "complete": 0, "failed": 0, "unknown": 0}
            for ref in refs:
                status = run_command(
                    ["kaggle", "kernels", "status", ref],
                    env=env,
                    timeout=30.0,
                )
                status_value = status_class(str(status.get("output_tail") or ""))
                statuses[status_value] = statuses.get(status_value, 0) + 1
                active += int(status_value in {"running", "queued"})
            gpu = dict(quota.get("gpu_quota") or {}) if quota.get("ok") else {}
            effective = float(gpu.get("effective_remaining_after_reserved_seconds") or 0.0)
            exhausted = bool(gpu.get("quota_exhausted_by_used") or gpu.get("reserved_exceeds_remaining"))
            authenticated = bool(owner and listing.get("ok"))
            eligible = bool(authenticated and quota.get("ok") and not exhausted and effective >= 3600 and active == 0)
            owner_hash = stable_hash({"owner": owner}) if owner else ""
            summary = {
                "candidate_index": index,
                "authenticated": authenticated,
                "owner_hash": owner_hash,
                "quota_api_ok": quota.get("ok") is True,
                "quota_refresh_time": quota.get("quota_refresh_time", ""),
                "gpu_effective_remaining_seconds": effective,
                "gpu_quota_exhausted": exhausted,
                "recent_kernel_status_counts": statuses,
                "active_kernel_count": active,
                "two_gpu_slots_read_only_preflight": eligible,
                "credential_values_public": False,
                "credential_paths_public": False,
            }
            public.append(summary)
            if eligible:
                private_candidates.append(
                    {
                        "index": index,
                        "env_values": dict(section.get("env") or {}),
                        "owner": owner,
                        "owner_hash": owner_hash,
                        "effective_remaining": effective,
                    }
                )
    private_candidates.sort(key=lambda item: (-item["effective_remaining"], item["index"]))
    return public, private_candidates


def inspect_checkpoint_bundle(path: str | Path, worker_bundle: dict[str, Any]) -> dict[str, Any]:
    source = Path(path)
    report = {
        "preserved": source.is_file(),
        "worker_hash_match": False,
        "archive_safe": False,
        "checkpoint_manifest_count": 0,
        "all_checkpoint_files_hash_verified": False,
        "all_final_steps_verified": False,
        "model_revision_verified": False,
        "manifest_summaries": [],
        "unique_archive_members": False,
        "file_hash": sha256_file(source) if source.is_file() else "",
        "byte_count": source.stat().st_size if source.is_file() else 0,
        "checkpoint_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    if not source.is_file():
        return report
    report["worker_hash_match"] = report["file_hash"] == worker_bundle.get("file_hash")
    with zipfile.ZipFile(source, "r") as archive:
        names = archive.namelist()
        unique_members = len(names) == len(set(names))
        safe = all(
            name and not name.startswith(("/", "\\")) and ".." not in Path(name).parts
            for name in names
        )
        manifests = [name for name in names if name.endswith("_checkpoint.json")]
        parsed = [json.loads(archive.read(name)) for name in manifests]
        file_hashes_ok = True
        summaries = []
        for name, manifest in zip(manifests, parsed, strict=True):
            parent = str(Path(name).parent)
            parts = Path(name).parts
            run_kind = next(
                (value for value in parts if value in {"baseline", "resumed"}),
                "",
            )
            component_hashes_present = True
            for file_key, hash_key in (
                ("adapter_file", "adapter_file_hash"),
                ("optimizer_file", "optimizer_file_hash"),
                ("grad_scaler_file", "grad_scaler_file_hash"),
                ("rng_file", "rng_file_hash"),
            ):
                member = str(Path(parent) / str(manifest.get(file_key) or ""))
                if member not in names or sha256_bytes(archive.read(member)) != manifest.get(hash_key):
                    file_hashes_ok = False
                    component_hashes_present = False
            summaries.append(
                {
                    "run_kind": run_kind,
                    "stage_id": int(manifest.get("stage_id", -1)),
                    "layer_start": int(manifest.get("layer_start", -1)),
                    "layer_end": int(manifest.get("layer_end", -1)),
                    "global_step": int(manifest.get("global_step") or 0),
                    "optimizer_step": int(manifest.get("optimizer_step") or 0),
                    "dataset_cursor": int(manifest.get("dataset_cursor") or 0),
                    "device": str(manifest.get("device") or ""),
                    "model_id": str(manifest.get("model_id") or ""),
                    "model_revision": str(manifest.get("model_revision") or ""),
                    "component_hashes_verified": component_hashes_present,
                    "grad_scaler_state_present": manifest.get("grad_scaler_state_present") is True,
                    "rng_state_present": manifest.get("rng_state_present") is True,
                    "adapter_tensor_count": int(manifest.get("adapter_tensor_count") or 0),
                    "adapter_tensor_hash": str(manifest.get("adapter_tensor_hash") or ""),
                    "manifest_content_hash": str(manifest.get("content_hash") or ""),
                    "manifest_content_hash_verified": manifest.get("content_hash")
                    == stable_hash(
                        {
                            key: value
                            for key, value in manifest.items()
                            if key != "content_hash"
                        }
                    ),
                }
            )
        report.update(
            {
                "archive_safe": safe,
                "unique_archive_members": unique_members,
                "archive_member_count": len(names),
                "checkpoint_manifest_count": len(manifests),
                "all_checkpoint_files_hash_verified": file_hashes_ok and bool(manifests),
                "all_final_steps_verified": bool(parsed)
                and all(int(item.get("global_step") or 0) == 8 for item in parsed),
                "model_revision_verified": bool(parsed)
                and all(
                    item.get("model_id") == MODEL_ID
                    and item.get("model_revision") == MODEL_REVISION
                    for item in parsed
                ),
                "all_manifest_content_hashes_verified": bool(summaries)
                and all(item["manifest_content_hash_verified"] for item in summaries),
                "manifest_summaries": sorted(
                    summaries,
                    key=lambda item: (item["run_kind"], item["stage_id"]),
                ),
            }
        )
    report["verified"] = bool(
        report["preserved"]
        and report["worker_hash_match"]
        and report["archive_safe"]
        and report["unique_archive_members"]
        and report["checkpoint_manifest_count"] == 4
        and report["all_checkpoint_files_hash_verified"]
        and report["all_final_steps_verified"]
        and report["model_revision_verified"]
        and report["all_manifest_content_hashes_verified"]
    )
    return report


def inspect_adapter_bundle(
    path: str | Path,
    worker_bundle: dict[str, Any],
    *,
    expected_model_id: str = MODEL_ID,
    expected_model_revision: str = MODEL_REVISION,
    expected_layer_count: int = 28,
) -> dict[str, Any]:
    source = Path(path)
    report = {
        "preserved": source.is_file(),
        "worker_hash_match": False,
        "standard_peft_layout": False,
        "archive_safe": False,
        "unique_archive_members": False,
        "safetensors_header_verified": False,
        "adapter_tensor_count": 0,
        "layer_indexes": [],
        "file_hash": sha256_file(source) if source.is_file() else "",
        "byte_count": source.stat().st_size if source.is_file() else 0,
        "adapter_tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    if not source.is_file():
        return report
    report["worker_hash_match"] = report["file_hash"] == worker_bundle.get("file_hash")
    with zipfile.ZipFile(source, "r") as archive:
        names = archive.namelist()
        report["unique_archive_members"] = len(names) == len(set(names))
        report["archive_safe"] = all(
            name and not name.startswith(("/", "\\")) and ".." not in Path(name).parts
            for name in names
        )
        report["standard_peft_layout"] = {
            "adapter_model.safetensors",
            "adapter_config.json",
        }.issubset(set(names))
        if report["standard_peft_layout"]:
            config_bytes = archive.read("adapter_config.json")
            adapter_bytes = archive.read("adapter_model.safetensors")
            config = json.loads(config_bytes)
            report["base_model_verified"] = (
                config.get("base_model_name_or_path") == expected_model_id
            )
            report["model_revision_verified"] = (
                config.get("revision") == expected_model_revision
            )
            report["adapter_config_hash"] = sha256_bytes(config_bytes)
            report["adapter_file_hash"] = sha256_bytes(adapter_bytes)
            report["adapter_config_byte_count"] = len(config_bytes)
            report["adapter_file_byte_count"] = len(adapter_bytes)
            if len(adapter_bytes) >= 8:
                header_length = int.from_bytes(adapter_bytes[:8], "little")
                if 0 < header_length <= len(adapter_bytes) - 8:
                    header = json.loads(adapter_bytes[8 : 8 + header_length])
                    tensor_entries = {
                        str(name): value
                        for name, value in header.items()
                        if name != "__metadata__" and isinstance(value, dict)
                    }
                    offsets_valid = bool(tensor_entries) and all(
                        isinstance(value.get("shape"), list)
                        and isinstance(value.get("data_offsets"), list)
                        and len(value["data_offsets"]) == 2
                        and int(value["data_offsets"][1]) >= int(value["data_offsets"][0])
                        for value in tensor_entries.values()
                    )
                    data_byte_count = len(adapter_bytes) - 8 - header_length
                    ranges = sorted(
                        (
                            int(value["data_offsets"][0]),
                            int(value["data_offsets"][1]),
                        )
                        for value in tensor_entries.values()
                        if isinstance(value.get("data_offsets"), list)
                        and len(value["data_offsets"]) == 2
                    )
                    offsets_valid = bool(
                        offsets_valid
                        and len(ranges) == len(tensor_entries)
                        and all(0 <= start <= end <= data_byte_count for start, end in ranges)
                        and all(
                            ranges[index - 1][1] <= ranges[index][0]
                            for index in range(1, len(ranges))
                        )
                    )
                    layer_indexes = sorted(
                        {
                            int(match.group(1))
                            for name in tensor_entries
                            if (match := re.search(r"\.layers\.(\d+)\.", name)) is not None
                        }
                    )
                    report["safetensors_header_verified"] = offsets_valid
                    report["adapter_tensor_count"] = len(tensor_entries)
                    report["adapter_tensor_names_hash"] = stable_hash(sorted(tensor_entries))
                    report["layer_indexes"] = layer_indexes
    report["verified"] = bool(
        report["preserved"]
        and report["worker_hash_match"]
        and report["archive_safe"]
        and report["unique_archive_members"]
        and report["standard_peft_layout"]
        and report.get("base_model_verified")
        and report.get("model_revision_verified")
        and report["safetensors_header_verified"]
        and report["layer_indexes"] == list(range(int(expected_layer_count)))
    )
    return report


def _extract_adapter(path: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as archive:
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            (destination / name).write_bytes(archive.read(name))


def _preserve_public_kernel_log(private_output: Path, destination: Path) -> dict[str, Any]:
    logs = sorted(private_output.glob("*.log"))
    if not logs:
        return {"present": False, "public_artifact_safe": True}
    text = logs[-1].read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"https?://[^\s]+", "<private-url>", text)
    text = re.sub(r"/(?:root|tmp|home|kaggle)/[^\s]+", "<private-path>", text)
    text = re.sub(
        r"(?i)(token|authorization|cookie|kaggle[_-]?(?:key|api_token))[=:]\S+",
        r"\1=<redacted>",
        text,
    )
    text = re.sub(r"KGA[A-Za-z0-9_-]{8,}", "KGA<redacted>", text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return {
        "present": True,
        "file_name": destination.name,
        "file_hash": sha256_file(destination),
        "byte_count": destination.stat().st_size,
        "tail": text[-4000:],
        "credentials_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def evaluate_live_evidence(
    *,
    workers: list[dict[str, Any]],
    rendezvous: dict[str, Any],
    checkpoint_bundles: list[dict[str, Any]],
    adapter_bundle: dict[str, Any],
    max_running: int,
) -> dict[str, Any]:
    by_role = {str(item.get("role") or ""): item for item in workers}
    payloads = list(rendezvous.get("payloads") or [])
    events = [
        event
        for worker in workers
        for run in (worker.get("worker") or {}).get("runs", {}).values()
        for event in run.get("events") or []
    ]
    overlap = four_stage_overlap_summary(events)
    stage_ids = {
        int(ready.get("stage_id", -1))
        for worker in workers
        for values in (worker.get("worker") or {}).get("stage_ready", {}).values()
        for ready in values
    }
    devices_by_role = {
        role: {
            str(ready.get("device") or "")
            for values in (worker.get("worker") or {}).get("stage_ready", {}).values()
            for ready in values
        }
        for role, worker in by_role.items()
    }
    coordinator_recoveries = [
        item
        for worker in workers
        for item in ((worker.get("worker") or {}).get("coordinator_restart_stage_recoveries") or [])
    ]
    transport_reports = {
        role: dict((worker.get("worker") or {}).get("transport_reliability") or {})
        for role, worker in by_role.items()
    }
    optimizer_records = [
        (run_kind, int(stage.get("stage_id", -1)), int(step.get("step") or 0))
        for worker in workers
        for run_kind, run in ((worker.get("worker") or {}).get("runs") or {}).items()
        for step in run.get("step_reports") or []
        for stage in step.get("stages") or []
    ]
    result = {
        "same_account_two_kernel_reports": set(by_role) == {"kernel_a", "kernel_b"},
        "both_workers_ok": len(by_role) == 2 and all(item.get("ok") is True for item in by_role.values()),
        "all_four_stage_ids": stage_ids == {0, 1, 2, 3},
        "two_devices_per_kernel": all(
            devices_by_role.get(role) == {"cuda:0", "cuda:1"}
            for role in ("kernel_a", "kernel_b")
        ),
        "both_runs_eight_steps": len(by_role) == 2
        and all(
            all(
                int(run.get("steps_completed") or 0) == 8
                for run in (worker.get("worker") or {}).get("runs", {}).values()
            )
            for worker in by_role.values()
        ),
        "activation_payload_count": sum(item.get("kind") == "activation" for item in payloads),
        "gradient_payload_count": sum(item.get("kind") == "gradient" for item in payloads),
        "stage_adapter_payload_count": sum(item.get("kind") == "stage_adapter" for item in payloads),
        "controlled_restart_verified": bool(
            ((by_role.get("kernel_b") or {}).get("worker") or {}).get(
                "controlled_restart_verified"
            )
        ),
        "resume_adapter_equivalence_verified": len(by_role) == 2
        and all(
            ((item.get("worker") or {}).get("resume_adapter_equivalence") or {}).get(
                "verified"
            )
            is True
            for item in by_role.values()
        ),
        "resume_loss_equivalence_verified": bool(
            (((by_role.get("kernel_b") or {}).get("worker") or {}).get(
                "resume_loss_equivalence"
            ) or {}).get("verified")
        ),
        "loss_reduced": bool(
            all(
                run.get("loss_reduced") is True
                for run in (((by_role.get("kernel_b") or {}).get("worker") or {}).get(
                    "runs"
                ) or {}).values()
            )
        ),
        "evaluation_verified": bool(
            ((((by_role.get("kernel_b") or {}).get("worker") or {}).get("evaluation") or {}).get(
                "evaluation_verified"
            ))
        ),
        "standard_peft_export_verified": bool(
            ((((by_role.get("kernel_b") or {}).get("worker") or {}).get("export") or {}).get(
                "standard_peft_format"
            ))
        ),
        "checkpoint_archives_verified": len(checkpoint_bundles) == 2
        and all(item.get("verified") is True for item in checkpoint_bundles),
        "adapter_archive_verified": adapter_bundle.get("verified") is True,
        "coordinator_restart_verified": rendezvous.get("coordinator_restart_verified") is True,
        "post_restart_registered_roles": list(
            rendezvous.get("post_restart_registered_roles") or []
        ),
        "all_four_stages_checkpoint_recovered_after_coordinator_restart": {
            int(item.get("stage_id", -1)) for item in coordinator_recoveries
        }
        == {0, 1, 2, 3},
        "coordinator_restart_recovery_count": len(coordinator_recoveries),
        "bounded_transport_retry_verified": set(transport_reports) == {"kernel_a", "kernel_b"}
        and all(item.get("bounded_retry_enabled") is True for item in transport_reports.values()),
        "post_restart_transport_reregistration_verified": set(transport_reports)
        == {"kernel_a", "kernel_b"}
        and all(
            int(item.get("reconnect_registration_count") or 0) >= 1
            for item in transport_reports.values()
        ),
        "optimizer_step_identity_count": len(optimizer_records),
        "optimizer_steps_unique": len(optimizer_records) == len(set(optimizer_records)) == 64,
        "max_observed_running_kernel_count": int(max_running),
        **overlap,
    }
    result["verified"] = bool(
        result["same_account_two_kernel_reports"]
        and result["both_workers_ok"]
        and result["all_four_stage_ids"]
        and result["two_devices_per_kernel"]
        and result["both_runs_eight_steps"]
        and result["activation_payload_count"] == 64
        and result["gradient_payload_count"] == 64
        and result["stage_adapter_payload_count"] == 1
        and result["controlled_restart_verified"]
        and result["resume_adapter_equivalence_verified"]
        and result["resume_loss_equivalence_verified"]
        and result["loss_reduced"]
        and result["evaluation_verified"]
        and result["standard_peft_export_verified"]
        and result["checkpoint_archives_verified"]
        and result["adapter_archive_verified"]
        and int(max_running) >= 2
        and result["four_stage_compute_overlap_verified"]
    )
    result["beta_recovery_verified"] = bool(
        result["coordinator_restart_verified"]
        and result["post_restart_registered_roles"] == ["kernel_a", "kernel_b"]
        and result["all_four_stages_checkpoint_recovered_after_coordinator_restart"]
        and result["coordinator_restart_recovery_count"] == 4
        and result["bounded_transport_retry_verified"]
        and result["post_restart_transport_reregistration_verified"]
        and result["optimizer_steps_unique"]
    )
    return result


def build_beta_benchmark(
    *,
    workers: list[dict[str, Any]],
    rendezvous: dict[str, Any],
    attempt_started_epoch: float,
    attempt_elapsed_seconds: float,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    compute_events = [
        item
        for worker in workers
        for run in ((worker.get("worker") or {}).get("runs") or {}).values()
        for item in run.get("events") or []
        if int(item.get("ended_ns") or 0) > int(item.get("started_ns") or 0)
    ]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in compute_events:
        grouped.setdefault(
            (str(item.get("run_kind") or ""), int(item.get("step", -1))), []
        ).append(item)
    step_latencies = []
    for (run_kind, step), items in sorted(grouped.items()):
        started = min(int(item["started_ns"]) for item in items)
        ended = max(int(item["ended_ns"]) for item in items)
        step_latencies.append(
            {
                "run_kind": run_kind,
                "step": step + 1,
                "latency_ms": round((ended - started) / 1_000_000.0, 6),
            }
        )
    optimizer_events = [
        item
        for item in rendezvous.get("events") or []
        if item.get("operation") == "optimizer_step"
    ]
    first_optimizer_at = min(
        (float(item.get("at") or 0) for item in optimizer_events if item.get("at")),
        default=0.0,
    )
    peak_allocated = max(
        (
            int(stage.get("peak_allocated_bytes") or 0)
            for worker in workers
            for run in ((worker.get("worker") or {}).get("runs") or {}).values()
            for step in run.get("step_reports") or []
            for stage in step.get("stages") or []
        ),
        default=0,
    )
    payloads = list(rendezvous.get("payloads") or [])
    restart = (list(rendezvous.get("coordinator_restarts") or []) or [{}])[-1]
    benchmark = {
        "schema": "crowdtensor_training_qwen15b_beta_benchmark_v1",
        "deployment_and_training_seconds": round(float(attempt_elapsed_seconds), 6),
        "completed_within_1800_seconds": float(attempt_elapsed_seconds) <= 1800.0,
        "first_optimizer_step_seconds": (
            round(first_optimizer_at - float(attempt_started_epoch), 6)
            if first_optimizer_at >= attempt_started_epoch
            else None
        ),
        "step_latencies": step_latencies,
        "step_latency_count": len(step_latencies),
        "maximum_step_latency_ms": max(
            (float(item["latency_ms"]) for item in step_latencies), default=0.0
        ),
        "maximum_four_stage_overlap_ms": round(
            int((evidence.get("maximum_four_stage_overlap") or {}).get("duration_ns") or 0)
            / 1_000_000.0,
            6,
        ),
        "private_network_payload_count": len(payloads),
        "private_network_bytes": sum(int(item.get("byte_count") or 0) for item in payloads),
        "peak_gpu_allocated_bytes": peak_allocated,
        "coordinator_recovery_seconds": float(restart.get("duration_seconds") or 0),
        "generated_token_count": 0,
        "raw_tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    benchmark["benchmark_complete"] = bool(
        benchmark["completed_within_1800_seconds"]
        and benchmark["first_optimizer_step_seconds"] is not None
        and benchmark["step_latency_count"] == 16
        and benchmark["maximum_four_stage_overlap_ms"] > 0
        and benchmark["private_network_payload_count"] == 129
        and benchmark["private_network_bytes"] > 0
        and benchmark["peak_gpu_allocated_bytes"] > 0
        and benchmark["coordinator_recovery_seconds"] > 0
    )
    return benchmark


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
    parser.add_argument("--attempt-ledger", required=True)
    parser.add_argument("--attempt-limit", type=int, default=2)
    parser.add_argument("--allocation-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--push-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--output-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--tunnel-attempts", type=int, default=3)
    parser.add_argument("--route-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--coordinator-restart-after-step", type=int, choices=[0, 4], default=0)
    parser.add_argument("--coordinator-restart-downtime-seconds", type=float, default=3.0)
    parser.add_argument("--cancel-file", default="")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.attempt_limit < 1:
        parser.error("--attempt-limit must be positive")
    if args.allocation_timeout_seconds <= 0 or args.allocation_timeout_seconds > 1800:
        parser.error("--allocation-timeout-seconds must be in (0, 1800]")
    if args.tunnel_attempts < 1 or args.tunnel_attempts > 3:
        parser.error("--tunnel-attempts must be in [1, 3]")
    if (
        args.coordinator_restart_downtime_seconds < 0.5
        or args.coordinator_restart_downtime_seconds > 30
    ):
        parser.error("--coordinator-restart-downtime-seconds must be in [0.5, 30]")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    private = output / ".private-runtime"
    private.mkdir(parents=True, exist_ok=True)
    report_path = output / "training_qwen15b_four_gpu_live_probe.json"
    ledger = Path(args.attempt_ledger).resolve()
    cancel_file = Path(args.cancel_file).resolve() if args.cancel_file else None

    def require_not_cancelled() -> None:
        if cancel_file is not None and cancel_file.is_file():
            raise RuntimeError("qwen15b_user_cancelled")

    initial_budget = allocation_budget_summary(_load(ledger))
    if initial_budget.get("total_attempt_limit_unbounded") is not True and int(
        args.attempt_limit
    ) > int(initial_budget["effective_attempt_limit"]):
        parser.error("--attempt-limit exceeds the validated allocation amendment")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "evidence_ready": False,
        "qwen15b_four_gpu_alpha_verified": False,
        "training_qwen15b_beta_live_verified": False,
        "beta_mode": int(args.coordinator_restart_after_step) == 4,
        "live_run_performed": not bool(args.preflight_only),
        "mock_runtime_used": False,
        "cpu_fallback_used": False,
        "tiny_or_random_model_used": False,
        "started_at": utc_now(),
        "attempt": 0,
        "allocation_started": False,
        "attempt_limit": int(args.attempt_limit),
        "allocation_budget": initial_budget,
        "requested_model": MODEL_ID,
        "requested_model_revision": MODEL_REVISION,
        "requested_topology": "kaggle-2x-t4x2",
        "requested_steps": 8,
        "requested_kernel_count": 2,
        "requested_accelerator": "NvidiaTeslaT4",
        "same_authorized_account": True,
        "multi_account_gate_substitution": False,
        "tpu_used": False,
        "blockers": [],
        "status_observations": [],
        "worker_reports": [],
        "coordinator_restart_after_step": int(args.coordinator_restart_after_step),
        "cleanup": {
            "kernels_deleted": False,
            "only_attempt_kernel_refs_targeted": True,
            "private_packages_removed": False,
            "coordinator_stopped": False,
            "tunnel_stopped": False,
            "private_runtime_removed": False,
            "checkpoint_archives_preserved": False,
            "adapter_archive_preserved": False,
            "rendezvous_private_payloads_removed": False,
        },
        "activation_values_public": False,
        "gradient_values_public": False,
        "adapter_tensor_values_public": False,
        "token_ids_public": False,
        "raw_training_text_public": False,
        "credentials_public": False,
        "credential_paths_public": False,
        "coordinator_url_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    attempt = 0
    attempt_started_epoch = 0.0
    outcome = "not_started"
    server = None
    server_thread = None
    coordinator_runtime: RestartableQwenCoordinator | None = None
    restart_monitor_thread: threading.Thread | None = None
    restart_monitor_stop = threading.Event()
    restart_monitor_errors: list[str] = []
    tunnel_process = None
    rendezvous = None
    refs: list[str] = []
    cleanup_refs: list[str] = []
    selected_env_values: dict[str, str] = {}
    packages: list[dict[str, Any]] = []
    try:
        token_files = list(args.token_file or [])
        sections = _credential_sections(
            token_files,
            raw_token_file=str(args.raw_token_file),
            raw_token_username=str(args.raw_token_username),
        )
        if not sections:
            raise RuntimeError("qwen15b_private_kaggle_credentials_required")
        require_not_cancelled()
        preflight, candidates = preflight_accounts(sections)
        report["account_preflight"] = preflight
        report["account_preflight_count"] = len(preflight)
        report["eligible_account_count"] = len(candidates)
        if args.preflight_only:
            report["preflight_only"] = True
            report["blockers"].append("qwen15b_preflight_only_no_allocation")
            outcome = "preflight_only"
            raise PreflightOnlyComplete(outcome)
        if not candidates:
            raise RuntimeError("qwen15b_same_account_two_gpu_slots_unavailable")
        require_not_cancelled()
        selected = candidates[0]
        selected_env_values = dict(selected["env_values"])
        owner = str(selected["owner"])
        report["selected_account"] = {
            "owner_hash": selected["owner_hash"],
            "candidate_index": int(selected["index"]),
            "effective_remaining_seconds": float(selected["effective_remaining"]),
            "credential_values_public": False,
        }

        source_manifest_path = Path(args.source_manifest).resolve()
        source_manifest = _load(source_manifest_path)
        if not source_manifest_path.is_file():
            raise RuntimeError("qwen15b_pinned_source_manifest_missing")
        report["source_manifest"] = {
            "present": True,
            "file_hash": sha256_file(source_manifest_path),
            "model_id": source_manifest.get("model_id"),
            "model_revision": source_manifest.get("model_revision"),
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        config_bytes = fetch_bytes(_hf_url(MODEL_ID, MODEL_REVISION, "config.json"))
        if sha256_bytes(config_bytes) != source_manifest.get("config_hash"):
            raise RuntimeError("qwen15b_pinned_config_hash_mismatch")
        config = json.loads(config_bytes)
        tokenized = Path(args.tokenized_payload).resolve()
        if not tokenized.is_file():
            raise RuntimeError("qwen15b_private_tokenized_payload_missing")

        run_id = f"qwen15b-four-gpu-{int(time.time())}-{secrets.token_hex(3)}"
        coordinator_token = secrets.token_urlsafe(32)
        port = _free_port()
        coordinator_runtime = RestartableQwenCoordinator(
            private_root=private,
            port=port,
            run_id=run_id,
            token=coordinator_token,
        )
        coordinator_runtime.start()
        rendezvous = coordinator_runtime.rendezvous
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
        report["route_preflight"] = route
        report["route_preflight_verified"] = True
        require_not_cancelled()

        suffix = f"{str(int(time.time()))[-8:]}-{secrets.token_hex(2)}"
        for role in ("kernel_a", "kernel_b"):
            packages.append(
                build_package(
                    private / f"package-{role}",
                    owner=owner,
                    slug=safe_slug(f"ct-qwen15b-alpha-{role}-{suffix}"),
                    role=role,
                    config=config,
                    tokenized_payload_path=tokenized,
                    coordinator_url=tunnel_url,
                    coordinator_token=coordinator_token,
                    run_id=run_id,
                    coordinator_restart_after_step=int(
                        args.coordinator_restart_after_step
                    ),
                )
            )

        attempt = reserve_attempt(ledger, limit=int(args.attempt_limit))
        report["attempt"] = attempt
        report["allocation_started"] = True
        attempt_started = time.monotonic()
        attempt_started_epoch = time.time()
        if args.coordinator_restart_after_step:
            def monitor_restart() -> None:
                deadline = attempt_started + float(args.allocation_timeout_seconds)
                while not restart_monitor_stop.is_set() and time.monotonic() < deadline:
                    try:
                        assert coordinator_runtime is not None
                        current = coordinator_runtime.rendezvous
                        if current is None:
                            time.sleep(0.01)
                            continue
                        status = current.public_status()
                        completed_stage_ids = {
                            int(item.get("stage_id", -1))
                            for item in status.get("events") or []
                            if item.get("run_kind") == "resumed"
                            and item.get("operation") == "optimizer_step"
                            and int(item.get("step") or 0)
                            == int(args.coordinator_restart_after_step)
                        }
                        if completed_stage_ids == {0, 1, 2, 3}:
                            report["coordinator_restart"] = coordinator_runtime.restart(
                                after_step=int(args.coordinator_restart_after_step),
                                downtime_seconds=float(
                                    args.coordinator_restart_downtime_seconds
                                ),
                            )
                            return
                    except BaseException as exc:
                        restart_monitor_errors.append(
                            f"{type(exc).__name__}:{hashlib.sha256(str(exc).encode('utf-8')).hexdigest()[:12]}"
                        )
                        return
                    time.sleep(0.01)

            restart_monitor_thread = threading.Thread(
                target=monitor_restart,
                name="qwen15b-coordinator-restart-monitor",
                daemon=True,
            )
            restart_monitor_thread.start()
        cleanup_refs = [str(package["kernel_ref"]) for package in packages]
        _write(
            private / "active_resources.json",
            {
                "schema": "crowdtensor_qwen15b_private_resources_v1",
                "kernel_refs": cleanup_refs,
                "same_owner": True,
            },
        )
        with tempfile.TemporaryDirectory(prefix="ct-qwen15b-selected-account-") as config_dir:
            env = clean_env(selected_env_values, config_dir=Path(config_dir))

            def push(package: dict[str, Any]) -> dict[str, Any]:
                remaining = max(1.0, float(args.allocation_timeout_seconds) - (time.monotonic() - attempt_started))
                step = run_command(
                    [
                        "kaggle",
                        "kernels",
                        "push",
                        "-p",
                        str(package["package_dir"]),
                        "-t",
                        "1800",
                        "--accelerator",
                        "NvidiaTeslaT4",
                    ],
                    env=env,
                    timeout=min(float(args.push_timeout_seconds), remaining),
                )
                step["role"] = package["role"]
                step["accepted"] = push_accepted(step)
                step["ref"] = extract_kernel_ref(
                    str(step.get("output_tail") or ""), package["kernel_ref"]
                )
                return step

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                pushes = list(executor.map(push, packages))
            require_not_cancelled()
            report["pushes"] = [
                {key: value for key, value in item.items() if key != "ref"} for item in pushes
            ]
            refs = [str(item["ref"]) for item in pushes if item.get("accepted")]
            if len(refs) != 2:
                raise RuntimeError("qwen15b_dual_kernel_concurrent_push_incomplete")
            report["kernel_ref_hashes"] = [stable_hash({"ref": ref}) for ref in refs]
            role_by_ref = {str(item["ref"]): str(item["role"]) for item in pushes}
            deadline = attempt_started + float(args.allocation_timeout_seconds)
            terminal: dict[str, str] = {}
            max_running = 0
            while time.monotonic() < deadline:
                require_not_cancelled()
                states = {}
                for ref in refs:
                    status = run_command(
                        ["kaggle", "kernels", "status", ref],
                        env=env,
                        timeout=min(float(args.status_timeout_seconds), max(1.0, deadline - time.monotonic())),
                    )
                    states[ref] = status_class(str(status.get("output_tail") or ""))
                running = sum(value == "running" for value in states.values())
                max_running = max(max_running, running)
                report["status_observations"].append(
                    {
                        "observed_at": utc_now(),
                        "running_count": running,
                        "queued_count": sum(value == "queued" for value in states.values()),
                        "complete_count": sum(value == "complete" for value in states.values()),
                        "failed_count": sum(value == "failed" for value in states.values()),
                    }
                )
                _write(report_path, report)
                if all(value in TERMINAL for value in states.values()):
                    terminal = states
                    break
                time.sleep(min(float(args.poll_interval_seconds), max(0.1, deadline - time.monotonic())))
            if len(terminal) != 2:
                raise RuntimeError("qwen15b_dual_kernel_attempt_timeout")
            report["max_observed_running_kernel_count"] = max_running

            workers = []
            checkpoint_reports = []
            adapter_report: dict[str, Any] = {}
            for ref in refs:
                require_not_cancelled()
                role = role_by_ref[ref]
                private_output = private / f"output-{role}"
                output_step = run_command(
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
                    timeout=float(args.output_timeout_seconds),
                )
                report.setdefault("outputs", []).append({"role": role, **output_step})
                report.setdefault("kernel_logs", {})[role] = _preserve_public_kernel_log(
                    private_output,
                    output / "logs" / f"{role}.log",
                )
                worker = _load(private_output / WORKER_REPORT)
                if worker:
                    workers.append(worker)
                    _write(output / "workers" / f"{role}.json", worker)
                checkpoint_name = f"training_qwen15b_{role}_checkpoint_bundle.zip"
                source = private_output / checkpoint_name
                destination = output / "checkpoints" / checkpoint_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source.is_file():
                    shutil.move(str(source), destination)
                checkpoint_reports.append(
                    {
                        "role": role,
                        **inspect_checkpoint_bundle(
                            destination,
                            dict(worker.get("checkpoint_bundle") or {}),
                        ),
                    }
                )
                if role == "kernel_b":
                    adapter_source = private_output / "training_qwen15b_standard_peft_adapter.zip"
                    adapter_destination = output / "training_qwen15b_standard_peft_adapter.zip"
                    if adapter_source.is_file():
                        shutil.move(str(adapter_source), adapter_destination)
                    adapter_report = inspect_adapter_bundle(
                        adapter_destination,
                        dict(worker.get("adapter_bundle") or {}),
                    )
                    if adapter_report.get("verified"):
                        _extract_adapter(adapter_destination, output / "exported_adapter")

            report["worker_reports"] = workers
            report["checkpoint_bundles"] = checkpoint_reports
            report["adapter_bundle"] = adapter_report
            report["cleanup"]["checkpoint_archives_preserved"] = bool(
                len(checkpoint_reports) == 2
                and all(item.get("verified") for item in checkpoint_reports)
            )
            report["cleanup"]["adapter_archive_preserved"] = adapter_report.get("verified") is True
            if args.coordinator_restart_after_step:
                restart_monitor_stop.set()
                if restart_monitor_thread is not None:
                    restart_monitor_thread.join(timeout=10.0)
                if restart_monitor_errors:
                    raise RuntimeError("qwen15b_coordinator_restart_monitor_failed")
                if not (report.get("coordinator_restart") or {}).get("verified"):
                    raise RuntimeError("qwen15b_coordinator_restart_not_observed")
            if coordinator_runtime is None or coordinator_runtime.rendezvous is None:
                raise RuntimeError("qwen15b_coordinator_runtime_missing")
            rendezvous = coordinator_runtime.rendezvous
            rendezvous_status = rendezvous.public_status()
            report["rendezvous"] = rendezvous_status
            evidence = evaluate_live_evidence(
                workers=workers,
                rendezvous=rendezvous_status,
                checkpoint_bundles=checkpoint_reports,
                adapter_bundle=adapter_report,
                max_running=max_running,
            )
            report["evidence"] = evidence
            benchmark = build_beta_benchmark(
                workers=workers,
                rendezvous=rendezvous_status,
                attempt_started_epoch=attempt_started_epoch,
                attempt_elapsed_seconds=time.monotonic() - attempt_started,
                evidence=evidence,
            )
            report["benchmark"] = benchmark
            _write(output / "training_qwen15b_beta_benchmark.json", benchmark)
            report["qwen15b_four_gpu_alpha_verified"] = evidence["verified"]
            report["training_qwen15b_beta_live_verified"] = bool(
                evidence["verified"]
                and benchmark.get("benchmark_complete") is True
                and (
                    not args.coordinator_restart_after_step
                    or evidence["beta_recovery_verified"]
                )
            )
            report["ok"] = bool(
                report["training_qwen15b_beta_live_verified"]
                if args.coordinator_restart_after_step
                else evidence["verified"]
            )
            if not report["ok"]:
                report["blockers"].append("qwen15b_four_gpu_live_acceptance_incomplete")
            outcome = "verified" if report["ok"] else "acceptance_incomplete"
    except PreflightOnlyComplete:
        pass
    except BaseException as exc:
        code = str(exc).split(":", 1)[0][:160] or type(exc).__name__
        report["blockers"].append(code)
        report["error_class"] = type(exc).__name__
        outcome = code
    finally:
        restart_monitor_stop.set()
        if restart_monitor_thread is not None:
            restart_monitor_thread.join(timeout=5.0)
        if cleanup_refs and selected_env_values:
            deleted = 0
            with tempfile.TemporaryDirectory(prefix="ct-qwen15b-cleanup-") as config_dir:
                cleanup_env = clean_env(selected_env_values, config_dir=Path(config_dir))
                for ref in sorted(set(cleanup_refs)):
                    step = run_command(
                        ["kaggle", "kernels", "delete", ref, "-y"],
                        env=cleanup_env,
                        timeout=float(args.delete_timeout_seconds),
                    )
                    deleted += int(delete_succeeded_or_absent(step))
            report["cleanup"]["kernels_deleted"] = deleted == len(set(cleanup_refs))
        else:
            report["cleanup"]["kernels_deleted"] = True
        if coordinator_runtime is not None and coordinator_runtime.rendezvous is not None:
            rendezvous = coordinator_runtime.rendezvous
        if rendezvous is not None:
            cleanup = rendezvous.cleanup()
            report["rendezvous_cleanup"] = cleanup
            report["cleanup"]["rendezvous_private_payloads_removed"] = cleanup.get(
                "private_payloads_removed"
            ) is True
        else:
            report["cleanup"]["rendezvous_private_payloads_removed"] = True
        report["cleanup"]["coordinator_stopped"] = bool(
            coordinator_runtime is None or coordinator_runtime.stop()
        )
        report["cleanup"]["tunnel_stopped"] = stop_process(tunnel_process)
        for package in packages:
            shutil.rmtree(Path(str(package.get("package_dir") or "")).parent, ignore_errors=True)
        report["cleanup"]["private_packages_removed"] = all(
            not (private / f"package-{role}").exists() for role in ("kernel_a", "kernel_b")
        )
        shutil.rmtree(private, ignore_errors=True)
        report["cleanup"]["private_runtime_removed"] = not private.exists()
        report["cleanup"]["checkpoint_archives_verified_before_cleanup"] = bool(
            report["cleanup"].get("checkpoint_archives_preserved")
        )
        cleanup_ok = all(
            report["cleanup"].get(key) is True
            for key in (
                "kernels_deleted",
                "only_attempt_kernel_refs_targeted",
                "private_packages_removed",
                "coordinator_stopped",
                "tunnel_stopped",
                "private_runtime_removed",
                "rendezvous_private_payloads_removed",
            )
        )
        safety_errors = public_safety_errors(report)
        report["public_artifact_safe"] = not safety_errors
        if safety_errors:
            report["safety_errors"] = safety_errors
        report["finished_at"] = utc_now()
        report["blockers"] = sorted(set(report.get("blockers") or []))
        report["ok"] = bool(report.get("ok") and cleanup_ok and report["public_artifact_safe"])
        report["qwen15b_four_gpu_alpha_verified"] = bool(
            report.get("qwen15b_four_gpu_alpha_verified") and report["ok"]
        )
        report["training_qwen15b_beta_live_verified"] = bool(
            report.get("training_qwen15b_beta_live_verified") and report["ok"]
        )
        report["evidence_ready"] = bool(cleanup_ok and report["public_artifact_safe"])
        _write(report_path, report)
        if attempt:
            finish_attempt(ledger, attempt=attempt, outcome=outcome)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(
                f"training_qwen15b_four_gpu_probe ok={report['ok']} "
                f"attempt={report['attempt']} blockers={','.join(report['blockers']) or 'none'}"
            )
    return 0 if report["ok"] else (1 if report["evidence_ready"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
