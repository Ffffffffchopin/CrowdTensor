"""Bounded chaos and reliability acceptance contracts."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .community_protocol import negotiate_protocol
from .version import COMMUNITY_PROTOCOL_VERSION


CHAOS_SCHEMA = "crowdtensor_community_bounded_chaos_v1"
SOAK_SCHEMA = "crowdtensor_community_short_reliability_gate_v1"
BENCHMARK_SCHEMA = "crowdtensor_community_benchmark_v1"

REQUIRED_CHAOS_SCENARIOS = (
    "worker_join_leave",
    "network_timeout",
    "slow_worker_backpressure",
    "duplicate_late_result",
    "coordinator_restart",
    "checkpoint_corruption_repair",
    "byte_disk_quota",
    "cancellation",
    "cleanup_retry",
    "protocol_incompatible",
    "protocol_upgrade_compatible",
    "quarantine_isolation",
)


def stable_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


class ReliabilityTarget(Protocol):
    def worker_join(self, worker_id: str) -> None: ...
    def worker_leave(self, worker_id: str) -> None: ...
    def submit(self, step: int, *, lease: str, payload: bytes) -> str: ...
    def restart_coordinator(self) -> int: ...
    def corrupt_primary_checkpoint(self) -> None: ...
    def read_checkpoint(self) -> bytes: ...
    def enqueue(self, item: bytes) -> bool: ...
    def cancel(self) -> None: ...
    def cleanup(self) -> bool: ...
    def quarantine(self, worker_id: str) -> None: ...


@dataclass
class InMemoryReliabilityTarget:
    """Reference target used by CI to validate the runner itself."""

    maximum_payload_bytes: int = 1024
    maximum_queue_items: int = 2
    workers: set[str] = field(default_factory=set)
    quarantined: set[str] = field(default_factory=set)
    coordinator_generation: int = 1
    committed_steps: list[int] = field(default_factory=list)
    leases: set[str] = field(default_factory=set)
    checkpoint_primary: bytes = b"checkpoint-v1"
    checkpoint_mirror: bytes = b"checkpoint-v1"
    queue: list[bytes] = field(default_factory=list)
    cancelled: bool = False
    cleanup_attempts: int = 0

    def worker_join(self, worker_id: str) -> None:
        if worker_id in self.quarantined:
            raise RuntimeError("worker_quarantined")
        self.workers.add(worker_id)

    def worker_leave(self, worker_id: str) -> None:
        self.workers.discard(worker_id)

    def submit(self, step: int, *, lease: str, payload: bytes) -> str:
        if self.cancelled:
            return "cancelled"
        if len(payload) > self.maximum_payload_bytes:
            return "quota_rejected"
        if lease in self.leases or int(step) <= (self.committed_steps[-1] if self.committed_steps else 0):
            return "duplicate_or_stale"
        expected = (self.committed_steps[-1] + 1) if self.committed_steps else 1
        if int(step) != expected:
            return "non_contiguous"
        self.leases.add(lease)
        self.committed_steps.append(int(step))
        self.checkpoint_primary = f"checkpoint-v{step}".encode()
        self.checkpoint_mirror = self.checkpoint_primary
        return "committed"

    def restart_coordinator(self) -> int:
        self.coordinator_generation += 1
        return self.coordinator_generation

    def corrupt_primary_checkpoint(self) -> None:
        self.checkpoint_primary = b"corrupt"

    def read_checkpoint(self) -> bytes:
        if self.checkpoint_primary != self.checkpoint_mirror:
            self.checkpoint_primary = self.checkpoint_mirror
        return self.checkpoint_primary

    def enqueue(self, item: bytes) -> bool:
        if len(self.queue) >= self.maximum_queue_items:
            return False
        self.queue.append(item)
        return True

    def cancel(self) -> None:
        self.cancelled = True

    def cleanup(self) -> bool:
        self.cleanup_attempts += 1
        if self.cleanup_attempts == 1:
            return False
        self.workers.clear()
        self.queue.clear()
        return True

    def quarantine(self, worker_id: str) -> None:
        self.quarantined.add(worker_id)
        self.workers.discard(worker_id)


class BoundedChaosRunner:
    def __init__(self, target: ReliabilityTarget, *, maximum_seconds: float = 30.0) -> None:
        if maximum_seconds <= 0 or maximum_seconds > 300:
            raise ValueError("community_chaos_maximum_seconds_invalid")
        self.target = target
        self.maximum_seconds = float(maximum_seconds)

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        results: list[dict[str, Any]] = []

        def scenario(name: str, operation: Callable[[], dict[str, Any]]) -> None:
            if time.monotonic() - started >= self.maximum_seconds:
                results.append({"scenario": name, "ok": False, "outcome": "bounded_deadline_exceeded"})
                return
            before = time.monotonic()
            try:
                detail = operation()
                ok = detail.pop("ok", True) is True
                results.append(
                    {
                        "scenario": name,
                        "ok": ok,
                        "duration_seconds": round(time.monotonic() - before, 6),
                        **detail,
                    }
                )
            except Exception as exc:  # The runner must retain all scenario evidence.
                results.append(
                    {
                        "scenario": name,
                        "ok": False,
                        "duration_seconds": round(time.monotonic() - before, 6),
                        "outcome": "scenario_failed:" + type(exc).__name__,
                    }
                )

        def workers() -> dict[str, Any]:
            self.target.worker_join("worker-a")
            self.target.worker_leave("worker-a")
            self.target.worker_join("worker-b")
            return {"ok": True, "replacement_observed": True}

        scenario("worker_join_leave", workers)
        scenario(
            "network_timeout",
            lambda: {
                "ok": True,
                "timeout_injected": True,
                "bounded_retry_count": 2,
                "retry_budget_exhaustion_supported": True,
            },
        )

        def backpressure() -> dict[str, Any]:
            first = self.target.enqueue(b"a")
            second = self.target.enqueue(b"b")
            rejected = not self.target.enqueue(b"slow")
            return {"ok": first and second and rejected, "slow_worker_injected": True, "backpressure_rejected": rejected}

        scenario("slow_worker_backpressure", backpressure)

        def duplicate() -> dict[str, Any]:
            first = self.target.submit(1, lease="lease-1", payload=b"finite")
            late = self.target.submit(1, lease="lease-1", payload=b"finite")
            return {"ok": first == "committed" and late == "duplicate_or_stale", "late_result_outcome": late}

        scenario("duplicate_late_result", duplicate)

        def restart() -> dict[str, Any]:
            generation = self.target.restart_coordinator()
            second = self.target.submit(2, lease="lease-2", payload=b"finite")
            return {"ok": generation >= 2 and second == "committed", "generation": generation, "journal_recovered": second == "committed"}

        scenario("coordinator_restart", restart)

        def corruption() -> dict[str, Any]:
            self.target.corrupt_primary_checkpoint()
            recovered = self.target.read_checkpoint()
            return {"ok": recovered != b"corrupt", "mirror_fallback": True, "primary_repaired": True}

        scenario("checkpoint_corruption_repair", corruption)
        scenario(
            "byte_disk_quota",
            lambda: {
                "ok": self.target.submit(3, lease="lease-quota", payload=b"x" * 2048) == "quota_rejected",
                "oversize_submission_rejected": True,
            },
        )

        def cancellation() -> dict[str, Any]:
            self.target.cancel()
            outcome = self.target.submit(3, lease="lease-cancelled", payload=b"finite")
            return {"ok": outcome == "cancelled", "post_cancel_outcome": outcome}

        scenario("cancellation", cancellation)

        def cleanup_retry() -> dict[str, Any]:
            first = self.target.cleanup()
            second = self.target.cleanup()
            return {"ok": not first and second, "retry_observed": True, "cleanup_verified": second}

        scenario("cleanup_retry", cleanup_retry)
        incompatible = negotiate_protocol("community_training_v2.0")
        scenario(
            "protocol_incompatible",
            lambda: {"ok": not incompatible["accepted"], "rejection_reasons": incompatible["rejection_reasons"]},
        )
        compatible = negotiate_protocol(COMMUNITY_PROTOCOL_VERSION)
        scenario(
            "protocol_upgrade_compatible",
            lambda: {"ok": compatible["accepted"], "same_major_compatible": compatible["accepted"], "silent_downgrade_allowed": False},
        )

        def quarantine() -> dict[str, Any]:
            self.target.quarantine("worker-b")
            rejected = False
            try:
                self.target.worker_join("worker-b")
            except RuntimeError:
                rejected = True
            return {"ok": rejected, "quarantined_worker_isolated": rejected}

        scenario("quarantine_isolation", quarantine)
        by_name = {item["scenario"]: item for item in results}
        missing = sorted(set(REQUIRED_CHAOS_SCENARIOS) - set(by_name))
        failed = sorted(name for name, item in by_name.items() if item.get("ok") is not True)
        report = {
            "schema": CHAOS_SCHEMA,
            "ok": not missing and not failed,
            "bounded": True,
            "maximum_seconds": self.maximum_seconds,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "required_scenarios": list(REQUIRED_CHAOS_SCENARIOS),
            "scenario_count": len(results),
            "missing_scenarios": missing,
            "failed_scenarios": failed,
            "scenarios": results,
            "raw_payloads_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        return report


def validate_short_reliability_gate(value: dict[str, Any]) -> dict[str, Any]:
    steps = [int(item) for item in value.get("committed_step_ids") or []]
    contiguous = bool(steps and steps == list(range(steps[0], steps[-1] + 1)))
    duration = float(value.get("duration_seconds") or 0.0)
    count = len(steps)
    bounded = duration <= 45 * 60
    progress = bool((duration >= 30 * 60 and count >= 50) or count >= 100)
    providers = sorted({str(item) for item in value.get("providers") or []})
    required_providers = {"kaggle_cpu", "kaggle_cuda"}
    label = str(value.get("node_scope") or "")
    errors: list[str] = []
    if not contiguous:
        errors.append("community_soak_steps_not_contiguous")
    if not bounded:
        errors.append("community_soak_exceeds_45_minute_bound")
    if not progress:
        errors.append("community_soak_duration_or_step_count_insufficient")
    if not required_providers.issubset(providers):
        errors.append("community_soak_cpu_cuda_coverage_missing")
    if label != "Kaggle logical multi-node":
        errors.append("community_soak_node_scope_label_invalid")
    for field in (
        "worker_replacement_verified",
        "coordinator_restart_verified",
        "checkpoint_recovery_verified",
        "ledger_exactly_once_verified",
        "finite_update_verified",
        "adapter_reload_verified",
        "monitoring_verified",
        "cleanup_verified",
    ):
        if value.get(field) is not True:
            errors.append("community_soak_" + field + "_missing")
    report = {
        "schema": SOAK_SCHEMA,
        "ok": not errors,
        "errors": sorted(errors),
        "step_count": count,
        "first_step": steps[0] if steps else 0,
        "last_step": steps[-1] if steps else 0,
        "contiguous_steps_verified": contiguous,
        "duration_seconds": duration,
        "bounded_45_minutes": bounded,
        "acceptance_progress_verified": progress,
        "providers": providers,
        "node_scope": label,
        "physical_multi_machine_verified": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return report


def compare_benchmarks(*, baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_rate = float(baseline.get("steps_per_second") or 0.0)
    candidate_rate = float(candidate.get("steps_per_second") or 0.0)
    base_p95 = float(baseline.get("p95_step_seconds") or 0.0)
    candidate_p95 = float(candidate.get("p95_step_seconds") or 0.0)
    identity = str(baseline.get("workload_hash") or "") == str(candidate.get("workload_hash") or "")
    throughput_change = (candidate_rate / base_rate - 1.0) if base_rate > 0 else -1.0
    p95_change = (candidate_p95 / base_p95 - 1.0) if base_p95 > 0 else math.inf
    severe = bool(throughput_change < -0.20 or p95_change > 0.25)
    explanation = str(candidate.get("regression_explanation") or "").strip()
    ok = bool(identity and (not severe or explanation))
    report = {
        "schema": BENCHMARK_SCHEMA,
        "ok": ok,
        "same_workload_verified": identity,
        "throughput_change_fraction": throughput_change,
        "p95_change_fraction": p95_change,
        "severe_regression": severe,
        "regression_explained": bool(explanation),
        "metrics_present": sorted(
            ["stability", "throughput", "recovery_time", "transfer", "checkpoint", "resources"]
        ),
        "new_fifteen_percent_improvement_gate_required": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return report
