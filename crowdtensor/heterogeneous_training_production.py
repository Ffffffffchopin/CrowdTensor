"""Production control surface for heterogeneous CPU/CUDA/JAX-TPU training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

from .heterogeneous_training_beta import HeterogeneousTrainingBetaController
from .heterogeneous_training_manifest import (
    stable_hash,
    validate_training_manifest,
)
from .heterogeneous_training_scheduler import (
    build_placement_plan,
    estimate_stage_resources,
)
from .model_adapter import get_model_adapter


CONFIG_SCHEMA = "crowdtensor_heterogeneous_training_production_config_v1"
PRIVATE_JOB_SCHEMA = "crowdtensor_heterogeneous_training_production_private_job_v1"
VALIDATION_SCHEMA = "crowdtensor_heterogeneous_training_production_validation_v1"
PLAN_SCHEMA = "crowdtensor_heterogeneous_training_production_plan_v1"
STATUS_SCHEMA = "crowdtensor_heterogeneous_training_production_status_v1"
PERFORMANCE_SCHEMA = "crowdtensor_heterogeneous_training_performance_comparison_v1"

PINNED_MODEL_ID = "Qwen/Qwen2.5-7B"
PINNED_MODEL_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
REQUIRED_ACCELERATORS = ["cpu", "cuda", "jax_tpu"]


def _write_json(path: Path, value: Any, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("heterogeneous_training_production_json_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("heterogeneous_training_production_json_invalid")
    return value


def _bounded_int(
    value: Any,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"heterogeneous_training_production_{name}_invalid") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"heterogeneous_training_production_{name}_invalid")
    return result


def _bounded_float(
    value: Any,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"heterogeneous_training_production_{name}_invalid") from exc
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"heterogeneous_training_production_{name}_invalid")
    return result


def default_production_config() -> dict[str, Any]:
    config = {
        "schema": CONFIG_SCHEMA,
        "model": {
            "model_id": PINNED_MODEL_ID,
            "model_revision": PINNED_MODEL_REVISION,
            "training_mode": "peft_lora",
        },
        "soak": {
            "target_steps": 100,
            "minimum_duration_seconds": 3600,
            "full_live_gate_timeout_seconds": 21600,
            "checkpoint_every_steps": 1,
            "adapter_reload_required": True,
        },
        "placement": {
            "required_accelerators": REQUIRED_ACCELERATORS,
            "dynamic_rebalance": True,
            "resource_telemetry": True,
            "checkpoint_freshness": True,
            "worker_replacements": REQUIRED_ACCELERATORS,
            "coordinator_restart_required": True,
        },
        "performance": {
            "baseline_steps": 20,
            "candidate_steps": 20,
            "minimum_improvement_fraction": 0.15,
            "maximum_unexplained_p95_regression_fraction": 0.05,
            "minimum_windows": 3,
            "same_workload_required": True,
            "same_topology_required": True,
        },
        "fault_governance": {
            "retry_attempts": 8,
            "retry_base_seconds": 0.5,
            "retry_cap_seconds": 5.0,
            "deterministic_jitter": True,
            "lease_seconds": 300.0,
            "quarantine_threshold": 3,
            "quarantine_seconds": 300.0,
            "fault_classes": sorted(
                [
                    "network_timeout",
                    "duplicate_or_stale_result",
                    "worker_crash",
                    "checkpoint_corrupt",
                    "coordinator_restart",
                    "cleanup_retry",
                ]
            ),
        },
        "monitoring": {
            "prometheus_metrics": True,
            "structured_events": True,
            "health_readiness_status": True,
            "low_cardinality_labels": True,
        },
        "acquisition": {
            "authorization_mode": "unlimited_authorized",
            "maximum_window_seconds": 43200,
            "maximum_full_live_gate_seconds": 21600,
        },
        "private_inputs": {
            "hf_token_env": "HF_TOKEN",
            "kaggle_credentials": "environment_or_private_token_file",
        },
    }
    config["content_hash"] = stable_hash(config)
    return config


def validate_production_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != CONFIG_SCHEMA:
        raise ValueError("heterogeneous_training_production_config_schema_invalid")
    source = json.loads(json.dumps(value))
    supplied_hash = str(source.pop("content_hash", ""))
    model = dict(source.get("model") or {})
    if (
        model.get("model_id") != PINNED_MODEL_ID
        or model.get("model_revision") != PINNED_MODEL_REVISION
        or model.get("training_mode") != "peft_lora"
    ):
        raise ValueError("heterogeneous_training_production_model_contract_invalid")
    soak = dict(source.get("soak") or {})
    canonical_soak = {
        "target_steps": _bounded_int(
            soak.get("target_steps"),
            name="target_steps",
            minimum=100,
            maximum=10000,
        ),
        "minimum_duration_seconds": _bounded_int(
            soak.get("minimum_duration_seconds"),
            name="minimum_duration_seconds",
            minimum=3600,
            maximum=21600,
        ),
        "full_live_gate_timeout_seconds": _bounded_int(
            soak.get("full_live_gate_timeout_seconds"),
            name="full_live_gate_timeout_seconds",
            minimum=3600,
            maximum=21600,
        ),
        "checkpoint_every_steps": _bounded_int(
            soak.get("checkpoint_every_steps"),
            name="checkpoint_every_steps",
            minimum=1,
            maximum=10,
        ),
        "adapter_reload_required": soak.get("adapter_reload_required") is True,
    }
    if not canonical_soak["adapter_reload_required"]:
        raise ValueError("heterogeneous_training_production_adapter_reload_required")
    placement = dict(source.get("placement") or {})
    required = sorted({str(item) for item in placement.get("required_accelerators") or []})
    replacements = sorted({str(item) for item in placement.get("worker_replacements") or []})
    if required != sorted(REQUIRED_ACCELERATORS) or replacements != sorted(
        REQUIRED_ACCELERATORS
    ):
        raise ValueError("heterogeneous_training_production_accelerator_contract_invalid")
    if not all(
        placement.get(key) is True
        for key in (
            "dynamic_rebalance",
            "resource_telemetry",
            "checkpoint_freshness",
            "coordinator_restart_required",
        )
    ):
        raise ValueError("heterogeneous_training_production_placement_contract_invalid")
    performance = dict(source.get("performance") or {})
    canonical_performance = {
        "baseline_steps": _bounded_int(
            performance.get("baseline_steps"),
            name="baseline_steps",
            minimum=10,
            maximum=1000,
        ),
        "candidate_steps": _bounded_int(
            performance.get("candidate_steps"),
            name="candidate_steps",
            minimum=10,
            maximum=1000,
        ),
        "minimum_improvement_fraction": _bounded_float(
            performance.get("minimum_improvement_fraction"),
            name="minimum_improvement_fraction",
            minimum=0.15,
            maximum=1.0,
        ),
        "maximum_unexplained_p95_regression_fraction": _bounded_float(
            performance.get("maximum_unexplained_p95_regression_fraction"),
            name="maximum_p95_regression_fraction",
            minimum=0.0,
            maximum=0.25,
        ),
        "minimum_windows": _bounded_int(
            performance.get("minimum_windows"),
            name="minimum_windows",
            minimum=3,
            maximum=20,
        ),
        "same_workload_required": performance.get("same_workload_required") is True,
        "same_topology_required": performance.get("same_topology_required") is True,
    }
    if not (
        canonical_performance["same_workload_required"]
        and canonical_performance["same_topology_required"]
    ):
        raise ValueError("heterogeneous_training_production_benchmark_identity_required")
    fault = dict(source.get("fault_governance") or {})
    required_faults = {
        "network_timeout",
        "duplicate_or_stale_result",
        "worker_crash",
        "checkpoint_corrupt",
        "coordinator_restart",
        "cleanup_retry",
    }
    fault_classes = sorted({str(item) for item in fault.get("fault_classes") or []})
    if not required_faults.issubset(fault_classes):
        raise ValueError("heterogeneous_training_production_fault_matrix_incomplete")
    canonical_fault = {
        "retry_attempts": _bounded_int(
            fault.get("retry_attempts"), name="retry_attempts", minimum=1, maximum=20
        ),
        "retry_base_seconds": _bounded_float(
            fault.get("retry_base_seconds"),
            name="retry_base_seconds",
            minimum=0.01,
            maximum=10.0,
        ),
        "retry_cap_seconds": _bounded_float(
            fault.get("retry_cap_seconds"),
            name="retry_cap_seconds",
            minimum=0.1,
            maximum=60.0,
        ),
        "deterministic_jitter": fault.get("deterministic_jitter") is True,
        "lease_seconds": _bounded_float(
            fault.get("lease_seconds"),
            name="lease_seconds",
            minimum=5.0,
            maximum=1800.0,
        ),
        "quarantine_threshold": _bounded_int(
            fault.get("quarantine_threshold"),
            name="quarantine_threshold",
            minimum=1,
            maximum=20,
        ),
        "quarantine_seconds": _bounded_float(
            fault.get("quarantine_seconds"),
            name="quarantine_seconds",
            minimum=1.0,
            maximum=86400.0,
        ),
        "fault_classes": fault_classes,
    }
    if not canonical_fault["deterministic_jitter"]:
        raise ValueError("heterogeneous_training_production_retry_jitter_required")
    monitoring = dict(source.get("monitoring") or {})
    if not all(
        monitoring.get(key) is True
        for key in (
            "prometheus_metrics",
            "structured_events",
            "health_readiness_status",
            "low_cardinality_labels",
        )
    ):
        raise ValueError("heterogeneous_training_production_monitoring_contract_invalid")
    acquisition = dict(source.get("acquisition") or {})
    if acquisition.get("authorization_mode") != "unlimited_authorized":
        raise ValueError("heterogeneous_training_production_authorization_mode_invalid")
    canonical_acquisition = {
        "authorization_mode": "unlimited_authorized",
        "maximum_window_seconds": _bounded_int(
            acquisition.get("maximum_window_seconds"),
            name="acquisition_window_seconds",
            minimum=600,
            maximum=43200,
        ),
        "maximum_full_live_gate_seconds": _bounded_int(
            acquisition.get("maximum_full_live_gate_seconds"),
            name="full_live_gate_seconds",
            minimum=600,
            maximum=21600,
        ),
    }
    private_inputs = dict(source.get("private_inputs") or {})
    hf_env = str(private_inputs.get("hf_token_env") or "")
    if not hf_env or not hf_env.replace("_", "").isalnum():
        raise ValueError("heterogeneous_training_production_hf_env_invalid")
    canonical = {
        "schema": CONFIG_SCHEMA,
        "model": model,
        "soak": canonical_soak,
        "placement": {
            "required_accelerators": sorted(REQUIRED_ACCELERATORS),
            "dynamic_rebalance": True,
            "resource_telemetry": True,
            "checkpoint_freshness": True,
            "worker_replacements": sorted(REQUIRED_ACCELERATORS),
            "coordinator_restart_required": True,
        },
        "performance": canonical_performance,
        "fault_governance": canonical_fault,
        "monitoring": {
            "prometheus_metrics": True,
            "structured_events": True,
            "health_readiness_status": True,
            "low_cardinality_labels": True,
        },
        "acquisition": canonical_acquisition,
        "private_inputs": {
            "hf_token_env": hf_env,
            "kaggle_credentials": str(
                private_inputs.get("kaggle_credentials")
                or "environment_or_private_token_file"
            ),
        },
    }
    canonical["content_hash"] = stable_hash(canonical)
    if supplied_hash and supplied_hash != canonical["content_hash"]:
        raise ValueError("heterogeneous_training_production_config_hash_mismatch")
    return canonical


def load_production_config(path: str | Path) -> dict[str, Any]:
    return validate_production_config(_read_json(path))


def production_manifest(config: dict[str, Any]) -> dict[str, Any]:
    canonical = validate_production_config(config)
    manifest = get_model_adapter("qwen2_lora_v1").production_manifest(
        target_steps=int(canonical["soak"]["target_steps"]),
        accelerators=REQUIRED_ACCELERATORS,
    )
    manifest.pop("content_hash", None)
    manifest["checkpoint"]["checkpoint_every_steps"] = int(
        canonical["soak"]["checkpoint_every_steps"]
    )
    manifest["checkpoint"]["retention_steps"] = max(
        2, int(canonical["soak"]["checkpoint_every_steps"]) + 1
    )
    return validate_training_manifest(manifest)


def validate_production_request(
    config: dict[str, Any],
    *,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    canonical = validate_production_config(config)
    env = os.environ if environment is None else environment
    hf_env = str(canonical["private_inputs"]["hf_token_env"])
    report = {
        "schema": VALIDATION_SCHEMA,
        "ok": True,
        "configuration_valid": True,
        "model_supported": True,
        "model_id": PINNED_MODEL_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "target_steps": int(canonical["soak"]["target_steps"]),
        "minimum_duration_seconds": int(
            canonical["soak"]["minimum_duration_seconds"]
        ),
        "required_accelerators": sorted(REQUIRED_ACCELERATORS),
        "hf_token_available": bool(str(env.get(hf_env) or "")),
        "hf_token_env_name_hash": "sha256:"
        + hashlib.sha256(hf_env.encode("utf-8")).hexdigest(),
        "credential_values_public": False,
        "credential_paths_public": False,
        "private_env_names_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return report


def build_production_plan(
    config: dict[str, Any],
    *,
    capabilities: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    canonical = validate_production_config(config)
    manifest = production_manifest(canonical)
    capability_rows = [dict(item) for item in capabilities]
    estimates = []
    for stage in manifest["stages"]:
        for device_type in stage["allowed_device_types"]:
            estimates.append(
                estimate_stage_resources(
                    manifest, stage, device_type=device_type
                )
            )
    placement = (
        build_placement_plan(manifest, capability_rows)
        if capability_rows
        else {}
    )
    report = {
        "schema": PLAN_SCHEMA,
        "ok": not capability_rows or placement.get("complete_stage_coverage") is True,
        "configuration_hash": canonical["content_hash"],
        "training_manifest_hash": manifest["content_hash"],
        "model_id": PINNED_MODEL_ID,
        "target_steps": int(manifest["training"]["target_steps"]),
        "minimum_duration_seconds": int(
            canonical["soak"]["minimum_duration_seconds"]
        ),
        "stage_count": len(manifest["stages"]),
        "required_accelerators": sorted(REQUIRED_ACCELERATORS),
        "minimum_topology": {
            "single_cuda_miners": 3,
            "jax_tpu_v5e8_miners": 1,
            "cpu_miners": 1,
        },
        "resource_estimates": estimates,
        "live_capability_count": len(capability_rows),
        "placement": placement,
        "soak_gate_bounded": True,
        "performance_gate_fraction": float(
            canonical["performance"]["minimum_improvement_fraction"]
        ),
        "credential_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return report


def retry_delay_seconds(
    policy: dict[str, Any],
    *,
    operation: str,
    attempt: int,
) -> float:
    canonical = validate_production_config(
        {
            **default_production_config(),
            "fault_governance": policy,
            "content_hash": "",
        }
    )["fault_governance"]
    index = int(attempt)
    if index < 0 or index >= int(canonical["retry_attempts"]):
        raise ValueError("heterogeneous_training_production_retry_attempt_invalid")
    jitter_byte = hashlib.sha256(
        f"{operation}:{index}".encode("utf-8")
    ).digest()[0]
    jitter = 0.75 + float(jitter_byte) / 255.0 * 0.5
    return min(
        float(canonical["retry_cap_seconds"]),
        float(canonical["retry_base_seconds"]) * (2**index) * jitter,
    )


def execute_bounded_operation(
    operation: Any,
    *,
    operation_name: str,
    policy: dict[str, Any],
    sleep: Any = time.sleep,
) -> tuple[Any, dict[str, Any]]:
    """Execute an idempotent operation with finite backoff and safe evidence."""

    attempts = int(policy["retry_attempts"])
    history = []
    for attempt in range(attempts):
        try:
            result = operation()
        except (ConnectionError, OSError, TimeoutError) as exc:
            final = attempt + 1 >= attempts
            delay = (
                0.0
                if final
                else retry_delay_seconds(
                    policy,
                    operation=operation_name,
                    attempt=attempt,
                )
            )
            history.append(
                {
                    "attempt": attempt + 1,
                    "outcome": "failed",
                    "failure_class": type(exc).__name__,
                    "retry_delay_seconds": delay,
                }
            )
            if final:
                raise RuntimeError(
                    "heterogeneous_training_production_retry_exhausted"
                ) from exc
            sleep(delay)
            continue
        history.append(
            {
                "attempt": attempt + 1,
                "outcome": "succeeded",
                "failure_class": "",
                "retry_delay_seconds": 0.0,
            }
        )
        report = {
            "schema": "crowdtensor_heterogeneous_training_bounded_operation_v1",
            "ok": True,
            "operation_name_hash": "sha256:"
            + hashlib.sha256(str(operation_name).encode("utf-8")).hexdigest(),
            "attempt_count": len(history),
            "retry_count": len(history) - 1,
            "attempt_limit": attempts,
            "history": history,
            "bounded_retry": True,
            "exponential_backoff_with_deterministic_jitter": True,
            "failure_details_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        return result, report
    raise RuntimeError("heterogeneous_training_production_retry_exhausted")


def _median(values: Iterable[float]) -> float:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        return 0.0
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def compare_performance_windows(
    *,
    baseline: Iterable[dict[str, Any]],
    candidate: Iterable[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    baseline_rows = [dict(item) for item in baseline]
    candidate_rows = [dict(item) for item in candidate]
    minimum_windows = int(policy["minimum_windows"])
    identities = {
        str(item.get("workload_hash") or "")
        for item in [*baseline_rows, *candidate_rows]
    }
    topologies = {
        str(item.get("topology_hash") or "")
        for item in [*baseline_rows, *candidate_rows]
    }
    identity_ok = len(identities) == 1 and "" not in identities
    topology_ok = len(topologies) == 1 and "" not in topologies
    counts_ok = (
        len(baseline_rows) >= minimum_windows
        and len(candidate_rows) >= minimum_windows
    )
    baseline_throughput = _median(
        float(item.get("step_throughput_per_second") or 0.0)
        for item in baseline_rows
    )
    candidate_throughput = _median(
        float(item.get("step_throughput_per_second") or 0.0)
        for item in candidate_rows
    )
    baseline_p50 = _median(
        float(item.get("p50_step_latency_seconds") or 0.0)
        for item in baseline_rows
    )
    candidate_p50 = _median(
        float(item.get("p50_step_latency_seconds") or 0.0)
        for item in candidate_rows
    )
    baseline_p95 = _median(
        float(item.get("p95_step_latency_seconds") or 0.0)
        for item in baseline_rows
    )
    candidate_p95 = _median(
        float(item.get("p95_step_latency_seconds") or 0.0)
        for item in candidate_rows
    )
    throughput_improvement = (
        candidate_throughput / baseline_throughput - 1.0
        if baseline_throughput > 0
        else 0.0
    )
    latency_improvement = (
        1.0 - candidate_p50 / baseline_p50 if baseline_p50 > 0 else 0.0
    )
    best = max(throughput_improvement, latency_improvement)
    p95_regression = (
        candidate_p95 / baseline_p95 - 1.0 if baseline_p95 > 0 else 0.0
    )
    threshold = float(policy["minimum_improvement_fraction"])
    p95_limit = float(policy["maximum_unexplained_p95_regression_fraction"])
    passed = bool(
        counts_ok
        and identity_ok
        and topology_ok
        and best >= threshold
        and p95_regression <= p95_limit
    )
    report = {
        "schema": PERFORMANCE_SCHEMA,
        "ok": passed,
        "performance_gate_passed": passed,
        "minimum_window_count": minimum_windows,
        "baseline_window_count": len(baseline_rows),
        "candidate_window_count": len(candidate_rows),
        "same_workload_verified": identity_ok,
        "same_topology_verified": topology_ok,
        "baseline_median_step_throughput_per_second": baseline_throughput,
        "candidate_median_step_throughput_per_second": candidate_throughput,
        "throughput_improvement_fraction": throughput_improvement,
        "baseline_median_p50_step_latency_seconds": baseline_p50,
        "candidate_median_p50_step_latency_seconds": candidate_p50,
        "p50_latency_improvement_fraction": latency_improvement,
        "p95_regression_fraction": p95_regression,
        "required_improvement_fraction": threshold,
        "maximum_p95_regression_fraction": p95_limit,
        "workload_or_topology_reduction_used": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return report


class HeterogeneousTrainingProductionController:
    """Idempotent owner workflow layered over the durable Beta runtime."""

    def __init__(self, job_dir: str | Path) -> None:
        self.job_dir = Path(job_dir).expanduser().resolve()
        self.private_dir = self.job_dir / ".private-production"
        self.private_job_path = self.private_dir / "job.json"
        private = _read_json(self.private_job_path)
        if private.get("schema") != PRIVATE_JOB_SCHEMA:
            raise ValueError("heterogeneous_training_production_job_schema_invalid")
        self.private = private
        self.config = load_production_config(private["config_path"])
        self.beta = HeterogeneousTrainingBetaController(self.job_dir)
        self.runtime = self.beta.runtime
        self.public_validation_path = self.job_dir / "training_production_validation.json"
        self.public_plan_path = self.job_dir / "training_production_plan.json"
        self.public_status_path = self.job_dir / "training_production_status.json"
        self.public_cleanup_path = self.job_dir / "training_production_cleanup.json"

    @classmethod
    def is_job(cls, job_dir: str | Path) -> bool:
        return (
            Path(job_dir).expanduser().resolve()
            / ".private-production"
            / "job.json"
        ).is_file()

    @classmethod
    def create(
        cls,
        job_dir: str | Path,
        *,
        config: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
        model_config_path: str | Path | None = None,
        tokenized_payload_path: str | Path | None = None,
        hf_token: str = "",
        checkpoint_storage: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any] | "HeterogeneousTrainingProductionController":
        output = Path(job_dir).expanduser().resolve()
        if cls.is_job(output):
            controller = cls(output)
            if dry_run:
                return controller.plan()
            return controller
        if config is not None and config_path is not None:
            raise ValueError("heterogeneous_training_production_config_conflict")
        canonical = validate_production_config(
            config
            if config is not None
            else load_production_config(config_path)
            if config_path is not None
            else default_production_config()
        )
        validation = validate_production_request(canonical)
        plan = build_production_plan(canonical)
        if dry_run:
            return {
                "schema": "crowdtensor_heterogeneous_training_production_dry_run_v1",
                "ok": validation["ok"] and plan["ok"],
                "dry_run": True,
                "validation": validation,
                "plan": plan,
                "private_paths_public": False,
                "public_artifact_safe": True,
            }
        output.mkdir(parents=True, exist_ok=True)
        private = output / ".private-production"
        private.mkdir(parents=True, exist_ok=True)
        private.chmod(0o700)
        config_destination = private / "production_config.json"
        manifest_destination = private / "training_manifest.json"
        _write_json(config_destination, canonical)
        _write_json(manifest_destination, production_manifest(canonical))
        primary_storage = dict(checkpoint_storage or {"backend": "local"})
        mirrored_storage = (
            primary_storage
            if primary_storage.get("backend") == "mirrored"
            else {
                "backend": "mirrored",
                "primary": primary_storage,
                "mirror_root": str(private / "checkpoint-recovery-mirror"),
            }
        )
        HeterogeneousTrainingBetaController.create(
            output,
            manifest_path=manifest_destination,
            config_path=model_config_path,
            tokenized_payload_path=tokenized_payload_path,
            hf_token=hf_token,
            checkpoint_storage=mirrored_storage,
            checkpoint_retention_steps=max(
                2, int(canonical["soak"]["checkpoint_every_steps"]) + 1
            ),
            lease_seconds=float(canonical["fault_governance"]["lease_seconds"]),
            max_online_miners=32,
            enable_jax_tpu=True,
            tensor_lookup_optimization_after_step=int(
                canonical["performance"]["baseline_steps"]
            ),
        )
        _write_json(
            private / "job.json",
            {
                "schema": PRIVATE_JOB_SCHEMA,
                "config_path": str(config_destination),
                "manifest_path": str(manifest_destination),
                "created_at": time.time(),
                "public_artifact": False,
            },
        )
        controller = cls(output)
        _write_json(controller.public_validation_path, validation, mode=0o644)
        _write_json(controller.public_plan_path, plan, mode=0o644)
        controller.status()
        return controller

    def validate(self) -> dict[str, Any]:
        report = validate_production_request(self.config)
        _write_json(self.public_validation_path, report, mode=0o644)
        return report

    def plan(self) -> dict[str, Any]:
        capabilities = [
            dict(item.get("capability") or {})
            for item in self.runtime.public_status()["miners"]
            if item.get("state") == "online" and item.get("capability")
        ]
        report = build_production_plan(self.config, capabilities=capabilities)
        _write_json(self.public_plan_path, report, mode=0o644)
        return report

    def status(self) -> dict[str, Any]:
        beta = self.beta.status()
        metrics = self.runtime.metrics_snapshot()
        blockers = list(beta.get("blockers") or [])
        if beta["overall_state"] == "waiting_for_miners":
            blockers.append("training_production_waiting_for_miners")
        report = {
            "schema": STATUS_SCHEMA,
            "job_id": beta["job_id"],
            "overall_state": beta["overall_state"],
            "current_phase": beta["current_phase"],
            "committed_step": int(beta["committed_step"]),
            "target_steps": int(beta["target_steps"]),
            "minimum_soak_duration_seconds": int(
                self.config["soak"]["minimum_duration_seconds"]
            ),
            "required_accelerators": sorted(REQUIRED_ACCELERATORS),
            "placement_generation": int(beta["placement_generation"]),
            "owner_paused": bool(beta["runtime"].get("owner_paused")),
            "coordinator_generation": int(
                beta["runtime"].get("coordinator_generation") or 0
            ),
            "metrics": metrics,
            "runtime": beta["runtime"],
            "blockers": sorted(set(blockers)),
            "next_resume_command": "crowdtensor train resume <job-dir>",
            "next_resume_command_redacts_credentials": True,
            "next_resume_command_uses_public_placeholder": True,
            "credential_values_public": False,
            "credential_paths_public": False,
            "private_env_names_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        _write_json(self.public_status_path, report, mode=0o644)
        return report

    def pause(self) -> dict[str, Any]:
        result = self.beta.pause()
        return {**self.status(), "pause_transition_applied": result["pause_transition_applied"], "command_ok": True}

    def resume(self) -> dict[str, Any]:
        result = self.beta.resume()
        return {**self.status(), "resume_transition_applied": result["resume_transition_applied"], "command_ok": True}

    def stop(self) -> dict[str, Any]:
        result = self.beta.cancel()
        return {**self.status(), "stop_transition_applied": result.get("cancel_transition_applied", True), "command_ok": True}

    def rebalance(self, *, reason: str = "owner_requested") -> dict[str, Any]:
        result = self.beta.rebalance(reason=reason)
        return {**self.status(), "rebalance_transition_applied": result["rebalance_transition_applied"], "command_ok": True}

    def cleanup(self) -> dict[str, Any]:
        result = self.beta.cleanup()
        report = {
            "schema": "crowdtensor_heterogeneous_training_production_cleanup_v1",
            "ok": result.get("ok") is True,
            "command_ok": result.get("ok") is True,
            "job_id": result["job_id"],
            "global_step": int(result["global_step"]),
            "active_miner_leases_revoked": result[
                "active_miner_leases_revoked"
            ],
            "tensor_transport_cleanup": result["tensor_transport_cleanup"],
            "live_resources_left_running": result[
                "live_resources_left_running"
            ],
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        _write_json(self.public_cleanup_path, report, mode=0o644)
        return report

    def remove_private_runtime(self) -> dict[str, Any]:
        cleanup = self.cleanup()
        shutil.rmtree(self.private_dir, ignore_errors=True)
        return {
            **cleanup,
            "production_private_state_removed": not self.private_dir.exists(),
        }
