"""Bounded research protocols and numerical-free resource-trace replay.

Replay models delivery of independent Work Units, not optimizer progress or
quorum aggregation. Numerical calibration lives in backends.benchmark.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
from pathlib import Path
from typing import Any

from .contracts import stable_hash


PROTOCOL_SCHEMA = "crowdtensor_training_benchmark_protocol_v1"
TRACE_SCHEMA = "crowdtensor_training_resource_trace_v1"
REPORT_SCHEMA = "crowdtensor_training_benchmark_report_v1"


class BenchmarkError(ValueError):
    """Invalid or unsafe benchmark input, with a public error code."""


def sealed(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "content_hash"}
    return {**unsigned, "content_hash": stable_hash(unsigned)}


def _integer(value: Any, minimum: int, maximum: int) -> bool:
    return type(value) is int and minimum <= value <= maximum


def load_protocol(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema", "study_id", "seeds", "conditions", "fixed_steps",
        "horizon_seconds", "cold_start_seconds", "setup_seconds",
        "upload_seconds", "model_bytes", "delta_bytes", "cpu_total_steps",
        "content_hash",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise BenchmarkError("benchmark_protocol_fields_invalid")
    if value["schema"] != PROTOCOL_SCHEMA or value != sealed(value):
        raise BenchmarkError("benchmark_protocol_hash_or_schema_invalid")
    if value["study_id"] != "intermittent-work-unit-baselines-v1":
        raise BenchmarkError("benchmark_study_unsupported")
    seeds = value["seeds"]
    if (
        not isinstance(seeds, list) or not 1 <= len(seeds) <= 10
        or not all(_integer(seed, 0, 2**31 - 1) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise BenchmarkError("benchmark_seeds_invalid")
    if value["conditions"] != ["stable", "intermittent"]:
        raise BenchmarkError("benchmark_conditions_invalid")
    steps = value["fixed_steps"]
    if (
        not isinstance(steps, dict) or set(steps) != {"fixed_short", "fixed_long"}
        or not all(_integer(item, 1, 16) for item in steps.values())
        or steps["fixed_short"] >= steps["fixed_long"]
    ):
        raise BenchmarkError("benchmark_fixed_steps_invalid")
    bounds = {
        "horizon_seconds": (10, 3600), "cold_start_seconds": (0, 60),
        "setup_seconds": (1, 60), "upload_seconds": (1, 60),
        "model_bytes": (1, 10**12), "delta_bytes": (1, 10**9),
        "cpu_total_steps": (4, 64),
    }
    if any(not _integer(value[key], *bound) for key, bound in bounds.items()):
        raise BenchmarkError("benchmark_budget_invalid")
    if any(value["cpu_total_steps"] % (2 * size) for size in steps.values()):
        raise BenchmarkError("benchmark_step_budget_not_divisible_by_quorum")
    return value


def make_trace(protocol: dict[str, Any], *, condition: str, seed: int) -> dict[str, Any]:
    """Generate exogenous availability; no policy affects random-number draws."""

    if condition not in protocol["conditions"] or seed not in protocol["seeds"]:
        raise BenchmarkError("benchmark_trace_selection_invalid")
    rng = random.Random(seed)
    horizon = protocol["horizon_seconds"]
    windows = []
    for index in range(2):
        start = 0
        while start < horizon:
            duration = horizon if condition == "stable" else rng.randint(8, 24)
            end = min(horizon, start + duration)
            windows.append({
                "worker": f"logical-{index}", "start": start, "end": end,
                "seconds_per_step": index + 1,
            })
            start = end + rng.randint(2, 8)
    return sealed({
        "schema": TRACE_SCHEMA, "condition": condition, "seed": seed,
        "origin": "synthetic", "horizon_seconds": horizon,
        "windows": sorted(windows, key=lambda item: (item["start"], item["worker"])),
    })


def validate_trace(trace: dict[str, Any]) -> None:
    if trace.get("schema") != TRACE_SCHEMA or trace != sealed(trace):
        raise BenchmarkError("benchmark_trace_hash_or_schema_invalid")
    if not _integer(trace.get("horizon_seconds"), 1, 3600):
        raise BenchmarkError("benchmark_trace_horizon_invalid")
    windows = trace.get("windows")
    if not isinstance(windows, list) or len(windows) > 2048:
        raise BenchmarkError("benchmark_trace_windows_invalid")
    previous: dict[str, int] = {}
    for window in windows:
        if not isinstance(window, dict) or set(window) != {
            "worker", "start", "end", "seconds_per_step"
        }:
            raise BenchmarkError("benchmark_trace_window_invalid")
        worker = window["worker"]
        if worker not in {"logical-0", "logical-1"}:
            raise BenchmarkError("benchmark_trace_worker_invalid")
        if (
            not _integer(window["start"], 0, trace["horizon_seconds"])
            or not _integer(window["end"], 1, trace["horizon_seconds"])
            or window["start"] >= window["end"]
            or window["start"] < previous.get(worker, 0)
            or not _integer(window["seconds_per_step"], 1, 60)
        ):
            raise BenchmarkError("benchmark_trace_window_invalid")
        previous[worker] = window["end"]


def replay_trace(
    protocol: dict[str, Any], trace: dict[str, Any], *, policy: str
) -> dict[str, Any]:
    """Account for setup, interrupted compute, and partial upload separately.

Each availability window is an ephemeral runtime with a cold cache. A delivery
completed exactly at departure is counted before departure. There is no round
barrier here: deliveries must never be reported as committed training progress.
"""

    validate_trace(trace)
    if policy not in protocol["fixed_steps"]:
        raise BenchmarkError("benchmark_policy_not_implemented")
    steps = protocol["fixed_steps"][policy]
    metrics = dict.fromkeys((
        "available_worker_seconds", "cold_start_seconds", "setup_seconds",
        "compute_seconds", "upload_seconds", "download_bytes", "upload_bytes",
        "computed_steps", "delivered_steps", "delivered_work_units",
        "interrupted_work_units",
    ), 0.0)
    events = []
    for window_index, window in enumerate(trace["windows"]):
        start, end = window["start"], window["end"]
        rate = window["seconds_per_step"]
        metrics["available_worker_seconds"] += end - start
        cold = min(protocol["cold_start_seconds"], end - start)
        metrics["cold_start_seconds"] += cold
        fraction = cold / protocol["cold_start_seconds"] if protocol["cold_start_seconds"] else 1
        metrics["download_bytes"] += protocol["model_bytes"] * fraction
        now = start + cold
        while now < end:
            issued = now
            phase_seconds = {}
            for phase, requested in (
                ("setup", protocol["setup_seconds"]),
                ("compute", steps * rate),
                ("upload", protocol["upload_seconds"]),
            ):
                elapsed = min(requested, end - now)
                phase_seconds[phase] = elapsed
                metrics[phase + "_seconds"] += elapsed
                now += elapsed
            computed = phase_seconds["compute"] / rate
            metrics["computed_steps"] += computed
            metrics["upload_bytes"] += (
                protocol["delta_bytes"] * phase_seconds["upload"] / protocol["upload_seconds"]
            )
            delivered = phase_seconds["upload"] == protocol["upload_seconds"]
            metrics["delivered_steps"] += steps if delivered else 0
            metrics["delivered_work_units" if delivered else "interrupted_work_units"] += 1
            events.append({
                "window": window_index, "worker": window["worker"],
                "issued_at": issued, "finished_at": now, "requested_steps": steps,
                "computed_steps": computed, "delivered": delivered,
            })
    metrics["undelivered_compute_steps"] = metrics["computed_steps"] - metrics["delivered_steps"]
    metrics["delivered_steps_per_available_worker_second"] = (
        metrics["delivered_steps"] / metrics["available_worker_seconds"]
        if metrics["available_worker_seconds"] else None
    )
    return sealed({
        "policy": policy, "condition": trace["condition"], "seed": trace["seed"],
        "trace_hash": trace["content_hash"], "protocol_hash": protocol["content_hash"],
        "evidence_kind": "synthetic_delivery_cost_model",
        "quorum_aggregation_simulated": False, "model_quality": None,
        "metrics": metrics, "events": events,
    })


def write_json(path: Path, value: Any) -> None:
    # Exclusive creation keeps reruns from replacing evidence from an earlier run.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run_benchmark(
    protocol_path: str | Path, output_dir: str | Path, *, backend: str
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if backend not in {"trace", "cpu-fixture"}:
        raise BenchmarkError("benchmark_backend_unsupported")
    root = Path(output_dir).expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=False, mode=0o700)
    except FileExistsError as exc:
        raise BenchmarkError("benchmark_output_already_exists") from exc
    write_json(root / "protocol.json", protocol)
    write_json(root / "run-state.json", {"state": "started", "backend": backend})
    if backend == "trace":
        trials = []
        for seed in protocol["seeds"]:
            for condition in protocol["conditions"]:
                trace = make_trace(protocol, condition=condition, seed=seed)
                write_json(root / f"trace-{condition}-{seed}.json", trace)
                for policy in protocol["fixed_steps"]:
                    trials.append(replay_trace(protocol, trace, policy=policy))
        scope = "synthetic_delivery_cost_model"
    else:
        from crowdtensor.backends.benchmark import run_cpu_calibration

        trials = run_cpu_calibration(protocol, root)
        scope = "local_cpu_fixture_calibration"
    report = sealed({
        "schema": REPORT_SCHEMA, "command_ok": True,
        "study_id": protocol["study_id"], "protocol_hash": protocol["content_hash"],
        "backend": backend, "evidence_kind": scope,
        "python_version": platform.python_version(),
        "trial_count": len(trials), "trials": trials,
        "source_hashes": {
            name: "sha256:" + hashlib.sha256((Path(__file__).parents[1] / name).read_bytes()).hexdigest()
            for name in (
                "core/benchmark.py", "backends/benchmark.py", "hf_lora_training.py",
                "backends/elastic_peft.py", "core/controller.py", "named_tensor_optimizer.py",
                "volunteer_training_cell.py", "volunteer_training_coordinator.py",
                "core/contracts.py", "training_contract.py", "model_adapter.py",
            )
        },
        "adaptive_policy_evaluated": False,
        "physical_multi_host_verified": False,
        "community_participation_measured": False,
        "statistical_significance_claimed": False,
        "public_artifact_safe": True,
    })
    write_json(root / "report.json", report)
    rows = [{
        "policy": trial["policy"], "condition": trial["condition"], "seed": trial["seed"],
        **trial["metrics"],
    } for trial in trials]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (root / "metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "complete.json", {"report_hash": report["content_hash"]})
    return report
