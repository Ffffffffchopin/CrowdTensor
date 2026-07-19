from crowdtensor.community_reliability import (
    BoundedChaosRunner,
    InMemoryReliabilityTarget,
    compare_benchmarks,
    validate_short_reliability_gate,
)


def test_bounded_chaos_runner_covers_every_required_fault() -> None:
    report = BoundedChaosRunner(InMemoryReliabilityTarget(), maximum_seconds=10).run()
    assert report["ok"] is True
    assert report["scenario_count"] == 12
    assert report["missing_scenarios"] == []
    assert report["failed_scenarios"] == []
    assert report["elapsed_seconds"] <= 10


def complete_soak(*, steps: int = 100, duration: float = 1200.0) -> dict:
    return {
        "committed_step_ids": list(range(1, steps + 1)),
        "duration_seconds": duration,
        "providers": ["kaggle_cuda", "kaggle_cpu"],
        "node_scope": "Kaggle logical multi-node",
        "worker_replacement_verified": True,
        "coordinator_restart_verified": True,
        "checkpoint_recovery_verified": True,
        "ledger_exactly_once_verified": True,
        "finite_update_verified": True,
        "adapter_reload_verified": True,
        "monitoring_verified": True,
        "cleanup_verified": True,
    }


def test_short_soak_accepts_100_steps_before_45_minutes_and_labels_scope() -> None:
    report = validate_short_reliability_gate(complete_soak())
    assert report["ok"] is True
    assert report["contiguous_steps_verified"] is True
    assert report["physical_multi_machine_verified"] is False


def test_short_soak_accepts_50_steps_only_after_30_minutes() -> None:
    assert validate_short_reliability_gate(complete_soak(steps=50, duration=1800))["ok"] is True
    failed = validate_short_reliability_gate(complete_soak(steps=50, duration=1799))
    assert failed["ok"] is False
    assert "community_soak_duration_or_step_count_insufficient" in failed["errors"]


def test_short_soak_rejects_gaps_wrong_scope_missing_cpu_and_cleanup() -> None:
    value = complete_soak()
    value["committed_step_ids"].remove(50)
    value["node_scope"] = "physical multi-machine"
    value["providers"] = ["kaggle_cuda"]
    value["cleanup_verified"] = False
    result = validate_short_reliability_gate(value)
    assert result["ok"] is False
    assert set(result["errors"]) >= {
        "community_soak_steps_not_contiguous",
        "community_soak_node_scope_label_invalid",
        "community_soak_cpu_cuda_coverage_missing",
        "community_soak_cleanup_verified_missing",
    }


def test_benchmark_requires_explanation_for_severe_regression() -> None:
    baseline = {"workload_hash": "sha256:a", "steps_per_second": 1.0, "p95_step_seconds": 1.0}
    candidate = {"workload_hash": "sha256:a", "steps_per_second": 0.7, "p95_step_seconds": 1.4}
    failed = compare_benchmarks(baseline=baseline, candidate=candidate)
    candidate["regression_explanation"] = "Kaggle shared-host variance"
    explained = compare_benchmarks(baseline=baseline, candidate=candidate)
    assert failed["ok"] is False
    assert explained["ok"] is True
    assert explained["new_fifteen_percent_improvement_gate_required"] is False
