from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from crowdtensor.core.benchmark import (
    BenchmarkError, load_protocol, make_trace, replay_trace, run_benchmark,
    sealed, validate_trace, write_json,
)


PROTOCOL = Path(__file__).resolve().parents[1] / "research/intermittent-training/protocol.json"


def _single_window(protocol, *, end=9):
    trace = make_trace(protocol, condition="stable", seed=protocol["seeds"][0])
    trace["windows"] = [{"worker": "logical-0", "start": 0, "end": end, "seconds_per_step": 2}]
    return sealed(trace)


def test_replay_accounts_for_partial_setup_and_compute_without_claiming_training():
    protocol = load_protocol(PROTOCOL)
    report = replay_trace(protocol, _single_window(protocol), policy="fixed_short")
    metrics = report["metrics"]
    assert metrics["delivered_steps"] == 1
    assert metrics["computed_steps"] == 1
    assert metrics["delivered_work_units"] == 1
    assert metrics["interrupted_work_units"] == 1
    assert metrics["available_worker_seconds"] == 9
    assert sum(metrics[key] for key in (
        "cold_start_seconds", "setup_seconds", "compute_seconds", "upload_seconds"
    )) == 9
    assert metrics["download_bytes"] == protocol["model_bytes"]
    assert metrics["upload_bytes"] == protocol["delta_bytes"]
    assert report["model_quality"] is None
    assert report["quorum_aggregation_simulated"] is False


def test_partial_upload_is_charged_but_never_counted_as_delivered():
    protocol = load_protocol(PROTOCOL)
    protocol["upload_seconds"] = 2
    protocol = sealed(protocol)
    report = replay_trace(protocol, _single_window(protocol, end=8), policy="fixed_short")
    assert report["metrics"]["computed_steps"] == 1
    assert report["metrics"]["delivered_steps"] == 0
    assert report["metrics"]["undelivered_compute_steps"] == 1
    assert report["metrics"]["upload_bytes"] == protocol["delta_bytes"] / 2
    completed = replay_trace(protocol, _single_window(protocol, end=9), policy="fixed_short")
    assert completed["metrics"]["delivered_steps"] == 1


def test_identical_trace_for_policies_and_no_future_dependent_task_length():
    protocol = load_protocol(PROTOCOL)
    trace = make_trace(protocol, condition="intermittent", seed=17)
    first = replay_trace(protocol, trace, policy="fixed_short")
    assert first == replay_trace(protocol, trace, policy="fixed_short")
    second = replay_trace(protocol, trace, policy="fixed_long")
    assert first["trace_hash"] == second["trace_hash"]
    near = replay_trace(protocol, _single_window(protocol, end=8), policy="fixed_long")
    far = replay_trace(protocol, _single_window(protocol, end=80), policy="fixed_long")
    assert near["events"][0]["requested_steps"] == far["events"][0]["requested_steps"] == 4
    assert near["events"][0]["issued_at"] == far["events"][0]["issued_at"]
    assert first["metrics"]["computed_steps"] >= first["metrics"]["delivered_steps"]


def test_all_offline_trace_is_bounded_and_has_no_fabricated_rate():
    protocol = load_protocol(PROTOCOL)
    trace = _single_window(protocol)
    trace["windows"] = []
    result = replay_trace(protocol, sealed(trace), policy="fixed_short")
    assert result["metrics"]["delivered_steps"] == 0
    assert result["metrics"]["delivered_steps_per_available_worker_second"] is None


def test_trace_rejects_overlap_and_tampering():
    protocol = load_protocol(PROTOCOL)
    trace = _single_window(protocol)
    trace["windows"].append(dict(trace["windows"][0]))
    with pytest.raises(BenchmarkError, match="window_invalid"):
        validate_trace(sealed(trace))
    with pytest.raises(BenchmarkError, match="hash_or_schema"):
        validate_trace(trace)


@pytest.mark.parametrize("field,value", [
    ("seeds", [17, 17]), ("seeds", [True]), ("horizon_seconds", 100000),
    ("cpu_total_steps", 7), ("setup_seconds", 0),
    ("fixed_steps", {"fixed_short": 4, "fixed_long": 4}),
])
def test_protocol_rejects_ambiguous_or_unbounded_runs(tmp_path, field, value):
    protocol = copy.deepcopy(load_protocol(PROTOCOL))
    protocol[field] = value
    write_json(tmp_path / "invalid.json", sealed(protocol))
    with pytest.raises(BenchmarkError):
        load_protocol(tmp_path / "invalid.json")


def test_replay_artifacts_are_reproducible_and_rerun_preserves_evidence(tmp_path):
    first = run_benchmark(PROTOCOL, tmp_path / "one", backend="trace")
    second = run_benchmark(PROTOCOL, tmp_path / "two", backend="trace")
    assert first == second
    assert first["trial_count"] == 12
    assert (tmp_path / "one/metrics.csv").read_bytes() == (tmp_path / "two/metrics.csv").read_bytes()
    assert json.loads((tmp_path / "one/complete.json").read_text())["report_hash"] == first["content_hash"]
    with pytest.raises(BenchmarkError, match="already_exists"):
        run_benchmark(PROTOCOL, tmp_path / "one", backend="trace")
    assert json.loads((tmp_path / "one/report.json").read_text()) == first


def test_cli_replay_has_no_framework_imports(tmp_path):
    code = (
        "import sys; from crowdtensor.cli import main; "
        "args=sys.argv[1:]; "
        "\ntry: main(args)\nexcept SystemExit as exc: assert exc.code == 0\n"
        "assert not {'torch','transformers','peft','jax'}.intersection(sys.modules)"
    )
    result = subprocess.run([
        sys.executable, "-c", code, "train", "benchmark", "--protocol", str(PROTOCOL),
        "--output-dir", str(tmp_path / "cli"), "--json",
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["trial_count"] == 12


def test_real_cpu_reference_and_elastic_recover_with_equal_steps(tmp_path, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    protocol = load_protocol(PROTOCOL)
    protocol["seeds"] = [17]
    write_json(tmp_path / "one-seed.json", sealed(protocol))
    report = run_benchmark(tmp_path / "one-seed.json", tmp_path / "cpu", backend="cpu-fixture")
    assert capsys.readouterr().out == ""
    assert report["trial_count"] == 6
    trials = report["trials"]
    assert {trial["metrics"]["accepted_optimizer_steps"] for trial in trials} == {8}
    assert {trial["metrics"]["accepted_tokens"] for trial in trials} == {256}
    for policy in ("fixed_short", "fixed_long", "upstream_checkpoint"):
        pair = [trial for trial in trials if trial["policy"] == policy]
        assert pair[0]["adapter_hash"] == pair[1]["adapter_hash"]
        assert pair[0]["quality"]["mean_loss"] == pair[1]["quality"]["mean_loss"]
    for trial in trials:
        assert trial["natural_language_benchmark"] is False
        if trial["policy"] != "upstream_checkpoint":
            assert trial["duplicate_receipt_idempotent"] is True
            assert trial["checkpoint_lineage_verified"] is True
            if trial["condition"] == "intermittent":
                assert trial["stale_result_rejected"] is True
                assert trial["metrics"]["reassignments"] == 1
    serialized = json.dumps(report)
    assert str(tmp_path) not in serialized
    for marker in ('"invite_token"', '"lease_token"', '"input_ids"', '"labels"'):
        assert marker not in serialized
