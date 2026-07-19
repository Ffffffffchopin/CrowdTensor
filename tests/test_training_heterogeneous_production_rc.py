from copy import deepcopy
import json
import py_compile

from crowdtensor.heterogeneous_training_manifest import stable_hash
from crowdtensor.heterogeneous_training_miner import (
    _committed_telemetry_sampling_interval,
)
from scripts.training_heterogeneous_production_fault_probe import (
    run_probe as run_fault_probe,
)
from scripts.training_heterogeneous_production_live_probe import (
    BASELINE_END_STEP,
    MINIMUM_REQUIRED_STEPS,
    PERFORMANCE_GATE_STEP,
    PERFORMANCE_WINDOW_COUNT,
    PERFORMANCE_WINDOW_SIZE,
    TARGET_STEPS,
    _observe_placement_blocker,
    _performance_windows,
    _public_blocker,
    _replacement_evidence,
)
from scripts.training_heterogeneous_production_live_replay import replay
from scripts.training_heterogeneous_production_rc_check import (
    LIVE_SCHEMA,
    SCHEMA,
    check_report,
)
from scripts.training_heterogeneous_production_rc_pack import pack
from scripts.training_heterogeneous_production_regression_pack import (
    pack as pack_regression,
)
from scripts.training_heterogeneous_production_workflow_probe import (
    run_probe as run_workflow_probe,
)
from scripts.training_heterogeneous_beta_kaggle_package import (
    build_package as build_cpu_gpu_package,
)
from scripts.training_heterogeneous_tpu_beta_kaggle_package import (
    build_package as build_tpu_package,
)


def worker(label: str, *, old_step: int, new: bool, generation: int) -> dict:
    target = old_step + 1 if new else old_step
    stage_id = 1
    return {
        "kernel_role": "gpu_a",
        "label": label,
        "returncode": 0,
        "report": {
            "ok": True,
            "device_policy": "cuda",
            "miner_id_hash": "sha256:" + ("b" if new else "a") * 64,
            "graceful_drain_applied": not new,
            "offline_transition_applied": True,
            "central_checkpoint_restore_count": 1 if new else 0,
            "client": {"checkpoint_download_count": 1 if new else 0},
            "steps": [
                {
                    "target_step": target,
                    "placement_generation": generation,
                    "stages": [
                        {
                            "stage_id": stage_id,
                            "target_step": target,
                            "placement_generation": generation,
                            "checkpoint_components_validated": True,
                            "checkpoint_hash": "sha256:" + ("d" if new else "c") * 64,
                            "archive_hash": "sha256:" + ("f" if new else "e") * 64,
                        }
                    ],
                }
            ],
            "stage_process_ready_history": [
                {
                    "resumed": new,
                    "resumed_global_step": old_step if new else 0,
                    "placement_generation": generation,
                    "stage_id": stage_id,
                }
            ],
        },
    }


def test_replacement_evidence_requires_checkpoint_and_generation_fencing() -> None:
    workers = [
        worker("gpu_old", old_step=50, new=False, generation=1),
        worker("gpu_replacement", old_step=50, new=True, generation=2),
    ]

    evidence = _replacement_evidence(workers, "cuda")
    damaged = deepcopy(workers)
    damaged[1]["report"]["stage_process_ready_history"][0][
        "resumed_global_step"
    ] = 49
    missing_download = deepcopy(damaged)
    missing_download[1]["report"]["client"]["checkpoint_download_count"] = 0

    assert evidence["verified"] is True
    assert evidence["restored_checkpoint_step"] == 50
    assert evidence["restore_evidence_source"] == "stage_ready_history"
    assert _replacement_evidence(damaged, "cuda")["verified"] is True
    assert (
        _replacement_evidence(damaged, "cuda")["restore_evidence_source"]
        == "generation_fenced_contiguous_checkpoint_handoff"
    )
    assert _replacement_evidence(missing_download, "cuda")["verified"] is False


def test_replacement_evidence_selects_cross_kernel_stage_takeover() -> None:
    old = worker("gpu_old", old_step=70, new=False, generation=1)
    idle = worker("gpu_replacement", old_step=70, new=True, generation=2)
    idle["report"]["steps"] = []
    idle["report"]["stage_process_ready_history"] = []
    idle["report"]["client"]["checkpoint_download_count"] = 0
    takeover = worker("gpu_stable_b0", old_step=70, new=True, generation=2)
    takeover["kernel_role"] = "gpu_b"
    takeover["report"]["miner_id_hash"] = "sha256:" + "9" * 64
    takeover["report"]["stage_process_ready_history"] = [
        {
            "resumed": True,
            "resumed_global_step": 100,
            "placement_generation": 5,
            "stage_id": 1,
        }
    ]

    evidence = _replacement_evidence([old, idle, takeover], "cuda")

    assert evidence["verified"] is True
    assert evidence["replacement_selection"] == "cross_kernel_dynamic_reassignment"
    assert evidence["replacement_kernel_role"] == "gpu_b"
    assert evidence["replacement_first_step"] == 71
    assert evidence["restored_checkpoint_step"] == 70
    assert evidence["checkpoint_ready_event_matched"] is False


def test_live_replay_reclassifies_only_verified_cross_kernel_takeover(
    tmp_path,
) -> None:
    def remote(
        label: str,
        *,
        kind: str,
        role: str,
        step: int,
        new: bool,
        generation: int,
        stage_id: int,
        identity_character: str,
    ) -> dict:
        item = worker(label, old_step=step, new=new, generation=generation)
        item["kernel_role"] = role
        item["report"]["device_policy"] = kind
        item["report"]["miner_id_hash"] = (
            "sha256:" + identity_character * 64
        )
        for stage in item["report"]["steps"][0]["stages"]:
            stage["stage_id"] = stage_id
        for ready in item["report"]["stage_process_ready_history"]:
            ready["stage_id"] = stage_id
        return item

    gpu_old = remote(
        "gpu_old",
        kind="cuda",
        role="gpu_a",
        step=70,
        new=False,
        generation=1,
        stage_id=1,
        identity_character="1",
    )
    gpu_idle = remote(
        "gpu_replacement",
        kind="cuda",
        role="gpu_a",
        step=70,
        new=True,
        generation=2,
        stage_id=1,
        identity_character="2",
    )
    gpu_idle["report"]["steps"] = []
    gpu_idle["report"]["stage_process_ready_history"] = []
    gpu_idle["report"]["client"]["checkpoint_download_count"] = 0
    gpu_takeover = remote(
        "gpu_stable_b0",
        kind="cuda",
        role="gpu_b",
        step=70,
        new=True,
        generation=2,
        stage_id=1,
        identity_character="3",
    )
    pairs = {
        "cpu": [
            remote(
                "cpu_old",
                kind="cpu",
                role="cpu",
                step=90,
                new=False,
                generation=3,
                stage_id=4,
                identity_character="4",
            ),
            remote(
                "cpu_replacement",
                kind="cpu",
                role="cpu",
                step=90,
                new=True,
                generation=4,
                stage_id=4,
                identity_character="5",
            ),
        ],
        "tpu": [
            remote(
                "tpu_old",
                kind="jax_tpu",
                role="tpu",
                step=100,
                new=False,
                generation=4,
                stage_id=2,
                identity_character="6",
            ),
            remote(
                "tpu_replacement",
                kind="jax_tpu",
                role="tpu",
                step=100,
                new=True,
                generation=5,
                stage_id=2,
                identity_character="7",
            ),
        ],
    }

    def kernel(role: str, results: list[dict], *, ok: bool) -> dict:
        return {
            "schema": "fixture_kernel_v1",
            "kernel_role": role,
            "ok": ok,
            "worker_results": [
                {
                    "label": item["label"],
                    "returncode": item["returncode"],
                    "report": item["report"],
                }
                for item in results
            ],
            "pause_observation": (
                {
                    "verified": False,
                    "observations": [
                        {
                            "committed_step": 70,
                            "placement_generation": 2,
                            "runtime_state": "running",
                            "missing_stage_ids": [],
                        }
                    ],
                }
                if role == "gpu_a"
                else {"verified": True, "observations": []}
            ),
            "all_worker_processes_stopped": True,
            "private_runtime_removed": True,
            "blockers": (
                ["heterogeneous_kaggle_worker_acceptance_incomplete"]
                if role == "gpu_a"
                else []
            ),
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }

    kernels = [
        kernel("gpu_a", [gpu_old, gpu_idle], ok=False),
        kernel("gpu_b", [gpu_takeover], ok=True),
        kernel("cpu", pairs["cpu"], ok=True),
        kernel("tpu", pairs["tpu"], ok=True),
    ]
    source = {
        "schema": LIVE_SCHEMA,
        "live_run_performed": True,
        "accepted_providers": [
            "kaggle_cpu",
            "kaggle_cuda",
            "kaggle_jax_tpu",
        ],
        "blockers": ["training_production_worker_replacement_gate_failed"],
        "credential_values_public": False,
        "coordinator_url_public": False,
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
        "public_safety_errors": [],
    }
    source["content_hash"] = stable_hash(source)
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    kernel_paths = []
    for value in kernels:
        path = tmp_path / f"{value['kernel_role']}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        kernel_paths.append(path)

    report = replay(
        source_live_path=source_path,
        kernel_paths=kernel_paths,
        output_dir=tmp_path / "replayed",
    )

    assert report["external_runtime_verified"] is True
    assert report["effective_kernel_evidence_verified"] is True
    assert report["blockers"] == []
    assert report["worker_replacements"]["cuda"]["replacement_kernel_role"] == "gpu_b"
    assert report["evidence_replay"]["live_run_reexecuted"] is False


def test_performance_windows_bind_same_live_commit_timeline() -> None:
    status = {
        "commits": [
            {"target_step": step, "committed_at": float(step * 10)}
            for step in range(1, PERFORMANCE_GATE_STEP + 6)
        ]
    }
    baseline, candidate, intervals = _performance_windows(
        status,
        workload_hash="sha256:" + "1" * 64,
        topology_hash="sha256:" + "2" * 64,
    )

    assert len(baseline) == PERFORMANCE_WINDOW_COUNT
    assert len(candidate) == PERFORMANCE_WINDOW_COUNT
    assert all(
        item["sample_count"] == PERFORMANCE_WINDOW_SIZE
        for item in baseline + candidate
    )
    assert len(intervals) == PERFORMANCE_GATE_STEP + 4
    assert {item["workload_hash"] for item in baseline + candidate} == {
        "sha256:" + "1" * 64
    }
    assert MINIMUM_REQUIRED_STEPS == 100
    assert TARGET_STEPS >= 400
    assert (
        _committed_telemetry_sampling_interval(
            committed_step=BASELINE_END_STEP,
            optimization_after_step=BASELINE_END_STEP,
        )
        == 1
    )
    assert (
        _committed_telemetry_sampling_interval(
            committed_step=BASELINE_END_STEP + 1,
            optimization_after_step=BASELINE_END_STEP,
        )
        == 5
    )
    assert (
        _committed_telemetry_sampling_interval(
            committed_step=BASELINE_END_STEP + 5,
            optimization_after_step=BASELINE_END_STEP,
        )
        == 5
    )


def test_live_probe_classifies_empty_interrupt_and_persistent_placement_blocker() -> None:
    status = {
        "runtime_state": "paused_waiting_for_miners",
        "live_miner_count": 6,
        "placement_error": {
            "code": "heterogeneous_placement_capacity_exhausted",
            "diagnostics": {"stage_id": 4},
        },
    }
    observation = {}
    for _ in range(3):
        observation = _observe_placement_blocker(
            status,
            previous_code=str(observation.get("code") or ""),
            previous_count=int(observation.get("consecutive_observations") or 0),
        )

    assert _public_blocker(KeyboardInterrupt()) == (
        "training_production_live_failed:KeyboardInterrupt"
    )
    assert observation["terminal"] is True
    assert observation["code"] == "heterogeneous_placement_capacity_exhausted"
    assert observation["stage_id"] == 4


def test_production_packages_render_bounded_backend_replacements(tmp_path) -> None:
    common = {
        "owner": "fixture-owner",
        "coordinator_url": "https://private.invalid",
        "coordinator_token": "private-token",
        "transport_optimization_after_step": 20,
    }
    cpu = build_cpu_gpu_package(
        tmp_path / "cpu",
        slug="fixture-cpu",
        role="cpu",
        replacement_after_steps=70,
        **common,
    )
    gpu = build_cpu_gpu_package(
        tmp_path / "gpu",
        slug="fixture-gpu",
        role="gpu_a",
        replacement_after_steps=50,
        **common,
    )
    tpu = build_tpu_package(
        tmp_path / "tpu",
        slug="fixture-tpu",
        replacement_after_steps=80,
        **common,
    )
    for package in (cpu, gpu, tpu):
        py_compile.compile(
            str(package["package_dir"] + "/kernel.py"), doraise=True
        )
    cpu_source = (tmp_path / "cpu/private-kernel/kernel.py").read_text()
    gpu_source = (tmp_path / "gpu/private-kernel/kernel.py").read_text()
    tpu_source = (tmp_path / "tpu/private-kernel/kernel.py").read_text()

    assert cpu["replacement_process_included"] is True
    assert cpu["replacement_after_steps"] == 70
    assert '"cpu_old",' in cpu_source
    assert '"cpu_replacement",' in cpu_source
    assert gpu["replacement_after_steps"] == 50
    assert "REPLACEMENT_AFTER_STEPS = 50" in gpu_source
    assert '"automatic_takeover_observed": True' in gpu_source
    assert tpu["replacement_after_steps"] == 80
    assert "REPLACEMENT_AFTER_STEPS = 80" in tpu_source
    assert (
        'int(workers[0].get("steps_completed") or 0) == REPLACEMENT_AFTER_STEPS'
        in tpu_source
    )
    assert "TRANSPORT_OPTIMIZATION_AFTER_STEP = 20" in tpu_source


def test_local_pack_is_valid_blocker_and_strict_checker_rejects_it(tmp_path) -> None:
    workflow = run_workflow_probe(tmp_path / "workflow")
    fault = run_fault_probe(tmp_path / "fault")
    report = pack(
        workflow_path=tmp_path
        / "workflow"
        / "training_heterogeneous_production_workflow_probe.json",
        fault_path=tmp_path
        / "fault"
        / "training_heterogeneous_production_fault_probe.json",
        output_dir=tmp_path / "rc",
    )

    default = check_report(report)
    strict = check_report(report, require_ready=True)

    assert workflow["ok"] is True
    assert fault["ok"] is True
    assert default["ok"] is True
    assert default["training_production_rc_ready"] is False
    assert strict["ok"] is False
    assert "training_production_rc_not_ready" in strict["errors"]


def test_regression_pack_reads_structured_junit_without_raw_command(tmp_path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="3" failures="0" errors="0" skipped="1" time="2.5">
  <testcase classname="tests.test_training_cli" name="one" time="1.0" />
  <testcase classname="tests.test_training_public_safety" name="two" time="1.0" />
  <testcase classname="" name="tests.test_heterogeneous_jax_qwen_training" time="0.5"><skipped /></testcase>
</testsuite></testsuites>
""",
        encoding="utf-8",
    )

    report = pack_regression(
        junit_path=junit,
        output_dir=tmp_path / "summary",
        warning_count=2,
    )

    assert report["ok"] is True
    assert report["passed"] == 2
    assert report["skipped"] == 1
    assert report["failed"] == 0
    assert report["test_file_count"] == 3
    assert report["warnings"] == 2
    assert report["raw_commands_public"] is False


def ready_report() -> dict:
    performance = {
        "schema": "crowdtensor_heterogeneous_training_performance_comparison_v1",
        "performance_gate_passed": True,
        "same_workload_verified": True,
        "same_topology_verified": True,
        "workload_or_topology_reduction_used": False,
        "throughput_improvement_fraction": 0.2,
        "p50_latency_improvement_fraction": 0.18,
        "p95_regression_fraction": -0.1,
        "maximum_p95_regression_fraction": 0.05,
        "baseline_window_count": 5,
        "candidate_window_count": 5,
    }
    replacements = {}
    for index, (kind, step, stage_id, old_role, new_role) in enumerate(
        (
            ("cuda", 50, 1, "gpu_a", "gpu_b"),
            ("cpu", 70, 4, "cpu", "cpu"),
            ("jax_tpu", 80, 2, "tpu", "tpu"),
        ),
        start=1,
    ):
        replacements[kind] = {
            "verified": True,
            "identity_changed": True,
            "old_worker_drained": True,
            "replacement_worker_accepted": True,
            "same_stage_handoff_verified": True,
            "contiguous_step_handoff_verified": True,
            "generation_fencing_verified": True,
            "checkpoint_restore_verified": True,
            "checkpoint_ready_event_matched": kind != "cuda",
            "checkpoint_download_count": 1,
            "removed_after_step": step,
            "replacement_first_step": step + 1,
            "restored_checkpoint_step": step,
            "previous_generation": index,
            "replacement_generation": index + 1,
            "old_identity_hash": "sha256:" + str(index) * 64,
            "replacement_identity_hash": "sha256:" + str(index + 3) * 64,
            "old_stage_ids": [stage_id],
            "replacement_stage_ids": [stage_id],
            "source_checkpoint_archive_hashes": ["sha256:" + "a" * 64],
            "replacement_checkpoint_archive_hashes": ["sha256:" + "b" * 64],
            "restore_evidence_source": (
                "generation_fenced_contiguous_checkpoint_handoff"
                if kind == "cuda"
                else "stage_ready_history"
            ),
            "replacement_selection": (
                "cross_kernel_dynamic_reassignment"
                if kind == "cuda"
                else "designated_replacement"
            ),
            "old_kernel_role": old_role,
            "replacement_kernel_role": new_role,
        }
    kernel_evidence = [
        {
            "kernel_role": role,
            "raw_ok": role != "gpu_a",
            "effective_ok": True,
            "worker_results_valid": True,
            "cleanup_verified": True,
            "automatic_cross_kernel_takeover_observed": role == "gpu_a",
            "raw_failure_reclassified": role == "gpu_a",
            "kernel_report_hash": "sha256:" + character * 64,
        }
        for role, character in zip(
            ("cpu", "gpu_a", "gpu_b", "tpu"), ("c", "d", "e", "f")
        )
    ]
    report = {
        "schema": SCHEMA,
        "model_id": "Qwen/Qwen2.5-7B",
        "minimum_required_steps": 100,
        "minimum_required_duration_seconds": 3600,
        "minimum_performance_improvement_fraction": 0.15,
        "training_production_rc_ready": True,
        "workflow_summary": {
            "source_schema": "crowdtensor_heterogeneous_training_production_workflow_probe_v1",
            "workflow_verified": True,
            "monitoring_contract_verified": True,
            "cleanup_verified": True,
            "next_resume_command_uses_public_placeholder": True,
        },
        "fault_summary": {
            "source_schema": "crowdtensor_heterogeneous_training_production_fault_probe_v1",
            "fault_injection_suite_ready": True,
            "generation_fencing_verified": True,
            "lease_reclaim_verified": True,
            "circuit_breaker_verified": True,
            "checkpoint_fallback_verified": True,
            "coordinator_journal_recovery_verified": True,
            "bounded_retry_verified": True,
            "cleanup_verified": True,
        },
        "live_summary": {
            "source_schema": LIVE_SCHEMA,
            "source_content_hash_valid": True,
            "live_run_performed": True,
            "external_runtime_verified": True,
            "accepted_providers": [
                "kaggle_cpu",
                "kaggle_cuda",
                "kaggle_jax_tpu",
            ],
            "committed_steps": list(range(1, 101)),
            "committed_step_count": 100,
            "soak_duration_seconds": 4000.0,
            "full_live_gate_elapsed_seconds": 5000.0,
            "maximum_checkpoint_interval_steps": 1,
            "finite_updates_all_stages": True,
            "changed_lora_hashes_all_stages": True,
            "atomic_ledger_verified": True,
            "checkpoint_integrity_verified": True,
            "adapter_cpu_reload_verified": True,
            "activation_gradient_transfer_verified": True,
            "monitoring_live_verified": True,
            "coordinator_restart_live_verified": True,
            "stale_result_rejected": True,
            "worker_replacements": replacements,
            "kernel_evidence": kernel_evidence,
            "effective_kernel_evidence_verified": True,
            "performance": performance,
            "optimization_summary": {
                "performance_window_count_per_phase": 5,
                "inline_tensor_message_upload_count": 500,
                "inline_tensor_message_download_count": 500,
                "large_payload_connection_isolation_count": 100,
                "persistent_http_max_body_bytes": 4 * 1024 * 1024,
            },
            "benchmark": {
                "step_throughput_per_second": 0.1,
                "p50_step_latency_seconds": 10.0,
                "p95_step_latency_seconds": 12.0,
                "checkpoint_overhead_seconds": 2.0,
                "transfer_bytes": 1000,
            },
            "cleanup": {
                "all_remote_kernels_deleted": True,
                "temporary_private_packages_removed": True,
                "coordinator_stopped": True,
                "tunnel_stopped": True,
                "tensor_payloads_removed": True,
                "live_resources_left_running": False,
            },
        },
        "quality_summary": {"passed": 300, "failed": 0},
        "blockers": [],
        "public_artifact_safe": True,
        "credential_values_public": False,
        "private_paths_public": False,
    }
    report["content_hash"] = stable_hash(report)
    return report


def test_strict_checker_accepts_complete_evidence_and_rejects_tampering() -> None:
    report = ready_report()
    checked = check_report(report, require_ready=True)
    tampered = deepcopy(report)
    tampered["live_summary"]["committed_steps"].remove(51)
    tampered["content_hash"] = stable_hash(
        {key: value for key, value in tampered.items() if key != "content_hash"}
    )
    rejected = check_report(tampered, require_ready=True)
    transport_tampered = deepcopy(report)
    transport_tampered["live_summary"]["optimization_summary"][
        "large_payload_connection_isolation_count"
    ] = 0
    transport_tampered["content_hash"] = stable_hash(
        {
            key: value
            for key, value in transport_tampered.items()
            if key != "content_hash"
        }
    )
    transport_rejected = check_report(transport_tampered, require_ready=True)
    replacement_tampered = deepcopy(report)
    replacement_tampered["live_summary"]["worker_replacements"]["cuda"][
        "checkpoint_download_count"
    ] = 0
    replacement_tampered["content_hash"] = stable_hash(
        {
            key: value
            for key, value in replacement_tampered.items()
            if key != "content_hash"
        }
    )
    replacement_rejected = check_report(
        replacement_tampered, require_ready=True
    )

    assert checked["ok"] is True
    assert checked["training_production_rc_ready"] is True
    assert rejected["ok"] is False
    assert "training_production_commit_ledger_invalid" in rejected["errors"]
    assert transport_rejected["ok"] is False
    assert (
        "training_production_large_payload_isolation_missing"
        in transport_rejected["errors"]
    )
    assert replacement_rejected["ok"] is False
    assert (
        "training_production_live_replacement_missing:cuda"
        in replacement_rejected["errors"]
    )
