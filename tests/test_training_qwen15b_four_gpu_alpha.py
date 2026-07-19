from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.training_qwen15b_four_gpu_alpha_check import SCHEMA, check
from scripts.training_qwen15b_four_gpu_alpha_pack import pack


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dist/training-qwen15b-source-20260712-r1/training_qwen15b_source_probe.json"
DATASET = ROOT / "dist/training-qwen15b-dataset-20260712-r1/training_qwen15b_dataset_prepare.json"
MODEL = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"


def _hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _ready(role: str, run_kind: str) -> list[dict]:
    stage_ids = [0, 1] if role == "kernel_a" else [2, 3]
    return [
        {
            "stage_id": stage,
            "pid": 1000 + stage + (100 if run_kind == "resumed" else 0),
            "device": f"cuda:{stage % 2}",
            "cuda_live": True,
            "grad_scaler_enabled": True,
            "load_report": {
                "stage_id": stage,
                "layer_start": stage * 7,
                "layer_end": (stage + 1) * 7,
                "loaded_layer_indexes": list(range(stage * 7, (stage + 1) * 7)),
                "meta_device_construction": True,
                "loaded_full_model": False,
                "stage_owned_module_construction": True,
                "foreign_layer_count": 0,
                "only_lora_trainable": True,
                "gradient_checkpointing": True,
                "lora_injected": True,
                "trainable_parameter_count": 128,
                "trainable_parameter_dtypes": ["float32"],
                "fp32_lora_parameters_for_grad_scaler": True,
                "cuda_fp16_autocast": False,
                "cuda_fp32_stable_compute": True,
                "stage_boundary_dtype": "float16",
            },
        }
        for stage in stage_ids
    ]


def _worker(role: str) -> dict:
    stage_ids = [0, 1] if role == "kernel_a" else [2, 3]

    def run_report(run_kind: str) -> dict:
        ready = {item["stage_id"]: item for item in _ready(role, run_kind)}
        restart = (
            {
                "stage_id": 2,
                "after_step": 4,
                "old_pid": ready[2]["pid"],
                "new_pid": 2202,
                "new_pid_verified": True,
                "forced_stop_verified": True,
                "checkpoint_resume_verified": True,
                "resumed_global_step": 4,
                "resumed_dataset_cursor": 16,
            }
            if role == "kernel_b" and run_kind == "resumed"
            else None
        )
        events = []
        step_reports = []
        payloads = []
        run_offset = 10_000_000_000 if run_kind == "resumed" else 0
        for step in range(8):
            stages = []
            for stage in stage_ids:
                pid = ready[stage]["pid"]
                if restart and stage == 2 and step >= 4:
                    pid = restart["new_pid"]
                stages.append(
                    {
                        "stage_id": stage,
                        "pid": pid,
                        "device": f"cuda:{stage % 2}",
                        "global_step": step + 1,
                        "dataset_cursor": (step + 1) * 4,
                        "gradient_scale_before": 128.0,
                        "gradient_scale_after": 128.0,
                        "lora_gradient_norm": 0.5 + stage / 10,
                        "gradient_clip_norm": 1.0,
                        "gradient_clipping_applied": True,
                        "optimizer_step_applied": True,
                        "checkpoint_hash": _hash(f"{role}-{run_kind}-{step}-{stage}-checkpoint"),
                        "adapter_tensor_hash": _hash(f"{role}-{run_kind}-{step}-{stage}-adapter"),
                        "peak_allocated_bytes": 1_000_000 + stage,
                        "peak_reserved_bytes": 2_000_000 + stage,
                    }
                )
            step_reports.append({"step": step + 1, "stages": stages})
            for microbatch in range(4):
                base = 1_000_000 + run_offset + step * 1_000_000 + microbatch * 10_000

                def event(stage: int, operation: str, start: int, end: int, *, loss=None):
                    pid = ready[stage]["pid"]
                    if restart and stage == 2 and step >= 4:
                        pid = restart["new_pid"]
                    events.append(
                        {
                            "run_kind": run_kind,
                            "operation": operation,
                            "stage_id": stage,
                            "step": step,
                            "microbatch": microbatch,
                            "pid": pid,
                            "device": f"cuda:{stage % 2}",
                            "started_ns": base + start,
                            "ended_ns": base + end,
                            "loss": loss,
                        }
                    )

                if role == "kernel_a":
                    event(0, "forward", 100, 1_000)
                    event(1, "forward", 200, 1_100)
                    event(1, "backward", 2_100, 2_800)
                    event(0, "backward", 2_200, 2_900)
                else:
                    event(2, "forward", 300, 1_200)
                    event(3, "forward_backward", 400, 1_300, loss=2.5 - step * 0.05)
                    event(2, "backward", 2_000, 2_700)
                for kind in ("activation", "gradient"):
                    payloads.append(
                        {
                            "kind": kind,
                            "step": step,
                            "microbatch": microbatch,
                            "payload_hash": _hash(f"{run_kind}-{kind}-{step}-{microbatch}"),
                        }
                    )
        if restart:
            events.extend(
                [
                    {
                        "run_kind": run_kind,
                        "operation": "stage_stopped",
                        "stage_id": 2,
                        "step": 4,
                        "microbatch": -1,
                        "pid": restart["old_pid"],
                        "device": "cuda:0",
                        "started_ns": 0,
                        "ended_ns": 0,
                        "loss": None,
                    },
                    {
                        "run_kind": run_kind,
                        "operation": "stage_restarted",
                        "stage_id": 2,
                        "step": 4,
                        "microbatch": -1,
                        "pid": restart["new_pid"],
                        "device": "cuda:0",
                        "started_ns": 0,
                        "ended_ns": 0,
                        "loss": None,
                    },
                ]
            )
        losses = [2.5 - step * 0.05 + microbatch * 0.001 for step in range(8) for microbatch in range(4)]
        means = [sum(losses[index * 4 : index * 4 + 4]) / 4 for index in range(8)]
        return {
            "schema": "crowdtensor_qwen15b_four_gpu_runtime_v1",
            "role": role,
            "run_kind": run_kind,
            "steps_completed": 8,
            "microbatches_per_step": 4,
            "real_forward": True,
            "real_backward": True,
            "events": events,
            "payloads": payloads,
            "step_reports": step_reports,
            "dataset_row_indexes": list(range(32)) if role == "kernel_a" else [],
            "adapter_hashes": {str(stage): _hash(f"{role}-{run_kind}-{stage}-final") for stage in stage_ids},
            "losses": losses if role == "kernel_b" else [],
            "step_mean_losses": means if role == "kernel_b" else [],
            "loss_start": means[0] if role == "kernel_b" else None,
            "loss_end": means[-1] if role == "kernel_b" else None,
            "loss_reduced": True if role == "kernel_b" else None,
            "controlled_restarts": [restart] if restart else [],
        }

    runs = {run_kind: run_report(run_kind) for run_kind in ("baseline", "resumed")}
    checkpoint_hash = _hash(f"{role}-checkpoint-archive")
    worker = {
        "model_id": MODEL,
        "model_revision": REVISION,
        "parameter_count": 1_543_714_304,
        "base_weights_frozen": True,
        "positive_lora_gradient_norms": True,
        "stage_ready": {
            run_kind: _ready(role, run_kind) for run_kind in ("baseline", "resumed")
        },
        "runs": runs,
        "resume_adapter_equivalence": {"verified": True},
        "controlled_restart_verified": role == "kernel_b",
        "evaluation": {},
        "export": {},
    }
    if role == "kernel_b":
        worker["resume_loss_equivalence"] = {
            "verified": True,
            "loss_count": 32,
            "maximum_absolute_difference": 0.0,
            "atol": 0.005,
            "rtol": 0.005,
        }
        worker["evaluation"] = {
            "evaluation_verified": True,
            "standard_peft_cpu_load": True,
            "standard_peft_cuda_load": True,
            "cuda_compute_dtype": "float32",
            "adapter_changes_logits": True,
            "validation_loss_reduced": True,
        }
        worker["export"] = {
            "standard_peft_format": True,
            "model_id": MODEL,
            "model_revision": REVISION,
            "layer_indexes": list(range(28)),
            "adapter_file_hash": _hash("adapter-model"),
            "adapter_config_hash": _hash("adapter-config"),
            "adapter_tensor_count": 28,
            "adapter_tensor_names_hash": _hash("adapter-tensor-names"),
        }
    outer = {
        "ok": True,
        "role": role,
        "worker": worker,
        "dependencies": {
            "transformers": "5.9.0",
            "peft": "0.19.1",
            "safetensors": "0.7.0",
            "torchao_before": "0.10.0",
            "torchao_after": "",
            "incompatible_torchao_removed": True,
        },
        "dependency_smoke": {
            "schema": "crowdtensor_qwen15b_dependency_smoke_v1",
            "verified": True,
            "peft_import_verified": True,
            "lora_injection_verified": True,
            "forward_verified": True,
            "backward_verified": True,
            "only_lora_trainable": True,
            "positive_lora_gradient_count": 2,
            "started_ns": 100,
            "ended_ns": 200,
        },
        "cuda_mixed_precision_smoke": {
            "schema": "crowdtensor_qwen15b_cuda_mixed_precision_smoke_v1",
            "verified": True,
            "cuda_live": True,
            "fp32_lora_parameters": True,
            "fp32_stable_compute": True,
            "fp16_stage_boundary": True,
            "grad_scaler_unscale_step_verified": True,
            "finite_gradient_count": 2,
            "started_ns": 210,
            "ended_ns": 290,
        },
        "stage_runtime_started_ns": 300,
        "dependency_smoke_before_stage_runtime": True,
        "checkpoint_bundle": {"file_hash": checkpoint_hash},
    }
    if role == "kernel_b":
        outer["adapter_bundle"] = {"file_hash": _hash("adapter-archive")}
    return outer


def _rendezvous(workers: list[dict]) -> dict:
    payloads = []
    for run_kind in ("baseline", "resumed"):
        for kind in ("activation", "gradient"):
            for step in range(8):
                for microbatch in range(4):
                    payloads.append(
                        {
                            "run_kind": run_kind,
                            "kind": kind,
                            "step": step,
                            "microbatch": microbatch,
                            "producer_role": "kernel_a" if kind == "activation" else "kernel_b",
                            "payload_hash": _hash(f"{run_kind}-{kind}-{step}-{microbatch}"),
                            "byte_count": 1024,
                            "tensor_count": 2 if kind == "activation" else 1,
                            "created_at": 1.0 + step,
                        }
                    )
    payloads.append(
        {
            "run_kind": "resumed",
            "kind": "stage_adapter",
            "step": 8,
            "microbatch": -1,
            "producer_role": "kernel_a",
            "payload_hash": _hash("stage-adapter"),
            "byte_count": 4096,
            "tensor_count": 28,
            "created_at": 10.0,
        }
    )
    registrations = []
    for outer in workers:
        role = outer["role"]
        ready = outer["worker"]["stage_ready"]["resumed"]
        pids = [item["pid"] for item in ready]
        if role == "kernel_b":
            pids[0] = outer["worker"]["runs"]["resumed"]["controlled_restarts"][0]["new_pid"]
        registrations.append(
            {
                "role": role,
                "stage_ids": [item["stage_id"] for item in ready],
                "stage_pids": pids,
                "cuda_devices": ["cuda:0", "cuda:1"],
                "cuda_device_name_hashes": [_hash(f"{role}-gpu0"), _hash(f"{role}-gpu1")],
                "cuda_live": True,
                "worker_id_hash": _hash(role),
            }
        )
    return {
        "schema": "crowdtensor_qwen15b_four_gpu_rendezvous_v1",
        "registered_roles": ["kernel_a", "kernel_b"],
        "registrations": registrations,
        "payloads": payloads,
        "events": [],
        "completions": [
            {
                "role": role,
                "ok": True,
                "baseline_steps_completed": 8,
                "resumed_steps_completed": 8,
                "stage_ids": list((0, 1) if role == "kernel_a" else (2, 3)),
            }
            for role in ("kernel_a", "kernel_b")
        ],
    }


def _checkpoint_bundle(role: str) -> dict:
    stage_ids = (0, 1) if role == "kernel_a" else (2, 3)
    summaries = []
    for run_kind in ("baseline", "resumed"):
        for stage in stage_ids:
            summaries.append(
                {
                    "run_kind": run_kind,
                    "stage_id": stage,
                    "layer_start": stage * 7,
                    "layer_end": (stage + 1) * 7,
                    "global_step": 8,
                    "optimizer_step": 8,
                    "dataset_cursor": 32,
                    "device": f"cuda:{stage % 2}",
                    "model_id": MODEL,
                    "model_revision": REVISION,
                    "component_hashes_verified": True,
                    "grad_scaler_state_present": True,
                    "rng_state_present": True,
                    "adapter_tensor_count": 14,
                    "adapter_tensor_hash": _hash(f"{role}-{run_kind}-{stage}-manifest-adapter"),
                    "manifest_content_hash": _hash(f"{role}-{run_kind}-{stage}-manifest"),
                    "manifest_content_hash_verified": True,
                }
            )
    return {
        "role": role,
        "verified": True,
        "preserved": True,
        "worker_hash_match": True,
        "archive_safe": True,
        "unique_archive_members": True,
        "all_checkpoint_files_hash_verified": True,
        "all_final_steps_verified": True,
        "model_revision_verified": True,
        "all_manifest_content_hashes_verified": True,
        "checkpoint_manifest_count": 4,
        "file_hash": _hash(f"{role}-checkpoint-archive"),
        "byte_count": 8192,
        "manifest_summaries": summaries,
    }


def _allocation_budget() -> dict:
    return {
        "schema": "crowdtensor_qwen15b_four_gpu_allocation_budget_summary_v1",
        "amendment_present": True,
        "amendment_valid": True,
        "original_attempt_limit": 2,
        "effective_attempt_limit": None,
        "total_attempt_limit_unbounded": True,
        "one_attempt_per_probe_invocation": True,
        "automatic_retry_loop": False,
        "additional_attempts_authorized": None,
        "prior_attempts_preserved": True,
        "same_authorized_account_only": True,
        "allocation_timeout_seconds": 1800,
        "authorization_hash": _hash("unbounded-allocation-authorization"),
        "authorization_text_public": False,
        "credential_values_public": False,
        "public_artifact_safe": True,
    }


def _complete_report() -> dict:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    workers = [_worker("kernel_a"), _worker("kernel_b")]
    all_events = [
        event
        for outer in workers
        for run in outer["worker"]["runs"].values()
        for event in run["events"]
        if event["operation"] in {"forward", "backward", "forward_backward"}
    ]
    live = {
        "ok": True,
        "qwen15b_four_gpu_alpha_verified": True,
        "requested_model": MODEL,
        "requested_model_revision": REVISION,
        "requested_kernel_count": 2,
        "same_authorized_account": True,
        "multi_account_gate_substitution": False,
        "attempt": 5,
        "attempt_limit": 5,
        "allocation_budget": _allocation_budget(),
        "max_observed_running_kernel_count": 2,
        "worker_reports": workers,
        "rendezvous": _rendezvous(workers),
        "checkpoint_bundles": [_checkpoint_bundle("kernel_a"), _checkpoint_bundle("kernel_b")],
        "adapter_bundle": {
            "verified": True,
            "preserved": True,
            "worker_hash_match": True,
            "archive_safe": True,
            "unique_archive_members": True,
            "standard_peft_layout": True,
            "base_model_verified": True,
            "model_revision_verified": True,
            "safetensors_header_verified": True,
            "file_hash": _hash("adapter-archive"),
            "adapter_file_hash": _hash("adapter-model"),
            "adapter_config_hash": _hash("adapter-config"),
            "byte_count": 4096,
            "adapter_tensor_count": 28,
            "adapter_tensor_names_hash": _hash("adapter-tensor-names"),
            "layer_indexes": list(range(28)),
        },
        "evidence": {
            "verified": True,
            "four_stage_compute_overlap_verified": True,
            "maximum_four_stage_overlap": {
                "run_kind": "baseline",
                "step": 0,
                "started_ns": 1_000_400,
                "ended_ns": 1_001_000,
                "duration_ns": 600,
            },
            "interval_count": len(all_events),
            "activation_payload_count": 64,
            "gradient_payload_count": 64,
            "resume_adapter_equivalence_verified": True,
            "resume_loss_equivalence_verified": True,
            "controlled_restart_verified": True,
            "checkpoint_archives_verified": True,
            "adapter_archive_verified": True,
        },
        "cleanup": {
            "kernels_deleted": True,
            "only_attempt_kernel_refs_targeted": True,
            "private_packages_removed": True,
            "coordinator_stopped": True,
            "tunnel_stopped": True,
            "private_runtime_removed": True,
            "rendezvous_private_payloads_removed": True,
            "checkpoint_archives_verified_before_cleanup": True,
        },
    }
    allocation_budget = {
        **_allocation_budget(),
        "attempts_used": 5,
        "successful_attempt": 5,
        "probe_invocation_attempt_ceiling": 5,
        "budget_exhausted": False,
        "ledger_history_must_be_preserved": True,
        "additional_attempt_requires_explicit_user_amendment": False,
        "probe_invocation_ceiling_is_not_total_policy_limit": True,
    }
    return {
        "schema": SCHEMA,
        "goal_achieved": True,
        "source": source,
        "dataset": dataset,
        "test_summary": {"ok": True, "passed": 100, "failed": 0},
        "live_report": live,
        "allocation_budget": allocation_budget,
        "allocation_history": {
            "schema": "crowdtensor_qwen15b_four_gpu_alpha_allocation_history_v1",
            "ledger_present": True,
            "attempt_count": 5,
            "completed_attempt_count": 5,
            "attempt_numbers_sequential": True,
            "attempt_records_hash": _hash("allocation-attempt-records"),
            "successful_attempt": 5,
            "verified_attempt_numbers": [5],
            "successful_attempt_matches_ledger": True,
            "immutable_history_preserved": True,
            "public_artifact_safe": True,
        },
        "runtime_remediation": {
            "fp16_autocast_non_finite_activation_observed": True,
            "fp16_autocast_abandoned": True,
            "frozen_stage_weight_compute_dtype": "float32",
            "lora_parameter_dtype": "float32",
            "cuda_fp16_autocast": False,
            "cuda_fp32_stable_compute": True,
            "fp16_stage_boundary_transport": True,
            "grad_scaler_unscale_step_verified": True,
            "non_finite_activation_logits_loss_gradient_gates": True,
            "remediation_local_tests_passed": True,
            "remediation_gpu_live_verified": True,
        },
        "artifacts": {"precision_failure_report": {"present": True}},
        "activation_values_public": False,
        "gradient_values_public": False,
        "adapter_tensor_values_public": False,
        "token_ids_public": False,
        "raw_training_text_public": False,
        "credentials_public": False,
        "coordinator_url_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def test_strict_checker_accepts_complete_real_four_gpu_contract() -> None:
    result = check(_complete_report(), require_ready=True)
    assert result["ok"] is True
    assert result["qwen15b_four_gpu_alpha_ready"] is True
    assert result["errors"] == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("toy_model", "qwen15b_source_parameter_count_below_1b"),
        ("full_model_load", "qwen15b_stage_owned_loading_invalid"),
        ("cpu_stage", "qwen15b_cuda_stage_execution_missing"),
        ("no_overlap", "qwen15b_four_gpu_compute_overlap_missing"),
        ("missing_gradient", "qwen15b_cross_kernel_gradient_transport_missing"),
        ("loss_not_reduced", "qwen15b_baseline_loss_not_reduced"),
        ("fake_resume", "qwen15b_controlled_stage_restart_missing"),
        ("no_cpu_peft", "qwen15b_standard_peft_evaluation_missing"),
        ("cleanup_incomplete", "qwen15b_cleanup_kernels_deleted_missing"),
        ("payload_leak", "qwen15b_public_payload_b64"),
        ("finite_allocation_budget", "qwen15b_unbounded_allocation_authorization_missing"),
        ("missing_precision_remediation", "qwen15b_fp32_stable_compute_remediation_missing"),
        ("allocation_history_incomplete", "qwen15b_allocation_attempt_history_invalid"),
    ],
)
def test_strict_checker_rejects_false_alpha_evidence(mutation: str, expected: str) -> None:
    report = copy.deepcopy(_complete_report())
    if mutation == "toy_model":
        report["source"]["source"]["parameter_count"] = 500_000_000
    elif mutation == "full_model_load":
        report["live_report"]["worker_reports"][0]["worker"]["stage_ready"]["baseline"][0][
            "load_report"
        ]["loaded_full_model"] = True
    elif mutation == "cpu_stage":
        report["live_report"]["worker_reports"][0]["worker"]["stage_ready"]["baseline"][0][
            "device"
        ] = "cpu"
    elif mutation == "no_overlap":
        report["live_report"]["evidence"]["four_stage_compute_overlap_verified"] = False
    elif mutation == "missing_gradient":
        report["live_report"]["evidence"]["gradient_payload_count"] = 63
    elif mutation == "loss_not_reduced":
        run = report["live_report"]["worker_reports"][1]["worker"]["runs"]["baseline"]
        run["loss_end"] = 3.0
        run["loss_reduced"] = False
    elif mutation == "fake_resume":
        report["live_report"]["worker_reports"][1]["worker"][
            "controlled_restart_verified"
        ] = False
    elif mutation == "no_cpu_peft":
        report["live_report"]["worker_reports"][1]["worker"]["evaluation"][
            "standard_peft_cpu_load"
        ] = False
    elif mutation == "cleanup_incomplete":
        report["live_report"]["cleanup"]["kernels_deleted"] = False
    elif mutation == "payload_leak":
        report["live_report"]["payload_b64"] = "private"
    elif mutation == "finite_allocation_budget":
        report["allocation_budget"]["total_attempt_limit_unbounded"] = False
    elif mutation == "missing_precision_remediation":
        report["runtime_remediation"]["cuda_fp32_stable_compute"] = False
    elif mutation == "allocation_history_incomplete":
        report["allocation_history"]["completed_attempt_count"] = 4
    report["goal_achieved"] = False
    result = check(report, require_ready=True)
    assert result["ok"] is False
    assert expected in result["errors"]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_step_record", "qwen15b_per_step_records_incomplete"),
        ("missing_rendezvous_payload", "qwen15b_rendezvous_payload_identity_set_invalid"),
        ("forged_overlap_summary", "qwen15b_four_gpu_compute_overlap_missing"),
        ("restart_pid_chronology", "qwen15b_compute_event_metadata_invalid"),
        ("checkpoint_manifest_removed", "qwen15b_checkpoint_manifest_coverage_invalid"),
        ("dependency_smoke_false", "qwen15b_dependency_lora_smoke_missing"),
        ("payload_tensor_metadata", "qwen15b_rendezvous_payload_metadata_invalid"),
        ("adapter_layer_coverage", "qwen15b_adapter_archive_integrity_invalid"),
        ("zero_step_gradient", "qwen15b_per_step_stage_record_invalid"),
        ("dataset_order_reused", "qwen15b_dataset_cursor_order_invalid"),
    ],
)
def test_strict_checker_recomputes_detailed_live_evidence(
    mutation: str, expected: str
) -> None:
    report = copy.deepcopy(_complete_report())
    live = report["live_report"]
    if mutation == "missing_step_record":
        live["worker_reports"][0]["worker"]["runs"]["baseline"]["step_reports"].pop()
    elif mutation == "missing_rendezvous_payload":
        live["rendezvous"]["payloads"].pop(0)
    elif mutation == "forged_overlap_summary":
        for outer in live["worker_reports"]:
            for run in outer["worker"]["runs"].values():
                for event in run["events"]:
                    if event["stage_id"] == 3 and event["operation"] == "forward_backward":
                        event["started_ns"] += 5_000
                        event["ended_ns"] += 5_000
    elif mutation == "restart_pid_chronology":
        resumed = live["worker_reports"][1]["worker"]["runs"]["resumed"]
        old_pid = resumed["controlled_restarts"][0]["old_pid"]
        for event in resumed["events"]:
            if event["stage_id"] == 2 and event["operation"] in {"forward", "backward"} and event["step"] >= 4:
                event["pid"] = old_pid
    elif mutation == "checkpoint_manifest_removed":
        live["checkpoint_bundles"][0]["manifest_summaries"].pop()
    elif mutation == "dependency_smoke_false":
        live["worker_reports"][0]["dependency_smoke"]["verified"] = False
    elif mutation == "payload_tensor_metadata":
        live["rendezvous"]["payloads"][0]["tensor_count"] = 0
    elif mutation == "adapter_layer_coverage":
        live["adapter_bundle"]["layer_indexes"].pop()
    elif mutation == "zero_step_gradient":
        live["worker_reports"][0]["worker"]["runs"]["baseline"]["step_reports"][0][
            "stages"
        ][0]["lora_gradient_norm"] = 0.0
    elif mutation == "dataset_order_reused":
        live["worker_reports"][0]["worker"]["runs"]["baseline"][
            "dataset_row_indexes"
        ] = [index % 4 for index in range(32)]
    report["goal_achieved"] = False
    result = check(report, require_ready=True)
    assert result["ok"] is False
    assert expected in result["errors"]


def test_pack_writes_valid_blocker_without_claiming_live_success(tmp_path) -> None:
    tests = tmp_path / "tests.json"
    tests.write_text(json.dumps({"ok": True, "passed": 21, "failed": 0}), encoding="utf-8")
    report = pack(
        tmp_path / "alpha",
        source_report=SOURCE,
        dataset_report=DATASET,
        test_summary=tests,
        live_report=None,
    )
    assert report["goal_achieved"] is False
    assert report["checker"]["ok"] is True
    assert report["checker"]["qwen15b_four_gpu_alpha_ready"] is False
    strict = check(report, require_ready=True)
    assert strict["ok"] is False
    assert "qwen15b_live_alpha_evidence_missing" in strict["errors"]


def test_pack_preserves_unbounded_allocation_and_precision_remediation(tmp_path) -> None:
    complete = _complete_report()
    tests = tmp_path / "tests.json"
    live = tmp_path / "live.json"
    ledger = tmp_path / "allocation_attempts.json"
    precision_failure = tmp_path / "attempt4.json"
    tests.write_text(json.dumps({"ok": True, "passed": 310, "failed": 0}), encoding="utf-8")
    live.write_text(json.dumps(complete["live_report"]), encoding="utf-8")
    ledger.write_text(
        json.dumps(
            {
                "qwen15b_four_gpu_attempts": [
                    {
                        "attempt": attempt,
                        "completed": True,
                        "outcome": "verified" if attempt == 5 else "acceptance_incomplete",
                    }
                    for attempt in range(1, 6)
                ]
            }
        ),
        encoding="utf-8",
    )
    precision_failure.write_text(
        json.dumps({"blockers": ["non_finite_stage_activation:stage0"]}),
        encoding="utf-8",
    )
    report = pack(
        tmp_path / "alpha",
        source_report=SOURCE,
        dataset_report=DATASET,
        test_summary=tests,
        live_report=live,
        allocation_ledger=ledger,
        precision_failure_report=precision_failure,
    )
    assert report["goal_achieved"] is True
    assert report["allocation_budget"]["total_attempt_limit_unbounded"] is True
    assert report["allocation_budget"]["effective_attempt_limit"] is None
    assert report["allocation_history"]["verified_attempt_numbers"] == [5]
    assert report["runtime_remediation"]["cuda_fp32_stable_compute"] is True
    assert report["runtime_remediation"]["incompatible_torchao_pre_0_16_observed"] is True
    assert report["runtime_remediation"]["kaggle_observed_torchao_version"] == "0.10.0"
    assert check(report, require_ready=True)["ok"] is True
