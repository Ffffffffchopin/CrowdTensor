#!/usr/bin/env python3
"""Check the Qwen 1.5B four-GPU Pipeline Training Alpha artifact."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path
from typing import Any


SCHEMA = "crowdtensor_training_qwen15b_four_gpu_alpha_v1"
MODEL_ID = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
PARAMETER_MINIMUM = 1_000_000_000
RUN_KINDS = ("baseline", "resumed")
ROLE_STAGES = {"kernel_a": (0, 1), "kernel_b": (2, 3)}
STAGE_RANGES = {0: (0, 7), 1: (7, 14), 2: (14, 21), 3: (21, 28)}
SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _positive_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _dependency_smoke_errors(by_role: dict[str, dict[str, Any]]) -> list[str]:
    errors = []
    for role in ROLE_STAGES:
        outer = _dict(by_role.get(role))
        smoke = _dict(outer.get("dependency_smoke"))
        if (
            smoke.get("schema") != "crowdtensor_qwen15b_dependency_smoke_v1"
            or any(
                smoke.get(key) is not True
                for key in (
                    "verified",
                    "peft_import_verified",
                    "lora_injection_verified",
                    "forward_verified",
                    "backward_verified",
                    "only_lora_trainable",
                )
            )
            or int(smoke.get("positive_lora_gradient_count") or 0) < 1
        ):
            errors.append("qwen15b_dependency_lora_smoke_missing")
        smoke_started = int(smoke.get("started_ns") or 0)
        smoke_ended = int(smoke.get("ended_ns") or 0)
        cuda_smoke = _dict(outer.get("cuda_mixed_precision_smoke"))
        cuda_smoke_started = int(cuda_smoke.get("started_ns") or 0)
        cuda_smoke_ended = int(cuda_smoke.get("ended_ns") or 0)
        if (
            cuda_smoke.get("schema")
            != "crowdtensor_qwen15b_cuda_mixed_precision_smoke_v1"
            or any(
                cuda_smoke.get(key) is not True
                for key in (
                    "verified",
                    "cuda_live",
                    "fp32_lora_parameters",
                    "fp32_stable_compute",
                    "fp16_stage_boundary",
                    "grad_scaler_unscale_step_verified",
                )
            )
            or int(cuda_smoke.get("finite_gradient_count") or 0) < 1
        ):
            errors.append("qwen15b_cuda_mixed_precision_smoke_missing")
        stage_started = int(outer.get("stage_runtime_started_ns") or 0)
        if (
            smoke_started <= 0
            or smoke_ended < smoke_started
            or cuda_smoke_started <= 0
            or cuda_smoke_ended < cuda_smoke_started
            or stage_started < max(smoke_ended, cuda_smoke_ended)
            or outer.get("dependency_smoke_before_stage_runtime") is not True
        ):
            errors.append("qwen15b_dependency_smoke_not_before_stage_runtime")
        dependencies = _dict(outer.get("dependencies"))
        if (
            dependencies.get("transformers") != "5.9.0"
            or dependencies.get("peft") != "0.19.1"
            or dependencies.get("safetensors") != "0.7.0"
        ):
            errors.append("qwen15b_live_dependency_versions_invalid")
        torchao_after = str(dependencies.get("torchao_after") or "")
        match = re.match(r"^(\d+)\.(\d+)", torchao_after)
        if match and (int(match.group(1)), int(match.group(2))) < (0, 16):
            errors.append("qwen15b_incompatible_torchao_remained_after_preflight")
    return errors


def _expected_compute_keys(role: str) -> Counter[tuple[int, str, int, int]]:
    operations = (
        ((0, "forward"), (1, "forward"), (1, "backward"), (0, "backward"))
        if role == "kernel_a"
        else ((2, "forward"), (3, "forward_backward"), (2, "backward"))
    )
    return Counter(
        (stage_id, operation, step, microbatch)
        for step in range(8)
        for microbatch in range(4)
        for stage_id, operation in operations
    )


def _compute_overlap(events: list[dict[str, Any]]) -> dict[str, Any]:
    intervals = [
        {
            "stage_id": int(item.get("stage_id", -1)),
            "started_ns": int(item.get("started_ns") or 0),
            "ended_ns": int(item.get("ended_ns") or 0),
            "run_kind": str(item.get("run_kind") or ""),
            "step": int(item.get("step", -1)),
        }
        for item in events
        if int(item.get("stage_id", -1)) in {0, 1, 2, 3}
        and int(item.get("ended_ns") or 0) > int(item.get("started_ns") or 0)
    ]
    best: dict[str, Any] | None = None
    for run_kind in RUN_KINDS:
        for step in range(8):
            selected = [
                item
                for item in intervals
                if item["run_kind"] == run_kind and item["step"] == step
            ]
            points = []
            for item in selected:
                points.append((item["started_ns"], 1, item["stage_id"]))
                points.append((item["ended_ns"], -1, item["stage_id"]))
            points.sort(key=lambda value: (value[0], value[1]))
            active: dict[int, int] = {}
            overlap_started: int | None = None
            for at, delta, stage_id in points:
                had_all = all(active.get(index, 0) > 0 for index in range(4))
                active[stage_id] = max(0, active.get(stage_id, 0) + delta)
                has_all = all(active.get(index, 0) > 0 for index in range(4))
                if not had_all and has_all:
                    overlap_started = at
                elif had_all and not has_all and overlap_started is not None:
                    duration = at - overlap_started
                    if duration > 0 and (best is None or duration > int(best["duration_ns"])):
                        best = {
                            "run_kind": run_kind,
                            "step": step,
                            "started_ns": overlap_started,
                            "ended_ns": at,
                            "duration_ns": duration,
                        }
                    overlap_started = None
    return {
        "verified": best is not None,
        "maximum": best or {},
        "interval_count": len(intervals),
    }


def _runtime_detail_errors(
    by_role: dict[str, dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    errors = []
    all_events: list[dict[str, Any]] = []
    for role, expected_stages_tuple in ROLE_STAGES.items():
        expected_stages = set(expected_stages_tuple)
        worker = _dict(_dict(by_role.get(role)).get("worker"))
        ready_by_run = _dict(worker.get("stage_ready"))
        runs = _dict(worker.get("runs"))
        for run_kind in RUN_KINDS:
            run = _dict(runs.get(run_kind))
            ready = _list(ready_by_run.get(run_kind))
            ready_by_stage = {int(item.get("stage_id", -1)): item for item in ready}
            if (
                run.get("schema") != "crowdtensor_qwen15b_four_gpu_runtime_v1"
                or run.get("role") != role
                or run.get("run_kind") != run_kind
                or int(run.get("microbatches_per_step") or 0) != 4
            ):
                errors.append("qwen15b_runtime_identity_invalid")
            if role == "kernel_a" and _list(run.get("dataset_row_indexes")) != list(range(32)):
                errors.append("qwen15b_dataset_cursor_order_invalid")
            if set(ready_by_stage) != expected_stages:
                errors.append("qwen15b_stage_ready_identity_invalid")
            for stage_id, item in ready_by_stage.items():
                layer_start, layer_end = STAGE_RANGES.get(stage_id, (-1, -1))
                load = _dict(item.get("load_report"))
                if (
                    item.get("device") != f"cuda:{stage_id % 2}"
                    or int(item.get("pid") or 0) <= 0
                    or item.get("grad_scaler_enabled") is not True
                    or int(load.get("stage_id", -1)) != stage_id
                    or int(load.get("layer_start", -1)) != layer_start
                    or int(load.get("layer_end", -1)) != layer_end
                    or _list(load.get("loaded_layer_indexes")) != list(range(layer_start, layer_end))
                    or load.get("lora_injected") is not True
                    or int(load.get("trainable_parameter_count") or 0) <= 0
                    or _list(load.get("trainable_parameter_dtypes")) != ["float32"]
                    or load.get("fp32_lora_parameters_for_grad_scaler") is not True
                    or load.get("cuda_fp16_autocast") is not False
                    or load.get("cuda_fp32_stable_compute") is not True
                    or load.get("stage_boundary_dtype") != "float16"
                ):
                    errors.append("qwen15b_stage_ready_runtime_metadata_invalid")

            restart = _list(run.get("controlled_restarts"))
            restart_record = restart[0] if role == "kernel_b" and run_kind == "resumed" and len(restart) == 1 else {}
            if role == "kernel_b" and run_kind == "resumed":
                old_pid = int(_dict(ready_by_stage.get(2)).get("pid") or 0)
                if (
                    len(restart) != 1
                    or int(restart_record.get("stage_id", -1)) != 2
                    or int(restart_record.get("after_step") or 0) != 4
                    or int(restart_record.get("old_pid") or 0) != old_pid
                    or int(restart_record.get("new_pid") or 0) <= 0
                    or int(restart_record.get("new_pid") or 0) == old_pid
                    or int(restart_record.get("resumed_global_step") or 0) != 4
                    or int(restart_record.get("resumed_dataset_cursor") or 0) != 16
                ):
                    errors.append("qwen15b_controlled_restart_record_invalid")
            elif restart:
                errors.append("qwen15b_unexpected_controlled_restart_record")

            reports = _list(run.get("step_reports"))
            reports_by_step = {int(item.get("step") or 0): item for item in reports}
            if len(reports) != 8 or set(reports_by_step) != set(range(1, 9)):
                errors.append("qwen15b_per_step_records_incomplete")
            for step, step_report in reports_by_step.items():
                stages = _list(_dict(step_report).get("stages"))
                by_stage = {int(item.get("stage_id", -1)): item for item in stages}
                if len(stages) != 2 or set(by_stage) != expected_stages:
                    errors.append("qwen15b_per_step_stage_records_incomplete")
                    continue
                for stage_id, item in by_stage.items():
                    expected_pid = int(_dict(ready_by_stage.get(stage_id)).get("pid") or 0)
                    if role == "kernel_b" and run_kind == "resumed" and stage_id == 2 and step > 4:
                        expected_pid = int(restart_record.get("new_pid") or 0)
                    allocated = int(item.get("peak_allocated_bytes") or 0)
                    reserved = int(item.get("peak_reserved_bytes") or 0)
                    if (
                        int(item.get("global_step") or 0) != step
                        or int(item.get("dataset_cursor") or 0) != step * 4
                        or int(item.get("pid") or 0) != expected_pid
                        or item.get("device") != f"cuda:{stage_id % 2}"
                        or item.get("optimizer_step_applied") is not True
                        or not _positive_finite(item.get("lora_gradient_norm"))
                        or not _positive_finite(item.get("gradient_clip_norm"))
                        or item.get("gradient_clipping_applied") is not True
                        or not _positive_finite(item.get("gradient_scale_before"))
                        or not _positive_finite(item.get("gradient_scale_after"))
                        or not _sha256(item.get("checkpoint_hash"))
                        or not _sha256(item.get("adapter_tensor_hash"))
                        or allocated <= 0
                        or reserved < allocated
                    ):
                        errors.append("qwen15b_per_step_stage_record_invalid")

            events = _list(run.get("events"))
            compute_events = [
                item
                for item in events
                if item.get("operation") in {"forward", "backward", "forward_backward"}
            ]
            actual_keys = Counter(
                (
                    int(item.get("stage_id", -1)),
                    str(item.get("operation") or ""),
                    int(item.get("step", -1)),
                    int(item.get("microbatch", -1)),
                )
                for item in compute_events
            )
            if actual_keys != _expected_compute_keys(role):
                errors.append("qwen15b_per_microbatch_compute_records_incomplete")
            for item in compute_events:
                stage_id = int(item.get("stage_id", -1))
                step = int(item.get("step", -1))
                expected_pid = int(_dict(ready_by_stage.get(stage_id)).get("pid") or 0)
                if role == "kernel_b" and run_kind == "resumed" and stage_id == 2 and step >= 4:
                    expected_pid = int(restart_record.get("new_pid") or 0)
                if (
                    item.get("run_kind") != run_kind
                    or item.get("device") != f"cuda:{stage_id % 2}"
                    or int(item.get("pid") or 0) != expected_pid
                    or int(item.get("started_ns") or 0) <= 0
                    or int(item.get("ended_ns") or 0) <= int(item.get("started_ns") or 0)
                    or (stage_id == 3 and not _positive_finite(item.get("loss")))
                ):
                    errors.append("qwen15b_compute_event_metadata_invalid")
                all_events.append(item)
            stopped = [item for item in events if item.get("operation") == "stage_stopped"]
            restarted = [item for item in events if item.get("operation") == "stage_restarted"]
            if role == "kernel_b" and run_kind == "resumed":
                if (
                    len(stopped) != 1
                    or len(restarted) != 1
                    or int(stopped[0].get("stage_id", -1)) != 2
                    or int(stopped[0].get("step") or 0) != 4
                    or int(stopped[0].get("pid") or 0) != int(restart_record.get("old_pid") or 0)
                    or int(restarted[0].get("pid") or 0) != int(restart_record.get("new_pid") or 0)
                ):
                    errors.append("qwen15b_restart_event_evidence_invalid")

            payload_records = _list(run.get("payloads"))
            payload_keys = Counter(
                (
                    str(item.get("kind") or ""),
                    int(item.get("step", -1)),
                    int(item.get("microbatch", -1)),
                )
                for item in payload_records
            )
            expected_payload_keys = Counter(
                (kind, step, microbatch)
                for kind in ("activation", "gradient")
                for step in range(8)
                for microbatch in range(4)
            )
            if payload_keys != expected_payload_keys or any(
                not _sha256(item.get("payload_hash")) for item in payload_records
            ):
                errors.append("qwen15b_worker_payload_records_incomplete")

            if role == "kernel_b":
                losses = _list(run.get("losses"))
                means = _list(run.get("step_mean_losses"))
                if len(losses) != 32 or len(means) != 8 or any(
                    not _positive_finite(value) for value in losses + means
                ):
                    errors.append("qwen15b_per_microbatch_loss_records_invalid")
                elif any(
                    not math.isclose(
                        float(means[index]),
                        sum(float(value) for value in losses[index * 4 : index * 4 + 4]) / 4.0,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    )
                    for index in range(8)
                ):
                    errors.append("qwen15b_step_loss_aggregation_invalid")
    return sorted(set(errors)), all_events


def _payload_metadata_errors(
    live: dict[str, Any], by_role: dict[str, dict[str, Any]]
) -> list[str]:
    errors = []
    rendezvous = _dict(live.get("rendezvous"))
    payloads = _list(rendezvous.get("payloads"))
    identities = [
        (
            str(item.get("run_kind") or ""),
            str(item.get("kind") or ""),
            int(item.get("step", -1)),
            int(item.get("microbatch", -1)),
        )
        for item in payloads
    ]
    expected = {
        (run_kind, kind, step, microbatch)
        for run_kind in RUN_KINDS
        for kind in ("activation", "gradient")
        for step in range(8)
        for microbatch in range(4)
    }
    expected.add(("resumed", "stage_adapter", 8, -1))
    if len(identities) != len(set(identities)) or set(identities) != expected:
        errors.append("qwen15b_rendezvous_payload_identity_set_invalid")
    payload_by_identity = {identity: item for identity, item in zip(identities, payloads, strict=True)}
    for identity, item in payload_by_identity.items():
        _run_kind, kind, _step, _microbatch = identity
        expected_role = "kernel_a" if kind in {"activation", "stage_adapter"} else "kernel_b"
        expected_tensors = 2 if kind == "activation" else 1
        if (
            item.get("producer_role") != expected_role
            or not _sha256(item.get("payload_hash"))
            or int(item.get("byte_count") or 0) <= 0
            or int(item.get("tensor_count") or 0) < expected_tensors
            or not _positive_finite(item.get("created_at"))
            or "payload_b64" in item
        ):
            errors.append("qwen15b_rendezvous_payload_metadata_invalid")
    for role in ROLE_STAGES:
        worker = _dict(_dict(by_role.get(role)).get("worker"))
        for run_kind in RUN_KINDS:
            for item in _list(_dict(_dict(worker.get("runs")).get(run_kind)).get("payloads")):
                identity = (
                    run_kind,
                    str(item.get("kind") or ""),
                    int(item.get("step", -1)),
                    int(item.get("microbatch", -1)),
                )
                if _dict(payload_by_identity.get(identity)).get("payload_hash") != item.get(
                    "payload_hash"
                ):
                    errors.append("qwen15b_worker_rendezvous_payload_hash_mismatch")
    registrations = {str(item.get("role") or ""): item for item in _list(rendezvous.get("registrations"))}
    if set(rendezvous.get("registered_roles") or []) != set(ROLE_STAGES) or set(registrations) != set(ROLE_STAGES):
        errors.append("qwen15b_rendezvous_registration_incomplete")
    for role, stage_ids in ROLE_STAGES.items():
        registration = _dict(registrations.get(role))
        if (
            registration.get("cuda_live") is not True
            or _list(registration.get("stage_ids")) != list(stage_ids)
            or _list(registration.get("cuda_devices")) != ["cuda:0", "cuda:1"]
            or len(_list(registration.get("stage_pids"))) != 2
            or any(int(value) <= 0 for value in _list(registration.get("stage_pids")))
        ):
            errors.append("qwen15b_rendezvous_registration_invalid")
    completions = {str(item.get("role") or ""): item for item in _list(rendezvous.get("completions"))}
    if set(completions) != set(ROLE_STAGES) or any(
        item.get("ok") is not True
        or int(item.get("baseline_steps_completed") or 0) != 8
        or int(item.get("resumed_steps_completed") or 0) != 8
        for item in completions.values()
    ):
        errors.append("qwen15b_rendezvous_completion_invalid")
    return sorted(set(errors))


def _archive_detail_errors(
    live: dict[str, Any], by_role: dict[str, dict[str, Any]]
) -> list[str]:
    errors = []
    checkpoints = {
        str(item.get("role") or ""): item for item in _list(live.get("checkpoint_bundles"))
    }
    if set(checkpoints) != set(ROLE_STAGES):
        errors.append("qwen15b_checkpoint_archive_roles_incomplete")
    for role, stage_ids in ROLE_STAGES.items():
        archive = _dict(checkpoints.get(role))
        if any(
            archive.get(key) is not True
            for key in (
                "verified",
                "preserved",
                "worker_hash_match",
                "archive_safe",
                "unique_archive_members",
                "all_checkpoint_files_hash_verified",
                "all_final_steps_verified",
                "model_revision_verified",
                "all_manifest_content_hashes_verified",
            )
        ) or not _sha256(archive.get("file_hash")) or int(archive.get("byte_count") or 0) <= 0:
            errors.append("qwen15b_checkpoint_archive_integrity_invalid")
        summaries = _list(archive.get("manifest_summaries"))
        identities = {
            (str(item.get("run_kind") or ""), int(item.get("stage_id", -1)))
            for item in summaries
        }
        expected = {(run_kind, stage_id) for run_kind in RUN_KINDS for stage_id in stage_ids}
        if len(summaries) != 4 or identities != expected:
            errors.append("qwen15b_checkpoint_manifest_coverage_invalid")
        for item in summaries:
            stage_id = int(item.get("stage_id", -1))
            layer_start, layer_end = STAGE_RANGES.get(stage_id, (-1, -1))
            if (
                int(item.get("layer_start", -1)) != layer_start
                or int(item.get("layer_end", -1)) != layer_end
                or int(item.get("global_step") or 0) != 8
                or int(item.get("optimizer_step") or 0) != 8
                or int(item.get("dataset_cursor") or 0) != 32
                or item.get("device") != f"cuda:{stage_id % 2}"
                or item.get("model_id") != MODEL_ID
                or item.get("model_revision") != MODEL_REVISION
                or item.get("component_hashes_verified") is not True
                or item.get("grad_scaler_state_present") is not True
                or item.get("rng_state_present") is not True
                or int(item.get("adapter_tensor_count") or 0) <= 0
                or not _sha256(item.get("adapter_tensor_hash"))
                or not _sha256(item.get("manifest_content_hash"))
                or item.get("manifest_content_hash_verified") is not True
            ):
                errors.append("qwen15b_checkpoint_manifest_metadata_invalid")
        worker_bundle = _dict(_dict(by_role.get(role)).get("checkpoint_bundle"))
        if worker_bundle.get("file_hash") != archive.get("file_hash"):
            errors.append("qwen15b_checkpoint_worker_archive_hash_mismatch")
    adapter = _dict(live.get("adapter_bundle"))
    if (
        any(
            adapter.get(key) is not True
            for key in (
                "verified",
                "preserved",
                "worker_hash_match",
                "archive_safe",
                "unique_archive_members",
                "standard_peft_layout",
                "base_model_verified",
                "model_revision_verified",
                "safetensors_header_verified",
            )
        )
        or not _sha256(adapter.get("file_hash"))
        or not _sha256(adapter.get("adapter_file_hash"))
        or not _sha256(adapter.get("adapter_config_hash"))
        or int(adapter.get("byte_count") or 0) <= 0
        or int(adapter.get("adapter_tensor_count") or 0) <= 0
        or _list(adapter.get("layer_indexes")) != list(range(28))
    ):
        errors.append("qwen15b_adapter_archive_integrity_invalid")
    kernel_b_outer = _dict(by_role.get("kernel_b"))
    kernel_b = _dict(kernel_b_outer.get("worker"))
    export = _dict(kernel_b.get("export"))
    if (
        _dict(kernel_b_outer.get("adapter_bundle")).get("file_hash") != adapter.get("file_hash")
        or export.get("adapter_file_hash") != adapter.get("adapter_file_hash")
        or export.get("adapter_config_hash") != adapter.get("adapter_config_hash")
        or int(export.get("adapter_tensor_count") or 0) != int(
            adapter.get("adapter_tensor_count") or -1
        )
        or export.get("adapter_tensor_names_hash") != adapter.get("adapter_tensor_names_hash")
    ):
        errors.append("qwen15b_adapter_worker_archive_hash_mismatch")
    return sorted(set(errors))


def _public_safety_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, default=str)
    lowered = encoded.lower()
    errors = []
    forbidden_fragments = {
        "qwen15b_public_payload_b64": '"payload_b64"',
        "qwen15b_public_token_tensor": '"token_ids":',
        "qwen15b_public_activation_tensor": '"activation":',
        "qwen15b_public_gradient_tensor": '"activation_gradient":',
        "qwen15b_public_raw_text": '"raw_training_text":',
        "qwen15b_public_root_path": "/root/",
        "qwen15b_public_tmp_path": "/tmp/",
        "qwen15b_public_kaggle_path": "/kaggle/",
        "qwen15b_public_coordinator_route": "trycloudflare.com",
        "qwen15b_public_bearer": "bearer ",
        "qwen15b_public_cookie": "cookie:",
    }
    for code, fragment in forbidden_fragments.items():
        if fragment in lowered:
            errors.append(code)
    if re.search(r"KGA[A-Za-z0-9_-]{8,}", encoded):
        errors.append("qwen15b_public_kaggle_token")
    for key in (
        "activation_values_public",
        "gradient_values_public",
        "adapter_tensor_values_public",
        "token_ids_public",
        "raw_training_text_public",
        "credentials_public",
        "coordinator_url_public",
        "private_paths_public",
    ):
        if f'"{key}": true' in lowered:
            errors.append(f"qwen15b_{key}_true")
    return sorted(set(errors))


def _source_errors(report: dict[str, Any]) -> list[str]:
    errors = []
    source = _dict(report.get("source"))
    manifest = _dict(source.get("source"))
    ownership = _dict(source.get("ownership"))
    if source.get("ok") is not True or manifest.get("source_verified") is not True:
        errors.append("qwen15b_source_not_verified")
    if manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION:
        errors.append("qwen15b_source_model_not_pinned")
    if int(manifest.get("parameter_count") or 0) < PARAMETER_MINIMUM:
        errors.append("qwen15b_source_parameter_count_below_1b")
    if manifest.get("official_weight_index_present") is not False or manifest.get(
        "weight_index_generated_from_header"
    ) is not True:
        errors.append("qwen15b_source_stage_index_invalid")
    stages = _list(ownership.get("stages"))
    if (
        len(stages) != 4
        or [int(item.get("layer_start", -1)) for item in stages] != [0, 7, 14, 21]
        or [int(item.get("layer_end", -1)) for item in stages] != [7, 14, 21, 28]
        or ownership.get("all_source_tensors_covered") is not True
        or ownership.get("only_tied_embedding_lm_head_duplicated") is not True
        or ownership.get("four_distinct_kernel_device_placements") is not True
    ):
        errors.append("qwen15b_four_stage_ownership_invalid")
    if (
        not stages
        or stages[0].get("owns_embedding") is not True
        or stages[-1].get("owns_norm") is not True
        or stages[-1].get("owns_lm_head") is not True
    ):
        errors.append("qwen15b_embedding_norm_lm_head_ownership_invalid")
    return errors


def _dataset_errors(report: dict[str, Any]) -> list[str]:
    errors = []
    dataset = _dict(report.get("dataset"))
    manifest = _dict(dataset.get("manifest"))
    if dataset.get("ok") is not True or dataset.get("private_payload_present") is not True:
        errors.append("qwen15b_private_dataset_not_prepared")
    if (
        manifest.get("model_id") != MODEL_ID
        or manifest.get("model_revision") != MODEL_REVISION
        or manifest.get("dataset_id") != "Salesforce/wikitext"
        or int(manifest.get("sequence_length") or 0) != 64
        or int(manifest.get("train_sequence_count") or 0) < 32
        or int(manifest.get("validation_sequence_count") or 0) < 4
    ):
        errors.append("qwen15b_dataset_contract_invalid")
    if dataset.get("raw_text_public") is not False or dataset.get("token_ids_public") is not False:
        errors.append("qwen15b_dataset_public_safety_invalid")
    return errors


def _allocation_policy_errors(report: dict[str, Any]) -> list[str]:
    errors = []
    live = _dict(report.get("live_report"))
    budget = _dict(report.get("allocation_budget"))
    live_budget = _dict(live.get("allocation_budget"))
    required_equal_fields = (
        "schema",
        "amendment_present",
        "amendment_valid",
        "original_attempt_limit",
        "effective_attempt_limit",
        "total_attempt_limit_unbounded",
        "one_attempt_per_probe_invocation",
        "automatic_retry_loop",
        "additional_attempts_authorized",
        "prior_attempts_preserved",
        "same_authorized_account_only",
        "allocation_timeout_seconds",
        "authorization_hash",
        "authorization_text_public",
        "credential_values_public",
        "public_artifact_safe",
    )
    if (
        budget.get("schema")
        != "crowdtensor_qwen15b_four_gpu_allocation_budget_summary_v1"
        or live_budget.get("schema") != budget.get("schema")
        or any(budget.get(key) != live_budget.get(key) for key in required_equal_fields)
        or budget.get("amendment_present") is not True
        or budget.get("amendment_valid") is not True
        or int(budget.get("original_attempt_limit") or 0) != 2
        or budget.get("effective_attempt_limit") is not None
        or budget.get("total_attempt_limit_unbounded") is not True
        or budget.get("one_attempt_per_probe_invocation") is not True
        or budget.get("automatic_retry_loop") is not False
        or budget.get("additional_attempts_authorized") is not None
        or budget.get("prior_attempts_preserved") is not True
        or budget.get("same_authorized_account_only") is not True
        or int(budget.get("allocation_timeout_seconds") or 0) != 1800
        or not _sha256(budget.get("authorization_hash"))
        or budget.get("authorization_text_public") is not False
        or budget.get("credential_values_public") is not False
        or budget.get("public_artifact_safe") is not True
        or int(budget.get("attempts_used") or 0) != int(live.get("attempt") or 0)
        or int(budget.get("successful_attempt") or 0) != int(live.get("attempt") or 0)
        or int(budget.get("probe_invocation_attempt_ceiling") or 0)
        != int(live.get("attempt_limit") or 0)
        or budget.get("budget_exhausted") is not False
        or budget.get("ledger_history_must_be_preserved") is not True
        or budget.get("additional_attempt_requires_explicit_user_amendment") is not False
        or budget.get("probe_invocation_ceiling_is_not_total_policy_limit") is not True
    ):
        errors.append("qwen15b_unbounded_allocation_authorization_missing")
    history = _dict(report.get("allocation_history"))
    live_attempt = int(live.get("attempt") or 0)
    if (
        history.get("schema")
        != "crowdtensor_qwen15b_four_gpu_alpha_allocation_history_v1"
        or history.get("ledger_present") is not True
        or int(history.get("attempt_count") or 0) < live_attempt
        or int(history.get("completed_attempt_count") or 0)
        != int(history.get("attempt_count") or -1)
        or history.get("attempt_numbers_sequential") is not True
        or not _sha256(history.get("attempt_records_hash"))
        or int(history.get("successful_attempt") or 0) != live_attempt
        or _list(history.get("verified_attempt_numbers")) != [live_attempt]
        or history.get("successful_attempt_matches_ledger") is not True
        or history.get("immutable_history_preserved") is not True
        or history.get("public_artifact_safe") is not True
    ):
        errors.append("qwen15b_allocation_attempt_history_invalid")
    return errors


def _runtime_remediation_errors(report: dict[str, Any]) -> list[str]:
    remediation = _dict(report.get("runtime_remediation"))
    if (
        remediation.get("fp16_autocast_non_finite_activation_observed") is not True
        or remediation.get("fp16_autocast_abandoned") is not True
        or remediation.get("frozen_stage_weight_compute_dtype") != "float32"
        or remediation.get("lora_parameter_dtype") != "float32"
        or remediation.get("cuda_fp16_autocast") is not False
        or remediation.get("cuda_fp32_stable_compute") is not True
        or remediation.get("fp16_stage_boundary_transport") is not True
        or remediation.get("grad_scaler_unscale_step_verified") is not True
        or remediation.get("non_finite_activation_logits_loss_gradient_gates") is not True
        or remediation.get("remediation_local_tests_passed") is not True
        or remediation.get("remediation_gpu_live_verified") is not True
    ):
        return ["qwen15b_fp32_stable_compute_remediation_missing"]
    artifacts = _dict(report.get("artifacts"))
    if _dict(artifacts.get("precision_failure_report")).get("present") is not True:
        return ["qwen15b_fp16_non_finite_failure_artifact_missing"]
    return []


def _readiness_errors(report: dict[str, Any]) -> list[str]:
    errors = []
    live = _dict(report.get("live_report"))
    evidence = _dict(live.get("evidence"))
    workers = _list(live.get("worker_reports"))
    by_role = {str(item.get("role") or ""): item for item in workers}
    errors.extend(_allocation_policy_errors(report))
    errors.extend(_runtime_remediation_errors(report))
    if live.get("qwen15b_four_gpu_alpha_verified") is not True or evidence.get(
        "verified"
    ) is not True:
        errors.append("qwen15b_live_alpha_evidence_missing")
    if live.get("requested_model") != MODEL_ID or live.get("requested_model_revision") != MODEL_REVISION:
        errors.append("qwen15b_live_model_request_not_pinned")
    if (
        live.get("same_authorized_account") is not True
        or live.get("multi_account_gate_substitution") is not False
        or int(live.get("requested_kernel_count") or 0) != 2
    ):
        errors.append("qwen15b_same_account_dual_kernel_not_verified")
    if int(live.get("max_observed_running_kernel_count") or 0) < 2:
        errors.append("qwen15b_two_concurrent_kernels_not_observed")
    if len(workers) != 2 or set(by_role) != {"kernel_a", "kernel_b"} or any(
        item.get("ok") is not True for item in by_role.values()
    ):
        errors.append("qwen15b_worker_reports_incomplete")

    errors.extend(_dependency_smoke_errors(by_role))
    runtime_errors, runtime_events = _runtime_detail_errors(by_role)
    errors.extend(runtime_errors)
    errors.extend(_payload_metadata_errors(live, by_role))
    errors.extend(_archive_detail_errors(live, by_role))

    stages = []
    for role, outer in by_role.items():
        worker = _dict(outer.get("worker"))
        if worker.get("model_id") != MODEL_ID or worker.get("model_revision") != MODEL_REVISION:
            errors.append("qwen15b_worker_model_not_pinned")
        if int(worker.get("parameter_count") or 0) < PARAMETER_MINIMUM:
            errors.append("qwen15b_worker_parameter_count_below_1b")
        if worker.get("base_weights_frozen") is not True or worker.get(
            "positive_lora_gradient_norms"
        ) is not True:
            errors.append("qwen15b_real_lora_backward_missing")
        for run_kind in ("baseline", "resumed"):
            ready = _list(_dict(worker.get("stage_ready")).get(run_kind))
            runs = _dict(worker.get("runs"))
            run = _dict(runs.get(run_kind))
            if int(run.get("steps_completed") or 0) != 8:
                errors.append(f"qwen15b_{run_kind}_eight_steps_missing")
            if run.get("real_forward") is not True or run.get("real_backward") is not True:
                errors.append(f"qwen15b_{run_kind}_real_forward_backward_missing")
            if len(ready) != 2:
                errors.append(f"qwen15b_{role}_{run_kind}_two_stages_missing")
            if len({(str(item.get("device")), int(item.get("pid") or 0)) for item in ready}) != 2:
                errors.append(f"qwen15b_{role}_{run_kind}_distinct_cuda_processes_missing")
            for item in ready:
                stages.append((run_kind, role, item))
                load = _dict(item.get("load_report"))
                if item.get("cuda_live") is not True or item.get("device") not in {
                    "cuda:0",
                    "cuda:1",
                }:
                    errors.append("qwen15b_cuda_stage_execution_missing")
                if (
                    load.get("meta_device_construction") is not True
                    or load.get("loaded_full_model") is not False
                    or load.get("stage_owned_module_construction") is not True
                    or load.get("foreign_layer_count") != 0
                    or load.get("only_lora_trainable") is not True
                    or load.get("gradient_checkpointing") is not True
                ):
                    errors.append("qwen15b_stage_owned_loading_invalid")
    baseline_stage_ids = {
        int(item.get("stage_id", -1))
        for run_kind, _role, item in stages
        if run_kind == "baseline"
    }
    if baseline_stage_ids != {0, 1, 2, 3}:
        errors.append("qwen15b_four_live_cuda_stages_missing")

    derived_overlap = _compute_overlap(runtime_events)
    if (
        derived_overlap.get("verified") is not True
        or evidence.get("four_stage_compute_overlap_verified") is not True
        or _dict(evidence.get("maximum_four_stage_overlap")) != derived_overlap.get("maximum")
        or int(evidence.get("interval_count") or 0) != int(derived_overlap.get("interval_count") or 0)
    ):
        errors.append("qwen15b_four_gpu_compute_overlap_missing")
    rendezvous_payloads = _list(_dict(live.get("rendezvous")).get("payloads"))
    activation_count = sum(item.get("kind") == "activation" for item in rendezvous_payloads)
    gradient_count = sum(item.get("kind") == "gradient" for item in rendezvous_payloads)
    if int(evidence.get("activation_payload_count") or 0) != 64 or activation_count != 64:
        errors.append("qwen15b_cross_kernel_activation_transport_missing")
    if int(evidence.get("gradient_payload_count") or 0) != 64 or gradient_count != 64:
        errors.append("qwen15b_cross_kernel_gradient_transport_missing")
    if evidence.get("resume_adapter_equivalence_verified") is not True or evidence.get(
        "resume_loss_equivalence_verified"
    ) is not True:
        errors.append("qwen15b_resume_numeric_equivalence_missing")
    kernel_b = _dict(_dict(by_role.get("kernel_b")).get("worker"))
    restart = _list(_dict(_dict(kernel_b.get("runs")).get("resumed")).get("controlled_restarts"))
    if (
        kernel_b.get("controlled_restart_verified") is not True
        or len(restart) != 1
        or restart[0].get("forced_stop_verified") is not True
        or restart[0].get("new_pid_verified") is not True
        or restart[0].get("checkpoint_resume_verified") is not True
        or int(restart[0].get("after_step") or 0) != 4
        or int(restart[0].get("resumed_global_step") or 0) != 4
    ):
        errors.append("qwen15b_controlled_stage_restart_missing")
    for run_kind in ("baseline", "resumed"):
        run = _dict(_dict(kernel_b.get("runs")).get(run_kind))
        if run.get("loss_reduced") is not True or not (
            float(run.get("loss_end") or float("inf")) < float(run.get("loss_start") or 0.0)
        ):
            errors.append(f"qwen15b_{run_kind}_loss_not_reduced")
    evaluation = _dict(kernel_b.get("evaluation"))
    if (
        evaluation.get("evaluation_verified") is not True
        or evaluation.get("standard_peft_cpu_load") is not True
        or evaluation.get("standard_peft_cuda_load") is not True
        or evaluation.get("cuda_compute_dtype") != "float32"
        or evaluation.get("adapter_changes_logits") is not True
        or evaluation.get("validation_loss_reduced") is not True
    ):
        errors.append("qwen15b_standard_peft_evaluation_missing")
    export = _dict(kernel_b.get("export"))
    if (
        export.get("standard_peft_format") is not True
        or export.get("model_id") != MODEL_ID
        or export.get("model_revision") != MODEL_REVISION
        or _list(export.get("layer_indexes")) != list(range(28))
    ):
        errors.append("qwen15b_standard_peft_export_missing")
    if evidence.get("checkpoint_archives_verified") is not True or evidence.get(
        "adapter_archive_verified"
    ) is not True:
        errors.append("qwen15b_checkpoint_or_adapter_archive_unverified")
    cleanup = _dict(live.get("cleanup"))
    for key in (
        "kernels_deleted",
        "only_attempt_kernel_refs_targeted",
        "private_packages_removed",
        "coordinator_stopped",
        "tunnel_stopped",
        "private_runtime_removed",
        "rendezvous_private_payloads_removed",
        "checkpoint_archives_verified_before_cleanup",
    ):
        if cleanup.get(key) is not True:
            errors.append(f"qwen15b_cleanup_{key}_missing")
    return sorted(set(errors))


def check(report: dict[str, Any], *, require_ready: bool = False) -> dict[str, Any]:
    structural = []
    if report.get("schema") != SCHEMA:
        structural.append("qwen15b_alpha_schema_invalid")
    structural.extend(_source_errors(report))
    structural.extend(_dataset_errors(report))
    structural.extend(_public_safety_errors(report))
    tests = _dict(report.get("test_summary"))
    if tests.get("ok") is not True or int(tests.get("failed") or 0) != 0:
        structural.append("qwen15b_required_tests_not_passing")
    readiness = _readiness_errors(report)
    ready = not structural and not readiness
    if bool(report.get("goal_achieved")) != ready:
        structural.append("qwen15b_goal_achieved_flag_incoherent")
    errors = sorted(set(structural + (readiness if require_ready else [])))
    return {
        "schema": "crowdtensor_training_qwen15b_four_gpu_alpha_check_v1",
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "readiness_error_count": len(readiness),
        "readiness_errors": readiness,
        "qwen15b_four_gpu_alpha_ready": ready,
        "goal_achieved": ready,
        "require_ready": bool(require_ready),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    result = check(report, require_ready=args.require_ready)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"training_qwen15b_four_gpu_alpha_check ok={result['ok']} "
            f"ready={result['qwen15b_four_gpu_alpha_ready']} errors={result['error_count']}"
        )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
