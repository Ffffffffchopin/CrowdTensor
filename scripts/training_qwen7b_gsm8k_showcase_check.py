#!/usr/bin/env python3
"""Strictly validate a Qwen2.5-7B elastic GSM8K showcase RC."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from crowdtensor.qwen15b_training import sha256_file, stable_hash
from crowdtensor.qwen7b_gsm8k_showcase import (
    DATASET_ID,
    DATASET_MANIFEST_SCHEMA,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_PARAMETER_COUNT,
    MODEL_REVISION,
    SOURCE_LAYOUT_SCHEMA,
    dataset_manifest_content_hash,
)
from scripts.training_cuda_kaggle_common import public_safety_errors
from scripts.training_qwen7b_gsm8k_showcase_pack import SCHEMA
from scripts.training_qwen7b_gsm8k_benchmark_check import (
    check as check_benchmark,
)
from scripts.training_qwen7b_gsm8k_cleanup_check import check as check_cleanup
from scripts.training_qwen7b_gsm8k_elastic_check import check as check_training


def check(path: str | Path, *, require_ready: bool = False) -> dict[str, Any]:
    report_path = Path(path).resolve()
    errors: list[str] = []
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except BaseException as exc:
        return {
            "schema": "crowdtensor_qwen7b_gsm8k_showcase_check_v1",
            "ok": False,
            "showcase_ready": False,
            "errors": ["showcase_report_load_failed:" + type(exc).__name__],
            "error_count": 1,
        }
    if report.get("schema") != SCHEMA:
        errors.append("showcase_schema_mismatch")
    expected_hash = stable_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    if report.get("content_hash") != expected_hash:
        errors.append("showcase_content_hash_mismatch")
    model = dict(report.get("model") or {})
    if (
        model.get("model_id") != MODEL_ID
        or model.get("model_revision") != MODEL_REVISION
        or int(model.get("parameter_count") or 0) != MODEL_PARAMETER_COUNT
    ):
        errors.append("showcase_model_identity_invalid")
    dataset = dict(report.get("dataset") or {})
    if (
        dataset.get("dataset_id") != DATASET_ID
        or dataset.get("dataset_revision") != DATASET_REVISION
        or int(dataset.get("benchmark_example_count") or 0) < 128
        or dataset.get("train_test_split_isolation_verified") is not True
    ):
        errors.append("showcase_dataset_contract_invalid")
    training = dict(report.get("training") or {})
    if (
        int(training.get("optimizer_steps") or 0) < 256
        or int(training.get("training_non_padding_token_count") or 0) < 262_144
        or int(training.get("replacement_after_step") or 0) <= 0
        or float(training.get("learning_rate") or 0.0) <= 0.0
        or int(training.get("lora_rank") or 0) != 4
        or int(training.get("lora_alpha") or 0) != 8
        or int(training.get("attempt_number") or 0) != 3
        or int(training.get("maximum_training_attempts") or 0) != 3
    ):
        errors.append("showcase_training_budget_incomplete")
    metrics = dict(report.get("metrics") or {})
    if (
        metrics.get("primary_metric") != "normalized_gsm8k_exact_match"
        or int(metrics.get("example_count") or 0) < 128
        or not 0.0 <= float(metrics.get("before_rate", -1.0)) <= 1.0
        or not 0.0 <= float(metrics.get("after_rate", -1.0)) <= 1.0
        or abs(
            float(metrics.get("absolute_improvement") or 0.0)
            - (
                float(metrics.get("after_rate") or 0.0)
                - float(metrics.get("before_rate") or 0.0)
            )
        )
        > 1e-12
    ):
        errors.append("showcase_primary_metric_invalid")
    if not (
        metrics.get("practical_improvement") is True
        or metrics.get("statistical_improvement") is True
    ):
        errors.append("showcase_primary_metric_not_improved")
    for field in ("before_wilson_95pct", "after_wilson_95pct", "paired_bootstrap_improvement_95pct"):
        interval = metrics.get(field)
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or float(interval[0]) > float(interval[1])
        ):
            errors.append("showcase_confidence_interval_invalid:" + field)
    gates = dict(report.get("gates") or {})
    required_gates = {
        "pinned_7b_model",
        "pinned_gsm8k_isolated_dataset",
        "confirmatory_holdout_not_used_for_development",
        "preregistration_bound_before_training",
        "real_two_generation_elastic_training",
        "large_training_budget",
        "training_contract_matches_preregistration",
        "replacement_exactly_once_verified",
        "baseline_benchmark_chronological",
        "same_runtime_before_after_benchmark",
        "baseline_reproduction_consistent",
        "standard_peft_reloaded",
        "primary_metric_improved",
        "no_integrity_degradation",
        "complete_cleanup",
        "claim_boundaries_preserved",
    }
    if set(gates) != required_gates or any(gates.get(key) is not True for key in required_gates):
        errors.append("showcase_required_gates_incomplete")
    artifacts = dict(report.get("artifacts") or {})
    hashes = dict(report.get("artifact_hashes") or {})
    required_artifacts = {
        "source",
        "dataset",
        "preregistration",
        "training",
        "baseline",
        "post_benchmark",
        "development_benchmark",
        "cleanup_audit",
        "adapter",
        "model_card",
        "showcase_doc",
        "reproduce_doc",
    }
    root = report_path.parent
    resolved: dict[str, Path] = {}
    if set(artifacts) != required_artifacts or set(hashes) != required_artifacts:
        errors.append("showcase_artifact_set_invalid")
    else:
        for name in sorted(required_artifacts):
            value = Path(str(artifacts[name]))
            target = (root / value).resolve()
            if value.is_absolute() or root not in target.parents:
                errors.append("showcase_artifact_path_invalid:" + name)
            elif not target.is_file():
                errors.append("showcase_artifact_missing:" + name)
            elif sha256_file(target) != hashes[name]:
                errors.append("showcase_artifact_hash_mismatch:" + name)
            else:
                resolved[name] = target
    adapter = resolved.get("adapter")
    source_artifact = resolved.get("source")
    dataset_artifact = resolved.get("dataset")
    preregistration_artifact = resolved.get("preregistration")
    training_artifact = resolved.get("training")
    loaded_inputs: dict[str, dict[str, Any]] = {}
    for name, artifact in (
        ("source", source_artifact),
        ("dataset", dataset_artifact),
        ("preregistration", preregistration_artifact),
        ("training", training_artifact),
        ("baseline", resolved.get("baseline")),
        ("post_benchmark", resolved.get("post_benchmark")),
        ("development_benchmark", resolved.get("development_benchmark")),
        ("cleanup_audit", resolved.get("cleanup_audit")),
    ):
        if artifact is None:
            continue
        try:
            value = json.loads(artifact.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(name)
            loaded_inputs[name] = value
            if public_safety_errors(value):
                errors.append("showcase_input_artifact_public_safety_failed:" + name)
        except BaseException:
            errors.append("showcase_input_artifact_unreadable:" + name)
    source_input = loaded_inputs.get("source")
    if source_input is not None and (
        source_input.get("schema") != SOURCE_LAYOUT_SCHEMA
        or source_input.get("content_hash")
        != stable_hash(
            {key: value for key, value in source_input.items() if key != "content_hash"}
        )
        or source_input.get("source_verified") is not True
    ):
        errors.append("showcase_source_artifact_invalid")
    dataset_input = loaded_inputs.get("dataset")
    if dataset_input is not None and (
        dataset_input.get("schema") != DATASET_MANIFEST_SCHEMA
        or dataset_input.get("content_hash")
        != dataset_manifest_content_hash(dataset_input)
        or dataset_input.get("train_test_split_isolation_verified") is not True
    ):
        errors.append("showcase_dataset_artifact_invalid")
    preregistration_input = loaded_inputs.get("preregistration")
    if preregistration_input is not None and (
        preregistration_input.get("schema")
        != "crowdtensor_qwen7b_gsm8k_showcase_preregistration_v1"
        or preregistration_input.get("content_hash")
        != stable_hash(
            {
                key: value
                for key, value in preregistration_input.items()
                if key != "content_hash"
            }
        )
    ):
        errors.append("showcase_preregistration_artifact_invalid")
    if (
        source_artifact is not None
        and source_input is not None
        and dataset_artifact is not None
        and preregistration_input is not None
        and (
            (preregistration_input.get("model") or {}).get("source_layout_hash")
            != sha256_file(source_artifact)
            or (preregistration_input.get("dataset") or {}).get(
                "dataset_manifest_hash"
            )
            != sha256_file(dataset_artifact)
            or (preregistration_input.get("dataset") or {}).get(
                "benchmark_indexes_hash"
            )
            != (dataset_input or {}).get("benchmark_indexes_hash")
            or (preregistration_input.get("dataset") or {}).get(
                "benchmark_prompt_hash"
            )
            != (dataset_input or {}).get("benchmark_prompt_hash")
            or (preregistration_input.get("dataset") or {}).get(
                "benchmark_gold_hash"
            )
            != (dataset_input or {}).get("benchmark_gold_hash")
            or (preregistration_input.get("dataset") or {}).get(
                "benchmark_excluded_indexes_hash"
            )
            != (dataset_input or {}).get("benchmark_excluded_indexes_hash")
            or int(
                (preregistration_input.get("dataset") or {}).get(
                    "benchmark_overlap_with_excluded_count",
                    -1,
                )
            )
            != int(
                (dataset_input or {}).get(
                    "benchmark_overlap_with_excluded_count",
                    -2,
                )
            )
        )
    ):
        errors.append("showcase_preregistration_binding_invalid")
    training_input = loaded_inputs.get("training")
    if (
        source_artifact is not None
        and source_input is not None
        and dataset_artifact is not None
        and dataset_input is not None
        and training_input is not None
        and (
            (training_input.get("source") or {}).get("source_layout_hash")
            != sha256_file(source_artifact)
            or (training_input.get("source") or {}).get("source_content_hash")
            != source_input.get("content_hash")
            or (training_input.get("dataset") or {}).get("dataset_manifest_hash")
            != sha256_file(dataset_artifact)
            or (training_input.get("dataset") or {}).get(
                "private_train_payload_hash"
            )
            != dataset_input.get("private_train_payload_hash")
            or (training_input.get("dataset") or {}).get("train_token_hash")
            != dataset_input.get("train_token_hash")
            or (training_input.get("dataset") or {}).get("validation_token_hash")
            != dataset_input.get("validation_token_hash")
        )
    ):
        errors.append("showcase_training_input_binding_invalid")
    for name in ("baseline", "post_benchmark"):
        benchmark_input = loaded_inputs.get(name)
        if benchmark_input is not None and dataset_input is not None and (
            benchmark_input.get("benchmark_payload_hash")
            != dataset_input.get("private_benchmark_payload_hash")
            or benchmark_input.get("validation_payload_hash")
            != dataset_input.get("validation_token_hash")
            or (benchmark_input.get("worker") or {}).get("benchmark_prompt_hash")
            != dataset_input.get("benchmark_prompt_hash")
            or (benchmark_input.get("worker") or {}).get("benchmark_gold_hash")
            != dataset_input.get("benchmark_gold_hash")
        ):
            errors.append("showcase_benchmark_input_binding_invalid:" + name)
    cleanup_input = loaded_inputs.get("cleanup_audit")
    if cleanup_input is not None and (
        (cleanup_input.get("evidence_hashes") or {}).get("dataset_manifest")
        != hashes.get("dataset")
        or (cleanup_input.get("evidence_hashes") or {}).get("training_report")
        != hashes.get("training")
        or (cleanup_input.get("evidence_hashes") or {}).get("baseline_report")
        != hashes.get("baseline")
        or (cleanup_input.get("evidence_hashes") or {}).get(
            "post_benchmark_report"
        )
        != hashes.get("post_benchmark")
    ):
        errors.append("showcase_cleanup_input_binding_invalid")
    development_input = loaded_inputs.get("development_benchmark")
    if (
        development_input is not None
        and preregistration_input is not None
        and (
            (preregistration_input.get("development_evidence") or {}).get(
                "prior_benchmark_report_hash"
            )
            != hashes.get("development_benchmark")
            or development_input.get("mode") != "both"
            or development_input.get("ok") is not True
            or (development_input.get("cleanup") or {}).get(
                "live_resources_left_running"
            )
            is not False
        )
    ):
        errors.append("showcase_development_evidence_binding_invalid")
    post_input = loaded_inputs.get("post_benchmark")
    if (
        development_input is not None
        and post_input is not None
        and dataset_input is not None
    ):
        development_indexes = sorted(
            int(item.get("example_index", -1))
            for item in (
                ((development_input.get("worker") or {}).get("passes") or {})
                .get("base", {})
                .get("records", [])
            )
        )
        confirmatory_indexes = sorted(
            int(item.get("example_index", -1))
            for item in (
                ((post_input.get("worker") or {}).get("passes") or {})
                .get("base", {})
                .get("records", [])
            )
        )
        if (
            len(development_indexes) != 128
            or len(set(development_indexes)) != 128
            or len(confirmatory_indexes) != 128
            or len(set(confirmatory_indexes)) != 128
            or set(development_indexes) & set(confirmatory_indexes)
            or stable_hash(development_indexes)
            != dataset_input.get("benchmark_excluded_indexes_hash")
            or stable_hash(confirmatory_indexes)
            != dataset_input.get("benchmark_indexes_hash")
        ):
            errors.append("showcase_confirmatory_holdout_disjointness_invalid")
    if training_artifact is not None:
        training_check = check_training(training_artifact, require_ready=True)
        if training_check.get("ok") is not True:
            errors.append("showcase_training_strict_check_failed")
    for name in ("baseline", "post_benchmark"):
        benchmark_artifact = resolved.get(name)
        if benchmark_artifact is not None:
            benchmark_check = check_benchmark(
                benchmark_artifact, require_ready=True
            )
            if benchmark_check.get("ok") is not True:
                errors.append("showcase_benchmark_strict_check_failed:" + name)
    development_artifact = resolved.get("development_benchmark")
    if development_artifact is not None:
        development_checked = check_benchmark(
            development_artifact, require_ready=True
        )
        if development_checked.get("ok") is not True:
            errors.append("showcase_benchmark_strict_check_failed:development")
    cleanup_artifact = resolved.get("cleanup_audit")
    if cleanup_artifact is not None:
        cleanup_checked = check_cleanup(cleanup_artifact, require_ready=True)
        if cleanup_checked.get("ok") is not True:
            errors.append("showcase_cleanup_strict_check_failed")
    for name in ("model_card", "showcase_doc", "reproduce_doc"):
        document = resolved.get(name)
        if document is not None:
            text = document.read_text(encoding="utf-8")
            if public_safety_errors(text):
                errors.append("showcase_document_public_safety_failed:" + name)
    reproduce = resolved.get("reproduce_doc")
    if reproduce is not None:
        text = reproduce.read_text(encoding="utf-8")
        required_commands = {
            "training_qwen7b_source_prepare.py",
            "training_qwen7b_gsm8k_dataset_prepare.py",
            "training_qwen7b_gsm8k_preregister.py",
            "training_qwen7b_gsm8k_elastic_live_probe.py",
            "training_qwen7b_gsm8k_benchmark_live_probe.py",
            "training_qwen7b_gsm8k_cleanup.py",
            "training_qwen7b_gsm8k_showcase_pack.py",
            "training_qwen7b_gsm8k_showcase_check.py",
        }
        if any(command not in text for command in required_commands):
            errors.append("showcase_reproduction_commands_incomplete")
    if adapter is not None:
        try:
            with zipfile.ZipFile(adapter, "r") as archive:
                names = archive.namelist()
                if (
                    len(names) != len(set(names))
                    or not {"adapter_config.json", "adapter_model.safetensors"}.issubset(names)
                    or any(
                        not name
                        or name.startswith(("/", "\\"))
                        or ".." in Path(name).parts
                        for name in names
                    )
                ):
                    errors.append("showcase_adapter_archive_invalid")
                else:
                    config = json.loads(archive.read("adapter_config.json"))
                    if (
                        config.get("base_model_name_or_path") != MODEL_ID
                        or config.get("revision") != MODEL_REVISION
                    ):
                        errors.append("showcase_adapter_identity_invalid")
        except BaseException:
            errors.append("showcase_adapter_archive_unreadable")
    claims = dict(report.get("claims") or {})
    if (
        claims.get("gsm8k_held_out_improvement_only") is not True
        or any(
            claims.get(key) is not False
            for key in (
                "general_reasoning_improvement_claimed",
                "independent_physical_multi_host_claimed",
                "full_parameter_training_claimed",
                "general_availability_claimed",
                "service_level_agreement_claimed",
            )
        )
    ):
        errors.append("showcase_claim_boundaries_invalid")
    if any(
        report.get(key) is not False
        for key in (
            "raw_text_public",
            "token_ids_public",
            "generated_text_public",
            "gold_answers_public",
            "credentials_public",
            "credential_paths_public",
            "private_paths_public",
        )
    ):
        errors.append("showcase_public_safety_flags_invalid")
    safety = public_safety_errors(report)
    if safety:
        errors.append("showcase_public_safety_scan_failed")
    ready_claim = bool(
        report.get("ok") is True
        and report.get("showcase_ready") is True
        and report.get("goal_achieved") is True
        and not report.get("blockers")
    )
    if ready_claim != all(gates.get(key) is True for key in required_gates):
        errors.append("showcase_ready_claim_mismatch")
    if require_ready and not ready_claim:
        errors.append("showcase_readiness_required")
    return {
        "schema": "crowdtensor_qwen7b_gsm8k_showcase_check_v1",
        "ok": not errors,
        "showcase_ready": ready_claim and not errors,
        "public_artifact_safe": not safety,
        "errors": errors,
        "error_count": len(errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(args.report, require_ready=args.require_ready)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"ok={result['ok']} ready={result['showcase_ready']} "
            f"errors={result['error_count']}"
        )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
