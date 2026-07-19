#!/usr/bin/env python3
"""Freeze the 7B GSM8K showcase training and benchmark decision rule."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from crowdtensor.qwen15b_training import sha256_file, stable_hash
from crowdtensor.qwen7b_gsm8k_showcase import (
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
)


SCHEMA = "crowdtensor_qwen7b_gsm8k_showcase_preregistration_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--source-layout", required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--attempt-number", type=int, default=1)
    parser.add_argument("--development-benchmark-report", default="")
    parser.add_argument("--require-confirmatory-fresh-holdout", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_manifest).resolve()
    source_path = Path(args.source_layout).resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not 0.0 < args.learning_rate <= 1e-3:
        parser.error("--learning-rate must be in (0, 1e-3]")
    if not 1 <= args.attempt_number <= 3:
        parser.error("--attempt-number must be in [1, 3]")
    if args.require_confirmatory_fresh_holdout and not (
        dataset.get("confirmatory_fresh_holdout") is True
        and int(dataset.get("benchmark_excluded_example_count") or 0) >= 128
        and int(dataset.get("benchmark_overlap_with_excluded_count", -1)) == 0
    ):
        parser.error("dataset manifest is not a zero-overlap confirmatory holdout")
    development_hash = ""
    if args.development_benchmark_report:
        development_path = Path(args.development_benchmark_report).resolve()
        development = json.loads(development_path.read_text(encoding="utf-8"))
        if development.get("mode") != "both" or development.get("ok") is not True:
            parser.error("development benchmark report is not a successful both run")
        development_hash = sha256_file(development_path)
    report = {
        "schema": SCHEMA,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "parameter_count": int(source["parameter_count"]),
            "source_layout_hash": sha256_file(source_path),
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_manifest_hash": sha256_file(dataset_path),
            "benchmark_split": "test",
            "benchmark_example_count": 128,
            "benchmark_indexes_hash": dataset["benchmark_indexes_hash"],
            "benchmark_prompt_hash": dataset["benchmark_prompt_hash"],
            "benchmark_gold_hash": dataset["benchmark_gold_hash"],
            "benchmark_excluded_indexes_hash": dataset.get(
                "benchmark_excluded_indexes_hash"
            ),
            "benchmark_excluded_example_count": int(
                dataset.get("benchmark_excluded_example_count") or 0
            ),
            "benchmark_overlap_with_excluded_count": int(
                dataset.get("benchmark_overlap_with_excluded_count") or 0
            ),
            "confirmatory_fresh_holdout": dataset.get(
                "confirmatory_fresh_holdout"
            )
            is True,
            "train_test_split_isolation_required": True,
        },
        "training": {
            "method": "four-stage LoRA supervised fine-tuning",
            "optimizer_steps": 256,
            "replacement_step": 128,
            "microbatches_per_step": 4,
            "sequence_length": 256,
            "minimum_non_padding_training_tokens": 262144,
            "learning_rate": float(args.learning_rate),
            "lora_rank": 4,
            "lora_alpha": 8,
            "first_generation_kernels_must_be_deleted": True,
            "fresh_replacement_generation_required": True,
            "exactly_once_contiguous_steps_required": True,
        },
        "benchmark": {
            "generation": {
                "do_sample": False,
                "max_new_tokens": 256,
                "batch_size": 8,
                "quantization": "bitsandbytes-nf4-double-quant",
                "compute_dtype": "float16",
            },
            "primary_metric": "normalized_gsm8k_exact_match",
            "primary_practical_improvement_threshold": 0.02,
            "primary_statistical_rule": (
                "paired-bootstrap-95pct-improvement-ci-lower-bound-greater-than-zero"
            ),
            "paired_bootstrap_resamples": 10000,
            "paired_bootstrap_seed": 20260719,
            "confidence_level": 0.95,
            "secondary_metrics": [
                "strict_marker_exact_match",
                "valid_answer_rate",
                "held_out_validation_loss",
                "held_out_validation_perplexity",
            ],
            "maximum_valid_answer_rate_degradation": 0.01,
            "maximum_validation_loss_relative_degradation": 0.02,
            "strict_marker_improvement_is_not_primary_success": True,
            "same_prompt_and_gold_hash_required": True,
            "standard_peft_reload_required": True,
        },
        "attempt_budget": {
            "maximum_full_live_training_attempts": 3,
            "current_training_attempt_number": int(args.attempt_number),
            "queue_or_quota_only_evidence_is_success": False,
            "mock_or_protocol_fixture_is_success": False,
        },
        "development_evidence": {
            "prior_benchmark_used_for_method_development": bool(development_hash),
            "prior_benchmark_report_hash": development_hash,
            "prior_benchmark_examples_reused_for_confirmation": False,
            "raw_development_records_public": False,
        },
        "claims": {
            "kaggle_logical_multi_node": True,
            "independent_physical_multi_host": False,
            "lora_not_full_parameter_training": True,
            "general_reasoning_improvement_claimed": False,
            "production_ga_or_sla_claimed": False,
        },
        "raw_text_public": False,
        "token_ids_public": False,
        "generated_text_public": False,
        "gold_answers_public": False,
        "credentials_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    path = output / "training_qwen7b_gsm8k_preregistration.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.json:
        print(
            json.dumps(
                {
                    "ok": True,
                    "schema": report["schema"],
                    "content_hash": report["content_hash"],
                    "primary_metric": report["benchmark"]["primary_metric"],
                    "practical_threshold": report["benchmark"][
                        "primary_practical_improvement_threshold"
                    ],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
