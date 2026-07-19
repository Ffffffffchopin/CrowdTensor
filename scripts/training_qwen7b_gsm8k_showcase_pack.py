#!/usr/bin/env python3
"""Pack the public-safe Qwen2.5-7B elastic GSM8K showcase RC."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from crowdtensor.qwen15b_training import sha256_file, stable_hash
from crowdtensor.qwen7b_gsm8k_showcase import (
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_PARAMETER_COUNT,
    MODEL_REVISION,
)
from scripts.training_cuda_kaggle_common import public_safety_errors, utc_now


SCHEMA = "crowdtensor_qwen7b_gsm8k_elastic_showcase_rc_v1"


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Qwen7B showcase input is not an object")
    return value


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total < 1:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _paired_bootstrap(
    before: list[bool],
    after: list[bool],
    *,
    resamples: int = 10_000,
    seed: int = 20260719,
) -> list[float]:
    if len(before) != len(after) or not before:
        return [0.0, 0.0]
    rng = random.Random(int(seed))
    differences = []
    count = len(before)
    for _ in range(int(resamples)):
        total = 0
        for _sample in range(count):
            index = rng.randrange(count)
            total += int(after[index]) - int(before[index])
        differences.append(total / count)
    differences.sort()
    lower = differences[int(0.025 * (len(differences) - 1))]
    upper = differences[int(0.975 * (len(differences) - 1))]
    return [lower, upper]


def _pass(worker: dict[str, Any], name: str) -> dict[str, Any]:
    return dict((worker.get("passes") or {}).get(name) or {})


def _timestamp(value: Any) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _aligned_vectors(
    before: dict[str, Any], after: dict[str, Any]
) -> tuple[list[bool], list[bool], bool]:
    left = list(before.get("records") or [])
    right = list(after.get("records") or [])
    aligned = bool(
        len(left) == len(right) == 128
        and all(
            int(first.get("example_index", -1)) == int(second.get("example_index", -2))
            and first.get("prompt_hash") == second.get("prompt_hash")
            and first.get("gold_hash") == second.get("gold_hash")
            for first, second in zip(left, right, strict=True)
        )
    )
    return (
        [bool(item.get("normalized_exact_match")) for item in left],
        [bool(item.get("normalized_exact_match")) for item in right],
        aligned,
    )


def build_report(
    *,
    source: dict[str, Any],
    dataset: dict[str, Any],
    preregistration: dict[str, Any],
    training: dict[str, Any],
    baseline: dict[str, Any],
    post_benchmark: dict[str, Any],
    development_benchmark: dict[str, Any],
    cleanup_audit: dict[str, Any],
    adapter_path: Path,
) -> dict[str, Any]:
    baseline_worker = dict(baseline.get("worker") or {})
    post_worker = dict(post_benchmark.get("worker") or {})
    chronological_base = _pass(baseline_worker, "base")
    same_runtime_base = _pass(post_worker, "base")
    adapted = _pass(post_worker, "adapter")
    development_worker = dict(development_benchmark.get("worker") or {})
    development_base = _pass(development_worker, "base")
    development_adapter = _pass(development_worker, "adapter")
    base_validation = _pass(post_worker, "base_validation")
    adapter_validation = _pass(post_worker, "adapter_validation")
    before_vector, after_vector, aligned = _aligned_vectors(
        same_runtime_base, adapted
    )
    before_count = sum(before_vector)
    after_count = sum(after_vector)
    total = len(before_vector)
    before_rate = before_count / total if total else 0.0
    after_rate = after_count / total if total else 0.0
    improvement = after_rate - before_rate
    bootstrap = _paired_bootstrap(before_vector, after_vector)
    threshold = float(
        preregistration.get("benchmark", {}).get(
            "primary_practical_improvement_threshold", 0.02
        )
    )
    practical = improvement >= threshold
    statistical = bootstrap[0] > 0.0
    valid_before = float(same_runtime_base.get("valid_answer_rate") or 0.0)
    valid_after = float(adapted.get("valid_answer_rate") or 0.0)
    loss_before = float(base_validation.get("mean_loss") or 0.0)
    loss_after = float(adapter_validation.get("mean_loss") or 0.0)
    max_valid_drop = float(
        preregistration.get("benchmark", {}).get(
            "maximum_valid_answer_rate_degradation", 0.01
        )
    )
    max_loss_degradation = float(
        preregistration.get("benchmark", {}).get(
            "maximum_validation_loss_relative_degradation", 0.02
        )
    )
    chronological_consistency = bool(
        chronological_base.get("example_count") == 128
        and chronological_base.get("records_hash")
        == same_runtime_base.get("records_hash")
    )
    preregistered_before_training = bool(
        _timestamp(preregistration.get("registered_at")) > 0.0
        and _timestamp(training.get("started_at"))
        >= _timestamp(preregistration.get("registered_at"))
    )
    preregistered_before_baseline_result = bool(
        _timestamp(preregistration.get("registered_at")) > 0.0
        and _timestamp(baseline.get("finished_at"))
        >= _timestamp(preregistration.get("registered_at"))
    )
    baseline_before_training = bool(
        _timestamp(baseline.get("finished_at")) > 0.0
        and _timestamp(training.get("started_at"))
        >= _timestamp(baseline.get("finished_at"))
    )
    registered_training = dict(preregistration.get("training") or {})
    training_budget = dict(training.get("training_budget") or {})
    adapter_file_hash = ""
    if adapter_path.is_file():
        adapter_file_hash = sha256_file(adapter_path)
    gates = {
        "pinned_7b_model": source.get("model_id") == MODEL_ID
        and source.get("model_revision") == MODEL_REVISION
        and int(source.get("parameter_count") or 0) == MODEL_PARAMETER_COUNT
        and source.get("source_verified") is True,
        "pinned_gsm8k_isolated_dataset": dataset.get("dataset_id") == DATASET_ID
        and dataset.get("dataset_revision") == DATASET_REVISION
        and int(dataset.get("benchmark_example_count") or 0) == 128
        and dataset.get("train_test_split_isolation_verified") is True,
        "confirmatory_holdout_not_used_for_development": dataset.get(
            "confirmatory_fresh_holdout"
        )
        is True
        and int(dataset.get("benchmark_excluded_example_count") or 0) >= 128
        and int(dataset.get("benchmark_overlap_with_excluded_count", -1)) == 0
        and preregistration.get("dataset", {}).get("confirmatory_fresh_holdout")
        is True
        and preregistration.get("development_evidence", {}).get(
            "prior_benchmark_used_for_method_development"
        )
        is True
        and preregistration.get("development_evidence", {}).get(
            "prior_benchmark_examples_reused_for_confirmation"
        )
        is False
        and development_benchmark.get("ok") is True
        and development_benchmark.get("mode") == "both"
        and stable_hash(
            sorted(
                int(item.get("example_index", -1))
                for item in development_base.get("records") or []
            )
        )
        == dataset.get("benchmark_excluded_indexes_hash"),
        "preregistration_bound_before_training": preregistration.get("schema")
        == "crowdtensor_qwen7b_gsm8k_showcase_preregistration_v1"
        and preregistration.get("benchmark", {}).get("primary_metric")
        == "normalized_gsm8k_exact_match"
        and preregistered_before_training
        and preregistered_before_baseline_result,
        "real_two_generation_elastic_training": training.get("ok") is True
        and training.get("training_ready") is True
        and training.get("live_run_performed") is True
        and training.get("mock_runtime_used") is False
        and training.get("cpu_fallback_used") is False,
        "large_training_budget": int(training.get("target_steps") or 0) >= 256
        and int(
            (training.get("training_budget") or {}).get(
                "training_non_padding_token_count"
            )
            or 0
        )
        >= 262_144,
        "training_contract_matches_preregistration": int(
            registered_training.get("optimizer_steps") or 0
        )
        == int(training.get("target_steps") or 0)
        and int(registered_training.get("replacement_step") or 0)
        == int(training.get("replacement_step") or 0)
        and int(registered_training.get("microbatches_per_step") or 0)
        == int(training.get("microbatches_per_step") or 0)
        and int(registered_training.get("sequence_length") or 0)
        == int(training_budget.get("sequence_length") or 0)
        and float(registered_training.get("learning_rate") or 0.0)
        == float(training.get("learning_rate") or 0.0)
        and int(registered_training.get("lora_rank") or 0)
        == int(training.get("lora_rank") or 0)
        and int(registered_training.get("lora_alpha") or 0)
        == int(training.get("lora_alpha") or 0)
        and int(
            (preregistration.get("attempt_budget") or {}).get(
                "current_training_attempt_number"
            )
            or 0
        )
        <= int(
            (preregistration.get("attempt_budget") or {}).get(
                "maximum_full_live_training_attempts"
            )
            or 0
        )
        == 3,
        "replacement_exactly_once_verified": (
            training.get("evidence") or {}
        ).get("exactly_once_optimizer_commits_verified")
        is True
        and (training.get("evidence") or {}).get(
            "old_kernels_deleted_before_replacement"
        )
        is True
        and (training.get("evidence") or {}).get(
            "entirely_new_miner_sessions_verified"
        )
        is True,
        "baseline_benchmark_chronological": baseline.get("ok") is True
        and baseline.get("mode") == "base"
        and baseline.get("live_run_performed") is True
        and baseline_before_training,
        "same_runtime_before_after_benchmark": post_benchmark.get("ok") is True
        and post_benchmark.get("mode") == "both"
        and same_runtime_base.get("example_count") == 128
        and adapted.get("example_count") == 128
        and aligned,
        "baseline_reproduction_consistent": chronological_consistency,
        "standard_peft_reloaded": post_worker.get(
            "standard_peft_reload_verified"
        )
        is True
        and bool(adapter_file_hash)
        and (training.get("adapter") or {}).get("archive_hash")
        == adapter_file_hash
        and post_benchmark.get("adapter_archive_hash") == adapter_file_hash
        and post_worker.get("adapter_file_hash")
        == (training.get("adapter") or {}).get("adapter_file_hash"),
        "primary_metric_improved": practical or statistical,
        "no_integrity_degradation": valid_after + max_valid_drop >= valid_before
        and loss_before > 0.0
        and loss_after <= loss_before * (1.0 + max_loss_degradation),
        "complete_cleanup": all(
            value.get("live_resources_left_running") is False
            for value in (
                dict(training.get("cleanup") or {}),
                dict(baseline.get("cleanup") or {}),
                dict(post_benchmark.get("cleanup") or {}),
                dict(development_benchmark.get("cleanup") or {}),
            )
        )
        and cleanup_audit.get("ok") is True
        and cleanup_audit.get("cleanup_ready") is True
        and cleanup_audit.get("all_private_payloads_removed") is True
        and int(cleanup_audit.get("dataset_manifest_count") or 0) >= 3
        and cleanup_audit.get("runtime_private_directories_absent") is True
        and cleanup_audit.get("live_resources_left_running") is False,
        "claim_boundaries_preserved": training.get("physical_multi_host_verified")
        is False
        and training.get("full_parameter_training_claimed") is False,
    }
    metrics = {
        "primary_metric": "normalized_gsm8k_exact_match",
        "example_count": total,
        "before_correct": before_count,
        "after_correct": after_count,
        "before_rate": before_rate,
        "after_rate": after_rate,
        "absolute_improvement": improvement,
        "relative_improvement": (
            improvement / before_rate if before_rate > 0.0 else None
        ),
        "before_wilson_95pct": _wilson(before_count, total),
        "after_wilson_95pct": _wilson(after_count, total),
        "paired_bootstrap_improvement_95pct": bootstrap,
        "practical_threshold": threshold,
        "practical_improvement": practical,
        "statistical_improvement": statistical,
        "before_valid_answer_rate": valid_before,
        "after_valid_answer_rate": valid_after,
        "before_strict_exact_match": float(
            same_runtime_base.get("strict_exact_match") or 0.0
        ),
        "after_strict_exact_match": float(
            adapted.get("strict_exact_match") or 0.0
        ),
        "before_validation_loss": loss_before,
        "after_validation_loss": loss_after,
        "before_validation_perplexity": float(
            base_validation.get("perplexity") or 0.0
        ),
        "after_validation_perplexity": float(
            adapter_validation.get("perplexity") or 0.0
        ),
        "correctness_vectors_hash": stable_hash(
            {"before": before_vector, "after": after_vector}
        ),
        "correctness_values_public": False,
    }
    report = {
        "schema": SCHEMA,
        "ok": all(gates.values()),
        "showcase_ready": all(gates.values()),
        "goal_achieved": all(gates.values()),
        "generated_at": utc_now(),
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "parameter_count": MODEL_PARAMETER_COUNT,
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "benchmark_example_count": 128,
            "train_test_split_isolation_verified": True,
            "confirmatory_fresh_holdout": dataset.get(
                "confirmatory_fresh_holdout"
            )
            is True,
            "benchmark_excluded_example_count": int(
                dataset.get("benchmark_excluded_example_count") or 0
            ),
            "benchmark_overlap_with_excluded_count": int(
                dataset.get("benchmark_overlap_with_excluded_count") or 0
            ),
        },
        "training": {
            **dict(training.get("training_budget") or {}),
            "learning_rate": float(training.get("learning_rate") or 0.0),
            "lora_rank": int(training.get("lora_rank") or 0),
            "lora_alpha": int(training.get("lora_alpha") or 0),
            "attempt_number": int(
                (preregistration.get("attempt_budget") or {}).get(
                    "current_training_attempt_number"
                )
                or 0
            ),
            "maximum_training_attempts": int(
                (preregistration.get("attempt_budget") or {}).get(
                    "maximum_full_live_training_attempts"
                )
                or 0
            ),
        },
        "metrics": metrics,
        "development_result": {
            "example_count": int(development_base.get("example_count") or 0),
            "before_correct": int(
                development_base.get("normalized_exact_match_count") or 0
            ),
            "after_correct": int(
                development_adapter.get("normalized_exact_match_count") or 0
            ),
            "used_for_final_success_claim": False,
        },
        "adapter": {
            "archive_hash": adapter_file_hash,
            "standard_peft_format": True,
            "standard_peft_reload_verified": post_worker.get(
                "standard_peft_reload_verified"
            )
            is True,
            "adapter_tensor_values_public": False,
        },
        "gates": gates,
        "blockers": sorted(key for key, value in gates.items() if not value),
        "execution_boundary": (
            "Kaggle logical multi-node LoRA SFT; not independent physical "
            "multi-host or full-parameter training"
        ),
        "claims": {
            "gsm8k_held_out_improvement_only": True,
            "general_reasoning_improvement_claimed": False,
            "independent_physical_multi_host_claimed": False,
            "full_parameter_training_claimed": False,
            "general_availability_claimed": False,
            "service_level_agreement_claimed": False,
        },
        "cleanup": {
            "training": dict(training.get("cleanup") or {}),
            "baseline_benchmark": dict(baseline.get("cleanup") or {}),
            "post_benchmark": dict(post_benchmark.get("cleanup") or {}),
            "development_benchmark": dict(
                development_benchmark.get("cleanup") or {}
            ),
            "final_audit": {
                "cleanup_ready": cleanup_audit.get("cleanup_ready"),
                "all_private_payloads_removed": cleanup_audit.get(
                    "all_private_payloads_removed"
                ),
                "runtime_private_directories_absent": cleanup_audit.get(
                    "runtime_private_directories_absent"
                ),
            },
            "live_resources_left_running": not gates["complete_cleanup"],
        },
        "raw_text_public": False,
        "token_ids_public": False,
        "generated_text_public": False,
        "gold_answers_public": False,
        "credentials_public": False,
        "credential_paths_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    safety = public_safety_errors(report)
    if safety:
        report["ok"] = False
        report["showcase_ready"] = False
        report["goal_achieved"] = False
        report["blockers"].append("public_artifact_safety_failed")
        report["public_artifact_safe"] = False
    report["content_hash"] = stable_hash(report)
    return report


def _model_card(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    training = report["training"]
    return f"""---
base_model: {MODEL_ID}
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
datasets:
- {DATASET_ID}
---

# CrowdTensor Qwen2.5-7B GSM8K LoRA

This standard PEFT Adapter is the output of a real CrowdTensor elastic
four-stage LoRA run over two generations of Kaggle T4x2 Runtimes.

## Result

| Metric | Frozen base | Adapter | Change |
| --- | ---: | ---: | ---: |
| GSM8K normalized exact match (128 held-out test items) | {metrics['before_rate']:.2%} | {metrics['after_rate']:.2%} | {metrics['absolute_improvement'] * 100:+.2f} pp |
| Valid answer rate | {metrics['before_valid_answer_rate']:.2%} | {metrics['after_valid_answer_rate']:.2%} | {metrics['after_valid_answer_rate'] - metrics['before_valid_answer_rate']:+.2%} |
| Reserved-train validation loss | {metrics['before_validation_loss']:.6f} | {metrics['after_validation_loss']:.6f} | {metrics['after_validation_loss'] - metrics['before_validation_loss']:+.6f} |

The preregistered primary metric is normalized GSM8K exact match. Strict
`####` formatting is secondary and cannot make the release pass by itself.

## Training

- Base revision: `{MODEL_REVISION}`
- Dataset revision: `{DATASET_REVISION}` (`main`, MIT)
- 256 optimizer steps, four microbatches per step, sequence length 256
- 262,144 non-padding training tokens
- LoRA rank {training['lora_rank']}, alpha {training['lora_alpha']}, learning rate {training['learning_rate']:.8g}
- Final full-training attempt {training['attempt_number']} of {training['maximum_training_attempts']}
- Four model stages across two concurrent T4x2 Kernels
- First Kernel generation deleted at step 128; fresh Kernels restored central
  checkpoints and completed steps 129-256 exactly once

## Use

Run `examples/qwen7b_gsm8k_compare.py` with this Adapter ZIP or an extracted
Adapter directory. The example validates the model identity, loads the pinned
base in NF4, and prints base and Adapter generations for one supplied question.

## Limitations

This is a bounded GSM8K supervised fine-tuning result. It does not establish a
general reasoning, chat, safety, or out-of-domain improvement. Execution was
Kaggle logical multi-node, not independently administered physical hosts. This
is LoRA, not full-parameter training, and carries no GA, uptime, or service SLA.
Review the Apache-2.0 base license and MIT dataset license before redistribution.
"""


def _showcase_doc(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    training = report["training"]
    bootstrap = metrics["paired_bootstrap_improvement_95pct"]
    return f"""# Qwen2.5-7B Elastic GSM8K Showcase

CrowdTensor trained a real `{MODEL_PARAMETER_COUNT:,}`-parameter
`Qwen2.5-7B-Instruct` LoRA Adapter using two concurrent Kaggle T4x2 Runtimes,
then deleted both Runtimes and resumed on two fresh T4x2 Runtimes from the
central step-128 checkpoint.

## Held-out Result

- Before: {metrics['before_correct']}/128 ({metrics['before_rate']:.2%})
- After: {metrics['after_correct']}/128 ({metrics['after_rate']:.2%})
- Absolute improvement: {metrics['absolute_improvement'] * 100:+.2f} percentage points
- Paired bootstrap 95% improvement interval: [{bootstrap[0] * 100:+.2f}, {bootstrap[1] * 100:+.2f}] percentage points
- Validation loss: {metrics['before_validation_loss']:.6f} -> {metrics['after_validation_loss']:.6f}

The 128 test items, prompt template, generation settings, exact-match parser,
training budget, practical threshold, and bootstrap rule were hash-bound before
the 256-step training run. Test examples were stored in a separate private
payload and were not attached to either training Kernel generation.

This was full-training attempt {training['attempt_number']} of {training['maximum_training_attempts']}.
A prior disjoint development benchmark
showed that the `1e-4` update was too strong despite lower validation loss. The
final `{training['learning_rate']:.8g}` learning rate was therefore registered before this training run,
and the confirmatory 128-item holdout excludes every development example.

## Elastic Evidence

- Steps 1-128 and 129-256 are contiguous and exactly once.
- All first-generation Kernels were deleted before replacement launch.
- The Coordinator observed a zero-Miner pause while retaining four stage
  checkpoints independently of old Kernel disks.
- Four new stage sessions restored Adapter, optimizer, GradScaler, and RNG
  state before continuing.
- The exported standard PEFT Adapter was reloaded into a 4-bit 7B model for the
  same-runtime before/after benchmark.
- Temporary training/benchmark Kernels, private Kaggle Datasets, tunnel,
  Coordinator, checkpoint payloads, and local private runtime were removed.

## Verify

```bash
PYTHONPATH=. python scripts/training_qwen7b_gsm8k_showcase_check.py \\
  --report training_qwen7b_gsm8k_showcase_rc.json \\
  --require-ready --json
```

## Boundary

This demonstrates Kaggle logical multi-node elastic LoRA SFT and a held-out
GSM8K improvement. It is not physical multi-host evidence, full-parameter
training, broad reasoning proof, GA, or an SLA.
"""


def _reproduce_doc(report: dict[str, Any]) -> str:
    training = report["training"]
    return f"""# Reproduce the Qwen2.5-7B GSM8K Showcase

The commands below regenerate private tokenized inputs from pinned public
sources. Keep Kaggle credentials and generated private payloads outside public
artifacts. Register the decision rule before launching the baseline; both the
registration and baseline must finish before training starts.

```bash
export CT_KAGGLE_TOKEN_FILE=/absolute/private/path/to/kaggle-token-file
export CT_KAGGLE_RAW_TOKEN_FILE=/absolute/private/path/to/raw-kaggle-token
export CT_KAGGLE_RAW_TOKEN_USERNAME=private-account-name

PYTHONPATH=. python scripts/training_qwen7b_source_prepare.py \\
  --output-dir dist/qwen7b-source --json
PYTHONPATH=. python scripts/training_qwen7b_gsm8k_dataset_prepare.py \\
  --output-dir dist/qwen7b-gsm8k-data \\
  --sequence-length 256 --train-sequence-count 1024 \\
  --validation-sequence-count 16 --benchmark-example-count 128 \\
  --benchmark-seed 20260721 \\
  --exclude-benchmark-payload /absolute/private/path/to/development-benchmark-payload.json \\
  --json
PYTHONPATH=. python scripts/training_qwen7b_gsm8k_preregister.py \\
  --output-dir dist/qwen7b-gsm8k-preregistration \\
  --source-layout dist/qwen7b-source/qwen7b_source_layout.json \\
  --dataset-manifest dist/qwen7b-gsm8k-data/training_qwen7b_gsm8k_dataset_prepare.json \\
  --learning-rate {training['learning_rate']:.8g} \\
  --attempt-number {training['attempt_number']} \\
  --development-benchmark-report /absolute/private/path/to/development-benchmark-report.json \\
  --require-confirmatory-fresh-holdout \\
  --json
PYTHONPATH=. python scripts/training_qwen7b_gsm8k_benchmark_live_probe.py \\
  --output-dir dist/qwen7b-gsm8k-baseline \\
  --token-file "$CT_KAGGLE_TOKEN_FILE" \\
  --raw-token-file "$CT_KAGGLE_RAW_TOKEN_FILE" \\
  --raw-token-username "$CT_KAGGLE_RAW_TOKEN_USERNAME" \\
  --benchmark-payload dist/qwen7b-gsm8k-data/qwen7b_gsm8k_benchmark_private.json \\
  --train-payload dist/qwen7b-gsm8k-data/qwen7b_gsm8k_train_private.json \\
  --mode base --max-new-tokens 256 --batch-size 8 --json
PYTHONPATH=. python scripts/training_qwen7b_gsm8k_elastic_live_probe.py \\
  --output-dir dist/qwen7b-gsm8k-training \\
  --token-file "$CT_KAGGLE_TOKEN_FILE" \\
  --raw-token-file "$CT_KAGGLE_RAW_TOKEN_FILE" \\
  --raw-token-username "$CT_KAGGLE_RAW_TOKEN_USERNAME" \\
  --source-layout dist/qwen7b-source/qwen7b_source_layout.json \\
  --dataset-manifest dist/qwen7b-gsm8k-data/training_qwen7b_gsm8k_dataset_prepare.json \\
  --train-payload dist/qwen7b-gsm8k-data/qwen7b_gsm8k_train_private.json \\
  --target-steps 256 --replacement-step 128 --microbatches-per-step 4 \\
  --learning-rate {training['learning_rate']:.8g} \\
  --lora-rank {training['lora_rank']} --lora-alpha {training['lora_alpha']} --json
PYTHONPATH=. python scripts/training_qwen7b_gsm8k_benchmark_live_probe.py \\
  --output-dir dist/qwen7b-gsm8k-post \\
  --token-file "$CT_KAGGLE_TOKEN_FILE" \\
  --raw-token-file "$CT_KAGGLE_RAW_TOKEN_FILE" \\
  --raw-token-username "$CT_KAGGLE_RAW_TOKEN_USERNAME" \\
  --benchmark-payload dist/qwen7b-gsm8k-data/qwen7b_gsm8k_benchmark_private.json \\
  --train-payload dist/qwen7b-gsm8k-data/qwen7b_gsm8k_train_private.json \\
  --adapter dist/qwen7b-gsm8k-training/training_qwen7b_standard_peft_adapter.zip \\
  --mode both --max-new-tokens 256 --batch-size 8 --json
PYTHONPATH=. python scripts/training_qwen7b_gsm8k_cleanup.py \\
  --output-dir dist/qwen7b-gsm8k-cleanup \\
  --dataset-manifest dist/qwen7b-gsm8k-data/training_qwen7b_gsm8k_dataset_prepare.json \\
  --training-report dist/qwen7b-gsm8k-training/training_qwen7b_gsm8k_elastic_live_probe.json \\
  --baseline-report dist/qwen7b-gsm8k-baseline/training_qwen7b_gsm8k_benchmark_live_probe.json \\
  --post-benchmark-report dist/qwen7b-gsm8k-post/training_qwen7b_gsm8k_benchmark_live_probe.json \\
  --private-train-payload dist/qwen7b-gsm8k-data/qwen7b_gsm8k_train_private.json \\
  --private-benchmark-payload dist/qwen7b-gsm8k-data/qwen7b_gsm8k_benchmark_private.json \\
  --remove --json
PYTHONPATH=. python scripts/training_qwen7b_gsm8k_showcase_pack.py \\
  --output-dir dist/qwen7b-gsm8k-showcase-rc \\
  --source dist/qwen7b-source/qwen7b_source_layout.json \\
  --dataset dist/qwen7b-gsm8k-data/training_qwen7b_gsm8k_dataset_prepare.json \\
  --preregistration dist/qwen7b-gsm8k-preregistration/training_qwen7b_gsm8k_preregistration.json \\
  --training dist/qwen7b-gsm8k-training/training_qwen7b_gsm8k_elastic_live_probe.json \\
  --baseline dist/qwen7b-gsm8k-baseline/training_qwen7b_gsm8k_benchmark_live_probe.json \\
  --post-benchmark dist/qwen7b-gsm8k-post/training_qwen7b_gsm8k_benchmark_live_probe.json \\
  --development-benchmark /absolute/private/path/to/development-benchmark-report.json \\
  --cleanup-audit dist/qwen7b-gsm8k-cleanup/training_qwen7b_gsm8k_cleanup_audit.json \\
  --adapter dist/qwen7b-gsm8k-training/training_qwen7b_standard_peft_adapter.zip \\
  --json
```

Verify the packed result with:

```bash
PYTHONPATH=. python scripts/training_qwen7b_gsm8k_showcase_check.py \\
  --report dist/qwen7b-gsm8k-showcase-rc/training_qwen7b_gsm8k_showcase_rc.json \\
  --require-ready --json
```

The fixed identities are `{MODEL_ID}` revision `{MODEL_REVISION}` and
`{DATASET_ID}` revision `{DATASET_REVISION}`. This workflow uses Kaggle logical
multi-node LoRA SFT, not independently administered physical machines or
full-parameter training.
"""


def pack(
    output_dir: str | Path,
    *,
    source_path: str | Path,
    dataset_path: str | Path,
    preregistration_path: str | Path,
    training_path: str | Path,
    baseline_path: str | Path,
    post_benchmark_path: str | Path,
    development_benchmark_path: str | Path,
    cleanup_audit_path: str | Path,
    adapter_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists():
        shutil.rmtree(output)
    evidence = output / "evidence"
    artifacts = output / "artifacts"
    evidence.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    inputs = {
        "source": Path(source_path).resolve(),
        "dataset": Path(dataset_path).resolve(),
        "preregistration": Path(preregistration_path).resolve(),
        "training": Path(training_path).resolve(),
        "baseline": Path(baseline_path).resolve(),
        "post_benchmark": Path(post_benchmark_path).resolve(),
        "development_benchmark": Path(development_benchmark_path).resolve(),
        "cleanup_audit": Path(cleanup_audit_path).resolve(),
    }
    copied: dict[str, str] = {}
    hashes: dict[str, str] = {}
    loaded: dict[str, dict[str, Any]] = {}
    for name, source_file in inputs.items():
        destination = evidence / f"{name}.json"
        shutil.copyfile(source_file, destination)
        copied[name] = str(destination.relative_to(output))
        hashes[name] = sha256_file(destination)
        loaded[name] = _load(destination)
    adapter_source = Path(adapter_path).resolve()
    adapter_destination = artifacts / "training_qwen7b_standard_peft_adapter.zip"
    shutil.copyfile(adapter_source, adapter_destination)
    copied["adapter"] = str(adapter_destination.relative_to(output))
    hashes["adapter"] = sha256_file(adapter_destination)
    report = build_report(
        source=loaded["source"],
        dataset=loaded["dataset"],
        preregistration=loaded["preregistration"],
        training=loaded["training"],
        baseline=loaded["baseline"],
        post_benchmark=loaded["post_benchmark"],
        development_benchmark=loaded["development_benchmark"],
        cleanup_audit=loaded["cleanup_audit"],
        adapter_path=adapter_destination,
    )
    model_card = output / "MODEL_CARD.md"
    showcase_doc = output / "SHOWCASE.md"
    reproduce_doc = output / "REPRODUCE.md"
    model_card.write_text(_model_card(report), encoding="utf-8")
    showcase_doc.write_text(_showcase_doc(report), encoding="utf-8")
    reproduce_doc.write_text(_reproduce_doc(report), encoding="utf-8")
    copied["model_card"] = str(model_card.relative_to(output))
    copied["showcase_doc"] = str(showcase_doc.relative_to(output))
    copied["reproduce_doc"] = str(reproduce_doc.relative_to(output))
    hashes["model_card"] = sha256_file(model_card)
    hashes["showcase_doc"] = sha256_file(showcase_doc)
    hashes["reproduce_doc"] = sha256_file(reproduce_doc)
    report["artifacts"] = copied
    report["artifact_hashes"] = hashes
    report["content_hash"] = stable_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    path = output / "training_qwen7b_gsm8k_showcase_rc.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--post-benchmark", required=True)
    parser.add_argument("--development-benchmark", required=True)
    parser.add_argument("--cleanup-audit", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = pack(
        args.output_dir,
        source_path=args.source,
        dataset_path=args.dataset,
        preregistration_path=args.preregistration,
        training_path=args.training,
        baseline_path=args.baseline,
        post_benchmark_path=args.post_benchmark,
        development_benchmark_path=args.development_benchmark,
        cleanup_audit_path=args.cleanup_audit,
        adapter_path=args.adapter,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "ok": report["ok"],
                    "showcase_ready": report["showcase_ready"],
                    "blockers": report["blockers"],
                    "metrics": report["metrics"],
                },
                sort_keys=True,
            )
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
