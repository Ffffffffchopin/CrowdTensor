from __future__ import annotations

import json
import zipfile
from pathlib import Path

import torch
from safetensors.torch import save_file

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
from scripts import training_qwen7b_gsm8k_showcase_check as showcase_check
from scripts.training_qwen7b_gsm8k_showcase_pack import pack


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _records(correct: int, *, offset: int = 0) -> list[dict]:
    return [
        {
            "example_index": index + offset,
            "prompt_hash": stable_hash({"prompt": index + offset}),
            "gold_hash": stable_hash({"gold": index + offset}),
            "normalized_exact_match": index < correct,
        }
        for index in range(128)
    ]


def _generation(correct: int, *, offset: int = 0) -> dict:
    records = _records(correct, offset=offset)
    return {
        "example_count": 128,
        "normalized_exact_match_count": correct,
        "normalized_exact_match": correct / 128,
        "strict_exact_match": max(0, correct - 2) / 128,
        "valid_answer_rate": 1.0,
        "records": records,
        "records_hash": stable_hash(records),
    }


def _fixture(tmp_path: Path, *, after_correct: int = 50) -> dict[str, Path]:
    source = {
        "schema": SOURCE_LAYOUT_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "parameter_count": MODEL_PARAMETER_COUNT,
        "source_verified": True,
    }
    source["content_hash"] = stable_hash(source)
    dataset = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "benchmark_example_count": 128,
        "train_test_split_isolation_verified": True,
        "benchmark_indexes_hash": stable_hash(list(range(128))),
        "benchmark_prompt_hash": stable_hash({"prompts": list(range(128))}),
        "benchmark_gold_hash": stable_hash({"gold": list(range(128))}),
        "private_train_payload_hash": stable_hash({"private": "train"}),
        "private_benchmark_payload_hash": stable_hash({"private": "benchmark"}),
        "train_token_hash": stable_hash({"tokens": "train"}),
        "validation_token_hash": stable_hash({"tokens": "validation"}),
        "confirmatory_fresh_holdout": True,
        "benchmark_excluded_example_count": 128,
        "benchmark_overlap_with_excluded_count": 0,
        "benchmark_excluded_indexes_hash": stable_hash(list(range(128, 256))),
    }
    dataset["content_hash"] = dataset_manifest_content_hash(dataset)
    source_path = _write(tmp_path / "source.json", source)
    dataset_path = _write(tmp_path / "dataset.json", dataset)
    development = {
        "ok": True,
        "mode": "both",
        "worker": {
            "passes": {
                "base": _generation(40, offset=128),
                "adapter": _generation(30, offset=128),
            }
        },
        "cleanup": {"live_resources_left_running": False},
    }
    development_path = _write(tmp_path / "development.json", development)
    preregistration = {
        "schema": "crowdtensor_qwen7b_gsm8k_showcase_preregistration_v1",
        "registered_at": "2026-07-19T00:00:00+00:00",
        "benchmark": {
            "primary_metric": "normalized_gsm8k_exact_match",
            "primary_practical_improvement_threshold": 0.02,
            "maximum_valid_answer_rate_degradation": 0.01,
            "maximum_validation_loss_relative_degradation": 0.02,
        },
        "training": {
            "optimizer_steps": 256,
            "replacement_step": 128,
            "microbatches_per_step": 4,
            "sequence_length": 256,
            "learning_rate": 0.0001,
            "lora_rank": 4,
            "lora_alpha": 8,
        },
        "model": {"source_layout_hash": sha256_file(source_path)},
        "dataset": {
            "dataset_manifest_hash": sha256_file(dataset_path),
            "benchmark_indexes_hash": dataset["benchmark_indexes_hash"],
            "benchmark_prompt_hash": dataset["benchmark_prompt_hash"],
            "benchmark_gold_hash": dataset["benchmark_gold_hash"],
            "confirmatory_fresh_holdout": True,
            "benchmark_excluded_indexes_hash": dataset[
                "benchmark_excluded_indexes_hash"
            ],
            "benchmark_overlap_with_excluded_count": 0,
        },
        "attempt_budget": {
            "current_training_attempt_number": 3,
            "maximum_full_live_training_attempts": 3,
        },
        "development_evidence": {
            "prior_benchmark_used_for_method_development": True,
            "prior_benchmark_report_hash": sha256_file(development_path),
            "prior_benchmark_examples_reused_for_confirmation": False,
        },
    }
    preregistration["content_hash"] = stable_hash(preregistration)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    adapter_model = adapter_dir / "adapter_model.safetensors"
    save_file({"base_model.model.model.layers.0.x": torch.ones(1)}, adapter_model)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": MODEL_ID,
                "revision": MODEL_REVISION,
            }
        )
    )
    adapter_zip = tmp_path / "adapter.zip"
    with zipfile.ZipFile(adapter_zip, "w") as archive:
        archive.write(adapter_dir / "adapter_config.json", "adapter_config.json")
        archive.write(adapter_model, "adapter_model.safetensors")
    training = {
        "ok": True,
        "training_ready": True,
        "live_run_performed": True,
        "started_at": "2026-07-19T02:00:00+00:00",
        "target_steps": 256,
        "replacement_step": 128,
        "microbatches_per_step": 4,
        "learning_rate": 0.0001,
        "lora_rank": 4,
        "lora_alpha": 8,
        "mock_runtime_used": False,
        "cpu_fallback_used": False,
        "physical_multi_host_verified": False,
        "full_parameter_training_claimed": False,
        "training_budget": {
            "optimizer_steps": 256,
            "training_non_padding_token_count": 262144,
            "replacement_after_step": 128,
            "sequence_length": 256,
        },
        "evidence": {
            "exactly_once_optimizer_commits_verified": True,
            "old_kernels_deleted_before_replacement": True,
            "entirely_new_miner_sessions_verified": True,
        },
        "source": {
            "source_layout_hash": sha256_file(source_path),
            "source_content_hash": source["content_hash"],
        },
        "dataset": {
            "dataset_manifest_hash": sha256_file(dataset_path),
            "private_train_payload_hash": dataset.get("private_train_payload_hash"),
            "train_token_hash": dataset.get("train_token_hash"),
            "validation_token_hash": dataset.get("validation_token_hash"),
        },
        "adapter": {
            "adapter_file_hash": sha256_file(adapter_model),
            "archive_hash": sha256_file(adapter_zip),
        },
        "cleanup": {"live_resources_left_running": False},
    }
    base = _generation(40)
    baseline = {
        "ok": True,
        "mode": "base",
        "live_run_performed": True,
        "finished_at": "2026-07-19T01:00:00+00:00",
        "worker": {
            "passes": {"base": base},
            "benchmark_prompt_hash": dataset["benchmark_prompt_hash"],
            "benchmark_gold_hash": dataset["benchmark_gold_hash"],
        },
        "benchmark_payload_hash": dataset.get("private_benchmark_payload_hash"),
        "validation_payload_hash": dataset.get("validation_token_hash"),
        "cleanup": {"live_resources_left_running": False},
    }
    post = {
        "ok": True,
        "mode": "both",
        "benchmark_payload_hash": dataset.get("private_benchmark_payload_hash"),
        "validation_payload_hash": dataset.get("validation_token_hash"),
        "adapter_archive_hash": sha256_file(adapter_zip),
        "worker": {
            "standard_peft_reload_verified": True,
            "adapter_file_hash": sha256_file(adapter_model),
            "benchmark_prompt_hash": dataset["benchmark_prompt_hash"],
            "benchmark_gold_hash": dataset["benchmark_gold_hash"],
            "passes": {
                "base": base,
                "adapter": _generation(after_correct),
                "base_validation": {"mean_loss": 2.0, "perplexity": 7.38},
                "adapter_validation": {"mean_loss": 1.8, "perplexity": 6.05},
            },
        },
        "cleanup": {"live_resources_left_running": False},
    }
    preregistration_path = _write(tmp_path / "preregistration.json", preregistration)
    training_path = _write(tmp_path / "training.json", training)
    baseline_path = _write(tmp_path / "baseline.json", baseline)
    post_path = _write(tmp_path / "post.json", post)
    cleanup = {
        "schema": "crowdtensor_qwen7b_gsm8k_cleanup_audit_v1",
        "ok": True,
        "cleanup_ready": True,
        "remove_requested": True,
        "evidence_hashes": {
            "dataset_manifest": sha256_file(dataset_path),
            "additional_dataset_manifests": stable_hash(
                ["sha256:" + "1" * 64, "sha256:" + "2" * 64]
            ),
            "training_report": sha256_file(training_path),
            "baseline_report": sha256_file(baseline_path),
            "post_benchmark_report": sha256_file(post_path),
        },
        "private_payloads": {
            role: {
                "expected_hash": dataset[field],
                "present_before_cleanup": True,
                "hash_verified_before_cleanup": True,
                "removed": True,
                "path_public": False,
                "raw_content_public": False,
            }
            for role, field in (
                ("train", "private_train_payload_hash"),
                ("benchmark", "private_benchmark_payload_hash"),
            )
        },
        "dataset_manifest_count": 3,
        "all_private_payloads_removed": True,
        "dataset_transient_directories_absent": True,
        "runtime_private_directories_absent": True,
        "training_live_cleanup_verified": True,
        "baseline_live_cleanup_verified": True,
        "post_benchmark_live_cleanup_verified": True,
        "live_resources_left_running": False,
        "blockers": [],
        "raw_text_public": False,
        "token_ids_public": False,
        "gold_answers_public": False,
        "credentials_public": False,
        "credential_paths_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    for index in (1, 2):
        for role, field in (
            ("train", "private_train_payload_hash"),
            ("benchmark", "private_benchmark_payload_hash"),
        ):
            cleanup["private_payloads"][f"additional_{index}_{role}"] = {
                "expected_hash": dataset[field],
                "present_before_cleanup": True,
                "hash_verified_before_cleanup": True,
                "removed": True,
                "path_public": False,
                "raw_content_public": False,
            }
    cleanup["content_hash"] = stable_hash(cleanup)
    cleanup_path = _write(tmp_path / "cleanup.json", cleanup)
    return {
        "source": source_path,
        "dataset": dataset_path,
        "preregistration": preregistration_path,
        "training": training_path,
        "baseline": baseline_path,
        "post": post_path,
        "development": development_path,
        "cleanup": cleanup_path,
        "adapter": adapter_zip,
    }


def test_showcase_pack_and_strict_checker_accept_improved_7b_run(
    tmp_path: Path, monkeypatch
) -> None:
    values = _fixture(tmp_path)
    output = tmp_path / "rc"
    report = pack(
        output,
        source_path=values["source"],
        dataset_path=values["dataset"],
        preregistration_path=values["preregistration"],
        training_path=values["training"],
        baseline_path=values["baseline"],
        post_benchmark_path=values["post"],
        development_benchmark_path=values["development"],
        cleanup_audit_path=values["cleanup"],
        adapter_path=values["adapter"],
    )
    assert report["showcase_ready"] is True
    assert report["metrics"]["absolute_improvement"] == 10 / 128
    assert (output / "MODEL_CARD.md").is_file()
    assert (output / "SHOWCASE.md").is_file()
    assert (output / "REPRODUCE.md").is_file()
    assert report["artifacts"]["model_card"] == "MODEL_CARD.md"
    assert report["artifacts"]["showcase_doc"] == "SHOWCASE.md"
    assert report["artifacts"]["reproduce_doc"] == "REPRODUCE.md"
    monkeypatch.setattr(
        showcase_check,
        "check_training",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        showcase_check,
        "check_benchmark",
        lambda *_args, **_kwargs: {"ok": True},
    )
    checked = showcase_check.check(
        output / "training_qwen7b_gsm8k_showcase_rc.json", require_ready=True
    )
    assert checked["ok"] is True


def test_showcase_checker_rejects_tampered_release_document(
    tmp_path: Path, monkeypatch
) -> None:
    values = _fixture(tmp_path)
    output = tmp_path / "rc"
    pack(
        output,
        source_path=values["source"],
        dataset_path=values["dataset"],
        preregistration_path=values["preregistration"],
        training_path=values["training"],
        baseline_path=values["baseline"],
        post_benchmark_path=values["post"],
        development_benchmark_path=values["development"],
        cleanup_audit_path=values["cleanup"],
        adapter_path=values["adapter"],
    )
    (output / "SHOWCASE.md").write_text("tampered\n", encoding="utf-8")
    monkeypatch.setattr(
        showcase_check,
        "check_training",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        showcase_check,
        "check_benchmark",
        lambda *_args, **_kwargs: {"ok": True},
    )
    checked = showcase_check.check(
        output / "training_qwen7b_gsm8k_showcase_rc.json", require_ready=True
    )
    assert checked["ok"] is False
    assert "showcase_artifact_hash_mismatch:showcase_doc" in checked["errors"]


def test_showcase_checker_rejects_rebound_benchmark_input(
    tmp_path: Path, monkeypatch
) -> None:
    values = _fixture(tmp_path)
    output = tmp_path / "rc"
    pack(
        output,
        source_path=values["source"],
        dataset_path=values["dataset"],
        preregistration_path=values["preregistration"],
        training_path=values["training"],
        baseline_path=values["baseline"],
        post_benchmark_path=values["post"],
        development_benchmark_path=values["development"],
        cleanup_audit_path=values["cleanup"],
        adapter_path=values["adapter"],
    )
    report_path = output / "training_qwen7b_gsm8k_showcase_rc.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    post_path = output / report["artifacts"]["post_benchmark"]
    post = json.loads(post_path.read_text(encoding="utf-8"))
    post["benchmark_payload_hash"] = stable_hash({"different": "benchmark"})
    post["content_hash"] = stable_hash(
        {key: value for key, value in post.items() if key != "content_hash"}
    )
    _write(post_path, post)
    report["artifact_hashes"]["post_benchmark"] = sha256_file(post_path)
    report["content_hash"] = stable_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    _write(report_path, report)
    monkeypatch.setattr(
        showcase_check,
        "check_training",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        showcase_check,
        "check_benchmark",
        lambda *_args, **_kwargs: {"ok": True},
    )
    checked = showcase_check.check(report_path, require_ready=True)
    assert checked["ok"] is False
    assert (
        "showcase_benchmark_input_binding_invalid:post_benchmark"
        in checked["errors"]
    )


def test_showcase_pack_rejects_no_primary_metric_improvement(tmp_path: Path) -> None:
    values = _fixture(tmp_path, after_correct=40)
    report = pack(
        tmp_path / "blocked",
        source_path=values["source"],
        dataset_path=values["dataset"],
        preregistration_path=values["preregistration"],
        training_path=values["training"],
        baseline_path=values["baseline"],
        post_benchmark_path=values["post"],
        development_benchmark_path=values["development"],
        cleanup_audit_path=values["cleanup"],
        adapter_path=values["adapter"],
    )
    assert report["showcase_ready"] is False
    assert "primary_metric_improved" in report["blockers"]
