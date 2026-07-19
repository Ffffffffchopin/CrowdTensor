from __future__ import annotations

import json
import zipfile
from contextlib import ExitStack
from pathlib import Path

import torch
import pytest
from safetensors.torch import load_file, save

from crowdtensor.qwen15b_four_gpu_runtime import _training_row_inputs_and_labels
from crowdtensor.qwen15b_training import (
    QWEN_STAGE_CHECKPOINT_SCHEMA,
    QwenStageSpec,
    materialize_stage_shard_from_layout,
    read_safetensors_header,
    stable_hash,
)
from crowdtensor.elastic_training_runtime import _validated_checkpoint_manifest
from crowdtensor.qwen7b_gsm8k_showcase import (
    MODEL_ID,
    MODEL_REVISION,
    _packed_rows,
    _prompt_ids,
    extract_gsm8k_answer,
    select_benchmark_indexes,
)
from scripts.training_qwen15b_four_gpu_package import build_package
from scripts.training_qwen7b_gsm8k_benchmark_package import (
    build_package as build_benchmark_package,
)
from scripts.training_qwen7b_gsm8k_benchmark_live_probe import (
    SCHEMA as BENCHMARK_SCHEMA,
    _materialize_standard_peft_adapter,
)
from scripts.training_qwen7b_gsm8k_benchmark_check import check as check_benchmark
from examples.qwen7b_gsm8k_compare import _adapter_directory


class _Tokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}


def test_chat_template_mapping_and_gsm8k_answer_extraction() -> None:
    assert _prompt_ids(_Tokenizer(), "private question") == [1, 2, 3]
    assert extract_gsm8k_answer("work\n#### 1,234") == "1234"
    assert extract_gsm8k_answer("therefore 42") == "42"
    assert extract_gsm8k_answer("therefore 42", require_marker=True) == ""


def test_packed_sft_rows_preserve_masked_labels() -> None:
    rows = _packed_rows(
        [([1, 2, 3, 4], [-100, -100, 3, 4])],
        sequence_length=8,
        row_count=2,
    )
    assert len(rows) == 2
    assert all(len(row["input_ids"]) == len(row["labels"]) == 8 for row in rows)
    assert sum(row["supervised_token_count"] for row in rows) == 8
    inputs, labels = _training_row_inputs_and_labels(rows[0])
    assert inputs == rows[0]["input_ids"]
    assert labels == rows[0]["labels"]


def test_confirmatory_benchmark_selection_is_deterministic_and_disjoint() -> None:
    excluded = set(range(128))
    first = select_benchmark_indexes(
        total_count=1319,
        example_count=128,
        seed=20260720,
        excluded_indexes=excluded,
    )
    second = select_benchmark_indexes(
        total_count=1319,
        example_count=128,
        seed=20260720,
        excluded_indexes=excluded,
    )
    assert first == second
    assert len(first) == len(set(first)) == 128
    assert not set(first) & excluded


def test_multifile_stage_materializer_reads_only_owned_ranges(tmp_path: Path) -> None:
    values = {
        "model.embed_tokens.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "model.layers.0.self_attn.q_proj.weight": torch.arange(
            4, dtype=torch.float32
        ).reshape(2, 2),
    }
    files = {
        "model-00001.safetensors": save(
            {"model.embed_tokens.weight": values["model.embed_tokens.weight"]}
        ),
        "model-00002.safetensors": save(
            {
                "model.layers.0.self_attn.q_proj.weight": values[
                    "model.layers.0.self_attn.q_proj.weight"
                ]
            }
        ),
    }
    shards = {}
    for name, payload in files.items():
        path = tmp_path / name
        path.write_bytes(payload)
        header_length, header = read_safetensors_header(path)
        shards[name] = {"header_length": header_length, "header": header}
    layout = {
        "model_id": "example/qwen",
        "model_revision": "revision",
        "weight_map": {
            "model.embed_tokens.weight": "model-00001.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00002.safetensors",
        },
        "shards": shards,
    }
    layout["content_hash"] = stable_hash(layout)
    requested = []

    def reader(url: str, start: int, end: int) -> bytes:
        filename = url.rsplit("/", 1)[-1]
        requested.append((filename, start, end))
        return files[filename][start : end + 1]

    output = tmp_path / "stage0.safetensors"
    report = materialize_stage_shard_from_layout(
        spec=QwenStageSpec(0, "A", 0, 0, 1, owns_embedding=True),
        source_layout=layout,
        output_path=output,
        range_reader=reader,
    )
    loaded = load_file(output)
    assert set(loaded) == set(values)
    assert all(torch.equal(loaded[name], value) for name, value in values.items())
    assert report["multi_file_source"] is True
    assert report["full_model_file_downloaded"] is False
    assert {item[0] for item in requested} == set(files)


def test_dynamic_7b_training_and_benchmark_packages(tmp_path: Path) -> None:
    tokenized = tmp_path / "train.json"
    tokenized.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "train": [
                    {
                        "input_ids": [1, 2],
                        "labels": [-100, 2],
                    }
                ],
                "validation": [[1, 2]],
            }
        ),
        encoding="utf-8",
    )
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "content_hash": "sha256:" + "a" * 64,
                "weight_map": {"x": "model.safetensors"},
                "shards": {"model.safetensors": {"header": {}}},
            }
        ),
        encoding="utf-8",
    )
    package = build_package(
        tmp_path / "training-package",
        owner="example",
        slug="qwen7b-training",
        role="kernel_a",
        config={"model_type": "qwen2", "num_hidden_layers": 28},
        tokenized_payload_path=tokenized,
        coordinator_url="https://example.invalid",
        coordinator_token="private",
        run_id="run",
        elastic_mode=True,
        miner_id_hash="sha256:" + "b" * 64,
        registration_nonce="private",
        expected_start_step=0,
        segment_end_step=1,
        target_steps=2,
        microbatch_count=1,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        parameter_count=7_615_616_512,
        source_layout_path=layout,
        defer_evaluation=True,
    )
    assert package["schema"] == "crowdtensor_qwen_four_gpu_package_v2"
    assert package["multi_file_source_layout"] is True
    assert package["evaluation_deferred_to_isolated_benchmark"] is True
    kernel_text = Path(package["package_dir"]).joinpath("kernel.py").read_text()
    encoded_package = json.dumps(package)
    assert "https://example.invalid" not in encoded_package
    assert '"coordinator_token": "private"' not in encoded_package
    compile(kernel_text, "kernel.py", "exec")

    benchmark = build_benchmark_package(
        tmp_path / "benchmark-package",
        owner="example",
        slug="qwen7b-benchmark",
        dataset_ref="example/private-input",
        mode="both",
        expected_input_hashes={
            name: "sha256:" + "c" * 64
            for name in (
                "qwen7b_gsm8k_benchmark_private.json",
                "qwen7b_gsm8k_validation_private.json",
                "adapter_config.json",
                "adapter_model.safetensors",
            )
        },
    )
    assert benchmark["ok"] is True
    assert benchmark["input_hash_binding_ready"] is True
    with pytest.raises(ValueError, match="expected input hashes invalid"):
        build_benchmark_package(
            tmp_path / "invalid-benchmark-package",
            owner="example",
            slug="qwen7b-invalid-benchmark",
            dataset_ref="example/private-input",
            mode="both",
        )
    benchmark_kernel = Path(benchmark["package_dir"]).joinpath("kernel.py").read_text()
    assert "training_qwen7b_standard_peft_adapter.zip" not in benchmark_kernel
    assert 'locate("adapter_model.safetensors")' in benchmark_kernel
    compile(
        benchmark_kernel,
        "benchmark-kernel.py",
        "exec",
    )


def test_benchmark_materializes_only_verified_standard_peft_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "adapter.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "adapter_config.json",
            json.dumps(
                {
                    "base_model_name_or_path": MODEL_ID,
                    "revision": MODEL_REVISION,
                }
            ),
        )
        archive.writestr("adapter_model.safetensors", b"adapter")
    output = tmp_path / "materialized"
    _materialize_standard_peft_adapter(source, output)
    assert sorted(path.name for path in output.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]


def test_benchmark_rejects_unsafe_adapter_archive(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("adapter_config.json", "{}")
        archive.writestr("adapter_model.safetensors", b"adapter")
        archive.writestr("../private", b"unsafe")
    with pytest.raises(RuntimeError, match="adapter_archive_invalid"):
        _materialize_standard_peft_adapter(source, tmp_path / "materialized")


def test_compare_example_accepts_standard_peft_zip(tmp_path: Path) -> None:
    source = tmp_path / "adapter.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "adapter_config.json",
            json.dumps(
                {
                    "base_model_name_or_path": MODEL_ID,
                    "revision": MODEL_REVISION,
                }
            ),
        )
        archive.writestr("adapter_model.safetensors", b"adapter")
    with ExitStack() as stack:
        directory = _adapter_directory(str(source), stack)
        assert (directory / "adapter_config.json").is_file()
        assert (directory / "adapter_model.safetensors").is_file()


def test_benchmark_checker_requires_materialized_adapter_input(tmp_path: Path) -> None:
    records = [
        {
            "example_index": index,
            "prompt_hash": stable_hash({"prompt": index}),
            "gold_hash": stable_hash({"gold": index}),
            "answer_valid": True,
            "normalized_exact_match": index < 64,
            "strict_exact_match": index < 60,
        }
        for index in range(128)
    ]
    generation = {
        "example_count": 128,
        "normalized_exact_match_count": 64,
        "valid_answer_count": 128,
        "strict_exact_match_count": 60,
        "normalized_exact_match": 0.5,
        "records": records,
        "records_hash": stable_hash(records),
    }
    worker = {
        "mode": "both",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "kaggle_kernel": True,
        "cuda_device_count": 2,
        "benchmark_example_count": 128,
        "standard_peft_reload_verified": True,
        "input_hashes_verified": True,
        "input_match_counts": {
            name: {"candidate_count": 1, "hash_match_count": 1}
            for name in (
                "qwen7b_gsm8k_benchmark_private.json",
                "qwen7b_gsm8k_validation_private.json",
                "adapter_config.json",
                "adapter_model.safetensors",
            )
        },
        "adapter_file_hash": "sha256:" + "a" * 64,
        "benchmark_prompt_hash": "sha256:" + "b" * 64,
        "benchmark_gold_hash": "sha256:" + "c" * 64,
        "passes": {
            "base": generation,
            "adapter": generation,
            "base_validation": {"sequence_count": 8},
            "adapter_validation": {"sequence_count": 8},
        },
        "ok": True,
        "public_artifact_safe": True,
    }
    worker["content_hash"] = stable_hash(worker)
    report = {
        "schema": BENCHMARK_SCHEMA,
        "ok": True,
        "live_run_performed": True,
        "mode": "both",
        "adapter_input_materialized": False,
        "dataset_attachment_preflight": {"ready": True},
        "worker": worker,
        "cleanup": {
            "kernel_deleted": True,
            "private_dataset_deleted": True,
            "private_runtime_removed": True,
            "live_resources_left_running": False,
        },
        "blockers": [],
        "raw_text_public": False,
        "token_ids_public": False,
        "generated_text_public": False,
        "gold_answers_public": False,
        "credentials_public": False,
        "credential_paths_public": False,
        "private_paths_public": False,
    }
    report["content_hash"] = stable_hash(report)
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    checked = check_benchmark(path, require_ready=True)
    assert checked["ok"] is False
    assert "benchmark_standard_peft_reload_invalid" in checked["errors"]
    report["adapter_input_materialized"] = True
    report["content_hash"] = stable_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    path.write_text(json.dumps(report), encoding="utf-8")
    accepted = check_benchmark(path, require_ready=True)
    assert accepted["ok"] is True


def test_legacy_elastic_checkpoint_validator_binds_dynamic_model_identity() -> None:
    digest = "sha256:" + "a" * 64
    manifest = {
        "schema": QWEN_STAGE_CHECKPOINT_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "stage_id": 0,
        "layer_start": 0,
        "layer_end": 7,
        "global_step": 1,
        "optimizer_step": 1,
        "dataset_cursor": 4,
        "device": "cuda:0",
        "adapter_file": "stage0_adapter.safetensors",
        "adapter_file_hash": digest,
        "adapter_tensor_hash": digest,
        "adapter_tensor_count": 1,
        "optimizer_file": "stage0_optimizer.pt",
        "optimizer_file_hash": digest,
        "grad_scaler_file": "stage0_grad_scaler.pt",
        "grad_scaler_file_hash": digest,
        "grad_scaler_state_present": True,
        "rng_file": "stage0_rng.pt",
        "rng_file_hash": digest,
        "rng_state_present": True,
        "tensor_values_public": False,
        "token_ids_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    manifest["content_hash"] = stable_hash(manifest)
    validated = _validated_checkpoint_manifest(
        manifest,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
    )
    assert validated["model_id"] == MODEL_ID
    with pytest.raises(ValueError, match="elastic_checkpoint_ownership_invalid"):
        _validated_checkpoint_manifest(manifest)
