from __future__ import annotations

import json
from pathlib import Path

from crowdtensor.qwen15b_training import sha256_file
from crowdtensor.qwen7b_gsm8k_showcase import DATASET_MANIFEST_SCHEMA
from scripts.training_qwen7b_gsm8k_cleanup import cleanup
from scripts.training_qwen7b_gsm8k_cleanup_check import check


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> dict[str, Path]:
    dataset_dir = tmp_path / "dataset"
    train = dataset_dir / "qwen7b_gsm8k_train_private.json"
    benchmark = dataset_dir / "qwen7b_gsm8k_benchmark_private.json"
    train.parent.mkdir(parents=True)
    train.write_text("private train\n", encoding="utf-8")
    benchmark.write_text("private benchmark\n", encoding="utf-8")
    manifest = _write(
        dataset_dir / "training_qwen7b_gsm8k_dataset_prepare.json",
        {
            "schema": DATASET_MANIFEST_SCHEMA,
            "private_train_payload_hash": sha256_file(train),
            "private_benchmark_payload_hash": sha256_file(benchmark),
        },
    )
    training = _write(
        tmp_path / "training" / "training.json",
        {
            "cleanup": {
                "all_four_kernels_deleted": True,
                "coordinator_stopped": True,
                "tunnel_stopped": True,
                "private_runtime_removed": True,
                "rendezvous_payloads_removed": True,
                "uncommitted_checkpoint_blobs_removed": True,
                "live_resources_left_running": False,
            }
        },
    )
    benchmark_cleanup = {
        "cleanup": {
            "kernel_deleted": True,
            "private_dataset_deleted": True,
            "private_runtime_removed": True,
            "live_resources_left_running": False,
        }
    }
    baseline = _write(tmp_path / "baseline" / "baseline.json", benchmark_cleanup)
    post = _write(tmp_path / "post" / "post.json", benchmark_cleanup)
    return {
        "manifest": manifest,
        "training": training,
        "baseline": baseline,
        "post": post,
        "train": train,
        "benchmark": benchmark,
    }


def test_cleanup_removes_hash_verified_private_payloads(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    output = tmp_path / "audit"
    report = cleanup(
        output,
        dataset_manifest_path=values["manifest"],
        training_report_path=values["training"],
        baseline_report_path=values["baseline"],
        post_benchmark_report_path=values["post"],
        private_train_payload_path=values["train"],
        private_benchmark_payload_path=values["benchmark"],
        remove=True,
    )
    assert report["cleanup_ready"] is True
    assert not values["train"].exists()
    assert not values["benchmark"].exists()
    checked = check(
        output / "training_qwen7b_gsm8k_cleanup_audit.json",
        require_ready=True,
    )
    assert checked["ok"] is True


def test_cleanup_refuses_hash_mismatched_private_payload(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    manifest = json.loads(values["manifest"].read_text(encoding="utf-8"))
    manifest["private_train_payload_hash"] = "sha256:" + "0" * 64
    _write(values["manifest"], manifest)
    report = cleanup(
        tmp_path / "audit",
        dataset_manifest_path=values["manifest"],
        training_report_path=values["training"],
        baseline_report_path=values["baseline"],
        post_benchmark_report_path=values["post"],
        private_train_payload_path=values["train"],
        private_benchmark_payload_path=values["benchmark"],
        remove=True,
    )
    assert report["cleanup_ready"] is False
    assert values["train"].is_file()
    assert "cleanup_private_payload_hash_mismatch:train" in report["blockers"]


def test_cleanup_removes_additional_manifest_payloads(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    extra_dir = tmp_path / "extra-dataset"
    extra_dir.mkdir()
    extra_train = extra_dir / "qwen7b_gsm8k_train_private.json"
    extra_benchmark = extra_dir / "qwen7b_gsm8k_benchmark_private.json"
    extra_train.write_text("old private train\n", encoding="utf-8")
    extra_benchmark.write_text("old private benchmark\n", encoding="utf-8")
    extra_manifest = _write(
        extra_dir / "training_qwen7b_gsm8k_dataset_prepare.json",
        {
            "schema": DATASET_MANIFEST_SCHEMA,
            "private_train_payload_hash": sha256_file(extra_train),
            "private_benchmark_payload_hash": sha256_file(extra_benchmark),
        },
    )
    output = tmp_path / "audit"
    report = cleanup(
        output,
        dataset_manifest_path=values["manifest"],
        training_report_path=values["training"],
        baseline_report_path=values["baseline"],
        post_benchmark_report_path=values["post"],
        private_train_payload_path=values["train"],
        private_benchmark_payload_path=values["benchmark"],
        additional_dataset_manifest_paths=[extra_manifest],
        remove=True,
    )
    assert report["cleanup_ready"] is True
    assert report["dataset_manifest_count"] == 2
    assert len(report["private_payloads"]) == 4
    assert not extra_train.exists()
    assert not extra_benchmark.exists()
    assert check(
        output / "training_qwen7b_gsm8k_cleanup_audit.json",
        require_ready=True,
    )["ok"] is True
