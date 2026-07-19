from __future__ import annotations

import json

import pytest

from crowdtensor import cli
from crowdtensor import training_cuda_job
from crowdtensor.cli import parse_args
from crowdtensor.training_cuda_job import cleanup_cuda_training_job
from crowdtensor.elastic_training_runtime import ElasticTrainingRuntime
from crowdtensor.training_foundation import training_status
from tests.test_heterogeneous_qwen_training import tiny_source


def test_training_cli_parses_all_user_paths(tmp_path) -> None:
    lora = parse_args(["train", "lora", "--output-dir", str(tmp_path), "--local-steps", "4"])
    status = parse_args(["train", "status", str(tmp_path)])
    resume = parse_args(["train", "resume", str(tmp_path)])
    export = parse_args(["train", "export", str(tmp_path), "--output-dir", str(tmp_path / "export")])
    cleanup = parse_args(["train", "cleanup", str(tmp_path)])
    cancel = parse_args(["train", "cancel", str(tmp_path)])
    serve = parse_args(["train", "serve", "--store", str(tmp_path / "jobs.sqlite3")])
    elastic_status = parse_args(
        [
            "train",
            "elastic-status",
            "--state",
            str(tmp_path / "elastic.sqlite3"),
            "--run-id",
            "run-1",
        ]
    )
    assert lora.train_action == "lora"
    assert status.train_action == "status"
    assert resume.train_action == "resume"
    assert export.train_action == "export"
    assert cleanup.train_action == "cleanup"
    assert cancel.train_action == "cancel"
    assert serve.train_action == "serve"
    assert elastic_status.train_action == "elastic-status"
    assert serve.port == 8791


def test_heterogeneous_training_cli_parses_manifest_product_path(tmp_path) -> None:
    create = parse_args(
        [
            "train",
            "create",
            str(tmp_path / "job"),
            "--heterogeneous",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--hf-token-env",
            "PRIVATE_HF_TOKEN",
        ]
    )
    serve = parse_args(
        ["train", "serve", "--elastic-job", str(tmp_path / "job")]
    )

    assert create.heterogeneous is True
    assert create.manifest.endswith("manifest.json")
    assert create.hf_token_env == "PRIVATE_HF_TOKEN"
    assert serve.elastic_job.endswith("job")


def test_heterogeneous_tpu_training_cli_has_explicit_pinned_path(tmp_path) -> None:
    create = parse_args(
        [
            "train",
            "create",
            str(tmp_path / "job"),
            "--heterogeneous",
            "--tpu",
            "--model",
            "Qwen/Qwen2.5-7B",
        ]
    )

    assert create.heterogeneous is True
    assert create.tpu is True
    assert create.model == "Qwen/Qwen2.5-7B"

    with pytest.raises(SystemExit, match="--tpu currently requires"):
        parse_args(
            [
                "train",
                "create",
                str(tmp_path / "invalid"),
                "--tpu",
                "--model",
                "Qwen/Qwen2.5-1.5B",
            ]
        )


def test_heterogeneous_training_cli_executes_owner_lifecycle(tmp_path, capsys) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest, config, _model = tiny_source(source)
    manifest_path = tmp_path / "manifest.json"
    config_path = tmp_path / "config.json"
    tokenized_path = tmp_path / "tokenized.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    tokenized_path.write_text(
        json.dumps(
            {
                "schema": "crowdtensor_heterogeneous_tokenized_private_v1",
                "training_manifest_hash": manifest["content_hash"],
                "model_id": manifest["model"]["model_id"],
                "model_revision": manifest["model"]["model_revision"],
                "sequence_length": manifest["training"]["sequence_length"],
                "train": [
                    [1, 2, 3, 4, 5, 6],
                    [6, 5, 4, 3, 2, 1],
                ],
                "validation": [[1, 2, 3, 4, 5, 6]],
            }
        ),
        encoding="utf-8",
    )
    job = tmp_path / "job"

    def invoke(arguments):
        with pytest.raises(SystemExit) as raised:
            cli.main([*arguments, "--json"])
        assert raised.value.code == 0
        return json.loads(capsys.readouterr().out)

    created = invoke(
        [
            "train",
            "create",
            str(job),
            "--heterogeneous",
            "--manifest",
            str(manifest_path),
            "--config",
            str(config_path),
            "--tokenized-payload",
            str(tokenized_path),
        ]
    )
    status = invoke(["train", "status", str(job)])
    resumed = invoke(["train", "resume", str(job)])
    invited = invoke(
        [
            "train",
            "invite",
            str(job),
            "--coordinator",
            "https://private.invalid",
            "--output-file",
            str(tmp_path / "private-invite.json"),
        ]
    )
    cancelled = invoke(["train", "cancel", str(job)])
    cleaned = invoke(["train", "cleanup", str(job)])

    assert created["runtime"]["heterogeneous_scheduler_enabled"] is True
    assert status["overall_state"] == "waiting_for_miners"
    assert resumed["resume_not_required"] is True
    assert invited["invite_file_written"] is True
    assert cancelled["overall_state"] == "cancelled"
    assert cleaned["overall_state"] == "cleaned"
    assert cleaned["tensor_transport_cleanup"]["all_messages_removed"] is True


def test_elastic_training_cli_status_reads_public_persistent_state(tmp_path, capsys) -> None:
    state = tmp_path / "elastic.sqlite3"
    ElasticTrainingRuntime(state, run_id="elastic-cli", target_steps=8)
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "train",
                "elastic-status",
                "--state",
                str(state),
                "--run-id",
                "elastic-cli",
                "--json",
            ]
        )
    assert raised.value.code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["runtime_state"] == "paused_waiting_for_miners"
    assert report["committed_step"] == 0
    assert report["live_miner_count"] == 0
    assert report["session_tokens_public"] is False


def test_missing_training_status_is_public_safe_and_resumable(tmp_path) -> None:
    status = training_status(tmp_path / "missing-job")
    assert status["overall_state"] == "not_found"
    assert status["blockers"] == ["training_job_status_missing"]
    assert status["next_resume_command"].startswith("crowdtensor train resume ")
    assert status["private_paths_public"] is False


def test_training_cli_failure_prints_safe_blocker_and_resume(tmp_path, capsys) -> None:
    job = tmp_path / "missing-job"
    with pytest.raises(SystemExit) as raised:
        cli.main(["train", "export", str(job), "--json"])
    assert raised.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["blockers"] == ["training_export_failed:FileNotFoundError"]
    assert payload["failure_detail_public"] is False
    assert payload["next_resume_command"] == f"crowdtensor train resume {job}"


def test_cuda_training_cli_parses_backend_and_private_runtime_inputs(tmp_path) -> None:
    lora = parse_args(
        [
            "train",
            "lora",
            "--backend",
            "cuda",
            "--output-dir",
            str(tmp_path),
            "--kaggle-token-file",
            "/private/token",
        ]
    )
    resume = parse_args(["train", "resume", str(tmp_path), "--backend", "cuda"])
    cleanup = parse_args(
        [
            "train",
            "cleanup",
            str(tmp_path),
            "--backend",
            "cuda",
            "--kaggle-token-file",
            "/private/token",
        ]
    )
    assert lora.backend == "cuda"
    assert lora.kaggle_token_file == "/private/token"
    assert lora.allocation_timeout_seconds == 1800
    assert resume.backend == "cuda"
    assert cleanup.kaggle_token_file == "/private/token"


def test_qwen15b_four_gpu_training_cli_parses_exact_alpha_contract(tmp_path) -> None:
    args = parse_args(
        [
            "train",
            "lora",
            "--backend",
            "cuda",
            "--model",
            "Qwen/Qwen2.5-1.5B",
            "--topology",
            "kaggle-2x-t4x2",
            "--steps",
            "8",
            "--output-dir",
            str(tmp_path),
            "--kaggle-token-file",
            "/private/accounts",
            "--kaggle-raw-token-file",
            "/private/dedicated",
        ]
    )
    assert args.model == "Qwen/Qwen2.5-1.5B"
    assert args.topology == "kaggle-2x-t4x2"
    assert args.steps == 8
    assert args.kaggle_raw_token_file == "/private/dedicated"


def test_qwen15b_beta_status_watch_contract(tmp_path) -> None:
    args = parse_args(
        [
            "train",
            "status",
            str(tmp_path),
            "--watch",
            "--watch-interval-seconds",
            "0.1",
            "--watch-timeout-seconds",
            "30",
        ]
    )
    assert args.watch is True
    assert args.watch_interval_seconds == 0.1
    assert args.watch_timeout_seconds == 30


def test_qwen15b_missing_private_tokens_writes_distinct_resumable_status(tmp_path, capsys) -> None:
    job = tmp_path / "qwen15b-job"
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "train",
                "lora",
                "--backend",
                "cuda",
                "--model",
                "Qwen/Qwen2.5-1.5B",
                "--topology",
                "kaggle-2x-t4x2",
                "--steps",
                "8",
                "--output-dir",
                str(job),
                "--json",
            ]
        )
    assert raised.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "Qwen/Qwen2.5-1.5B"
    assert payload["topology"] == "kaggle-2x-t4x2"
    assert payload["blockers"] == ["private_kaggle_token_input_required"]
    assert (job / "training_qwen15b_status.json").is_file()
    assert not (job / "training_cuda_status.json").exists()
    with pytest.raises(SystemExit) as status_exit:
        cli.main(["train", "status", str(job), "--json"])
    assert status_exit.value.code == 1
    status = json.loads(capsys.readouterr().out)
    assert status["current_phase"] == "account_preflight"


def test_cuda_training_cli_missing_private_token_writes_safe_resumable_status(tmp_path, capsys) -> None:
    job = tmp_path / "cuda-job"
    with pytest.raises(SystemExit) as raised:
        cli.main(["train", "lora", "--backend", "cuda", "--output-dir", str(job), "--json"])
    assert raised.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "cuda"
    assert payload["blockers"] == ["private_kaggle_token_input_required"]
    assert payload["resume_private_inputs"]["credential_values_public"] is False
    assert "/private" not in json.dumps(payload)
    stored = json.loads((job / "training_cuda_status.json").read_text(encoding="utf-8"))
    assert stored["overall_state"] == "blocked"


def test_cuda_status_and_cleanup_auto_detect_cuda_job(tmp_path, capsys) -> None:
    job = tmp_path / "cuda-job"
    with pytest.raises(SystemExit):
        cli.main(["train", "lora", "--backend", "cuda", "--output-dir", str(job), "--json"])
    capsys.readouterr()
    with pytest.raises(SystemExit) as status_exit:
        cli.main(["train", "status", str(job), "--json"])
    assert status_exit.value.code == 1
    status = json.loads(capsys.readouterr().out)
    assert status["backend"] == "cuda"
    with pytest.raises(SystemExit) as cleanup_exit:
        cli.main(["train", "cleanup", str(job), "--json"])
    assert cleanup_exit.value.code == 0
    cleanup = json.loads(capsys.readouterr().out)
    assert cleanup["temporary_kaggle_kernels_deleted"] is True


def test_cuda_cleanup_does_not_overclaim_orphan_kernel_without_private_token(tmp_path) -> None:
    job = tmp_path / "cuda-job"
    state_path = job / "attempts" / "single-1" / ".private-cleanup" / "active_resources.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "schema": "crowdtensor_cuda_training_private_cleanup_resources_v1",
                "provider": "kaggle",
                "kernel_refs": ["private-owner/private-kernel"],
                "credentials_embedded": False,
            }
        ),
        encoding="utf-8",
    )
    report = cleanup_cuda_training_job(job)
    assert report["ok"] is False
    assert report["temporary_kaggle_kernels_deleted"] is False
    assert report["live_resources_left_running"] is True
    assert report["blockers"] == ["cuda_training_cleanup_private_kaggle_token_required"]
    assert state_path.is_file()
    assert "private-owner" not in json.dumps(report, sort_keys=True)


def test_cuda_cleanup_uses_private_recovery_ledger_and_preserves_checkpoint(
    tmp_path, monkeypatch
) -> None:
    job = tmp_path / "cuda-job"
    state_path = job / "attempts" / "two-node-1" / ".private-cleanup" / "active_resources.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "schema": "crowdtensor_cuda_training_private_cleanup_resources_v1",
                "provider": "kaggle",
                "kernel_refs": ["private-owner/stage0", "private-owner/stage1"],
                "credentials_embedded": False,
            }
        ),
        encoding="utf-8",
    )
    private_runtime = job / "attempts" / "two-node-1" / ".private-runtime"
    private_runtime.mkdir(parents=True)
    checkpoint = job / "attempts" / "two-node-1" / "checkpoints" / "stage0_checkpoint_bundle.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    token_file = tmp_path / "private-token"
    token_file.write_text("private", encoding="utf-8")
    monkeypatch.setattr(
        training_cuda_job,
        "_delete_private_kaggle_refs",
        lambda refs, **_kwargs: (len(set(refs)), len(set(refs))),
    )
    report = cleanup_cuda_training_job(job, kaggle_token_file=str(token_file))
    assert report["ok"] is True
    assert report["temporary_kaggle_kernels_deleted"] is True
    assert report["kaggle_delete_verified_count"] == 2
    assert report["checkpoint_bundle_count"] == 1
    assert checkpoint.is_file()
    assert not state_path.exists()
    assert not private_runtime.exists()
