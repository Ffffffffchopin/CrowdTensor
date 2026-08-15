from __future__ import annotations

import argparse
import json

import pytest

from crowdtensor.core import cli
from crowdtensor.core.execution import PROVIDER_SNAPSHOT_SCHEMA
from crowdtensor.core import (
    ProviderSnapshot,
    ResourceAvailability,
    stable_hash,
)


def _args(tmp_path) -> list[str]:
    return [
        "init",
        str(tmp_path / "project"),
        "--model",
        "org/model",
        "--model-revision",
        "model-revision",
        "--dataset",
        "org/data",
        "--dataset-revision",
        "data-revision",
        "--model-adapter",
        "qwen2_lora_v1",
        "--target-steps",
        "4",
    ]


def _parse(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="train_action", required=True)
    cli.add_training_v2_parsers(subparsers)
    args = parser.parse_args(arguments)
    cli.validate_training_v2_args(args)
    return args


def test_v2_commands_parse_and_validate_without_framework_dependencies(tmp_path) -> None:
    initialized = _parse(_args(tmp_path))
    inspected = _parse(["inspect", str(tmp_path / "project")])
    assert initialized.train_action == "init"
    assert initialized.mode == "elastic-delta"
    assert inspected.train_action == "inspect"
    with pytest.raises(SystemExit, match="target-steps"):
        _parse([*_args(tmp_path), "--target-steps", "0"])

    operated = _parse(
        [
            "run",
            str(tmp_path / "operator"),
            "--campaign-dir",
            str(tmp_path / "campaign"),
            "--prepare-only",
        ]
    )
    contributed = _parse(
        [
            "join",
            str(tmp_path / "contributor"),
            "--coordinator-url",
            "https://training.example.test",
            "--code",
            "one-time-code",
            "--max-work-units",
            "2",
        ]
    )
    assert operated.prepare_only is True
    assert operated.work_unit_steps == 0
    assert operated.max_work_units == 1
    assert operated.execution_timeout_seconds == 3600.0
    assert contributed.max_work_units == 2


def test_v2_cli_surfaces_safe_stable_execution_error(tmp_path, capsys) -> None:
    workspace = tmp_path / "stable"
    arguments = [
        "init",
        str(workspace),
        "--model",
        "org/model",
        "--model-revision",
        "model-revision",
        "--dataset",
        "org/data",
        "--dataset-revision",
        "data-revision",
        "--model-adapter",
        "qwen2_lora_v1",
        "--mode",
        "stable-sharded",
        "--target-steps",
        "4",
        "--json",
    ]
    with pytest.raises(SystemExit) as initialized:
        cli.main(arguments)
    assert initialized.value.code == 0
    capsys.readouterr()

    from crowdtensor.cli import main as top_level_main

    with pytest.raises(SystemExit) as run:
        top_level_main(
            [
                "train",
                "run",
                str(workspace),
                "--work-unit-steps",
                "2",
                "--max-work-units",
                "2",
                "--execution-timeout-seconds",
                "120",
                "--json",
            ]
        )
    assert run.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["blocker"] == "stable_execution_plan_required"
    assert report["private_paths_public"] is False


def test_v2_cli_initializes_and_inspects_idempotently(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as first:
        cli.main([*_args(tmp_path), "--json"])
    assert first.value.code == 0
    created = json.loads(capsys.readouterr().out)
    assert created["command_ok"] is True
    assert created["created"] is True
    assert created["status"]["mode"] == "elastic_delta"

    with pytest.raises(SystemExit) as second:
        cli.main([*_args(tmp_path), "--json"])
    assert second.value.code == 0
    repeat = json.loads(capsys.readouterr().out)
    assert repeat["created"] is False
    assert repeat["status"]["project_hash"] == created["status"]["project_hash"]

    with pytest.raises(SystemExit) as inspected:
        cli.main(["inspect", str(tmp_path / "project"), "--json"])
    assert inspected.value.code == 0
    status = json.loads(capsys.readouterr().out)
    assert status["execution_state"] == "initialized"
    assert str(tmp_path) not in json.dumps(status)


def test_v2_cli_reports_public_safe_conflicts(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as initialized:
        cli.main([*_args(tmp_path), "--json"])
    assert initialized.value.code == 0
    capsys.readouterr()
    changed = _args(tmp_path)
    changed[changed.index("model-revision")] = "different-revision"
    with pytest.raises(SystemExit) as conflict:
        cli.main([*changed, "--json"])
    assert conflict.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["command_ok"] is False
    assert report["blocker"] == "workspace_init_failed:WorkspaceError"
    assert report["private_paths_public"] is False


def test_v2_cli_surfaces_safe_volunteer_error_codes(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as result:
        cli.main(
            [
                "join",
                str(tmp_path / "contributor"),
                "--coordinator-url",
                "http://training.example.test",
                "--code",
                "unused-code",
                "--json",
            ]
        )
    assert result.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["blocker"] == "volunteer_session_public_https_required"
    assert report["private_paths_public"] is False
    assert report["credential_values_public"] is False
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)


def test_v2_cli_lists_backends_without_loading_frameworks(capsys) -> None:
    with pytest.raises(SystemExit) as result:
        cli.main(["backends", "--json"])
    assert result.value.code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "crowdtensor_training_backend_registry_v2"
    assert {item["backend_id"] for item in report["backends"]} == {
        "accelerate_fsdp2",
        "volunteer_peft",
    }


def test_v2_cli_plan_routes_workspace_and_keeps_runtime_probe_explicit(tmp_path, capsys) -> None:
    workspace = tmp_path / "stable"
    with pytest.raises(SystemExit) as initialized:
        cli.main(
            [
                "init",
                str(workspace),
                "--model",
                "org/model",
                "--model-revision",
                "model-revision",
                "--dataset",
                "org/data",
                "--dataset-revision",
                "data-revision",
                "--model-adapter",
                "qwen2_lora_v1",
                "--mode",
                "stable-sharded",
                "--target-steps",
                "4",
                "--json",
            ]
        )
    assert initialized.value.code == 0
    capsys.readouterr()
    resource = ProviderSnapshot(
        provider_id="fixture",
        resource_id="fixture.gpu-0",
        machine_id_hash=stable_hash("machine"),
        device_type="cuda",
        device_count=2,
        total_memory_bytes=24 * 1024**3,
        free_memory_bytes=20 * 1024**3,
        availability=ResourceAvailability.STABLE_WINDOW,
        source_hash=stable_hash("capability"),
        capabilities=("cuda", "distributed_collective", "stable_sharded"),
        supported_dtypes=("bfloat16", "float16"),
        stable_group_id="stable-window",
    )
    capability = tmp_path / "provider.json"
    capability.write_text(
        json.dumps(resource.to_dict()), encoding="utf-8"
    )
    with pytest.raises(SystemExit) as planned:
        cli.main(
            [
                "plan",
                str(workspace),
                "--capability",
                str(capability),
                "--trainer-entrypoint",
                "train.py",
                "--trainer-contract-verified",
                "--json",
            ]
        )
    assert planned.value.code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["command_ok"] is True
    assert report["runtime_probe_performed"] is False
    assert report["execution_ready"] is False
    assert "runtime_unavailable" in report["plan"]["blockers"]
    assert str(tmp_path) not in json.dumps(report)


def test_v2_cli_lifecycle_routes_through_top_level_compatibility_cli(tmp_path, capsys) -> None:
    workspace = tmp_path / "project"
    with pytest.raises(SystemExit) as initialized:
        from crowdtensor.cli import main as top_level_main

        top_level_main(["train", *_args(tmp_path), "--json"])
    assert initialized.value.code == 0
    created = json.loads(capsys.readouterr().out)
    assert created["status"]["lifecycle_state"] == "initialized"
    assert "peft_lora_v1" in created["status"]["project"]["optimization_plugins"]

    with pytest.raises(SystemExit) as paused:
        top_level_main(
            [
                "train",
                "pause",
                str(workspace),
                "--reason",
                "owner_window_closed",
                "--json",
            ]
        )
    assert paused.value.code == 0
    pause_report = json.loads(capsys.readouterr().out)
    assert pause_report["state"] == "paused"
    assert pause_report["pause_reason"] == "owner_window_closed"

    with pytest.raises(SystemExit) as status:
        top_level_main(["train", "status", str(workspace), "--json"])
    assert status.value.code == 0
    status_report = json.loads(capsys.readouterr().out)
    assert status_report["lifecycle_state"] == "paused"

    with pytest.raises(SystemExit) as resumed:
        top_level_main(["train", "resume", str(workspace), "--json"])
    assert resumed.value.code == 0
    resume_report = json.loads(capsys.readouterr().out)
    assert resume_report["state"] == "initialized"

    with pytest.raises(SystemExit) as joined:
        top_level_main(["train", "join", str(workspace), "--json"])
    assert joined.value.code == 1
    join_report = json.loads(capsys.readouterr().out)
    assert join_report["blockers"] == ["v2_controller_join_pending"]

    with pytest.raises(SystemExit) as run:
        top_level_main(["train", "run", str(workspace), "--json"])
    assert run.value.code == 1
    run_report = json.loads(capsys.readouterr().out)
    assert run_report["blockers"] == ["execution_plan_required"]

    export_dir = tmp_path / "public-export"
    with pytest.raises(SystemExit) as exported:
        top_level_main(
            [
                "train",
                "export",
                str(workspace),
                "--output-dir",
                str(export_dir),
                "--json",
            ]
        )
    assert exported.value.code == 0
    export_report = json.loads(capsys.readouterr().out)
    assert export_report["weights_exported"] is False
    assert (export_dir / "status.json").is_file()
