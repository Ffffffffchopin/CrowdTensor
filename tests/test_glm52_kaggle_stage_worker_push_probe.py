from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from scripts import glm52_kaggle_stage_worker_push_probe as probe
from scripts import glm52_kaggle_stage_worker_push_probe_check as check
from scripts import glm52_kaggle_stage_runtime_check as runtime_check


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_stage_push_"))


def _package_report(base: Path) -> dict:
    packages = []
    for index, provider in enumerate(probe.REQUIRED_PROVIDERS):
        package_dir = base / f"pkg-{provider}"
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "kernel-metadata.json").write_text(
            json.dumps({"id": f"tester/ct-glm52-{index}-{provider}"}) + "\n",
            encoding="utf-8",
        )
        (package_dir / "kernel.py").write_text(
            "# test kernel\n"
            f"{probe.PRIVATE_RUNTIME_ENV_INLINE_SENTINEL}"
            "print('ok')\n",
            encoding="utf-8",
        )
        packages.append({
            "provider": provider,
            "stage_id": index,
            "package_dir": str(package_dir),
            "private_kernel": True,
            "public_artifact_safe": True,
        })
    return {
        "schema": "glm52_kaggle_stage_worker_package_v1",
        "ok": True,
        "glm52_stage_worker_package_ready": True,
        "packages": packages,
        "public_artifact_safe": True,
    }


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _stage_runtime_report(provider: str, stage_id: int) -> dict:
    return {
        "schema": runtime_check.STAGE_SCHEMA,
        "ok": True,
        "public_artifact_safe": True,
        "model_id": runtime_check.MODEL_ID,
        "compatible_weight_repo": runtime_check.COMPATIBLE_WEIGHT_REPO,
        "provider": provider,
        "stage_id": stage_id,
        "stage_layer_range": [stage_id * 2, stage_id * 2 + 1],
        "coordinator_request_id_hash": _hash("b"),
        "stage_execution_verified": True,
        "stage_output_hash": _hash("a"),
        "weight_tensor_values_loaded": True,
        "weight_value_byte_count": 16,
        "weight_value_sha256": _hash("c"),
        "weight_tensor_values_public": False,
        "live_run_performed": True,
        "fallback_model_used": False,
        "queue_only_evidence": False,
        "metadata_only": False,
        "stage_smoke_only": False,
        "activation_public": False,
        "kv_cache_public": False,
        "safety": {
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _tpu_watch(*, ready: bool, stage_report_path: Path | None = None) -> dict:
    status = "KernelWorkerStatus.COMPLETE" if ready else "KernelWorkerStatus.QUEUED"
    return {
        "schema": "glm52_mcp_tpu_stage_runtime_watch_v1",
        "ref": "tester/ct-glm52-tpu-stage",
        "last_status": status,
        "last_status_class": "complete" if ready else "queued",
        "stage_runtime_report_verified": ready,
        "tpu_stage_runtime_ready": ready,
        "same_request_decode_verified": False,
        "stage_runtime_report": {
            "path": str(stage_report_path or ""),
            "present": bool(stage_report_path),
            "sha256": _hash("d") if stage_report_path else "",
        },
        "observations": [{"attempt": 1, "status": status, "ok": True}],
        "blockers": [] if ready else [
            "glm52_mcp_tpu_stage_runtime_not_ready",
            "glm52_mcp_tpu_stage_runtime_scheduler_queued",
        ],
        "public_artifact_safe": True,
    }


def test_preflight_builds_push_commands_without_live_overclaim() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "preflight",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
    ])

    report = probe.build_report(args)

    assert report["live_run_performed"] is False
    assert report["stage_runtime_reports_collected"] == 0
    assert report["stage_runtime_reports_verified"] == 0
    assert {push["provider"] for push in report["pushes"]} == set(probe.REQUIRED_PROVIDERS)
    assert all(push["pushed"] is False for push in report["pushes"])
    push_by_provider = {push["provider"]: push for push in report["pushes"]}
    assert "--accelerator NvidiaTeslaT4" in push_by_provider["kaggle_cuda"]["push_command"]
    assert "--accelerator tpuV5e8" in push_by_provider["kaggle_jax_tpu"]["push_command"]
    assert "--accelerator" not in push_by_provider["kaggle_cpu"]["push_command"]
    assert check.validate_report(report) == []
    assert "live_run_not_performed" in check.validate_report(report, require_live=True)


def test_raw_token_file_builds_private_kaggle_env() -> None:
    out = _tmp_dir()
    token_file = _write(out / "kaggle.json", {"username": "owner", "key": "secret-key"})
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "preflight",
        "--stage-worker-package-report",
        str(package_path),
        "--raw-token-file",
        str(token_file),
    ])

    env, config_dir = probe.kaggle_env_for_token_section(args)
    try:
        assert env is not None
        assert env["KAGGLE_USERNAME"] == "owner"
        assert env["KAGGLE_KEY"] == "secret-key"
        assert "KAGGLE_CONFIG_DIR" in env
    finally:
        if config_dir is not None:
            config_dir.cleanup()


def test_raw_key_only_token_file_uses_username_hint() -> None:
    token_file = _tmp_dir() / "key.txt"
    token_file.write_text("unused", encoding="utf-8")
    token_env = probe.parse_raw_token_file(token_file, username_hint="owner")
    assert token_env["KAGGLE_USERNAME"] == "owner"
    assert token_env["KAGGLE_KEY"] == "unused"
    assert token_env["KAGGLE_API_TOKEN"] == "unused"


def test_multi_account_raw_token_file_selects_requested_section() -> None:
    token_file = _tmp_dir() / "accounts.md"
    token_file.write_text(
        "\n".join(
            [
                "# first-owner",
                "export KAGGLE_API_TOKEN=private-first",
                "export MY_KAGGLE_TOKEN=private-first",
                "# second-owner",
                "export KAGGLE_API_TOKEN=private-second",
                "export MY_KAGGLE_TOKEN=private-second",
            ]
        ),
        encoding="utf-8",
    )

    token_env = probe.parse_raw_token_file(
        token_file, username_hint="first-owner"
    )

    assert token_env == {
        "KAGGLE_API_TOKEN": "private-first",
        "KAGGLE_USERNAME": "first-owner",
    }


def test_multi_account_raw_token_file_requires_matching_hint() -> None:
    token_file = _tmp_dir() / "accounts.md"
    token_file.write_text(
        "# first\nexport KAGGLE_API_TOKEN=private-first\n"
        "# second\nexport KAGGLE_API_TOKEN=private-second\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="kaggle_raw_token_section_not_found"):
        probe.parse_raw_token_file(token_file, username_hint="missing")


def test_preflight_push_command_can_include_kernel_timeout() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "preflight",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_cuda",
        "--kernel-timeout-seconds",
        "3600",
    ])

    report = probe.build_report(args)
    push = report["pushes"][0]

    assert " -t 3600 " in f" {push['push_command']} "
    assert "--accelerator NvidiaTeslaT4" in push["push_command"]
    assert check.validate_report(report) == []


def test_preflight_can_filter_by_stage_ids() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "preflight",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_cpu",
        "--stage-ids",
        "2",
    ])

    report = probe.build_report(args)

    assert [push["stage_id"] for push in report["pushes"]] == [2]
    assert [push["provider"] for push in report["pushes"]] == ["kaggle_cpu"]
    assert "glm52_stage_worker_push_provider_missing:kaggle_cuda" in report["blockers"]
    assert "glm52_stage_worker_push_provider_missing:kaggle_jax_tpu" in report["blockers"]
    assert check.validate_report(report) == []
    assert "required_provider_pushes_missing" in check.validate_report(report, require_live=True)


def test_preflight_can_filter_grouped_package_by_any_stage_id() -> None:
    out = _tmp_dir()
    package_report = _package_report(out)
    cpu_entry = next(entry for entry in package_report["packages"] if entry["provider"] == "kaggle_cpu")
    cpu_entry["stage_id"] = 10
    cpu_entry["stage_ids"] = [10, 11, 12]
    cpu_entry["grouped_stage_worker"] = True
    package_path = _write(out / "package.json", package_report)
    args = probe.parse_args([
        "--mode",
        "preflight",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_cpu",
        "--stage-ids",
        "11",
    ])

    report = probe.build_report(args)

    assert len(report["pushes"]) == 1
    assert report["pushes"][0]["stage_id"] == 10
    assert report["pushes"][0]["stage_ids"] == [10, 11, 12]
    assert check.validate_report(report) == []


def test_live_mode_prefers_manifest_kernel_ref_over_metadata_slug() -> None:
    out = _tmp_dir()
    package_report = _package_report(out)
    package_report["packages"] = [item for item in package_report["packages"] if item["provider"] == "kaggle_cuda"]
    entry = package_report["packages"][0]
    entry["kernel_ref"] = "tester/ct-glm52-manifest-ref"
    metadata_path = Path(entry["package_dir"]) / "kernel-metadata.json"
    metadata_path.write_text(json.dumps({"id": "ct-glm52-manifest-ref"}) + "\n", encoding="utf-8")
    package_path = _write(out / "package.json", package_report)
    args = probe.parse_args([
        "--mode",
        "live",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_cuda",
        "--wait-seconds",
        "0",
    ])
    commands = []

    def fake_runner(command, **kwargs):
        commands.append(command)
        joined = " ".join(command)
        if "status" in joined:
            return subprocess.CompletedProcess(command, 0, stdout="status RUNNING", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)

    assert report["pushes"][0]["kernel_ref"] == "tester/ct-glm52-manifest-ref"
    assert any(command[:3] == ["kaggle", "kernels", "status"] and command[3] == "tester/ct-glm52-manifest-ref" for command in commands)
    assert not any(command[:3] == ["kaggle", "kernels", "status"] and command[3] == "ct-glm52-manifest-ref" for command in commands)


def test_live_mode_records_push_status_output_and_cleanup_with_fake_runner() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "live",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--wait-seconds",
        "0",
    ])

    def fake_runner(command, **kwargs):
        joined = " ".join(command)
        if "status" in joined:
            return subprocess.CompletedProcess(command, 0, stdout="status COMPLETE", stderr="")
        if command[:3] == ["kaggle", "kernels", "output"]:
            output_dir = Path(command[command.index("-p") + 1])
            _prefix, stage_id, provider = output_dir.name.split("-", 2)
            _write(
                output_dir / "glm52_kaggle_stage_runtime_report.json",
                _stage_runtime_report(provider, int(stage_id)),
            )
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)

    assert report["live_run_performed"] is True
    assert report["stage_runtime_reports_collected"] == 3
    assert report["stage_runtime_reports_verified"] == 3
    assert all(push["stage_report_present"] is True for push in report["pushes"])
    assert all(push["stage_runtime_verified"] is True for push in report["pushes"])
    assert all(push["cleanup_performed"] is True for push in report["pushes"])
    assert check.validate_report(report, require_live=True) == []


def test_live_mode_treats_kaggle_push_error_stdout_as_not_pushed() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "live",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_cuda",
        "--wait-seconds",
        "0",
    ])
    commands = []

    def fake_runner(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Kernel push error: Maximum batch GPU session count of 2 reached.\n",
            stderr="",
        )

    report = probe.build_report(args, runner=fake_runner)
    push = report["pushes"][0]

    assert push["provider"] == "kaggle_cuda"
    assert push["pushed"] is False
    assert push["push_error_blocker"] == "kaggle_gpu_batch_session_limit_reached"
    assert "kaggle_gpu_batch_session_limit_reached" in report["blockers"]
    assert "glm52_stage_worker_push_failed:kaggle_cuda" in report["blockers"]
    assert not any(command[:3] == ["kaggle", "kernels", "status"] for command in commands)
    assert not any(command[:3] == ["kaggle", "kernels", "delete"] for command in commands)
    assert check.validate_report(report) == []
    assert "required_provider_pushes_missing" in check.validate_report(report, require_live=True)


def test_live_mode_records_push_timeout_as_specific_blocker() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "live",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_cuda",
        "--wait-seconds",
        "0",
    ])

    def fake_runner(command, **kwargs):
        if command[:3] == ["kaggle", "kernels", "push"]:
            raise subprocess.TimeoutExpired(command, timeout=kwargs.get("timeout"))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)
    push = report["pushes"][0]

    assert push["pushed"] is False
    assert push["push_error_blocker"] == "kaggle_kernel_push_timeout:kaggle_cuda"
    assert "kaggle_kernel_push_timeout:kaggle_cuda" in report["blockers"]
    assert "glm52_stage_worker_push_failed:kaggle_cuda" in report["blockers"]
    assert check.validate_report(report) == []


def test_live_mode_records_status_wait_timeout_blocker() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "live",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_cuda",
        "--wait-seconds",
        "0",
        "--poll-interval-seconds",
        "0.1",
    ])

    def fake_runner(command, **kwargs):
        if command[:3] == ["kaggle", "kernels", "status"]:
            return subprocess.CompletedProcess(command, 0, stdout="status RUNNING", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)
    push = report["pushes"][0]

    assert push["pushed"] is True
    assert push["terminal_status"] == "RUNNING"
    assert "kaggle_kernel_wait_timeout:kaggle_cuda" in report["blockers"]
    assert "glm52_stage_worker_live_reports_missing" in report["blockers"]
    assert check.validate_report(report) == []


def test_live_mode_records_output_empty_response_blocker() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "live",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_cpu",
        "--wait-seconds",
        "0",
    ])

    def fake_runner(command, **kwargs):
        if command[:3] == ["kaggle", "kernels", "status"]:
            return subprocess.CompletedProcess(command, 0, stdout="status COMPLETE", stderr="")
        if command[:3] == ["kaggle", "kernels", "output"]:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)
    push = report["pushes"][0]

    assert push["terminal_status"] == "COMPLETE"
    assert push["output_collected"] is False
    assert "kaggle_kernel_output_empty_response:kaggle_cpu" in report["blockers"]
    assert "glm52_stage_worker_live_reports_missing" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_live_output_without_verified_stage_report() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "live",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--wait-seconds",
        "0",
    ])

    def fake_runner(command, **kwargs):
        joined = " ".join(command)
        if "status" in joined:
            return subprocess.CompletedProcess(command, 0, stdout="status COMPLETE", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)
    errors = check.validate_report(report, require_live=True)

    assert report["stage_runtime_reports_collected"] == 0
    assert report["stage_runtime_reports_verified"] == 0
    assert "stage_report_missing:kaggle_cuda" in errors
    assert "stage_report_not_verified:kaggle_cuda" in errors
    assert "stage_runtime_reports_not_verified" in errors


def test_checker_accepts_partial_live_stage_report_as_blocker_evidence() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "live",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_cpu",
        "--wait-seconds",
        "0",
    ])

    def fake_runner(command, **kwargs):
        joined = " ".join(command)
        if "status" in joined:
            return subprocess.CompletedProcess(command, 0, stdout="status COMPLETE", stderr="")
        if command[:3] == ["kaggle", "kernels", "output"]:
            output_dir = Path(command[command.index("-p") + 1])
            _write(
                output_dir / "glm52_kaggle_stage_runtime_report.json",
                _stage_runtime_report("kaggle_cpu", 2),
            )
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)

    assert report["stage_runtime_reports_collected"] == 1
    assert report["stage_runtime_reports_verified"] == 1
    assert check.validate_report(report) == []
    strict_errors = check.validate_report(report, require_live=True)
    assert "required_provider_pushes_missing" in strict_errors
    assert "stage_runtime_reports_not_collected" in strict_errors
    assert "stage_runtime_reports_not_verified" in strict_errors


def test_checker_rejects_preflight_push_overclaim() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args(["--stage-worker-package-report", str(package_path)])
    report = probe.build_report(args)
    report["pushes"][0]["pushed"] = True

    errors = check.validate_report(report)

    assert "preflight_push_overclaim:kaggle_cuda" in errors


def test_live_mode_can_use_token_section_without_public_token_leak() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    token_path = out / "tokens.md"
    token_path.write_text(
        "# tpuowner\n"
        "export KAGGLE_USERNAME='tpuowner'\n"
        "export KAGGLE_KEY='KGA_TEST_SECRET_VALUE'\n",
        encoding="utf-8",
    )
    args = probe.parse_args([
        "--mode",
        "live",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--wait-seconds",
        "0",
        "--token-file",
        str(token_path),
        "--token-section",
        "tpuowner",
    ])
    seen_envs = []

    def fake_runner(command, **kwargs):
        seen_envs.append(dict(kwargs.get("env") or {}))
        joined = " ".join(command)
        if "status" in joined:
            return subprocess.CompletedProcess(command, 0, stdout="status COMPLETE", stderr="")
        if command[:3] == ["kaggle", "kernels", "output"]:
            output_dir = Path(command[command.index("-p") + 1])
            _prefix, stage_id, provider = output_dir.name.split("-", 2)
            _write(
                output_dir / "glm52_kaggle_stage_runtime_report.json",
                _stage_runtime_report(provider, int(stage_id)),
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)
    encoded = json.dumps(report, sort_keys=True)

    assert seen_envs
    assert all(env.get("KAGGLE_USERNAME") == "tpuowner" for env in seen_envs)
    assert all(env.get("KAGGLE_KEY") == "KGA_TEST_SECRET_VALUE" for env in seen_envs)
    assert all(env.get("KAGGLE_CONFIG_DIR") for env in seen_envs)
    assert "KGA_TEST_SECRET_VALUE" not in encoded
    assert check.validate_report(report, require_live=True) == []


def test_live_mode_injects_coordinator_env_file_without_public_leak() -> None:
    out = _tmp_dir()
    package_report = _package_report(out)
    package_path = _write(out / "package.json", package_report)
    coordinator_token = "CT_COORDINATOR_TEST_SECRET"
    coordinator_url = "https://coordinator.example/private-session"
    token_file = out / "coordinator-token.txt"
    token_file.write_text(coordinator_token, encoding="utf-8")
    args = probe.parse_args([
        "--mode",
        "live",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--wait-seconds",
        "0",
        "--coordinator-url",
        coordinator_url,
        "--coordinator-token-file",
        str(token_file),
        "--coordinator-task-timeout-seconds",
        "7200",
        "--coordinator-poll-interval-seconds",
        "1",
        "--coordinator-stage-task-limit",
        "8",
        "--full-prefix-prefill-length",
        "1",
        "--full-prefix-dsa-mask-topk",
        "1",
        "--full-prefix-executed-expert-count",
        "2",
        "--full-prefix-top-k",
        "1",
        "--full-prefix-row-block-size",
        "512",
        "--full-prefix-max-tensor-bytes",
        "33554432",
        "--full-prefix-max-block-bytes",
        "16777216",
        "--cpu-group-stage-attempt-seconds",
        "2.5",
        "--cpu-group-stage-poll-seconds",
        "0.5",
    ])
    observed_private_payloads = []

    def fake_runner(command, **kwargs):
        if command[:3] == ["kaggle", "kernels", "push"]:
            package_dir = Path(command[command.index("-p") + 1])
            private_env = package_dir / probe.PRIVATE_RUNTIME_ENV_FILENAME
            assert private_env.is_file()
            payload = json.loads(private_env.read_text(encoding="utf-8"))
            observed_private_payloads.append(payload)
            kernel_text = (package_dir / "kernel.py").read_text(encoding="utf-8")
            assert coordinator_token in kernel_text
            assert coordinator_url in kernel_text
        joined = " ".join(command)
        if "status" in joined:
            return subprocess.CompletedProcess(command, 0, stdout="status COMPLETE", stderr="")
        if command[:3] == ["kaggle", "kernels", "output"]:
            output_dir = Path(command[command.index("-p") + 1])
            _prefix, stage_id, provider = output_dir.name.split("-", 2)
            _write(
                output_dir / "glm52_kaggle_stage_runtime_report.json",
                _stage_runtime_report(provider, int(stage_id)),
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)
    encoded = json.dumps(report, sort_keys=True)

    assert observed_private_payloads
    assert all(payload["CT_GLM52_COORDINATOR_TOKEN"] == coordinator_token for payload in observed_private_payloads)
    assert all(payload["CT_GLM52_COORDINATOR_URL"] == coordinator_url for payload in observed_private_payloads)
    assert all(payload["CT_GLM52_COORDINATOR_STAGE_TASK_LIMIT"] == "8" for payload in observed_private_payloads)
    assert all(payload["CT_GLM52_FULL_PREFIX_PREFILL_LENGTH"] == "1" for payload in observed_private_payloads)
    assert all(payload["CT_GLM52_FULL_PREFIX_DSA_MASK_TOPK"] == "1" for payload in observed_private_payloads)
    assert all(payload["CT_GLM52_FULL_PREFIX_EXECUTED_EXPERT_COUNT"] == "2" for payload in observed_private_payloads)
    assert all(payload["CT_GLM52_FULL_PREFIX_TOP_K"] == "1" for payload in observed_private_payloads)
    assert all(payload["CT_GLM52_FULL_PREFIX_ROW_BLOCK_SIZE"] == "512" for payload in observed_private_payloads)
    assert all(payload["CT_GLM52_FULL_PREFIX_MAX_TENSOR_BYTES"] == "33554432" for payload in observed_private_payloads)
    assert all(payload["CT_GLM52_FULL_PREFIX_MAX_BLOCK_BYTES"] == "16777216" for payload in observed_private_payloads)
    assert all(payload["CT_GLM52_CPU_GROUP_STAGE_ATTEMPT_SECONDS"] == "2.5" for payload in observed_private_payloads)
    assert all(payload["CT_GLM52_CPU_GROUP_STAGE_POLL_SECONDS"] == "0.5" for payload in observed_private_payloads)
    for entry in package_report["packages"]:
        assert not (Path(entry["package_dir"]) / probe.PRIVATE_RUNTIME_ENV_FILENAME).exists()
        kernel_text = (Path(entry["package_dir"]) / "kernel.py").read_text(encoding="utf-8")
        assert coordinator_token not in kernel_text
        assert coordinator_url not in kernel_text
        assert probe.PRIVATE_RUNTIME_ENV_INLINE_SENTINEL in kernel_text
    assert coordinator_token not in encoded
    assert coordinator_url not in encoded
    assert all(push["coordinator_private_runtime_env_uploaded"] is True for push in report["pushes"])
    assert all(push["coordinator_private_runtime_env_inlined"] is True for push in report["pushes"])
    assert all(push["coordinator_url_public"] is False for push in report["pushes"])
    assert all(push["coordinator_token_public"] is False for push in report["pushes"])
    assert all(push["private_runtime_env_local_removed"] is True for push in report["pushes"])
    assert all(push["private_runtime_env_kernel_restored"] is True for push in report["pushes"])
    assert check.validate_report(report, require_live=True) == []


def test_live_mode_injects_hf_token_private_env_without_public_leak(monkeypatch) -> None:
    out = _tmp_dir()
    package_report = _package_report(out)
    package_path = _write(out / "package.json", package_report)
    env_name = "CT_TEST_HF_TOKEN_ENV"
    hf_secret = "hf_stage_worker_test_secret"
    monkeypatch.setenv(env_name, hf_secret)
    args = probe.parse_args([
        "--mode",
        "live",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_cpu",
        "--wait-seconds",
        "0",
        "--hf-token-env",
        env_name,
    ])
    observed_private_payloads = []

    def fake_runner(command, **kwargs):
        if command[:3] == ["kaggle", "kernels", "push"]:
            package_dir = Path(command[command.index("-p") + 1])
            private_env = package_dir / probe.PRIVATE_RUNTIME_ENV_FILENAME
            assert private_env.is_file()
            payload = json.loads(private_env.read_text(encoding="utf-8"))
            observed_private_payloads.append(payload)
            kernel_text = (package_dir / "kernel.py").read_text(encoding="utf-8")
            assert hf_secret in kernel_text
        if command[:3] == ["kaggle", "kernels", "status"]:
            return subprocess.CompletedProcess(command, 0, stdout="status COMPLETE", stderr="")
        if command[:3] == ["kaggle", "kernels", "output"]:
            output_dir = Path(command[command.index("-p") + 1])
            _write(
                output_dir / "glm52_kaggle_stage_runtime_report.json",
                _stage_runtime_report("kaggle_cpu", 2),
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)
    encoded = json.dumps(report, sort_keys=True)
    push = report["pushes"][0]

    assert observed_private_payloads
    assert observed_private_payloads[0]["HF_TOKEN"] == hf_secret
    assert observed_private_payloads[0]["HUGGING_FACE_HUB_TOKEN"] == hf_secret
    assert push["hf_token_env_supported"] is True
    assert push["hf_token_env_configured"] is True
    assert push["hf_token_env_configured_count"] == 1
    assert push["hf_token_private_runtime_env_uploaded"] is True
    assert push["hf_token_public"] is False
    assert push["hf_token_env_name_hashes"][0].startswith("sha256:")
    assert push["coordinator_private_runtime_env_uploaded"] is True
    assert hf_secret not in encoded
    assert env_name not in encoded
    for entry in package_report["packages"]:
        assert not (Path(entry["package_dir"]) / probe.PRIVATE_RUNTIME_ENV_FILENAME).exists()
        assert hf_secret not in (Path(entry["package_dir"]) / "kernel.py").read_text(encoding="utf-8")
    assert check.validate_report(report) == []


def test_live_mode_can_retain_nonterminal_tpu_kernel() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "live",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_jax_tpu",
        "--wait-seconds",
        "0",
        "--retain-nonterminal-tpu",
    ])
    commands = []

    def fake_runner(command, **kwargs):
        commands.append(command)
        joined = " ".join(command)
        if "status" in joined:
            return subprocess.CompletedProcess(command, 0, stdout="status QUEUED", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)
    push = report["pushes"][0]

    assert push["provider"] == "kaggle_jax_tpu"
    assert push["terminal_status"] == "QUEUED"
    assert push["cleanup_performed"] is False
    assert push["steps"][-1]["skipped"] is True
    assert "kaggle_tpu_kernel_retained_for_queue" in report["blockers"]
    assert not any(command[:3] == ["kaggle", "kernels", "delete"] for command in commands)


def test_live_mode_can_retain_nonterminal_gpu_kernel() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "live",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_cuda",
        "--wait-seconds",
        "0",
        "--retain-nonterminal-gpu",
    ])
    commands = []

    def fake_runner(command, **kwargs):
        commands.append(command)
        joined = " ".join(command)
        if "status" in joined:
            return subprocess.CompletedProcess(command, 0, stdout="status RUNNING", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)
    push = report["pushes"][0]

    assert push["provider"] == "kaggle_cuda"
    assert push["terminal_status"] == "RUNNING"
    assert push["cleanup_performed"] is False
    assert push["steps"][-1]["skipped"] is True
    assert push["steps"][-1]["reason"] == "retain_nonterminal_gpu"
    assert "kaggle_gpu_kernel_retained_for_queue_or_run" in report["blockers"]
    assert not any(command[:3] == ["kaggle", "kernels", "delete"] for command in commands)
    assert check.validate_report(report) == []


def test_live_mode_can_retain_nonterminal_cpu_kernel() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "live",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_cpu",
        "--wait-seconds",
        "0",
        "--retain-nonterminal-cpu",
    ])
    commands = []

    def fake_runner(command, **kwargs):
        commands.append(command)
        joined = " ".join(command)
        if "status" in joined:
            return subprocess.CompletedProcess(command, 0, stdout="status RUNNING", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)
    push = report["pushes"][0]

    assert push["provider"] == "kaggle_cpu"
    assert push["terminal_status"] == "RUNNING"
    assert push["cleanup_performed"] is False
    assert push["steps"][-1]["skipped"] is True
    assert push["steps"][-1]["reason"] == "retain_nonterminal_cpu"
    assert "kaggle_cpu_kernel_retained_for_run" in report["blockers"]
    assert not any(command[:3] == ["kaggle", "kernels", "delete"] for command in commands)
    assert check.validate_report(report) == []


def test_collect_mode_retains_nonterminal_cpu_kernel_without_repush() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "collect",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_cpu",
        "--wait-seconds",
        "0",
    ])
    commands = []

    def fake_runner(command, **kwargs):
        commands.append(command)
        if command[:3] == ["kaggle", "kernels", "status"]:
            return subprocess.CompletedProcess(command, 0, stdout="status RUNNING", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)
    push = report["pushes"][0]

    assert report["mode"] == "collect"
    assert push["provider"] == "kaggle_cpu"
    assert push["existing_kernel_observed"] is True
    assert push["pushed"] is False
    assert push["terminal_status"] == "RUNNING"
    assert push["cleanup_performed"] is False
    assert push["steps"][-1]["reason"] == "retain_nonterminal_cpu"
    assert "kaggle_cpu_kernel_retained_for_run" in report["blockers"]
    assert not any(command[:3] == ["kaggle", "kernels", "push"] for command in commands)
    assert not any(command[:3] == ["kaggle", "kernels", "delete"] for command in commands)
    assert check.validate_report(report) == []


def test_collect_mode_downloads_complete_report_and_cleans_kernel() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    args = probe.parse_args([
        "--mode",
        "collect",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_cpu",
        "--wait-seconds",
        "0",
    ])
    commands = []

    def fake_runner(command, **kwargs):
        commands.append(command)
        if command[:3] == ["kaggle", "kernels", "status"]:
            return subprocess.CompletedProcess(command, 0, stdout="status COMPLETE", stderr="")
        if command[:3] == ["kaggle", "kernels", "output"]:
            output_dir = Path(command[command.index("-p") + 1])
            _write(
                output_dir / "glm52_kaggle_stage_runtime_report.json",
                _stage_runtime_report("kaggle_cpu", 2),
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)
    push = report["pushes"][0]

    assert report["mode"] == "collect"
    assert report["stage_runtime_reports_collected"] == 1
    assert report["stage_runtime_reports_verified"] == 1
    assert push["pushed"] is False
    assert push["output_collected"] is True
    assert push["stage_report_present"] is True
    assert push["stage_runtime_verified"] is True
    assert push["cleanup_performed"] is True
    assert not any(command[:3] == ["kaggle", "kernels", "push"] for command in commands)
    assert any(command[:3] == ["kaggle", "kernels", "delete"] for command in commands)
    assert check.validate_report(report) == []


def test_import_mode_aggregates_gpu_cpu_reports_and_queued_tpu_watch() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    gpu = _write(out / "gpu.json", _stage_runtime_report("kaggle_cuda", 0))
    cpu = _write(out / "cpu.json", _stage_runtime_report("kaggle_cpu", 2))
    tpu_watch = _write(out / "tpu-watch.json", _tpu_watch(ready=False))
    args = probe.parse_args([
        "--mode",
        "import",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--import-stage-report",
        str(gpu),
        "--import-stage-report",
        str(cpu),
        "--tpu-watch-report",
        str(tpu_watch),
        "--import-cleanup-verified",
    ])

    report = probe.build_report(args)
    errors = check.validate_report(report, require_live=True)

    assert report["mode"] == "import"
    assert report["stage_runtime_reports_collected"] == 2
    assert report["stage_runtime_reports_verified"] == 2
    assert "glm52_mcp_tpu_stage_runtime_scheduler_queued" in report["blockers"]
    assert "stage_runtime_reports_not_collected" in errors
    assert "stage_runtime_reports_not_verified" in errors
    assert check.validate_report(report) == []


def test_import_mode_accepts_all_three_verified_provider_reports_for_require_live() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    gpu = _write(out / "gpu.json", _stage_runtime_report("kaggle_cuda", 0))
    tpu = _write(out / "tpu.json", _stage_runtime_report("kaggle_jax_tpu", 1))
    cpu = _write(out / "cpu.json", _stage_runtime_report("kaggle_cpu", 2))
    tpu_watch = _write(out / "tpu-watch.json", _tpu_watch(ready=True, stage_report_path=tpu))
    args = probe.parse_args([
        "--mode",
        "import",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--import-stage-report",
        str(gpu),
        "--import-stage-report",
        str(cpu),
        "--tpu-watch-report",
        str(tpu_watch),
        "--import-cleanup-verified",
    ])

    report = probe.build_report(args)

    assert report["stage_runtime_reports_collected"] == 3
    assert report["stage_runtime_reports_verified"] == 3
    assert check.validate_report(report, require_live=True) == []
    assert report["stage_runtime_adapter_verified"] is False
    assert report["same_request_route_verified"] is False


def test_import_mode_does_not_require_tpu_watch_when_direct_tpu_report_is_verified() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    gpu = _write(out / "gpu.json", _stage_runtime_report("kaggle_cuda", 0))
    tpu = _write(out / "tpu.json", _stage_runtime_report("kaggle_jax_tpu", 1))
    cpu = _write(out / "cpu.json", _stage_runtime_report("kaggle_cpu", 2))
    args = probe.parse_args([
        "--mode",
        "import",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--import-stage-report",
        str(gpu),
        "--import-stage-report",
        str(tpu),
        "--import-stage-report",
        str(cpu),
        "--import-cleanup-verified",
    ])

    report = probe.build_report(args)

    assert report["stage_runtime_reports_collected"] == 3
    assert report["stage_runtime_reports_verified"] == 3
    assert "glm52_mcp_tpu_stage_runtime_not_ready" not in report["blockers"]
    assert check.validate_report(report, require_live=True) == []


def test_import_mode_keeps_multiple_same_provider_stage_reports() -> None:
    out = _tmp_dir()
    package = _package_report(out)
    for stage_id in [3, 4]:
        package_dir = out / f"pkg-cpu-{stage_id}"
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "kernel-metadata.json").write_text(
            json.dumps({"id": f"tester/ct-glm52-cpu-{stage_id}"}) + "\n",
            encoding="utf-8",
        )
        package["packages"].append({
            "provider": "kaggle_cpu",
            "stage_id": stage_id,
            "package_dir": str(package_dir),
            "private_kernel": True,
            "public_artifact_safe": True,
        })
    package_path = _write(out / "package.json", package)
    cpu2 = _write(out / "cpu2.json", _stage_runtime_report("kaggle_cpu", 2))
    cpu3 = _write(out / "cpu3.json", _stage_runtime_report("kaggle_cpu", 3))
    cpu4 = _write(out / "cpu4.json", _stage_runtime_report("kaggle_cpu", 4))
    args = probe.parse_args([
        "--mode",
        "import",
        "--stage-worker-package-report",
        str(package_path),
        "--output-dir",
        str(out / "probe"),
        "--providers",
        "kaggle_cpu",
        "--stage-ids",
        "2,3,4",
        "--import-stage-report",
        str(cpu2),
        "--import-stage-report",
        str(cpu3),
        "--import-stage-report",
        str(cpu4),
        "--import-cleanup-verified",
    ])

    report = probe.build_report(args)

    assert report["stage_runtime_reports_collected"] == 3
    assert report["stage_runtime_reports_verified"] == 3
    assert [item["stage_id"] for item in report["pushes"]] == [2, 3, 4]
    assert all(item["stage_runtime_verified"] is True for item in report["pushes"])
    assert check.validate_report(report) == []
