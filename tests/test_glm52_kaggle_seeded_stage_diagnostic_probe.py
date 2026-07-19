from __future__ import annotations

import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from scripts import glm52_kaggle_seeded_stage_diagnostic_probe as probe
from scripts import glm52_kaggle_stage_runtime_check as runtime_check
from scripts import glm52_kaggle_stage_worker_push_probe as push_probe


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_seeded_stage_diag_"))


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _package_report(base: Path) -> dict:
    packages = []
    for stage_id in [21, 22, 23]:
        package_dir = base / f"pkg-stage-{stage_id}"
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "kernel-metadata.json").write_text(
            json.dumps({"id": f"tester/ct-glm52-stage-worker-{stage_id}-kaggle-cpu"}) + "\n",
            encoding="utf-8",
        )
        (package_dir / "kernel.py").write_text(
            "# test kernel\n"
            f"{push_probe.PRIVATE_RUNTIME_ENV_INLINE_SENTINEL}"
            "print('ok')\n",
            encoding="utf-8",
        )
        packages.append(
            {
                "provider": "kaggle_cpu",
                "stage_id": stage_id,
                "stage_count": 39,
                "stage_layer_range": [stage_id * 2, stage_id * 2 + 2],
                "compatible_weight_repo": runtime_check.COMPATIBLE_WEIGHT_REPO,
                "package_dir": str(package_dir),
                "kernel_ref": f"tester/ct-glm52-stage-worker-{stage_id}-kaggle-cpu",
                "private_kernel": True,
                "public_artifact_safe": True,
            }
        )
    return {
        "schema": "glm52_kaggle_stage_worker_package_v1",
        "ok": True,
        "glm52_stage_worker_package_ready": True,
        "coordinator_request_id_hash": _hash("b"),
        "packages": packages,
        "public_artifact_safe": True,
    }


def _token_file(base: Path) -> Path:
    return _write(
        base / "kaggle_tokens.md",
        {},
    )


def _write_token_file(base: Path) -> Path:
    path = base / "kaggle_tokens.md"
    path.write_text(
        "# cpuowner\n"
        "export KAGGLE_USERNAME='cpuowner'\n"
        "export KAGGLE_KEY='KGA_TEST_SECRET_VALUE'\n",
        encoding="utf-8",
    )
    return path


def _post(url: str, token: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload, sort_keys=True).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-CrowdTensor-GLM52-Token": token,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        loaded = json.loads(response.read().decode("utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _stage_runtime_report(task: dict, *, coordinator_url: str) -> dict:
    stage_id = int(task["stage_id"])
    return {
        "schema": runtime_check.STAGE_SCHEMA,
        "ok": True,
        "public_artifact_safe": True,
        "model_id": runtime_check.MODEL_ID,
        "compatible_weight_repo": runtime_check.COMPATIBLE_WEIGHT_REPO,
        "provider": "kaggle_cpu",
        "stage_id": stage_id,
        "stage_layer_range": [stage_id * 2, stage_id * 2 + 2],
        "coordinator_request_id_hash": str(task.get("coordinator_request_id_hash") or _hash("b")),
        "stage_execution_verified": True,
        "stage_decode_verified": True,
        "stage_full_decode_verified": True,
        "stage_runtime_adapter_verified": True,
        "same_request_route_verified": True,
        "stage_output_hash": _hash("a"),
        "weight_tensor_values_loaded": True,
        "stage_owned_weight_values_loaded": True,
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
        "coordinator_stage_decode_verified": True,
        "coordinator_stage_tasks_accepted": 1,
        "coordinator_stage_last_submit_accepted": True,
        "coordinator_stage_last_submit_ready": False,
        "coordinator_stage_last_full_prefix_adapter_verified": True,
        "coordinator_stage_last_full_prefix_probe_ready": True,
        "coordinator_stage_last_full_prefix_input_activation_consumed": True,
        "coordinator_stage_last_full_prefix_output_activation_private_ready": True,
        "coordinator_stage_last_full_prefix_output_activation_hash": _hash("e"),
        "coordinator_stage_last_full_prefix_stdout_hash": _hash("f"),
        "coordinator_stage_last_full_prefix_stderr_hash": _hash("0"),
        "coordinator_url_public": False,
        "coordinator_url_hash": probe.sha_json(coordinator_url),
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


def test_seeded_stage_diagnostic_runs_target_stage_without_overclaim() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    token_path = _write_token_file(out)
    args = probe.parse_args(
        [
            "--stage-worker-package-report",
            str(package_path),
            "--output-dir",
            str(out / "diag"),
            "--stage-id",
            "22",
            "--seed-hidden-shape",
            "3,8",
            "--coordinator-bind-host",
            "127.0.0.1",
            "--coordinator-public-host",
            "127.0.0.1",
            "--wait-seconds",
            "0",
            "--token-file",
            str(token_path),
            "--token-section",
            "cpuowner",
        ]
    )
    claimed_tasks: list[dict] = []

    def fake_runner(command, **kwargs):
        if command[:3] == ["kaggle", "kernels", "push"]:
            package_dir = Path(command[command.index("-p") + 1])
            private_env = json.loads((package_dir / push_probe.PRIVATE_RUNTIME_ENV_FILENAME).read_text(encoding="utf-8"))
            claim = _post(
                private_env["CT_GLM52_COORDINATOR_URL"],
                private_env["CT_GLM52_COORDINATOR_TOKEN"],
                "/claim",
                {"miner_id": "fake-stage22", "stage_id": 22},
            )
            task = claim["task"]
            claimed_tasks.append(task)
            submit = _post(
                private_env["CT_GLM52_COORDINATOR_URL"],
                private_env["CT_GLM52_COORDINATOR_TOKEN"],
                "/submit",
                {
                    "task_id": task["task_id"],
                    "stage_id": 22,
                    "generation_step": 0,
                    "public_artifact_safe": True,
                    "stage_decode_verified": True,
                    "stage_output_hash": _hash("a"),
                    "output_hash": _hash("a"),
                    "weight_value_sha256": _hash("c"),
                    "weight_value_byte_count": 16,
                    "provider_runtime_verified": True,
                    "provider_device_count": 1,
                    "stage_decode_report_hash": _hash("d"),
                    "activation": {
                        "schema": "glm52_private_stage_activation_v1",
                        "activation_hash": _hash("e"),
                        "hidden_shape": [3, 8],
                        "hidden_dtype": "float16",
                        "hidden_b64": "PRIVATE_NEXT_STAGE_PAYLOAD",
                        "activation_public": False,
                    },
                    "activation_hash": _hash("e"),
                },
            )
            assert submit["accepted"] is True
        if command[:3] == ["kaggle", "kernels", "status"]:
            return subprocess.CompletedProcess(command, 0, stdout="status COMPLETE", stderr="")
        if command[:3] == ["kaggle", "kernels", "output"]:
            output_dir = Path(command[command.index("-p") + 1])
            _write(
                output_dir / "glm52_kaggle_stage_runtime_report.json",
                _stage_runtime_report(claimed_tasks[-1], coordinator_url="private-url-not-public"),
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)
    encoded = json.dumps(report, sort_keys=True)

    assert report["target_stage_diagnostic_verified"] is True
    assert report["target_stage_push_verified"] is True
    assert report["target_stage_coordinator_completed"] is True
    assert report["same_request_decode_verified"] is False
    assert report["generated_token_count"] == 0
    assert report["completion_boundary"]["seeded_diagnostic_is_not_deployment_rc"] is True
    assert report["completion_boundary"]["synthetic_previous_activation_used"] is True
    assert report["target_stage_report_diagnostics"]["coordinator_stage_last_submit_accepted"] is True
    assert "PRIVATE_NEXT_STAGE_PAYLOAD" not in encoded
    assert "KGA_TEST_SECRET_VALUE" not in encoded
    assert probe.public_redaction_errors(report) == []


def test_seeded_stage_diagnostic_records_target_failure_diagnostics() -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    token_path = _write_token_file(out)
    args = probe.parse_args(
        [
            "--stage-worker-package-report",
            str(package_path),
            "--output-dir",
            str(out / "diag"),
            "--stage-id",
            "22",
            "--seed-hidden-shape",
            "3,8",
            "--coordinator-bind-host",
            "127.0.0.1",
            "--coordinator-public-host",
            "127.0.0.1",
            "--wait-seconds",
            "0",
            "--token-file",
            str(token_path),
            "--token-section",
            "cpuowner",
        ]
    )

    def fake_runner(command, **kwargs):
        if command[:3] == ["kaggle", "kernels", "status"]:
            return subprocess.CompletedProcess(command, 0, stdout="status COMPLETE", stderr="")
        if command[:3] == ["kaggle", "kernels", "output"]:
            output_dir = Path(command[command.index("-p") + 1])
            failed = _stage_runtime_report(
                {
                    "stage_id": 22,
                    "coordinator_request_id_hash": _hash("b"),
                },
                coordinator_url="private-url-not-public",
            )
            failed["ok"] = False
            failed["stage_decode_verified"] = False
            failed["stage_execution_verified"] = False
            failed["stage_full_decode_verified"] = False
            failed["stage_runtime_adapter_verified"] = False
            failed["same_request_route_verified"] = False
            failed["stage_output_hash"] = ""
            failed["blockers"] = ["glm52_full_prefix_stage_runtime_probe_not_verified"]
            failed["coordinator_stage_last_full_prefix_probe_errors"] = [
                {
                    "phase": "adapter",
                    "error_type": "RuntimeError",
                    "error_public": "synthetic failure",
                    "error_digest": _hash("1"),
                }
            ]
            _write(output_dir / "glm52_kaggle_stage_runtime_report.json", failed)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = probe.build_report(args, runner=fake_runner)

    assert report["target_stage_diagnostic_verified"] is False
    assert report["same_request_decode_verified"] is False
    assert "glm52_seeded_target_stage_push_not_verified" in report["blockers"]
    assert "glm52_seeded_target_stage_coordinator_submit_missing" in report["blockers"]
    assert report["target_stage_report_diagnostics"]["coordinator_stage_last_full_prefix_probe_errors"][0]["error_public"] == "synthetic failure"
    assert probe.public_redaction_errors(report) == []
