from __future__ import annotations

import json
import time
import tempfile
import urllib.request
from pathlib import Path

from scripts import glm52_kaggle_same_request_live_check as live_check
from scripts import glm52_kaggle_same_request_live_probe as live
from scripts import glm52_kaggle_stage_worker_push_probe as push_probe


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_same_request_live_"))


def _write(path: Path, payload: dict | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _hash(label: str) -> str:
    return live.bridge.sha_json({"test": label})


def _package_report(base: Path) -> dict:
    specs = [
        ("kaggle_cpu", 20, [4, 6], "cpu-owner"),
        ("kaggle_cuda", 10, [0, 2], "gpu-owner"),
        ("kaggle_jax_tpu", 30, [2, 4], "tpu-owner"),
    ]
    packages = []
    for provider, stage_id, layer_range, owner in specs:
        package_dir = base / f"pkg-{stage_id}-{provider}"
        package_dir.mkdir(parents=True, exist_ok=True)
        kernel_ref = f"{owner}/ct-glm52-stage-worker-{stage_id}"
        _write(package_dir / "kernel-metadata.json", {"id": kernel_ref})
        packages.append(
            {
                "provider": provider,
                "stage_id": stage_id,
                "stage_count": len(specs),
                "stage_layer_range": layer_range,
                "kaggle_owner": owner,
                "kernel_ref": kernel_ref,
                "package_dir": str(package_dir),
                "compatible_weight_repo": live.same_request_probe.COMPATIBLE_WEIGHT_REPO,
                "public_artifact_safe": True,
            }
        )
    return {
        "schema": "glm52_kaggle_stage_worker_package_v1",
        "ok": True,
        "coordinator_request_id_hash": _hash("request"),
        "packages": packages,
        "public_artifact_safe": True,
    }


def _grouped_cpu_package_report(base: Path) -> dict:
    packages = []
    entries = [
        ("kaggle_cuda", [0], [[0, 2]], "gpu-owner"),
        ("kaggle_cpu", [1, 2], [[2, 4], [4, 6]], "cpu-owner"),
        ("kaggle_jax_tpu", [3], [[6, 8]], "tpu-owner"),
    ]
    for provider, stage_ids, ranges, owner in entries:
        first = stage_ids[0]
        package_dir = base / f"pkg-{first}-{provider}"
        package_dir.mkdir(parents=True, exist_ok=True)
        kernel_ref = f"{owner}/ct-glm52-stage-worker-{first}"
        _write(package_dir / "kernel-metadata.json", {"id": kernel_ref})
        stage_specs = [
            {
                "stage_id": stage_id,
                "stage_count": 4,
                "provider": provider,
                "stage_layer_range": layer_range,
                "compatible_weight_repo": live.same_request_probe.COMPATIBLE_WEIGHT_REPO,
                "public_artifact_safe": True,
            }
            for stage_id, layer_range in zip(stage_ids, ranges, strict=True)
        ]
        packages.append(
            {
                "provider": provider,
                "stage_id": first,
                "stage_ids": stage_ids,
                "stage_specs": stage_specs if len(stage_ids) > 1 else [],
                "stage_count": 4,
                "stage_layer_range": [ranges[0][0], ranges[-1][1]],
                "kaggle_owner": owner,
                "kernel_ref": kernel_ref,
                "package_dir": str(package_dir),
                "compatible_weight_repo": live.same_request_probe.COMPATIBLE_WEIGHT_REPO,
                "grouped_stage_worker": len(stage_ids) > 1,
                "public_artifact_safe": True,
            }
        )
    return {
        "schema": "glm52_kaggle_stage_worker_package_v1",
        "ok": True,
        "coordinator_request_id_hash": _hash("grouped-request"),
        "packages": packages,
        "public_artifact_safe": True,
    }


def _post_json(url: str, token: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-CrowdTensor-GLM52-Token": token,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        loaded = json.loads(response.read().decode("utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _stage_runtime_report(provider: str, stage_id: int, task: dict, *, final: bool) -> dict:
    return {
        "schema": "glm52_kaggle_stage_runtime_report_v1",
        "ok": True,
        "public_artifact_safe": True,
        "model_id": live.same_request_probe.MODEL_ID,
        "compatible_weight_repo": live.same_request_probe.COMPATIBLE_WEIGHT_REPO,
        "provider": provider,
        "stage_id": stage_id,
        "stage_layer_range": list(task.get("stage_layer_range") or []),
        "coordinator_request_id_hash": str(task.get("coordinator_request_id_hash") or ""),
        "stage_execution_verified": True,
        "stage_decode_verified": True,
        "stage_output_hash": _hash(f"stage-output-{stage_id}"),
        "weight_tensor_values_loaded": True,
        "stage_owned_weight_values_loaded": True,
        "weight_value_byte_count": 128 + stage_id,
        "weight_value_sha256": _hash(f"weight-{stage_id}"),
        "weight_tensor_values_public": False,
        "live_run_performed": True,
        "fallback_model_used": False,
        "queue_only_evidence": False,
        "metadata_only": False,
        "stage_smoke_only": False,
        "activation_public": False,
        "kv_cache_public": False,
        "safety": live.same_request_probe.safety_flags(),
        "final_stage": final,
    }


def test_mapping_parser_and_provider_token_selection() -> None:
    args = live.parse_args(
        [
            "--stage-worker-package-report",
            "package.json",
            "--token-file",
            "~/.config/crowdtensor/kaggle-tokens.md",
            "--token-section",
            "cpuowner",
            "--provider-token-file-map",
            "kaggle_cuda=/tmp/gpu.md,kaggle_jax_tpu=/tmp/tpu.md",
            "--provider-token-section-map",
            "kaggle_jax_tpu=tpuowner",
            "--provider-raw-token-file-map",
            "kaggle_cuda=/tmp/gpu.raw",
            "--provider-raw-token-username-map",
            "kaggle_cuda=gpuowner",
        ]
    )

    assert live.parse_mapping("a=1,b=2") == {"a": "1", "b": "2"}
    assert live.token_config_for_provider("kaggle_cuda", args) == {
        "token_file": "/tmp/gpu.md",
        "token_section": "cpuowner",
        "raw_token_file": "/tmp/gpu.raw",
        "raw_token_username": "gpuowner",
    }
    assert live.token_config_for_provider("kaggle_jax_tpu", args)["token_section"] == "tpuowner"
    assert live.token_config_for_provider("kaggle_cpu", args)["token_file"] == "~/.config/crowdtensor/kaggle-tokens.md"


def test_live_probe_routes_stages_by_layer_order_and_requires_full_completion(monkeypatch) -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    gpu_raw = _write(out / "gpu.raw", "gpu-secret")
    token_file = _write(
        out / "tokens.md",
        "# tpuowner\n"
        "export KAGGLE_USERNAME='tpu-owner'\n"
        "export KAGGLE_KEY='tpu-secret'\n"
        "# cpuowner\n"
        "export KAGGLE_USERNAME='cpu-owner'\n"
        "export KAGGLE_KEY='cpu-secret'\n",
    )
    observed_stage_ids: list[int] = []
    observed_token_modes: dict[str, tuple[str, str, str, str]] = {}

    def fake_push_build_report(push_args, *, runner):
        provider = push_args.providers[0]
        stage_id = next(iter(push_args.stage_ids))
        observed_stage_ids.append(stage_id)
        observed_token_modes[provider] = (
            push_args.token_file,
            push_args.token_section,
            push_args.raw_token_file,
            push_args.raw_token_username,
        )
        token = Path(push_args.coordinator_token_file).read_text(encoding="utf-8").strip()
        claim = _post_json(
            f"{push_args.coordinator_url}/claim",
            token,
            {"miner_id": f"{provider}-{stage_id}", "stage_id": stage_id},
        )
        task = claim["task"]
        assert task["stage_id"] == stage_id
        final = task.get("is_final_stage") is True
        payload = {
            "task_id": task["task_id"],
            "stage_id": stage_id,
            "generation_step": int(task.get("generation_step") or 0),
            "public_artifact_safe": True,
            "stage_decode_verified": True,
            "stage_output_hash": _hash(f"stage-output-{stage_id}"),
            "output_hash": _hash(f"output-{stage_id}"),
            "weight_value_sha256": _hash(f"weight-{stage_id}"),
            "weight_value_byte_count": 128 + stage_id,
            "provider_runtime_verified": True,
            "provider_device_count": 1,
            "kv_cache": {"stage_local_kv_cache_verified": True},
        }
        if final:
            payload["generated_token_hash"] = _hash("generated-token")
        else:
            payload["activation"] = {
                "activation_hash": _hash(f"activation-{stage_id}"),
                "hidden_shape": [1, 8],
                "hidden_dtype": "float16",
                "hidden_b64": "PRIVATE_TEST_ACTIVATION",
                "activation_public": False,
            }
        submit = _post_json(f"{push_args.coordinator_url}/submit", token, payload)
        assert submit["accepted"] is True
        output_dir = Path(push_args.output_dir) / "notebook-output" / f"stage-{stage_id}-{provider}"
        stage_report_path = _write(
            output_dir / "glm52_kaggle_stage_runtime_report.json",
            _stage_runtime_report(provider, stage_id, task, final=final),
        )
        return {
            "schema": push_probe.SCHEMA,
            "generated_at": live.utc_now(),
            "mode": "live",
            "ok": True,
            "glm52_stage_worker_push_probe_ready": True,
            "live_run_performed": True,
            "stage_runtime_reports_collected": 1,
            "stage_runtime_reports_verified": 1,
            "same_request_route_verified": False,
            "stage_runtime_adapter_verified": False,
            "pushes": [
                {
                    "schema": "glm52_kaggle_stage_worker_push_entry_v1",
                    "provider": provider,
                    "stage_id": stage_id,
                    "kernel_ref": f"owner/stage-{stage_id}",
                    "pushed": True,
                    "push_error_blocker": "",
                    "terminal_status": "COMPLETE",
                    "output_collected": True,
                    "stage_report_path": str(stage_report_path),
                    "stage_report_present": True,
                    "stage_runtime_verified": True,
                    "cleanup_performed": True,
                    "public_artifact_safe": True,
                }
            ],
            "blockers": [],
            "completion_boundary": {
                "preflight_is_not_runtime_success": True,
                "push_required": True,
                "terminal_kernel_output_required": True,
                "stage_runtime_check_required": True,
                "same_request_probe_required": True,
            },
            "public_artifact_safe": True,
            "safety": live.same_request_probe.safety_flags(),
        }

    monkeypatch.setattr(live.push_probe, "build_report", fake_push_build_report)
    args = live.parse_args(
        [
            "--output-dir",
            str(out / "live"),
            "--stage-worker-package-report",
            str(package_path),
            "--coordinator-bind-host",
            "127.0.0.1",
            "--coordinator-public-host",
            "127.0.0.1",
            "--provider-token-file-map",
            f"kaggle_cuda={token_file},kaggle_jax_tpu={token_file},kaggle_cpu={token_file}",
            "--provider-token-section-map",
            "kaggle_jax_tpu=tpuowner,kaggle_cpu=cpuowner",
            "--provider-raw-token-file-map",
            f"kaggle_cuda={gpu_raw}",
            "--provider-raw-token-username-map",
            "kaggle_cuda=gpuowner",
        ]
    )

    report = live.run_live(args)

    assert observed_stage_ids == [10, 30, 20]
    assert report["stage_order"] == [10, 30, 20]
    assert report["same_request_decode_verified"] is True
    assert report["full_stage_count_verified"] is True
    assert report["generated_token_count"] == 1
    assert report["stage_runtime_reports_verified"] == 3
    assert report["coordinator_stage_reports_collected"] == 3
    assert set(report["accepted_providers"]) == set(live.REQUIRED_PROVIDERS)
    assert observed_token_modes["kaggle_cuda"][2] == str(gpu_raw)
    assert observed_token_modes["kaggle_cuda"][3] == "gpuowner"
    assert observed_token_modes["kaggle_jax_tpu"][1] == "tpuowner"
    assert observed_token_modes["kaggle_cpu"][1] == "cpuowner"
    encoded = json.dumps(report, sort_keys=True)
    assert "PRIVATE_TEST_ACTIVATION" not in encoded
    assert live_check.validate_report(report, require_verified=True) == []


def test_live_probe_multitoken_uses_concurrent_stage_workers(monkeypatch) -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    accepted_by_stage: dict[int, int] = {}

    def fake_push_build_report(push_args, *, runner):
        provider = push_args.providers[0]
        stage_id = next(iter(push_args.stage_ids))
        token = Path(push_args.coordinator_token_file).read_text(encoding="utf-8").strip()
        accepted = 0
        last_task: dict = {}
        deadline = time.monotonic() + 5.0
        while accepted < push_args.coordinator_stage_task_limit and time.monotonic() < deadline:
            claim = _post_json(
                f"{push_args.coordinator_url}/claim",
                token,
                {"miner_id": f"{provider}-{stage_id}", "stage_id": stage_id},
            )
            if claim.get("done"):
                break
            task = claim.get("task") if isinstance(claim.get("task"), dict) else {}
            if not task:
                time.sleep(0.005)
                continue
            last_task = task
            step = int(task.get("generation_step") or 0)
            final = task.get("is_final_stage") is True
            payload = {
                "task_id": task["task_id"],
                "stage_id": stage_id,
                "generation_step": step,
                "public_artifact_safe": True,
                "stage_decode_verified": True,
                "stage_output_hash": _hash(f"stage-output-{stage_id}-{step}"),
                "output_hash": _hash(f"output-{stage_id}-{step}"),
                "weight_value_sha256": _hash(f"weight-{stage_id}"),
                "weight_value_byte_count": 128 + stage_id,
                "provider_runtime_verified": True,
                "provider_device_count": 1,
                "kv_cache": {"stage_local_kv_cache_verified": True},
            }
            if final:
                payload["generated_token_hash"] = _hash(f"generated-token-{step}")
            else:
                payload["activation"] = {
                    "activation_hash": _hash(f"activation-{stage_id}-{step}"),
                    "hidden_shape": [1, 8],
                    "hidden_dtype": "float16",
                    "hidden_b64": "PRIVATE_TEST_ACTIVATION",
                    "activation_public": False,
                }
            submit = _post_json(f"{push_args.coordinator_url}/submit", token, payload)
            assert submit["accepted"] is True
            accepted += 1
            if submit.get("ready") is True:
                break
        accepted_by_stage[stage_id] = accepted
        output_dir = Path(push_args.output_dir) / "notebook-output" / f"stage-{stage_id}-{provider}"
        stage_report = _stage_runtime_report(provider, stage_id, last_task or {"stage_layer_range": [0, 1]}, final=False)
        stage_report["coordinator_stage_tasks_accepted"] = accepted
        stage_report_path = _write(output_dir / "glm52_kaggle_stage_runtime_report.json", stage_report)
        return {
            "schema": push_probe.SCHEMA,
            "generated_at": live.utc_now(),
            "mode": "live",
            "ok": True,
            "glm52_stage_worker_push_probe_ready": True,
            "live_run_performed": True,
            "stage_runtime_reports_collected": 1,
            "stage_runtime_reports_verified": 1,
            "same_request_route_verified": True,
            "stage_runtime_adapter_verified": True,
            "pushes": [
                {
                    "schema": "glm52_kaggle_stage_worker_push_entry_v1",
                    "provider": provider,
                    "stage_id": stage_id,
                    "pushed": True,
                    "push_error_blocker": "",
                    "terminal_status": "COMPLETE",
                    "output_collected": True,
                    "stage_report_path": str(stage_report_path),
                    "stage_report_present": True,
                    "stage_runtime_verified": True,
                    "cleanup_performed": True,
                    "public_artifact_safe": True,
                }
            ],
            "blockers": [],
            "public_artifact_safe": True,
            "safety": live.same_request_probe.safety_flags(),
        }

    monkeypatch.setattr(live.push_probe, "build_report", fake_push_build_report)
    args = live.parse_args(
        [
            "--output-dir",
            str(out / "live"),
            "--stage-worker-package-report",
            str(package_path),
            "--coordinator-bind-host",
            "127.0.0.1",
            "--coordinator-public-host",
            "127.0.0.1",
            "--max-new-tokens",
            "3",
            "--stage-push-parallelism",
            "3",
        ]
    )

    report = live.run_live(args)

    assert report["target_generated_token_count"] == 3
    assert report["expected_stage_task_count"] == 9
    assert report["generated_token_count"] == 3
    assert len(report["generated_token_hashes"]) == 3
    assert report["coordinator_stage_reports_collected"] == 9
    assert report["worker_stage_decode_task_count"] == 9
    assert accepted_by_stage == {10: 3, 30: 3, 20: 3}
    assert report["same_request_decode_verified"] is True
    assert live_check.validate_report(report, require_verified=True) == []


def test_live_probe_multitoken_expands_grouped_cpu_package_without_overclaim(monkeypatch) -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _grouped_cpu_package_report(out))
    accepted_by_package: dict[tuple[int, ...], int] = {}
    observed_stage_id_groups: list[list[int]] = []
    observed_task_limits: dict[tuple[int, ...], int] = {}
    observed_tuning: dict[tuple[int, ...], dict[str, int | float]] = {}

    def fake_push_build_report(push_args, *, runner):
        provider = push_args.providers[0]
        stage_ids = sorted(push_args.stage_ids)
        observed_stage_id_groups.append(stage_ids)
        observed_task_limits[tuple(stage_ids)] = push_args.coordinator_stage_task_limit
        observed_tuning[tuple(stage_ids)] = {
            "full_prefix_prefill_length": push_args.full_prefix_prefill_length,
            "full_prefix_dsa_mask_topk": push_args.full_prefix_dsa_mask_topk,
            "full_prefix_executed_expert_count": push_args.full_prefix_executed_expert_count,
            "full_prefix_top_k": push_args.full_prefix_top_k,
            "full_prefix_row_block_size": push_args.full_prefix_row_block_size,
            "full_prefix_max_tensor_bytes": push_args.full_prefix_max_tensor_bytes,
            "full_prefix_max_block_bytes": push_args.full_prefix_max_block_bytes,
            "cpu_group_stage_attempt_seconds": push_args.cpu_group_stage_attempt_seconds,
            "cpu_group_stage_poll_seconds": push_args.cpu_group_stage_poll_seconds,
        }
        token = Path(push_args.coordinator_token_file).read_text(encoding="utf-8").strip()
        accepted = 0
        processed_tasks = []
        last_task_by_stage: dict[int, dict] = {}
        deadline = time.monotonic() + 6.0
        while accepted < push_args.coordinator_stage_task_limit and time.monotonic() < deadline:
            made_progress = False
            for stage_id in stage_ids:
                if accepted >= push_args.coordinator_stage_task_limit:
                    break
                claim = _post_json(
                    f"{push_args.coordinator_url}/claim",
                    token,
                    {"miner_id": f"{provider}-{stage_id}", "stage_id": stage_id},
                )
                if claim.get("done"):
                    break
                task = claim.get("task") if isinstance(claim.get("task"), dict) else {}
                if not task:
                    continue
                last_task_by_stage[stage_id] = task
                step = int(task.get("generation_step") or 0)
                final = task.get("is_final_stage") is True
                payload = {
                    "task_id": task["task_id"],
                    "stage_id": stage_id,
                    "generation_step": step,
                    "public_artifact_safe": True,
                    "stage_decode_verified": True,
                    "stage_output_hash": _hash(f"group-stage-output-{stage_id}-{step}"),
                    "output_hash": _hash(f"group-output-{stage_id}-{step}"),
                    "weight_value_sha256": _hash(f"group-weight-{stage_id}"),
                    "weight_value_byte_count": 256 + stage_id,
                    "provider_runtime_verified": True,
                    "provider_device_count": 1,
                    "kv_cache": {"stage_local_kv_cache_verified": True},
                }
                if final:
                    payload["generated_token_hash"] = _hash(f"group-generated-token-{step}")
                else:
                    payload["activation"] = {
                        "activation_hash": _hash(f"group-activation-{stage_id}-{step}"),
                        "hidden_shape": [1, 8],
                        "hidden_dtype": "float16",
                        "hidden_b64": "PRIVATE_TEST_ACTIVATION",
                        "activation_public": False,
                    }
                submit = _post_json(f"{push_args.coordinator_url}/submit", token, payload)
                assert submit["accepted"] is True
                accepted += 1
                made_progress = True
                processed_tasks.append({
                    "task_id_hash": _hash(f"group-task-{stage_id}-{step}-{accepted}"),
                    "stage_id": stage_id,
                    "accepted": True,
                    "stage_output_hash": payload["stage_output_hash"],
                })
                if submit.get("ready") is True:
                    break
            if not made_progress:
                time.sleep(0.005)
        accepted_by_package[tuple(stage_ids)] = accepted
        first_stage = stage_ids[0]
        last_task = last_task_by_stage.get(first_stage) or {"stage_layer_range": [first_stage * 2, first_stage * 2 + 2]}
        output_dir = Path(push_args.output_dir) / "notebook-output" / f"stage-{first_stage}-{provider}"
        stage_report = _stage_runtime_report(provider, first_stage, last_task, final=False)
        stage_report["stage_ids"] = stage_ids
        stage_report["stage_ids_verified"] = stage_ids
        stage_report["stage_layer_range"] = [stage_ids[0] * 2, stage_ids[-1] * 2 + 2]
        stage_report["coordinator_stage_tasks_accepted"] = accepted
        stage_report["coordinator_stage_processed_tasks"] = processed_tasks
        stage_report_path = _write(output_dir / "glm52_kaggle_stage_runtime_report.json", stage_report)
        return {
            "schema": push_probe.SCHEMA,
            "generated_at": live.utc_now(),
            "mode": "live",
            "ok": True,
            "glm52_stage_worker_push_probe_ready": True,
            "live_run_performed": True,
            "stage_runtime_reports_collected": 1,
            "stage_runtime_reports_verified": 1,
            "same_request_route_verified": True,
            "stage_runtime_adapter_verified": True,
            "pushes": [
                {
                    "schema": "glm52_kaggle_stage_worker_push_entry_v1",
                    "provider": provider,
                    "stage_id": first_stage,
                    "stage_ids": stage_ids,
                    "pushed": True,
                    "push_error_blocker": "",
                    "terminal_status": "COMPLETE",
                    "output_collected": True,
                    "stage_report_path": str(stage_report_path),
                    "stage_report_present": True,
                    "stage_runtime_verified": True,
                    "cleanup_performed": True,
                    "public_artifact_safe": True,
                }
            ],
            "blockers": [],
            "public_artifact_safe": True,
            "safety": live.same_request_probe.safety_flags(),
        }

    monkeypatch.setattr(live.push_probe, "build_report", fake_push_build_report)
    args = live.parse_args(
        [
            "--output-dir",
            str(out / "live"),
            "--stage-worker-package-report",
            str(package_path),
            "--coordinator-bind-host",
            "127.0.0.1",
            "--coordinator-public-host",
            "127.0.0.1",
            "--max-new-tokens",
            "2",
            "--stage-push-parallelism",
            "3",
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
        ]
    )

    report = live.run_live(args)

    assert sorted(observed_stage_id_groups) == [[0], [1, 2], [3]]
    assert observed_task_limits[(1, 2)] == 4
    assert observed_tuning[(1, 2)] == {
        "full_prefix_prefill_length": 1,
        "full_prefix_dsa_mask_topk": 1,
        "full_prefix_executed_expert_count": 2,
        "full_prefix_top_k": 1,
        "full_prefix_row_block_size": 512,
        "full_prefix_max_tensor_bytes": 33554432,
        "full_prefix_max_block_bytes": 16777216,
        "cpu_group_stage_attempt_seconds": 2.5,
        "cpu_group_stage_poll_seconds": 0.5,
    }
    assert accepted_by_package == {(0,): 2, (1, 2): 4, (3,): 2}
    assert report["runtime_tuning"]["full_prefix_prefill_length"] == 1
    assert report["runtime_tuning"]["cpu_group_stage_poll_seconds"] == 0.5
    assert report["stage_count"] == 4
    assert report["stage_runtime_reports_collected"] == 4
    assert report["stage_runtime_reports_verified"] == 4
    assert report["stage_worker_package_reports_collected"] == 3
    assert report["stage_worker_package_reports_verified"] == 3
    assert report["coordinator_stage_reports_collected"] == 8
    assert report["worker_stage_decode_reports_collected"] == 4
    assert report["worker_stage_decode_task_count"] == 8
    assert report["generated_token_count"] == 2
    assert report["same_request_decode_verified"] is True
    assert live_check.validate_report(report, require_verified=True) == []


def test_live_checker_rejects_partial_overclaim() -> None:
    report = {
        "schema": live.SCHEMA,
        "ok": True,
        "mode": "live",
        "model_id": live.same_request_probe.MODEL_ID,
        "public_artifact_safe": True,
        "coordinator_url_public": False,
        "coordinator_token_public": False,
        "same_request_decode_verified": True,
        "generated_token_count": 1,
        "accepted_providers": list(live.REQUIRED_PROVIDERS),
        "stage_count": 39,
        "stage_order": list(range(39)),
        "stage_runtime_reports_collected": 3,
        "stage_runtime_reports_verified": 3,
        "coordinator_stage_reports_collected": 3,
        "worker_stage_decode_reports_collected": 3,
        "full_stage_count_verified": False,
        "coordinator_status": {
            "ready": True,
            "generated_token_count": 1,
            "completed_task_count": 3,
            "pending_count": 0,
        },
        "cleanup_status": {
            "temporary_kaggle_kernels_deleted": True,
            "temporary_private_packages_removed": True,
            "live_resources_left_running": False,
            "public_artifact_safe": True,
        },
        "same_request_check": {"ok": True, "error_count": 0, "errors": []},
        "blockers": [],
        "safety": live.same_request_probe.safety_flags(),
    }

    errors = live_check.validate_report(report, require_verified=True)

    assert "full_stage_count_not_verified" in errors
    assert "stage_runtime_reports_collected_count_mismatch" in errors
    assert "coordinator_completed_task_count_mismatch" in errors


def test_cleanup_from_push_does_not_retain_rejected_kernel_push() -> None:
    cleanup = live.cleanup_from_push(
        {
            "pushes": [
                {
                    "provider": "kaggle_cuda",
                    "stage_id": 0,
                    "pushed": False,
                    "push_error_blocker": "kaggle_gpu_quota_or_session_rejected",
                    "cleanup_performed": False,
                }
            ]
        }
    )

    assert cleanup["temporary_kaggle_kernels_deleted"] is True
    assert cleanup["temporary_private_packages_removed"] is True
    assert cleanup["live_resources_left_running"] is False
    assert cleanup["retained_or_uncleaned_kernels"] == []
    assert cleanup["blockers"] == []


def test_live_probe_stops_when_stage_runtime_does_not_submit_to_coordinator(monkeypatch) -> None:
    out = _tmp_dir()
    package_path = _write(out / "package.json", _package_report(out))
    attempted: list[int] = []

    def fake_push_build_report(push_args, *, runner):
        provider = push_args.providers[0]
        stage_id = next(iter(push_args.stage_ids))
        attempted.append(stage_id)
        output_dir = Path(push_args.output_dir) / "notebook-output" / f"stage-{stage_id}-{provider}"
        stage_report = _stage_runtime_report(provider, stage_id, {
            "stage_layer_range": [0, 2],
            "coordinator_request_id_hash": _hash("request"),
        }, final=False)
        stage_report["stage_decode_verified"] = False
        stage_report_path = _write(output_dir / "glm52_kaggle_stage_runtime_report.json", stage_report)
        return {
            "schema": push_probe.SCHEMA,
            "generated_at": live.utc_now(),
            "mode": "live",
            "ok": True,
            "glm52_stage_worker_push_probe_ready": True,
            "live_run_performed": True,
            "stage_runtime_reports_collected": 1,
            "stage_runtime_reports_verified": 1,
            "same_request_route_verified": False,
            "stage_runtime_adapter_verified": False,
            "pushes": [
                {
                    "schema": "glm52_kaggle_stage_worker_push_entry_v1",
                    "provider": provider,
                    "stage_id": stage_id,
                    "pushed": True,
                    "push_error_blocker": "",
                    "terminal_status": "COMPLETE",
                    "output_collected": True,
                    "stage_report_path": str(stage_report_path),
                    "stage_report_present": True,
                    "stage_runtime_verified": True,
                    "cleanup_performed": True,
                    "public_artifact_safe": True,
                }
            ],
            "blockers": [],
            "completion_boundary": {
                "preflight_is_not_runtime_success": True,
                "push_required": True,
                "terminal_kernel_output_required": True,
                "stage_runtime_check_required": True,
                "same_request_probe_required": True,
            },
            "public_artifact_safe": True,
            "safety": live.same_request_probe.safety_flags(),
        }

    monkeypatch.setattr(live.push_probe, "build_report", fake_push_build_report)
    args = live.parse_args([
        "--output-dir",
        str(out / "live"),
        "--stage-worker-package-report",
        str(package_path),
        "--coordinator-bind-host",
        "127.0.0.1",
        "--coordinator-public-host",
        "127.0.0.1",
    ])

    report = live.run_live(args)

    assert attempted == [10]
    assert report["same_request_decode_verified"] is False
    assert report["coordinator_status"]["completed_task_count"] == 0
    assert "glm52_stage_worker_stage_decode_not_verified:kaggle_cuda:stage10" in report["blockers"]
    assert "glm52_stage_worker_coordinator_submit_missing:kaggle_cuda:stage10" in report["blockers"]
    assert "glm52_stage_worker_push_stopped_before_all_stages" in report["blockers"]
    assert "same_request_live_not_verified" in live_check.validate_report(report, require_verified=True)
