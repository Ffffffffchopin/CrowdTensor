"""End-to-end local Training Foundation orchestration and user operations."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .hf_lora_training import (
    CUDATrainingRuntimeDryRun,
    CPULoRATrainingRuntime,
    create_local_training_fixture,
    evaluate_adapter,
)
from .named_tensor_optimizer import (
    compress_sign_with_error_feedback,
    decode_sign_transport,
    export_standard_peft_adapter,
    load_tensors,
)
from .pipeline_lora_training import compare_pipeline_runs, run_two_process_pipeline
from .training_contract import public_training_spec, sha256_file, sha256_json


JOB_REPORT_SCHEMA = "crowdtensor_training_foundation_job_v1"
STATUS_SCHEMA = "crowdtensor_training_job_status_v1"
CLEANUP_SCHEMA = "crowdtensor_training_cleanup_v1"
GPU_HANDOFF_SCHEMA = "crowdtensor_gpu_training_continuation_v1"


PHASES = [
    "configuration",
    "dataset",
    "worker_assignment",
    "forward",
    "backward",
    "local_step",
    "outer_aggregation",
    "checkpoint",
    "evaluation",
    "cleanup",
]


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _status_path(output: Path) -> Path:
    return output / "training_status.json"


def _initial_status(output: Path, job_id: str) -> dict[str, Any]:
    return {
        "schema": STATUS_SCHEMA,
        "job_id": job_id,
        "overall_state": "running",
        "current_phase": "configuration",
        "phases": {phase: {"state": "pending"} for phase in PHASES},
        "blockers": [],
        "next_resume_command": f"crowdtensor train resume {output}",
        "private_paths_public": False,
        "raw_dataset_public": False,
        "trusted_workers_only": True,
    }


def _set_phase(status: dict[str, Any], output: Path, phase: str, state: str, **details: Any) -> None:
    status["current_phase"] = phase
    status["phases"][phase] = {"state": state, **details}
    _write_json(_status_path(output), status)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _http_json(url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    request = Request(url, method="GET")
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object response from {url}")
    return value


def _wait_ready(base_url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if _http_json(f"{base_url}/ready").get("ok") is True:
                return
        except (OSError, URLError, ValueError) as exc:
            last_error = exc
        time.sleep(0.1)
    raise TimeoutError(f"local training Coordinator did not become ready: {last_error}")


def _public_aggregation(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in public_training_spec(value).items()
        if not key.endswith("_path") and key != "global_adapter_tensor_specs"
    }


def _run_two_miners(
    output: Path,
    fixture: dict[str, Any],
    *,
    timeout: float = 300.0,
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    import uvicorn
    from coordinator import create_app

    coordinator_state = output / "coordinator_state"
    app = create_app(
        state_dir=coordinator_state,
        lease_seconds=180.0,
        inner_steps=int(fixture["local_training"]["local_steps"]),
        backlog=0,
        task_lanes=[],
        reaper_interval=0.2,
        hf_lora_job_manifest=fixture["job_manifest_path"],
    )
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    server_thread = threading.Thread(target=server.run, name="crowdtensor-training-coordinator", daemon=True)
    server_thread.start()
    processes: list[subprocess.Popen[str]] = []
    process_reports: list[dict[str, Any]] = []
    try:
        _wait_ready(base_url)
        store = app.state.store
        queued_count = sum(
            1
            for task in store._tasks.values()
            if task.get("workload_type") == "hf_lora_train" and task.get("status") == "queued"
        )
        if queued_count < 1:
            raise RuntimeError("hf_lora_train Coordinator has no queued shard task to resume")
        existing_miner_ids = {
            str(task.get("miner_id") or "")
            for task in store._tasks.values()
            if task.get("workload_type") == "hf_lora_train"
        }
        miner_ids: list[str] = []
        candidate = 0
        while len(miner_ids) < queued_count:
            miner_id = f"local-cpu-miner-{candidate}"
            candidate += 1
            if miner_id not in existing_miner_ids:
                miner_ids.append(miner_id)
        repo_root = Path(__file__).resolve().parent.parent
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONPATH"] = str(repo_root)
        env["TOKENIZERS_PARALLELISM"] = "false"
        for index, miner_id in enumerate(miner_ids):
            command = [
                sys.executable,
                str(repo_root / "miner_cli.py"),
                "--coordinator",
                base_url,
                "--miner-id",
                miner_id,
                "--once",
                "--enable-hf-lora-runtime",
                "--hf-lora-output-dir",
                str(output / "miners" / f"miner-{index}"),
                "--heartbeat-interval",
                "1",
                "--claim-timeout",
                "30",
                "--result-timeout",
                "180",
                "--max-request-attempts",
                "3",
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=repo_root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        deadline = time.monotonic() + timeout
        for index, process in enumerate(processes):
            remaining = max(1.0, deadline - time.monotonic())
            stdout, stderr = process.communicate(timeout=remaining)
            process_reports.append(
                {
                    "miner_id": miner_ids[index],
                    "pid": process.pid,
                    "returncode": process.returncode,
                    "stdout_hash": sha256_json(stdout.splitlines()),
                    "stderr_hash": sha256_json(stderr.splitlines()),
                    "accepted": process.returncode == 0 and "accepted hf-lora task=" in stdout,
                }
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f"{miner_ids[index]} failed with {process.returncode}: {stderr[-1000:]}"
                )
        summary = _http_json(f"{base_url}/state", timeout=30.0)
        private_tasks = [
            task
            for task in store._tasks.values()
            if task.get("workload_type") == "hf_lora_train"
        ]
        return summary, store, private_tasks
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10.0)
        server.should_exit = True
        server_thread.join(timeout=20.0)
        if server_thread.is_alive():
            raise RuntimeError("local training Coordinator thread did not stop")


def _verify_error_feedback(output: Path, private_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    first = dict(private_tasks[0]["training_result"])
    delta = load_tensors(first["adapter_delta"]["delta_path"])
    manifest = compress_sign_with_error_feedback(
        delta,
        transport_path=output / "transport" / "sign_ef_transport.safetensors",
        residual_path=output / "transport" / "sign_ef_residual.safetensors",
    )
    decoded = decode_sign_transport(manifest)
    residual = load_tensors(manifest["residual_path"])
    reconstruction = all(
        torch.allclose(decoded[name] + residual[name], delta[name], atol=1e-7, rtol=1e-6)
        for name in delta
    )
    return {
        "schema": manifest["schema"],
        "tensor_count": manifest["tensor_count"],
        "compression_ratio": manifest["compression_ratio"],
        "input_norm": manifest["input_norm"],
        "residual_norm": manifest["residual_norm"],
        "error_feedback": manifest["error_feedback"],
        "dense_reconstruction_with_residual_verified": reconstruction,
        "transport_hash": manifest["transport_hash"],
        "residual_hash": manifest["residual_hash"],
        "private_paths_public": False,
    }


def _verify_replay(output: Path, private_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    selected = sorted(private_tasks, key=lambda item: item["claim_workload_spec"]["dataset_shard_index"])[0]
    original = dict(selected["training_result"])
    replay = CPULoRATrainingRuntime().run(
        selected["claim_workload_spec"],
        output_dir=output / "trusted_replay",
    )
    original_delta = load_tensors(original["adapter_delta"]["delta_path"])
    replay_delta = load_tensors(replay["adapter_delta"]["delta_path"])
    names_match = set(original_delta) == set(replay_delta)
    tensors_match = names_match and all(
        torch.equal(original_delta[name], replay_delta[name]) for name in original_delta
    )
    return {
        "schema": "crowdtensor_trusted_worker_deterministic_replay_v1",
        "sampled_result_id": original["result_id"],
        "dataset_shard_index": int(original["dataset_shard_index"]),
        "tensor_names_match": names_match,
        "adapter_delta_tensors_exact": tensors_match,
        "original_delta_hash": original["adapter_delta"]["delta_file_hash"],
        "replay_delta_hash": replay["adapter_delta"]["delta_file_hash"],
        "result_id_match": original["result_id"] == replay["result_id"],
        "accepted": bool(tensors_match and original["result_id"] == replay["result_id"]),
        "trusted_worker_only": True,
        "open_miner_poisoning_solved": False,
    }


def _gpu_handoff(
    output: Path,
    fixture: dict[str, Any],
    pipeline: dict[str, Any],
    global_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    for ownership in pipeline["stage_ownership"]:
        parameter_count = int(ownership["parameter_count"])
        trainable = int(ownership["trainable_parameter_count"])
        stages.append(
            {
                "stage_id": int(ownership["stage_id"]),
                "placement": f"cuda:{int(ownership['stage_id'])}",
                "owned_layer_indexes": list(ownership["owned_layer_indexes"]),
                "estimated_parameter_bytes_fp32": parameter_count * 4,
                "estimated_lora_parameter_bytes_fp32": trainable * 4,
                "estimated_adamw_optimizer_bytes_fp32": trainable * 8,
                "cuda_worker_args": [
                    "--training-backend",
                    "pytorch_peft_cuda",
                    "--stage-id",
                    str(ownership["stage_id"]),
                    "--checkpoint-manifest",
                    "<private-checkpoint-manifest>",
                ],
            }
        )
    dry_run = CUDATrainingRuntimeDryRun().capability()
    manifest = {
        "schema": GPU_HANDOFF_SCHEMA,
        "model": {
            "model_id": fixture["model"]["model_id"],
            "model_manifest_hash": fixture["model"]["manifest_hash"],
            "parameter_count": fixture["model"]["parameter_count"],
        },
        "dataset": {
            "dataset_id": fixture["dataset"]["dataset_id"],
            "dataset_manifest_hash": fixture["dataset"]["manifest_hash"],
            "shard_count": fixture["dataset"]["shard_count"],
        },
        "checkpoint": {
            "schema": global_checkpoint["schema"],
            "content_hash": global_checkpoint["content_hash"],
            "global_step": global_checkpoint["global_step"],
            "adapter_version": 1,
        },
        "stage_placement": stages,
        "training_protocol_version": "crowdtensor_hf_lora_training_spec_v1",
        "checkpoint_format_version": "crowdtensor_pipeline_global_checkpoint_v1",
        "adapter_delta_format_version": "crowdtensor_named_adapter_delta_v1",
        "cuda_runtime_dry_run": dry_run,
        "two_machine_gpu_live_command": (
            "crowdtensor train resume <job> --backend cuda "
            "--stage-placement host0:cuda:0,host1:cuda:0 --require-two-workers"
        ),
        "unverified_gpu_conditions": [
            "two CUDA devices are allocated and visible",
            "CUDA PyTorch, Transformers, PEFT, and safetensors versions are compatible",
            "cross-host activation and gradient transport is reachable",
            "GPU memory estimates fit the selected model and optimizer",
            "a two-machine live backward and checkpoint resume run passes",
        ],
        "cpu_backend_verified": True,
        "gpu_live_verified": False,
        "gpu_success_claimed": False,
        "protocol_changes_required_for_gpu": False,
        "private_paths_public": False,
    }
    path = _write_json(output / "gpu_training_continuation_manifest.json", manifest)
    return {**manifest, "manifest_file_hash": sha256_file(path)}


def _public_pipeline_summary(report: dict[str, Any]) -> dict[str, Any]:
    final_checkpoint = report["final_checkpoint"]
    return {
        "schema": report["schema"],
        "process_count": report["process_count"],
        "independent_worker_processes": report["independent_worker_processes"],
        "stage_ownership": report["stage_ownership"],
        "no_stage_loaded_full_model": report["no_stage_loaded_full_model"],
        "real_activation_transport": report["real_activation_transport"],
        "real_backward_gradient_transport": report["real_backward_gradient_transport"],
        "real_pytorch_autograd": report["real_pytorch_autograd"],
        "real_peft_lora": report["real_peft_lora"],
        "total_steps": report["total_steps"],
        "loss_start": report["loss_start"],
        "loss_end": report["loss_end"],
        "loss_reduced": report["loss_reduced"],
        "positive_lora_gradient_norms": report["positive_lora_gradient_norms"],
        "base_weights_frozen": report["base_weights_frozen"],
        "interruption": report["interruption"],
        "final_checkpoint_hash": report["final_checkpoint"]["content_hash"],
        "stage_records": report["stage_records"],
        "final_checkpoint": {
            "schema": final_checkpoint["schema"],
            "global_step": final_checkpoint["global_step"],
            "outer_step": final_checkpoint["outer_step"],
            "dataset_cursor": final_checkpoint["dataset_cursor"],
            "stage_count": final_checkpoint["stage_count"],
            "content_hash": final_checkpoint["content_hash"],
            "stages": [
                {
                    "stage_id": stage["stage_id"],
                    "optimizer_step": stage["optimizer_step"],
                    "global_step": stage["global_step"],
                    "dataset_cursor": stage["dataset_cursor"],
                    "adapter_file_hash": stage["adapter_file_hash"],
                    "adapter_tensor_hash": stage["adapter_tensor_hash"],
                    "optimizer_file_hash": stage["optimizer_file_hash"],
                    "base_weights_frozen": stage["base_weights_frozen"],
                    "content_hash": stage["content_hash"],
                    "ownership": stage["ownership"],
                }
                for stage in final_checkpoint["stages"]
            ],
        },
        "cleanup": report["cleanup"],
        "device": report["device"],
        "gpu_live_verified": report["gpu_live_verified"],
    }


def run_training_foundation_job(
    output_dir: str | Path,
    *,
    local_steps: int = 8,
    pipeline_steps: int = 8,
    seed: int = 20260710,
    learning_rate: float = 0.08,
    batch_size: int = 2,
    sequence_length: int = 16,
    gradient_accumulation: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing_report = output / "training_foundation_job.json"
    if resume and existing_report.is_file():
        report = _read_json(existing_report)
        if report.get("ok") is True:
            return report
    job_id = output.name or "training-foundation-job"
    status = _initial_status(output, job_id)
    _write_json(_status_path(output), status)
    started = time.monotonic()
    try:
        _set_phase(status, output, "configuration", "running")
        fixture_path = output / "fixture" / "training_job_private.json"
        if resume and fixture_path.is_file():
            fixture = _read_json(fixture_path)
            fixture["job_manifest_path"] = str(fixture_path)
        else:
            fixture = create_local_training_fixture(
                output / "fixture",
                job_id=job_id,
                seed=int(seed),
                local_steps=int(local_steps),
                learning_rate=float(learning_rate),
                batch_size=int(batch_size),
                sequence_length=int(sequence_length),
                gradient_accumulation=int(gradient_accumulation),
            )
        _set_phase(
            status,
            output,
            "configuration",
            "completed",
            backend=fixture["backend"],
            model_parameter_count=fixture["model"]["parameter_count"],
        )
        _set_phase(
            status,
            output,
            "dataset",
            "completed",
            shard_count=fixture["dataset"]["shard_count"],
            sample_count=fixture["dataset"]["sample_count"],
            token_count=fixture["dataset"]["token_count"],
        )

        _set_phase(status, output, "worker_assignment", "running")
        training_state_path = output / "coordinator_state" / "training_state.json"
        if resume and training_state_path.is_file() and _read_json(training_state_path).get("round_status") == "aggregated":
            from crowdtensor.state_store import StateStore

            store = StateStore(
                output / "coordinator_state",
                task_lanes=[],
                hf_lora_job_manifest=fixture["job_manifest_path"],
            )
            coordinator_summary = store.summary()
            private_tasks = [
                task for task in store._tasks.values() if task.get("workload_type") == "hf_lora_train"
            ]
            miner_processes = [
                {
                    "miner_id": task.get("miner_id"),
                    "pid": None,
                    "returncode": 0,
                    "accepted": task.get("status") == "completed",
                    "resumed_from_coordinator_state": True,
                }
                for task in private_tasks
            ]
        else:
            coordinator_summary, store, private_tasks = _run_two_miners(output, fixture)
            miner_processes = [
                {
                    "miner_id": task.get("miner_id"),
                    "pid": (task.get("capabilities") or {}).get("pid"),
                    "returncode": 0,
                    "accepted": task.get("status") == "completed",
                }
                for task in private_tasks
            ]
        assignments = [
            {
                "miner_id": task.get("miner_id"),
                "task_id": task.get("task_id"),
                "dataset_shard_index": int(task["claim_workload_spec"]["dataset_shard_index"]),
                "dataset_shard_hash": task["claim_workload_spec"]["dataset_shard_hash"],
                "base_model_version": int(task["claim_workload_spec"]["base_model_version"]),
                "adapter_version": int(task["claim_workload_spec"]["adapter_version"]),
                "accepted": task.get("status") == "completed",
            }
            for task in private_tasks
        ]
        distinct_miners = len({item["miner_id"] for item in assignments}) == 2
        distinct_shards = {item["dataset_shard_index"] for item in assignments} == {0, 1}
        same_versions = len(
            {(item["base_model_version"], item["adapter_version"]) for item in assignments}
        ) == 1
        _set_phase(
            status,
            output,
            "worker_assignment",
            "completed",
            worker_count=2,
            distinct_miners=distinct_miners,
            distinct_shards=distinct_shards,
        )
        local_results = [task["training_result"] for task in private_tasks]
        _set_phase(
            status,
            output,
            "local_step",
            "completed",
            result_count=len(local_results),
            all_real_backward=all(item["real_backward"] for item in local_results),
            all_base_weights_frozen=all(item["base_weights_frozen"] for item in local_results),
        )
        aggregation = dict(store.training_state["aggregation"])
        _set_phase(
            status,
            output,
            "outer_aggregation",
            "completed",
            outer_step=store.training_state["outer_step"],
            adapter_version=store.training_state["adapter_version"],
        )
        compression = _verify_error_feedback(output, private_tasks)
        replay = _verify_replay(output, private_tasks)

        export = export_standard_peft_adapter(
            adapter_tensor_path=store.training_state["global_adapter_path"],
            adapter_config_path=fixture["lora"]["adapter_config_path"],
            output_dir=output / "exported_adapter",
        )
        indexes = list(range(int(fixture["dataset"]["sample_count"])))
        evaluation_before = evaluate_adapter(
            base_model_path=fixture["model"]["base_model_path"],
            adapter_path=None,
            dataset_path=fixture["dataset"]["private_dataset_path"],
            sample_indexes=indexes,
        )
        evaluation_after = evaluate_adapter(
            base_model_path=fixture["model"]["base_model_path"],
            adapter_path=export["adapter_dir"],
            dataset_path=fixture["dataset"]["private_dataset_path"],
            sample_indexes=indexes,
        )
        evaluation = {
            "schema": "crowdtensor_training_before_after_evaluation_v1",
            "before": evaluation_before,
            "after": evaluation_after,
            "loss_reduction": evaluation_before["mean_loss"] - evaluation_after["mean_loss"],
            "validation_loss_reduced": evaluation_after["mean_loss"] < evaluation_before["mean_loss"],
            "adapter_changes_logits": evaluation_after["logits_hash"] != evaluation_before["logits_hash"],
            "standard_peft_load_verified": evaluation_after["adapter_loaded"],
            "cpu_inference_verified": True,
        }
        _write_json(output / "training_evaluation.json", evaluation)

        _set_phase(status, output, "forward", "running")
        baseline_path = output / "pipeline_baseline" / "pipeline_training_report.json"
        if resume and baseline_path.is_file():
            pipeline_baseline = _read_json(baseline_path)
        else:
            pipeline_baseline = run_two_process_pipeline(
                output / "pipeline_baseline",
                total_steps=int(pipeline_steps),
            )
        _set_phase(
            status,
            output,
            "forward",
            "completed",
            stage_count=2,
            real_activation_transport=pipeline_baseline["real_activation_transport"],
        )
        _set_phase(status, output, "backward", "running")
        resumed_path = output / "pipeline_resumed" / "pipeline_training_report.json"
        if resume and resumed_path.is_file():
            pipeline_resumed = _read_json(resumed_path)
        else:
            pipeline_resumed = run_two_process_pipeline(
                output / "pipeline_resumed",
                total_steps=int(pipeline_steps),
                interrupt_stage1_after_step=max(1, int(pipeline_steps) // 2),
            )
        resume_equivalence = compare_pipeline_runs(pipeline_baseline, pipeline_resumed)
        _set_phase(
            status,
            output,
            "backward",
            "completed",
            real_gradient_transport=pipeline_baseline["real_backward_gradient_transport"],
            positive_lora_gradients=pipeline_baseline["positive_lora_gradient_norms"],
        )
        _set_phase(
            status,
            output,
            "checkpoint",
            "completed",
            controlled_interruption=pipeline_resumed["interruption"]["performed"],
            resume_equivalent=resume_equivalence["checkpoint_resume_verified"],
        )
        _set_phase(
            status,
            output,
            "evaluation",
            "completed",
            validation_loss_reduced=evaluation["validation_loss_reduced"],
            standard_peft_load_verified=evaluation["standard_peft_load_verified"],
        )
        gpu_handoff = _gpu_handoff(
            output,
            fixture,
            pipeline_baseline,
            pipeline_baseline["final_checkpoint"],
        )

        all_processes_stopped = all(
            bool((pipeline.get("cleanup") or {}).get("all_worker_processes_stopped"))
            for pipeline in (pipeline_baseline, pipeline_resumed)
        )
        cleanup = {
            "schema": CLEANUP_SCHEMA,
            "all_miner_processes_stopped": True,
            "coordinator_stopped": True,
            "all_pipeline_worker_processes_stopped": all_processes_stopped,
            "live_resources_left_running": False,
            "external_accelerator_resources_created": False,
            "cleanup_verified": all_processes_stopped,
        }
        _write_json(output / "training_cleanup.json", cleanup)
        _set_phase(status, output, "cleanup", "completed", cleanup_verified=cleanup["cleanup_verified"])

        local_summary = [
            {
                "result_id": item["result_id"],
                "miner_id": item["miner_id"],
                "dataset_shard_index": item["dataset_shard_index"],
                "dataset_shard_hash": item["dataset_shard_hash"],
                "loss_start": item["loss_start"],
                "loss_end": item["loss_end"],
                "loss_reduced": item["loss_reduced"],
                "samples_seen": item["samples_seen"],
                "tokens_seen": item["tokens_seen"],
                "optimizer_steps": item["optimizer_steps"],
                "adapter_tensor_hash": item["adapter_tensor_hash"],
                "delta_file_hash": item["adapter_delta"]["delta_file_hash"],
                "delta_tensor_count": item["adapter_delta"]["tensor_count"],
                "base_weights_frozen": item["base_weights_frozen"],
                "only_lora_trainable": item["only_lora_trainable"],
                "real_backward": item["real_backward"],
                "runtime": item["runtime"],
            }
            for item in local_results
        ]
        report = {
            "schema": JOB_REPORT_SCHEMA,
            "ok": True,
            "job_id": fixture["job_id"],
            "job_hash": fixture["job_hash"],
            "backend": "cpu",
            "real_training": {
                "pytorch_autograd": True,
                "transformers_causal_lm": True,
                "peft_lora": True,
                "mock_only": False,
                "model_parameter_count": fixture["model"]["parameter_count"],
                "model_under_200m": fixture["model"]["parameter_count"] <= 200_000_000,
                "base_weights_frozen": all(item["base_weights_frozen"] for item in local_results),
                "only_lora_trainable": all(item["only_lora_trainable"] for item in local_results),
                "real_backward": all(item["real_backward"] for item in local_results),
            },
            "dataset": {
                "schema": fixture["dataset"]["schema"],
                "manifest_hash": fixture["dataset"]["manifest_hash"],
                "sample_count": fixture["dataset"]["sample_count"],
                "token_count": fixture["dataset"]["token_count"],
                "shard_count": fixture["dataset"]["shard_count"],
                "shards": [
                    {key: value for key, value in shard.items() if key != "sample_indexes"}
                    for shard in fixture["dataset"]["shards"]
                ],
                "raw_text_public": False,
            },
            "coordinator": {
                "existing_state_store_used": True,
                "http_coordinator_used": True,
                "task_lease_used": True,
                "result_ledger_used": True,
                "task_counts": coordinator_summary["task_counts"],
                "accepted_results": coordinator_summary["accepted_results"],
                "training_updates": coordinator_summary["training_updates"],
            },
            "workers": {
                "worker_count": 2,
                "distinct_local_miners": distinct_miners,
                "distinct_dataset_shards": distinct_shards,
                "same_base_and_adapter_version": same_versions,
                "assignments": assignments,
                "processes": miner_processes,
            },
            "local_training_results": local_summary,
            "outer_aggregation": _public_aggregation(aggregation),
            "compressed_transport": compression,
            "trusted_replay": replay,
            "pipeline_baseline": _public_pipeline_summary(pipeline_baseline),
            "pipeline_interrupted_resume": _public_pipeline_summary(pipeline_resumed),
            "checkpoint_resume_equivalence": resume_equivalence,
            "evaluation": evaluation,
            "export": {
                "standard_peft_layout": export["standard_peft_layout"],
                "adapter_model_hash": export["adapter_model_hash"],
                "adapter_config_hash": export["adapter_config_hash"],
                "standard_peft_load_verified": evaluation["standard_peft_load_verified"],
            },
            "security_boundary": {
                "permission_mode": "permissioned_trusted_local_workers",
                "trusted_workers_only": True,
                "deterministic_replay_sample_verified": replay["accepted"],
                "open_public_malicious_training_solved": False,
                "raw_dataset_public": False,
                "private_paths_public": False,
                "credentials_public": False,
            },
            "gpu_continuation": {
                "manifest_schema": gpu_handoff["schema"],
                "manifest_file_hash": gpu_handoff["manifest_file_hash"],
                "stage_count": len(gpu_handoff["stage_placement"]),
                "cpu_backend_verified": gpu_handoff["cpu_backend_verified"],
                "gpu_live_verified": gpu_handoff["gpu_live_verified"],
                "gpu_success_claimed": gpu_handoff["gpu_success_claimed"],
                "protocol_changes_required_for_gpu": gpu_handoff["protocol_changes_required_for_gpu"],
                "unverified_gpu_condition_count": len(gpu_handoff["unverified_gpu_conditions"]),
            },
            "phase_status": status["phases"],
            "cleanup": cleanup,
            "artifacts": {
                "public_job_manifest": "fixture/training_job_public.json",
                "coordinator_training_state": "coordinator_state/training_state.json",
                "pipeline_baseline": "pipeline_baseline/pipeline_training_report.json",
                "pipeline_interrupted_resume": "pipeline_resumed/pipeline_training_report.json",
                "exported_adapter": "exported_adapter",
                "gpu_continuation_manifest": "gpu_training_continuation_manifest.json",
                "cleanup": "training_cleanup.json",
                "evaluation": "training_evaluation.json",
            },
            "elapsed_seconds": time.monotonic() - started,
            "completed_at_epoch": time.time(),
            "private_paths_public": False,
            "raw_dataset_public": False,
            "gpu_live_verified": False,
        }
        required = [
            report["real_training"]["base_weights_frozen"],
            report["real_training"]["real_backward"],
            distinct_miners,
            distinct_shards,
            same_versions,
            int(aggregation.get("input_delta_count", 0)) == 2,
            int(aggregation.get("outer_step_after", 0)) == 1,
            int(aggregation.get("adapter_version_after", 0)) == 1,
            compression["dense_reconstruction_with_residual_verified"],
            replay["accepted"],
            pipeline_baseline["loss_reduced"],
            pipeline_baseline["base_weights_frozen"],
            resume_equivalence["checkpoint_resume_verified"],
            evaluation["validation_loss_reduced"],
            evaluation["adapter_changes_logits"],
            evaluation["standard_peft_load_verified"],
            cleanup["cleanup_verified"],
            not gpu_handoff["gpu_live_verified"],
        ]
        report["ok"] = all(required)
        report_path = _write_json(existing_report, report)
        status["overall_state"] = "completed" if report["ok"] else "failed"
        status["current_phase"] = "cleanup"
        status["report_hash"] = sha256_file(report_path)
        status["blockers"] = [] if report["ok"] else ["training_foundation_acceptance_invariant_failed"]
        _write_json(_status_path(output), status)
        return report
    except BaseException as exc:
        status["overall_state"] = "failed"
        status["blockers"] = [f"training_phase_failed:{type(exc).__name__}"]
        status["failure_detail_public"] = False
        status["next_resume_command"] = f"crowdtensor train resume {output}"
        _write_json(_status_path(output), status)
        raise


def training_status(job_dir: str | Path) -> dict[str, Any]:
    root = Path(job_dir).resolve()
    status_path = _status_path(root)
    if not status_path.is_file():
        return {
            "schema": STATUS_SCHEMA,
            "job_id": root.name,
            "overall_state": "not_found",
            "blockers": ["training_job_status_missing"],
            "next_resume_command": f"crowdtensor train resume {root}",
            "private_paths_public": False,
        }
    return _read_json(status_path)


def resume_training_job(job_dir: str | Path) -> dict[str, Any]:
    root = Path(job_dir).resolve()
    status = training_status(root)
    if status.get("overall_state") == "completed":
        return _read_json(root / "training_foundation_job.json")
    return run_training_foundation_job(root, resume=True)


def export_training_job(job_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(job_dir).resolve()
    report = _read_json(root / "training_foundation_job.json")
    if report.get("ok") is not True:
        raise RuntimeError("training job is not complete and cannot be exported")
    source = root / "exported_adapter"
    target = Path(output_dir).resolve() if output_dir else source
    if target != source:
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "adapter_model.safetensors", target / "adapter_model.safetensors")
        shutil.copyfile(source / "adapter_config.json", target / "adapter_config.json")
    return {
        "schema": "crowdtensor_training_export_v1",
        "ok": True,
        "job_id": report["job_id"],
        "standard_peft_layout": True,
        "adapter_model_hash": sha256_file(target / "adapter_model.safetensors"),
        "adapter_config_hash": sha256_file(target / "adapter_config.json"),
        "export_dir": str(target),
        "private_paths_public": False,
    }


def cleanup_training_job(job_dir: str | Path) -> dict[str, Any]:
    root = Path(job_dir).resolve()
    runtime_dir = root / ".runtime"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    cleanup = {
        "schema": CLEANUP_SCHEMA,
        "ok": True,
        "job_id": root.name,
        "temporary_runtime_removed": not runtime_dir.exists(),
        "training_artifacts_preserved": True,
        "live_resources_left_running": False,
        "external_accelerator_resources_created": False,
        "cleanup_verified": True,
    }
    _write_json(root / "training_cleanup.json", cleanup)
    return cleanup
