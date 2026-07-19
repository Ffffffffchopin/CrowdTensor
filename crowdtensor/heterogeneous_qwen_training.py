"""Manifest-driven Qwen LoRA stage runtime for CPU and single/multi GPU Miners."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .heterogeneous_qwen_source import qwen_stage_spec
from .heterogeneous_training_checkpoint import (
    load_stage_checkpoint,
    save_stage_checkpoint,
)
from .heterogeneous_training_manifest import validate_training_manifest
from .qwen15b_training import (
    QwenStageTrainer,
    _qwen_trainable_grad_norm,
    load_qwen_pipeline_stage,
    qwen_stage_adapter_hash,
    qwen_stage_adapter_state,
    qwen_stage_base_hash,
    stable_hash,
)


STAGE_RUNTIME_SCHEMA = "crowdtensor_heterogeneous_qwen_stage_runtime_v1"
_SPAWN_START_LOCK = threading.Lock()


def _ensure_spawn_import_path() -> str:
    package_parent = str(Path(__file__).resolve().parent.parent)
    sys.path[:] = [
        package_parent,
        *(value for value in sys.path if value != package_parent),
    ]
    python_path = [
        value
        for value in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if value
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [package_parent, *(value for value in python_path if value != package_parent)]
    )
    return package_parent


def _spawn_process_with_pipe(
    target: Callable[..., Any], args: tuple[Any, ...]
) -> tuple[Any, Any, Any]:
    with _SPAWN_START_LOCK:
        _ensure_spawn_import_path()
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=target, args=(child, *args), daemon=False
        )
        process.start()
    return parent, child, process


def _tensor_hash(value: Any) -> str:
    import torch

    tensor = value.detach().to("cpu").contiguous()
    return "sha256:" + hashlib.sha256(
        tensor.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _public_error(exc: BaseException) -> str:
    text = str(exc)
    if "out of memory" in text.lower() or "cuda_oom" in text.lower():
        return "cuda_oom"
    for code in (
        "qwen15b_non_finite_stage_activation",
        "qwen15b_non_finite_stage_boundary_activation",
        "qwen15b_non_finite_logits",
        "qwen15b_non_finite_loss",
        "qwen15b_non_finite_activation_gradient",
        "qwen15b_non_finite_incoming_gradient",
        "qwen15b_non_finite_lora_gradient",
    ):
        if code in text:
            return code.replace("qwen15b_", "")
    if isinstance(exc, TimeoutError):
        return "stage_timeout"
    return "stage_runtime_failed:" + type(exc).__name__


class HeterogeneousQwenStageTrainer(QwenStageTrainer):
    """Qwen stage autograd with manifest-bound scheduler and checkpoint state."""

    def __init__(
        self,
        module: Any,
        spec: Any,
        *,
        training_manifest: dict[str, Any],
        placement_generation: int,
        device: str,
        checkpoint_dir: str | Path,
        resume: bool = False,
    ) -> None:
        import torch

        self.training_manifest = validate_training_manifest(training_manifest)
        self.placement_generation = int(placement_generation)
        lora = self.training_manifest["lora"]
        super().__init__(
            module,
            spec,
            device=device,
            checkpoint_dir=checkpoint_dir,
            learning_rate=float(lora["learning_rate"]),
            gradient_clip_norm=float(lora["gradient_clip_norm"]),
            grad_scaler_init_scale=128.0,
            resume=False,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda _step: 1.0
        )
        self.loaded_checkpoint = None
        if resume:
            self.loaded_checkpoint = load_stage_checkpoint(
                self.module,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.checkpoint_dir,
                training_manifest=self.training_manifest,
                stage_spec=self.spec,
                device=self.device,
            )

    def _boundary(self, value: Any) -> Any:
        import torch

        if value is None:
            return None
        dtype = getattr(
            torch, str(self.training_manifest["precision"]["boundary_dtype"])
        )
        result = value.detach().to("cpu", dtype=dtype).contiguous()
        if not bool(torch.isfinite(result.float()).all().item()):
            raise RuntimeError("heterogeneous_non_finite_stage_boundary")
        return result

    def forward(self, microbatch_id: int, value: Any) -> dict[str, Any]:
        result = super().forward(microbatch_id, value)
        boundary = self._boundary(result["activation"])
        return {
            **result,
            "activation": boundary,
            "shape": list(boundary.shape),
            "dtype": str(boundary.dtype).replace("torch.", ""),
            "activation_hash": _tensor_hash(boundary),
        }

    def loss_backward(
        self,
        microbatch_id: int,
        hidden_states: Any,
        labels: Any,
        *,
        microbatch_count: int,
    ) -> dict[str, Any]:
        result = super().loss_backward(
            microbatch_id,
            hidden_states,
            labels,
            microbatch_count=microbatch_count,
        )
        boundary = self._boundary(result["activation_gradient"])
        return {
            **result,
            "activation_gradient": boundary,
            "gradient_hash": _tensor_hash(boundary),
        }

    def backward(self, microbatch_id: int, activation_gradient: Any) -> dict[str, Any]:
        result = super().backward(microbatch_id, activation_gradient)
        boundary = self._boundary(result["activation_gradient"])
        return {
            **result,
            "activation_gradient": boundary,
            "outgoing_gradient_hash": _tensor_hash(boundary) if boundary is not None else "",
        }

    def abort_step(self) -> dict[str, Any]:
        self.cached_outputs.clear()
        self.cached_inputs.clear()
        self.optimizer.zero_grad(set_to_none=True)
        return {"aborted": True, "graphs_cleared": True}

    def finish_step(self, *, global_step: int, dataset_cursor: int) -> dict[str, Any]:
        import torch

        if self.cached_outputs or self.cached_inputs:
            raise RuntimeError("Qwen stage cannot step with unfinished microbatches")
        scale_before = float(self.scaler.get_scale())
        self.scaler.unscale_(self.optimizer)
        gradient_norm = _qwen_trainable_grad_norm(self.module)
        if not math.isfinite(gradient_norm):
            raise RuntimeError("qwen15b_non_finite_lora_gradient")
        torch.nn.utils.clip_grad_norm_(
            self.trainable_parameters, self.gradient_clip_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        checkpoint = save_stage_checkpoint(
            self.module,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.checkpoint_dir,
            training_manifest=self.training_manifest,
            stage_spec=self.spec,
            global_step=int(global_step),
            dataset_cursor=int(dataset_cursor),
            placement_generation=self.placement_generation,
            device=self.device,
        )
        target = torch.device(self.device)
        peak_allocated = (
            int(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else 0
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved(target)) if target.type == "cuda" else 0
        )
        return {
            "global_step": int(global_step),
            "dataset_cursor": int(dataset_cursor),
            "placement_generation": self.placement_generation,
            "gradient_scale_before": scale_before,
            "gradient_scale_after": float(self.scaler.get_scale()),
            "lora_gradient_norm": gradient_norm,
            "gradient_clip_norm": float(self.gradient_clip_norm),
            "gradient_clipping_applied": True,
            "optimizer_step_applied": True,
            "scheduler_step_applied": True,
            "scheduler_last_epoch": int(self.scheduler.last_epoch),
            "checkpoint_hash": checkpoint["content_hash"],
            "adapter_tensor_hash": checkpoint["adapter_tensor_hash"],
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        }


def _stage_process_main(connection: Any, settings: dict[str, Any]) -> None:
    phase = "bootstrap"
    try:
        import torch

        manifest = validate_training_manifest(settings["training_manifest"])
        spec = qwen_stage_spec(
            manifest,
            stage_id=int(settings["stage_id"]),
            device_index=int(settings.get("device_index") or 0),
        )
        device = str(settings["device"])
        target = torch.device(device)
        if target.type == "cuda":
            torch.cuda.set_device(target)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(target)
        torch.manual_seed(int(manifest["training"]["seed"]) + int(spec.stage_id))
        if target.type == "cuda":
            torch.cuda.manual_seed_all(
                int(manifest["training"]["seed"]) + int(spec.stage_id)
            )
        torch.use_deterministic_algorithms(True, warn_only=True)
        phase = "model_load"
        module, load_report = load_qwen_pipeline_stage(
            dict(settings["config"]),
            spec,
            settings["shard_path"],
            device=device,
            compute_dtype=manifest["precision"][
                "cuda_compute_dtype" if target.type == "cuda" else "cpu_compute_dtype"
            ],
            inject_lora=True,
            lora_rank=int(manifest["lora"]["rank"]),
            lora_alpha=int(manifest["lora"]["alpha"]),
            lora_target_modules=manifest["lora"]["target_modules"],
            lora_dropout=float(manifest["lora"]["dropout"]),
            lora_seed=int(manifest["training"]["seed"]),
            gradient_checkpointing=True,
            model_id=manifest["model"]["model_id"],
            model_revision=manifest["model"]["model_revision"],
        )
        module.train()
        phase = "trainer_init"
        trainer = HeterogeneousQwenStageTrainer(
            module,
            spec,
            training_manifest=manifest,
            placement_generation=int(settings["placement_generation"]),
            device=device,
            checkpoint_dir=settings["checkpoint_dir"],
            resume=bool(settings.get("resume")),
        )
        phase = "base_hash"
        base_hash = qwen_stage_base_hash(module)
        ready = {
            "type": "ready",
            "schema": STAGE_RUNTIME_SCHEMA,
            "stage_id": int(spec.stage_id),
            "pid": os.getpid(),
            "device": device,
            "device_type": target.type,
            "placement_generation": int(settings["placement_generation"]),
            "load_report": load_report,
            "base_hash_before": base_hash,
            "adapter_hash_before": qwen_stage_adapter_hash(module),
            "resumed": trainer.loaded_checkpoint is not None,
            "resumed_global_step": int(
                (trainer.loaded_checkpoint or {}).get("global_step", 0)
            ),
            "resumed_dataset_cursor": int(
                (trainer.loaded_checkpoint or {}).get("dataset_cursor", 0)
            ),
            "resumed_placement_generation": int(
                (trainer.loaded_checkpoint or {}).get("placement_generation", 0)
            ),
            "single_device_stage_process": True,
            "public_artifact_safe": True,
        }
        connection.send(ready)
        while True:
            request = connection.recv()
            request_id = int(request["request_id"])
            operation = str(request["operation"])
            phase = operation
            if operation == "begin_step":
                trainer.begin_step()
                result: Any = {"begun": True}
            elif operation == "forward":
                result = trainer.forward(
                    int(request["microbatch_id"]), request["value"]
                )
            elif operation == "loss_backward":
                result = trainer.loss_backward(
                    int(request["microbatch_id"]),
                    request["hidden_states"],
                    request["labels"],
                    microbatch_count=int(request["microbatch_count"]),
                )
            elif operation == "backward":
                result = trainer.backward(
                    int(request["microbatch_id"]), request["activation_gradient"]
                )
            elif operation == "finish_step":
                result = trainer.finish_step(
                    global_step=int(request["global_step"]),
                    dataset_cursor=int(request["dataset_cursor"]),
                )
            elif operation == "abort_step":
                result = trainer.abort_step()
            elif operation == "adapter_state":
                result = {
                    "adapter_state": qwen_stage_adapter_state(module),
                    "adapter_hash": qwen_stage_adapter_hash(module),
                }
            elif operation == "status":
                result = {
                    "base_hash_before": base_hash,
                    "base_hash_after": qwen_stage_base_hash(module),
                    "adapter_hash": qwen_stage_adapter_hash(module),
                    "compute_intervals": list(trainer.compute_intervals),
                    "placement_generation": trainer.placement_generation,
                }
            elif operation == "stop":
                result = {"stopped": True, "stage_id": int(spec.stage_id)}
                connection.send(
                    {"request_id": request_id, "ok": True, "result": result}
                )
                break
            else:
                raise ValueError("heterogeneous_stage_operation_invalid")
            connection.send(
                {"request_id": request_id, "ok": True, "result": result}
            )
    except BaseException as exc:
        try:
            connection.send(
                {
                    "type": "error",
                    "ok": False,
                    "error_class": type(exc).__name__,
                    "error_code": _public_error(exc),
                    "error_phase": phase,
                }
            )
        except BaseException:
            pass
        raise
    finally:
        connection.close()


class HeterogeneousStageProcessClient:
    def __init__(
        self,
        *,
        training_manifest: dict[str, Any],
        config: dict[str, Any],
        stage_id: int,
        shard_path: str | Path,
        checkpoint_dir: str | Path,
        device: str,
        placement_generation: int,
        resume: bool,
        ready_timeout: float = 1800.0,
        keepalive: Callable[[], Any] | None = None,
        keepalive_interval_seconds: float = 5.0,
        require_tpu: bool = True,
        expected_tpu_devices: int = 8,
    ) -> None:
        target = str(device)
        device_index = int(target.split(":", 1)[1]) if target.startswith("cuda:") else 0
        settings = {
            "training_manifest": validate_training_manifest(training_manifest),
            "config": dict(config),
            "stage_id": int(stage_id),
            "shard_path": str(Path(shard_path).resolve()),
            "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
            "device": target,
            "device_index": device_index,
            "placement_generation": int(placement_generation),
            "resume": bool(resume),
            "require_tpu": bool(require_tpu),
            "expected_tpu_devices": int(expected_tpu_devices),
        }
        process_target = _stage_process_main
        if target.startswith("jax_tpu:"):
            from .heterogeneous_jax_qwen_training import jax_stage_process_main

            process_target = jax_stage_process_main
        parent, child, process = _spawn_process_with_pipe(
            process_target, (settings,)
        )
        child.close()
        deadline = time.monotonic() + float(ready_timeout)
        ready_available = False
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if parent.poll(
                min(max(0.1, float(keepalive_interval_seconds)), remaining)
            ):
                ready_available = True
                break
            if keepalive is not None:
                keepalive()
        if not ready_available:
            process.terminate()
            process.join(timeout=30.0)
            raise TimeoutError("heterogeneous_stage_start_timeout")
        ready = parent.recv()
        if ready.get("type") != "ready":
            process.join(timeout=5.0)
            raise RuntimeError(
                "heterogeneous_stage_start_failed:"
                f"{ready.get('error_phase', 'unknown')}:"
                f"{ready.get('error_code', 'unknown')}"
            )
        self.stage_id = int(stage_id)
        self.device = target
        self.placement_generation = int(placement_generation)
        self.connection = parent
        self.process = process
        self.ready = dict(ready)
        self._request_id = 0
        self._keepalive = keepalive
        self._keepalive_interval_seconds = max(
            0.1, float(keepalive_interval_seconds)
        )

    def call(self, operation: str, *, timeout: float = 1800.0, **payload: Any) -> Any:
        self._request_id += 1
        request = {
            "request_id": self._request_id,
            "operation": str(operation),
            **payload,
        }
        self.connection.send(request)
        deadline = time.monotonic() + float(timeout)
        response_available = False
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if self.connection.poll(
                min(self._keepalive_interval_seconds, remaining)
            ):
                response_available = True
                break
            if self._keepalive is not None:
                self._keepalive()
        if not response_available:
            raise TimeoutError(f"heterogeneous_stage_{operation}_timeout")
        response = self.connection.recv()
        if response.get("ok") is not True or int(response.get("request_id", -1)) != self._request_id:
            raise RuntimeError(
                "heterogeneous_stage_request_failed:"
                f"{response.get('error_phase', operation)}:"
                f"{response.get('error_code', 'unknown')}"
            )
        return response["result"]

    def stop(self, *, timeout: float = 300.0) -> dict[str, Any]:
        result = self.call("stop", timeout=timeout)
        self.process.join(timeout=30.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=30.0)
        self.connection.close()
        return dict(result)

    def force_stop(self) -> bool:
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=30.0)
        self.connection.close()
        return not self.process.is_alive()

    def public_ready(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.ready.items()
            if key not in {"adapter_state"}
        }
