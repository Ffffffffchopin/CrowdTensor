"""Two-Kernel, four-stage Qwen 1.5B pipeline training runtime."""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing as mp
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

from .qwen15b_training import (
    MODEL_ID,
    MODEL_REVISION,
    QwenStageSpec,
    QwenStageTrainer,
    assemble_qwen_standard_peft_state,
    canonical_stage_specs,
    load_qwen_pipeline_stage,
    qwen_stage_adapter_hash,
    qwen_stage_adapter_state,
    qwen_stage_base_hash,
    stable_hash,
)


RUNTIME_SCHEMA = "crowdtensor_qwen15b_four_gpu_runtime_v1"
DEFAULT_STEPS = 8
DEFAULT_MICROBATCHES = 4


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def serialize_private_tensors(values: dict[str, Any]) -> bytes:
    from safetensors.torch import save

    tensors = {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in values.items()
    }
    if not tensors:
        raise ValueError("private tensor payload cannot be empty")
    return save(tensors)


def deserialize_private_tensors(payload: bytes) -> dict[str, Any]:
    from safetensors.torch import load

    values = load(payload)
    if not values:
        raise ValueError("private tensor payload decoded to no tensors")
    return values


def _stage_error_code(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "worker_timeout"
    text = str(exc).lower()
    classifications = (
        ("incompatible version of torchao", "incompatible_torchao_pre_0_16"),
        ("out of memory", "cuda_out_of_memory"),
        ("cannot re-initialize cuda", "cuda_process_initialization_failed"),
        ("stage-selective range response length mismatch", "hf_range_length_mismatch"),
        ("ignored byte range", "hf_range_ignored"),
        ("source assignment mismatch", "stage_source_assignment_mismatch"),
        ("retained meta parameters", "stage_meta_parameter_unmaterialized"),
        ("strict source load", "stage_strict_source_load_failed"),
        ("non-lora trainable", "stage_lora_injection_invalid"),
        ("qwen15b_non_finite_stage_activation", "non_finite_stage_activation"),
        (
            "qwen15b_non_finite_stage_boundary_activation",
            "non_finite_stage_boundary_activation",
        ),
        ("qwen15b_non_finite_logits", "non_finite_logits"),
        ("qwen15b_non_finite_loss", "non_finite_loss"),
        ("qwen15b_non_finite_activation_gradient", "non_finite_activation_gradient"),
        ("qwen15b_non_finite_incoming_gradient", "non_finite_incoming_gradient"),
        ("qwen15b_non_finite_lora_gradient", "non_finite_lora_gradient"),
        ("attempting to unscale fp16 gradients", "grad_scaler_fp16_gradient_rejected"),
    )
    for fragment, code in classifications:
        if fragment in text:
            return code
    return f"{type(exc).__name__.lower()}_{hashlib.sha256(str(exc).encode('utf-8')).hexdigest()[:12]}"


class QwenHTTPTransport:
    """Authenticated HTTP transport whose public methods never return its URL/token."""

    def __init__(
        self,
        *,
        coordinator_url: str,
        token: str,
        run_id: str,
        timeout: float = 30.0,
        retry_attempts: int = 12,
        retry_base_seconds: float = 0.5,
        retry_max_seconds: float = 5.0,
    ) -> None:
        self._url = str(coordinator_url).rstrip("/")
        self._token = str(token)
        self._run_id = str(run_id)
        self._timeout = float(timeout)
        self._retry_attempts = int(retry_attempts)
        self._retry_base_seconds = float(retry_base_seconds)
        self._retry_max_seconds = float(retry_max_seconds)
        if self._retry_attempts < 1:
            raise ValueError("qwen15b_transport_retry_attempts_must_be_positive")
        self._last_registration_payload: dict[str, Any] | None = None
        self._reconnect_required = False
        self._retry_count = 0
        self._reconnect_registration_count = 0
        self._transient_error_classes: list[str] = []
        self._retry_sleep_seconds = 0.0

    def _request_once(
        self,
        path: str,
        *,
        method: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        data = None
        headers = {
            "User-Agent": "crowdtensor-qwen15b-four-gpu/1",
            "x-crowdtensor-miner-token": self._token,
        }
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self._url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("qwen15b_coordinator_response_invalid")
        return value

    @staticmethod
    def _retryable(exc: BaseException) -> bool:
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code == 429 or exc.code >= 500
        return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, OSError))

    @staticmethod
    def _terminal_error(exc: BaseException) -> RuntimeError:
        if isinstance(exc, urllib.error.HTTPError):
            return RuntimeError(f"qwen15b_coordinator_http_{exc.code}")
        if isinstance(exc, RuntimeError) and str(exc).startswith("qwen15b_"):
            return exc
        return RuntimeError("qwen15b_coordinator_transport_unavailable")

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        not_found_ok: bool = False,
    ) -> dict[str, Any] | None:
        last_error: BaseException | None = None
        for attempt in range(self._retry_attempts):
            try:
                if (
                    self._reconnect_required
                    and path != "/qwen15b-training/register"
                    and self._last_registration_payload is not None
                ):
                    self._request_once(
                        "/qwen15b-training/register",
                        method="POST",
                        payload=self._last_registration_payload,
                    )
                    self._reconnect_registration_count += 1
                    self._reconnect_required = False
                return self._request_once(path, method=method, payload=payload)
            except urllib.error.HTTPError as exc:
                if not_found_ok and exc.code == 404:
                    return None
                last_error = exc
            except BaseException as exc:
                last_error = exc
            assert last_error is not None
            if not self._retryable(last_error) or attempt + 1 >= self._retry_attempts:
                raise self._terminal_error(last_error) from last_error
            self._retry_count += 1
            self._reconnect_required = True
            self._transient_error_classes.append(type(last_error).__name__)
            delay = min(
                self._retry_max_seconds,
                self._retry_base_seconds * (2 ** min(attempt, 5)),
            )
            self._retry_sleep_seconds += delay
            time.sleep(delay)
        raise self._terminal_error(last_error or RuntimeError("unknown"))

    def register(self, *, role: str, ready: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "run_id": self._run_id,
            "role": role,
            "worker_id_hash": stable_hash(
                {"role": role, "pids": [int(item["pid"]) for item in ready]}
            ),
            "stage_ids": [int(item["stage_id"]) for item in ready],
            "stage_pids": [int(item["pid"]) for item in ready],
            "cuda_devices": [str(item["device"]) for item in ready],
            "cuda_device_name_hashes": [
                str(item["cuda_device_name_hash"]) for item in ready
            ],
            "cuda_live": all(item.get("cuda_live") is True for item in ready),
        }
        self._last_registration_payload = dict(payload)
        return dict(
            self._request(
                "/qwen15b-training/register",
                method="POST",
                payload=payload,
            )
            or {}
        )

    def status(self) -> dict[str, Any]:
        return dict(self._request("/qwen15b-training/status") or {})

    def wait_roles(self, *, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            status = self.status()
            if set(status.get("registered_roles") or []) == {"kernel_a", "kernel_b"}:
                return status
            time.sleep(1.0)
        raise TimeoutError("qwen15b_training_peer_registration_timeout")

    def wait_generation(self, *, minimum: int, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            status = self.status()
            if int(status.get("coordinator_generation") or 0) >= int(minimum):
                return status
            time.sleep(0.1)
        raise TimeoutError("qwen15b_training_coordinator_restart_timeout")

    def public_retry_report(self) -> dict[str, Any]:
        return {
            "schema": "crowdtensor_qwen15b_transport_reliability_v1",
            "bounded_retry_enabled": True,
            "retry_attempt_limit": self._retry_attempts,
            "retry_count": self._retry_count,
            "reconnect_registration_count": self._reconnect_registration_count,
            "retry_sleep_seconds": round(self._retry_sleep_seconds, 6),
            "transient_error_classes": sorted(set(self._transient_error_classes)),
            "retryable_http_statuses": [429, "5xx"],
            "coordinator_url_public": False,
            "coordinator_token_public": False,
            "public_artifact_safe": True,
        }

    def put_tensors(
        self,
        *,
        role: str,
        run_kind: str,
        kind: str,
        step: int,
        microbatch: int,
        tensors: dict[str, Any],
    ) -> dict[str, Any]:
        raw = serialize_private_tensors(tensors)
        result = self._request(
            "/qwen15b-training/payload",
            method="POST",
            payload={
                "run_id": self._run_id,
                "run_kind": run_kind,
                "kind": kind,
                "step": int(step),
                "microbatch": int(microbatch),
                "producer_role": role,
                "payload_b64": base64.b64encode(raw).decode("ascii"),
                "payload_hash": sha256_bytes(raw),
                "tensor_count": len(tensors),
            },
        )
        return dict(result or {})

    def get_tensors(
        self,
        *,
        run_kind: str,
        kind: str,
        step: int,
        microbatch: int,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        query = urllib.parse.urlencode({"run_id": self._run_id})
        value = self._request(
            f"/qwen15b-training/payload/{run_kind}/{kind}/{int(step)}/{int(microbatch)}?{query}",
            not_found_ok=True,
        )
        if value is None:
            return None
        encoded = str(value.get("payload_b64") or "")
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
        if sha256_bytes(raw) != str(value.get("payload_hash") or ""):
            raise RuntimeError("qwen15b_training_private_payload_hash_mismatch")
        return deserialize_private_tensors(raw), {
            "payload_hash": value["payload_hash"],
            "byte_count": int(value.get("byte_count") or 0),
            "tensor_count": int(value.get("tensor_count") or 0),
        }

    def event(
        self,
        *,
        role: str,
        run_kind: str,
        operation: str,
        stage_id: int,
        step: int = -1,
        microbatch: int = -1,
        pid: int = 0,
        device: str = "",
        started_ns: int = 0,
        ended_ns: int = 0,
        loss: float | None = None,
        gradient_norm: float | None = None,
        checkpoint_hash: str = "",
        adapter_hash: str = "",
    ) -> dict[str, Any]:
        value = self._request(
            "/qwen15b-training/event",
            method="POST",
            payload={
                "run_id": self._run_id,
                "role": role,
                "run_kind": run_kind,
                "operation": operation,
                "stage_id": int(stage_id),
                "step": int(step),
                "microbatch": int(microbatch),
                "pid": int(pid),
                "device": str(device),
                "started_ns": int(started_ns),
                "ended_ns": int(ended_ns),
                "loss": loss,
                "gradient_norm": gradient_norm,
                "checkpoint_hash": checkpoint_hash,
                "adapter_hash": adapter_hash,
            },
        )
        return dict(value or {})

    def complete(self, *, role: str, summary: dict[str, Any]) -> dict[str, Any]:
        return dict(
            self._request(
                "/qwen15b-training/complete",
                method="POST",
                payload={"run_id": self._run_id, "role": role, "summary": summary},
            )
            or {}
        )


def _stage_process_main(connection: Any, settings: dict[str, Any]) -> None:
    phase = "process_bootstrap"
    try:
        import torch

        phase = "cuda_process_setup"
        spec = QwenStageSpec(**dict(settings["spec"]))
        device = str(settings["device"])
        if torch.device(device).type == "cuda":
            torch.cuda.set_device(torch.device(device))
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(int(settings["seed"]))
        torch.use_deterministic_algorithms(True, warn_only=True)
        phase = "stage_model_load"
        module, load_report = load_qwen_pipeline_stage(
            dict(settings["config"]),
            spec,
            settings["shard_path"],
            device=device,
            compute_dtype=settings.get("compute_dtype"),
            inject_lora=True,
            lora_rank=int(settings["lora_rank"]),
            lora_alpha=int(settings["lora_alpha"]),
            lora_seed=int(settings["seed"]),
            gradient_checkpointing=True,
            model_id=str(settings.get("model_id") or MODEL_ID),
            model_revision=str(settings.get("model_revision") or MODEL_REVISION),
        )
        module.train()
        phase = "stage_trainer_init"
        trainer = QwenStageTrainer(
            module,
            spec,
            device=device,
            checkpoint_dir=settings["checkpoint_dir"],
            learning_rate=float(settings["learning_rate"]),
            gradient_clip_norm=float(settings["gradient_clip_norm"]),
            grad_scaler_init_scale=float(settings["grad_scaler_init_scale"]),
            resume=bool(settings.get("resume")),
            model_id=str(settings.get("model_id") or MODEL_ID),
            model_revision=str(settings.get("model_revision") or MODEL_REVISION),
        )
        phase = "stage_base_hash"
        base_hash = qwen_stage_base_hash(module)
        phase = "stage_ready"
        device_name = torch.cuda.get_device_name(torch.device(device)) if torch.cuda.is_available() else "cpu"
        ready = {
            "type": "ready",
            "stage_id": int(spec.stage_id),
            "pid": os.getpid(),
            "device": device,
            "cuda_live": bool(torch.cuda.is_available() and torch.device(device).type == "cuda"),
            "cuda_device_name_hash": stable_hash({"device_name": device_name}),
            "load_report": load_report,
            "base_hash_before": base_hash,
            "resumed": trainer.loaded_checkpoint is not None,
            "resumed_global_step": int((trainer.loaded_checkpoint or {}).get("global_step", 0)),
            "resumed_dataset_cursor": int(
                (trainer.loaded_checkpoint or {}).get("dataset_cursor", 0)
            ),
            "loaded_checkpoint_hash": str(
                (trainer.loaded_checkpoint or {}).get("content_hash") or ""
            ),
            "grad_scaler_enabled": bool(trainer.scaler.is_enabled()),
        }
        connection.send(ready)
        while True:
            request = connection.recv()
            request_id = int(request["request_id"])
            operation = str(request["operation"])
            phase = f"stage_{operation}"
            if operation == "begin_step":
                trainer.begin_step()
                result: Any = {"begun": True}
            elif operation == "forward":
                result = trainer.forward(int(request["microbatch_id"]), request["value"])
            elif operation == "loss_backward":
                result = trainer.loss_backward(
                    int(request["microbatch_id"]),
                    request["hidden_states"],
                    request["labels"],
                    microbatch_count=int(request["microbatch_count"]),
                )
            elif operation == "backward":
                result = trainer.backward(
                    int(request["microbatch_id"]),
                    request["activation_gradient"],
                )
            elif operation == "finish_step":
                result = trainer.finish_step(
                    global_step=int(request["global_step"]),
                    dataset_cursor=int(request["dataset_cursor"]),
                )
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
                }
            elif operation == "stop":
                result = {
                    "stopped": True,
                    "stage_id": int(spec.stage_id),
                }
                connection.send({"request_id": request_id, "ok": True, "result": result})
                break
            else:
                raise ValueError("qwen15b_stage_process_operation_invalid")
            connection.send({"request_id": request_id, "ok": True, "result": result})
    except BaseException as exc:
        try:
            connection.send(
                {
                    "type": "error",
                    "ok": False,
                    "error_class": type(exc).__name__,
                    "error_code": _stage_error_code(exc),
                    "error_phase": phase,
                }
            )
        except BaseException:
            pass
        raise
    finally:
        connection.close()


class StageProcessClient:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        spec: QwenStageSpec,
        shard_path: str | Path,
        checkpoint_dir: str | Path,
        resume: bool,
        seed: int,
        learning_rate: float,
        lora_rank: int,
        lora_alpha: int,
        ready_timeout: float = 900.0,
        device_override: str | None = None,
        model_id: str = MODEL_ID,
        model_revision: str = MODEL_REVISION,
    ) -> None:
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        execution_device = str(device_override or spec.device)
        settings = {
            "config": dict(config),
            "spec": asdict(spec),
            "shard_path": str(Path(shard_path).resolve()),
            "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
            "device": execution_device,
            "compute_dtype": "float32",
            "resume": bool(resume),
            "seed": int(seed),
            "learning_rate": float(learning_rate),
            "gradient_clip_norm": 1.0,
            "grad_scaler_init_scale": 128.0,
            "lora_rank": int(lora_rank),
            "lora_alpha": int(lora_alpha),
            "model_id": str(model_id),
            "model_revision": str(model_revision),
        }
        process = context.Process(target=_stage_process_main, args=(child, settings), daemon=False)
        process.start()
        child.close()
        if not parent.poll(float(ready_timeout)):
            process.terminate()
            process.join(timeout=30.0)
            raise TimeoutError(f"Qwen stage {spec.stage_id} startup timed out")
        ready = parent.recv()
        if ready.get("type") != "ready":
            process.join(timeout=5.0)
            phase = str(ready.get("error_phase") or "unknown")
            code = str(ready.get("error_code") or "unknown")
            raise RuntimeError(
                f"qwen15b_stage_startup_failed:{phase}:{code}:stage{spec.stage_id}"
            )
        self.spec = spec
        self.device = execution_device
        self.connection = parent
        self.process = process
        self.ready = dict(ready)
        self.busy: dict[str, Any] | None = None
        self._request_id = 0

    @property
    def pid(self) -> int:
        return int(self.ready["pid"])

    def send(self, operation: str, **payload: Any) -> int:
        if self.busy is not None:
            raise RuntimeError(f"Qwen stage {self.spec.stage_id} is already busy")
        self._request_id += 1
        request = {"request_id": self._request_id, "operation": operation, **payload}
        self.connection.send(request)
        self.busy = {"request_id": self._request_id, "operation": operation, **payload}
        return self._request_id

    def poll(self, timeout: float = 0.0) -> bool:
        return self.connection.poll(float(timeout))

    def recv(self) -> tuple[dict[str, Any], Any]:
        if self.busy is None:
            raise RuntimeError(f"Qwen stage {self.spec.stage_id} has no outstanding request")
        request = self.busy
        response = self.connection.recv()
        self.busy = None
        if response.get("ok") is not True or int(response.get("request_id", -1)) != int(
            request["request_id"]
        ):
            phase = str(response.get("error_phase") or request.get("operation") or "unknown")
            code = str(response.get("error_code") or "unknown")
            raise RuntimeError(
                f"qwen15b_stage_request_failed:{phase}:{code}:stage{self.spec.stage_id}"
            )
        return request, response["result"]

    def call(self, operation: str, *, timeout: float = 900.0, **payload: Any) -> Any:
        self.send(operation, **payload)
        if not self.poll(timeout):
            raise TimeoutError(f"Qwen stage {self.spec.stage_id} {operation} timed out")
        return self.recv()[1]

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


def _event_for_result(
    transport: QwenHTTPTransport,
    *,
    role: str,
    run_kind: str,
    client: StageProcessClient,
    operation: str,
    step: int,
    microbatch: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    interval = dict(result.get("compute_interval") or {})
    public = {
        "run_kind": run_kind,
        "operation": operation,
        "stage_id": int(client.spec.stage_id),
        "step": int(step),
        "microbatch": int(microbatch),
        "pid": int(client.pid),
        "device": client.device,
        "started_ns": int(interval.get("started_ns") or 0),
        "ended_ns": int(interval.get("ended_ns") or 0),
        "loss": float(result["loss"]) if result.get("loss") is not None else None,
    }
    transport.event(role=role, **public)
    return public


def _begin_step(clients: Iterable[StageProcessClient]) -> None:
    values = list(clients)
    for client in values:
        client.send("begin_step")
    for client in values:
        if not client.poll(120.0):
            raise TimeoutError("Qwen stage begin_step timed out")
        client.recv()


def _finish_step(
    clients: Iterable[StageProcessClient],
    *,
    transport: QwenHTTPTransport,
    role: str,
    run_kind: str,
    global_step: int,
    dataset_cursor: int,
) -> list[dict[str, Any]]:
    values = list(clients)
    for client in values:
        client.send(
            "finish_step",
            global_step=int(global_step),
            dataset_cursor=int(dataset_cursor),
        )
    reports = []
    for client in values:
        if not client.poll(300.0):
            raise TimeoutError("Qwen stage optimizer/checkpoint timed out")
        _request, result = client.recv()
        public = {
            "stage_id": int(client.spec.stage_id),
            "pid": int(client.pid),
            "device": client.device,
            **dict(result),
        }
        reports.append(public)
        transport.event(
            role=role,
            run_kind=run_kind,
            operation="optimizer_step",
            stage_id=client.spec.stage_id,
            step=global_step,
            pid=client.pid,
            device=client.device,
            gradient_norm=float(result.get("lora_gradient_norm") or 0.0),
            checkpoint_hash=str(result.get("checkpoint_hash") or ""),
            adapter_hash=str(result.get("adapter_tensor_hash") or ""),
        )
    return reports


def _restart_pair_from_checkpoint(
    clients: list[StageProcessClient],
    *,
    replacement_factory: Callable[[], list[StageProcessClient]],
    transport: QwenHTTPTransport,
    role: str,
    run_kind: str,
    after_step: int,
    dataset_cursor: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old_clients = list(clients)
    stop_results = []
    for client in old_clients:
        stopped = client.force_stop()
        stop_results.append(stopped)
        transport.event(
            role=role,
            run_kind=run_kind,
            operation="stage_stopped",
            stage_id=client.spec.stage_id,
            step=after_step,
            pid=client.pid,
            device=client.device,
        )
    replacements = replacement_factory()
    if len(replacements) != len(old_clients):
        for client in replacements:
            client.force_stop()
        raise RuntimeError("qwen15b_checkpoint_pair_restart_count_invalid")
    records = []
    public_events = []
    for old, replacement, stopped in zip(old_clients, replacements, stop_results):
        ready = replacement.ready
        if (
            replacement.spec.stage_id != old.spec.stage_id
            or replacement.pid == old.pid
            or ready.get("resumed") is not True
            or int(ready.get("resumed_global_step") or 0) != int(after_step)
            or int(ready.get("resumed_dataset_cursor") or 0) != int(dataset_cursor)
            or not str(ready.get("loaded_checkpoint_hash") or "").startswith("sha256:")
        ):
            for client in replacements:
                client.force_stop()
            raise RuntimeError("qwen15b_checkpoint_pair_restart_contract_invalid")
        record = {
            "stage_id": int(old.spec.stage_id),
            "after_step": int(after_step),
            "old_pid": int(old.pid),
            "new_pid": int(replacement.pid),
            "new_pid_verified": replacement.pid != old.pid,
            "forced_stop_verified": bool(stopped),
            "checkpoint_resume_verified": True,
            "resumed_global_step": int(ready["resumed_global_step"]),
            "resumed_dataset_cursor": int(ready["resumed_dataset_cursor"]),
            "loaded_checkpoint_hash": str(ready["loaded_checkpoint_hash"]),
        }
        records.append(record)
        public_events.extend(
            [
                {
                    "run_kind": run_kind,
                    "operation": "stage_stopped",
                    "stage_id": int(old.spec.stage_id),
                    "step": int(after_step),
                    "microbatch": -1,
                    "pid": int(old.pid),
                    "device": old.device,
                    "started_ns": 0,
                    "ended_ns": 0,
                    "loss": None,
                },
                {
                    "run_kind": run_kind,
                    "operation": "stage_restarted",
                    "stage_id": int(replacement.spec.stage_id),
                    "step": int(after_step),
                    "microbatch": -1,
                    "pid": int(replacement.pid),
                    "device": replacement.device,
                    "started_ns": 0,
                    "ended_ns": 0,
                    "loss": None,
                },
            ]
        )
    clients[:] = replacements
    transport.register(role=role, ready=[client.ready for client in replacements])
    for replacement in replacements:
        transport.event(
            role=role,
            run_kind=run_kind,
            operation="stage_restarted",
            stage_id=replacement.spec.stage_id,
            step=after_step,
            pid=replacement.pid,
            device=replacement.device,
            checkpoint_hash=str(replacement.ready.get("loaded_checkpoint_hash") or ""),
        )
    return records, public_events


def _training_row_inputs_and_labels(row: Any) -> tuple[list[int], list[int]]:
    if isinstance(row, dict):
        input_ids = list(row.get("input_ids") or [])
        labels = list(row.get("labels") or [])
    else:
        input_ids = list(row or [])
        labels = list(input_ids)
    if not input_ids or len(input_ids) != len(labels):
        raise ValueError("Qwen training row input/label shape mismatch")
    return [int(value) for value in input_ids], [int(value) for value in labels]


def run_kernel_a_once(
    *,
    run_kind: str,
    clients: list[StageProcessClient],
    transport: QwenHTTPTransport,
    train_rows: list[Any],
    steps: int = DEFAULT_STEPS,
    start_step: int = 0,
    microbatch_count: int = DEFAULT_MICROBATCHES,
    wait_timeout: float = 900.0,
    step_commit_callback: Callable[
        [int, int, list[dict[str, Any]]], dict[str, Any]
    ]
    | None = None,
    restart_pair_after_step: int = 0,
    restart_pair_factory: Callable[[], list[StageProcessClient]] | None = None,
    restart_generation: int = 1,
) -> dict[str, Any]:
    """Drive stages 0/1 while overlapping local forward/backward with Kernel B."""

    import torch

    stage0, stage1 = clients
    if [stage0.spec.stage_id, stage1.spec.stage_id] != [0, 1]:
        raise ValueError("Kernel A must own Qwen stages 0 and 1")
    events: list[dict[str, Any]] = []
    step_reports: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    if int(start_step) < 0 or int(steps) < 1:
        raise ValueError("Qwen elastic step range invalid")
    required_rows = (int(start_step) + int(steps)) * int(microbatch_count)
    if len(train_rows) < required_rows:
        raise ValueError("Qwen training dataset has too few microbatches")
    restart_records: list[dict[str, Any]] = []
    for local_step in range(int(steps)):
        step = int(start_step) + local_step
        _begin_step(clients)
        next_stage0_forward = 0
        stage0_forward_outputs: deque[tuple[int, Any]] = deque()
        stage1_backward_ready: deque[tuple[int, Any]] = deque()
        stage0_backward_ready: deque[tuple[int, Any]] = deque()
        uploaded: set[int] = set()
        gradients: dict[int, Any] = {}
        completed_stage0_backward: set[int] = set()
        deadline = time.monotonic() + float(wait_timeout)
        last_poll = 0.0
        while len(completed_stage0_backward) < int(microbatch_count):
            if time.monotonic() >= deadline:
                raise TimeoutError("Qwen Kernel A pipeline step timed out")
            progressed = False
            if stage0.busy is None:
                if stage0_backward_ready:
                    microbatch, gradient = stage0_backward_ready.popleft()
                    stage0.send(
                        "backward",
                        microbatch_id=microbatch,
                        activation_gradient=gradient,
                    )
                    progressed = True
                elif next_stage0_forward < int(microbatch_count):
                    microbatch = next_stage0_forward
                    next_stage0_forward += 1
                    row_index = step * int(microbatch_count) + microbatch
                    input_ids, _labels = _training_row_inputs_and_labels(
                        train_rows[row_index]
                    )
                    tokens = torch.tensor([input_ids], dtype=torch.long)
                    stage0.send("forward", microbatch_id=microbatch, value=tokens)
                    progressed = True
            if stage1.busy is None:
                if stage1_backward_ready:
                    microbatch, gradient = stage1_backward_ready.popleft()
                    stage1.send(
                        "backward",
                        microbatch_id=microbatch,
                        activation_gradient=gradient,
                    )
                    progressed = True
                elif stage0_forward_outputs:
                    microbatch, activation = stage0_forward_outputs.popleft()
                    stage1.send("forward", microbatch_id=microbatch, value=activation)
                    progressed = True
            for client in clients:
                if client.busy is not None and client.poll(0.0):
                    request, result = client.recv()
                    operation = str(request["operation"])
                    microbatch = int(request.get("microbatch_id", -1))
                    if client.spec.stage_id == 0 and operation == "forward":
                        stage0_forward_outputs.append((microbatch, result["activation"]))
                        events.append(
                            _event_for_result(
                                transport,
                                role="kernel_a",
                                run_kind=run_kind,
                                client=client,
                                operation="forward",
                                step=step,
                                microbatch=microbatch,
                                result=result,
                            )
                        )
                    elif client.spec.stage_id == 1 and operation == "forward":
                        row_index = step * int(microbatch_count) + microbatch
                        _input_ids, label_ids = _training_row_inputs_and_labels(
                            train_rows[row_index]
                        )
                        labels = torch.tensor([label_ids], dtype=torch.long)
                        payload_report = transport.put_tensors(
                            role="kernel_a",
                            run_kind=run_kind,
                            kind="activation",
                            step=step,
                            microbatch=microbatch,
                            tensors={"activation": result["activation"], "labels": labels},
                        )
                        payloads.append(
                            {
                                "kind": "activation",
                                "step": step,
                                "microbatch": microbatch,
                                "payload_hash": payload_report["payload_hash"],
                            }
                        )
                        uploaded.add(microbatch)
                        events.append(
                            _event_for_result(
                                transport,
                                role="kernel_a",
                                run_kind=run_kind,
                                client=client,
                                operation="forward",
                                step=step,
                                microbatch=microbatch,
                                result=result,
                            )
                        )
                    elif client.spec.stage_id == 1 and operation == "backward":
                        stage0_backward_ready.append(
                            (microbatch, result["activation_gradient"])
                        )
                        events.append(
                            _event_for_result(
                                transport,
                                role="kernel_a",
                                run_kind=run_kind,
                                client=client,
                                operation="backward",
                                step=step,
                                microbatch=microbatch,
                                result=result,
                            )
                        )
                    elif client.spec.stage_id == 0 and operation == "backward":
                        completed_stage0_backward.add(microbatch)
                        events.append(
                            _event_for_result(
                                transport,
                                role="kernel_a",
                                run_kind=run_kind,
                                client=client,
                                operation="backward",
                                step=step,
                                microbatch=microbatch,
                                result=result,
                            )
                        )
                    else:
                        raise RuntimeError("Qwen Kernel A stage response was out of protocol")
                    progressed = True
            now = time.monotonic()
            if now - last_poll >= 0.05:
                last_poll = now
                for microbatch in sorted(uploaded - set(gradients)):
                    private = transport.get_tensors(
                        run_kind=run_kind,
                        kind="gradient",
                        step=step,
                        microbatch=microbatch,
                    )
                    if private is not None:
                        tensors, metadata = private
                        gradients[microbatch] = tensors["activation_gradient"]
                        stage1_backward_ready.append(
                            (microbatch, tensors["activation_gradient"])
                        )
                        payloads.append(
                            {
                                "kind": "gradient",
                                "step": step,
                                "microbatch": microbatch,
                                **metadata,
                            }
                        )
                        progressed = True
            if not progressed:
                time.sleep(0.005)
        finishes = _finish_step(
            clients,
            transport=transport,
            role="kernel_a",
            run_kind=run_kind,
            global_step=step + 1,
            dataset_cursor=(step + 1) * int(microbatch_count),
        )
        barrier = (
            step_commit_callback(
                step + 1,
                (step + 1) * int(microbatch_count),
                finishes,
            )
            if step_commit_callback is not None
            else {}
        )
        step_reports.append(
            {"step": step + 1, "stages": finishes, "elastic_barrier": barrier}
        )
        if barrier.get("continue_training") is False:
            break
        if int(restart_pair_after_step) == step + 1:
            if restart_pair_factory is None:
                raise RuntimeError("qwen15b_checkpoint_pair_restart_factory_missing")
            transport.wait_generation(minimum=restart_generation, timeout=wait_timeout)
            records, restart_events = _restart_pair_from_checkpoint(
                clients,
                replacement_factory=restart_pair_factory,
                transport=transport,
                role="kernel_a",
                run_kind=run_kind,
                after_step=step + 1,
                dataset_cursor=(step + 1) * int(microbatch_count),
            )
            restart_records.extend(records)
            events.extend(restart_events)
            stage0, stage1 = clients
    adapter_states = []
    adapter_hashes = {}
    statuses = []
    for client in clients:
        private = client.call("adapter_state", timeout=120.0)
        adapter_states.append(private["adapter_state"])
        adapter_hashes[str(client.spec.stage_id)] = private["adapter_hash"]
        statuses.append(client.call("status", timeout=300.0))
    return {
        "schema": RUNTIME_SCHEMA,
        "role": "kernel_a",
        "run_kind": run_kind,
        "steps_completed": len(step_reports),
        "start_step": int(start_step),
        "end_step": int(start_step) + len(step_reports),
        "microbatches_per_step": int(microbatch_count),
        "events": events,
        "payloads": payloads,
        "step_reports": step_reports,
        "dataset_row_indexes": list(
            range(
                int(start_step) * int(microbatch_count),
                (int(start_step) + len(step_reports)) * int(microbatch_count),
            )
        ),
        "adapter_hashes": adapter_hashes,
        "adapter_states_private": adapter_states,
        "stage_statuses": statuses,
        "coordinator_restart_stage_recoveries": restart_records,
        "coordinator_restart_all_owned_stages_verified": bool(
            not restart_pair_after_step
            or {int(item["stage_id"]) for item in restart_records} == {0, 1}
        ),
        "real_forward": True,
        "real_backward": True,
    }


def run_kernel_b_once(
    *,
    run_kind: str,
    clients: list[StageProcessClient],
    transport: QwenHTTPTransport,
    steps: int = DEFAULT_STEPS,
    start_step: int = 0,
    microbatch_count: int = DEFAULT_MICROBATCHES,
    wait_timeout: float = 900.0,
    step_commit_callback: Callable[
        [int, int, list[dict[str, Any]]], dict[str, Any]
    ]
    | None = None,
    restart_stage2_after_step: int = 0,
    restart_stage2_factory: Callable[[], StageProcessClient] | None = None,
    restart_pair_after_step: int = 0,
    restart_pair_factory: Callable[[], list[StageProcessClient]] | None = None,
    restart_generation: int = 1,
) -> dict[str, Any]:
    """Drive stages 2/3 and return scaled boundary gradients to Kernel A."""

    stage2, stage3 = clients
    if [stage2.spec.stage_id, stage3.spec.stage_id] != [2, 3]:
        raise ValueError("Kernel B must own Qwen stages 2 and 3")
    events: list[dict[str, Any]] = []
    step_reports: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    losses: list[float] = []
    step_mean_losses: list[float] = []
    restart_records: list[dict[str, Any]] = []
    coordinator_restart_records: list[dict[str, Any]] = []
    if int(start_step) < 0 or int(steps) < 1:
        raise ValueError("Qwen elastic step range invalid")
    for local_step in range(int(steps)):
        step = int(start_step) + local_step
        step_loss_offset = len(losses)
        _begin_step(clients)
        activations: deque[tuple[int, Any, Any]] = deque()
        stage2_outputs: deque[tuple[int, Any, Any]] = deque()
        stage2_backward_ready: deque[tuple[int, Any]] = deque()
        fetched: set[int] = set()
        uploaded_gradients: set[int] = set()
        labels_by_microbatch: dict[int, Any] = {}
        deadline = time.monotonic() + float(wait_timeout)
        last_poll = 0.0
        while len(uploaded_gradients) < int(microbatch_count):
            if time.monotonic() >= deadline:
                raise TimeoutError("Qwen Kernel B pipeline step timed out")
            progressed = False
            if stage2.busy is None:
                if stage2_backward_ready:
                    microbatch, gradient = stage2_backward_ready.popleft()
                    stage2.send(
                        "backward",
                        microbatch_id=microbatch,
                        activation_gradient=gradient,
                    )
                    progressed = True
                elif activations:
                    microbatch, activation, labels = activations.popleft()
                    labels_by_microbatch[microbatch] = labels
                    stage2.send("forward", microbatch_id=microbatch, value=activation)
                    progressed = True
            if stage3.busy is None and stage2_outputs:
                microbatch, hidden, labels = stage2_outputs.popleft()
                stage3.send(
                    "loss_backward",
                    microbatch_id=microbatch,
                    hidden_states=hidden,
                    labels=labels,
                    microbatch_count=int(microbatch_count),
                )
                progressed = True
            for client in clients:
                if client.busy is not None and client.poll(0.0):
                    request, result = client.recv()
                    operation = str(request["operation"])
                    microbatch = int(request.get("microbatch_id", -1))
                    if client.spec.stage_id == 2 and operation == "forward":
                        stage2_outputs.append(
                            (microbatch, result["activation"], labels_by_microbatch[microbatch])
                        )
                        events.append(
                            _event_for_result(
                                transport,
                                role="kernel_b",
                                run_kind=run_kind,
                                client=client,
                                operation="forward",
                                step=step,
                                microbatch=microbatch,
                                result=result,
                            )
                        )
                    elif client.spec.stage_id == 3 and operation == "loss_backward":
                        stage2_backward_ready.append(
                            (microbatch, result["activation_gradient"])
                        )
                        losses.append(float(result["loss"]))
                        events.append(
                            _event_for_result(
                                transport,
                                role="kernel_b",
                                run_kind=run_kind,
                                client=client,
                                operation="forward_backward",
                                step=step,
                                microbatch=microbatch,
                                result=result,
                            )
                        )
                    elif client.spec.stage_id == 2 and operation == "backward":
                        outgoing = result["activation_gradient"]
                        payload_report = transport.put_tensors(
                            role="kernel_b",
                            run_kind=run_kind,
                            kind="gradient",
                            step=step,
                            microbatch=microbatch,
                            tensors={"activation_gradient": outgoing},
                        )
                        payloads.append(
                            {
                                "kind": "gradient",
                                "step": step,
                                "microbatch": microbatch,
                                "payload_hash": payload_report["payload_hash"],
                            }
                        )
                        uploaded_gradients.add(microbatch)
                        events.append(
                            _event_for_result(
                                transport,
                                role="kernel_b",
                                run_kind=run_kind,
                                client=client,
                                operation="backward",
                                step=step,
                                microbatch=microbatch,
                                result=result,
                            )
                        )
                    else:
                        raise RuntimeError("Qwen Kernel B stage response was out of protocol")
                    progressed = True
            now = time.monotonic()
            if now - last_poll >= 0.05:
                last_poll = now
                for microbatch in range(int(microbatch_count)):
                    if microbatch in fetched:
                        continue
                    private = transport.get_tensors(
                        run_kind=run_kind,
                        kind="activation",
                        step=step,
                        microbatch=microbatch,
                    )
                    if private is not None:
                        tensors, metadata = private
                        fetched.add(microbatch)
                        activations.append(
                            (microbatch, tensors["activation"], tensors["labels"])
                        )
                        payloads.append(
                            {
                                "kind": "activation",
                                "step": step,
                                "microbatch": microbatch,
                                **metadata,
                            }
                        )
                        progressed = True
            if not progressed:
                time.sleep(0.005)
        finishes = _finish_step(
            clients,
            transport=transport,
            role="kernel_b",
            run_kind=run_kind,
            global_step=step + 1,
            dataset_cursor=(step + 1) * int(microbatch_count),
        )
        barrier = (
            step_commit_callback(
                step + 1,
                (step + 1) * int(microbatch_count),
                finishes,
            )
            if step_commit_callback is not None
            else {}
        )
        step_reports.append(
            {"step": step + 1, "stages": finishes, "elastic_barrier": barrier}
        )
        current_losses = losses[step_loss_offset:]
        if len(current_losses) != int(microbatch_count):
            raise RuntimeError("Qwen final stage did not produce one loss per microbatch")
        step_mean_losses.append(sum(current_losses) / len(current_losses))
        if barrier.get("continue_training") is False:
            break
        if int(restart_pair_after_step) == step + 1:
            if restart_pair_factory is None:
                raise RuntimeError("qwen15b_checkpoint_pair_restart_factory_missing")
            transport.wait_generation(minimum=restart_generation, timeout=wait_timeout)
            records, restart_events = _restart_pair_from_checkpoint(
                clients,
                replacement_factory=restart_pair_factory,
                transport=transport,
                role="kernel_b",
                run_kind=run_kind,
                after_step=step + 1,
                dataset_cursor=(step + 1) * int(microbatch_count),
            )
            coordinator_restart_records.extend(records)
            restart_records.extend(
                item for item in records if int(item.get("stage_id", -1)) == 2
            )
            events.extend(restart_events)
            stage2, stage3 = clients
        elif int(restart_stage2_after_step) == step + 1:
            if restart_stage2_factory is None:
                raise RuntimeError("Qwen stage2 restart factory is required")
            old_pid = stage2.pid
            stopped = stage2.force_stop()
            if not stopped:
                raise RuntimeError("Qwen stage2 forced stop did not terminate its process")
            transport.event(
                role="kernel_b",
                run_kind=run_kind,
                operation="stage_stopped",
                stage_id=2,
                step=step + 1,
                pid=old_pid,
                device=stage2.device,
            )
            replacement = restart_stage2_factory()
            if (
                replacement.pid == old_pid
                or replacement.ready.get("resumed") is not True
                or int(replacement.ready.get("resumed_global_step") or 0) != step + 1
                or int(replacement.ready.get("resumed_dataset_cursor") or 0)
                != (step + 1) * int(microbatch_count)
            ):
                replacement.force_stop()
                raise RuntimeError("Qwen stage2 checkpoint restart contract was not satisfied")
            stage2 = replacement
            clients[0] = replacement
            transport.register(role="kernel_b", ready=[stage2.ready, stage3.ready])
            transport.event(
                role="kernel_b",
                run_kind=run_kind,
                operation="stage_restarted",
                stage_id=2,
                step=step + 1,
                pid=stage2.pid,
                device=stage2.device,
                checkpoint_hash=str(stage2.ready.get("loaded_checkpoint_hash") or ""),
            )
            restart_record = {
                "stage_id": 2,
                "after_step": step + 1,
                "old_pid": old_pid,
                "new_pid": stage2.pid,
                "new_pid_verified": stage2.pid != old_pid,
                "forced_stop_verified": stopped,
                "checkpoint_resume_verified": True,
                "resumed_global_step": int(stage2.ready["resumed_global_step"]),
                "resumed_dataset_cursor": int(stage2.ready["resumed_dataset_cursor"]),
            }
            restart_records.append(restart_record)
            events.extend(
                [
                    {
                        "run_kind": run_kind,
                        "operation": "stage_stopped",
                        "stage_id": 2,
                        "step": step + 1,
                        "microbatch": -1,
                        "pid": old_pid,
                        "device": stage2.device,
                        "started_ns": 0,
                        "ended_ns": 0,
                        "loss": None,
                    },
                    {
                        "run_kind": run_kind,
                        "operation": "stage_restarted",
                        "stage_id": 2,
                        "step": step + 1,
                        "microbatch": -1,
                        "pid": stage2.pid,
                        "device": stage2.device,
                        "started_ns": 0,
                        "ended_ns": 0,
                        "loss": None,
                    },
                ]
            )
    adapter_states = []
    adapter_hashes = {}
    statuses = []
    for client in clients:
        private = client.call("adapter_state", timeout=120.0)
        adapter_states.append(private["adapter_state"])
        adapter_hashes[str(client.spec.stage_id)] = private["adapter_hash"]
        statuses.append(client.call("status", timeout=300.0))
    return {
        "schema": RUNTIME_SCHEMA,
        "role": "kernel_b",
        "run_kind": run_kind,
        "steps_completed": len(step_reports),
        "start_step": int(start_step),
        "end_step": int(start_step) + len(step_reports),
        "microbatches_per_step": int(microbatch_count),
        "events": events,
        "payloads": payloads,
        "step_reports": step_reports,
        "losses": losses,
        "step_mean_losses": step_mean_losses,
        "loss_start": step_mean_losses[0] if step_mean_losses else None,
        "loss_end": step_mean_losses[-1] if step_mean_losses else None,
        "loss_reduced": bool(
            step_mean_losses and step_mean_losses[-1] < step_mean_losses[0]
        ),
        "adapter_hashes": adapter_hashes,
        "adapter_states_private": adapter_states,
        "stage_statuses": statuses,
        "controlled_restarts": restart_records,
        "coordinator_restart_stage_recoveries": coordinator_restart_records,
        "coordinator_restart_all_owned_stages_verified": bool(
            not restart_pair_after_step
            or {int(item["stage_id"]) for item in coordinator_restart_records} == {2, 3}
        ),
        "controlled_restart_verified": bool(
            (
                len(restart_records) == 1
                and restart_records[0]["new_pid_verified"]
                and restart_records[0]["forced_stop_verified"]
                and restart_records[0]["checkpoint_resume_verified"]
            )
            if restart_pair_after_step
            else not restart_stage2_after_step
            or (
                len(restart_records) == 1
                and restart_records[0]["new_pid_verified"]
                and restart_records[0]["forced_stop_verified"]
                and restart_records[0]["checkpoint_resume_verified"]
            )
        ),
        "real_forward": True,
        "real_backward": True,
    }


def four_stage_overlap_summary(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    intervals = [
        {
            "stage_id": int(item.get("stage_id", -1)),
            "started_ns": int(item.get("started_ns") or 0),
            "ended_ns": int(item.get("ended_ns") or 0),
            "run_kind": str(item.get("run_kind") or ""),
            "step": int(item.get("step", -1)),
        }
        for item in events
        if int(item.get("stage_id", -1)) in {0, 1, 2, 3}
        and int(item.get("ended_ns") or 0) > int(item.get("started_ns") or 0)
    ]
    best: dict[str, Any] | None = None
    for run_kind in ("baseline", "resumed", "elastic"):
        for step in range(DEFAULT_STEPS):
            selected = [
                item
                for item in intervals
                if item["run_kind"] == run_kind and item["step"] == step
            ]
            points: list[tuple[int, int, int]] = []
            for item in selected:
                points.append((item["started_ns"], 1, item["stage_id"]))
                points.append((item["ended_ns"], -1, item["stage_id"]))
            points.sort(key=lambda value: (value[0], value[1]))
            active: dict[int, int] = {}
            all_started: int | None = None
            for at, delta, stage_id in points:
                had_all = all(active.get(index, 0) > 0 for index in range(4))
                if delta > 0:
                    active[stage_id] = active.get(stage_id, 0) + 1
                else:
                    active[stage_id] = max(0, active.get(stage_id, 0) - 1)
                has_all = all(active.get(index, 0) > 0 for index in range(4))
                if not had_all and has_all:
                    all_started = at
                elif had_all and not has_all and all_started is not None:
                    duration = at - all_started
                    if best is None or duration > int(best["duration_ns"]):
                        best = {
                            "run_kind": run_kind,
                            "step": step,
                            "started_ns": all_started,
                            "ended_ns": at,
                            "duration_ns": duration,
                        }
                    all_started = None
    return {
        "four_stage_compute_overlap_verified": best is not None and int(best["duration_ns"]) > 0,
        "maximum_four_stage_overlap": best or {},
        "stage_ids": [0, 1, 2, 3],
        "interval_count": len(intervals),
    }


def public_runtime_report(value: dict[str, Any]) -> dict[str, Any]:
    """Remove process-local tensors and paths before writing worker artifacts."""

    private_keys = {"adapter_states_private", "activation", "activation_gradient", "labels"}

    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                str(key): clean(child)
                for key, child in item.items()
                if str(key) not in private_keys and not str(key).endswith("_path")
            }
        if isinstance(item, list):
            return [clean(child) for child in item]
        return item

    report = clean(value)
    report.update(
        {
            "activation_values_public": False,
            "gradient_values_public": False,
            "adapter_tensor_values_public": False,
            "token_ids_public": False,
            "raw_training_text_public": False,
            "credentials_public": False,
            "coordinator_url_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
    )
    return report
