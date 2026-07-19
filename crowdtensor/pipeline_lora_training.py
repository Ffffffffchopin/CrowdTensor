"""Two-process CPU pipeline training reference with LoRA-only stage ownership."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Protocol

from .training_contract import sha256_file, sha256_json, tensor_bytes, tensor_specs


PIPELINE_SCHEMA = "crowdtensor_two_process_pipeline_training_v1"
CUDA_PIPELINE_SCHEMA = "crowdtensor_two_process_cuda_pipeline_training_v1"
STAGE_CHECKPOINT_SCHEMA = "crowdtensor_pipeline_stage_checkpoint_v1"
GLOBAL_CHECKPOINT_SCHEMA = "crowdtensor_pipeline_global_checkpoint_v1"


class StageRuntime(Protocol):
    stage_id: int

    def forward(self, value: Any) -> Any: ...

    def checkpoint(self, step: int, cursor: int) -> dict[str, Any]: ...


def _tensor_hash(tensor: Any) -> str:
    return "sha256:" + hashlib.sha256(tensor_bytes(tensor)).hexdigest()


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _trainable_state(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _base_state(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if "lora_" not in name
    }


def _state_hash(values: dict[str, Any]) -> str:
    return sha256_json(tensor_specs(values))


def _grad_norm(model: Any) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            total += float(parameter.grad.detach().float().norm().item()) ** 2
    return math.sqrt(total)


def _build_stage(stage_id: int, config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import torch
    from peft import LoraConfig, inject_adapter_in_model

    torch.manual_seed(int(config["seed"]) + 1000 * int(stage_id))
    hidden = int(config["hidden_size"])
    heads = int(config["num_heads"])
    feedforward = int(config["intermediate_size"])
    total_layers = int(config["num_layers"])
    split = int(config["split_index"])
    vocab = int(config["vocab_size"])
    max_positions = int(config["max_positions"])

    class Stage0(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = torch.nn.Embedding(vocab, hidden)
            self.position_embedding = torch.nn.Embedding(max_positions, hidden)
            self.layers = torch.nn.ModuleList(
                [
                    torch.nn.TransformerEncoderLayer(
                        hidden,
                        heads,
                        dim_feedforward=feedforward,
                        dropout=0.0,
                        activation="gelu",
                        batch_first=True,
                    )
                    for _ in range(split)
                ]
            )

        def forward(self, input_ids: Any) -> Any:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
            mask = torch.triu(
                torch.ones(input_ids.shape[1], input_ids.shape[1], dtype=torch.bool, device=input_ids.device),
                diagonal=1,
            )
            for layer in self.layers:
                hidden_states = layer(hidden_states, src_mask=mask)
            return hidden_states

    class Stage1(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [
                    torch.nn.TransformerEncoderLayer(
                        hidden,
                        heads,
                        dim_feedforward=feedforward,
                        dropout=0.0,
                        activation="gelu",
                        batch_first=True,
                    )
                    for _ in range(total_layers - split)
                ]
            )
            self.norm = torch.nn.LayerNorm(hidden)
            self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

        def forward(self, hidden_states: Any) -> Any:
            mask = torch.triu(
                torch.ones(hidden_states.shape[1], hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device),
                diagonal=1,
            )
            for layer in self.layers:
                hidden_states = layer(hidden_states, src_mask=mask)
            return self.lm_head(self.norm(hidden_states))

    model = Stage0() if int(stage_id) == 0 else Stage1()
    for parameter in model.parameters():
        parameter.requires_grad = False
    lora = LoraConfig(
        r=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        target_modules=["linear1", "linear2"],
        lora_dropout=0.0,
        bias="none",
    )
    model = inject_adapter_in_model(lora, model)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError(f"pipeline stage {stage_id} has non-LoRA trainable parameters")
    owned_layers = list(range(0, split)) if int(stage_id) == 0 else list(range(split, total_layers))
    ownership = {
        "stage_id": int(stage_id),
        "owned_layer_indexes": owned_layers,
        "owns_embedding": int(stage_id) == 0,
        "owns_norm_and_lm_head": int(stage_id) == 1,
        "loaded_full_model": False,
        "parameter_count": sum(int(value.numel()) for value in model.parameters()),
        "trainable_parameter_count": sum(
            int(value.numel()) for value in model.parameters() if value.requires_grad
        ),
        "trainable_tensor_names": trainable,
    }
    return model, ownership


def _save_stage_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    stage_id: int,
    step: int,
    cursor: int,
    checkpoint_dir: Path,
    base_hash: str,
    ownership: dict[str, Any],
    scaler: Any | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = checkpoint_dir / f"stage{stage_id}_adapter.safetensors"
    optimizer_path = checkpoint_dir / f"stage{stage_id}_optimizer.pt"
    adapter = _trainable_state(model)
    save_file(adapter, str(adapter_path))
    torch.save(optimizer.state_dict(), optimizer_path)
    base_hash_after = _state_hash(_base_state(model))
    manifest = {
        "schema": STAGE_CHECKPOINT_SCHEMA,
        "stage_id": int(stage_id),
        "optimizer_step": int(step),
        "global_step": int(step),
        "dataset_cursor": int(cursor),
        "adapter_path": str(adapter_path.resolve()),
        "adapter_file_hash": sha256_file(adapter_path),
        "adapter_tensor_hash": _state_hash(adapter),
        "adapter_tensor_specs": tensor_specs(adapter),
        "optimizer_path": str(optimizer_path.resolve()),
        "optimizer_file_hash": sha256_file(optimizer_path),
        "base_hash_before": base_hash,
        "base_hash_after": base_hash_after,
        "base_weights_frozen": base_hash == base_hash_after,
        "ownership": ownership,
    }
    if scaler is not None:
        scaler_path = checkpoint_dir / f"stage{stage_id}_grad_scaler.pt"
        torch.save(scaler.state_dict(), scaler_path)
        manifest.update(
            {
                "grad_scaler_path": str(scaler_path.resolve()),
                "grad_scaler_file_hash": sha256_file(scaler_path),
                "grad_scaler_state_present": True,
                "cuda_placement": str(device),
            }
        )
    manifest["content_hash"] = sha256_json(
        {key: value for key, value in manifest.items() if not key.endswith("_path")}
    )
    path = _write_json(checkpoint_dir / f"stage{stage_id}_checkpoint.json", manifest)
    return {**manifest, "manifest_path": str(path.resolve())}


def _load_stage_checkpoint(
    model: Any,
    optimizer: Any,
    checkpoint_dir: Path,
    stage_id: int,
    *,
    scaler: Any | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    manifest_path = checkpoint_dir / f"stage{stage_id}_checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapter_path = Path(manifest["adapter_path"])
    optimizer_path = Path(manifest["optimizer_path"])
    if sha256_file(adapter_path) != manifest["adapter_file_hash"]:
        raise RuntimeError(f"stage {stage_id} adapter checkpoint hash mismatch")
    if sha256_file(optimizer_path) != manifest["optimizer_file_hash"]:
        raise RuntimeError(f"stage {stage_id} optimizer checkpoint hash mismatch")
    adapter = load_file(str(adapter_path), device="cpu")
    incompatible = model.load_state_dict(adapter, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"stage {stage_id} checkpoint has unexpected adapter tensors")
    optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=True))
    if str(device).startswith("cuda"):
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if hasattr(value, "to"):
                    state[key] = value.to(device)
    if scaler is not None:
        scaler_path = Path(str(manifest.get("grad_scaler_path") or ""))
        if not scaler_path.is_file() or sha256_file(scaler_path) != manifest.get("grad_scaler_file_hash"):
            raise RuntimeError(f"stage {stage_id} GradScaler checkpoint hash mismatch")
        scaler.load_state_dict(torch.load(scaler_path, map_location="cpu", weights_only=True))
    return manifest


class CUDAStageRuntime:
    """One genuinely stage-owned CUDA process in the two-stage training pipeline."""

    def __init__(
        self,
        stage_id: int,
        config: dict[str, Any],
        checkpoint_dir: str | Path,
        *,
        device: str,
        resume: bool = False,
    ) -> None:
        import torch

        from .hf_lora_training import _new_grad_scaler

        self.stage_id = int(stage_id)
        self.device = str(device)
        if self.stage_id not in {0, 1}:
            raise ValueError("CUDA pipeline stage id must be 0 or 1")
        if not self.device.startswith("cuda:"):
            raise ValueError("CUDA stage placement must use cuda:<index>")
        self.device_index = int(self.device.split(":", 1)[1])
        if not torch.cuda.is_available() or self.device_index >= int(torch.cuda.device_count()):
            raise RuntimeError("cuda_pipeline_stage_device_unavailable")
        torch.cuda.set_device(self.device_index)
        torch.manual_seed(int(config["seed"]) + 1000 * self.stage_id)
        torch.cuda.manual_seed(int(config["seed"]) + 1000 * self.stage_id)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device_index)
        self.model, self.ownership = _build_stage(self.stage_id, config)
        self.model.to(self.device)
        self.model.train()
        self.base_hash = _state_hash(_base_state(self.model))
        self.trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters,
            lr=float(config["learning_rate"]),
            weight_decay=0.0,
        )
        self.scaler = _new_grad_scaler(torch)
        self.gradient_clip_norm = float(config.get("gradient_clip_norm", 1.0))
        self.checkpoint_dir = Path(checkpoint_dir)
        self.loaded_checkpoint: dict[str, Any] | None = None
        if resume:
            self.loaded_checkpoint = _load_stage_checkpoint(
                self.model,
                self.optimizer,
                self.checkpoint_dir,
                self.stage_id,
                scaler=self.scaler,
                device=self.device,
            )
        self.cached: dict[int, Any] = {}

    def ready(self) -> dict[str, Any]:
        import torch

        return {
            "type": "ready",
            "stage_id": self.stage_id,
            "pid": os.getpid(),
            "ownership": self.ownership,
            "base_hash": self.base_hash,
            "resumed": self.loaded_checkpoint is not None,
            "resumed_optimizer_step": int((self.loaded_checkpoint or {}).get("optimizer_step", 0)),
            "resumed_cursor": int((self.loaded_checkpoint or {}).get("dataset_cursor", 0)),
            "cuda_device": self.device,
            "cuda_device_index": self.device_index,
            "cuda_device_name_hash": sha256_json(
                {"device_name": torch.cuda.get_device_name(self.device_index)}
            ),
            "cuda_live": True,
            "fp16_autocast": True,
            "grad_scaler": True,
        }

    def checkpoint(self, step: int, cursor: int) -> dict[str, Any]:
        return _save_stage_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            stage_id=self.stage_id,
            step=step,
            cursor=cursor,
            checkpoint_dir=self.checkpoint_dir,
            base_hash=self.base_hash,
            ownership=self.ownership,
            scaler=self.scaler,
            device=self.device,
        )

    def forward(self, input_ids: Any, *, step: int) -> dict[str, Any]:
        import torch

        if self.stage_id != 0:
            raise RuntimeError("only CUDA stage0 accepts token forward")
        self.optimizer.zero_grad(set_to_none=True)
        tokens = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            activation = self.model(tokens)
        # Stage1 scales the loss and returns an already-scaled activation
        # gradient. Prime stage0's lazy GradScaler state so unscale_/step can
        # consume that external gradient with the matching scale.
        self.scaler.scale(activation.detach().new_zeros(()))
        self.cached[int(step)] = activation
        cpu_activation = activation.detach().cpu().contiguous()
        return {
            "type": "forward",
            "stage_id": 0,
            "step": int(step),
            "activation": cpu_activation.numpy(),
            "forward_hash": _tensor_hash(cpu_activation),
            "activation_shape": list(cpu_activation.shape),
            "activation_dtype": str(cpu_activation.dtype).replace("torch.", ""),
            "private_cpu_transport": True,
        }

    def forward_backward(
        self,
        activation: Any,
        labels: Any,
        *,
        step: int,
        cursor: int,
    ) -> dict[str, Any]:
        import torch

        if self.stage_id != 1:
            raise RuntimeError("only CUDA stage1 accepts activation forward/backward")
        self.optimizer.zero_grad(set_to_none=True)
        hidden = torch.from_numpy(activation).to(self.device).requires_grad_(True)
        target = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        scale_before = float(self.scaler.get_scale())
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = self.model(hidden)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )
        self.scaler.scale(loss).backward()
        scaled_activation_gradient = hidden.grad.detach().cpu().contiguous()
        self.scaler.unscale_(self.optimizer)
        lora_gradient_norm = _grad_norm(self.model)
        torch.nn.utils.clip_grad_norm_(self.trainable_parameters, self.gradient_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        checkpoint = self.checkpoint(step + 1, cursor)
        torch.cuda.synchronize(self.device_index)
        return {
            "type": "backward",
            "stage_id": 1,
            "step": int(step),
            "loss": float(loss.detach().float().item()),
            "logits_hash": _tensor_hash(logits),
            "activation_gradient": scaled_activation_gradient.numpy(),
            "backward_gradient_hash": _tensor_hash(scaled_activation_gradient),
            "gradient_scale": scale_before,
            "lora_gradient_norm": lora_gradient_norm,
            "optimizer_step": int(step) + 1,
            "checkpoint_hash": checkpoint["content_hash"],
            "checkpoint": checkpoint,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device_index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device_index)),
        }

    def backward(self, activation_gradient: Any, *, step: int, cursor: int, gradient_scale: float) -> dict[str, Any]:
        import torch

        if self.stage_id != 0:
            raise RuntimeError("only CUDA stage0 accepts activation gradient")
        activation = self.cached.pop(int(step))
        scaled_gradient = torch.from_numpy(activation_gradient).to(self.device)
        local_scale = float(self.scaler.get_scale())
        if local_scale != float(gradient_scale):
            raise RuntimeError("CUDA pipeline GradScaler values diverged between stages")
        activation.backward(scaled_gradient)
        self.scaler.unscale_(self.optimizer)
        lora_gradient_norm = _grad_norm(self.model)
        torch.nn.utils.clip_grad_norm_(self.trainable_parameters, self.gradient_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        checkpoint = self.checkpoint(step + 1, cursor)
        torch.cuda.synchronize(self.device_index)
        return {
            "type": "backward_applied",
            "stage_id": 0,
            "step": int(step),
            "backward_gradient_hash": _tensor_hash(scaled_gradient),
            "gradient_scale": local_scale,
            "lora_gradient_norm": lora_gradient_norm,
            "optimizer_step": int(step) + 1,
            "checkpoint_hash": checkpoint["content_hash"],
            "checkpoint": checkpoint,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device_index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device_index)),
        }

    def status(self) -> dict[str, Any]:
        import torch

        return {
            "type": "status",
            "stage_id": self.stage_id,
            "base_hash": self.base_hash,
            "base_hash_after": _state_hash(_base_state(self.model)),
            "adapter_tensor_hash": _state_hash(_trainable_state(self.model)),
            "cuda_device": self.device,
            "cuda_device_index": self.device_index,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device_index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device_index)),
        }


def _stage_worker(
    stage_id: int,
    connection: Any,
    config: dict[str, Any],
    checkpoint_dir: str,
    resume: bool,
) -> None:
    try:
        import torch

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        model, ownership = _build_stage(stage_id, config)
        model.to("cpu")
        model.train()
        base_hash = _state_hash(_base_state(model))
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(config["learning_rate"]),
            weight_decay=0.0,
        )
        loaded_checkpoint = None
        if resume:
            loaded_checkpoint = _load_stage_checkpoint(model, optimizer, Path(checkpoint_dir), stage_id)
        connection.send(
            {
                "type": "ready",
                "stage_id": stage_id,
                "pid": os.getpid(),
                "ownership": ownership,
                "base_hash": base_hash,
                "resumed": bool(resume),
                "resumed_optimizer_step": int((loaded_checkpoint or {}).get("optimizer_step", 0)),
                "resumed_cursor": int((loaded_checkpoint or {}).get("dataset_cursor", 0)),
            }
        )
        cached: dict[int, Any] = {}
        while True:
            message = connection.recv()
            command = message.get("command")
            if command == "stop":
                connection.send({"type": "stopped", "stage_id": stage_id, "pid": os.getpid()})
                break
            if command == "status":
                connection.send(
                    {
                        "type": "status",
                        "stage_id": stage_id,
                        "base_hash": base_hash,
                        "base_hash_after": _state_hash(_base_state(model)),
                        "adapter_tensor_hash": _state_hash(_trainable_state(model)),
                    }
                )
                continue
            step = int(message["step"])
            cursor = int(message["cursor"])
            if int(stage_id) == 0 and command == "forward":
                optimizer.zero_grad(set_to_none=True)
                input_ids = torch.tensor(message["input_ids"], dtype=torch.long)
                activation = model(input_ids)
                cached[step] = activation
                connection.send(
                    {
                        "type": "forward",
                        "stage_id": 0,
                        "step": step,
                        "activation": activation.detach().cpu().numpy(),
                        "forward_hash": _tensor_hash(activation),
                        "activation_shape": list(activation.shape),
                    }
                )
                continue
            if int(stage_id) == 1 and command == "forward_backward":
                optimizer.zero_grad(set_to_none=True)
                activation = torch.from_numpy(message["activation"]).requires_grad_(True)
                labels = torch.tensor(message["labels"], dtype=torch.long)
                logits = model(activation)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                )
                loss.backward()
                activation_gradient = activation.grad.detach().cpu().contiguous()
                lora_gradient_norm = _grad_norm(model)
                optimizer.step()
                checkpoint = _save_stage_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    stage_id=1,
                    step=step + 1,
                    cursor=cursor,
                    checkpoint_dir=Path(checkpoint_dir),
                    base_hash=base_hash,
                    ownership=ownership,
                )
                connection.send(
                    {
                        "type": "backward",
                        "stage_id": 1,
                        "step": step,
                        "loss": float(loss.detach().item()),
                        "logits_hash": _tensor_hash(logits),
                        "activation_gradient": activation_gradient.numpy(),
                        "backward_gradient_hash": _tensor_hash(activation_gradient),
                        "lora_gradient_norm": lora_gradient_norm,
                        "optimizer_step": step + 1,
                        "checkpoint_hash": checkpoint["content_hash"],
                        "checkpoint": checkpoint,
                    }
                )
                continue
            if int(stage_id) == 0 and command == "backward":
                activation = cached.pop(step)
                gradient = torch.from_numpy(message["activation_gradient"])
                activation.backward(gradient)
                lora_gradient_norm = _grad_norm(model)
                optimizer.step()
                checkpoint = _save_stage_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    stage_id=0,
                    step=step + 1,
                    cursor=cursor,
                    checkpoint_dir=Path(checkpoint_dir),
                    base_hash=base_hash,
                    ownership=ownership,
                )
                connection.send(
                    {
                        "type": "backward_applied",
                        "stage_id": 0,
                        "step": step,
                        "backward_gradient_hash": _tensor_hash(gradient),
                        "lora_gradient_norm": lora_gradient_norm,
                        "optimizer_step": step + 1,
                        "checkpoint_hash": checkpoint["content_hash"],
                        "checkpoint": checkpoint,
                    }
                )
                continue
            raise RuntimeError(f"invalid pipeline stage command {command} for stage {stage_id}")
    except BaseException as exc:
        try:
            connection.send(
                {
                    "type": "error",
                    "stage_id": stage_id,
                    "error_class": type(exc).__name__,
                    "error": str(exc),
                }
            )
        finally:
            connection.close()
        raise


def _cuda_stage_worker(
    stage_id: int,
    connection: Any,
    config: dict[str, Any],
    checkpoint_dir: str,
    resume: bool,
) -> None:
    try:
        runtime = CUDAStageRuntime(
            stage_id,
            config,
            checkpoint_dir,
            device=f"cuda:{stage_id}",
            resume=resume,
        )
        connection.send(runtime.ready())
        while True:
            message = connection.recv()
            command = message.get("command")
            if command == "stop":
                connection.send({"type": "stopped", "stage_id": stage_id, "pid": os.getpid()})
                break
            if command == "status":
                connection.send(runtime.status())
                continue
            if command == "forward":
                connection.send(runtime.forward(message["input_ids"], step=int(message["step"])))
                continue
            if command == "forward_backward":
                connection.send(
                    runtime.forward_backward(
                        message["activation"],
                        message["labels"],
                        step=int(message["step"]),
                        cursor=int(message["cursor"]),
                    )
                )
                continue
            if command == "backward":
                connection.send(
                    runtime.backward(
                        message["activation_gradient"],
                        step=int(message["step"]),
                        cursor=int(message["cursor"]),
                        gradient_scale=float(message["gradient_scale"]),
                    )
                )
                continue
            raise RuntimeError(f"invalid CUDA pipeline command {command} for stage {stage_id}")
    except BaseException as exc:
        error_code = "cuda_pipeline_stage_failed"
        try:
            import torch

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                error_code = "cuda_pipeline_out_of_memory"
        except Exception:
            pass
        try:
            connection.send(
                {
                    "type": "error",
                    "stage_id": int(stage_id),
                    "error_class": type(exc).__name__,
                    "error_code": error_code,
                    "error": str(exc)[:240],
                }
            )
        finally:
            connection.close()
        raise


def _recv(connection: Any, *, timeout: float = 30.0) -> dict[str, Any]:
    if not connection.poll(timeout):
        raise TimeoutError("pipeline stage response timed out")
    response = connection.recv()
    if response.get("type") == "error":
        raise RuntimeError(
            f"pipeline stage {response.get('stage_id')} failed: "
            f"{response.get('error_class')}: {response.get('error')}"
        )
    return response


def _start_stage(
    context: Any,
    stage_id: int,
    config: dict[str, Any],
    checkpoint_dir: Path,
    *,
    resume: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    parent, child = context.Pipe()
    process = context.Process(
        target=_stage_worker,
        args=(stage_id, child, config, str(checkpoint_dir), resume),
        name=f"crowdtensor-pipeline-stage{stage_id}",
    )
    process.start()
    child.close()
    ready = _recv(parent, timeout=60.0)
    if ready.get("type") != "ready":
        raise RuntimeError(f"pipeline stage {stage_id} did not become ready")
    return process, parent, ready


def _start_cuda_stage(
    context: Any,
    stage_id: int,
    config: dict[str, Any],
    checkpoint_dir: Path,
    *,
    resume: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    parent, child = context.Pipe()
    process = context.Process(
        target=_cuda_stage_worker,
        args=(stage_id, child, config, str(checkpoint_dir), resume),
        name=f"crowdtensor-cuda-pipeline-stage{stage_id}",
    )
    process.start()
    child.close()
    ready = _recv(parent, timeout=120.0)
    if ready.get("type") != "ready" or ready.get("cuda_live") is not True:
        raise RuntimeError(f"CUDA pipeline stage {stage_id} did not become ready")
    return process, parent, ready


def _stop_stage(process: Any, connection: Any) -> int:
    if process.is_alive():
        connection.send({"command": "stop"})
        _recv(connection, timeout=10.0)
        process.join(timeout=10.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10.0)
    connection.close()
    return int(process.exitcode or 0)


def _global_checkpoint(
    checkpoint_dir: Path,
    *,
    step: int,
    cursor: int,
    stage0: dict[str, Any],
    stage1: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema": GLOBAL_CHECKPOINT_SCHEMA,
        "global_step": int(step),
        "outer_step": int(step),
        "dataset_cursor": int(cursor),
        "stages": [stage0, stage1],
        "stage_count": 2,
        "complete": True,
    }
    manifest["content_hash"] = sha256_json(
        {
            "schema": manifest["schema"],
            "global_step": manifest["global_step"],
            "outer_step": manifest["outer_step"],
            "dataset_cursor": manifest["dataset_cursor"],
            "stage_hashes": [stage0["content_hash"], stage1["content_hash"]],
        }
    )
    path = _write_json(checkpoint_dir / "global_checkpoint.json", manifest)
    return {**manifest, "manifest_path": str(path.resolve())}


def deterministic_pipeline_rows(*, row_count: int = 16, sequence_length: int = 12) -> list[list[int]]:
    return [
        [1] + [3 + ((position + index % 4) % 8) for position in range(sequence_length - 2)] + [2]
        for index in range(row_count)
    ]


def run_two_process_pipeline(
    output_dir: str | Path,
    *,
    total_steps: int = 12,
    interrupt_stage1_after_step: int | None = None,
    seed: int = 20260710,
) -> dict[str, Any]:
    """Run at most two workers, optionally hard-stop/restart stage1 mid-run."""

    started = time.monotonic()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoint"
    config = {
        "seed": int(seed),
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 4,
        "num_layers": 4,
        "split_index": 2,
        "max_positions": 32,
        "lora_rank": 4,
        "lora_alpha": 8,
        "learning_rate": 0.12,
        "batch_size": 2,
        "sequence_length": 12,
        "device": "cpu",
    }
    rows = deterministic_pipeline_rows(sequence_length=config["sequence_length"])
    context = mp.get_context("spawn")
    processes: dict[int, Any] = {}
    connections: dict[int, Any] = {}
    ready: dict[int, dict[str, Any]] = {}
    stage_records: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    losses: list[float] = []
    cursor = 0
    interruption: dict[str, Any] = {
        "requested": interrupt_stage1_after_step is not None,
        "performed": False,
        "stage_id": 1,
    }
    cleanup: dict[str, Any] = {}
    try:
        for stage_id in (0, 1):
            process, connection, stage_ready = _start_stage(
                context,
                stage_id,
                config,
                checkpoint_dir,
                resume=False,
            )
            processes[stage_id] = process
            connections[stage_id] = connection
            ready[stage_id] = stage_ready
        for step in range(int(total_steps)):
            batch = [rows[(cursor + offset) % len(rows)] for offset in range(config["batch_size"])]
            next_cursor = (cursor + config["batch_size"]) % len(rows)
            connections[0].send(
                {"command": "forward", "step": step, "cursor": next_cursor, "input_ids": batch}
            )
            stage0_forward = _recv(connections[0])
            connections[1].send(
                {
                    "command": "forward_backward",
                    "step": step,
                    "cursor": next_cursor,
                    "activation": stage0_forward["activation"],
                    "labels": batch,
                }
            )
            stage1_backward = _recv(connections[1])
            connections[0].send(
                {
                    "command": "backward",
                    "step": step,
                    "cursor": next_cursor,
                    "activation_gradient": stage1_backward["activation_gradient"],
                }
            )
            stage0_backward = _recv(connections[0])
            if stage0_backward["backward_gradient_hash"] != stage1_backward["backward_gradient_hash"]:
                raise RuntimeError("activation gradient changed in stage1-to-stage0 transport")
            losses.append(float(stage1_backward["loss"]))
            stage_records[0].append(
                {
                    "step": step,
                    "forward_hash": stage0_forward["forward_hash"],
                    "backward_gradient_hash": stage0_backward["backward_gradient_hash"],
                    "lora_gradient_norm": stage0_backward["lora_gradient_norm"],
                    "optimizer_step": stage0_backward["optimizer_step"],
                    "checkpoint_hash": stage0_backward["checkpoint_hash"],
                }
            )
            stage_records[1].append(
                {
                    "step": step,
                    "forward_hash": stage1_backward["logits_hash"],
                    "backward_gradient_hash": stage1_backward["backward_gradient_hash"],
                    "lora_gradient_norm": stage1_backward["lora_gradient_norm"],
                    "optimizer_step": stage1_backward["optimizer_step"],
                    "checkpoint_hash": stage1_backward["checkpoint_hash"],
                    "loss": stage1_backward["loss"],
                }
            )
            global_checkpoint = _global_checkpoint(
                checkpoint_dir,
                step=step + 1,
                cursor=next_cursor,
                stage0=stage0_backward["checkpoint"],
                stage1=stage1_backward["checkpoint"],
            )
            cursor = next_cursor
            if interrupt_stage1_after_step is not None and step + 1 == int(interrupt_stage1_after_step):
                old_pid = processes[1].pid
                processes[1].terminate()
                processes[1].join(timeout=10.0)
                old_exitcode = processes[1].exitcode
                connections[1].close()
                process, connection, stage_ready = _start_stage(
                    context,
                    1,
                    config,
                    checkpoint_dir,
                    resume=True,
                )
                processes[1] = process
                connections[1] = connection
                interruption = {
                    "requested": True,
                    "performed": True,
                    "stage_id": 1,
                    "after_step": step + 1,
                    "old_pid": old_pid,
                    "old_exitcode": old_exitcode,
                    "new_pid": process.pid,
                    "worker_restarted": process.pid != old_pid,
                    "checkpoint_loaded": bool(stage_ready.get("resumed")),
                    "resumed_optimizer_step": int(stage_ready.get("resumed_optimizer_step", 0)),
                    "resumed_dataset_cursor": int(stage_ready.get("resumed_cursor", 0)),
                }
        final_checkpoint = global_checkpoint
        status: dict[int, dict[str, Any]] = {}
        for stage_id in (0, 1):
            connections[stage_id].send({"command": "status"})
            status[stage_id] = _recv(connections[stage_id])
        for stage_id in (0, 1):
            cleanup[f"stage{stage_id}_exitcode"] = _stop_stage(processes[stage_id], connections[stage_id])
        cleanup["all_worker_processes_stopped"] = all(not process.is_alive() for process in processes.values())
        elapsed = time.monotonic() - started
        report = {
            "schema": PIPELINE_SCHEMA,
            "configuration": config,
            "process_count": 2,
            "independent_worker_processes": True,
            "stage_ownership": [ready[0]["ownership"], ready[1]["ownership"]],
            "no_stage_loaded_full_model": all(
                not value["loaded_full_model"] for value in [ready[0]["ownership"], ready[1]["ownership"]]
            ),
            "real_activation_transport": True,
            "real_backward_gradient_transport": True,
            "real_pytorch_autograd": True,
            "real_peft_lora": True,
            "total_steps": int(total_steps),
            "loss_history": losses,
            "loss_start": losses[0],
            "loss_end": losses[-1],
            "loss_reduced": losses[-1] < losses[0],
            "stage_records": {str(key): value for key, value in stage_records.items()},
            "positive_lora_gradient_norms": all(
                record["lora_gradient_norm"] > 0.0
                for records in stage_records.values()
                for record in records
            ),
            "base_weights_frozen": all(
                value["base_hash"] == status[index]["base_hash_after"]
                for index, value in ready.items()
            ),
            "interruption": interruption,
            "final_checkpoint": final_checkpoint,
            "elapsed_seconds": elapsed,
            "cleanup": cleanup,
            "device": "cpu",
            "gpu_live_verified": False,
        }
        report_path = _write_json(output / "pipeline_training_report.json", report)
        return {**report, "report_path": str(report_path.resolve())}
    finally:
        for stage_id, process in processes.items():
            if process.is_alive():
                process.terminate()
                process.join(timeout=10.0)
            connection = connections.get(stage_id)
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass


def _public_stage_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in checkpoint.items()
        if not key.endswith("_path") and key != "manifest_path"
    }


def _public_cuda_pipeline_report(report: dict[str, Any]) -> dict[str, Any]:
    public = {
        key: value
        for key, value in report.items()
        if key not in {"final_checkpoint", "private_report_path"} and not key.endswith("_path")
    }
    checkpoint = dict(report.get("final_checkpoint") or {})
    public["final_checkpoint"] = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"stages", "manifest_path"} and not key.endswith("_path")
    }
    public["final_checkpoint"]["stages"] = [
        _public_stage_checkpoint(dict(stage)) for stage in checkpoint.get("stages") or []
    ]
    public.update(
        {
            "activation_values_public": False,
            "gradient_values_public": False,
            "raw_training_text_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
    )
    return public


def run_two_cuda_process_pipeline(
    output_dir: str | Path,
    *,
    total_steps: int = 4,
    interrupt_stage1_after_step: int | None = None,
    seed: int = 20260710,
) -> dict[str, Any]:
    """Run a bounded two-process pipeline on distinct CUDA devices."""

    import torch

    if int(total_steps) < 4:
        raise ValueError("CUDA pipeline gate requires at least four optimizer steps")
    if not torch.cuda.is_available() or int(torch.cuda.device_count()) < 2:
        raise RuntimeError("cuda_pipeline_requires_two_visible_devices")
    started = time.monotonic()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoint"
    config = {
        "seed": int(seed),
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 4,
        "num_layers": 4,
        "split_index": 2,
        "max_positions": 32,
        "lora_rank": 4,
        "lora_alpha": 8,
        "learning_rate": 0.08,
        "gradient_clip_norm": 1.0,
        "batch_size": 2,
        "sequence_length": 12,
        "precision": "fp16_autocast",
        "placements": ["cuda:0", "cuda:1"],
    }
    rows = deterministic_pipeline_rows(sequence_length=config["sequence_length"])
    context = mp.get_context("spawn")
    processes: dict[int, Any] = {}
    connections: dict[int, Any] = {}
    initial_ready: dict[int, dict[str, Any]] = {}
    stage_records: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    losses: list[float] = []
    cursor = 0
    interruption: dict[str, Any] = {
        "requested": interrupt_stage1_after_step is not None,
        "performed": False,
        "stage_id": 1,
    }
    cleanup: dict[str, Any] = {}
    try:
        for stage_id in (0, 1):
            process, connection, stage_ready = _start_cuda_stage(
                context,
                stage_id,
                config,
                checkpoint_dir,
                resume=False,
            )
            processes[stage_id] = process
            connections[stage_id] = connection
            initial_ready[stage_id] = dict(stage_ready)
        for step in range(int(total_steps)):
            batch = [rows[(cursor + offset) % len(rows)] for offset in range(config["batch_size"])]
            next_cursor = (cursor + config["batch_size"]) % len(rows)
            connections[0].send(
                {"command": "forward", "step": step, "cursor": next_cursor, "input_ids": batch}
            )
            stage0_forward = _recv(connections[0], timeout=120.0)
            connections[1].send(
                {
                    "command": "forward_backward",
                    "step": step,
                    "cursor": next_cursor,
                    "activation": stage0_forward["activation"],
                    "labels": batch,
                }
            )
            stage1_backward = _recv(connections[1], timeout=120.0)
            connections[0].send(
                {
                    "command": "backward",
                    "step": step,
                    "cursor": next_cursor,
                    "activation_gradient": stage1_backward["activation_gradient"],
                    "gradient_scale": stage1_backward["gradient_scale"],
                }
            )
            stage0_backward = _recv(connections[0], timeout=120.0)
            if stage0_backward["backward_gradient_hash"] != stage1_backward["backward_gradient_hash"]:
                raise RuntimeError("CUDA activation gradient changed in private CPU transport")
            losses.append(float(stage1_backward["loss"]))
            stage_records[0].append(
                {
                    "step": step,
                    "pid": int(processes[0].pid),
                    "cuda_device": "cuda:0",
                    "forward_hash": stage0_forward["forward_hash"],
                    "backward_gradient_hash": stage0_backward["backward_gradient_hash"],
                    "lora_gradient_norm": stage0_backward["lora_gradient_norm"],
                    "optimizer_step": stage0_backward["optimizer_step"],
                    "checkpoint_hash": stage0_backward["checkpoint_hash"],
                    "peak_allocated_bytes": stage0_backward["peak_allocated_bytes"],
                    "peak_reserved_bytes": stage0_backward["peak_reserved_bytes"],
                }
            )
            stage_records[1].append(
                {
                    "step": step,
                    "pid": int(processes[1].pid),
                    "cuda_device": "cuda:1",
                    "forward_hash": stage1_backward["logits_hash"],
                    "backward_gradient_hash": stage1_backward["backward_gradient_hash"],
                    "lora_gradient_norm": stage1_backward["lora_gradient_norm"],
                    "optimizer_step": stage1_backward["optimizer_step"],
                    "checkpoint_hash": stage1_backward["checkpoint_hash"],
                    "loss": stage1_backward["loss"],
                    "peak_allocated_bytes": stage1_backward["peak_allocated_bytes"],
                    "peak_reserved_bytes": stage1_backward["peak_reserved_bytes"],
                }
            )
            global_checkpoint = _global_checkpoint(
                checkpoint_dir,
                step=step + 1,
                cursor=next_cursor,
                stage0=stage0_backward["checkpoint"],
                stage1=stage1_backward["checkpoint"],
            )
            cursor = next_cursor
            if interrupt_stage1_after_step is not None and step + 1 == int(interrupt_stage1_after_step):
                old_pid = int(processes[1].pid)
                processes[1].terminate()
                processes[1].join(timeout=15.0)
                old_exitcode = processes[1].exitcode
                connections[1].close()
                process, connection, stage_ready = _start_cuda_stage(
                    context,
                    1,
                    config,
                    checkpoint_dir,
                    resume=True,
                )
                processes[1] = process
                connections[1] = connection
                interruption = {
                    "requested": True,
                    "performed": True,
                    "stage_id": 1,
                    "after_step": step + 1,
                    "old_pid": old_pid,
                    "old_exitcode": old_exitcode,
                    "new_pid": int(process.pid),
                    "worker_restarted": int(process.pid) != old_pid,
                    "checkpoint_loaded": bool(stage_ready.get("resumed")),
                    "resumed_optimizer_step": int(stage_ready.get("resumed_optimizer_step", 0)),
                    "resumed_dataset_cursor": int(stage_ready.get("resumed_cursor", 0)),
                    "grad_scaler_state_present": True,
                }
        final_checkpoint = global_checkpoint
        statuses: dict[int, dict[str, Any]] = {}
        for stage_id in (0, 1):
            connections[stage_id].send({"command": "status"})
            statuses[stage_id] = _recv(connections[stage_id], timeout=60.0)
        for stage_id in (0, 1):
            cleanup[f"stage{stage_id}_exitcode"] = _stop_stage(processes[stage_id], connections[stage_id])
        cleanup["all_worker_processes_stopped"] = all(
            not process.is_alive() for process in processes.values()
        )
        stage_pids = sorted(
            {
                int(record["pid"])
                for records in stage_records.values()
                for record in records
            }
        )
        report = {
            "schema": CUDA_PIPELINE_SCHEMA,
            "configuration": config,
            "process_count": 2,
            "independent_worker_processes": True,
            "stage_pids": stage_pids,
            "distinct_stage_pids": len({initial_ready[0]["pid"], initial_ready[1]["pid"]}) == 2,
            "cuda_devices": [initial_ready[0]["cuda_device"], initial_ready[1]["cuda_device"]],
            "distinct_cuda_devices": initial_ready[0]["cuda_device"] != initial_ready[1]["cuda_device"],
            "cuda_device_name_hashes": [
                initial_ready[0]["cuda_device_name_hash"],
                initial_ready[1]["cuda_device_name_hash"],
            ],
            "stage_ownership": [initial_ready[0]["ownership"], initial_ready[1]["ownership"]],
            "no_stage_loaded_full_model": all(
                not value["ownership"]["loaded_full_model"] for value in initial_ready.values()
            ),
            "real_activation_transport": True,
            "private_cpu_activation_transport": True,
            "real_backward_gradient_transport": True,
            "real_pytorch_autograd": True,
            "real_peft_lora": True,
            "real_cuda_forward": True,
            "real_cuda_backward": True,
            "fp16_autocast": True,
            "grad_scaler": True,
            "gradient_clipping": True,
            "total_steps": int(total_steps),
            "loss_history": losses,
            "loss_start": losses[0],
            "loss_end": losses[-1],
            "loss_reduced": losses[-1] < losses[0],
            "stage_records": {str(key): value for key, value in stage_records.items()},
            "positive_lora_gradient_norms": all(
                record["lora_gradient_norm"] > 0.0
                for records in stage_records.values()
                for record in records
            ),
            "positive_cuda_memory": all(
                record["peak_allocated_bytes"] > 0
                for records in stage_records.values()
                for record in records
            ),
            "base_weights_frozen": all(
                initial_ready[index]["base_hash"] == statuses[index]["base_hash_after"]
                for index in (0, 1)
            ),
            "interruption": interruption,
            "final_checkpoint": final_checkpoint,
            "elapsed_seconds": time.monotonic() - started,
            "cleanup": cleanup,
            "device": "cuda",
            "cuda_used": True,
            "gpu_live_verified": True,
            "kaggle_kernel": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")),
        }
        private_report_path = _write_json(output / "cuda_pipeline_report_private.json", report)
        public_report = _public_cuda_pipeline_report(report)
        public_report_path = _write_json(output / "cuda_pipeline_report.json", public_report)
        return {
            **report,
            "private_report_path": str(private_report_path.resolve()),
            "report_path": str(public_report_path.resolve()),
        }
    finally:
        for stage_id, process in processes.items():
            if process.is_alive():
                process.terminate()
                process.join(timeout=15.0)
            connection = connections.get(stage_id)
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass


def compare_pipeline_runs(
    baseline: dict[str, Any],
    resumed: dict[str, Any],
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    comparisons: list[dict[str, Any]] = []
    all_close = True
    max_difference = 0.0
    for stage_id in (0, 1):
        left_manifest = baseline["final_checkpoint"]["stages"][stage_id]
        right_manifest = resumed["final_checkpoint"]["stages"][stage_id]
        left = load_file(left_manifest["adapter_path"], device="cpu")
        right = load_file(right_manifest["adapter_path"], device="cpu")
        names_match = set(left) == set(right)
        stage_close = names_match
        stage_max = 0.0
        if names_match:
            for name in left:
                difference = float((left[name].float() - right[name].float()).abs().max().item())
                stage_max = max(stage_max, difference)
                stage_close = stage_close and bool(torch.allclose(left[name], right[name], atol=atol, rtol=rtol))
        all_close = all_close and stage_close
        max_difference = max(max_difference, stage_max)
        comparisons.append(
            {
                "stage_id": stage_id,
                "tensor_names_match": names_match,
                "adapter_tensors_close": stage_close,
                "max_abs_difference": stage_max,
            }
        )
    final_loss_difference = abs(float(baseline["loss_end"]) - float(resumed["loss_end"]))
    return {
        "schema": "crowdtensor_pipeline_resume_equivalence_v1",
        "adapter_tensors_close": all_close,
        "max_abs_difference": max_difference,
        "atol": float(atol),
        "rtol": float(rtol),
        "final_loss_difference": final_loss_difference,
        "final_loss_close": final_loss_difference <= atol + rtol * abs(float(baseline["loss_end"])),
        "stages": comparisons,
        "controlled_interruption_performed": bool((resumed.get("interruption") or {}).get("performed")),
        "checkpoint_resume_verified": bool(
            all_close
            and final_loss_difference <= atol + rtol * abs(float(baseline["loss_end"]))
            and (resumed.get("interruption") or {}).get("performed")
            and (resumed.get("interruption") or {}).get("checkpoint_loaded")
        ),
    }
