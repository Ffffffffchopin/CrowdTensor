"""Real CPU Transformers/PEFT LoRA training runtime for CrowdTensor."""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Protocol

from .named_tensor_optimizer import adapter_delta, load_tensors, named_tensor_hash, save_tensors
from .training_contract import (
    DATASET_SCHEMA,
    JOB_SCHEMA,
    LORA_SCHEMA,
    MODEL_SCHEMA,
    RESULT_SCHEMA,
    TRAINING_SPEC_SCHEMA,
    WORKLOAD_TYPE,
    delta_manifest,
    public_training_spec,
    sha256_file,
    sha256_json,
    tensor_bytes,
    tensor_specs,
)


RUNTIME_SCHEMA = "crowdtensor_hf_lora_cpu_runtime_v1"
CUDA_RUNTIME_SCHEMA = "crowdtensor_hf_lora_cuda_runtime_v1"
CUDA_CHECKPOINT_SCHEMA = "crowdtensor_hf_lora_cuda_checkpoint_v1"
CUDA_BLOCKER_SCHEMA = "crowdtensor_hf_lora_cuda_blocker_v1"
EVALUATION_SCHEMA = "crowdtensor_lora_evaluation_v1"
FIXTURE_MODEL_ID = "crowdtensor-local-tiny-llama"


class CUDAOutOfMemoryError(RuntimeError):
    """Public-safe CUDA OOM classification raised by the training runtime."""

    code = "cuda_training_out_of_memory"


class CUDAUnavailableError(RuntimeError):
    """Raised before model loading when the requested CUDA placement is unavailable."""

    code = "cuda_training_device_unavailable"


class TrainingRuntime(Protocol):
    """Device-neutral local training interface used by Miner workloads."""

    backend: str

    def run(self, training_spec: dict[str, Any], *, output_dir: str | Path) -> dict[str, Any]: ...

    def capability(self) -> dict[str, Any]: ...


def _deps() -> tuple[Any, Any, Any, Any, Any, Any]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    return torch, LoraConfig, PeftModel, get_peft_model, LlamaConfig, LlamaForCausalLM


def configure_cpu_determinism(seed: int) -> Any:
    torch, *_ = _deps()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.set_num_threads(max(1, min(2, int(os.environ.get("CROWDTENSOR_CPU_THREADS", "1")))))
    torch.use_deterministic_algorithms(True)
    return torch


def configure_cuda_determinism(seed: int) -> Any:
    """Configure deterministic CUDA execution without changing visible devices."""

    torch, *_ = _deps()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _adapter_tensor_path(adapter_dir: str | Path) -> Path:
    directory = Path(adapter_dir)
    safe = directory / "adapter_model.safetensors"
    if safe.is_file():
        return safe
    raise FileNotFoundError(f"PEFT adapter_model.safetensors not found in {directory}")


def _base_state(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if "lora_" not in name and ".adapter" not in name
    }


def model_state_hash(model: Any, *, base_only: bool = False) -> str:
    values = _base_state(model) if base_only else {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    return sha256_json(tensor_specs(values))


def _fixture_rows(*, row_count: int, sequence_length: int, vocab_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        phase = index % 4
        tokens = [1]
        for position in range(sequence_length - 2):
            tokens.append(3 + ((position + phase) % min(8, vocab_size - 3)))
        tokens.append(2)
        rows.append({"sample_id": f"sample-{index:04d}", "input_ids": tokens})
    return rows


def create_local_training_fixture(
    output_dir: str | Path,
    *,
    job_id: str = "training-foundation-cpu-rc",
    seed: int = 20260710,
    row_count: int = 24,
    sequence_length: int = 16,
    local_steps: int = 12,
    learning_rate: float = 0.08,
    batch_size: int = 2,
    gradient_accumulation: int = 1,
) -> dict[str, Any]:
    """Create a fully local Transformers model, initial PEFT adapter and sharded data."""

    torch, LoraConfig, _PeftModel, get_peft_model, LlamaConfig, LlamaForCausalLM = _deps()
    configure_cpu_determinism(seed)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model_dir = root / "base_model"
    initial_adapter_dir = root / "initial_adapter"
    dataset_path = root / "private_dataset.jsonl"

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )
    model = LlamaForCausalLM(config)
    parameter_count = sum(int(value.numel()) for value in model.parameters())
    if parameter_count > 200_000_000:
        raise RuntimeError("local training fixture exceeds the 200M parameter boundary")
    model.save_pretrained(model_dir, safe_serialization=True)
    base_model_hash = model_state_hash(model)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    adapter_model = get_peft_model(model, lora_config)
    adapter_model.save_pretrained(initial_adapter_dir, safe_serialization=True)
    adapter_tensor_path = _adapter_tensor_path(initial_adapter_dir)
    initial_tensors = load_tensors(adapter_tensor_path)
    adapter_specs = tensor_specs(initial_tensors)
    base_adapter_hash = sha256_json(adapter_specs)

    rows = _fixture_rows(row_count=row_count, sequence_length=sequence_length, vocab_size=config.vocab_size)
    dataset_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    shard_indexes = [list(range(0, row_count, 2)), list(range(1, row_count, 2))]
    shard_manifests: list[dict[str, Any]] = []
    for shard_index, indexes in enumerate(shard_indexes):
        shard_public = {
            "schema": DATASET_SCHEMA,
            "dataset_id": f"{job_id}-dataset",
            "dataset_version": 1,
            "shard_index": shard_index,
            "sample_indexes": indexes,
            "sample_count": len(indexes),
            "token_count": sum(len(rows[index]["input_ids"]) for index in indexes),
            "data_cursor_start": 0,
            "raw_text_public": False,
        }
        shard_public["shard_hash"] = sha256_json(shard_public)
        shard_manifests.append(shard_public)
    dataset_manifest = {
        "schema": DATASET_SCHEMA,
        "dataset_id": f"{job_id}-dataset",
        "dataset_version": 1,
        "format": "deterministic_tokenized_jsonl",
        "sample_count": len(rows),
        "token_count": sum(len(row["input_ids"]) for row in rows),
        "sequence_length": sequence_length,
        "shard_count": 2,
        "shards": shard_manifests,
        "dataset_file_hash": sha256_file(dataset_path),
        "raw_text_public": False,
        "private_dataset_path": str(dataset_path),
    }
    dataset_manifest["manifest_hash"] = sha256_json(
        {key: value for key, value in dataset_manifest.items() if not key.endswith("_path")}
    )
    model_manifest = {
        "schema": MODEL_SCHEMA,
        "model_id": FIXTURE_MODEL_ID,
        "model_version": 1,
        "architecture": "LlamaForCausalLM",
        "parameter_count": parameter_count,
        "dtype": "float32",
        "base_model_hash": base_model_hash,
        "config_hash": sha256_json(config.to_dict()),
        "max_parameters": 200_000_000,
        "source": "local_deterministic_transformers_fixture",
        "base_model_path": str(model_dir),
    }
    model_manifest["manifest_hash"] = sha256_json(
        {key: value for key, value in model_manifest.items() if not key.endswith("_path")}
    )
    lora_manifest = {
        "schema": LORA_SCHEMA,
        "adapter_version": 0,
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules": list(lora_config.target_modules),
        "trainable_parameter_count": sum(
            int(value.numel()) for name, value in adapter_model.named_parameters() if value.requires_grad
        ),
        "tensor_specs": adapter_specs,
        "base_adapter_hash": base_adapter_hash,
        "adapter_path": str(initial_adapter_dir),
        "adapter_tensor_path": str(adapter_tensor_path),
        "adapter_config_path": str(initial_adapter_dir / "adapter_config.json"),
    }
    lora_manifest["manifest_hash"] = sha256_json(
        {key: value for key, value in lora_manifest.items() if not key.endswith("_path")}
    )
    job = {
        "schema": JOB_SCHEMA,
        "job_id": job_id,
        "job_version": 1,
        "workload_type": WORKLOAD_TYPE,
        "permission_mode": "permissioned_trusted_local_workers",
        "backend": "pytorch_transformers_peft_cpu",
        "seed": int(seed),
        "model": model_manifest,
        "dataset": dataset_manifest,
        "lora": lora_manifest,
        "local_training": {
            "local_steps": int(local_steps),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "sequence_length": int(sequence_length),
            "gradient_accumulation": int(gradient_accumulation),
            "optimizer": "adamw",
            "optimizer_contract": "torch_adamw_v1",
            "step_start": 0,
            "step_end": int(local_steps),
        },
        "outer_optimizer": {
            "schema": "crowdtensor_named_tensor_outer_optimizer_v1",
            "optimizer_type": "diloco_momentum",
            "outer_lr": 1.0,
            "momentum": 0.0,
            "outer_step": 0,
        },
        "private_paths_public": False,
        "raw_dataset_public": False,
        "gpu_live_verified": False,
    }
    job["job_hash"] = sha256_json(public_training_spec(job))
    job_path = _write_json(root / "training_job_private.json", job)
    public_path = _write_json(root / "training_job_public.json", public_training_spec(job))
    return {
        **job,
        "job_manifest_path": str(job_path),
        "public_job_manifest_path": str(public_path),
    }


def load_token_rows(dataset_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(dataset_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("input_ids"), list):
                raise ValueError("dataset row must contain an input_ids list")
            rows.append(value)
    if not rows:
        raise ValueError("training dataset is empty")
    return rows


def training_spec_for_claim(
    job: dict[str, Any],
    *,
    task_id: str,
    miner_id: str,
    shard_index: int,
    device: str | None = None,
) -> dict[str, Any]:
    shards = list((job.get("dataset") or {}).get("shards") or [])
    if shard_index < 0 or shard_index >= len(shards):
        raise ValueError("dataset shard index is out of range")
    shard = dict(shards[shard_index])
    local = dict(job.get("local_training") or {})
    spec = {
        "schema": TRAINING_SPEC_SCHEMA,
        "workload_type": WORKLOAD_TYPE,
        "job_id": job["job_id"],
        "job_hash": job["job_hash"],
        "round_id": f"{job['job_id']}-round-0001",
        "task_id": task_id,
        "miner_id": miner_id,
        "model_manifest_hash": job["model"]["manifest_hash"],
        "base_model_hash": job["model"]["base_model_hash"],
        "base_model_version": int(job["model"]["model_version"]),
        "base_model_path": job["model"]["base_model_path"],
        "base_adapter_hash": job["lora"]["base_adapter_hash"],
        "adapter_version": int(job["lora"]["adapter_version"]),
        "adapter_path": job["lora"]["adapter_path"],
        "adapter_tensor_path": job["lora"]["adapter_tensor_path"],
        "adapter_config_path": job["lora"]["adapter_config_path"],
        "dataset_path": job["dataset"]["private_dataset_path"],
        "dataset_manifest_hash": job["dataset"]["manifest_hash"],
        "dataset_shard_index": int(shard_index),
        "dataset_shard_hash": shard["shard_hash"],
        "sample_indexes": list(shard["sample_indexes"]),
        "sample_count": int(shard["sample_count"]),
        "token_count": int(shard["token_count"]),
        "data_cursor": int(shard.get("data_cursor_start", 0)),
        "seed": int(job["seed"]) + int(shard_index),
        "step_start": int(local.get("step_start", 0)),
        "step_end": int(local["step_end"]),
        "local_steps": int(local["local_steps"]),
        "learning_rate": float(local["learning_rate"]),
        "batch_size": int(local["batch_size"]),
        "sequence_length": int(local["sequence_length"]),
        "gradient_accumulation": int(local["gradient_accumulation"]),
        "optimizer_contract": str(local["optimizer_contract"]),
        "device": str(device or job.get("device") or "cpu"),
        "trusted_worker_required": True,
        "raw_dataset_public": False,
    }
    spec["claim_hash"] = sha256_json(public_training_spec(spec))
    return spec


def _batch(
    rows: list[dict[str, Any]],
    indexes: list[int],
    cursor: int,
    batch_size: int,
    torch: Any,
    *,
    device: str = "cpu",
) -> Any:
    selected = [rows[indexes[(cursor + offset) % len(indexes)]]["input_ids"] for offset in range(batch_size)]
    return torch.tensor(selected, dtype=torch.long, device=device)


def evaluate_adapter(
    *,
    base_model_path: str | Path,
    adapter_path: str | Path | None,
    dataset_path: str | Path,
    sample_indexes: list[int],
    batch_size: int = 2,
) -> dict[str, Any]:
    torch, _LoraConfig, PeftModel, _get_peft_model, _LlamaConfig, LlamaForCausalLM = _deps()
    model = LlamaForCausalLM.from_pretrained(base_model_path, local_files_only=True)
    adapter_loaded = bool(adapter_path)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False, local_files_only=True)
    model.to("cpu")
    model.eval()
    rows = load_token_rows(dataset_path)
    losses: list[float] = []
    first_logits = None
    with torch.no_grad():
        for start in range(0, len(sample_indexes), max(1, batch_size)):
            indexes = sample_indexes[start : start + max(1, batch_size)]
            input_ids = torch.tensor([rows[index]["input_ids"] for index in indexes], dtype=torch.long)
            output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            losses.append(float(output.loss.item()))
            if first_logits is None:
                first_logits = output.logits[0, -1].detach().cpu().contiguous()
    logits = first_logits if first_logits is not None else torch.empty(0)
    return {
        "schema": EVALUATION_SCHEMA,
        "adapter_loaded": adapter_loaded,
        "sample_count": len(sample_indexes),
        "mean_loss": sum(losses) / max(1, len(losses)),
        "logits_hash": "sha256:" + __import__("hashlib").sha256(tensor_bytes(logits)).hexdigest(),
        "logits_norm": float(logits.float().norm().item()),
        "device": "cpu",
    }


class CPULoRATrainingRuntime:
    backend = "pytorch_transformers_peft_cpu"

    def capability(self) -> dict[str, Any]:
        torch, *_ = _deps()
        return {
            "schema": RUNTIME_SCHEMA,
            "backend": self.backend,
            "device": "cpu",
            "real_pytorch_autograd": True,
            "real_transformers": True,
            "real_peft_lora": True,
            "torch_version": str(torch.__version__),
            "cuda_used": False,
            "gpu_live_verified": False,
        }

    def run(self, training_spec: dict[str, Any], *, output_dir: str | Path) -> dict[str, Any]:
        if training_spec.get("schema") != TRAINING_SPEC_SCHEMA:
            raise ValueError("unsupported HF LoRA training spec")
        if training_spec.get("device") != "cpu":
            raise ValueError("Training Foundation RC only validates the CPU backend")
        torch, _LoraConfig, PeftModel, _get_peft_model, _LlamaConfig, LlamaForCausalLM = _deps()
        seed = int(training_spec["seed"])
        configure_cpu_determinism(seed)
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()

        base = LlamaForCausalLM.from_pretrained(training_spec["base_model_path"], local_files_only=True)
        model = PeftModel.from_pretrained(
            base,
            training_spec["adapter_path"],
            is_trainable=True,
            local_files_only=True,
        )
        model.to("cpu")
        trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
        if not trainable or any("lora_" not in name for name, _value in trainable):
            raise RuntimeError("base model is not fully frozen for LoRA training")
        base_hash_before = model_state_hash(model, base_only=True)
        rows = load_token_rows(training_spec["dataset_path"])
        indexes = [int(value) for value in training_spec["sample_indexes"]]
        if not indexes:
            raise ValueError("dataset shard has no sample indexes")
        optimizer = torch.optim.AdamW(
            [value for _name, value in trainable],
            lr=float(training_spec["learning_rate"]),
            weight_decay=0.0,
        )
        batch_size = int(training_spec["batch_size"])
        grad_acc = int(training_spec["gradient_accumulation"])
        local_steps = int(training_spec["local_steps"])
        cursor = int(training_spec["data_cursor"])
        losses: list[float] = []
        samples_seen = 0
        tokens_seen = 0
        model.train()
        for _step in range(local_steps):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _micro in range(grad_acc):
                input_ids = _batch(rows, indexes, cursor, batch_size, torch)
                cursor = (cursor + batch_size) % len(indexes)
                result = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                loss = result.loss
                (loss / float(grad_acc)).backward()
                step_loss += float(loss.detach().item())
                samples_seen += batch_size
                tokens_seen += int(input_ids.numel())
            optimizer.step()
            losses.append(step_loss / float(grad_acc))

        base_hash_after = model_state_hash(model, base_only=True)
        if base_hash_before != base_hash_after:
            raise RuntimeError("base model weights changed during LoRA-only training")
        final_adapter_dir = output / "adapter"
        model.save_pretrained(final_adapter_dir, safe_serialization=True)
        initial_path = Path(training_spec["adapter_tensor_path"])
        final_path = _adapter_tensor_path(final_adapter_dir)
        initial_tensors = load_tensors(initial_path)
        final_tensors = load_tensors(final_path)
        delta = adapter_delta(initial_tensors, final_tensors)
        delta_path = save_tensors(delta, output / "adapter_delta.safetensors")
        result_id = sha256_json(
            {
                "task_id": training_spec["task_id"],
                "miner_id": training_spec["miner_id"],
                "claim_hash": training_spec["claim_hash"],
                "delta_hash": named_tensor_hash(delta),
            }
        )
        manifest = delta_manifest(
            delta_path=delta_path,
            job_id=training_spec["job_id"],
            round_id=training_spec["round_id"],
            result_id=result_id,
            miner_id=training_spec["miner_id"],
            model_manifest_hash=training_spec["model_manifest_hash"],
            base_model_hash=training_spec["base_model_hash"],
            base_adapter_hash=training_spec["base_adapter_hash"],
            base_model_version=int(training_spec["base_model_version"]),
            adapter_version=int(training_spec["adapter_version"]),
            dataset_shard_index=int(training_spec["dataset_shard_index"]),
            dataset_shard_hash=training_spec["dataset_shard_hash"],
            loss_start=float(losses[0]),
            loss_end=float(losses[-1]),
            samples_seen=samples_seen,
            tokens_seen=tokens_seen,
        )
        elapsed = time.monotonic() - started
        private_result = {
            "schema": RESULT_SCHEMA,
            "workload_type": WORKLOAD_TYPE,
            "result_id": result_id,
            "job_id": training_spec["job_id"],
            "round_id": training_spec["round_id"],
            "task_id": training_spec["task_id"],
            "miner_id": training_spec["miner_id"],
            "claim_hash": training_spec["claim_hash"],
            "dataset_shard_index": int(training_spec["dataset_shard_index"]),
            "dataset_shard_hash": training_spec["dataset_shard_hash"],
            "adapter_delta": manifest,
            "adapter_path": str(final_adapter_dir),
            "adapter_tensor_path": str(final_path),
            "adapter_tensor_hash": named_tensor_hash(final_tensors),
            "base_weights_frozen": True,
            "base_hash_before": base_hash_before,
            "base_hash_after": base_hash_after,
            "only_lora_trainable": True,
            "real_backward": True,
            "optimizer_steps": local_steps,
            "loss_history": losses,
            "loss_start": float(losses[0]),
            "loss_end": float(losses[-1]),
            "loss_reduced": bool(losses[-1] < losses[0]),
            "samples_seen": samples_seen,
            "tokens_seen": tokens_seen,
            "data_cursor": cursor,
            "elapsed_seconds": elapsed,
            "runtime": self.capability(),
            "raw_dataset_public": False,
            "private_paths_public": False,
        }
        private_result_path = _write_json(output / "training_result_private.json", private_result)
        public = public_training_spec(private_result)
        public["adapter_delta"] = {
            key: value for key, value in manifest.items() if not key.endswith("_path")
        }
        public_result_path = _write_json(output / "training_result_public.json", public)
        return {
            **private_result,
            "private_result_path": str(private_result_path),
            "public_result_path": str(public_result_path),
        }


def _cuda_device_index(device: str) -> int:
    value = str(device or "").strip().lower()
    if value == "cuda":
        return 0
    if not value.startswith("cuda:"):
        raise ValueError("CUDA training device must use cuda:<index>")
    try:
        index = int(value.split(":", 1)[1])
    except ValueError as exc:
        raise ValueError("CUDA training device index must be an integer") from exc
    if index < 0:
        raise ValueError("CUDA training device index must be non-negative")
    return index


def _new_grad_scaler(torch: Any, *, enabled: bool = True) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older supported torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _optimizer_to_device(optimizer: Any, device: Any) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if hasattr(value, "to"):
                state[key] = value.to(device)


def _gradient_l2_norm(parameters: list[Any]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().norm().item()) ** 2
    return total ** 0.5


def _save_cuda_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    scaler: Any,
    output_dir: Path,
    optimizer_step: int,
    cursor: int,
    base_hash: str,
    device: str,
    outer_step: int,
) -> dict[str, Any]:
    torch, *_ = _deps()
    from safetensors.torch import save_file

    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = checkpoint_dir / "adapter.safetensors"
    runtime_path = checkpoint_dir / "runtime.pt"
    adapter = {
        name: value.detach().cpu().contiguous()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    save_file(adapter, str(adapter_path))
    runtime_state = {
        "optimizer": optimizer.state_dict(),
        "grad_scaler": scaler.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(_cuda_device_index(device)),
        "optimizer_step": int(optimizer_step),
        "dataset_cursor": int(cursor),
        "outer_step": int(outer_step),
    }
    torch.save(runtime_state, runtime_path)
    base_hash_after = model_state_hash(model, base_only=True)
    manifest = {
        "schema": CUDA_CHECKPOINT_SCHEMA,
        "optimizer_step": int(optimizer_step),
        "global_step": int(optimizer_step),
        "outer_step": int(outer_step),
        "dataset_cursor": int(cursor),
        "cuda_placement": str(device),
        "adapter_path": str(adapter_path.resolve()),
        "adapter_file_hash": sha256_file(adapter_path),
        "adapter_tensor_hash": named_tensor_hash(adapter),
        "adapter_tensor_specs": tensor_specs(adapter),
        "runtime_path": str(runtime_path.resolve()),
        "runtime_file_hash": sha256_file(runtime_path),
        "optimizer_state_present": True,
        "grad_scaler_state_present": True,
        "base_hash_before": base_hash,
        "base_hash_after": base_hash_after,
        "base_weights_frozen": base_hash == base_hash_after,
        "private_paths_public": False,
    }
    manifest["content_hash"] = sha256_json(
        {key: value for key, value in manifest.items() if not key.endswith("_path")}
    )
    manifest_path = _write_json(checkpoint_dir / "checkpoint.json", manifest)
    return {**manifest, "checkpoint_path": str(manifest_path.resolve())}


def _load_cuda_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    scaler: Any,
    checkpoint_path: str | Path,
    device: str,
) -> dict[str, Any]:
    torch, *_ = _deps()
    from safetensors.torch import load_file

    manifest = _read_json(checkpoint_path)
    if manifest.get("schema") != CUDA_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported CUDA LoRA checkpoint schema")
    adapter_path = Path(str(manifest.get("adapter_path") or ""))
    runtime_path = Path(str(manifest.get("runtime_path") or ""))
    if not adapter_path.is_file() or sha256_file(adapter_path) != manifest.get("adapter_file_hash"):
        raise RuntimeError("CUDA checkpoint adapter hash mismatch")
    if not runtime_path.is_file() or sha256_file(runtime_path) != manifest.get("runtime_file_hash"):
        raise RuntimeError("CUDA checkpoint runtime hash mismatch")
    adapter = load_file(str(adapter_path), device="cpu")
    incompatible = model.load_state_dict(adapter, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError("CUDA checkpoint contains unexpected adapter tensors")
    runtime_state = torch.load(runtime_path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(runtime_state["optimizer"])
    _optimizer_to_device(optimizer, torch.device(device))
    scaler.load_state_dict(runtime_state["grad_scaler"])
    torch.set_rng_state(runtime_state["cpu_rng_state"])
    torch.cuda.set_rng_state(runtime_state["cuda_rng_state"], _cuda_device_index(device))
    return manifest


class CUDALoRATrainingRuntime:
    """Real single-device CUDA Transformers/PEFT LoRA Miner runtime."""

    backend = "pytorch_transformers_peft_cuda"

    def __init__(
        self,
        device: str = "cuda:0",
        *,
        gradient_clip_norm: float = 1.0,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.device = str(device)
        self.device_index = _cuda_device_index(self.device)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.checkpoint_path = str(checkpoint_path or "")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")

    def capability(self) -> dict[str, Any]:
        torch, *_ = _deps()
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        placement_available = available and self.device_index < count
        device_name = torch.cuda.get_device_name(self.device_index) if placement_available else ""
        return {
            "schema": CUDA_RUNTIME_SCHEMA,
            "backend": self.backend,
            "device": self.device,
            "device_index": self.device_index,
            "cuda_available": available,
            "cuda_device_count": count,
            "placement_available": placement_available,
            "device_name_hash": sha256_json({"device_name": device_name}) if device_name else "",
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda or ""),
            "real_pytorch_autograd": True,
            "real_transformers": True,
            "real_peft_lora": True,
            "fp16_autocast": True,
            "grad_scaler": True,
            "gradient_clipping": True,
            "cuda_used": False,
            "gpu_live_verified": False,
            "dry_run_only": False,
        }

    def _require_device(self, torch: Any) -> Any:
        if not torch.cuda.is_available() or self.device_index >= int(torch.cuda.device_count()):
            raise CUDAUnavailableError("cuda_training_device_unavailable")
        return torch.device(self.device)

    def _write_blocker(self, output: Path, *, code: str, error_class: str) -> None:
        _write_json(
            output / "cuda_training_blocker.json",
            {
                "schema": CUDA_BLOCKER_SCHEMA,
                "ok": False,
                "blocker": code,
                "error_class": error_class,
                "device": self.device,
                "credentials_public": False,
                "raw_dataset_public": False,
                "tensor_values_public": False,
                "public_artifact_safe": True,
            },
        )

    def run(self, training_spec: dict[str, Any], *, output_dir: str | Path) -> dict[str, Any]:
        if training_spec.get("schema") != TRAINING_SPEC_SCHEMA:
            raise ValueError("unsupported HF LoRA training spec")
        requested_device = str(training_spec.get("device") or self.device)
        if requested_device != self.device:
            raise ValueError("training spec CUDA placement does not match runtime placement")
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        torch, _LoraConfig, PeftModel, _get_peft_model, _LlamaConfig, LlamaForCausalLM = _deps()
        try:
            device = self._require_device(torch)
        except CUDAUnavailableError as exc:
            self._write_blocker(output, code=exc.code, error_class=type(exc).__name__)
            raise

        seed = int(training_spec["seed"])
        configure_cuda_determinism(seed)
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        try:
            base = LlamaForCausalLM.from_pretrained(training_spec["base_model_path"], local_files_only=True)
            model = PeftModel.from_pretrained(
                base,
                training_spec["adapter_path"],
                is_trainable=True,
                local_files_only=True,
            )
            model.to(device)
            model.train()
            trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
            if not trainable or any("lora_" not in name for name, _value in trainable):
                raise RuntimeError("base model is not fully frozen for LoRA training")
            trainable_parameters = [value for _name, value in trainable]
            base_hash_before = model_state_hash(model, base_only=True)
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=float(training_spec["learning_rate"]),
                weight_decay=0.0,
            )
            scaler = _new_grad_scaler(torch)
            cursor = int(training_spec["data_cursor"])
            optimizer_step = int(training_spec.get("step_start", 0))
            checkpoint_loaded = False
            checkpoint_source = str(training_spec.get("checkpoint_path") or self.checkpoint_path or "")
            if checkpoint_source:
                loaded = _load_cuda_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    checkpoint_path=checkpoint_source,
                    device=self.device,
                )
                cursor = int(loaded["dataset_cursor"])
                optimizer_step = int(loaded["optimizer_step"])
                checkpoint_loaded = True

            rows = load_token_rows(training_spec["dataset_path"])
            indexes = [int(value) for value in training_spec["sample_indexes"]]
            if not indexes:
                raise ValueError("dataset shard has no sample indexes")
            batch_size = int(training_spec["batch_size"])
            grad_acc = int(training_spec["gradient_accumulation"])
            target_step = int(training_spec.get("step_end", training_spec["local_steps"]))
            if optimizer_step > target_step:
                raise ValueError("checkpoint optimizer step exceeds training target")
            losses: list[float] = []
            gradient_norms: list[float] = []
            clipped_gradient_norms: list[float] = []
            scaler_history: list[float] = []
            samples_seen = 0
            tokens_seen = 0
            checkpoint: dict[str, Any] = {}
            while optimizer_step < target_step:
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for _micro in range(grad_acc):
                    input_ids = _batch(rows, indexes, cursor, batch_size, torch, device=self.device)
                    cursor = (cursor + batch_size) % len(indexes)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        result = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                        loss = result.loss / float(grad_acc)
                    scaler.scale(loss).backward()
                    step_loss += float(loss.detach().float().item())
                    samples_seen += batch_size
                    tokens_seen += int(input_ids.numel())
                scaler.unscale_(optimizer)
                gradient_norms.append(_gradient_l2_norm(trainable_parameters))
                clipped = torch.nn.utils.clip_grad_norm_(trainable_parameters, self.gradient_clip_norm)
                clipped_gradient_norms.append(float(clipped.detach().float().item()))
                scaler_history.append(float(scaler.get_scale()))
                scaler.step(optimizer)
                scaler.update()
                optimizer_step += 1
                losses.append(step_loss)
                checkpoint = _save_cuda_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    output_dir=output,
                    optimizer_step=optimizer_step,
                    cursor=cursor,
                    base_hash=base_hash_before,
                    device=self.device,
                    outer_step=int(training_spec.get("outer_step", 0)),
                )

            if not losses:
                raise RuntimeError("CUDA training target contains no remaining optimizer steps")
            base_hash_after = model_state_hash(model, base_only=True)
            if base_hash_before != base_hash_after:
                raise RuntimeError("base model weights changed during CUDA LoRA-only training")
            final_adapter_dir = output / "adapter"
            model.save_pretrained(final_adapter_dir, safe_serialization=True)
            initial_path = Path(training_spec["adapter_tensor_path"])
            final_path = _adapter_tensor_path(final_adapter_dir)
            initial_tensors = load_tensors(initial_path)
            final_tensors = load_tensors(final_path)
            delta = adapter_delta(initial_tensors, final_tensors)
            delta_path = save_tensors(delta, output / "adapter_delta.safetensors")
            result_id = sha256_json(
                {
                    "task_id": training_spec["task_id"],
                    "miner_id": training_spec["miner_id"],
                    "claim_hash": training_spec["claim_hash"],
                    "delta_hash": named_tensor_hash(delta),
                }
            )
            manifest = delta_manifest(
                delta_path=delta_path,
                job_id=training_spec["job_id"],
                round_id=training_spec["round_id"],
                result_id=result_id,
                miner_id=training_spec["miner_id"],
                model_manifest_hash=training_spec["model_manifest_hash"],
                base_model_hash=training_spec["base_model_hash"],
                base_adapter_hash=training_spec["base_adapter_hash"],
                base_model_version=int(training_spec["base_model_version"]),
                adapter_version=int(training_spec["adapter_version"]),
                dataset_shard_index=int(training_spec["dataset_shard_index"]),
                dataset_shard_hash=training_spec["dataset_shard_hash"],
                loss_start=float(losses[0]),
                loss_end=float(losses[-1]),
                samples_seen=samples_seen,
                tokens_seen=tokens_seen,
            )
            elapsed = time.monotonic() - started
            capability = self.capability()
            capability.update(
                {
                    "cuda_used": True,
                    "gpu_live_verified": True,
                    "device_name_hash": sha256_json(
                        {"device_name": torch.cuda.get_device_name(self.device_index)}
                    ),
                }
            )
            private_result = {
                "schema": RESULT_SCHEMA,
                "workload_type": WORKLOAD_TYPE,
                "result_id": result_id,
                "job_id": training_spec["job_id"],
                "round_id": training_spec["round_id"],
                "task_id": training_spec["task_id"],
                "miner_id": training_spec["miner_id"],
                "claim_hash": training_spec["claim_hash"],
                "dataset_shard_index": int(training_spec["dataset_shard_index"]),
                "dataset_shard_hash": training_spec["dataset_shard_hash"],
                "adapter_delta": manifest,
                "adapter_path": str(final_adapter_dir),
                "adapter_tensor_path": str(final_path),
                "adapter_tensor_hash": named_tensor_hash(final_tensors),
                "checkpoint_path": checkpoint.get("checkpoint_path", ""),
                "checkpoint_hash": checkpoint.get("content_hash", ""),
                "checkpoint_loaded": checkpoint_loaded,
                "base_weights_frozen": True,
                "base_hash_before": base_hash_before,
                "base_hash_after": base_hash_after,
                "only_lora_trainable": True,
                "real_backward": True,
                "optimizer_steps": optimizer_step,
                "loss_history": losses,
                "loss_start": float(losses[0]),
                "loss_end": float(losses[-1]),
                "loss_reduced": bool(losses[-1] < losses[0]),
                "gradient_norms": gradient_norms,
                "clipped_gradient_norms": clipped_gradient_norms,
                "grad_scaler_history": scaler_history,
                "gradient_clip_norm": self.gradient_clip_norm,
                "samples_seen": samples_seen,
                "tokens_seen": tokens_seen,
                "data_cursor": cursor,
                "elapsed_seconds": elapsed,
                "pid": os.getpid(),
                "cuda_device": self.device,
                "cuda_device_index": self.device_index,
                "cuda_device_name": torch.cuda.get_device_name(self.device_index),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "runtime": capability,
                "raw_dataset_public": False,
                "private_paths_public": False,
            }
            private_result_path = _write_json(output / "training_result_private.json", private_result)
            public = public_training_spec(private_result)
            public.pop("cuda_device_name", None)
            public["adapter_delta"] = {
                key: value for key, value in manifest.items() if not key.endswith("_path")
            }
            public_result_path = _write_json(output / "training_result_public.json", public)
            return {
                **private_result,
                "private_result_path": str(private_result_path),
                "public_result_path": str(public_result_path),
            }
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            self._write_blocker(output, code=CUDAOutOfMemoryError.code, error_class=type(exc).__name__)
            raise CUDAOutOfMemoryError(CUDAOutOfMemoryError.code) from exc


class CUDATrainingRuntimeDryRun:
    """Configuration-only handoff. It deliberately never initializes CUDA."""

    backend = "pytorch_peft_cuda"

    def capability(self) -> dict[str, Any]:
        import importlib.util

        packages = {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "transformers", "peft", "safetensors")
        }
        return {
            "schema": RUNTIME_SCHEMA,
            "backend": self.backend,
            "configuration_ready": all(packages.values()),
            "package_import_contract": packages,
            "cuda_initialized": False,
            "cuda_used": False,
            "gpu_live_verified": False,
            "dry_run_only": True,
        }

    def run(self, training_spec: dict[str, Any], *, output_dir: str | Path) -> dict[str, Any]:
        raise RuntimeError("CUDA runtime is dry-run-only in Training Foundation RC")
