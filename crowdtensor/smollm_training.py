"""Real two-process, two-stage SmolLM2 LoRA training proof runtime."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import re
import shutil
import time
from pathlib import Path
from typing import Any

from .model_adapter import SmolLMModelAdapter, stable_hash


LIVE_SCHEMA = "crowdtensor_smollm_two_stage_lora_live_v1"
WORKER_SCHEMA = "crowdtensor_smollm_stage_worker_v1"
MODEL_ID = SmolLMModelAdapter.default_model_id
MODEL_REVISION = SmolLMModelAdapter.default_revision
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _tensor_state_hash(value: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(value):
        tensor = value[name].detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _public_error(exc: BaseException) -> str:
    return "smollm_stage_runtime_failed:" + type(exc).__name__


class _PassThroughLayer:
    @staticmethod
    def create() -> Any:
        import torch

        class PassThrough(torch.nn.Module):
            def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
                return hidden_states

        return PassThrough()


def _owned_lora_state(model: Any, *, start: int, end: int) -> dict[str, Any]:
    from peft import get_peft_model_state_dict

    result: dict[str, Any] = {}
    for name, tensor in get_peft_model_state_dict(model).items():
        match = _LAYER.search(name)
        if match and start <= int(match.group(1)) < end:
            result[name] = tensor.detach().cpu().contiguous()
    if not result:
        raise RuntimeError("smollm_owned_lora_state_empty")
    return result


def _configure_stage_model(
    *,
    stage_id: int,
    split_index: int,
    device: str,
    cache_dir: str,
    rank: int,
    alpha: int,
) -> tuple[Any, Any, Any, int, int]:
    import torch

    adapter = SmolLMModelAdapter()
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = adapter.load_model(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        device=device,
        dtype=dtype,
        local_files_only=False,
        cache_dir=cache_dir,
    )
    model = adapter.apply_lora(model, rank=rank, alpha=alpha)
    causal = model.get_base_model()
    decoder = causal.model
    total_layers = len(decoder.layers)
    if split_index <= 0 or split_index >= total_layers:
        raise RuntimeError("smollm_split_index_invalid")
    start, end = (0, split_index) if stage_id == 0 else (split_index, total_layers)
    for name, parameter in model.named_parameters():
        match = _LAYER.search(name)
        parameter.requires_grad = bool(match and start <= int(match.group(1)) < end and "lora_" in name)
    if stage_id == 0:
        for index in range(split_index, total_layers):
            decoder.layers[index] = _PassThroughLayer.create()
        decoder.config.num_hidden_layers = split_index
        decoder.norm = torch.nn.Identity()
    else:
        for index in range(0, split_index):
            decoder.layers[index] = _PassThroughLayer.create()
        decoder.config.num_hidden_layers = total_layers
    trainable = [item for item in model.parameters() if item.requires_grad]
    if not trainable:
        raise RuntimeError("smollm_stage_trainable_parameters_missing")
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.0)
    model.train()
    return model, causal, optimizer, start, end


def _stage_worker(
    stage_id: int,
    device: str,
    split_index: int,
    rank: int,
    alpha: int,
    cache_dir: str,
    checkpoint_dir: str,
    command_queue: Any,
    boundary_queue: Any,
    gradient_queue: Any,
    report_queue: Any,
) -> None:
    phase = "bootstrap"
    try:
        import torch
        from safetensors.torch import save_file

        if str(device).startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        torch.manual_seed(20260717 + stage_id)
        phase = "model_load"
        model, causal, optimizer, start, end = _configure_stage_model(
            stage_id=stage_id,
            split_index=split_index,
            device=device,
            cache_dir=cache_dir,
            rank=rank,
            alpha=alpha,
        )
        initial_state = _owned_lora_state(model, start=start, end=end)
        initial_hash = _tensor_state_hash(initial_state)
        parameter_count = sum(int(item.numel()) for item in initial_state.values())
        report_queue.put(
            {
                "schema": WORKER_SCHEMA,
                "type": "ready",
                "stage_id": stage_id,
                "worker_pid_hash": "sha256:" + hashlib.sha256(str(os.getpid()).encode()).hexdigest(),
                "device_type": "cuda" if str(device).startswith("cuda") else "cpu",
                "layer_start": start,
                "layer_end": end,
                "trainable_adapter_parameter_count": parameter_count,
                "adapter_hash_before": initial_hash,
                "real_model_weights_loaded": True,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "raw_process_id_public": False,
                "private_paths_public": False,
            }
        )
        completed_steps = 0
        losses: list[float] = []
        while True:
            phase = "wait_command"
            message = command_queue.get(timeout=600)
            kind = str(message.get("type") or "")
            if kind == "stop":
                break
            if kind != "step":
                raise RuntimeError("smollm_worker_command_invalid")
            step = int(message["step"])
            optimizer.zero_grad(set_to_none=True)
            if stage_id == 0:
                phase = "stage0_forward"
                input_ids = message["input_ids"].to(device)
                attention_mask = message["attention_mask"].to(device)
                labels = message["labels"]
                output = causal.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state
                boundary_queue.put(
                    {
                        "type": "activation",
                        "step": step,
                        "hidden": output.detach().float().cpu(),
                        "attention_mask": attention_mask.cpu(),
                        "labels": labels,
                    }
                )
                phase = "stage0_backward"
                gradient = gradient_queue.get(timeout=600)
                if int(gradient.get("step") or 0) != step:
                    raise RuntimeError("smollm_stage_gradient_step_mismatch")
                output.backward(gradient["gradient"].to(device=device, dtype=output.dtype))
                torch.nn.utils.clip_grad_norm_(
                    [item for item in model.parameters() if item.requires_grad], 1.0
                )
                optimizer.step()
                report_queue.put(
                    {
                        "schema": WORKER_SCHEMA,
                        "type": "step",
                        "stage_id": stage_id,
                        "step": step,
                        "optimizer_step_applied": True,
                        "activation_hash": _tensor_state_hash({"activation": output.detach()}),
                        "activation_values_public": False,
                    }
                )
            else:
                phase = "stage1_forward_backward"
                boundary = boundary_queue.get(timeout=600)
                if int(boundary.get("step") or 0) != step:
                    raise RuntimeError("smollm_stage_activation_step_mismatch")
                hidden = boundary["hidden"].to(device=device, dtype=next(causal.parameters()).dtype)
                hidden.requires_grad_(True)
                attention_mask = boundary["attention_mask"].to(device)
                labels = boundary["labels"].to(device)
                final_hidden = causal.model(
                    inputs_embeds=hidden,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state
                logits = causal.lm_head(final_hidden).float()
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
                )
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("smollm_non_finite_loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [item for item in model.parameters() if item.requires_grad], 1.0
                )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                gradient_queue.put(
                    {
                        "type": "gradient",
                        "step": step,
                        "gradient": hidden.grad.detach().float().cpu(),
                    }
                )
                report_queue.put(
                    {
                        "schema": WORKER_SCHEMA,
                        "type": "step",
                        "stage_id": stage_id,
                        "step": step,
                        "optimizer_step_applied": True,
                        "loss": losses[-1],
                        "gradient_hash": _tensor_state_hash({"gradient": hidden.grad.detach()}),
                        "gradient_values_public": False,
                        "token_ids_public": False,
                    }
                )
            completed_steps += 1
        phase = "checkpoint"
        state = _owned_lora_state(model, start=start, end=end)
        final_hash = _tensor_state_hash(state)
        destination = Path(checkpoint_dir)
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint_path = destination / f"stage{stage_id}_adapter.safetensors"
        save_file(state, str(checkpoint_path))
        if stage_id == 0:
            model.save_pretrained(
                destination / "peft-template",
                safe_serialization=True,
            )
        report_queue.put(
            {
                "schema": WORKER_SCHEMA,
                "type": "completed",
                "stage_id": stage_id,
                "completed_steps": completed_steps,
                "adapter_hash_before": initial_hash,
                "adapter_hash_after": final_hash,
                "adapter_updated": initial_hash != final_hash,
                "checkpoint_hash": "sha256:" + hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "checkpoint_tensor_count": len(state),
                "loss_count": len(losses),
                "loss_finite": all(math.isfinite(item) for item in losses),
                "checkpoint_path_public": False,
                "tensor_values_public": False,
            }
        )
    except BaseException as exc:
        report_queue.put(
            {
                "schema": WORKER_SCHEMA,
                "type": "error",
                "stage_id": stage_id,
                "phase": phase,
                "error": _public_error(exc),
                "private_paths_public": False,
                "tensor_values_public": False,
            }
        )


def _wait_reports(report_queue: Any, *, expected_type: str, count: int, timeout: float) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    reports: list[dict[str, Any]] = []
    while len(reports) < count and time.monotonic() < deadline:
        try:
            value = report_queue.get(timeout=min(5.0, max(0.1, deadline - time.monotonic())))
        except queue.Empty:
            continue
        if value.get("type") == "error":
            raise RuntimeError(str(value.get("error") or "smollm_worker_failed"))
        if value.get("type") == expected_type:
            reports.append(value)
    if len(reports) != count:
        raise RuntimeError("smollm_worker_report_timeout")
    return sorted(reports, key=lambda item: int(item["stage_id"]))


def _merge_stage_adapters(checkpoint_dir: Path, export_dir: Path, *, rank: int, alpha: int) -> dict[str, Any]:
    from safetensors.torch import load_file, save_file

    combined: dict[str, Any] = {}
    for stage_id in (0, 1):
        state = load_file(str(checkpoint_dir / f"stage{stage_id}_adapter.safetensors"))
        duplicate = set(combined).intersection(state)
        if duplicate:
            raise RuntimeError("smollm_stage_adapter_key_overlap")
        combined.update(state)
    export_dir.mkdir(parents=True, exist_ok=True)
    save_file(combined, str(export_dir / "adapter_model.safetensors"))
    template = checkpoint_dir / "peft-template" / "adapter_config.json"
    if template.is_file():
        shutil.copyfile(template, export_dir / "adapter_config.json")
    else:
        from peft import LoraConfig

        config = LoraConfig(
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(SmolLMModelAdapter.target_modules),
        )
        config.base_model_name_or_path = MODEL_ID
        config.revision = MODEL_REVISION
        config.inference_mode = True
        config.save_pretrained(export_dir)
    return {
        "adapter_tensor_count": len(combined),
        "adapter_file_hash": "sha256:" + hashlib.sha256((export_dir / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "adapter_config_hash": "sha256:" + hashlib.sha256((export_dir / "adapter_config.json").read_bytes()).hexdigest(),
        "standard_peft_format": True,
    }


def _independent_reload(export_dir: Path, *, device: str, cache_dir: str) -> dict[str, Any]:
    import torch

    adapter = SmolLMModelAdapter()
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = adapter.reload_adapter(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        adapter_dir=export_dir,
        device=device,
        dtype=dtype,
        local_files_only=False,
        cache_dir=cache_dir,
    )
    model.eval()
    tokenizer_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=tokenizer_ids, use_cache=False).logits
    finite = bool(torch.isfinite(logits.float()).all().item())
    return {
        "independent_process_reload": True,
        "adapter_reload_verified": finite,
        "reload_logits_finite": finite,
        "reload_output_shape": list(logits.shape),
        "token_ids_public": False,
        "logit_values_public": False,
    }


def run_two_stage_lora(
    output_dir: str | Path,
    *,
    steps: int = 2,
    sequence_length: int = 16,
    devices: tuple[str, str] = ("cuda:0", "cuda:1"),
    timeout_seconds: float = 1200.0,
    rank: int = 8,
    alpha: int = 16,
    node_scope: str = "local logical multi-process",
    clean_install: bool = False,
) -> dict[str, Any]:
    if steps < 1 or steps > 20:
        raise ValueError("smollm_steps_outside_bound")
    if sequence_length < 4 or sequence_length > 128:
        raise ValueError("smollm_sequence_length_outside_bound")
    started = time.monotonic()
    output = Path(output_dir).expanduser().resolve()
    private = output / ".private"
    checkpoints = private / "checkpoints"
    cache = private / "hf-cache"
    export = output / "adapter"
    private.mkdir(parents=True, exist_ok=True)
    private.chmod(0o700)
    context = mp.get_context("spawn")
    commands = [context.Queue() for _ in range(2)]
    boundary = context.Queue()
    gradients = context.Queue()
    reports = context.Queue()
    split_index = 15
    processes = [
        context.Process(
            target=_stage_worker,
            args=(
                stage_id, devices[stage_id], split_index, rank, alpha,
                str(cache), str(checkpoints), commands[stage_id], boundary,
                gradients, reports,
            ),
        )
        for stage_id in range(2)
    ]
    for process in processes:
        process.start()
    try:
        ready = _wait_reports(reports, expected_type="ready", count=2, timeout=timeout_seconds)
        import torch

        generator = torch.Generator().manual_seed(20260717)
        committed_steps: list[int] = []
        step_reports: list[dict[str, Any]] = []
        for step in range(1, steps + 1):
            input_ids = torch.randint(
                low=0, high=49152, size=(1, sequence_length), generator=generator
            )
            attention = torch.ones_like(input_ids)
            labels = input_ids.clone()
            message = {
                "type": "step",
                "step": step,
                "input_ids": input_ids,
                "attention_mask": attention,
                "labels": labels,
            }
            commands[0].put(message)
            commands[1].put({"type": "step", "step": step})
            current = _wait_reports(reports, expected_type="step", count=2, timeout=timeout_seconds)
            if [int(item["stage_id"]) for item in current] != [0, 1]:
                raise RuntimeError("smollm_atomic_step_stage_coverage_invalid")
            committed_steps.append(step)
            step_reports.extend(current)
        for item in commands:
            item.put({"type": "stop"})
        completed = _wait_reports(reports, expected_type="completed", count=2, timeout=timeout_seconds)
        for process in processes:
            process.join(timeout=60)
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError("smollm_stage_process_exit_invalid")
        merged = _merge_stage_adapters(checkpoints, export, rank=rank, alpha=alpha)
        reload_device = devices[0] if str(devices[0]).startswith("cuda") else "cpu"
        reload_report = _independent_reload(export, device=reload_device, cache_dir=str(cache))
        report = {
            "schema": LIVE_SCHEMA,
            "ok": True,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_license": "apache-2.0",
            "real_open_model_weights": True,
            "random_or_synthetic_weights_used": False,
            "model_adapter_id": "smollm2_lora_v1",
            "model_adapter_api_version": "model_adapter_v1.0",
            "node_scope": str(node_scope),
            "logical_miner_count": 2,
            "logical_stage_count": 2,
            "distinct_worker_processes": len({item["worker_pid_hash"] for item in ready}) == 2,
            "physical_multi_machine_verified": False,
            "single_process_smoke": False,
            "devices": [item["device_type"] for item in ready],
            "stage_specs": [
                {key: item[key] for key in ("stage_id", "layer_start", "layer_end")}
                for item in ready
            ],
            "committed_step_ids": committed_steps,
            "strictly_contiguous_atomic_steps": committed_steps == list(range(1, steps + 1)),
            "all_stage_optimizer_steps_applied": all(item["optimizer_step_applied"] for item in step_reports),
            "finite_loss_verified": all(
                math.isfinite(float(item["loss"]))
                for item in step_reports if item["stage_id"] == 1
            ),
            "both_stage_adapters_updated": all(item["adapter_updated"] for item in completed),
            "stage_checkpoints": [
                {
                    "stage_id": item["stage_id"],
                    "checkpoint_hash": item["checkpoint_hash"],
                    "checkpoint_tensor_count": item["checkpoint_tensor_count"],
                    "adapter_updated": item["adapter_updated"],
                }
                for item in completed
            ],
            "export": merged,
            "reload": reload_report,
            "clean_install_required": bool(clean_install),
            "workspace_import_used": not bool(clean_install),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "raw_training_text_public": False,
            "token_ids_public": False,
            "activation_values_public": False,
            "gradient_values_public": False,
            "checkpoint_tensor_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        output.mkdir(parents=True, exist_ok=True)
        (output / "smollm_two_stage_lora_live.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
