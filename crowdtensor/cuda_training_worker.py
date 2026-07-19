"""Remote CUDA stage and LoRA Miner workload used by private Kaggle kernels."""

from __future__ import annotations

import base64
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .hf_lora_training import CUDALoRATrainingRuntime, load_token_rows
from .pipeline_lora_training import CUDAStageRuntime, deterministic_pipeline_rows
from .training_contract import sha256_file, sha256_json


WORKER_SCHEMA = "crowdtensor_cuda_two_node_worker_v1"
TRANSIENT_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _request_json(
    method: str,
    base_url: str,
    path: str,
    *,
    token: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
    transient_attempts: int = 1,
    retry_interval: float = 1.0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"x-crowdtensor-miner-token": token}
    if body is not None:
        headers["content-type"] = "application/json"
    attempts = max(1, int(transient_attempts))
    raw = ""
    for attempt in range(1, attempts + 1):
        request = Request(
            f"{base_url.rstrip('/')}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            break
        except HTTPError as exc:
            if attempt >= attempts or int(exc.code) not in TRANSIENT_HTTP_STATUS:
                raise
        except (URLError, TimeoutError, OSError):
            if attempt >= attempts:
                raise
        time.sleep(max(0.1, float(retry_interval)) * attempt)
    value = json.loads(raw) if raw else {}
    if not isinstance(value, dict):
        raise RuntimeError("CUDA training Coordinator returned a non-object response")
    return value


def _wait_json(
    base_url: str,
    path: str,
    *,
    token: str,
    timeout: float,
    interval: float = 1.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout)
    last_status = 0
    while time.monotonic() < deadline:
        try:
            return _request_json("GET", base_url, path, token=token, timeout=30.0)
        except HTTPError as exc:
            last_status = int(exc.code)
            if exc.code not in {404, 409, *TRANSIENT_HTTP_STATUS}:
                raise
        except (URLError, TimeoutError, OSError):
            last_status = 0
        time.sleep(max(0.1, float(interval)))
    raise TimeoutError(f"CUDA training Coordinator payload wait timed out with HTTP {last_status}")


def _tensor_payload(value: Any) -> tuple[str, str, list[int], str]:
    import torch
    from safetensors.torch import save

    tensor = torch.from_numpy(value).detach().cpu().contiguous()
    raw = save({"tensor": tensor})
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    return base64.b64encode(raw).decode("ascii"), digest, list(tensor.shape), str(tensor.dtype).replace("torch.", "")


def _payload_tensor(encoded: str) -> Any:
    from safetensors.torch import load

    raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    return load(raw)["tensor"].cpu().numpy()


def _pipeline_config(seed: int) -> dict[str, Any]:
    return {
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
    }


def run_cross_node_stage(
    *,
    role: str,
    coordinator_url: str,
    token: str,
    run_id: str,
    output_dir: str | Path,
    total_steps: int = 4,
    seed: int = 20260710,
    wait_timeout: float = 600.0,
) -> dict[str, Any]:
    import torch

    if role not in {"stage0", "stage1"}:
        raise ValueError("CUDA two-node worker role must be stage0 or stage1")
    stage_id = 0 if role == "stage0" else 1
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = _pipeline_config(seed)
    runtime = CUDAStageRuntime(
        stage_id,
        config,
        output / "checkpoint",
        device="cuda:0",
    )
    ready = runtime.ready()
    worker_id_hash = sha256_json({"run_id": run_id, "role": role, "pid": os.getpid()})
    _request_json(
        "POST",
        coordinator_url,
        "/cuda-training/register",
        token=token,
        payload={
            "run_id": run_id,
            "role": role,
            "worker_id_hash": worker_id_hash,
            "pid": os.getpid(),
            "cuda_device_index": 0,
            "cuda_device_name_hash": ready["cuda_device_name_hash"],
            "cuda_live": True,
        },
        transient_attempts=6,
        retry_interval=2.0,
    )
    rows = deterministic_pipeline_rows(sequence_length=config["sequence_length"])
    cursor = 0
    records: list[dict[str, Any]] = []
    losses: list[float] = []
    final_checkpoint: dict[str, Any] = {}
    for step in range(int(total_steps)):
        batch = [rows[(cursor + offset) % len(rows)] for offset in range(config["batch_size"])]
        next_cursor = (cursor + config["batch_size"]) % len(rows)
        if stage_id == 0:
            forward = runtime.forward(batch, step=step)
            encoded, digest, shape, dtype = _tensor_payload(forward["activation"])
            _request_json(
                "POST",
                coordinator_url,
                "/cuda-training/payload",
                token=token,
                payload={
                    "run_id": run_id,
                    "kind": "activation",
                    "step": step,
                    "producer_role": role,
                    "payload_b64": encoded,
                    "payload_hash": digest,
                    "shape": shape,
                    "dtype": dtype,
                },
                transient_attempts=6,
                retry_interval=2.0,
            )
            gradient = _wait_json(
                coordinator_url,
                f"/cuda-training/payload/gradient/{step}?{urlencode({'run_id': run_id})}",
                token=token,
                timeout=wait_timeout,
            )
            backward = runtime.backward(
                _payload_tensor(str(gradient["payload_b64"])),
                step=step,
                cursor=next_cursor,
                gradient_scale=float(gradient.get("gradient_scale") or 65536.0),
            )
            final_checkpoint = backward["checkpoint"]
            records.append(
                {
                    "step": step,
                    "forward_hash": digest,
                    "backward_gradient_hash": gradient["payload_hash"],
                    "lora_gradient_norm": backward["lora_gradient_norm"],
                    "optimizer_step": backward["optimizer_step"],
                    "checkpoint_hash": backward["checkpoint_hash"],
                    "peak_allocated_bytes": backward["peak_allocated_bytes"],
                    "peak_reserved_bytes": backward["peak_reserved_bytes"],
                }
            )
        else:
            activation = _wait_json(
                coordinator_url,
                f"/cuda-training/payload/activation/{step}?{urlencode({'run_id': run_id})}",
                token=token,
                timeout=wait_timeout,
            )
            backward = runtime.forward_backward(
                _payload_tensor(str(activation["payload_b64"])),
                batch,
                step=step,
                cursor=next_cursor,
            )
            encoded, digest, shape, dtype = _tensor_payload(backward["activation_gradient"])
            _request_json(
                "POST",
                coordinator_url,
                "/cuda-training/payload",
                token=token,
                payload={
                    "run_id": run_id,
                    "kind": "gradient",
                    "step": step,
                    "producer_role": role,
                    "payload_b64": encoded,
                    "payload_hash": digest,
                    "shape": shape,
                    "dtype": dtype,
                    "gradient_scale": backward["gradient_scale"],
                },
                transient_attempts=6,
                retry_interval=2.0,
            )
            final_checkpoint = backward["checkpoint"]
            losses.append(float(backward["loss"]))
            records.append(
                {
                    "step": step,
                    "forward_hash": backward["logits_hash"],
                    "activation_hash": activation["payload_hash"],
                    "backward_gradient_hash": digest,
                    "lora_gradient_norm": backward["lora_gradient_norm"],
                    "optimizer_step": backward["optimizer_step"],
                    "checkpoint_hash": backward["checkpoint_hash"],
                    "loss": backward["loss"],
                    "peak_allocated_bytes": backward["peak_allocated_bytes"],
                    "peak_reserved_bytes": backward["peak_reserved_bytes"],
                }
            )
        cursor = next_cursor
    status = runtime.status()
    summary = {
        "schema": "crowdtensor_cuda_cross_node_stage_v1",
        "role": role,
        "stage_id": stage_id,
        "pid": os.getpid(),
        "cuda_device": "cuda:0",
        "cuda_device_name_hash": ready["cuda_device_name_hash"],
        "steps_completed": int(total_steps),
        "records": records,
        "real_cuda_forward": True,
        "real_cuda_backward": True,
        "real_activation_transport": True,
        "real_backward_gradient_transport": True,
        "positive_lora_gradient_norms": all(item["lora_gradient_norm"] > 0 for item in records),
        "base_weights_frozen": ready["base_hash"] == status["base_hash_after"],
        "no_full_model_loaded": not ready["ownership"]["loaded_full_model"],
        "owned_layer_indexes": ready["ownership"]["owned_layer_indexes"],
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "loss_reduced": losses[-1] < losses[0] if losses else None,
        "final_adapter_hash": status["adapter_tensor_hash"],
        "checkpoint_hash": final_checkpoint.get("content_hash", ""),
        "checkpoint_grad_scaler_state_present": final_checkpoint.get("grad_scaler_state_present") is True,
        "peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in records),
        "peak_reserved_bytes": max(item["peak_reserved_bytes"] for item in records),
        "activation_values_public": False,
        "gradient_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    _request_json(
        "POST",
        coordinator_url,
        "/cuda-training/complete",
        token=token,
        payload={"run_id": run_id, "role": role, "summary": summary},
        transient_attempts=6,
        retry_interval=2.0,
    )
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _localized_training_spec(spec: dict[str, Any], fixture_dir: Path) -> dict[str, Any]:
    localized = dict(spec)
    localized.update(
        {
            "base_model_path": str(fixture_dir / "base_model"),
            "adapter_path": str(fixture_dir / "initial_adapter"),
            "adapter_tensor_path": str(fixture_dir / "initial_adapter" / "adapter_model.safetensors"),
            "adapter_config_path": str(fixture_dir / "initial_adapter" / "adapter_config.json"),
            "dataset_path": str(fixture_dir / "private_dataset.jsonl"),
            "device": "cuda:0",
        }
    )
    for key in ("base_model_path", "adapter_path", "adapter_tensor_path", "adapter_config_path", "dataset_path"):
        if not Path(localized[key]).exists():
            raise RuntimeError(f"remote CUDA training fixture missing {key}")
    return localized


def run_remote_lora_miner(
    *,
    role: str,
    coordinator_url: str,
    token: str,
    fixture_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    runtime = CUDALoRATrainingRuntime("cuda:0", gradient_clip_norm=1.0)
    capability = runtime.capability()
    claim = _request_json(
        "POST",
        coordinator_url,
        "/tasks/claim",
        token=token,
        payload={
            "miner_id": f"kaggle-cuda-{role}",
            "capabilities": {
                "runtime": "python-cli",
                "backend": "cuda",
                "protocol_version": "runtime_contract_v1",
                "supports_training_spec": True,
                "supported_workloads": ["hf_lora_train"],
                "hf_lora_runtime": capability,
            },
        },
    )
    spec = _localized_training_spec(dict(claim["workload_spec"]), Path(fixture_dir))
    result = runtime.run(spec, output_dir=output_dir)
    delta_path = Path(result["adapter_delta"]["delta_path"])
    response = _request_json(
        "POST",
        coordinator_url,
        f"/tasks/{claim['task_id']}/result",
        token=token,
        payload={
            "lease_token": claim["lease_token"],
            "attempt": claim["attempt"],
            "idempotency_key": sha256_json({"result_id": result["result_id"], "role": role}),
            "training_result": result,
            "training_adapter_delta_b64": base64.b64encode(delta_path.read_bytes()).decode("ascii"),
            "metrics": {
                "optimizer_steps": result["optimizer_steps"],
                "tokens_seen": result["tokens_seen"],
                "loss_start": result["loss_start"],
                "loss_end": result["loss_end"],
                "loss_reduced": result["loss_reduced"],
                "cuda_live": True,
            },
        },
        timeout=300.0,
        transient_attempts=6,
        retry_interval=2.0,
    )
    return {
        "schema": "crowdtensor_cuda_remote_lora_miner_v1",
        "role": role,
        "base_model_version": int(spec["base_model_version"]),
        "adapter_version": int(spec["adapter_version"]),
        "model_manifest_hash": spec["model_manifest_hash"],
        "base_model_hash": spec["base_model_hash"],
        "base_adapter_hash": spec["base_adapter_hash"],
        "dataset_shard_index": int(result["dataset_shard_index"]),
        "dataset_shard_hash": result["dataset_shard_hash"],
        "result_id": result["result_id"],
        "adapter_delta_file_hash": result["adapter_delta"]["delta_file_hash"],
        "adapter_delta_tensor_specs_hash": result["adapter_delta"]["tensor_specs_hash"],
        "adapter_delta_tensor_count": result["adapter_delta"]["tensor_count"],
        "adapter_delta_norm": result["adapter_delta"]["delta_norm"],
        "adapter_delta_format": "named_safetensors",
        "adapter_delta_named_tensors": True,
        "base_weights_frozen": result["base_weights_frozen"],
        "only_lora_trainable": result["only_lora_trainable"],
        "real_backward": result["real_backward"],
        "optimizer_steps": result["optimizer_steps"],
        "tokens_seen": result["tokens_seen"],
        "elapsed_seconds": result["elapsed_seconds"],
        "loss_start": result["loss_start"],
        "loss_end": result["loss_end"],
        "loss_reduced": result["loss_reduced"],
        "checkpoint_hash": result["checkpoint_hash"],
        "checkpoint_loaded": result["checkpoint_loaded"],
        "peak_allocated_bytes": result["peak_allocated_bytes"],
        "peak_reserved_bytes": result["peak_reserved_bytes"],
        "runtime": result["runtime"],
        "coordinator_accepted": response.get("accepted") is True,
        "coordinator_training_updated": response.get("training_updated") is True,
        "coordinator_adapter_version": int(response.get("adapter_version", 0)),
        "coordinator_outer_step": int(response.get("outer_step", 0)),
        "private_paths_public": False,
        "tensor_values_public": False,
        "public_artifact_safe": True,
    }


def wait_global_adapter(
    *,
    coordinator_url: str,
    token: str,
    run_id: str,
    timeout: float = 600.0,
) -> dict[str, Any]:
    return _wait_json(
        coordinator_url,
        f"/cuda-training/global-adapter?{urlencode({'run_id': run_id})}",
        token=token,
        timeout=timeout,
    )


def evaluate_global_adapter_on_cuda(
    *,
    role: str,
    coordinator_url: str,
    token: str,
    run_id: str,
    fixture_dir: str | Path,
    private_output_dir: str | Path,
    global_adapter: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from safetensors.torch import save
    from transformers import LlamaForCausalLM

    fixture = Path(fixture_dir)
    output = Path(private_output_dir)
    adapter_dir = output / "global_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_bytes = base64.b64decode(str(global_adapter["adapter_b64"]).encode("ascii"), validate=True)
    config_bytes = base64.b64decode(str(global_adapter["adapter_config_b64"]).encode("ascii"), validate=True)
    adapter_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    adapter_path.write_bytes(adapter_bytes)
    config_path.write_bytes(config_bytes)
    if sha256_file(adapter_path) != global_adapter["adapter_hash"]:
        raise RuntimeError("global adapter private download hash mismatch")
    rows = load_token_rows(fixture / "private_dataset.jsonl")
    input_ids = torch.tensor([rows[index]["input_ids"] for index in range(4)], dtype=torch.long, device="cuda:0")

    def evaluate(adapter: bool) -> tuple[float, Any]:
        base = LlamaForCausalLM.from_pretrained(fixture / "base_model", local_files_only=True).to("cuda:0")
        model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True) if adapter else base
        model.eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            loss = float(result.loss.detach().float().item())
            logits = result.logits[0, -1].detach().cpu().contiguous()
        del model, base, result
        gc.collect()
        torch.cuda.empty_cache()
        return loss, logits

    before_loss, before_logits = evaluate(False)
    after_loss, after_logits = evaluate(True)
    changed = not bool(torch.allclose(before_logits, after_logits, atol=1e-7, rtol=1e-6))
    raw_logits = save({"logits": after_logits})
    logits_hash = "sha256:" + hashlib.sha256(raw_logits).hexdigest()
    _request_json(
        "POST",
        coordinator_url,
        "/cuda-training/evaluation",
        token=token,
        payload={
            "run_id": run_id,
            "role": role,
            "logits_b64": base64.b64encode(raw_logits).decode("ascii"),
            "logits_hash": logits_hash,
            "shape": list(after_logits.shape),
            "dtype": str(after_logits.dtype).replace("torch.", ""),
            "before_loss": before_loss,
            "after_loss": after_loss,
            "adapter_changes_logits": changed,
            "standard_peft_cuda_load": True,
        },
        transient_attempts=6,
        retry_interval=2.0,
    )
    return {
        "schema": "crowdtensor_cuda_adapter_evaluation_v1",
        "role": role,
        "before_loss": before_loss,
        "after_loss": after_loss,
        "validation_loss_reduced": after_loss < before_loss,
        "adapter_changes_logits": changed,
        "after_logits_hash": logits_hash,
        "standard_peft_cuda_load": True,
        "cuda_device": "cuda:0",
        "logits_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
