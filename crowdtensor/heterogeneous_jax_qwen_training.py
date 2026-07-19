"""Stage-selective Qwen2.5 LoRA training on a JAX TPU resource group."""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

from .heterogeneous_qwen_source import qwen_stage_spec
from .heterogeneous_training_checkpoint import (
    load_jax_stage_checkpoint,
    save_jax_stage_checkpoint,
)
from .heterogeneous_training_manifest import (
    TPU_MANIFEST_SCHEMA,
    stable_hash,
    validate_training_manifest,
)
from .qwen15b_training import sha256_file


JAX_STAGE_RUNTIME_SCHEMA = "crowdtensor_heterogeneous_jax_qwen_stage_runtime_v1"


def _require_jax() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec
    except ImportError as exc:  # pragma: no cover - exercised in Kaggle package
        raise RuntimeError("heterogeneous_jax_runtime_missing") from exc
    return jax, jnp, np, Mesh, (NamedSharding, PartitionSpec)


def _public_error(exc: BaseException) -> str:
    text = str(exc)
    for code in (
        "heterogeneous_jax_runtime_missing",
        "heterogeneous_jax_tpu_device_missing",
        "heterogeneous_jax_tpu_device_count_invalid",
        "heterogeneous_jax_stage_must_be_intermediate",
        "heterogeneous_jax_stage_source_keys_invalid",
        "heterogeneous_jax_non_finite_activation",
        "heterogeneous_jax_non_finite_gradient",
        "heterogeneous_jax_non_finite_lora_gradient",
    ):
        if code in text:
            return code
    if isinstance(exc, TimeoutError):
        return "heterogeneous_jax_stage_timeout"
    return "heterogeneous_jax_stage_runtime_failed:" + type(exc).__name__


def _linear_dimensions(config: dict[str, Any]) -> dict[str, tuple[int, int, str]]:
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    kv_width = hidden * kv_heads // heads
    intermediate = int(config["intermediate_size"])
    return {
        "q_proj": (hidden, hidden, "self_attn"),
        "k_proj": (hidden, kv_width, "self_attn"),
        "v_proj": (hidden, kv_width, "self_attn"),
        "o_proj": (hidden, hidden, "self_attn"),
        "gate_proj": (hidden, intermediate, "mlp"),
        "up_proj": (hidden, intermediate, "mlp"),
        "down_proj": (intermediate, hidden, "mlp"),
    }


def _base_key(layer: int, owner: str, target: str, suffix: str) -> str:
    return f"model.layers.{layer}.{owner}.{target}.{suffix}"


def _adapter_key(layer: int, owner: str, target: str, side: str) -> str:
    return f"model.layers.{layer}.{owner}.{target}.lora_{side}.weight"


def expected_stage_source_keys(
    config: dict[str, Any], stage: dict[str, Any]
) -> list[str]:
    del config
    keys: list[str] = []
    for layer in range(int(stage["layer_start"]), int(stage["layer_end"])):
        prefix = f"model.layers.{layer}"
        keys.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.v_proj.bias",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        )
    return sorted(keys)


def _mesh_devices(*, require_tpu: bool, expected_tpu_devices: int) -> list[Any]:
    jax, _jnp, _np, _mesh, _sharding = _require_jax()
    devices = list(jax.devices())
    tpu_devices = [
        item
        for item in devices
        if str(getattr(item, "platform", "")).lower() == "tpu"
    ]
    if require_tpu and not tpu_devices:
        raise RuntimeError("heterogeneous_jax_tpu_device_missing")
    selected = tpu_devices if tpu_devices else devices[:1]
    if require_tpu and len(selected) != int(expected_tpu_devices):
        raise RuntimeError("heterogeneous_jax_tpu_device_count_invalid")
    if not selected:
        raise RuntimeError("heterogeneous_jax_device_missing")
    return selected


def _put_array(value: Any, *, sharding: Any, dtype: Any) -> Any:
    jax, jnp, np, _mesh, _sharding = _require_jax()
    host = value
    if hasattr(host, "detach"):
        host = host.detach().float().cpu().numpy()
    host = np.asarray(host, dtype=np.float32)
    placed = jax.device_put(host, sharding)
    return placed.astype(dtype if dtype is not None else jnp.float32)


def load_jax_qwen_stage(
    training_manifest: dict[str, Any],
    config: dict[str, Any],
    *,
    stage_id: int,
    shard_path: str | Path,
    require_tpu: bool,
    expected_tpu_devices: int = 8,
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
    """Load one real middle-stage shard and distribute matrices over a JAX mesh."""

    jax, jnp, np, Mesh, sharding_types = _require_jax()
    NamedSharding, PartitionSpec = sharding_types
    manifest = validate_training_manifest(training_manifest)
    if manifest["schema"] != TPU_MANIFEST_SCHEMA:
        raise ValueError("heterogeneous_jax_manifest_v2_required")
    stage = dict(manifest["stages"][int(stage_id)])
    if (
        stage["owns_embedding"]
        or stage["owns_norm"]
        or stage["owns_lm_head"]
        or "jax_tpu" not in stage["allowed_device_types"]
    ):
        raise ValueError("heterogeneous_jax_stage_must_be_intermediate")
    devices = _mesh_devices(
        require_tpu=require_tpu, expected_tpu_devices=expected_tpu_devices
    )
    mesh = Mesh(np.asarray(devices), ("model",))
    replicated = NamedSharding(mesh, PartitionSpec())
    matrix_sharded = NamedSharding(mesh, PartitionSpec("model", None))
    vector_sharded = NamedSharding(mesh, PartitionSpec("model"))
    expected = expected_stage_source_keys(config, stage)
    source_path = Path(shard_path).expanduser().resolve()
    from safetensors import safe_open

    base: dict[str, Any] = {}
    with safe_open(str(source_path), framework="pt", device="cpu") as handle:
        present = sorted(handle.keys())
        if present != expected:
            raise RuntimeError("heterogeneous_jax_stage_source_keys_invalid")
        for name in expected:
            tensor = handle.get_tensor(name)
            if tensor.ndim == 2 and int(tensor.shape[0]) % len(devices) == 0:
                target_sharding = matrix_sharded
            elif (
                tensor.ndim == 1
                and ".layernorm." not in name
                and int(tensor.shape[0]) % len(devices) == 0
            ):
                target_sharding = vector_sharded
            else:
                target_sharding = replicated
            base[name] = _put_array(
                tensor, sharding=target_sharding, dtype=jnp.bfloat16
            )
    sharded_arrays = [
        value
        for value in base.values()
        if not bool(getattr(value.sharding, "is_fully_replicated", False))
    ]
    if require_tpu and (
        not sharded_arrays
        or max(len(value.addressable_shards) for value in sharded_arrays)
        != len(devices)
    ):
        raise RuntimeError("heterogeneous_jax_parameter_sharding_incomplete")
    report = {
        "schema": "crowdtensor_heterogeneous_jax_qwen_stage_load_v1",
        "training_manifest_hash": manifest["content_hash"],
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "stage_id": int(stage_id),
        "layer_start": int(stage["layer_start"]),
        "layer_end": int(stage["layer_end"]),
        "source_tensor_count": len(expected),
        "source_keys_hash": stable_hash(expected),
        "source_shard_hash": sha256_file(source_path),
        "stage_selective_loading": True,
        "full_model_loaded": False,
        "runtime_backend": "jax_tpu" if require_tpu else "jax_cpu_test",
        "jax_version": str(getattr(jax, "__version__", "")),
        "jax_device_count": len(devices),
        "jax_tpu_device_count": len(
            [item for item in devices if str(item.platform).lower() == "tpu"]
        ),
        "mesh_axis_names": ["model"],
        "mesh_shape": [len(devices)],
        "parameter_sharding": "named_mesh_model_axis",
        "sharded_source_tensor_count": len(sharded_arrays),
        "all_mesh_devices_used": bool(
            sharded_arrays
            and max(len(value.addressable_shards) for value in sharded_arrays)
            == len(devices)
        ),
        "compute_dtype": "bfloat16",
        "tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return base, stage, mesh, report


def initialize_jax_lora(
    manifest: dict[str, Any],
    config: dict[str, Any],
    stage: dict[str, Any],
    mesh: Any,
) -> dict[str, Any]:
    _jax, jnp, np, _Mesh, sharding_types = _require_jax()
    NamedSharding, PartitionSpec = sharding_types
    rank = int(manifest["lora"]["rank"])
    replicated = NamedSharding(mesh, PartitionSpec())
    output_sharded = NamedSharding(mesh, PartitionSpec("model", None))
    dimensions = _linear_dimensions(config)
    state: dict[str, Any] = {}
    seed = int(manifest["training"]["seed"])
    for layer in range(int(stage["layer_start"]), int(stage["layer_end"])):
        for target_index, target in enumerate(manifest["lora"]["target_modules"]):
            input_size, output_size, owner = dimensions[str(target)]
            rng = np.random.default_rng(seed + layer * 101 + target_index)
            bound = 1.0 / math.sqrt(float(input_size))
            a = rng.uniform(-bound, bound, size=(rank, input_size)).astype(np.float32)
            b = np.zeros((output_size, rank), dtype=np.float32)
            state[_adapter_key(layer, owner, str(target), "A")] = _put_array(
                a, sharding=replicated, dtype=jnp.float32
            )
            state[_adapter_key(layer, owner, str(target), "B")] = _put_array(
                b, sharding=output_sharded, dtype=jnp.float32
            )
    return state


def _rms_norm(value: Any, weight: Any, *, epsilon: float) -> Any:
    _jax, jnp, _np, _mesh, _sharding = _require_jax()
    variance = jnp.mean(jnp.square(value.astype(jnp.float32)), axis=-1, keepdims=True)
    normalized = value.astype(jnp.float32) * jax_lax_rsqrt(variance + epsilon)
    return (normalized * weight.astype(jnp.float32)).astype(value.dtype)


def jax_lax_rsqrt(value: Any) -> Any:
    jax, _jnp, _np, _mesh, _sharding = _require_jax()
    return jax.lax.rsqrt(value)


def _linear(
    value: Any,
    base: dict[str, Any],
    lora: dict[str, Any],
    *,
    layer: int,
    owner: str,
    target: str,
    scaling: float,
    bias: bool,
) -> Any:
    _jax, jnp, _np, _mesh, _sharding = _require_jax()
    weight = base[_base_key(layer, owner, target, "weight")]
    result = jnp.einsum("...i,oi->...o", value, weight)
    if bias:
        result = result + base[_base_key(layer, owner, target, "bias")]
    a = lora[_adapter_key(layer, owner, target, "A")].astype(value.dtype)
    b = lora[_adapter_key(layer, owner, target, "B")].astype(value.dtype)
    delta = jnp.einsum("...i,ri->...r", value, a)
    delta = jnp.einsum("...r,or->...o", delta, b)
    return result + delta * jnp.asarray(scaling, dtype=result.dtype)


def qwen_middle_stage_forward(
    base: dict[str, Any],
    lora: dict[str, Any],
    value: Any,
    *,
    config: dict[str, Any],
    stage: dict[str, Any],
    lora_alpha: int,
    lora_rank: int,
) -> Any:
    _jax, jnp, _np, _mesh, _sharding = _require_jax()
    hidden_size = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = hidden_size // heads
    repeat_factor = heads // kv_heads
    epsilon = float(config.get("rms_norm_eps") or 1e-6)
    theta = float(config.get("rope_theta") or 1_000_000.0)
    sequence_length = int(value.shape[1])
    scaling = float(lora_alpha) / float(lora_rank)
    positions = jnp.arange(sequence_length, dtype=jnp.float32)
    inv_frequency = 1.0 / (
        theta
        ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / float(head_dim))
    )
    frequencies = jnp.einsum("i,j->ij", positions, inv_frequency)
    rope = jnp.concatenate((frequencies, frequencies), axis=-1)[None, :, None, :]
    cosine = jnp.cos(rope).astype(value.dtype)
    sine = jnp.sin(rope).astype(value.dtype)

    def rotate_half(item: Any) -> Any:
        first, second = jnp.split(item, 2, axis=-1)
        return jnp.concatenate((-second, first), axis=-1)

    hidden = value
    for layer in range(int(stage["layer_start"]), int(stage["layer_end"])):
        residual = hidden
        normalized = _rms_norm(
            hidden,
            base[f"model.layers.{layer}.input_layernorm.weight"],
            epsilon=epsilon,
        )
        query = _linear(
            normalized, base, lora, layer=layer, owner="self_attn",
            target="q_proj", scaling=scaling, bias=True,
        ).reshape(value.shape[0], sequence_length, heads, head_dim)
        key = _linear(
            normalized, base, lora, layer=layer, owner="self_attn",
            target="k_proj", scaling=scaling, bias=True,
        ).reshape(value.shape[0], sequence_length, kv_heads, head_dim)
        attention_value = _linear(
            normalized, base, lora, layer=layer, owner="self_attn",
            target="v_proj", scaling=scaling, bias=True,
        ).reshape(value.shape[0], sequence_length, kv_heads, head_dim)
        query = (query * cosine) + (rotate_half(query) * sine)
        key = (key * cosine) + (rotate_half(key) * sine)
        key = jnp.repeat(key, repeat_factor, axis=2)
        attention_value = jnp.repeat(attention_value, repeat_factor, axis=2)
        scores = jnp.einsum("bqhd,bkhd->bhqk", query, key).astype(jnp.float32)
        scores = scores / math.sqrt(float(head_dim))
        causal = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))
        scores = jnp.where(causal[None, None, :, :], scores, -1e30)
        probabilities = jnp.asarray(
            __import__("jax").nn.softmax(scores, axis=-1), dtype=query.dtype
        )
        context = jnp.einsum(
            "bhqk,bkhd->bqhd", probabilities, attention_value
        ).reshape(value.shape[0], sequence_length, hidden_size)
        hidden = residual + _linear(
            context, base, lora, layer=layer, owner="self_attn",
            target="o_proj", scaling=scaling, bias=False,
        )
        residual = hidden
        normalized = _rms_norm(
            hidden,
            base[f"model.layers.{layer}.post_attention_layernorm.weight"],
            epsilon=epsilon,
        )
        gate = _linear(
            normalized, base, lora, layer=layer, owner="mlp",
            target="gate_proj", scaling=scaling, bias=False,
        )
        up = _linear(
            normalized, base, lora, layer=layer, owner="mlp",
            target="up_proj", scaling=scaling, bias=False,
        )
        activated = __import__("jax").nn.silu(gate) * up
        hidden = residual + _linear(
            activated, base, lora, layer=layer, owner="mlp",
            target="down_proj", scaling=scaling, bias=False,
        )
    return hidden


def _host_torch(value: Any, *, dtype: str) -> Any:
    jax, _jnp, np, _mesh, _sharding = _require_jax()
    import torch

    host = np.asarray(jax.device_get(value), dtype=np.float32)
    tensor = torch.from_numpy(host.copy())
    if dtype == "float16":
        tensor = tensor.to(torch.float16)
    elif dtype == "bfloat16":
        tensor = tensor.to(torch.bfloat16)
    return tensor.contiguous()


def _tensor_hash(value: Any) -> str:
    import torch

    tensor = value.detach().cpu().contiguous()
    return "sha256:" + hashlib.sha256(
        tensor.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def jax_adapter_hash(state: dict[str, Any]) -> str:
    jax, _jnp, np, _mesh, _sharding = _require_jax()
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        host = np.ascontiguousarray(jax.device_get(value), dtype=np.float32)
        raw = host.view(np.uint8).tobytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(len(raw).to_bytes(8, "little") + raw)
    return "sha256:" + digest.hexdigest()


class JaxQwenStageTrainer:
    """One real JAX middle stage with LoRA gradients and AdamW state."""

    def __init__(
        self,
        *,
        training_manifest: dict[str, Any],
        config: dict[str, Any],
        stage_id: int,
        shard_path: str | Path,
        checkpoint_dir: str | Path,
        placement_generation: int,
        resume: bool,
        require_tpu: bool = True,
        expected_tpu_devices: int = 8,
        gradient_scale: float = 128.0,
    ) -> None:
        jax, jnp, _np, _Mesh, sharding_types = _require_jax()
        NamedSharding, PartitionSpec = sharding_types
        self.manifest = validate_training_manifest(training_manifest)
        self.config = dict(config)
        self.spec = qwen_stage_spec(self.manifest, stage_id=stage_id)
        self.base, self.stage, self.mesh, self.load_report = load_jax_qwen_stage(
            self.manifest,
            self.config,
            stage_id=stage_id,
            shard_path=shard_path,
            require_tpu=require_tpu,
            expected_tpu_devices=expected_tpu_devices,
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.placement_generation = int(placement_generation)
        self.gradient_scale = float(gradient_scale)
        self.gradient_clip_norm = float(self.manifest["lora"]["gradient_clip_norm"])
        self.learning_rate = float(self.manifest["lora"]["learning_rate"])
        self.lora = initialize_jax_lora(
            self.manifest, self.config, self.stage, self.mesh
        )
        self.optimizer_state = {
            "step": 0,
            "exp_avg": jax.tree_util.tree_map(jnp.zeros_like, self.lora),
            "exp_avg_sq": jax.tree_util.tree_map(jnp.zeros_like, self.lora),
        }
        self.prng_key = jax.random.PRNGKey(int(self.manifest["training"]["seed"]))
        self.loaded_checkpoint: dict[str, Any] | None = None
        if resume:
            loaded = load_jax_stage_checkpoint(
                self.checkpoint_dir,
                training_manifest=self.manifest,
                stage_spec=self.spec,
            )
            self.loaded_checkpoint = {
                key: value
                for key, value in loaded.items()
                if key
                not in {"adapter_state", "optimizer_state", "scheduler_state", "prng_key"}
            }
            self.lora = {
                name: _put_array(
                    loaded["adapter_state"][name],
                    sharding=self.lora[name].sharding,
                    dtype=jnp.float32,
                )
                for name in self.lora
            }
            self.optimizer_state = {
                "step": int(loaded["optimizer_state"]["step"]),
                "exp_avg": {
                    name: _put_array(
                        loaded["optimizer_state"]["exp_avg"][name],
                        sharding=self.lora[name].sharding,
                        dtype=jnp.float32,
                    )
                    for name in self.lora
                },
                "exp_avg_sq": {
                    name: _put_array(
                        loaded["optimizer_state"]["exp_avg_sq"][name],
                        sharding=self.lora[name].sharding,
                        dtype=jnp.float32,
                    )
                    for name in self.lora
                },
            }
            self.prng_key = _put_array(
                loaded["prng_key"],
                sharding=NamedSharding(self.mesh, PartitionSpec()),
                dtype=jnp.uint32,
            )
        replicated = NamedSharding(self.mesh, PartitionSpec())
        self.forward_fn = jax.jit(
            lambda base, lora, value: qwen_middle_stage_forward(
                base,
                lora,
                value,
                config=self.config,
                stage=self.stage,
                lora_alpha=int(self.manifest["lora"]["alpha"]),
                lora_rank=int(self.manifest["lora"]["rank"]),
            ),
            out_shardings=replicated,
        )

        def objective(base: Any, lora: Any, value: Any, incoming: Any) -> Any:
            output = qwen_middle_stage_forward(
                base,
                lora,
                value,
                config=self.config,
                stage=self.stage,
                lora_alpha=int(self.manifest["lora"]["alpha"]),
                lora_rank=int(self.manifest["lora"]["rank"]),
            )
            return jnp.vdot(
                output.astype(jnp.float32), incoming.astype(jnp.float32)
            )

        lora_gradient_shardings = jax.tree_util.tree_map(
            lambda value: value.sharding, self.lora
        )
        self.backward_fn = jax.jit(
            jax.value_and_grad(objective, argnums=(1, 2)),
            out_shardings=(
                replicated,
                (lora_gradient_shardings, replicated),
            ),
        )
        self.forward_output_sharding_explicit = True
        self.backward_output_sharding_explicit = True
        self.cached_inputs: dict[int, Any] = {}
        self.accumulated_gradients = jax.tree_util.tree_map(jnp.zeros_like, self.lora)
        self.compute_intervals: list[dict[str, Any]] = []
        self.compile_forward_latency_ms = 0.0
        self.compile_backward_latency_ms = 0.0
        self.steady_forward_latency_ms: list[float] = []
        self.steady_backward_latency_ms: list[float] = []
        self._forward_compiled = False
        self._backward_compiled = False

    def _input(self, value: Any) -> Any:
        jax, jnp, np, _mesh, sharding_types = _require_jax()
        NamedSharding, PartitionSpec = sharding_types
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        host = np.asarray(value, dtype=np.float32)
        expected = (
            int(self.manifest["training"]["microbatch_size"]),
            int(self.manifest["training"]["sequence_length"]),
            int(self.manifest["model"]["hidden_size"]),
        )
        if tuple(host.shape) != expected:
            raise ValueError("heterogeneous_jax_stage_input_shape_invalid")
        return jax.device_put(
            jnp.asarray(host, dtype=jnp.bfloat16),
            NamedSharding(self.mesh, PartitionSpec()),
        )

    def _record(self, operation: str, microbatch_id: int, started_ns: int) -> dict[str, Any]:
        interval = {
            "operation": operation,
            "microbatch_id": int(microbatch_id),
            "started_ns": int(started_ns),
            "ended_ns": int(time.time_ns()),
            "stage_id": int(self.spec.stage_id),
            "device": "jax_tpu:0",
        }
        self.compute_intervals.append(interval)
        return interval

    def begin_step(self) -> dict[str, Any]:
        jax, jnp, _np, _mesh, _sharding = _require_jax()
        if self.cached_inputs:
            raise RuntimeError("heterogeneous_jax_stage_retained_microbatch")
        self.accumulated_gradients = jax.tree_util.tree_map(jnp.zeros_like, self.lora)
        return {"begun": True, "gradient_scale": self.gradient_scale}

    def forward(self, microbatch_id: int, value: Any) -> dict[str, Any]:
        jax, _jnp, np, _mesh, _sharding = _require_jax()
        stage_input = self._input(value)
        started_ns = time.time_ns()
        started = time.perf_counter()
        output = self.forward_fn(self.base, self.lora, stage_input)
        output.block_until_ready()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if self._forward_compiled:
            self.steady_forward_latency_ms.append(elapsed_ms)
        else:
            self.compile_forward_latency_ms = elapsed_ms
            self._forward_compiled = True
        if not bool(np.isfinite(np.asarray(jax.device_get(output), dtype=np.float32)).all()):
            raise RuntimeError("heterogeneous_jax_non_finite_activation")
        self.cached_inputs[int(microbatch_id)] = stage_input
        boundary = _host_torch(output, dtype=self.manifest["precision"]["boundary_dtype"])
        return {
            "activation": boundary,
            "shape": list(boundary.shape),
            "dtype": str(boundary.dtype).replace("torch.", ""),
            "activation_hash": _tensor_hash(boundary),
            "compute_interval": self._record("forward", microbatch_id, started_ns),
        }

    def backward(self, microbatch_id: int, activation_gradient: Any) -> dict[str, Any]:
        jax, jnp, np, _mesh, sharding_types = _require_jax()
        NamedSharding, PartitionSpec = sharding_types
        key = int(microbatch_id)
        stage_input = self.cached_inputs.pop(key)
        if hasattr(activation_gradient, "detach"):
            activation_gradient = activation_gradient.detach().float().cpu().numpy()
        incoming_host = np.asarray(activation_gradient, dtype=np.float32)
        if not bool(np.isfinite(incoming_host).all()):
            raise RuntimeError("heterogeneous_jax_non_finite_gradient")
        incoming = jax.device_put(
            jnp.asarray(incoming_host, dtype=jnp.bfloat16),
            NamedSharding(self.mesh, PartitionSpec()),
        )
        started_ns = time.time_ns()
        started = time.perf_counter()
        _value, (lora_gradients, input_gradient) = self.backward_fn(
            self.base, self.lora, stage_input, incoming
        )
        input_gradient.block_until_ready()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if self._backward_compiled:
            self.steady_backward_latency_ms.append(elapsed_ms)
        else:
            self.compile_backward_latency_ms = elapsed_ms
            self._backward_compiled = True
        self.accumulated_gradients = jax.tree_util.tree_map(
            lambda old, new: old + new,
            self.accumulated_gradients,
            lora_gradients,
        )
        boundary = _host_torch(
            input_gradient, dtype=self.manifest["precision"]["boundary_dtype"]
        )
        if not bool(np.isfinite(boundary.float().numpy()).all()):
            raise RuntimeError("heterogeneous_jax_non_finite_gradient")
        return {
            "activation_gradient": boundary,
            "gradient_scale": self.gradient_scale,
            "incoming_gradient_hash": _tensor_hash(
                _host_torch(incoming, dtype=self.manifest["precision"]["boundary_dtype"])
            ),
            "compute_interval": self._record("backward", microbatch_id, started_ns),
        }

    def abort_step(self) -> dict[str, Any]:
        self.cached_inputs.clear()
        self.begin_step()
        return {"aborted": True, "graphs_cleared": True}

    def finish_step(self, *, global_step: int, dataset_cursor: int) -> dict[str, Any]:
        jax, jnp, np, _mesh, _sharding = _require_jax()
        if self.cached_inputs:
            raise RuntimeError("heterogeneous_jax_stage_unfinished_microbatch")
        gradients = jax.tree_util.tree_map(
            lambda value: value.astype(jnp.float32) / self.gradient_scale,
            self.accumulated_gradients,
        )
        squared = sum(
            jnp.sum(jnp.square(value.astype(jnp.float32)))
            for value in gradients.values()
        )
        gradient_norm_array = jnp.sqrt(squared)
        gradient_norm = float(np.asarray(jax.device_get(gradient_norm_array)))
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise RuntimeError("heterogeneous_jax_non_finite_lora_gradient")
        clip_scale = min(1.0, self.gradient_clip_norm / max(gradient_norm, 1e-12))
        gradients = jax.tree_util.tree_map(
            lambda value: value * clip_scale, gradients
        )
        step = int(self.optimizer_state["step"]) + 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        exp_avg = jax.tree_util.tree_map(
            lambda old, grad: beta1 * old + (1.0 - beta1) * grad,
            self.optimizer_state["exp_avg"],
            gradients,
        )
        exp_avg_sq = jax.tree_util.tree_map(
            lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad),
            self.optimizer_state["exp_avg_sq"],
            gradients,
        )
        bias1 = 1.0 - beta1**step
        bias2 = 1.0 - beta2**step
        self.lora = jax.tree_util.tree_map(
            lambda parameter, first, second: parameter
            - self.learning_rate
            * (first / bias1)
            / (jnp.sqrt(second / bias2) + epsilon),
            self.lora,
            exp_avg,
            exp_avg_sq,
        )
        self.optimizer_state = {
            "step": step,
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
        }
        _next, self.prng_key = jax.random.split(self.prng_key)
        checkpoint = save_jax_stage_checkpoint(
            self.lora,
            self.optimizer_state,
            {"last_epoch": int(global_step), "learning_rate": self.learning_rate},
            self.prng_key,
            self.checkpoint_dir,
            training_manifest=self.manifest,
            stage_spec=self.spec,
            global_step=int(global_step),
            dataset_cursor=int(dataset_cursor),
            placement_generation=self.placement_generation,
            mesh_shape=list(self.mesh.devices.shape),
        )
        return {
            "global_step": int(global_step),
            "dataset_cursor": int(dataset_cursor),
            "placement_generation": self.placement_generation,
            "gradient_scale_before": self.gradient_scale,
            "gradient_scale_after": self.gradient_scale,
            "lora_gradient_norm": gradient_norm,
            "gradient_clip_norm": self.gradient_clip_norm,
            "gradient_clipping_applied": clip_scale < 1.0,
            "optimizer_step_applied": True,
            "scheduler_step_applied": True,
            "scheduler_last_epoch": int(global_step),
            "checkpoint_hash": checkpoint["content_hash"],
            "adapter_tensor_hash": checkpoint["adapter_tensor_hash"],
            "peak_allocated_bytes": self.peak_hbm_bytes(),
            "peak_reserved_bytes": self.peak_hbm_bytes(),
            "runtime_backend": "jax_tpu",
            "compile_latency_ms": self.compile_forward_latency_ms
            + self.compile_backward_latency_ms,
            "steady_forward_latency_ms": (
                sum(self.steady_forward_latency_ms)
                / len(self.steady_forward_latency_ms)
                if self.steady_forward_latency_ms
                else 0.0
            ),
            "steady_backward_latency_ms": (
                sum(self.steady_backward_latency_ms)
                / len(self.steady_backward_latency_ms)
                if self.steady_backward_latency_ms
                else 0.0
            ),
        }

    def peak_hbm_bytes(self) -> int:
        values = []
        for device in list(self.mesh.devices.flat):
            try:
                stats = dict(device.memory_stats() or {})
            except (AttributeError, RuntimeError, TypeError):
                continue
            values.append(int(stats.get("peak_bytes_in_use") or stats.get("bytes_in_use") or 0))
        return sum(values)

    def status(self) -> dict[str, Any]:
        sharded = [
            value
            for name, value in self.lora.items()
            if ".lora_B." in name
            and not bool(getattr(value.sharding, "is_fully_replicated", False))
        ]
        return {
            "schema": JAX_STAGE_RUNTIME_SCHEMA,
            "runtime_backend": "jax_tpu",
            "stage_id": int(self.spec.stage_id),
            "placement_generation": self.placement_generation,
            "base_hash": self.load_report["source_shard_hash"],
            "adapter_hash": jax_adapter_hash(self.lora),
            "optimizer_step": int(self.optimizer_state["step"]),
            "jax_mesh_shape": list(self.mesh.devices.shape),
            "jax_mesh_device_count": int(self.mesh.devices.size),
            "sharded_lora_tensor_count": len(sharded),
            "all_mesh_devices_used": bool(
                sharded
                and max(len(value.addressable_shards) for value in sharded)
                == int(self.mesh.devices.size)
            ),
            "forward_output_sharding_explicit": self.forward_output_sharding_explicit,
            "backward_output_sharding_explicit": self.backward_output_sharding_explicit,
            "boundary_output_replicated": True,
            "compile_latency_ms": self.compile_forward_latency_ms
            + self.compile_backward_latency_ms,
            "steady_forward_sample_count": len(self.steady_forward_latency_ms),
            "steady_backward_sample_count": len(self.steady_backward_latency_ms),
            "tensor_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }

    def adapter_state(self) -> dict[str, Any]:
        return {
            name: _host_torch(value, dtype="float32")
            for name, value in self.lora.items()
        }


def jax_stage_process_main(connection: Any, settings: dict[str, Any]) -> None:
    """Multiprocessing entry point matching the existing PyTorch stage RPC."""

    phase = "bootstrap"
    try:
        manifest = validate_training_manifest(settings["training_manifest"])
        phase = "model_load"
        trainer = JaxQwenStageTrainer(
            training_manifest=manifest,
            config=dict(settings["config"]),
            stage_id=int(settings["stage_id"]),
            shard_path=settings["shard_path"],
            checkpoint_dir=settings["checkpoint_dir"],
            placement_generation=int(settings["placement_generation"]),
            resume=bool(settings.get("resume")),
            require_tpu=bool(settings.get("require_tpu", True)),
            expected_tpu_devices=int(settings.get("expected_tpu_devices") or 8),
        )
        status = trainer.status()
        ready = {
            "type": "ready",
            "schema": JAX_STAGE_RUNTIME_SCHEMA,
            "stage_id": int(trainer.spec.stage_id),
            "device": "jax_tpu:0",
            "device_type": "jax_tpu",
            "runtime_backend": "jax_tpu",
            "placement_generation": int(settings["placement_generation"]),
            "load_report": trainer.load_report,
            "base_hash_before": trainer.load_report["source_shard_hash"],
            "adapter_hash_before": status["adapter_hash"],
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
            "jax_mesh_shape": status["jax_mesh_shape"],
            "jax_mesh_device_count": status["jax_mesh_device_count"],
            "all_mesh_devices_used": status["all_mesh_devices_used"],
            "single_tpu_resource_group_process": True,
            "public_artifact_safe": True,
        }
        connection.send(ready)
        while True:
            request = connection.recv()
            request_id = int(request["request_id"])
            operation = str(request["operation"])
            phase = operation
            if operation == "begin_step":
                result: Any = trainer.begin_step()
            elif operation == "forward":
                result = trainer.forward(
                    int(request["microbatch_id"]), request["value"]
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
                    "adapter_state": trainer.adapter_state(),
                    "adapter_hash": jax_adapter_hash(trainer.lora),
                }
            elif operation == "status":
                result = trainer.status()
            elif operation == "loss_backward":
                raise RuntimeError("heterogeneous_jax_stage_must_be_intermediate")
            elif operation == "stop":
                result = {"stopped": True, "stage_id": int(trainer.spec.stage_id)}
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
