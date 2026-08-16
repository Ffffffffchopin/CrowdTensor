"""Versioned model-family boundary used by heterogeneous training runtimes."""

from __future__ import annotations

import abc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .version import MODEL_ADAPTER_API_VERSION, __version__


MODEL_ADAPTER_DESCRIPTOR_SCHEMA = "crowdtensor_model_adapter_descriptor_v1"
MODEL_ADAPTER_REGISTRY_SCHEMA = "crowdtensor_model_adapter_registry_v2"
MODEL_ADAPTER_CONFORMANCE_SCHEMA = "crowdtensor_model_adapter_conformance_v1"
MODEL_ADAPTER_ENTRY_POINT_GROUP = "crowdtensor.model_adapters.v1"
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{2,63}")


class ModelAdapterError(ValueError):
    """Raised for an explicitly unsupported or malformed model request."""


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def disable_incompatible_optional_torchao_dispatch(model: Any) -> bool:
    """Ignore an outdated optional TorchAO install for ordinary dense weights."""

    if importlib.util.find_spec("torchao") is None:
        return False
    try:
        from packaging.version import Version

        installed = Version(importlib.metadata.version("torchao"))
    except (ImportError, importlib.metadata.PackageNotFoundError):
        return False
    if installed >= Version("0.16.0"):
        return False
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if weight is not None and type(weight).__module__.startswith("torchao"):
            raise ModelAdapterError("model_adapter_outdated_torchao_quantized_weight_unsupported")
    from peft.tuners.lora import torchao as peft_torchao

    peft_torchao.is_torchao_available = lambda: False
    return True


@dataclass(frozen=True)
class StageSpec:
    stage_id: int
    layer_start: int
    layer_end: int
    owns_embedding: bool
    owns_norm: bool
    owns_lm_head: bool

    @property
    def layer_count(self) -> int:
        return self.layer_end - self.layer_start

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["layer_count"] = self.layer_count
        return result


class ModelAdapter(abc.ABC):
    """Stable family interface; implementations must fail closed."""

    adapter_id: str
    family: str
    model_types: tuple[str, ...]
    architectures: tuple[str, ...]
    default_model_id: str
    default_revision: str
    default_model_license: str = "unknown"
    target_modules: tuple[str, ...]
    default_config: Mapping[str, Any]
    recommended_stage_count: int = 2
    production_scheduler_supported: bool = False
    extra_capabilities: tuple[str, ...] = ()

    def descriptor(self) -> dict[str, Any]:
        value = {
            "schema": MODEL_ADAPTER_DESCRIPTOR_SCHEMA,
            "api_version": MODEL_ADAPTER_API_VERSION,
            "adapter_id": self.adapter_id,
            "family": self.family,
            "model_types": sorted(self.model_types),
            "architectures": sorted(self.architectures),
            "default_model_id": self.default_model_id,
            "default_revision": self.default_revision,
            "default_model_license": self.default_model_license,
            "capabilities": sorted(
                {
                    "config_discovery",
                    "stage_partition",
                    "stage_loading",
                    "peft_lora",
                    "checkpoint",
                    "export",
                    "independent_reload",
                    "resource_estimate",
                    *self.extra_capabilities,
                }
            ),
            "training_modes": ["peft_lora"],
            "recommended_stage_count": int(self.recommended_stage_count),
            "production_scheduler_supported": bool(
                self.production_scheduler_supported
            ),
            "entry_point_plugin_compatible": True,
            "full_parameter_training_supported": False,
            "automatic_arbitrary_architecture_partition_supported": False,
            "public_artifact_safe": True,
        }
        value["content_hash"] = stable_hash(value)
        return value

    def canonical_config(self) -> dict[str, Any]:
        value = json.loads(json.dumps(dict(self.default_config)))
        self.validate_config(value)
        return value

    def supports(self, *, model_id: str, config: Mapping[str, Any]) -> bool:
        model_type = str(config.get("model_type") or "").lower()
        architectures = {str(item) for item in config.get("architectures") or []}
        return bool(
            model_type in self.model_types
            or architectures.intersection(self.architectures)
            or str(model_id) == self.default_model_id
        )

    def validate_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        model_type = str(config.get("model_type") or "").lower()
        layers = int(config.get("num_hidden_layers") or config.get("n_layer") or 0)
        hidden = int(config.get("hidden_size") or config.get("n_embd") or 0)
        vocab = int(config.get("vocab_size") or 0)
        if model_type not in self.model_types or min(layers, hidden, vocab) <= 0:
            raise ModelAdapterError("model_adapter_config_unsupported")
        return {
            "model_type": model_type,
            "num_hidden_layers": layers,
            "hidden_size": hidden,
            "intermediate_size": int(config.get("intermediate_size") or hidden * 4),
            "num_attention_heads": int(config.get("num_attention_heads") or 1),
            "num_key_value_heads": int(
                config.get("num_key_value_heads")
                or config.get("num_attention_heads")
                or 1
            ),
            "vocab_size": vocab,
            "architectures": sorted(str(item) for item in config.get("architectures") or []),
        }

    def partition(self, config: Mapping[str, Any], *, stage_count: int) -> list[StageSpec]:
        canonical = self.validate_config(config)
        layers = int(canonical["num_hidden_layers"])
        count = int(stage_count)
        if count < 2 or count > layers:
            raise ModelAdapterError("model_adapter_stage_count_invalid")
        boundaries = [round(index * layers / count) for index in range(count + 1)]
        if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
            raise ModelAdapterError("model_adapter_partition_empty_stage")
        return [
            StageSpec(
                stage_id=index,
                layer_start=boundaries[index],
                layer_end=boundaries[index + 1],
                owns_embedding=index == 0,
                owns_norm=index == count - 1,
                owns_lm_head=index == count - 1,
            )
            for index in range(count)
        ]

    def estimate_resources(
        self,
        config: Mapping[str, Any],
        stage: StageSpec,
        *,
        dtype_bytes: int = 2,
        lora_rank: int = 8,
    ) -> dict[str, Any]:
        canonical = self.validate_config(config)
        hidden = int(canonical["hidden_size"])
        intermediate = int(canonical["intermediate_size"])
        vocab = int(canonical["vocab_size"])
        layer_parameters = stage.layer_count * (
            4 * hidden * hidden + 3 * hidden * intermediate + 2 * hidden
        )
        extra = (vocab * hidden if stage.owns_embedding else 0) + (
            vocab * hidden + hidden if stage.owns_lm_head else 0
        )
        parameter_count = layer_parameters + extra
        lora_parameters = stage.layer_count * len(self.target_modules) * 2 * hidden * int(lora_rank)
        return {
            "schema": "crowdtensor_model_adapter_resource_estimate_v1",
            "adapter_id": self.adapter_id,
            "stage_id": stage.stage_id,
            "parameter_count": parameter_count,
            "weight_bytes": parameter_count * int(dtype_bytes),
            "lora_parameter_count": lora_parameters,
            "optimizer_bytes": lora_parameters * 8,
            "estimate_only": True,
            "public_artifact_safe": True,
        }

    def load_model(
        self,
        *,
        model_id: str,
        revision: str,
        device: str,
        dtype: Any,
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
    ) -> Any:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            local_files_only=local_files_only,
            cache_dir=str(cache_dir) if cache_dir else None,
            trust_remote_code=False,
        )
        if not self.supports(model_id=model_id, config=model.config.to_dict()):
            raise ModelAdapterError("model_adapter_loaded_architecture_unsupported")
        return model.to(device)

    def apply_lora(
        self,
        model: Any,
        *,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> Any:
        from peft import LoraConfig, get_peft_model

        torchao_dispatch_disabled = disable_incompatible_optional_torchao_dispatch(model)
        config = LoraConfig(
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(self.target_modules),
        )
        result = get_peft_model(model, config)
        result._crowdtensor_outdated_optional_torchao_dispatch_disabled = (
            torchao_dispatch_disabled
        )
        return result

    def export_adapter(self, model: Any, output_dir: str | Path) -> dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(destination, safe_serialization=True)
        files = sorted(path.name for path in destination.iterdir() if path.is_file())
        if "adapter_config.json" not in files:
            raise ModelAdapterError("model_adapter_export_missing_config")
        return {
            "schema": "crowdtensor_model_adapter_export_v1",
            "adapter_id": self.adapter_id,
            "file_names": files,
            "safe_serialization": "adapter_model.safetensors" in files,
            "public_artifact_safe": True,
        }

    def reload_adapter(
        self,
        *,
        model_id: str,
        revision: str,
        adapter_dir: str | Path,
        device: str,
        dtype: Any,
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
    ) -> Any:
        from peft import PeftModel

        base = self.load_model(
            model_id=model_id,
            revision=revision,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        torchao_dispatch_disabled = disable_incompatible_optional_torchao_dispatch(base)
        result = PeftModel.from_pretrained(
            base, str(adapter_dir), is_trainable=False
        ).to(device)
        result._crowdtensor_outdated_optional_torchao_dispatch_disabled = (
            torchao_dispatch_disabled
        )
        return result

    @abc.abstractmethod
    def production_manifest(self, *, target_steps: int, accelerators: Iterable[str]) -> dict[str, Any]:
        """Return a validated scheduler manifest for this family."""


class QwenModelAdapter(ModelAdapter):
    adapter_id = "qwen2_lora_v1"
    family = "qwen2"
    model_types = ("qwen2",)
    architectures = ("Qwen2ForCausalLM",)
    default_model_id = "Qwen/Qwen2.5-7B"
    default_revision = "d149729398750b98c0af14eb82c78cfe92750796"
    default_model_license = "apache-2.0"
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    default_config = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "vocab_size": 152064,
    }
    recommended_stage_count = 5
    production_scheduler_supported = True

    def production_manifest(self, *, target_steps: int, accelerators: Iterable[str]) -> dict[str, Any]:
        from .adapters.manifests import (
            qwen25_7b_lora_manifest,
            qwen25_7b_lora_tpu_manifest,
        )

        selected = {str(item) for item in accelerators}
        if "jax_tpu" in selected:
            return qwen25_7b_lora_tpu_manifest(target_steps=int(target_steps))
        return qwen25_7b_lora_manifest(target_steps=int(target_steps))


class SmolLMModelAdapter(ModelAdapter):
    adapter_id = "smollm2_lora_v1"
    family = "smollm2"
    model_types = ("llama",)
    architectures = ("LlamaForCausalLM",)
    default_model_id = "HuggingFaceTB/SmolLM2-135M"
    default_revision = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
    default_model_license = "apache-2.0"
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    default_config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 30,
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "vocab_size": 49152,
    }
    extra_capabilities = ("bounded_two_stage_live",)

    def production_manifest(self, *, target_steps: int, accelerators: Iterable[str]) -> dict[str, Any]:
        raise ModelAdapterError("smollm2_full_production_scheduler_not_supported")


class SmolLM3ModelAdapter(ModelAdapter):
    adapter_id = "smollm3_lora_v1"
    family = "smollm3"
    model_types = ("smollm3",)
    architectures = ("SmolLM3ForCausalLM",)
    default_model_id = "HuggingFaceTB/SmolLM3-3B-Base"
    default_revision = "d78a42f79198603e614095753484a04c10c2b940"
    default_model_license = "apache-2.0"
    target_modules = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    default_config = {
        "model_type": "smollm3",
        "architectures": ["SmolLM3ForCausalLM"],
        "num_hidden_layers": 36,
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "vocab_size": 128256,
    }
    recommended_stage_count = 4
    extra_capabilities = ("commons_campaign", "instruction_sft")

    def production_manifest(
        self, *, target_steps: int, accelerators: Iterable[str]
    ) -> dict[str, Any]:
        raise ModelAdapterError("smollm3_full_production_scheduler_not_supported")


_BUILTIN_ADAPTERS: dict[str, ModelAdapter] = {
    adapter.adapter_id: adapter
    for adapter in (QwenModelAdapter(), SmolLMModelAdapter(), SmolLM3ModelAdapter())
}


def check_model_adapter_conformance(adapter: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(adapter, ModelAdapter):
        errors.append("model_adapter_plugin_object_type_invalid")
    else:
        if not _IDENTIFIER.fullmatch(str(adapter.adapter_id or "")):
            errors.append("model_adapter_id_invalid")
        if not _IDENTIFIER.fullmatch(str(adapter.family or "")):
            errors.append("model_adapter_family_invalid")
        if not adapter.model_types or any(
            not _IDENTIFIER.fullmatch(str(item)) for item in adapter.model_types
        ):
            errors.append("model_adapter_model_types_invalid")
        if not adapter.architectures or any(
            not str(item).endswith("ForCausalLM") for item in adapter.architectures
        ):
            errors.append("model_adapter_architectures_invalid")
        if not str(adapter.default_model_id or "").strip() or not str(
            adapter.default_revision or ""
        ).strip():
            errors.append("model_adapter_default_source_invalid")
        if not adapter.target_modules or any(
            not _IDENTIFIER.fullmatch(str(item)) for item in adapter.target_modules
        ):
            errors.append("model_adapter_lora_targets_invalid")
        if int(adapter.recommended_stage_count) < 2:
            errors.append("model_adapter_recommended_stage_count_invalid")
        try:
            config = adapter.canonical_config()
            canonical = adapter.validate_config(config)
            stages = adapter.partition(
                canonical, stage_count=int(adapter.recommended_stage_count)
            )
            estimates = [adapter.estimate_resources(canonical, item) for item in stages]
            if stages[0].layer_start != 0 or stages[-1].layer_end != int(
                canonical["num_hidden_layers"]
            ):
                errors.append("model_adapter_partition_coverage_invalid")
            if any(
                left.layer_end != right.layer_start
                for left, right in zip(stages, stages[1:])
            ):
                errors.append("model_adapter_partition_contiguity_invalid")
            if any(int(item.get("weight_bytes") or 0) <= 0 for item in estimates):
                errors.append("model_adapter_resource_estimate_invalid")
        except (KeyError, TypeError, ValueError, ModelAdapterError):
            config = {}
            stages = []
            errors.append("model_adapter_canonical_config_invalid")
        descriptor = adapter.descriptor()
        supplied = str(descriptor.get("content_hash") or "")
        expected = stable_hash(
            {key: value for key, value in descriptor.items() if key != "content_hash"}
        )
        if descriptor.get("schema") != MODEL_ADAPTER_DESCRIPTOR_SCHEMA:
            errors.append("model_adapter_descriptor_schema_invalid")
        if descriptor.get("api_version") != MODEL_ADAPTER_API_VERSION:
            errors.append("model_adapter_api_version_invalid")
        if supplied != expected:
            errors.append("model_adapter_descriptor_hash_invalid")
    report = {
        "schema": MODEL_ADAPTER_CONFORMANCE_SCHEMA,
        "ok": not errors,
        "adapter_id": str(getattr(adapter, "adapter_id", "") or ""),
        "family": str(getattr(adapter, "family", "") or ""),
        "api_version": MODEL_ADAPTER_API_VERSION,
        "errors": sorted(set(errors)),
        "canonical_config_verified": bool(not errors and config),
        "partition_verified": bool(not errors and stages),
        "entry_point_contract_verified": isinstance(adapter, ModelAdapter),
        "real_weight_live_required_separately": True,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return report


def _plugins_enabled() -> bool:
    return str(os.environ.get("CROWDTENSOR_DISABLE_MODEL_ADAPTER_PLUGINS") or "").lower() not in {
        "1",
        "true",
        "yes",
    }


def _entry_points() -> list[Any]:
    discovered = importlib.metadata.entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=MODEL_ADAPTER_ENTRY_POINT_GROUP))
    return list(discovered.get(MODEL_ADAPTER_ENTRY_POINT_GROUP, ()))


def _plugin_records() -> dict[str, tuple[ModelAdapter, dict[str, Any]]]:
    if not _plugins_enabled():
        return {}
    records: dict[str, tuple[ModelAdapter, dict[str, Any]]] = {}
    failures: list[str] = []
    points = sorted(
        _entry_points(), key=lambda item: (str(getattr(item, "name", "")), str(getattr(item, "value", "")))
    )
    for point in points:
        try:
            loaded = point.load()
            if isinstance(loaded, ModelAdapter):
                adapter = loaded
            elif isinstance(loaded, type) and issubclass(loaded, ModelAdapter):
                adapter = loaded()
            elif callable(loaded):
                adapter = loaded()
            else:
                adapter = loaded
            conformance = check_model_adapter_conformance(adapter)
            if conformance["ok"] is not True:
                raise ModelAdapterError("model_adapter_plugin_conformance_failed")
            adapter_id = str(adapter.adapter_id)
            if str(point.name) != adapter_id:
                raise ModelAdapterError("model_adapter_plugin_entry_point_name_mismatch")
            if adapter_id in _BUILTIN_ADAPTERS or adapter_id in records:
                raise ModelAdapterError("model_adapter_plugin_id_conflict")
            distribution = getattr(point, "dist", None)
            metadata = getattr(distribution, "metadata", {}) if distribution is not None else {}
            name = str(metadata.get("Name") or "") if hasattr(metadata, "get") else ""
            version = str(getattr(distribution, "version", "") or "")
            if not name or not version:
                raise ModelAdapterError("model_adapter_plugin_distribution_metadata_missing")
            records[adapter_id] = (
                adapter,
                {
                    "kind": "entry_point_plugin",
                    "entry_point_group": MODEL_ADAPTER_ENTRY_POINT_GROUP,
                    "entry_point_name": str(point.name),
                    "distribution_name": name,
                    "distribution_version": version,
                    "module_path_public": False,
                    "installed_location_public": False,
                },
            )
        except Exception as exc:
            reason = str(exc).splitlines()[0] if str(exc).startswith("model_adapter_") else (
                "model_adapter_plugin_load_failed:" + type(exc).__name__
            )
            failures.append(reason[:180])
    if failures:
        raise ModelAdapterError(
            "model_adapter_plugin_discovery_failed:" + sorted(set(failures))[0]
        )
    return records


def model_adapter_records() -> dict[str, tuple[ModelAdapter, dict[str, Any]]]:
    records = {
        adapter_id: (
            adapter,
            {
                "kind": "builtin",
                "distribution_name": "crowdtensord",
                "distribution_version": __version__,
                "module_path_public": False,
                "installed_location_public": False,
            },
        )
        for adapter_id, adapter in _BUILTIN_ADAPTERS.items()
    }
    records.update(_plugin_records())
    return records


def model_adapters() -> tuple[ModelAdapter, ...]:
    records = model_adapter_records()
    return tuple(records[key][0] for key in sorted(records))


def get_model_adapter(adapter_id: str) -> ModelAdapter:
    records = model_adapter_records()
    try:
        return records[str(adapter_id)][0]
    except KeyError as exc:
        raise ModelAdapterError("model_adapter_id_unsupported") from exc


def get_model_adapter_registration(adapter_id: str) -> dict[str, Any]:
    records = model_adapter_records()
    try:
        return dict(records[str(adapter_id)][1])
    except KeyError as exc:
        raise ModelAdapterError("model_adapter_id_unsupported") from exc


def resolve_model_adapter(*, model_id: str, config: Mapping[str, Any]) -> ModelAdapter:
    matches = [adapter for adapter in model_adapters() if adapter.supports(model_id=model_id, config=config)]
    if len(matches) != 1:
        raise ModelAdapterError("model_adapter_resolution_unsupported_or_ambiguous")
    return matches[0]


def adapter_registry_report() -> dict[str, Any]:
    records = model_adapter_records()
    descriptors: list[dict[str, Any]] = []
    for key in sorted(records):
        adapter, registration = records[key]
        descriptor = adapter.descriptor()
        descriptor.pop("content_hash", None)
        descriptor["registration"] = registration
        descriptor["content_hash"] = stable_hash(descriptor)
        descriptors.append(descriptor)
    value = {
        "schema": MODEL_ADAPTER_REGISTRY_SCHEMA,
        "api_version": MODEL_ADAPTER_API_VERSION,
        "entry_point_group": MODEL_ADAPTER_ENTRY_POINT_GROUP,
        "plugin_discovery_enabled": _plugins_enabled(),
        "builtin_adapter_count": sum(
            item[1]["kind"] == "builtin" for item in records.values()
        ),
        "plugin_adapter_count": sum(
            item[1]["kind"] == "entry_point_plugin" for item in records.values()
        ),
        "adapters": descriptors,
        "supported_model_families": sorted(
            adapter.family for adapter, _registration in records.values()
        ),
        "unsupported_capabilities": sorted(
            [
                "arbitrary_architecture_partition",
                "data_parallel_training",
                "full_parameter_training",
                "in_flight_stage_migration",
                "parameter_limit_exploration",
            ]
        ),
        "public_artifact_safe": True,
    }
    value["content_hash"] = stable_hash(value)
    return value
