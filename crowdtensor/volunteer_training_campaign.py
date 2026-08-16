"""Pinned public-source Campaign import for volunteer PEFT training."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hf_lora_training import configure_cpu_determinism
from .adapters.text_data import (
    DATA_PACK_MANIFEST_FILE,
    DATA_PACK_RECORDS_FILE,
    load_data_pack,
    load_instruction_records,
    tokenize_fixed_sequences,
    tokenize_instruction_records,
    validate_instruction_data_pack,
)
from .model_adapter import SmolLM3ModelAdapter, SmolLMModelAdapter, get_model_adapter
from .named_tensor_optimizer import load_tensors
from .training_contract import (
    DATASET_SCHEMA,
    JOB_SCHEMA,
    LORA_SCHEMA,
    MODEL_SCHEMA,
    WORKLOAD_TYPE,
    public_training_spec,
    sha256_file,
    sha256_json,
    tensor_specs,
)


IMPORT_SCHEMA = "crowdtensor_volunteer_campaign_import_v1"
MODEL_SOURCE_SCHEMA = "crowdtensor_volunteer_model_source_v1"
DATASET_SOURCE_SCHEMA = "crowdtensor_volunteer_dataset_source_v1"
IMPORT_PROFILE = "smollm2_135m_wikitext2_lora_v1"
COMMONS_IMPORT_PROFILE = "commons_instruction_sft_lora_v1"

_IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}\Z")

MODEL_ID = SmolLMModelAdapter.default_model_id
MODEL_REVISION = SmolLMModelAdapter.default_revision
MODEL_LICENSE = SmolLMModelAdapter.default_model_license
MODEL_ADAPTER_ID = SmolLMModelAdapter.adapter_id
COMMONS_MODEL_ID = SmolLM3ModelAdapter.default_model_id
COMMONS_MODEL_REVISION = SmolLM3ModelAdapter.default_revision
COMMONS_MODEL_LICENSE = SmolLM3ModelAdapter.default_model_license
COMMONS_MODEL_ADAPTER_ID = SmolLM3ModelAdapter.adapter_id
DATASET_ID = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_LICENSES = ("cc-by-sa-3.0", "gfdl")
DATASET_FILES = {
    "train": (
        f"{DATASET_CONFIG}/train-00000-of-00001.parquet",
        "sha256:e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
    ),
    "validation": (
        f"{DATASET_CONFIG}/validation-00000-of-00001.parquet",
        "sha256:204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
    ),
}


@dataclass(frozen=True)
class CampaignAssetBundle:
    model_dir: Path
    train_parquet: Path
    validation_parquet: Path
    model_source: dict[str, Any]
    dataset_source: dict[str, Any]


def _write_json(path: Path, value: dict[str, Any], *, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(mode)
    return path


def _file_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "relative_name": relative,
                "sha256": sha256_file(path),
                "byte_count": int(path.stat().st_size),
            }
        )
    if not records:
        raise RuntimeError("volunteer_campaign_model_snapshot_empty")
    return records


def _license_value(info: Any) -> Any:
    card = getattr(info, "card_data", None)
    if isinstance(card, dict):
        return card.get("license")
    return getattr(card, "license", None)


def _copy_snapshot(source: Path, destination: Path) -> list[dict[str, Any]]:
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        target.chmod(0o600)
    return _file_records(destination)


def download_pinned_campaign_assets(output_dir: str | Path) -> CampaignAssetBundle:
    """Fetch exact public revisions and return a fully verified private bundle."""

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    root = Path(output_dir).expanduser().resolve()
    cache = root / ".hf-cache"
    model_dir = root / "model"
    dataset_dir = root / "dataset"
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    model_info = api.model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if (
        str(model_info.sha or "") != MODEL_REVISION
        or model_info.gated is True
        or model_info.private is True
        or str(_license_value(model_info) or "").lower() != MODEL_LICENSE
    ):
        raise RuntimeError("volunteer_campaign_model_source_contract_mismatch")
    dataset_info = api.dataset_info(
        DATASET_ID, revision=DATASET_REVISION, files_metadata=True
    )
    licenses = _license_value(dataset_info)
    normalized_licenses = (
        sorted(str(item).lower() for item in licenses)
        if isinstance(licenses, (list, tuple))
        else [str(licenses or "").lower()]
    )
    if (
        str(dataset_info.sha or "") != DATASET_REVISION
        or dataset_info.gated is True
        or dataset_info.private is True
        or normalized_licenses != sorted(DATASET_LICENSES)
    ):
        raise RuntimeError("volunteer_campaign_dataset_source_contract_mismatch")

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(cache),
            local_files_only=False,
        )
    )
    model_files = _copy_snapshot(snapshot, model_dir)
    if not {"config.json", "model.safetensors", "tokenizer.json"}.issubset(
        {str(item["relative_name"]) for item in model_files}
    ):
        raise RuntimeError("volunteer_campaign_model_runtime_files_missing")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_records: list[dict[str, Any]] = []
    local_dataset: dict[str, Path] = {}
    for split, (remote_name, expected_hash) in DATASET_FILES.items():
        cached = Path(
            hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                revision=DATASET_REVISION,
                filename=remote_name,
                cache_dir=str(cache),
            )
        )
        target = dataset_dir / f"{split}.parquet"
        shutil.copyfile(cached, target)
        target.chmod(0o600)
        actual_hash = sha256_file(target)
        if actual_hash != expected_hash:
            raise RuntimeError(f"volunteer_campaign_{split}_snapshot_hash_mismatch")
        local_dataset[split] = target
        dataset_records.append(
            {
                "split": split,
                "relative_name": remote_name,
                "sha256": actual_hash,
                "byte_count": int(target.stat().st_size),
            }
        )

    model_source = {
        "schema": MODEL_SOURCE_SCHEMA,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "architecture": "LlamaForCausalLM",
        "adapter_id": MODEL_ADAPTER_ID,
        "source_public": True,
        "immutable_revision": True,
        "gated": False,
        "private": False,
        "imported_files": model_files,
        "imported_file_count": len(model_files),
        "imported_snapshot_hash": sha256_json(model_files),
        "source_verified": True,
    }
    dataset_source = {
        "schema": DATASET_SOURCE_SCHEMA,
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "config": DATASET_CONFIG,
        "licenses": sorted(DATASET_LICENSES),
        "source_public": True,
        "immutable_revision": True,
        "gated": False,
        "private": False,
        "source_files": dataset_records,
        "source_snapshot_hash": sha256_json(dataset_records),
        "source_verified": True,
        "raw_text_public": False,
        "token_ids_public": False,
    }
    shutil.rmtree(cache, ignore_errors=True)
    return CampaignAssetBundle(
        model_dir=model_dir,
        train_parquet=local_dataset["train"],
        validation_parquet=local_dataset["validation"],
        model_source=model_source,
        dataset_source=dataset_source,
    )


def _validate_asset_bundle(assets: CampaignAssetBundle) -> None:
    model_files = _file_records(assets.model_dir)
    if model_files != list(assets.model_source.get("imported_files") or []):
        raise RuntimeError("volunteer_campaign_model_snapshot_changed")
    if sha256_json(model_files) != assets.model_source.get("imported_snapshot_hash"):
        raise RuntimeError("volunteer_campaign_model_snapshot_hash_mismatch")
    source_files = {
        str(item.get("split") or ""): item
        for item in assets.dataset_source.get("source_files") or []
        if isinstance(item, dict)
    }
    for split, path in {
        "train": assets.train_parquet,
        "validation": assets.validation_parquet,
    }.items():
        record = source_files.get(split) or {}
        if sha256_file(path) != record.get("sha256"):
            raise RuntimeError("volunteer_campaign_dataset_snapshot_changed")
    if not (
        assets.model_source.get("source_verified") is True
        and assets.dataset_source.get("source_verified") is True
    ):
        raise RuntimeError("volunteer_campaign_source_not_verified")


def _materialize_tokenized_peft_fixture(
    output_dir: str | Path,
    *,
    model_dir: str | Path,
    model_source: dict[str, Any],
    dataset_source: dict[str, Any],
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    train_source_ids: list[str],
    validation_source_ids: list[str],
    job_id: str,
    import_profile: str,
    model_adapter_id: str,
    dataset_runtime_id: str,
    sequence_length: int,
    local_steps: int,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation: int,
    seed: int,
    lora_rank: int = 4,
    lora_alpha: int = 8,
    dataset_format: str = "deterministic_instruction_sft_jsonl",
    response_only_supervision: bool = True,
    work_shard_count: int = 2,
) -> dict[str, Any]:
    import math

    import torch
    from accelerate import init_empty_weights
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM

    shard_count = int(work_shard_count)
    if len(train_rows) < 4 or len(validation_rows) < 2:
        raise ValueError("volunteer Campaign needs training and validation sequences")
    if shard_count < 2 or shard_count > min(64, len(train_rows)):
        raise ValueError("volunteer_campaign_work_shard_count_out_of_bounds")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset_path = root / "private_dataset.jsonl"
    validation_path = root / "private_validation_dataset.jsonl"
    dataset_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
            for row in train_rows
        ),
        encoding="utf-8",
    )
    dataset_path.chmod(0o600)
    validation_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
            for row in validation_rows
        ),
        encoding="utf-8",
    )
    validation_path.chmod(0o600)

    configure_cpu_determinism(seed)
    config_object = AutoConfig.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    config = config_object.to_dict()
    model_adapter = get_model_adapter(model_adapter_id)
    if not model_adapter.supports(
        model_id=str(model_source["model_id"]), config=config
    ):
        raise RuntimeError("volunteer_campaign_model_architecture_unsupported")
    model_adapter.validate_config(config)
    with init_empty_weights():
        base = AutoModelForCausalLM.from_config(config_object)
        adapter = model_adapter.apply_lora(
            base, rank=int(lora_rank), alpha=int(lora_alpha), dropout=0.0
        )
    parameter_count = sum(int(value.numel()) for value in base.parameters())
    initial_adapter_dir = root / "initial_adapter"
    initial_adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_config = adapter.peft_config["default"]
    peft_config.base_model_name_or_path = str(model_source["model_id"])
    if hasattr(peft_config, "revision"):
        peft_config.revision = str(model_source["revision"])
    peft_config.save_pretrained(initial_adapter_dir)
    adapter_tensor_path = initial_adapter_dir / "adapter_model.safetensors"
    adapter_config_path = initial_adapter_dir / "adapter_config.json"
    torch.manual_seed(int(seed))
    materialized: dict[str, Any] = {}
    for name, value in sorted(
        get_peft_model_state_dict(adapter, save_embedding_layers=False).items()
    ):
        tensor = torch.empty(tuple(value.shape), dtype=value.dtype, device="cpu")
        if name.endswith("lora_A.weight"):
            torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
        elif name.endswith("lora_B.weight"):
            torch.nn.init.zeros_(tensor)
        else:
            raise RuntimeError("volunteer_campaign_initial_adapter_tensor_unsupported")
        materialized[str(name)] = tensor.contiguous()
    if not materialized:
        raise RuntimeError("volunteer_campaign_initial_adapter_empty")
    save_file(materialized, adapter_tensor_path)
    initial_tensors = load_tensors(adapter_tensor_path)
    adapter_specs = tensor_specs(initial_tensors)
    architecture = str(
        (config.get("architectures") or ["AutoModelForCausalLM"])[0]
    )
    del adapter, base

    shard_indexes = [
        list(range(shard_index, len(train_rows), shard_count))
        for shard_index in range(shard_count)
    ]
    shard_manifests: list[dict[str, Any]] = []
    for shard_index, indexes in enumerate(shard_indexes):
        public = {
            "schema": DATASET_SCHEMA,
            "dataset_id": dataset_runtime_id,
            "dataset_version": 1,
            "shard_index": shard_index,
            "sample_indexes": indexes,
            "sample_count": len(indexes),
            "token_count": sum(
                sum(int(item) for item in train_rows[index]["attention_mask"])
                for index in indexes
            ),
            "data_cursor_start": 0,
            "raw_text_public": False,
            "token_ids_public": False,
        }
        public["shard_hash"] = sha256_json(public)
        shard_manifests.append(public)

    model_manifest = {
        "schema": MODEL_SCHEMA,
        "model_id": model_source["model_id"],
        "model_version": 1,
        "architecture": architecture,
        "parameter_count": parameter_count,
        "dtype": str(config.get("torch_dtype") or "source_declared"),
        "base_model_hash": model_source["imported_snapshot_hash"],
        "config_hash": sha256_json(config),
        "source": "pinned_public_model_snapshot",
        "source_revision": model_source["revision"],
        "source_license": model_source["license"],
        "source_snapshot_hash": model_source["imported_snapshot_hash"],
        "model_adapter_id": model_adapter_id,
        "source_provenance": model_source,
        "base_model_path": str(Path(model_dir).resolve()),
    }
    model_manifest["manifest_hash"] = sha256_json(
        public_training_spec(model_manifest)
    )
    dataset_manifest = {
        "schema": DATASET_SCHEMA,
        "dataset_id": dataset_runtime_id,
        "dataset_version": 1,
        "format": str(dataset_format),
        "sample_count": len(train_rows),
        "token_count": sum(
            sum(int(item) for item in row["attention_mask"]) for row in train_rows
        ),
        "supervised_token_count": sum(
            sum(int(item != -100) for item in row["labels"]) for row in train_rows
        ),
        "sequence_length": int(sequence_length),
        "shard_count": shard_count,
        "shards": shard_manifests,
        "dataset_file_hash": sha256_file(dataset_path),
        "validation_file_hash": sha256_file(validation_path),
        "validation_sample_count": len(validation_rows),
        "train_source_ids_hash": sha256_json(train_source_ids),
        "validation_source_ids_hash": sha256_json(validation_source_ids),
        "train_token_hash": sha256_json(train_rows),
        "validation_token_hash": sha256_json(validation_rows),
        "source_provenance": dataset_source,
        "raw_text_public": False,
        "token_ids_public": False,
        "private_dataset_path": str(dataset_path),
        "private_validation_dataset_path": str(validation_path),
    }
    dataset_manifest["manifest_hash"] = sha256_json(
        public_training_spec(dataset_manifest)
    )
    lora_manifest = {
        "schema": LORA_SCHEMA,
        "adapter_version": 0,
        "adapter_id": model_adapter_id,
        "rank": int(lora_rank),
        "alpha": int(lora_alpha),
        "dropout": 0.0,
        "target_modules": sorted(model_adapter.target_modules),
        "trainable_parameter_count": sum(
            int(item["numel"]) for item in adapter_specs
        ),
        "tensor_specs": adapter_specs,
        "base_adapter_hash": sha256_json(adapter_specs),
        "adapter_path": str(initial_adapter_dir),
        "adapter_tensor_path": str(adapter_tensor_path),
        "adapter_config_path": str(adapter_config_path),
    }
    lora_manifest["manifest_hash"] = sha256_json(
        public_training_spec(lora_manifest)
    )
    import_manifest = {
        "schema": IMPORT_SCHEMA,
        "profile": import_profile,
        "model_adapter_id": model_adapter_id,
        "model_source": model_source,
        "dataset_source": dataset_source,
        "deterministic_tokenization": True,
        "response_only_supervision": bool(response_only_supervision),
        "raw_text_public": False,
        "token_ids_public": False,
        "private_paths_public": False,
        "source_verified": True,
    }
    import_manifest["content_hash"] = sha256_json(import_manifest)
    job = {
        "schema": JOB_SCHEMA,
        "job_id": job_id,
        "job_version": 1,
        "workload_type": WORKLOAD_TYPE,
        "permission_mode": "invite_authenticated_trusted_cells_beta",
        "backend": "pytorch_transformers_peft_cpu",
        "seed": int(seed),
        "model": model_manifest,
        "dataset": dataset_manifest,
        "lora": lora_manifest,
        "campaign_import": import_manifest,
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
        "token_ids_public": False,
        "gpu_live_verified": False,
    }
    job["job_hash"] = sha256_json(public_training_spec(job))
    private_path = _write_json(root / "training_job_private.json", job)
    public_path = _write_json(
        root / "training_job_public.json", public_training_spec(job), mode=0o644
    )
    return {
        **job,
        "job_manifest_path": str(private_path),
        "public_job_manifest_path": str(public_path),
    }


def build_smollm_wikitext_fixture(
    output_dir: str | Path,
    assets: CampaignAssetBundle,
    *,
    job_id: str,
    sequence_length: int = 16,
    train_sequence_count: int = 12,
    validation_sequence_count: int = 4,
    local_steps: int = 1,
    learning_rate: float = 2e-4,
    batch_size: int = 1,
    gradient_accumulation: int = 1,
    seed: int = 20260718,
) -> dict[str, Any]:
    """Materialize a fixture accepted by the durable volunteer Coordinator."""

    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer

    if not 8 <= int(sequence_length) <= 256:
        raise ValueError("volunteer Campaign sequence length must be in [8, 256]")
    if int(train_sequence_count) < 4 or int(validation_sequence_count) < 2:
        raise ValueError("volunteer Campaign needs training and validation sequences")
    if int(local_steps) < 1 or int(local_steps) > 64:
        raise ValueError("volunteer Campaign local steps must be in [1, 64]")
    _validate_asset_bundle(assets)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    model_dir = root / "base_model"
    _copy_snapshot(assets.model_dir, model_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    train_texts = parquet.read_table(assets.train_parquet, columns=["text"])[
        "text"
    ].to_pylist()
    validation_texts = parquet.read_table(
        assets.validation_parquet, columns=["text"]
    )["text"].to_pylist()
    train_rows, train_indexes = tokenize_fixed_sequences(
        train_texts,
        tokenizer,
        sequence_length=int(sequence_length),
        sequence_count=int(train_sequence_count),
    )
    validation_rows, validation_indexes = tokenize_fixed_sequences(
        validation_texts,
        tokenizer,
        sequence_length=int(sequence_length),
        sequence_count=int(validation_sequence_count),
    )
    tokenized_train = [
        {
            "sample_id": f"train-{index:06d}",
            "input_ids": tokens,
            "labels": list(tokens),
            "attention_mask": [1] * len(tokens),
        }
        for index, tokens in enumerate(train_rows)
    ]
    tokenized_validation = [
        {
            "sample_id": f"validation-{index:06d}",
            "input_ids": tokens,
            "labels": list(tokens),
            "attention_mask": [1] * len(tokens),
        }
        for index, tokens in enumerate(validation_rows)
    ]
    return _materialize_tokenized_peft_fixture(
        root,
        model_dir=model_dir,
        model_source=assets.model_source,
        dataset_source=assets.dataset_source,
        train_rows=tokenized_train,
        validation_rows=tokenized_validation,
        train_source_ids=[str(item) for item in train_indexes],
        validation_source_ids=[str(item) for item in validation_indexes],
        job_id=job_id,
        import_profile=IMPORT_PROFILE,
        model_adapter_id=MODEL_ADAPTER_ID,
        dataset_runtime_id=f"{job_id}-wikitext2",
        sequence_length=int(sequence_length),
        local_steps=int(local_steps),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        gradient_accumulation=int(gradient_accumulation),
        seed=int(seed),
        lora_rank=4,
        lora_alpha=8,
        dataset_format="deterministic_tokenized_jsonl",
        response_only_supervision=False,
    )


def build_commons_instruction_fixture(
    output_dir: str | Path,
    *,
    model_dir: str | Path,
    model_id: str,
    model_revision: str,
    model_license: str,
    model_adapter_id: str,
    train_data_packs: tuple[str | Path, ...],
    evaluation_data_pack: str | Path,
    job_id: str,
    model_source_attested: bool,
    sequence_length: int = 512,
    train_sequence_count: int = 0,
    validation_sequence_count: int = 0,
    local_steps: int = 1,
    learning_rate: float = 2e-4,
    batch_size: int = 1,
    gradient_accumulation: int = 1,
    seed: int = 20260816,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    work_shard_count: int = 4,
) -> dict[str, Any]:
    """Build a review-gated Commons instruction Campaign from immutable inputs."""

    from transformers import AutoTokenizer

    length = int(sequence_length)
    if length < 32 or length > 4096:
        raise ValueError("commons_campaign_sequence_length_out_of_bounds")
    if int(local_steps) < 1 or int(local_steps) > 64:
        raise ValueError("commons_campaign_local_steps_out_of_bounds")
    if int(lora_rank) < 1 or int(lora_rank) > 256:
        raise ValueError("commons_campaign_lora_rank_out_of_bounds")
    if int(lora_alpha) < 1 or int(lora_alpha) > 1024:
        raise ValueError("commons_campaign_lora_alpha_out_of_bounds")
    if model_source_attested is not True:
        raise ValueError("commons_campaign_model_source_attestation_required")
    revision = str(model_revision).strip().lower()
    if not _IMMUTABLE_REVISION.fullmatch(revision):
        raise ValueError("commons_campaign_immutable_model_revision_required")
    selected_model_id = str(model_id).strip()
    selected_license = str(model_license).strip().lower()
    if not selected_model_id or not selected_license:
        raise ValueError("commons_campaign_model_identity_required")

    local_model = Path(model_dir).expanduser().resolve()
    if not local_model.is_dir():
        raise FileNotFoundError("commons_campaign_model_snapshot_missing")
    imported_files = _file_records(local_model)
    imported_names = {str(item["relative_name"]) for item in imported_files}
    has_weights = any(
        name.endswith(".safetensors") and not name.startswith("adapter_")
        for name in imported_names
    )
    if not {"config.json", "tokenizer.json"}.issubset(imported_names) or not has_weights:
        raise RuntimeError("commons_campaign_model_runtime_files_missing")
    try:
        config = json.loads((local_model / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("commons_campaign_model_config_unreadable") from exc
    model_adapter = get_model_adapter(model_adapter_id)
    if not model_adapter.supports(model_id=selected_model_id, config=config):
        raise RuntimeError("commons_campaign_model_architecture_unsupported")
    model_adapter.validate_config(config)
    model_source = {
        "schema": MODEL_SOURCE_SCHEMA,
        "model_id": selected_model_id,
        "revision": revision,
        "license": selected_license,
        "architecture": str(
            (config.get("architectures") or ["AutoModelForCausalLM"])[0]
        ),
        "adapter_id": model_adapter_id,
        "source_public": True,
        "immutable_revision": True,
        "gated": False,
        "private": False,
        "imported_files": imported_files,
        "imported_file_count": len(imported_files),
        "imported_snapshot_hash": sha256_json(imported_files),
        "local_snapshot_verified": True,
        "remote_identity_operator_attested": True,
        "source_verified": True,
    }
    model_source["runtime_fetch"] = {
        "schema": "crowdtensor_huggingface_snapshot_fetch_v1",
        "provider": "huggingface_hub",
        "repo_id": selected_model_id,
        "revision": revision,
        "allow_patterns": sorted(imported_names),
        "file_manifest_hash": model_source["imported_snapshot_hash"],
        "trust_remote_code": False,
    }

    train_paths = tuple(Path(item).expanduser().resolve() for item in train_data_packs)
    evaluation_path = Path(evaluation_data_pack).expanduser().resolve()
    if not train_paths:
        raise ValueError("commons_campaign_training_data_pack_required")
    if len(set(train_paths)) != len(train_paths) or evaluation_path in train_paths:
        raise ValueError("commons_campaign_data_pack_role_conflict")

    pack_entries: list[dict[str, Any]] = []
    train_records: list[dict[str, str]] = []
    validation_records: list[dict[str, str]] = []
    train_source_ids: list[str] = []
    validation_source_ids: list[str] = []
    source_files: list[dict[str, Any]] = []
    seen_pack_ids: set[str] = set()
    seen_train_content: set[str] = set()
    evaluation_content: set[str] = set()

    for role, paths in (("train", train_paths), ("evaluation", (evaluation_path,))):
        for pack_path in paths:
            report = validate_instruction_data_pack(pack_path)
            if report.get("ok") is not True:
                raise ValueError("commons_campaign_data_pack_integrity_failed")
            pack = load_data_pack(pack_path)
            if not pack.admission_ready:
                raise ValueError("commons_campaign_data_pack_not_admission_ready")
            if not pack.public_records:
                raise ValueError("commons_campaign_data_pack_records_not_public")
            if pack.pack_id in seen_pack_ids:
                raise ValueError("commons_campaign_data_pack_id_duplicate")
            seen_pack_ids.add(pack.pack_id)
            records = load_instruction_records(pack_path)
            public_manifest = pack.to_dict()
            pack_entries.append({"role": role, "manifest": public_manifest})
            root = pack_path.parent if pack_path.name == DATA_PACK_MANIFEST_FILE else pack_path
            for name in (DATA_PACK_MANIFEST_FILE, DATA_PACK_RECORDS_FILE):
                source = root / name
                source_files.append(
                    {
                        "split": role,
                        "pack_id": pack.pack_id,
                        "relative_name": f"data-packs/{pack.pack_id}/{name}",
                        "sha256": sha256_file(source),
                        "byte_count": int(source.stat().st_size),
                    }
                )
            destination = train_records if role == "train" else validation_records
            source_ids = train_source_ids if role == "train" else validation_source_ids
            for record in records:
                content_hash = sha256_json(
                    {
                        "prompt": record["prompt"],
                        "response": record["response"],
                    }
                )
                if role == "train":
                    if content_hash in seen_train_content:
                        raise ValueError("commons_campaign_training_record_duplicate")
                    seen_train_content.add(content_hash)
                else:
                    if content_hash in evaluation_content:
                        raise ValueError("commons_campaign_evaluation_record_duplicate")
                    evaluation_content.add(content_hash)
                source_id = f"{pack.pack_id}:{record['record_id']}"
                canonical = dict(record)
                canonical["record_id"] = "sample-" + sha256_json(
                    {"source_id": source_id, "pack_hash": pack.content_hash}
                ).split(":", 1)[1][:24]
                destination.append(canonical)
                source_ids.append(source_id)

    if seen_train_content.intersection(evaluation_content):
        raise ValueError("commons_campaign_benchmark_contamination_detected")
    if len(train_records) < 4 or len(validation_records) < 2:
        raise ValueError("commons_campaign_data_pack_records_insufficient")

    tokenizer = AutoTokenizer.from_pretrained(
        local_model, local_files_only=True, trust_remote_code=False
    )
    train_rows, selected_train_ids = tokenize_instruction_records(
        train_records,
        tokenizer,
        sequence_length=length,
        sequence_count=int(train_sequence_count),
    )
    validation_rows, selected_validation_ids = tokenize_instruction_records(
        validation_records,
        tokenizer,
        sequence_length=length,
        sequence_count=int(validation_sequence_count),
    )
    selected_train = {
        record["record_id"]: source_id
        for record, source_id in zip(train_records, train_source_ids)
    }
    selected_validation = {
        record["record_id"]: source_id
        for record, source_id in zip(validation_records, validation_source_ids)
    }
    dataset_snapshot_hash = sha256_json(source_files)
    dataset_source = {
        "schema": DATASET_SOURCE_SCHEMA,
        "dataset_id": f"crowdtensor/{job_id}-commons",
        "revision": dataset_snapshot_hash.split(":", 1)[1][:40],
        "revision_kind": "content_hash_prefix",
        "config": "commons-instruction-sft-v1",
        "licenses": sorted(
            {entry["manifest"]["license_spdx"] for entry in pack_entries}
        ),
        "source_public": True,
        "immutable_revision": True,
        "gated": False,
        "private": False,
        "source_files": source_files,
        "source_snapshot_hash": dataset_snapshot_hash,
        "source_verified": True,
        "data_packs": pack_entries,
        "training_data_pack_count": len(train_paths),
        "evaluation_data_pack_id": load_data_pack(evaluation_path).pack_id,
        "public_training_record_count": len(train_records),
        "public_evaluation_record_count": len(validation_records),
        "all_data_packs_admission_ready": True,
        "all_records_redistributable": True,
        "benchmark_overlap_detected": False,
        "raw_text_public": False,
        "token_ids_public": False,
    }
    return _materialize_tokenized_peft_fixture(
        output_dir,
        model_dir=local_model,
        model_source=model_source,
        dataset_source=dataset_source,
        train_rows=train_rows,
        validation_rows=validation_rows,
        train_source_ids=[selected_train[item] for item in selected_train_ids],
        validation_source_ids=[
            selected_validation[item] for item in selected_validation_ids
        ],
        job_id=job_id,
        import_profile=COMMONS_IMPORT_PROFILE,
        model_adapter_id=model_adapter_id,
        dataset_runtime_id=f"{job_id}-commons",
        sequence_length=length,
        local_steps=int(local_steps),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        gradient_accumulation=int(gradient_accumulation),
        seed=int(seed),
        lora_rank=int(lora_rank),
        lora_alpha=int(lora_alpha),
        dataset_format="deterministic_instruction_sft_jsonl",
        response_only_supervision=True,
        work_shard_count=int(work_shard_count),
    )


def create_pinned_smollm_wikitext_fixture(
    output_dir: str | Path,
    *,
    job_id: str,
    sequence_length: int = 16,
    train_sequence_count: int = 12,
    validation_sequence_count: int = 4,
    local_steps: int = 1,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    assets_root = root / ".import-assets"
    try:
        assets = download_pinned_campaign_assets(assets_root)
        return build_smollm_wikitext_fixture(
            root,
            assets,
            job_id=job_id,
            sequence_length=sequence_length,
            train_sequence_count=train_sequence_count,
            validation_sequence_count=validation_sequence_count,
            local_steps=local_steps,
        )
    finally:
        shutil.rmtree(assets_root, ignore_errors=True)
