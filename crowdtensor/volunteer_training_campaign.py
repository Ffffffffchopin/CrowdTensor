"""Pinned public-source Campaign import for volunteer PEFT training."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hf_lora_training import configure_cpu_determinism
from .model_adapter import SmolLMModelAdapter, get_model_adapter
from .named_tensor_optimizer import load_tensors
from .qwen15b_training import _tokenize_split
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

MODEL_ID = SmolLMModelAdapter.default_model_id
MODEL_REVISION = SmolLMModelAdapter.default_revision
MODEL_LICENSE = SmolLMModelAdapter.default_model_license
MODEL_ADAPTER_ID = SmolLMModelAdapter.adapter_id
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
    from transformers import AutoTokenizer, LlamaForCausalLM

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
    train_rows, train_indexes = _tokenize_split(
        train_texts,
        tokenizer,
        sequence_length=int(sequence_length),
        sequence_count=int(train_sequence_count),
    )
    validation_rows, validation_indexes = _tokenize_split(
        validation_texts,
        tokenizer,
        sequence_length=int(sequence_length),
        sequence_count=int(validation_sequence_count),
    )
    dataset_path = root / "private_dataset.jsonl"
    dataset_path.write_text(
        "".join(
            json.dumps(
                {"sample_id": f"train-{index:06d}", "input_ids": tokens},
                sort_keys=True,
            )
            + "\n"
            for index, tokens in enumerate(train_rows)
        ),
        encoding="utf-8",
    )
    dataset_path.chmod(0o600)
    validation_path = root / "private_validation_dataset.jsonl"
    validation_path.write_text(
        "".join(
            json.dumps(
                {"sample_id": f"validation-{index:06d}", "input_ids": tokens},
                sort_keys=True,
            )
            + "\n"
            for index, tokens in enumerate(validation_rows)
        ),
        encoding="utf-8",
    )
    validation_path.chmod(0o600)

    configure_cpu_determinism(seed)
    base = LlamaForCausalLM.from_pretrained(model_dir, local_files_only=True)
    config = base.config.to_dict()
    if (
        str(config.get("model_type") or "") != "llama"
        or "LlamaForCausalLM" not in set(config.get("architectures") or [])
    ):
        raise RuntimeError("volunteer_campaign_model_architecture_unsupported")
    parameter_count = sum(int(value.numel()) for value in base.parameters())
    adapter = get_model_adapter(MODEL_ADAPTER_ID).apply_lora(
        base, rank=4, alpha=8, dropout=0.0
    )
    initial_adapter_dir = root / "initial_adapter"
    adapter.save_pretrained(initial_adapter_dir, safe_serialization=True)
    adapter_tensor_path = initial_adapter_dir / "adapter_model.safetensors"
    adapter_config_path = initial_adapter_dir / "adapter_config.json"
    initial_tensors = load_tensors(adapter_tensor_path)
    adapter_specs = tensor_specs(initial_tensors)
    del adapter, base

    shard_indexes = [
        list(range(0, len(train_rows), 2)),
        list(range(1, len(train_rows), 2)),
    ]
    shard_manifests: list[dict[str, Any]] = []
    for shard_index, indexes in enumerate(shard_indexes):
        public = {
            "schema": DATASET_SCHEMA,
            "dataset_id": f"{job_id}-wikitext2",
            "dataset_version": 1,
            "shard_index": shard_index,
            "sample_indexes": indexes,
            "sample_count": len(indexes),
            "token_count": sum(len(train_rows[index]) for index in indexes),
            "data_cursor_start": 0,
            "raw_text_public": False,
            "token_ids_public": False,
        }
        public["shard_hash"] = sha256_json(public)
        shard_manifests.append(public)

    model_manifest = {
        "schema": MODEL_SCHEMA,
        "model_id": MODEL_ID,
        "model_version": 1,
        "architecture": "LlamaForCausalLM",
        "parameter_count": parameter_count,
        "dtype": str(config.get("torch_dtype") or "source_declared"),
        "base_model_hash": assets.model_source["imported_snapshot_hash"],
        "config_hash": sha256_json(config),
        "source": "pinned_huggingface_public_snapshot",
        "source_revision": MODEL_REVISION,
        "source_license": MODEL_LICENSE,
        "source_snapshot_hash": assets.model_source["imported_snapshot_hash"],
        "model_adapter_id": MODEL_ADAPTER_ID,
        "source_provenance": assets.model_source,
        "base_model_path": str(model_dir),
    }
    model_manifest["manifest_hash"] = sha256_json(public_training_spec(model_manifest))
    dataset_manifest = {
        "schema": DATASET_SCHEMA,
        "dataset_id": f"{job_id}-wikitext2",
        "dataset_version": 1,
        "format": "deterministic_tokenized_jsonl",
        "sample_count": len(train_rows),
        "token_count": sum(len(row) for row in train_rows),
        "sequence_length": int(sequence_length),
        "shard_count": 2,
        "shards": shard_manifests,
        "dataset_file_hash": sha256_file(dataset_path),
        "validation_file_hash": sha256_file(validation_path),
        "validation_sample_count": len(validation_rows),
        "train_source_row_indexes_hash": sha256_json(train_indexes),
        "validation_source_row_indexes_hash": sha256_json(validation_indexes),
        "train_token_hash": sha256_json(train_rows),
        "validation_token_hash": sha256_json(validation_rows),
        "source_provenance": assets.dataset_source,
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
        "adapter_id": MODEL_ADAPTER_ID,
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules": sorted(SmolLMModelAdapter.target_modules),
        "trainable_parameter_count": sum(int(item["numel"]) for item in adapter_specs),
        "tensor_specs": adapter_specs,
        "base_adapter_hash": sha256_json(adapter_specs),
        "adapter_path": str(initial_adapter_dir),
        "adapter_tensor_path": str(adapter_tensor_path),
        "adapter_config_path": str(adapter_config_path),
    }
    lora_manifest["manifest_hash"] = sha256_json(public_training_spec(lora_manifest))
    import_manifest = {
        "schema": IMPORT_SCHEMA,
        "profile": IMPORT_PROFILE,
        "model_adapter_id": MODEL_ADAPTER_ID,
        "model_source": assets.model_source,
        "dataset_source": assets.dataset_source,
        "deterministic_tokenization": True,
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
