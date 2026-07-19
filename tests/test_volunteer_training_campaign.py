from __future__ import annotations

import shutil

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from crowdtensor.hf_lora_training import create_local_training_fixture
from crowdtensor.training_contract import sha256_file, sha256_json
from crowdtensor.volunteer_training_campaign import (
    CampaignAssetBundle,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_LICENSES,
    DATASET_REVISION,
    DATASET_SOURCE_SCHEMA,
    IMPORT_PROFILE,
    MODEL_ADAPTER_ID,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    MODEL_SOURCE_SCHEMA,
    build_smollm_wikitext_fixture,
)
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator
from crowdtensor.volunteer_training_protocol import validate_campaign_manifest


def _records(root):
    return [
        {
            "relative_name": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _mock_assets(tmp_path) -> CampaignAssetBundle:
    local = create_local_training_fixture(tmp_path / "tiny-source", row_count=8)
    model_dir = tmp_path / "mock-model"
    shutil.copytree(local["model"]["base_model_path"], model_dir)
    vocabulary = {"<unk>": 0, "<eos>": 1}
    vocabulary.update({f"word{index}": index + 2 for index in range(62)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<eos>",
    ).save_pretrained(model_dir)
    model_files = _records(model_dir)

    texts = [" ".join(f"word{index % 50}" for index in range(40))] * 20
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    parquet.write_table(pa.table({"text": texts}), train)
    parquet.write_table(pa.table({"text": texts[:8]}), validation)
    dataset_files = [
        {
            "split": split,
            "relative_name": f"{DATASET_CONFIG}/{split}.parquet",
            "sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
        }
        for split, path in (("train", train), ("validation", validation))
    ]
    return CampaignAssetBundle(
        model_dir=model_dir,
        train_parquet=train,
        validation_parquet=validation,
        model_source={
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
        },
        dataset_source={
            "schema": DATASET_SOURCE_SCHEMA,
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "config": DATASET_CONFIG,
            "licenses": sorted(DATASET_LICENSES),
            "source_public": True,
            "immutable_revision": True,
            "gated": False,
            "private": False,
            "source_files": dataset_files,
            "source_snapshot_hash": sha256_json(dataset_files),
            "source_verified": True,
            "raw_text_public": False,
            "token_ids_public": False,
        },
    )


def test_pinned_campaign_import_binds_sources_adapter_and_tokens(tmp_path) -> None:
    fixture = build_smollm_wikitext_fixture(
        tmp_path / "fixture",
        _mock_assets(tmp_path),
        job_id="import-test",
        sequence_length=8,
        train_sequence_count=4,
        validation_sequence_count=2,
        local_steps=1,
    )
    coordinator = VolunteerTrainingCoordinator.create_from_fixture(
        tmp_path / "campaign", fixture, target_rounds=3
    )
    manifest = validate_campaign_manifest(coordinator.campaign_manifest())
    assert manifest["model_adapter_id"] == MODEL_ADAPTER_ID
    assert manifest["campaign_import"]["profile"] == IMPORT_PROFILE
    assert manifest["model_source"]["revision"] == MODEL_REVISION
    assert manifest["dataset_source"]["revision"] == DATASET_REVISION
    assert manifest["round_policy"]["target_rounds"] == 3
    serialized = coordinator.campaign_path.read_text(encoding="utf-8")
    assert "input_ids" not in serialized
    assert str(tmp_path) not in serialized


def test_campaign_import_rejects_changed_source_file(tmp_path) -> None:
    assets = _mock_assets(tmp_path)
    (assets.model_dir / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="snapshot_changed"):
        build_smollm_wikitext_fixture(
            tmp_path / "fixture-invalid",
            assets,
            job_id="invalid",
            sequence_length=8,
            train_sequence_count=4,
            validation_sequence_count=2,
        )
