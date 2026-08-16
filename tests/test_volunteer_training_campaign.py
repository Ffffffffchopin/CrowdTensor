from __future__ import annotations

import json
import shutil

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from crowdtensor.adapters.text_data import create_instruction_data_pack
from crowdtensor.hf_lora_training import (
    CPULoRATrainingRuntime,
    create_local_training_fixture,
    training_spec_for_claim,
)
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
    COMMONS_IMPORT_PROFILE,
    build_commons_instruction_fixture,
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


def _instruction_pack(tmp_path, pack_id, records):
    source = tmp_path / f"{pack_id}.jsonl"
    source.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    destination = tmp_path / f"{pack_id}-pack"
    create_instruction_data_pack(
        source,
        destination,
        pack_id=pack_id,
        license_spdx="CC-BY-4.0",
        source_kind="contributor_authored",
        languages=("en",),
        domains=("reasoning",),
        contributor_id=f"contributor-{pack_id}",
        redistribution_allowed=True,
        training_allowed=True,
        personal_data_reviewed=True,
        copyright_reviewed=True,
        benchmark_contamination_reviewed=True,
        moderation_status="approved",
        public_records=True,
    )
    return destination


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


def test_commons_campaign_binds_reviewed_data_packs_and_response_labels(
    tmp_path,
) -> None:
    assets = _mock_assets(tmp_path)
    train_pack = _instruction_pack(
        tmp_path,
        "commons-train",
        [
            {
                "record_id": f"train-{index}",
                "prompt": f"word{index} word{index + 1}",
                "response": f"word{index + 2} word{index + 3}",
                "language": "en",
            }
            for index in range(6)
        ],
    )
    evaluation_pack = _instruction_pack(
        tmp_path,
        "commons-eval",
        [
            {
                "record_id": f"eval-{index}",
                "prompt": f"word{index + 20} word{index + 21}",
                "response": f"word{index + 22} word{index + 23}",
                "language": "en",
            }
            for index in range(3)
        ],
    )
    fixture = build_commons_instruction_fixture(
        tmp_path / "commons-fixture",
        model_dir=assets.model_dir,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        model_license=MODEL_LICENSE,
        model_adapter_id=MODEL_ADAPTER_ID,
        train_data_packs=(train_pack,),
        evaluation_data_pack=evaluation_pack,
        job_id="commons-test",
        model_source_attested=True,
        sequence_length=32,
        train_sequence_count=4,
        validation_sequence_count=2,
    )

    rows = [
        json.loads(line)
        for line in open(
            fixture["dataset"]["private_dataset_path"], encoding="utf-8"
        )
        if line.strip()
    ]
    assert all(len(row["input_ids"]) == 32 for row in rows)
    assert all(-100 in row["labels"] for row in rows)
    assert all(any(value != -100 for value in row["labels"]) for row in rows)
    assert fixture["campaign_import"]["profile"] == COMMONS_IMPORT_PROFILE
    assert fixture["campaign_import"]["response_only_supervision"] is True
    assert fixture["model"]["source_provenance"]["runtime_fetch"] == {
        "schema": "crowdtensor_huggingface_snapshot_fetch_v1",
        "provider": "huggingface_hub",
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "allow_patterns": sorted(
            item["relative_name"]
            for item in fixture["model"]["source_provenance"]["imported_files"]
        ),
        "file_manifest_hash": fixture["model"]["source_provenance"][
            "imported_snapshot_hash"
        ],
        "trust_remote_code": False,
    }
    assert fixture["dataset"]["shard_count"] == 4
    assert fixture["dataset"]["source_provenance"][
        "all_data_packs_admission_ready"
    ] is True

    spec = training_spec_for_claim(
        fixture,
        task_id="commons-cpu-task",
        miner_id="commons-cpu-cell",
        shard_index=0,
    )
    result = CPULoRATrainingRuntime().run(spec, output_dir=tmp_path / "commons-cell")
    assert result["real_backward"] is True
    assert result["only_lora_trainable"] is True
    assert result["adapter_delta"]["tensor_count"] > 0

    coordinator = VolunteerTrainingCoordinator.create_from_fixture(
        tmp_path / "commons-campaign", fixture, target_rounds=2
    )
    manifest = validate_campaign_manifest(coordinator.campaign_manifest())
    assert manifest["campaign_import"]["profile"] == COMMONS_IMPORT_PROFILE
    assert manifest["evaluation_contract"]["heldout_sample_count"] == 2
    serialized = json.dumps(manifest, sort_keys=True)
    assert "word0" not in serialized
    assert str(tmp_path) not in serialized
    evaluation = coordinator.evaluate_campaign(heldout_quality=True)
    snapshot = coordinator.public_campaign_snapshot()
    assert evaluation["held_out_quality_benchmark_performed"] is True
    assert evaluation["quality"]["evaluation_device"] == "cpu"
    assert snapshot["data"]["data_pack_count"] == 2
    assert snapshot["data"]["training_data_pack_count"] == 1
    assert snapshot["data"]["all_data_packs_admission_ready"] is True
    assert snapshot["checkpoint_lineage"]["ok"] is True
    assert snapshot["evaluation"]["current_evaluation_available"] is True
    assert snapshot["evaluation"]["held_out_quality_benchmark_performed"] is True
    stale = json.loads(
        (coordinator.root / "evaluation.json").read_text(encoding="utf-8")
    )
    stale["adapter_version"] = 999
    stale["content_hash"] = sha256_json(
        {key: value for key, value in stale.items() if key != "content_hash"}
    )
    (coordinator.root / "evaluation.json").write_text(
        json.dumps(stale, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    stale_snapshot = coordinator.public_campaign_snapshot()
    assert stale_snapshot["evaluation"]["current_evaluation_available"] is False
    assert stale_snapshot["evaluation"]["held_out_quality_benchmark_performed"] is False


def test_commons_campaign_rejects_train_evaluation_overlap(tmp_path) -> None:
    assets = _mock_assets(tmp_path)
    shared = {
        "prompt": "word1 word2",
        "response": "word3 word4",
        "language": "en",
    }
    train_pack = _instruction_pack(
        tmp_path,
        "overlap-train",
        [
            {"record_id": f"train-{index}", **(shared if index == 0 else {
                "prompt": f"word{index + 5}",
                "response": f"word{index + 15}",
                "language": "en",
            })}
            for index in range(4)
        ],
    )
    evaluation_pack = _instruction_pack(
        tmp_path,
        "overlap-eval",
        [
            {"record_id": "eval-shared", **shared},
            {
                "record_id": "eval-other",
                "prompt": "word40",
                "response": "word41",
                "language": "en",
            },
        ],
    )
    with pytest.raises(ValueError, match="benchmark_contamination_detected"):
        build_commons_instruction_fixture(
            tmp_path / "overlap-fixture",
            model_dir=assets.model_dir,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            model_license=MODEL_LICENSE,
            model_adapter_id=MODEL_ADAPTER_ID,
            train_data_packs=(train_pack,),
            evaluation_data_pack=evaluation_pack,
            job_id="overlap-test",
            model_source_attested=True,
            sequence_length=32,
        )
