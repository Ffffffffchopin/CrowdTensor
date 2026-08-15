from copy import deepcopy
import json
from pathlib import Path

import jsonschema
import pytest

from crowdtensor.adapters.manifests import (
    MANIFEST_SCHEMA,
    TPU_MANIFEST_SCHEMA,
    QWEN25_7B_MODEL_ID,
    QWEN25_7B_MODEL_REVISION,
    QWEN25_7B_WEIGHT_BYTES,
    TrainingManifestError,
    qwen25_7b_lora_manifest,
    qwen25_7b_lora_tpu_manifest,
    validate_training_manifest,
)


def test_qwen25_7b_manifest_is_pinned_complete_and_public_safe() -> None:
    manifest = qwen25_7b_lora_manifest()

    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["model"]["model_id"] == QWEN25_7B_MODEL_ID
    assert manifest["model"]["model_revision"] == QWEN25_7B_MODEL_REVISION
    assert sum(stage["estimated_weight_bytes"] for stage in manifest["stages"]) == QWEN25_7B_WEIGHT_BYTES
    assert [stage["layer_start"] for stage in manifest["stages"]] == [0, 7, 14, 20, 26]
    assert [stage["layer_end"] for stage in manifest["stages"]] == [7, 14, 20, 26, 28]
    assert manifest["stages"][-1]["allowed_device_types"] == ["cpu"]
    assert manifest["scheduler"]["required_device_types"] == ["cpu", "cuda"]
    assert manifest["training"]["target_steps"] == 6
    assert manifest["public_artifact_safe"] is True
    assert manifest["tensor_values_public"] is False


def test_qwen25_7b_tpu_manifest_preserves_topology_and_requires_all_backends() -> None:
    manifest = qwen25_7b_lora_tpu_manifest()

    assert manifest["schema"] == TPU_MANIFEST_SCHEMA
    assert manifest["model"]["model_id"] == QWEN25_7B_MODEL_ID
    assert manifest["model"]["model_revision"] == QWEN25_7B_MODEL_REVISION
    assert manifest["precision"]["jax_tpu_compute_dtype"] == "bfloat16"
    assert manifest["scheduler"]["required_device_types"] == [
        "cpu",
        "cuda",
        "jax_tpu",
    ]
    assert [(stage["layer_start"], stage["layer_end"]) for stage in manifest["stages"]] == [
        (0, 7),
        (7, 14),
        (14, 20),
        (20, 26),
        (26, 28),
    ]
    assert manifest["stages"][2]["allowed_device_types"] == ["jax_tpu"]
    assert manifest["stages"][2]["preferred_device_type"] == "jax_tpu"
    assert manifest["stages"][-1]["allowed_device_types"] == ["cpu"]
    assert manifest["training"]["target_steps"] == 6


def test_canonical_manifest_matches_published_json_schema() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "model_manifest_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    jsonschema.validate(qwen25_7b_lora_manifest(), schema)

    invalid = qwen25_7b_lora_manifest()
    invalid["private_paths_public"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)


def test_tpu_manifest_matches_published_v2_json_schema() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "model_manifest_v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    jsonschema.validate(qwen25_7b_lora_tpu_manifest(), schema)

    invalid = qwen25_7b_lora_tpu_manifest()
    invalid["precision"].pop("jax_tpu_compute_dtype")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)


def test_manifest_accepts_arbitrary_contiguous_stage_count() -> None:
    manifest = qwen25_7b_lora_manifest()
    manifest.pop("content_hash")
    first = deepcopy(manifest["stages"][0])
    second = deepcopy(manifest["stages"][1])
    merged = {
        **first,
        "layer_end": second["layer_end"],
        "layer_count": second["layer_end"] - first["layer_start"],
        "estimated_parameter_count": first["estimated_parameter_count"]
        + second["estimated_parameter_count"],
        "estimated_weight_bytes": first["estimated_weight_bytes"]
        + second["estimated_weight_bytes"],
        "estimated_compute_units": first["estimated_compute_units"]
        + second["estimated_compute_units"],
    }
    stages = [merged, *deepcopy(manifest["stages"][2:])]
    for stage_id, stage in enumerate(stages):
        stage["stage_id"] = stage_id
    manifest["stages"] = stages

    validated = validate_training_manifest(manifest)

    assert len(validated["stages"]) == 4
    assert validated["stages"][0]["layer_end"] == 14
    assert validated["stages"][-1]["stage_id"] == 3


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda value: value["stages"][1].update(layer_start=8),
            "heterogeneous_manifest_stage_layers_not_contiguous",
        ),
        (
            lambda value: value["stages"][0].update(allowed_device_types=[]),
            "heterogeneous_manifest_stage_device_types_invalid",
        ),
        (
            lambda value: value["model"].update(model_revision=""),
            "heterogeneous_manifest_model_revision_required",
        ),
    ],
)
def test_manifest_rejects_invalid_topology(mutate, code: str) -> None:
    manifest = qwen25_7b_lora_manifest()
    manifest.pop("content_hash")
    mutate(manifest)
    with pytest.raises(TrainingManifestError, match=code):
        validate_training_manifest(manifest)


def test_manifest_content_hash_fails_closed() -> None:
    manifest = qwen25_7b_lora_manifest()
    manifest["content_hash"] = "sha256:" + "0" * 64
    with pytest.raises(
        TrainingManifestError, match="heterogeneous_manifest_content_hash_mismatch"
    ):
        validate_training_manifest(manifest)
