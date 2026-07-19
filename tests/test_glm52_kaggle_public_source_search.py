from __future__ import annotations

from types import SimpleNamespace

from scripts import glm52_kaggle_public_source_search as search
from scripts import glm52_kaggle_public_source_search_check as check


def _model(**kwargs):
    return SimpleNamespace(**kwargs)


def _dataset(**kwargs):
    return SimpleNamespace(**kwargs)


def test_summarize_model_detects_attachable_glm52_weight_source() -> None:
    model = _model(
        ref="zai-org/glm-5-2",
        title="GLM-5.2",
        subtitle="GLM 5.2 safetensors weights",
        author="zai-org",
        url="https://www.kaggle.com/models/zai-org/glm-5-2",
        is_private=False,
        vote_count=3,
        description="not public",
        instances=[
            '{"slug":"default","framework":"MODEL_FRAMEWORK_PY_TORCH","versionNumber":1,'
            '"totalUncompressedBytes":440000000000,"licenseName":"MIT",'
            '"url":"https://www.kaggle.com/models/zai-org/glm-5-2/PyTorch/default"}'
        ],
    )

    summary = search.summarize_model(model, query="GLM-5.2")

    assert summary["glm52_text_match"] is True
    assert summary["weight_source_signal"] is True
    assert summary["attachable_glm52_weight_source_candidate"] is True
    assert summary["kaggle_kernel_model_sources"] == ["zai-org/glm-5-2/PyTorch/default/1"]
    assert summary["description_public_excerpt_included"] is False
    assert search.public_redaction_errors(summary) == []


def test_summarize_dataset_rejects_non_weight_glm_benchmark() -> None:
    dataset = _dataset(
        ref="s1m0n38/balatrobench",
        title="BalatroBench",
        subtitle="Metrics and gameplay logs mentioning GLM-5.2",
        owner_ref="s1m0n38",
        url="https://www.kaggle.com/datasets/s1m0n38/balatrobench",
        total_bytes=3_900_000_000,
        license_name="CC0",
        download_count=1,
        vote_count=0,
        description="not public",
    )

    summary = search.summarize_dataset(dataset, query="GLM-5.2")

    assert summary["glm52_text_match"] is True
    assert summary["weight_source_signal"] is False
    assert summary["attachable_glm52_weight_source_candidate"] is False
    assert summary["description_public_excerpt_included"] is False
    assert search.public_redaction_errors(summary) == []


def test_checker_accepts_no_source_blocker_report() -> None:
    report = {
        "schema": search.SCHEMA,
        "ok": True,
        "glm52_kaggle_public_source_search_ready": True,
        "public_artifact_safe": True,
        "query_count": 1,
        "model_result_count": 0,
        "dataset_result_count": 0,
        "compatible_model_source_count": 0,
        "compatible_dataset_source_count": 0,
        "kaggle_models_glm52_source_verified": False,
        "kaggle_datasets_glm52_source_verified": False,
        "kaggle_attach_source_verified": False,
        "recommended_kaggle_kernel_model_sources": [],
        "model_results": [],
        "dataset_results": [],
        "compatible_model_candidates": [],
        "compatible_dataset_candidates": [],
        "blockers": [
            "kaggle_models_glm52_weight_source_not_found",
            "kaggle_datasets_glm52_weight_source_not_found",
        ],
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
        },
    }

    assert check.validate_report(report) == []
