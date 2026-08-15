from __future__ import annotations

from crowdtensor.volunteer_campaign_proposal import (
    PROPOSAL_SCHEMA,
    proposal_template,
    validate_proposal,
)


def test_built_in_template_is_ready() -> None:
    template = proposal_template()
    result = validate_proposal(template)
    assert template["schema"] == PROPOSAL_SCHEMA
    assert result["ok"] is True
    assert result["campaign_proposal_ready"] is True


def test_proposal_rejects_ambiguous_safety_or_mutable_inputs() -> None:
    proposal = proposal_template()
    proposal["model"]["revision"] = "main"
    proposal["safety"]["poisoning_safety_claimed"] = True
    proposal["content_hash"] = "sha256:" + "0" * 64
    result = validate_proposal(proposal)
    assert result["ok"] is False
    assert "volunteer_campaign_proposal_model_revision_not_immutable" in result[
        "errors"
    ]
    assert "volunteer_campaign_proposal_safety_overclaim:poisoning_safety_claimed" in result[
        "errors"
    ]
    assert result["public_artifact_safe"] is True


def test_proposal_validation_is_public_safe_for_missing_sections() -> None:
    result = validate_proposal({"schema": PROPOSAL_SCHEMA})
    assert result["ok"] is False
    assert result["credential_values_public"] is False
    assert result["private_paths_public"] is False
    assert result["raw_data_public"] is False
    assert result["tensor_values_public"] is False
