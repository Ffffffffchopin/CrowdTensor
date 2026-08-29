from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from crowdtensor.training_contract import sha256_json


ROOT = Path(__file__).resolve().parents[1]


def _brief() -> dict:
    return json.loads(
        (ROOT / "campaigns" / "commons-3b-public-pilot.json").read_text(
            encoding="utf-8"
        )
    )


def test_public_campaign_brief_is_schema_valid_and_hash_bound() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "public_campaign_brief_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    brief = _brief()
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(brief, schema)
    assert brief["content_hash"] == sha256_json(
        {key: value for key, value in brief.items() if key != "content_hash"}
    )
    assert {item["id"] for item in brief["contribution_lanes"]} == {
        "compute",
        "data",
        "review",
    }


def test_public_campaign_brief_keeps_claim_and_privacy_boundaries_explicit() -> None:
    brief = _brief()
    assert brief["status"] == "launch_candidate"
    assert brief["trust_boundary"]["enrollment"] == "controlled"
    assert brief["trust_boundary"]["external_contributors_verified"] is False
    assert brief["trust_boundary"]["permissionless_admission"] is False
    assert brief["trust_boundary"]["statistical_significance_claimed"] is False
    assert brief["data"]["raw_records_published"] is False
    serialized = json.dumps(brief, sort_keys=True)
    for marker in ("invite_token", "lease_token", "credential_token", "tensor_values"):
        assert marker not in serialized
