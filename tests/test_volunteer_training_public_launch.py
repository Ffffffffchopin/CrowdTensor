from __future__ import annotations

import json

from scripts.volunteer_training_public_launch_check import (
    MULTI_HOST_SCHEMA,
    _validate_multi_host,
)
from crowdtensor.training_contract import sha256_json


def test_formal_launch_evidence_requires_independent_hosts_and_route(tmp_path) -> None:
    value = {
        "schema": MULTI_HOST_SCHEMA,
        "physical_multi_host_verified": True,
        "independent_host_identities_verified": False,
        "independent_admin_domains_verified": False,
        "real_network_route_verified": False,
        "cleanup_verified": True,
        "independent_host_count": 1,
        "credential_values_public": False,
        "private_paths_public": False,
        "raw_data_public": False,
        "tensor_values_public": False,
        "public_artifact_safe": True,
    }
    value["content_hash"] = sha256_json(value)
    path = tmp_path / "formal.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    ready, errors, _ = _validate_multi_host(path)
    assert ready is False
    assert "formal_multihost_host_count_insufficient" in errors
    assert "formal_multihost_field_missing:real_network_route_verified" in errors
