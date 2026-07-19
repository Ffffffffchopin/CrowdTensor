from copy import deepcopy

from crowdtensor.model_adapter import stable_hash
from crowdtensor.version import public_version
from scripts.community_maturity_rc_check import (
    REQUIRED_ARTIFACTS,
    REQUIRED_REQUIREMENT_KEYS,
    SCHEMA,
    check_report,
)


def valid_report() -> dict:
    value = {
        "schema": SCHEMA,
        "versions": public_version(),
        "node_scope": "Kaggle logical multi-node",
        "physical_multi_machine_verified": False,
        "community_maturity_rc_ready": True,
        "gates": {
            "p0_ready": True,
            "p1_ready": True,
            "p2_ready": True,
            "p3_ready": True,
            "p4_ready": True,
            "cleanup_ready": True,
            "wheel_identity_verified": True,
            "community_maturity_rc_ready": True,
        },
        "requirements": {
            section: {name: True for name in names}
            for section, names in REQUIRED_REQUIREMENT_KEYS.items()
        },
        "source_checks": {name: True for name in REQUIRED_ARTIFACTS},
        "artifacts": {
            name: {
                "relative_path": f"evidence/{name}.json",
                "sha256": "sha256:" + "a" * 64,
            }
            for name in REQUIRED_ARTIFACTS
        },
        "blockers": [],
        "credential_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    value["content_hash"] = stable_hash(value)
    return value


def test_rc_report_contract_accepts_ready_p0_p4() -> None:
    assert check_report(valid_report(), require_ready=True) == []


def test_rc_report_rejects_overclaim_and_inconsistent_readiness() -> None:
    value = deepcopy(valid_report())
    value["physical_multi_machine_verified"] = True
    value["gates"]["p1_ready"] = False
    value["blockers"] = []
    value["content_hash"] = stable_hash(
        {key: item for key, item in value.items() if key != "content_hash"}
    )
    errors = check_report(value, require_ready=True)
    assert "community_maturity_physical_multi_machine_overclaim" in errors
    assert "community_maturity_ready_consistency_invalid" in errors
    assert "community_maturity_rc_not_ready" in errors
