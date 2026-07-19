import pytest

from crowdtensor.community_protocol import (
    ProtocolCompatibilityError,
    negotiate_protocol,
    parse_protocol_version,
    require_compatible_protocol,
)


def test_protocol_accepts_same_version_and_required_capabilities() -> None:
    result = negotiate_protocol(
        "community_training_v1.0",
        peer_capabilities=["peft_lora", "atomic_checkpoint"],
        required_capabilities=["atomic_checkpoint"],
    )
    assert result["accepted"] is True
    assert result["missing_capabilities"] == []
    assert result["silent_downgrade_allowed"] is False


def test_protocol_accepts_older_minor_only_when_local_explicitly_supports_it() -> None:
    result = negotiate_protocol(
        "community_training_v1.0",
        local_version="community_training_v1.1",
    )
    assert result["accepted"] is True


@pytest.mark.parametrize(
    ("peer", "reason"),
    [
        ("community_training_v2.0", "community_protocol_major_incompatible"),
        ("community_training_v1.1", "community_protocol_peer_minor_newer"),
        ("other_training_v1.0", "community_protocol_family_incompatible"),
    ],
)
def test_protocol_rejects_unknown_major_minor_and_family(peer: str, reason: str) -> None:
    result = negotiate_protocol(peer)
    assert result["accepted"] is False
    assert reason in result["rejection_reasons"]
    with pytest.raises(ProtocolCompatibilityError):
        require_compatible_protocol(peer)


def test_protocol_rejects_invalid_and_capability_gap() -> None:
    with pytest.raises(ProtocolCompatibilityError, match="version_invalid"):
        parse_protocol_version("v1")
    result = negotiate_protocol(
        "community_training_v1.0",
        peer_capabilities=["peft_lora"],
        required_capabilities=["signed_task"],
    )
    assert result["accepted"] is False
    assert result["missing_capabilities"] == ["signed_task"]
