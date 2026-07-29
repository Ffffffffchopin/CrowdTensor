"""Release and compatibility versions for the Community training surface."""

from __future__ import annotations


__version__ = "0.2.0rc7"
COMMUNITY_PROTOCOL_VERSION = "community_training_v1.0"
MODEL_ADAPTER_API_VERSION = "model_adapter_v1.0"
EVIDENCE_API_VERSION = "community_evidence_v1.0"


def public_version() -> dict[str, str]:
    return {
        "package_version": __version__,
        "community_protocol_version": COMMUNITY_PROTOCOL_VERSION,
        "model_adapter_api_version": MODEL_ADAPTER_API_VERSION,
        "evidence_api_version": EVIDENCE_API_VERSION,
    }
