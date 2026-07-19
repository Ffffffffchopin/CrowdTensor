"""Public proposal contract for community-operated training Campaigns."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .training_contract import sha256_json
from .volunteer_training_protocol import with_public_safety


PROPOSAL_SCHEMA = "crowdtensor_volunteer_campaign_proposal_v1"
VALIDATION_SCHEMA = "crowdtensor_volunteer_campaign_proposal_validation_v1"
HEX_REVISION = re.compile(r"^[0-9a-f]{7,64}$")
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
SPDX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]{1,63}$")


def proposal_template() -> dict[str, Any]:
    proposal: dict[str, Any] = {
        "schema": PROPOSAL_SCHEMA,
        "proposal_id": "smollm2-wikitext-community-001",
        "title": "SmolLM2 WikiText community adaptation",
        "summary": "A bounded public LoRA Campaign with immutable inputs and reproducible evaluation.",
        "model": {
            "model_id": "HuggingFaceTB/SmolLM2-135M",
            "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
            "license_spdx": "Apache-2.0",
            "source_public": True,
            "redistribution_allowed": True,
            "parameter_count": 134_515_008,
        },
        "dataset": {
            "dataset_id": "Salesforce/wikitext",
            "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "licenses_spdx": ["CC-BY-SA-3.0", "GFDL-1.3-or-later"],
            "source_public": True,
            "redistribution_allowed": True,
            "personal_data_reviewed": True,
            "dataset_snapshot_hash": "sha256:3b764bde8301a060f872901bf28d621d7a879e3bfe16553ff6a4ea2fe8d4c058",
        },
        "training": {
            "method": "peft_lora",
            "target_rounds": 3,
            "minimum_quorum": 2,
            "local_steps": 1,
            "sequence_length": 16,
            "maximum_update_bytes": 268_435_456,
            "base_weights_frozen": True,
        },
        "evaluation": {
            "suite_id": "wikitext-heldout-v1",
            "baseline_required": True,
            "heldout_split_hash": "sha256:c6bb82e43dae09d261c93bdc7a5eeecda1861794e9130ddc1d13b832dc0c824e",
            "metrics": ["validation_loss", "perplexity"],
            "publish_before_after_results": True,
        },
        "compute": {
            "supported_devices": ["cpu", "cuda"],
            "minimum_memory_bytes": 4_294_967_296,
            "maximum_download_bytes": 8_589_934_592,
            "intermittent_contributors_supported": True,
        },
        "governance": {
            "governance_model": "named_maintainers",
            "maintainers": ["maintainer-a", "maintainer-b"],
            "decision_log_public": True,
            "conflict_disclosures_required": True,
            "pause_and_rollback_owner": "maintainer-a",
        },
        "publication": {
            "campaign_manifest_public": True,
            "checkpoints_public": True,
            "audit_ledger_public": True,
            "evaluation_results_public": True,
            "result_license_spdx": "Apache-2.0",
            "attribution_file_required": True,
        },
        "safety": {
            "allowed_use_summary": "Open research and reproducible language-model adaptation.",
            "prohibited_data_categories": [
                "private_credentials",
                "non-consensual_personal_data",
                "copyrighted_data_without_permission",
            ],
            "moderation_owner": "maintainer-b",
            "rollback_plan_public": True,
            "permissionless_admission": False,
            "sybil_resistance_claimed": False,
            "poisoning_safety_claimed": False,
        },
    }
    proposal["content_hash"] = sha256_json(proposal)
    return proposal


def _object(value: Any, field: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append("volunteer_campaign_proposal_" + field + "_invalid")
        return {}
    return value


def _true(value: dict[str, Any], field: str, errors: list[str], code: str) -> None:
    if value.get(field) is not True:
        errors.append(code)


def validate_proposal(value: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, dict) or value.get("schema") != PROPOSAL_SCHEMA:
        errors.append("volunteer_campaign_proposal_schema_invalid")
        value = value if isinstance(value, dict) else {}
    supplied_hash = str(value.get("content_hash") or "")
    expected_hash = sha256_json(
        {key: item for key, item in value.items() if key != "content_hash"}
    )
    if supplied_hash != expected_hash:
        errors.append("volunteer_campaign_proposal_content_hash_mismatch")
    if not IDENTIFIER.fullmatch(str(value.get("proposal_id") or "")):
        errors.append("volunteer_campaign_proposal_id_invalid")
    if not 8 <= len(str(value.get("title") or "")) <= 120:
        errors.append("volunteer_campaign_proposal_title_invalid")
    if not 24 <= len(str(value.get("summary") or "")) <= 500:
        errors.append("volunteer_campaign_proposal_summary_invalid")

    model = _object(value.get("model"), "model", errors)
    dataset = _object(value.get("dataset"), "dataset", errors)
    training = _object(value.get("training"), "training", errors)
    evaluation = _object(value.get("evaluation"), "evaluation", errors)
    compute = _object(value.get("compute"), "compute", errors)
    governance = _object(value.get("governance"), "governance", errors)
    publication = _object(value.get("publication"), "publication", errors)
    safety = _object(value.get("safety"), "safety", errors)

    if not str(model.get("model_id") or ""):
        errors.append("volunteer_campaign_proposal_model_id_missing")
    if not HEX_REVISION.fullmatch(str(model.get("revision") or "")):
        errors.append("volunteer_campaign_proposal_model_revision_not_immutable")
    if not SPDX.fullmatch(str(model.get("license_spdx") or "")):
        errors.append("volunteer_campaign_proposal_model_license_invalid")
    _true(model, "source_public", errors, "volunteer_campaign_proposal_model_not_public")
    _true(
        model,
        "redistribution_allowed",
        errors,
        "volunteer_campaign_proposal_model_redistribution_unverified",
    )
    if int(model.get("parameter_count") or 0) < 1:
        errors.append("volunteer_campaign_proposal_parameter_count_invalid")

    if not str(dataset.get("dataset_id") or ""):
        errors.append("volunteer_campaign_proposal_dataset_id_missing")
    if not HEX_REVISION.fullmatch(str(dataset.get("revision") or "")):
        errors.append("volunteer_campaign_proposal_dataset_revision_not_immutable")
    licenses = dataset.get("licenses_spdx")
    if not isinstance(licenses, list) or not licenses or any(
        not SPDX.fullmatch(str(item)) for item in licenses
    ):
        errors.append("volunteer_campaign_proposal_dataset_license_invalid")
    _true(dataset, "source_public", errors, "volunteer_campaign_proposal_dataset_not_public")
    _true(
        dataset,
        "redistribution_allowed",
        errors,
        "volunteer_campaign_proposal_dataset_redistribution_unverified",
    )
    _true(
        dataset,
        "personal_data_reviewed",
        errors,
        "volunteer_campaign_proposal_personal_data_review_missing",
    )
    if not SHA256.fullmatch(str(dataset.get("dataset_snapshot_hash") or "")):
        errors.append("volunteer_campaign_proposal_dataset_snapshot_hash_invalid")

    if training.get("method") != "peft_lora":
        errors.append("volunteer_campaign_proposal_training_method_unsupported")
    for field in ("target_rounds", "minimum_quorum", "local_steps", "sequence_length"):
        if int(training.get(field) or 0) < 1:
            errors.append("volunteer_campaign_proposal_training_" + field + "_invalid")
    if int(training.get("minimum_quorum") or 0) < 2:
        errors.append("volunteer_campaign_proposal_quorum_insufficient")
    _true(
        training,
        "base_weights_frozen",
        errors,
        "volunteer_campaign_proposal_base_weights_not_frozen",
    )
    if int(training.get("maximum_update_bytes") or 0) < 1024:
        errors.append("volunteer_campaign_proposal_update_limit_invalid")

    if not str(evaluation.get("suite_id") or ""):
        errors.append("volunteer_campaign_proposal_evaluation_suite_missing")
    _true(
        evaluation,
        "baseline_required",
        errors,
        "volunteer_campaign_proposal_baseline_missing",
    )
    _true(
        evaluation,
        "publish_before_after_results",
        errors,
        "volunteer_campaign_proposal_evaluation_publication_missing",
    )
    if not SHA256.fullmatch(str(evaluation.get("heldout_split_hash") or "")):
        errors.append("volunteer_campaign_proposal_heldout_hash_invalid")
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, list) or not metrics or any(
        not str(item).strip() for item in metrics
    ):
        errors.append("volunteer_campaign_proposal_metrics_missing")

    devices = compute.get("supported_devices")
    if not isinstance(devices, list) or not devices or not set(devices).issubset(
        {"cpu", "cuda"}
    ):
        errors.append("volunteer_campaign_proposal_compute_devices_invalid")
    _true(
        compute,
        "intermittent_contributors_supported",
        errors,
        "volunteer_campaign_proposal_intermittent_compute_missing",
    )
    for field in ("minimum_memory_bytes", "maximum_download_bytes"):
        if int(compute.get(field) or 0) < 1:
            errors.append("volunteer_campaign_proposal_compute_" + field + "_invalid")

    maintainers = governance.get("maintainers")
    if governance.get("governance_model") != "named_maintainers":
        errors.append("volunteer_campaign_proposal_governance_model_invalid")
    if not isinstance(maintainers, list) or len(set(maintainers)) < 2:
        errors.append("volunteer_campaign_proposal_maintainers_insufficient")
    _true(
        governance,
        "decision_log_public",
        errors,
        "volunteer_campaign_proposal_decision_log_not_public",
    )
    _true(
        governance,
        "conflict_disclosures_required",
        errors,
        "volunteer_campaign_proposal_conflict_policy_missing",
    )
    if not str(governance.get("pause_and_rollback_owner") or ""):
        errors.append("volunteer_campaign_proposal_rollback_owner_missing")

    for field in (
        "campaign_manifest_public",
        "checkpoints_public",
        "audit_ledger_public",
        "evaluation_results_public",
        "attribution_file_required",
    ):
        _true(
            publication,
            field,
            errors,
            "volunteer_campaign_proposal_publication_" + field + "_missing",
        )
    if not SPDX.fullmatch(str(publication.get("result_license_spdx") or "")):
        errors.append("volunteer_campaign_proposal_result_license_invalid")

    if len(str(safety.get("allowed_use_summary") or "")) < 16:
        errors.append("volunteer_campaign_proposal_allowed_use_missing")
    prohibited = safety.get("prohibited_data_categories")
    if not isinstance(prohibited, list) or len(prohibited) < 3:
        errors.append("volunteer_campaign_proposal_prohibited_data_policy_missing")
    if not str(safety.get("moderation_owner") or ""):
        errors.append("volunteer_campaign_proposal_moderation_owner_missing")
    _true(
        safety,
        "rollback_plan_public",
        errors,
        "volunteer_campaign_proposal_rollback_plan_not_public",
    )
    for field in (
        "permissionless_admission",
        "sybil_resistance_claimed",
        "poisoning_safety_claimed",
    ):
        if safety.get(field) is not False:
            errors.append("volunteer_campaign_proposal_safety_overclaim:" + field)

    report = with_public_safety(
        {
            "schema": VALIDATION_SCHEMA,
            "ok": not errors,
            "campaign_proposal_ready": not errors,
            "proposal_id": value.get("proposal_id"),
            "proposal_content_hash": supplied_hash,
            "immutable_model_source": not any(
                "model_revision" in error for error in errors
            ),
            "immutable_dataset_source": not any(
                "dataset_revision" in error or "dataset_snapshot" in error
                for error in errors
            ),
            "license_review_ready": not any("license" in error for error in errors),
            "evaluation_plan_ready": not any(
                "evaluation" in error or "baseline" in error or "heldout" in error
                or "metrics" in error
                for error in errors
            ),
            "governance_ready": not any(
                "governance" in error
                or "maintainer" in error
                or "decision_log" in error
                or "conflict" in error
                for error in errors
            ),
            "safety_boundary_explicit": not any("safety_overclaim" in error for error in errors),
            "error_count": len(errors),
            "errors": sorted(set(errors)),
        }
    )
    report["content_hash"] = sha256_json(report)
    return report


def load_and_validate_proposal(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    return validate_proposal(value)


def write_proposal_template(path: str | Path) -> dict[str, Any]:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    value = proposal_template()
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.chmod(0o644)
    return with_public_safety(
        {
            "schema": "crowdtensor_volunteer_campaign_proposal_template_result_v1",
            "ok": True,
            "proposal_id": value["proposal_id"],
            "proposal_content_hash": value["content_hash"],
            "template_written": True,
        }
    )
