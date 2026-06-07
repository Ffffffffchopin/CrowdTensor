#!/usr/bin/env python3
"""CI-safe contract check for Public P2P Swarm Inference v1.0 RC."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import public_p2p_swarm_inference_v1_rc_pack as pack  # noqa: E402


SCHEMA = "public_p2p_swarm_inference_v1_rc_check_v1"
REQUIRED_CODES = {
    "public_p2p_swarm_inference_v1_rc_ready",
    "signed_peer_announcement_ready",
    "peer_identity_ready",
    "peer_registry_health_ready",
    "ttl_refresh_ready",
    "local_signed_p2p_discovery_ready",
    "serve_join_generate_p2p_commands_ready",
    "coordinator_ledger_fallback_ready",
    "petals_style_public_preview_ready",
    "tiny_gpt2_multi_token_ready",
    "p2p_stage_rescue_ready",
    "public_p2p_v1_rc_model_metadata_ready",
    "not_libp2p",
    "not_dht",
    "not_nat_traversal",
    "not_decentralized_security",
    "not_economic_system",
    "not_large_model_throughput",
    "not_hivemind_petals_production_parity",
}
EXTERNAL_REQUIRED_CODES = {
    "external_p2p_runtime_verified",
    "external_p2p_generate_verified",
}
SECRET_FRAGMENTS = [
    "peer-secret-value",
    "CROWDTENSOR_P2P_PEER_SECRET=",
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "Bearer ",
    "hidden_state",
    "input_ids",
    "logits",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def completed(payload: dict[str, Any], *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


def fake_signed_local_v06() -> dict[str, Any]:
    return {
        "schema": "p2p_swarm_inference_v06_v1",
        "ok": True,
        "hf_model_id": pack.DEFAULT_HF_MODEL_ID,
        "p2p": {
            "ready": True,
            "hf_model_id": pack.DEFAULT_HF_MODEL_ID,
            "observed_hf_model_id": pack.DEFAULT_HF_MODEL_ID,
            "model_id_match": True,
            "catalog_peer_count": 3,
            "registry": {"signed_peer_count": 3, "healthy_peer_count": 3, "signed_announcement_required": True},
            "signed_peer_count": 3,
            "healthy_peer_count": 3,
            "stage0_peer_count": 1,
            "stage1_peer_count": 1,
            "stage_rescue_ready": True,
            "real_generate_ready": True,
            "real_stage_rescue_ready": True,
            "generation": {"generated_token_count": 2, "decoded_tokens_match": True},
        },
        "diagnosis_codes": [
            "p2p_swarm_inference_v06_ready",
            "p2pd_daemon_ready",
            "local_three_process_p2p_discovery_ready",
            "p2p_stage_discovery_ready",
            "p2p_generate_route_ready",
            "p2p_stage_rescue_ready",
            "p2p_real_generate_ready",
            "p2p_real_stage_rescue_ready",
            "tiny_gpt2_multi_token_ready",
            "distilgpt2_attempt_ready",
            "coordinator_to_p2p_transition_ready",
            "coordinator_result_fallback_ready",
        ],
    }


def fake_external_v06() -> dict[str, Any]:
    return {
        "schema": "p2p_swarm_inference_v06_v1",
        "ok": True,
        "hf_model_id": pack.DEFAULT_HF_MODEL_ID,
        "p2p": {
            "ready": True,
            "hf_model_id": pack.DEFAULT_HF_MODEL_ID,
            "observed_hf_model_id": pack.DEFAULT_HF_MODEL_ID,
            "model_id_match": True,
            "external_runtime_verified": True,
            "external_generate_verified": True,
            "stage0_peer_count": 1,
            "stage1_peer_count": 1,
            "generation": {"generated_token_count": 2, "decoded_tokens_match": True},
            "kaggle_lifecycle": {"kernels_deleted": True, "token_rotation_required": True},
        },
        "diagnosis_codes": [
            "p2p_swarm_inference_v06_ready",
            "p2p_swarm_inference_v06_kaggle_auto_ready",
            "external_p2p_runtime_verified",
            "external_p2p_generate_verified",
            "kaggle_kernels_deleted",
            "tiny_gpt2_multi_token_ready",
        ],
    }


def fake_kaggle_v06() -> dict[str, Any]:
    return {
        "schema": "p2p_swarm_inference_v06_kaggle_auto_v1",
        "ok": True,
        "hf_model_id": pack.DEFAULT_HF_MODEL_ID,
        "external_runtime_verified": True,
        "external_generate_verified": True,
        "generation": {"generated_token_count": 2, "decoded_tokens_match": True},
        "kaggle_lifecycle": {"kernels_deleted": True, "token_rotation_required": True},
        "diagnosis_codes": [
            "p2p_swarm_inference_v06_kaggle_auto_ready",
            "external_p2p_runtime_verified",
            "external_p2p_generate_verified",
            "kaggle_kernels_deleted",
            "tiny_gpt2_multi_token_ready",
        ],
    }


def validate_payload(payload: dict[str, Any], *, mode: str) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if payload.get("ok") is not True:
        errors.append("rc_not_ready")
    codes = set(payload.get("diagnosis_codes") or [])
    required_codes = set(REQUIRED_CODES)
    if mode in {pack.MODE_EVIDENCE_IMPORT, pack.MODE_KAGGLE_AUTO}:
        required_codes.update(EXTERNAL_REQUIRED_CODES)
    for code in sorted(required_codes - codes):
        errors.append(f"missing_code:{code}")
    rc = payload.get("rc") if isinstance(payload.get("rc"), dict) else {}
    required_rc = ["signed_local_ready", "generation_ready", "stage_rescue_ready", "runbook_ready"]
    if mode in {pack.MODE_EVIDENCE_IMPORT, pack.MODE_KAGGLE_AUTO}:
        required_rc.append("external_runtime_ready")
    for key in required_rc:
        if rc.get(key) is not True:
            errors.append(f"rc_missing:{key}")
    if rc.get("model_metadata_ready") is not True:
        errors.append("rc_missing:model_metadata_ready")
    p2p = payload.get("p2p") if isinstance(payload.get("p2p"), dict) else {}
    if p2p.get("hf_model_id") != pack.DEFAULT_HF_MODEL_ID:
        errors.append("p2p_hf_model_id_mismatch")
    required_model_keys = ["local_model"]
    if mode in {pack.MODE_EVIDENCE_IMPORT, pack.MODE_KAGGLE_AUTO}:
        required_model_keys.extend(["external_model", "kaggle_model"])
    for key in required_model_keys:
        model = p2p.get(key) if isinstance(p2p.get(key), dict) else {}
        if not model.get("observed_hf_model_id") or model.get("model_id_present") is not True:
            errors.append(f"{key}_id_missing")
        if model.get("model_id_match") is not True or model.get("compatible") is not True:
            errors.append(f"{key}_mismatch")
    if p2p.get("signed_announcement_required") is not True:
        errors.append("signed_announcement_not_required")
    if int(p2p.get("signed_peer_count") or 0) < 3:
        errors.append("signed_peer_count_too_low")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    for key in ["signed_peer_announcement", "peer_secret_gossiped", "tokens_gossiped", "not_libp2p", "not_dht", "not_nat_traversal"]:
        expected = False if key in {"peer_secret_gossiped", "tokens_gossiped"} else True
        if safety.get(key) is not expected:
            errors.append(f"safety_mismatch:{key}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    output_request = payload.get("output_request") if isinstance(payload.get("output_request"), dict) else {}
    if output_request.get("include_output") is not False:
        errors.append("output_request_include_output_not_false")
    if output_request.get("raw_prompt_public") is not False:
        errors.append("output_request_raw_prompt_public_not_false")
    if output_request.get("raw_generated_text_public") is not False:
        errors.append("output_request_raw_generated_text_public_not_false")
    if output_request.get("generated_token_ids_public") is not False:
        errors.append("output_request_generated_token_ids_public_not_false")
    if output_request.get("public_artifact_safe") is not True:
        errors.append("output_request_public_artifact_safe_missing")
    answer_scope = payload.get("answer_scope") if isinstance(payload.get("answer_scope"), dict) else {}
    if answer_scope.get("scope_state") != "no-local-answer":
        errors.append("answer_scope_state_mismatch")
    if answer_scope.get("visible_in_terminal") is not False:
        errors.append("answer_scope_visible_in_terminal_not_false")
    if answer_scope.get("terminal_only") is not False:
        errors.append("answer_scope_terminal_only_not_false")
    if answer_scope.get("saved_json_display") != "hash-only":
        errors.append("answer_scope_saved_json_not_hash_only")
    if answer_scope.get("saved_markdown_display") != "hash-only":
        errors.append("answer_scope_saved_markdown_not_hash_only")
    if answer_scope.get("public_artifact_safe") is not True:
        errors.append("answer_scope_public_artifact_safe_missing")
    shareable_summary = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
    if shareable_summary.get("saved_artifacts_public_safe") is not True:
        errors.append("shareable_saved_artifacts_public_safe_missing")
    if shareable_summary.get("raw_prompt_public") is not False:
        errors.append("shareable_raw_prompt_public_not_false")
    if shareable_summary.get("raw_generated_text_public") is not False:
        errors.append("shareable_raw_generated_text_public_not_false")
    if shareable_summary.get("generated_token_ids_public") is not False:
        errors.append("shareable_generated_token_ids_public_not_false")
    if shareable_summary.get("answer_scope_state") != "no-local-answer":
        errors.append("shareable_answer_scope_state_mismatch")
    if shareable_summary.get("local_answer_terminal_only") is not False:
        errors.append("shareable_local_answer_terminal_only_not_false")
    return errors


def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    joined = " ".join(command)
    if "p2p_swarm_inference_v06_pack.py" in joined and "local-smoke" in joined:
        return completed(fake_signed_local_v06())
    if "p2p_swarm_inference_v06_pack.py" in joined and "kaggle-auto" in joined:
        return completed(fake_external_v06())
    return completed({"schema": "unknown", "ok": True})


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="crowdtensor_public_p2p_v1_rc_check_")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    signed_local = output_dir / "signed_local.json"
    external = output_dir / "external.json"
    kaggle = output_dir / "kaggle.json"
    write_json(signed_local, fake_signed_local_v06())
    write_json(external, fake_external_v06())
    write_json(kaggle, fake_kaggle_v06())

    if args.mode == pack.MODE_LOCAL_SMOKE:
        parsed = pack.parse_args([
            pack.MODE_LOCAL_SMOKE,
            "--output-dir",
            str(output_dir / "rc"),
            "--peer-secret",
            "peer-secret-value",
            "--json",
        ])
        report = pack.build_report(parsed, runner=fake_runner)
    elif args.mode == pack.MODE_KAGGLE_AUTO:
        parsed = pack.parse_args([
            pack.MODE_KAGGLE_AUTO,
            "--output-dir",
            str(output_dir / "rc"),
            "--peer-secret",
            "peer-secret-value",
            "--kaggle-owner",
            "owner",
            "--json",
        ])
        report = pack.build_report(parsed, runner=fake_runner)
    else:
        parsed = pack.parse_args([
            pack.MODE_EVIDENCE_IMPORT,
            "--output-dir",
            str(output_dir / "rc"),
            "--signed-local-report",
            str(signed_local),
            "--v06-external-report",
            str(external),
            "--v06-kaggle-report",
            str(kaggle),
            "--json",
        ])
        report = pack.build_report(parsed, runner=fake_runner)

    errors = validate_payload(report, mode=args.mode)
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "errors": errors,
        "public_p2p_swarm_inference_v1_rc_schema": report.get("schema"),
        "public_p2p_swarm_inference_v1_rc_ok": report.get("ok"),
        "diagnosis_codes": ["public_p2p_swarm_inference_v1_rc_check_ready"] if not errors else ["public_p2p_swarm_inference_v1_rc_check_failed"],
        "artifacts": {
            "public_p2p_swarm_inference_v1_rc_json": str(output_dir / "rc" / "public_p2p_swarm_inference_v1_rc.json"),
            "public_p2p_swarm_inference_v1_rc_markdown": str(output_dir / "rc" / "public_p2p_swarm_inference_v1_rc.md"),
        },
    }
    write_json(output_dir / "public_p2p_swarm_inference_v1_rc_check.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Public P2P Swarm Inference v1.0 RC.")
    parser.add_argument("--mode", choices=[pack.MODE_LOCAL_SMOKE, pack.MODE_EVIDENCE_IMPORT, pack.MODE_KAGGLE_AUTO], default=pack.MODE_EVIDENCE_IMPORT)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    report = run_check(parse_args())
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
