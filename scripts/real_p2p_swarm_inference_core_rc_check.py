#!/usr/bin/env python3
"""CI-safe checks for Real P2P Swarm Inference Core RC."""

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

import real_p2p_swarm_inference_core_rc_pack as pack  # noqa: E402


SCHEMA = "real_p2p_swarm_inference_core_rc_check_v1"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fake_ready_report(mode: str, output_dir: Path) -> dict[str, Any]:
    report = {
        "schema": pack.SCHEMA,
        "generated_at": pack.utc_now(),
        "ok": True,
        "mode": mode,
        "output_dir": str(output_dir),
        "backend": "cpu",
        "hf_model_id": pack.DEFAULT_HF_MODEL_ID,
        "max_new_tokens": 8,
        "p2p": {
            "backend": "real",
            "discovery_backend": "libp2p-kad",
            "catalog_schema": "real_p2p_provider_catalog_v1",
            "provider_count": 3,
            "signed_provider_record_count": 3,
            "coordinator_provider_count": 1,
            "stage0_provider_count": 1,
            "stage1_provider_count": 1,
            "route": {"route_source": "real-p2p-discovery", "usable_now": True},
            "libp2p": {
                "schema": "libp2p_kad_backend_v1",
                "ok": True,
                "stable_peer_identity": True,
                "provider_record_transport": "libp2p-stream",
                "diagnosis_codes": [
                    "libp2p_discovery_backend_ready",
                    "p2p_peer_identity_ready",
                    "p2p_provider_dht_ready",
                ],
            },
            "peer_scoring": {
                "schema": "real_p2p_peer_scoring_v1",
                "peer_count": 3,
                "scored_peers": [
                    {"peer_id": "stage0", "score": 1.2, "trust_status": "trusted"},
                    {"peer_id": "stage1", "score": 1.1, "trust_status": "trusted"},
                ],
            },
        },
        "generation": {"generated_token_count": 8, "max_new_tokens": 8, "generated_text_hash": "sha256:fake"},
        "stage_assignment": {
            "completed_rows": 16,
            "stage0_miner_id": "stage0",
            "stage1_miner_id": "stage1",
            "distinct_stage_miners": True,
        },
        "live_requeue_summary": {
            "enabled": True,
            "failure_mode": "kill-stage0-after-claim",
            "target_stage": "stage0",
            "victim_miner_id": "stage0-victim",
            "rescue_miner_id": "stage0-rescue",
            "claim_observed": True,
            "victim_kernel_deleted": True,
            "lease_expired": True,
            "rescue_miner_used": True,
            "accepted_result_after_requeue": True,
            "victim_result_accepted": False,
        },
        "diagnosis_codes": [
            "real_p2p_swarm_inference_core_rc_ready",
            "libp2p_or_real_p2p_discovery_ready",
            "real_p2p_stage_discovery_ready",
            "real_p2p_generate_route_ready",
            "real_p2p_local_generate_ready",
            "generated_token_count_ready",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "coordinator_result_fallback_ready",
            "libp2p_discovery_backend_ready",
            "p2p_peer_identity_ready",
            "p2p_provider_dht_ready",
            "hivemind_petals_class_alpha_local_ready",
            "real_p2p_local_stage_requeue_ready",
            "local_stage_requeue_ready",
            "stage_requeue_ready",
            "live_stage0_requeue_ready",
            "rescue_miner_used",
            "accepted_result_after_requeue",
            "peer_scoring_ready",
        ],
        "safety": pack.safety_block(discovery_backend="libp2p-kad"),
    }
    return pack.sanitize_report(report, output_dir=output_dir, secret_values=[])


def completed(payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=0 if payload.get("ok") else 1, stdout=json.dumps(payload) + "\n", stderr="")


def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
    joined = " ".join(command)
    if "real_p2p_swarm_inference_core_rc_pack.py" in joined and "evidence-import" in joined:
        output_dir = Path(command[command.index("--output-dir") + 1])
        report_path = Path(command[command.index("--real-p2p-report") + 1])
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload = {**payload, "cli_schema": "real_p2p_swarm_inference_core_rc_cli_v1"}
        write_json(output_dir / "real_p2p_swarm_inference_core_rc.json", payload)
        return completed(payload)
    if "real_p2p_swarm_inference_core_rc_pack.py" in joined and "package" in joined:
        output_dir = Path(command[command.index("--output-dir") + 1])
        payload = pack.build_report(pack.parse_args(["package", "--output-dir", str(output_dir)]))
        return completed(payload)
    raise AssertionError(command)


def validate_output_scope(payload: dict[str, Any], *, label: str) -> list[str]:
    errors: list[str] = []
    output_request = payload.get("output_request") if isinstance(payload.get("output_request"), dict) else {}
    if output_request.get("include_output") is not False:
        errors.append(f"{label}:output_request_include_output_not_false")
    if output_request.get("raw_prompt_public") is not False:
        errors.append(f"{label}:output_request_raw_prompt_public_not_false")
    if output_request.get("raw_generated_text_public") is not False:
        errors.append(f"{label}:output_request_raw_generated_text_public_not_false")
    if output_request.get("generated_token_ids_public") is not False:
        errors.append(f"{label}:output_request_generated_token_ids_public_not_false")
    if output_request.get("public_artifact_safe") is not True:
        errors.append(f"{label}:output_request_public_artifact_safe_missing")
    answer_scope = payload.get("answer_scope") if isinstance(payload.get("answer_scope"), dict) else {}
    if answer_scope.get("scope_state") != "no-local-answer":
        errors.append(f"{label}:answer_scope_state_mismatch")
    if answer_scope.get("visible_in_terminal") is not False:
        errors.append(f"{label}:answer_scope_visible_in_terminal_not_false")
    if answer_scope.get("terminal_only") is not False:
        errors.append(f"{label}:answer_scope_terminal_only_not_false")
    if answer_scope.get("saved_json_display") != "hash-only":
        errors.append(f"{label}:answer_scope_saved_json_not_hash_only")
    if answer_scope.get("saved_markdown_display") != "hash-only":
        errors.append(f"{label}:answer_scope_saved_markdown_not_hash_only")
    if answer_scope.get("public_artifact_safe") is not True:
        errors.append(f"{label}:answer_scope_public_artifact_safe_missing")
    shareable_summary = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
    if shareable_summary.get("saved_artifacts_public_safe") is not True:
        errors.append(f"{label}:shareable_saved_artifacts_public_safe_missing")
    if shareable_summary.get("raw_prompt_public") is not False:
        errors.append(f"{label}:shareable_raw_prompt_public_not_false")
    if shareable_summary.get("raw_generated_text_public") is not False:
        errors.append(f"{label}:shareable_raw_generated_text_public_not_false")
    if shareable_summary.get("generated_token_ids_public") is not False:
        errors.append(f"{label}:shareable_generated_token_ids_public_not_false")
    if shareable_summary.get("answer_scope_state") != "no-local-answer":
        errors.append(f"{label}:shareable_answer_scope_state_mismatch")
    if shareable_summary.get("local_answer_terminal_only") is not False:
        errors.append(f"{label}:shareable_local_answer_terminal_only_not_false")
    return errors


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="crowdtensor_real_p2p_rc_check_")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = fake_ready_report("local-smoke", output_dir / "source")
    source_path = output_dir / "source" / "real_p2p_swarm_inference_core_rc.json"
    package_report = pack.build_report(pack.parse_args(["package", "--output-dir", str(output_dir / "package")]))
    evidence_args = [
        "evidence-import",
        "--output-dir",
        str(output_dir / "import"),
        "--real-p2p-report",
        str(source_path),
    ]
    evidence_report = pack.build_report(pack.parse_args(evidence_args))
    errors: list[str] = []
    if package_report.get("schema") != pack.SCHEMA or package_report.get("ok") is not True:
        errors.append("package_report_not_ready")
    if evidence_report.get("schema") != pack.SCHEMA or evidence_report.get("ok") is not True:
        errors.append("evidence_import_not_ready")
    errors.extend(validate_output_scope(package_report, label="package"))
    errors.extend(validate_output_scope(evidence_report, label="evidence_import"))
    required = {
        "real_p2p_swarm_inference_core_rc_ready",
        "real_p2p_core_rc_model_metadata_ready",
        "libp2p_or_real_p2p_discovery_ready",
        "real_p2p_stage_discovery_ready",
        "real_p2p_generate_route_ready",
        "real_p2p_local_generate_ready",
        "libp2p_discovery_backend_ready",
        "p2p_peer_identity_ready",
        "p2p_provider_dht_ready",
        "real_p2p_local_stage_requeue_ready",
        "local_stage_requeue_ready",
        "stage_requeue_ready",
        "live_stage0_requeue_ready",
        "rescue_miner_used",
        "accepted_result_after_requeue",
        "peer_scoring_ready",
    }
    if not required.issubset(set(evidence_report.get("diagnosis_codes") or [])):
        errors.append("required_readiness_codes_missing")
    if int(((evidence_report.get("generation") or {}).get("generated_token_count")) or 0) < 8:
        errors.append("generated_token_count_below_candidate_floor")
    imported = evidence_report.get("imported") if isinstance(evidence_report.get("imported"), dict) else {}
    model = imported.get("model") if isinstance(imported.get("model"), dict) else {}
    if not model.get("observed_hf_model_id") or model.get("model_id_present") is not True:
        errors.append("model_id_missing")
    if model.get("model_id_match") is not True or model.get("compatible") is not True:
        errors.append("model_id_mismatch")
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "output_dir": str(output_dir),
        "errors": errors,
        "package": {
            "ok": package_report.get("ok"),
            "diagnosis_codes": package_report.get("diagnosis_codes"),
        },
        "evidence_import": {
            "ok": evidence_report.get("ok"),
            "diagnosis_codes": evidence_report.get("diagnosis_codes"),
        },
        "diagnosis_codes": ["real_p2p_swarm_inference_core_rc_check_ready"] if not errors else ["real_p2p_swarm_inference_core_rc_check_blocked"],
    }
    write_json(output_dir / "real_p2p_swarm_inference_core_rc_check.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Real P2P Swarm Inference Core RC package contracts.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"Real P2P Core RC check ready: {result.get('ok')}")
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
