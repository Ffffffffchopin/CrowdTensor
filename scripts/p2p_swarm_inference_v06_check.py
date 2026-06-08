#!/usr/bin/env python3
"""CI-safe contract check for P2P Swarm Inference v0.6."""

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

import p2p_swarm_inference_v06_pack as pack  # noqa: E402
import public_swarm_preview_v04_check as v04_check  # noqa: E402


SCHEMA = "p2p_swarm_inference_v06_check_v1"
REQUIRED_CODES = {
    "p2p_swarm_inference_v06_ready",
    "p2p_discovery_routing_prototype_ready",
    "coordinator_to_p2p_transition_ready",
    "coordinator_result_fallback_ready",
    "external_two_stage_generation_ready",
    "external_stage_requeue_ready",
    "tiny_gpt2_multi_token_ready",
    "distilgpt2_attempt_ready",
    "completed_p2p_discovery_routing_prototype",
    "not_production_nat_traversal",
    "not_decentralized_security",
    "not_economic_system",
    "not_large_model_throughput",
}
SECRET_FRAGMENTS = [
    "admin-secret",
    "miner-secret",
    "observer-secret",
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


def fake_product_payload(model_id: str = "sshleifer/tiny-gpt2") -> dict[str, Any]:
    payload = v04_check.fake_product_mvp_payload(model_id)
    payload["diagnosis_codes"] = sorted(set(payload["diagnosis_codes"]) | {"product_swarm_mvp_ready"})
    return payload


def fake_preview_payload() -> dict[str, Any]:
    return {
        "schema": "public_swarm_preview_v04_v1",
        "ok": True,
        "preview": {
            "ready": True,
            "external_two_stage_generation_ready": True,
            "external_stage_requeue_ready": True,
            "multi_token_generation_ready": True,
            "tiny_gpt2_ci_fallback_ready": True,
            "optional_model_ready": True,
        },
        "diagnosis_codes": [
            "public_swarm_preview_v04_ready",
            "external_two_stage_generation_ready",
            "external_stage_requeue_ready",
            "multi_token_generation_ready",
            "tiny_gpt2_ci_fallback_ready",
            "optional_distilgpt2_or_gpt2_strict_ready",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "stage_latency_ready",
            "throughput_summary_ready",
            "memory_or_vram_summary_ready",
        ],
    }


def fake_local_p2p(model_id: str = pack.DEFAULT_HF_MODEL_ID) -> dict[str, Any]:
    return {
        "schema": "p2p_swarm_local_discovery_v1",
        "ok": True,
        "hf_model_id": model_id,
        "p2p_url": "http://127.0.0.1:9560",
        "catalog_peer_count": 3,
        "coordinator_peer_count": 1,
        "stage0_peer_count": 1,
        "stage1_peer_count": 1,
        "kv_cache": {
            "schema": "p2p_real_generate_dual_stage_kv_cache_v1",
            "ready": True,
            "expected_hit_count_per_stage": 1,
            "stage0": {"ready": True, "ready_count": 2, "hit_count": 1},
            "stage1": {"ready": True, "ready_count": 2, "hit_count": 1},
            "raw_activations_public": False,
            "raw_token_inputs_public": False,
        },
        "generate_route": {
            "usable_now": True,
            "route_source": "p2p-discovery",
            "coordinator_url_present": True,
            "matched_capabilities": {
                "real_llm_sharded_stage0": "stage0",
                "real_llm_sharded_stage1": "stage1",
            },
        },
        "diagnosis_codes": [
            "p2pd_daemon_ready",
            "local_three_process_p2p_discovery_ready",
            "p2p_stage_discovery_ready",
            "p2p_generate_route_ready",
            "p2p_stage_rescue_ready",
            "p2p_real_generate_ready",
            "p2p_real_generate_kv_cache_ready",
            "p2p_real_stage_rescue_ready",
            "p2p_rescue_generation_completed",
            "stage0_rescue_generation_completed",
            "stage1_rescue_generation_completed",
            "stage0_victim_requeued",
            "stage1_victim_requeued",
            "stage0_rescue_peer_discovered",
            "stage1_rescue_peer_discovered",
            "coordinator_to_p2p_transition_ready",
            "coordinator_result_fallback_ready",
        ],
        "rescue_probe": {
            "schema": "p2p_stage_rescue_probe_v1",
            "ok": True,
            "results": {
                "stage0": {"ok": True, "matched_before": "p2p-v06-victim-stage0", "matched_after": "p2p-v06-rescue-stage0"},
                "stage1": {"ok": True, "matched_before": "p2p-v06-victim-stage1", "matched_after": "p2p-v06-rescue-stage1"},
            },
            "diagnosis_codes": ["p2p_stage_rescue_ready", "stage0_rescue_peer_discovered", "stage1_rescue_peer_discovered"],
        },
        "real_generate_probe": {
            "schema": "p2p_real_generate_probe_v1",
            "ok": True,
            "route": {"usable_now": True, "route_source": "p2p-discovery"},
            "generation": {"generated_token_count": 2, "multi_token_generation_ready": True},
            "stage_assignment": {"distinct_stage_miners": True, "completed_rows": 4},
            "kv_cache": {
                "schema": "p2p_real_generate_dual_stage_kv_cache_v1",
                "ready": True,
                "expected_hit_count_per_stage": 1,
                "stage0": {"ready": True, "ready_count": 2, "hit_count": 1},
                "stage1": {"ready": True, "ready_count": 2, "hit_count": 1},
                "raw_activations_public": False,
                "raw_token_inputs_public": False,
            },
            "diagnosis_codes": [
                "p2p_real_generate_ready",
                "p2p_real_generate_kv_cache_ready",
                "p2p_generate_route_ready",
                "tiny_gpt2_multi_token_ready",
                "real_llm_stage0_kv_cache_v1_ready",
                "real_llm_stage1_kv_cache_v1_ready",
                "stage0_kv_cache_hits_ready",
                "stage1_kv_cache_hits_ready",
            ],
        },
        "real_stage_rescue_probe": {
            "schema": "p2p_real_stage_rescue_probe_v1",
            "ok": True,
            "results": {
                "stage0": {
                    "schema": "p2p_real_stage_rescue_case_v1",
                    "ok": True,
                    "failure_stage": "stage0",
                    "generation": {"generated_token_count": 1, "multi_token_generation_ready": True},
                    "requeue_summary": {
                        "victim_peer_id": "p2p-v06-rescue-stage0-victim",
                        "rescue_peer_id": "p2p-v06-rescue-stage0-rescue",
                        "lease_expired": True,
                        "victim_task_requeued": True,
                        "rescued_result": True,
                        "victim_result_accepted": False,
                        "victim_process_failed": True,
                    },
                    "diagnosis_codes": [
                        "p2p_real_stage_rescue_ready",
                        "p2p_rescue_generation_completed",
                        "stage0_rescue_generation_completed",
                        "stage0_victim_requeued",
                    ],
                },
                "stage1": {
                    "schema": "p2p_real_stage_rescue_case_v1",
                    "ok": True,
                    "failure_stage": "stage1",
                    "generation": {"generated_token_count": 1, "multi_token_generation_ready": True},
                    "requeue_summary": {
                        "victim_peer_id": "p2p-v06-rescue-stage1-victim",
                        "rescue_peer_id": "p2p-v06-rescue-stage1-rescue",
                        "lease_expired": True,
                        "victim_task_requeued": True,
                        "rescued_result": True,
                        "victim_result_accepted": False,
                        "victim_process_failed": True,
                    },
                    "diagnosis_codes": [
                        "p2p_real_stage_rescue_ready",
                        "p2p_rescue_generation_completed",
                        "stage1_rescue_generation_completed",
                        "stage1_victim_requeued",
                    ],
                },
            },
            "diagnosis_codes": [
                "p2p_real_stage_rescue_ready",
                "p2p_rescue_generation_completed",
                "stage0_rescue_generation_completed",
                "stage1_rescue_generation_completed",
                "stage0_victim_requeued",
                "stage1_victim_requeued",
            ],
        },
    }


def validate_payload(payload: dict[str, Any], *, mode: str, expected_model_id: str) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if payload.get("ok") is not True:
        errors.append("v06_not_ready")
    if payload.get("mode") != mode:
        errors.append("mode_mismatch")
    codes = set(payload.get("diagnosis_codes") or [])
    for code in sorted(REQUIRED_CODES - codes):
        errors.append(f"missing_code:{code}")
    if mode == pack.MODE_LOCAL_SMOKE:
        for code in ["local_three_process_p2p_discovery_ready", "p2p_stage_discovery_ready", "p2p_generate_route_ready", "p2p_stage_rescue_ready"]:
            if code not in codes:
                errors.append(f"missing_code:{code}")
    if mode == pack.MODE_PACKAGE and "p2p_two_machine_kaggle_runbook_ready" not in codes:
        errors.append("missing_code:p2p_two_machine_kaggle_runbook_ready")
    if mode == pack.MODE_EXTERNAL_EXISTING:
        for code in ["external_existing_p2p_verified", "external_p2p_runtime_verified", "external_p2p_route_ready", "external_p2p_stage_discovery_ready"]:
            if code not in codes:
                errors.append(f"missing_code:{code}")
    if mode == pack.MODE_KAGGLE_AUTO:
        for code in ["p2p_swarm_inference_v06_kaggle_auto_ready", "external_existing_p2p_verified", "external_p2p_runtime_verified", "external_p2p_generate_verified", "kaggle_kernels_deleted"]:
            if code not in codes:
                errors.append(f"missing_code:{code}")
    p2p = payload.get("p2p") if isinstance(payload.get("p2p"), dict) else {}
    if mode == pack.MODE_LOCAL_SMOKE and p2p.get("ready") is not True:
        errors.append("p2p_not_ready")
    if mode != pack.MODE_PACKAGE:
        if p2p.get("hf_model_id") != expected_model_id:
            errors.append("p2p_requested_model_id_mismatch")
        if p2p.get("observed_hf_model_id") != expected_model_id:
            errors.append("p2p_observed_model_id_mismatch")
        if not p2p.get("observed_hf_model_id") or p2p.get("model_id_present") is not True:
            errors.append("p2p_model_id_missing")
        if p2p.get("model_id_match") is not True:
            errors.append("p2p_model_id_mismatch")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    for key in [
        "p2p_discovery_routing_prototype",
        "coordinator_result_fallback",
        "not_production",
        "not_nat_traversal",
        "not_decentralized_security",
        "not_economic_system",
        "not_large_model_throughput",
    ]:
        if safety.get(key) is not True:
            errors.append(f"safety_missing:{key}")
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
    prompt_scope = payload.get("prompt_scope") if isinstance(payload.get("prompt_scope"), dict) else {}
    if prompt_scope.get("source") not in {"prompt-text", "prompt-texts", "prompt-texts-file", "none"}:
        errors.append("prompt_scope_source_mismatch")
    if not isinstance(prompt_scope.get("prompt_count"), int) or prompt_scope.get("prompt_count") < 0:
        errors.append("prompt_scope_count_mismatch")
    inline_prompt_text = prompt_scope.get("source") in {"prompt-text", "prompt-texts"}
    if prompt_scope.get("source") == "none" and prompt_scope.get("prompt_count") != 0:
        errors.append("prompt_scope_none_count_mismatch")
    if inline_prompt_text and prompt_scope.get("prompt_count") < 1:
        errors.append("prompt_scope_inline_count_mismatch")
    if prompt_scope.get("inline_prompt_text") is not inline_prompt_text:
        errors.append("prompt_scope_inline_prompt_text_mismatch")
    if prompt_scope.get("terminal_next_commands_local_private") is not inline_prompt_text:
        errors.append("prompt_scope_terminal_next_commands_mismatch")
    if prompt_scope.get("terminal_logs_local_private") is not inline_prompt_text:
        errors.append("prompt_scope_terminal_logs_mismatch")
    if prompt_scope.get("saved_artifacts_prompt_placeholders") is not True:
        errors.append("prompt_scope_saved_placeholders_mismatch")
    if prompt_scope.get("saved_artifacts_public_safe") is not True:
        errors.append("prompt_scope_saved_artifacts_public_safe_mismatch")
    if prompt_scope.get("prefer_prompt_file_or_stdin_for_shareable_logs") is not inline_prompt_text:
        errors.append("prompt_scope_shareable_log_guidance_mismatch")
    if prompt_scope.get("prompt_file_path_public") is not False:
        errors.append("prompt_scope_file_path_public_mismatch")
    if prompt_scope.get("raw_prompt_public") is not False:
        errors.append("prompt_scope_raw_prompt_public_mismatch")
    if prompt_scope.get("public_artifact_safe") is not True:
        errors.append("prompt_scope_public_artifact_safe_mismatch")
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
    if "crowdtensor.cli p2pd" in joined:
        return completed({"schema": "p2pd_cli_v1", "ok": True, "diagnosis_codes": ["p2pd_command_ready"]})
    if "crowdtensor.cli serve" in joined:
        return completed({"schema": "public_swarm_product_cli_v1", "ok": True, "diagnosis_codes": ["serve_command_ready", "p2p_coordinator_announce_ready"]})
    if "crowdtensor.cli join" in joined:
        return completed({"schema": "public_swarm_product_cli_v1", "ok": True, "diagnosis_codes": ["join_command_ready", "p2p_stage_miner_announce_ready"]})
    if "crowdtensor.cli generate" in joined:
        return completed({
            "schema": "public_swarm_product_cli_v1",
            "ok": True,
            "diagnosis_codes": ["p2p_generate_route_ready"],
            "route": {"usable_now": True, "route_source": "p2p-discovery"},
        })
    return completed({"schema": "unknown", "ok": True})


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="crowdtensor_p2p_v06_check_")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_report = output_dir / "preview_v04.json"
    product_report = output_dir / "product_mvp.json"
    optional_report = output_dir / "optional_mvp.json"
    local_p2p_report = output_dir / "local_p2p.json"
    write_json(preview_report, fake_preview_payload())
    write_json(product_report, fake_product_payload())
    write_json(optional_report, fake_product_payload(args.hf_model_id))
    write_json(local_p2p_report, fake_local_p2p(args.hf_model_id))

    parsed = pack.parse_args([
        args.mode,
        "--output-dir",
        str(output_dir / "v06"),
        "--preview-v04-report",
        str(preview_report),
        "--product-mvp-report",
        str(product_report),
        "--optional-model-report",
        str(optional_report),
        "--p2p-discovery-report",
        str(local_p2p_report),
        "--peer-bootstrap",
        "http://p2p.example",
        "--max-new-tokens",
        "2",
        "--hf-model-id",
        args.hf_model_id,
        "--json",
    ])
    if args.mode == pack.MODE_LOCAL_SMOKE:
        original = pack.run_local_p2p_discovery
        try:
            pack.run_local_p2p_discovery = lambda _args, output_dir, runner: (fake_local_p2p(args.hf_model_id), [], {})  # type: ignore[assignment]
            report = pack.build_report(parsed, runner=fake_runner)
        finally:
            pack.run_local_p2p_discovery = original  # type: ignore[assignment]
    elif args.mode == pack.MODE_EXTERNAL_EXISTING:
        original = pack.run_external_existing_probe
        try:
            pack.run_external_existing_probe = lambda _args, output_dir: (  # type: ignore[assignment]
                {
                    **fake_local_p2p(args.hf_model_id),
                    "schema": "p2p_external_existing_probe_v1",
                    "external_runtime_verified": True,
                    "external_generate_verified": False,
                    "diagnosis_codes": fake_local_p2p(args.hf_model_id)["diagnosis_codes"] + [
                        "external_p2p_runtime_verified",
                        "external_p2p_route_ready",
                        "external_p2p_stage_discovery_ready",
                        "external_p2p_generate_not_requested",
                    ],
                },
                [],
                {},
            )
            report = pack.build_report(parsed, runner=fake_runner)
        finally:
            pack.run_external_existing_probe = original  # type: ignore[assignment]
    elif args.mode == pack.MODE_KAGGLE_AUTO:
        original = pack.run_kaggle_auto
        try:
            def fake_kaggle_auto(_args: object, output_dir: Path) -> dict[str, Any]:
                payload = fake_local_p2p(args.hf_model_id)
                return {
                    **payload,
                    "schema": "p2p_swarm_inference_v06_kaggle_auto_v1",
                    "external_runtime_verified": True,
                    "external_generate_verified": True,
                    "kaggle_lifecycle": {"kernels_deleted": True},
                    "diagnosis_codes": payload["diagnosis_codes"] + [
                        "p2p_swarm_inference_v06_kaggle_auto_ready",
                        "external_existing_p2p_verified",
                        "external_p2p_runtime_verified",
                        "external_p2p_generate_verified",
                        "kaggle_kernels_deleted",
                    ],
                }

            pack.run_kaggle_auto = fake_kaggle_auto  # type: ignore[assignment]
            report = pack.build_report(parsed, runner=fake_runner)
        finally:
            pack.run_kaggle_auto = original  # type: ignore[assignment]
    else:
        report = pack.build_report(parsed, runner=fake_runner)
    errors = validate_payload(report, mode=args.mode, expected_model_id=args.hf_model_id)
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "mode": args.mode,
        "hf_model_id": args.hf_model_id,
        "output_dir": str(output_dir),
        "errors": errors,
        "p2p_swarm_inference_v06_schema": report.get("schema"),
        "p2p_swarm_inference_v06_ok": report.get("ok"),
        "diagnosis_codes": ["p2p_swarm_inference_v06_check_ready"] if not errors else ["p2p_swarm_inference_v06_check_failed"],
        "artifacts": {
            "p2p_swarm_inference_v06_json": str(output_dir / "v06" / "p2p_swarm_inference_v06.json"),
            "p2p_swarm_inference_v06_markdown": str(output_dir / "v06" / "p2p_swarm_inference_v06.md"),
        },
    }
    write_json(output_dir / "p2p_swarm_inference_v06_check.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate P2P Swarm Inference v0.6.")
    parser.add_argument("--mode", choices=pack.MODES, default=pack.MODE_EVIDENCE_IMPORT)
    parser.add_argument("--hf-model-id", default=pack.DEFAULT_HF_MODEL_ID)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    report = run_check(parse_args())
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
