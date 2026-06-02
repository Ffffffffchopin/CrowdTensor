#!/usr/bin/env python3
"""Acceptance checks for Usable Swarm Inference v1."""

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

import usable_swarm_inference_pack as pack  # noqa: E402


SCHEMA = "usable_swarm_inference_check_v1"
REQUIRED_CODES = {
    "usable_swarm_inference_ready",
    "usable_swarm_inference_v1_ready",
    "usable_p2p_route_ready",
    "usable_real_llm_generate_ready",
    "usable_real_llm_kv_cache_ready",
    "usable_multi_token_generation_ready",
    "usable_distinct_stage_miners_ready",
    "usable_stage_requeue_rescue_ready",
    "usable_swarm_model_match_ready",
    "serve_join_generate_p2p_primary_path",
    "read_only_workload",
    "not_production",
    "not_coordinator_free",
    "not_hivemind_petals_production",
    "not_large_model_serving",
}
SECRET_FRAGMENTS = [
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "CROWDTENSOR_P2P_PEER_SECRET=",
    "lease_token",
    "idempotency_key",
    "Bearer ",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
]


def completed(payload: dict[str, Any], returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


def fake_p2p_v06_payload(
    *,
    generated_tokens: int = 8,
    accepted_rows: int = 16,
    model_id: str = pack.DEFAULT_HF_MODEL_ID,
) -> dict[str, Any]:
    return {
        "schema": pack.P2P_V06_SCHEMA,
        "ok": True,
        "mode": "local-smoke",
        "hf_model_id": model_id,
        "p2p": {
            "ready": True,
            "hf_model_id": model_id,
            "observed_hf_model_id": model_id,
            "model_id_present": True,
            "model_id_match": True,
            "p2p_url": "http://127.0.0.1:9788",
            "catalog_peer_count": 4,
            "coordinator_peer_count": 1,
            "stage0_peer_count": 1,
            "stage1_peer_count": 1,
            "real_generate_ready": True,
            "kv_cache_ready": True,
            "kv_cache": {
                "schema": "p2p_real_generate_dual_stage_kv_cache_v1",
                "ready": True,
                "expected_hit_count_per_stage": max(0, generated_tokens - 1),
                "stage0": {"ready": True, "ready_count": generated_tokens, "hit_count": max(0, generated_tokens - 1)},
                "stage1": {"ready": True, "ready_count": generated_tokens, "hit_count": max(0, generated_tokens - 1)},
                "raw_activations_public": False,
                "raw_token_inputs_public": False,
            },
            "stage_rescue_ready": True,
            "real_stage_rescue_ready": True,
            "generate_route": {
                "schema": "session_route_decision_v1",
                "workload_type": pack.WORKLOAD_TYPE,
                "route_source": "p2p-discovery",
                "coordinator_url_present": True,
                "usable_now": True,
                "missing_capabilities": [],
                "matched_capabilities": {
                    "real_llm_sharded_stage0": "usable-stage0",
                    "real_llm_sharded_stage1": "usable-stage1",
                },
            },
        },
        "payload_summaries": {
            "local_p2p_discovery": {
                "rescue_probe": {"ok": True, "diagnosis_codes": ["p2p_stage_rescue_ready"]},
                "real_stage_rescue_probe": {
                    "ok": True,
                    "diagnosis_codes": [
                        "p2p_real_stage_rescue_ready",
                        "p2p_rescue_generation_completed",
                        "stage0_rescue_generation_completed",
                        "stage1_rescue_generation_completed",
                    ],
                },
                "real_generate_probe": {
                    "schema": "p2p_real_generate_probe_v1",
                    "ok": True,
                    "generation": {
                        "schema": "session_result_summary_v1",
                        "generated_token_count": generated_tokens,
                        "max_new_tokens": generated_tokens,
                        "generated_text_hash": "sha256:fake-generated",
                        "decoded_tokens_match": True,
                        "multi_token_generation_ready": True,
                        "raw_generated_text_public": False,
                        "generated_token_ids_public": False,
                    },
                    "stage_assignment": {
                        "distinct_stage_miners": True,
                        "stage0_miner_id": "usable-stage0",
                        "stage1_miner_id": "usable-stage1",
                        "completed_rows": accepted_rows,
                    },
                    "ledger": {"accepted_rows": accepted_rows, "error": ""},
                    "kv_cache": {
                        "schema": "p2p_real_generate_dual_stage_kv_cache_v1",
                        "ready": True,
                        "expected_hit_count_per_stage": max(0, generated_tokens - 1),
                        "stage0": {"ready": True, "ready_count": generated_tokens, "hit_count": max(0, generated_tokens - 1)},
                        "stage1": {"ready": True, "ready_count": generated_tokens, "hit_count": max(0, generated_tokens - 1)},
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
                        "distinct_stage_miners",
                        "stage_assignment_valid",
                    ],
                },
            }
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
            "coordinator_to_p2p_transition_ready",
            "coordinator_result_fallback_ready",
            "distinct_stage_miners",
            "stage_assignment_valid",
        ],
    }


def validate_report(
    payload: dict[str, Any],
    *,
    mode: str,
    expected_model_id: str,
    required_tokens: int = 8,
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if payload.get("ok") is not True:
        errors.append("usable_swarm_not_ready")
    if payload.get("mode") != mode:
        errors.append("mode_mismatch")
    usable = payload.get("usable_swarm") if isinstance(payload.get("usable_swarm"), dict) else {}
    if usable.get("ready") is not True:
        errors.append("usable_summary_not_ready")
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    p2p = readiness.get("p2p_product_path") if isinstance(readiness.get("p2p_product_path"), dict) else {}
    for key in [
        "route_ready",
        "p2p_counts_ready",
        "real_generate_ready",
        "kv_cache_ready",
        "generation_target_ready",
        "accepted_rows_ready",
        "distinct_stage_miners",
        "stage_rescue_ready",
        "real_stage_rescue_ready",
    ]:
        if p2p.get(key) is not True:
            errors.append(f"p2p_{key}_missing")
    if int(p2p.get("generated_token_count") or 0) < required_tokens:
        errors.append(f"generated_token_count_below_{required_tokens}")
    if int(p2p.get("accepted_rows") or 0) < required_tokens * 2:
        errors.append(f"accepted_rows_below_{required_tokens * 2}")
    if p2p.get("usable_evidence_source") not in {"source_gate", "nested_goal_evidence"}:
        errors.append("usable_evidence_source_missing")
    if p2p.get("route_source") != "p2p-discovery":
        errors.append("usable_route_source_not_p2p_discovery")
    model = p2p.get("model") if isinstance(p2p.get("model"), dict) else {}
    if model.get("expected_hf_model_id") != expected_model_id:
        errors.append("usable_expected_model_id_mismatch")
    if model.get("observed_hf_model_id") != expected_model_id:
        errors.append("usable_observed_model_id_mismatch")
    if not model.get("observed_hf_model_id") or model.get("model_id_present") is not True:
        errors.append("usable_model_id_missing")
    if model.get("model_id_match") is not True or model.get("compatible") is not True:
        errors.append("usable_model_id_mismatch")
    codes = set(payload.get("diagnosis_codes") or [])
    for code in sorted(REQUIRED_CODES - codes):
        errors.append(f"missing_code:{code}")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    for key in [
        "p2p_discovery_primary_path",
        "raw_prompt_public",
        "raw_generated_text_public",
        "generated_token_ids_public",
        "activation_payloads_redacted",
        "not_production",
        "not_coordinator_free",
        "not_hivemind_petals_production",
        "not_large_model_serving",
    ]:
        expected = False if key in {"raw_prompt_public", "raw_generated_text_public", "generated_token_ids_public"} else True
        if safety.get(key) is not expected:
            errors.append(f"safety_{key}_mismatch")
    if safety.get("read_only_workload") != pack.WORKLOAD_TYPE:
        errors.append("safety_read_only_workload_mismatch")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    for name in ["usable_swarm_inference_json", "usable_swarm_inference_markdown", "support_bundle_json", "runbook"]:
        artifact = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
        if artifact.get("present") is not True:
            errors.append(f"artifact_missing:{name}")
    return sorted(set(errors))


def local_pack_args(args: argparse.Namespace, output_dir: Path) -> list[str]:
    argv = [
        pack.MODE_LOCAL,
        "--output-dir",
        str(output_dir / "usable"),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--hf-model-id",
        args.hf_model_id,
        "--p2p-port",
        str(args.p2p_port),
        "--coordinator-port",
        str(args.coordinator_port),
        "--startup-timeout",
        str(args.startup_timeout),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--http-timeout",
        str(args.http_timeout),
    ]
    if args.prompt_texts:
        argv.extend(["--prompt-texts", args.prompt_texts])
    if args.stream_generation:
        argv.append("--stream-generation")
    if args.hf_cache_dir:
        argv.extend(["--hf-cache-dir", args.hf_cache_dir])
    return argv


def run_check(args: argparse.Namespace, *, runner: pack.Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    real_local = bool(args.real_local and args.mode == pack.MODE_LOCAL)

    if args.mode == pack.MODE_LOCAL:
        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if "p2p_swarm_inference_v06_pack.py" not in command[1]:
                raise AssertionError(command)
            if "--max-new-tokens" not in command:
                raise AssertionError(command)
            if command[command.index("--max-new-tokens") + 1] != str(args.max_new_tokens):
                raise AssertionError(command)
            if "--hf-model-id" not in command:
                raise AssertionError(command)
            if command[command.index("--hf-model-id") + 1] != args.hf_model_id:
                raise AssertionError(command)
            return completed(fake_p2p_v06_payload(generated_tokens=args.max_new_tokens, accepted_rows=args.max_new_tokens * 2, model_id=args.hf_model_id))

        report = pack.build_report(
            pack.parse_args(local_pack_args(args, output_dir)),
            runner=runner if real_local else fake_runner,
        )
    elif args.mode == pack.MODE_PACKAGE:
        report = pack.build_report(pack.parse_args([
            pack.MODE_PACKAGE,
            "--output-dir",
            str(output_dir / "usable"),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--hf-model-id",
            args.hf_model_id,
        ]))
    else:
        source = output_dir / "source" / "p2p_v06.json"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            json.dumps(fake_p2p_v06_payload(generated_tokens=args.max_new_tokens, accepted_rows=args.max_new_tokens * 2, model_id=args.hf_model_id)) + "\n",
            encoding="utf-8",
        )
        report = pack.build_report(pack.parse_args([
            pack.MODE_EVIDENCE_IMPORT,
            "--output-dir",
            str(output_dir / "usable"),
            "--p2p-report",
            str(source),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--hf-model-id",
            args.hf_model_id,
        ]))

    errors = validate_report(
        report,
        mode=args.mode,
        expected_model_id=args.hf_model_id,
        required_tokens=args.max_new_tokens,
    )
    check = {
        "schema": SCHEMA,
        "ok": not errors,
        "mode": args.mode,
        "real_local": real_local,
        "hf_model_id": args.hf_model_id,
        "max_new_tokens": args.max_new_tokens,
        "output_dir": str(output_dir),
        "errors": errors,
        "diagnosis_codes": ["usable_swarm_inference_check_ready"] if not errors else ["usable_swarm_inference_check_failed"],
    }
    (output_dir / "usable_swarm_inference_check.json").write_text(json.dumps(check, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return check


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Usable Swarm Inference v1.")
    parser.add_argument("--mode", choices=pack.MODES, default=pack.MODE_LOCAL)
    parser.add_argument("--real-local", action="store_true", help="run the real local p2pd/serve/join/generate smoke instead of the fast fake-runner contract")
    parser.add_argument("--hf-model-id", default=pack.DEFAULT_HF_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--prompt-texts", default="")
    parser.add_argument("--stream-generation", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--p2p-port", type=int, default=9788)
    parser.add_argument("--coordinator-port", type=int, default=9789)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--http-timeout", type=float, default=10.0)
    parser.add_argument("--output-dir", default=str(Path(tempfile.gettempdir()) / "crowdtensor_usable_swarm_check"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.real_local and args.mode != pack.MODE_LOCAL:
        raise SystemExit("--real-local is only valid with --mode local")
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    if args.p2p_port < 1 or args.coordinator_port < 1:
        raise SystemExit("--p2p-port and --coordinator-port must be positive")
    for name in ["startup_timeout", "timeout_seconds", "http_timeout"]:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = run_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"Usable Swarm Inference check ok={result.get('ok')} errors={','.join(result.get('errors') or [])}")
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
