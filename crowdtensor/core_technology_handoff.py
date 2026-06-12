"""Core technology handoff RC helpers.

This layer aggregates the Large-Model Shard Alpha and Inference RC evidence into
a stable handoff package for the control, user-facing, and operator/economics
layers.  It does not broaden the runtime claim beyond the evidence supplied by
the Inference RC.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crowdtensor import large_model_inference_rc as inference_rc
from crowdtensor import large_model_shard as alpha


HANDOFF_SCHEMA = "core_technology_handoff_rc_v1"
HANDOFF_SUPPORT_BUNDLE_SCHEMA = "core_technology_handoff_rc_support_bundle_v1"
HANDOFF_CHECK_SCHEMA = "core_technology_handoff_rc_check_v1"
DEPLOYMENT_RUNBOOK_SCHEMA = "core_technology_deployment_runbook_v1"
NEXT_LAYER_CONTRACT_SCHEMA = "core_technology_next_layer_contract_v1"
ADAPTER_CONFORMANCE_SCHEMA = "core_technology_adapter_conformance_v1"
TEST_GATE_SCHEMA = "core_technology_test_gate_summary_v1"


def stable_hash_payload(value: Any) -> str:
    return alpha.stable_hash_payload(value)


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    return alpha.artifact_entry(path, output_dir, kind=kind, schema=schema, ok=ok)


def artifact_summary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    count = len(artifacts)
    present = sum(1 for item in artifacts.values() if isinstance(item, dict) and item.get("present"))
    return {
        "schema": "core_technology_handoff_rc_artifact_summary_v1",
        "artifact_count": count,
        "present_artifact_count": present,
        "public_artifact_safe": True,
        "support_bundle": artifacts.get("support_bundle_json", {}).get("path") if artifacts else "",
        "inspect_first": artifacts.get("summary_markdown", {}).get("path") if artifacts else "",
    }


def public_redaction_errors(value: Any) -> list[str]:
    return alpha.public_redaction_errors(value)


def build_deployment_runbook(*, output_dir: Path, inference_report: dict[str, Any]) -> dict[str, Any]:
    runner = inference_report.get("runner_result") if isinstance(inference_report.get("runner_result"), dict) else {}
    runtime_probe = inference_report.get("runtime_adapter_probe") if isinstance(inference_report.get("runtime_adapter_probe"), dict) else {}
    blockers = list(inference_report.get("blockers") or [])
    max_new_tokens = int(runner.get("max_new_tokens") or inference_rc.DEFAULT_RC_MAX_NEW_TOKENS)
    runbook = {
        "schema": DEPLOYMENT_RUNBOOK_SCHEMA,
        "output_dir": str(output_dir),
        "ready": True,
        "local_fixture": {
            "command": "crowdtensor large-model-shard-rc --mode fixture --json",
            "ci_safe": True,
            "real_runtime_verified": False,
            "purpose": "Validate contracts, artifacts, redaction, planner, runner, benchmark, and serving hooks without a GGUF runtime.",
        },
        "local_real_runtime": {
            "command": (
                "crowdtensor large-model-shard-rc --mode real "
                "--model-path /models/llama-7b.Q4_K_M.gguf "
                "--rpc-endpoint http://127.0.0.1:50052 "
                f"--max-new-tokens {min(max_new_tokens, inference_rc.MAX_REAL_RUN_TOKENS)} --real-timeout-seconds 1200 --json"
            ),
            "requires": [
                "local GGUF model file",
                "llama-cli on PATH",
                "reachable controlled llama.cpp RPC worker or --start-workers setup",
                "enough CPU/RAM/GPU/VRAM for the selected partition",
            ],
            "timeout_seconds_max": inference_rc.MAX_REAL_RUN_TIMEOUT_SECONDS,
            "max_new_tokens_max": inference_rc.MAX_REAL_RUN_TOKENS,
        },
        "lan_vpn_two_worker_runtime": {
            "worker_commands": [
                item.get("command_line")
                for item in (inference_report.get("runtime_adapter") or {}).get("worker_commands", [])
                if isinstance(item, dict) and item.get("command_line")
            ],
            "client_command": (inference_report.get("runtime_adapter") or {}).get("client_command_line", ""),
            "controlled_network_only": True,
            "not_public_rpc_safe": True,
        },
        "import_retained_evidence": {
            "command": (
                "crowdtensor large-model-shard-rc "
                "--real-run-report /secure/private/large_model_real_run.json "
                "--real-benchmark-report /secure/private/large_model_benchmark.json --json"
            ),
            "real_run_required_fields": [
                "ttft_ms",
                "tokens_per_second",
                "wall_time_seconds",
                "generated_token_count",
                "output_digest",
            ],
            "benchmark_import_supplements_metrics_only": True,
        },
        "troubleshooting": {
            "blockers": blockers,
            "runtime_probe_codes": runtime_probe.get("diagnosis_codes") or [],
            "operator_actions": [
                "Install llama.cpp client/server binaries or pass their explicit paths.",
                "Provide a local GGUF model path; do not auto-download large models.",
                "Start controlled local/LAN/VPN RPC workers and verify endpoint reachability.",
                "Import a public-safe real-run report when external runtime proof already exists.",
            ],
        },
        "cleanup": {
            "process_leak_check": "ps -eo pid,comm,args | rg -i 'llama|rpc-server|large_model_inference|large-model-shard-rc' || true",
            "clean_artifacts": "crowdtensor clean-artifacts --dry-run",
            "runtime_processes_started_by_runner_must_be_terminated": True,
        },
        "diagnosis_codes": ["core_technology_deployment_runbook_ready"],
    }
    runbook["runbook_hash"] = stable_hash_payload(runbook)
    return runbook


def build_next_layer_contract(*, inference_report: dict[str, Any]) -> dict[str, Any]:
    serving = inference_report.get("serving_readiness_hooks") if isinstance(inference_report.get("serving_readiness_hooks"), dict) else {}
    partition = inference_report.get("partition_manifest") if isinstance(inference_report.get("partition_manifest"), dict) else {}
    benchmark = inference_report.get("benchmark") if isinstance(inference_report.get("benchmark"), dict) else {}
    correctness = inference_report.get("correctness_summary") if isinstance(inference_report.get("correctness_summary"), dict) else {}
    contract = {
        "schema": NEXT_LAYER_CONTRACT_SCHEMA,
        "ready": True,
        "control_layer": {
            "stable_entrypoints": [
                "crowdtensor large-model-shard-rc",
                "scripts/large_model_inference_rc_pack.py",
                "scripts/core_technology_handoff_pack.py",
            ],
            "route_health_schema": serving.get("health_aware_route_metadata_schema"),
            "runner_result_schema": inference_rc.RUNNER_RESULT_SCHEMA,
            "blocker_codes_source": "core_technology_handoff_rc_v1.blockers",
            "schedule_inputs": [
                "partition_manifest.assignments",
                "partition_manifest.tensor_split_plan",
                "device_profile.devices",
                "runtime_adapter_probe.rpc_endpoint_health",
            ],
        },
        "user_layer": {
            "safe_status_fields": [
                "ok",
                "real_runtime_verified",
                "real_7b_runtime_verified",
                "mode",
                "diagnosis_codes",
                "blockers",
            ],
            "streaming_event_schema": serving.get("streaming_event_schema"),
            "bounded_batch_request_schema": serving.get("bounded_batch_request_schema"),
            "answer_visibility": "public artifacts expose digests and readiness only; local generated text belongs to a human runtime command.",
        },
        "permissions_trust_billing_layer": {
            "core_signals": [
                "runtime_backend",
                "model_id",
                "partition_hash",
                "runner_result.real_runtime_verified",
                "benchmark.tokens_per_second",
                "benchmark.wall_time_seconds",
                "correctness_summary.output_digest",
                "route_health.healthy",
                "process_cleanup.completed",
            ],
            "not_implemented_here": [
                "accounts",
                "billing",
                "trust scores",
                "incentives",
                "staking",
                "slashing",
            ],
        },
        "sample_control_request": {
            "schema": "core_technology_control_request_v1",
            "mode": inference_report.get("mode"),
            "partition_hash": partition.get("partition_hash"),
            "max_new_tokens": (serving.get("bounded_batch_request") or {}).get("max_new_tokens"),
            "timeout_seconds": (serving.get("bounded_batch_request") or {}).get("timeout_seconds"),
            "cancel_requested": False,
            "raw_prompt_public": False,
        },
        "performance_contract": {
            "measurement_kind": benchmark.get("measurement_kind"),
            "real_runtime_verified": bool(benchmark.get("real_runtime_verified")),
            "ttft_ms": benchmark.get("ttft_ms"),
            "tokens_per_second": benchmark.get("tokens_per_second"),
            "wall_time_seconds": benchmark.get("wall_time_seconds"),
        },
        "correctness_contract": {
            "generated_token_count": correctness.get("generated_token_count"),
            "output_digest": correctness.get("output_digest"),
            "baseline_comparison": correctness.get("baseline_comparison"),
            "generated_token_ids_public": False,
        },
        "diagnosis_codes": ["core_technology_next_layer_contract_ready"],
    }
    contract["contract_hash"] = stable_hash_payload(contract)
    return contract


def build_adapter_conformance(*, inference_report: dict[str, Any]) -> dict[str, Any]:
    adapter_interface = inference_report.get("adapter_interface") if isinstance(inference_report.get("adapter_interface"), dict) else {}
    descriptors = adapter_interface.get("descriptors") if isinstance(adapter_interface.get("descriptors"), list) else []
    descriptor_checks = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            continue
        status = descriptor.get("status")
        descriptor_checks.append({
            "adapter_kind": descriptor.get("adapter_kind"),
            "status": status,
            "has_capability_contract": bool(descriptor.get("capabilities") or descriptor.get("operator_action")),
            "unsupported_diagnostic_ready": bool(status == "supported" or "unsupported_runtime_backend" in (descriptor.get("diagnosis_codes") or [])),
            "conformant": bool(
                descriptor.get("adapter_kind")
                and status in {"supported", "unsupported"}
                and (descriptor.get("capabilities") or descriptor.get("operator_action"))
                and (status == "supported" or "unsupported_runtime_backend" in (descriptor.get("diagnosis_codes") or []))
            ),
        })
    conformance = {
        "schema": ADAPTER_CONFORMANCE_SCHEMA,
        "ready": bool(descriptor_checks and all(item.get("conformant") for item in descriptor_checks)),
        "selected_runtime_backend": adapter_interface.get("selected_runtime_backend"),
        "selected_supported": bool(adapter_interface.get("selected_supported")),
        "descriptor_checks": descriptor_checks,
        "future_runtime_backends": [
            item.get("adapter_kind")
            for item in descriptor_checks
            if item.get("status") == "unsupported"
        ],
        "diagnosis_codes": ["core_technology_adapter_conformance_ready"]
        + ([] if descriptor_checks and all(item.get("conformant") for item in descriptor_checks) else ["core_technology_adapter_conformance_failed"]),
    }
    conformance["conformance_hash"] = stable_hash_payload(conformance)
    return conformance


def build_test_gate_summary(*, mode: str, full_pytest: bool = False) -> dict[str, Any]:
    commands = [
        "python -m py_compile crowdtensor/core_technology_handoff.py scripts/core_technology_handoff_pack.py scripts/core_technology_handoff_check.py",
        "python scripts/core_technology_handoff_check.py --mode fixture --json",
        "python scripts/large_model_inference_rc_check.py --mode fixture --json",
        "python -m pytest tests/test_core_technology_handoff.py tests/test_large_model_inference_rc.py tests/test_large_model_shard_alpha.py -q",
    ]
    if full_pytest:
        commands.append("python -m pytest -q")
    summary = {
        "schema": TEST_GATE_SCHEMA,
        "ready": True,
        "mode": mode,
        "ci_safe": True,
        "commands": commands,
        "full_pytest_requested": bool(full_pytest),
        "coverage": [
            "CLI/API entry",
            "adapter interface",
            "unsupported adapters",
            "device profile import/export",
            "planner v2",
            "runner plan/fixture/import",
            "real mode validation and timeout constraints",
            "benchmark v2",
            "correctness",
            "serving hooks",
            "deployment/runbook artifact generation",
            "aggregate handoff report",
            "redaction",
            "backward compatibility",
        ],
        "diagnosis_codes": ["core_technology_test_gates_ready"],
    }
    summary["test_gate_hash"] = stable_hash_payload(summary)
    return summary


def build_support_bundle(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": HANDOFF_SUPPORT_BUNDLE_SCHEMA,
        "report_schema": report.get("schema"),
        "ok": bool(report.get("ok")),
        "mode": report.get("mode"),
        "real_runtime_verified": bool(report.get("real_runtime_verified")),
        "real_7b_runtime_verified": bool(report.get("real_7b_runtime_verified")),
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "blockers": report.get("blockers") or [],
        "artifact_summary": report.get("artifact_summary"),
        "public_artifact_safe": bool((report.get("safety") or {}).get("public_artifact_safe")),
    }


def build_handoff_report(
    *,
    output_dir: Path,
    mode: str,
    inference_report: dict[str, Any],
    deployment_runbook: dict[str, Any],
    next_layer_contract: dict[str, Any],
    adapter_conformance: dict[str, Any],
    test_gate_summary: dict[str, Any],
) -> dict[str, Any]:
    real_verified = bool(inference_report.get("real_runtime_verified"))
    blockers = list(inference_report.get("blockers") or [])
    if not real_verified:
        for item in [
            "core_technology_real_7b_runtime_not_verified",
            "external_real_runtime_resources_required",
        ]:
            if item not in blockers:
                blockers.append(item)
    codes = [
        "core_technology_handoff_rc_ready",
        "core_technology_stable_entrypoint_ready",
        "core_technology_inference_rc_imported",
        "core_technology_deployment_runbook_ready",
        "core_technology_next_layer_contract_ready",
        "core_technology_adapter_conformance_ready",
        "core_technology_test_gates_ready",
        "core_technology_public_artifact_redaction_ready",
    ]
    for source in [inference_report, deployment_runbook, next_layer_contract, adapter_conformance, test_gate_summary]:
        codes.extend(source.get("diagnosis_codes") or [])
    if real_verified:
        codes.append("core_technology_real_runtime_verified")
    else:
        codes.extend([
            "core_technology_real_runtime_not_verified",
            "core_technology_handoff_fixture_or_import_ready",
        ])
    seen: set[str] = set()
    diagnosis_codes = [code for code in codes if not (code in seen or seen.add(code))]
    report = {
        "schema": HANDOFF_SCHEMA,
        "ok": bool(
            inference_report.get("ok")
            and deployment_runbook.get("ready")
            and next_layer_contract.get("ready")
            and adapter_conformance.get("ready")
            and test_gate_summary.get("ready")
        ),
        "mode": mode,
        "output_dir": str(output_dir),
        "stable_entrypoints": [
            "crowdtensor large-model-shard-rc",
            "crowdtensor core-tech-handoff",
            "scripts/core_technology_handoff_pack.py",
            "scripts/core_technology_handoff_check.py",
        ],
        "real_runtime_verified": real_verified,
        "real_7b_runtime_verified": bool(inference_report.get("real_7b_runtime_verified")),
        "capability_summary": {
            "can_plan_large_model_sharding": True,
            "can_run_ci_safe_fixture": True,
            "can_import_real_runtime_evidence": True,
            "can_attempt_controlled_real_runtime": True,
            "can_export_next_layer_contract": True,
            "can_support_control_layer_development": True,
            "can_support_user_layer_development": True,
            "can_support_permissions_trust_billing_layer_development": True,
            "requires_external_runtime_for_real_7b_claim": not bool(inference_report.get("real_7b_runtime_verified")),
        },
        "evidence_scope": "real-runtime" if real_verified else "fixture-diagnostic-handoff",
        "inference_rc_report": inference_report,
        "alpha_evidence": inference_report.get("alpha_report"),
        "adapter_interface": inference_report.get("adapter_interface"),
        "runtime_probe": inference_report.get("runtime_adapter_probe"),
        "device_profile": inference_report.get("device_profile"),
        "partition_planner": inference_report.get("partition_manifest"),
        "runner_result": inference_report.get("runner_result"),
        "benchmark": inference_report.get("benchmark"),
        "correctness_summary": inference_report.get("correctness_summary"),
        "serving_hooks": inference_report.get("serving_readiness_hooks"),
        "deployment_runbook": deployment_runbook,
        "next_layer_integration_contract": next_layer_contract,
        "adapter_conformance": adapter_conformance,
        "test_gate_summary": test_gate_summary,
        "blockers": [item for index, item in enumerate(blockers) if item and item not in blockers[:index]],
        "handoff_answers": {
            "what_core_can_do": [
                "Build CI-safe large-model sharding plans and evidence.",
                "Probe llama.cpp/GGUF/RPC runtime prerequisites.",
                "Profile devices from local probes or JSON imports.",
                "Plan layer placement, tensor split, KV reservation, and memory estimates.",
                "Run fixture/plan/real/import runner paths with benchmark and correctness evidence.",
                "Expose serving-readiness hooks and next-layer integration contracts.",
            ],
            "real_verified": bool(real_verified),
            "fixture_diagnostic_or_import": "fixture/diagnostic unless real runner or real-run import is supplied",
            "control_layer_call": "Use core_technology_handoff_rc_v1.next_layer_integration_contract.control_layer and stable_entrypoints.",
            "user_layer_call": "Use safe status fields, streaming event schema, and bounded batch schema; do not expose raw generated text from public artifacts.",
            "permissions_trust_billing_dependencies": next_layer_contract.get("permissions_trust_billing_layer", {}).get("core_signals", []),
            "external_runtime_future_work": [
                "Provide real GGUF model files and controlled llama.cpp/vLLM/SGLang/TensorRT/Petals-like runtimes.",
                "Run bounded 7B/13B/70B external proofs on real consumer devices.",
                "Optimize production throughput only after real runtime evidence exists.",
            ],
        },
        "boundary": {
            "core_technology_only": True,
            "inference_only": True,
            "not_training_or_finetuning": True,
            "not_permissions_accounts_billing": True,
            "not_incentives_staking_slashing": True,
            "not_public_p2p_nat_traversal": True,
            "not_production_petals_hivemind": True,
            "not_gpu_marketplace": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "kv_cache_public": False,
            "credentials_public": False,
            "lease_material_public": False,
            "idempotency_material_public": False,
        },
        "diagnosis_codes": diagnosis_codes,
    }
    report["handoff_hash"] = stable_hash_payload({
        "schema": report["schema"],
        "inference": inference_report.get("schema"),
        "contract": next_layer_contract.get("contract_hash"),
        "runbook": deployment_runbook.get("runbook_hash"),
        "adapter": adapter_conformance.get("conformance_hash"),
    })
    errors = public_redaction_errors(report)
    if errors:
        report["ok"] = False
        report.setdefault("errors", []).extend(errors)
        report["safety"]["public_artifact_safe"] = False
        if "core_technology_public_artifact_redaction_failed" not in report["diagnosis_codes"]:
            report["diagnosis_codes"].append("core_technology_public_artifact_redaction_failed")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CrowdTensor Core Technology Handoff RC",
        "",
        f"- Schema: `{report.get('schema')}`",
        f"- OK: `{bool(report.get('ok'))}`",
        f"- Mode: `{report.get('mode')}`",
        f"- Real runtime verified: `{bool(report.get('real_runtime_verified'))}`",
        f"- Real 7B runtime verified: `{bool(report.get('real_7b_runtime_verified'))}`",
        f"- Evidence scope: `{report.get('evidence_scope')}`",
        f"- Output: `{report.get('output_dir')}`",
        "",
        "## Stable Entrypoints",
        "",
    ]
    for item in report.get("stable_entrypoints") or []:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Handoff Answers", ""])
    answers = report.get("handoff_answers") if isinstance(report.get("handoff_answers"), dict) else {}
    for item in answers.get("what_core_can_do") or []:
        lines.append(f"- {item}")
    lines.extend([
        "",
        f"- Control layer: {answers.get('control_layer_call')}",
        f"- User layer: {answers.get('user_layer_call')}",
        f"- Fixture/import scope: {answers.get('fixture_diagnostic_or_import')}",
        "",
        "## Blockers",
        "",
    ])
    for item in report.get("blockers") or ["none"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Diagnosis Codes", ""])
    for code in report.get("diagnosis_codes") or []:
        lines.append(f"- `{code}`")
    lines.append("")
    return "\n".join(lines)
