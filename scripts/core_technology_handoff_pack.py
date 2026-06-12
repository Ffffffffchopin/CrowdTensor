#!/usr/bin/env python3
"""Build the CrowdTensor core technology Handoff RC evidence pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor import core_technology_handoff as handoff  # noqa: E402
from crowdtensor import large_model_inference_rc as inference_rc  # noqa: E402
from scripts import large_model_inference_rc_pack as inference_pack  # noqa: E402


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_output_dir = output_dir / "inference-rc"
    inference_args = inference_pack.parse_args([
        "--mode",
        args.mode,
        "--output-dir",
        str(inference_output_dir),
        "--runtime-backend",
        args.runtime_backend,
        "--model-id",
        args.model_id,
        "--model-path",
        args.model_path,
        "--quantization",
        args.quantization,
        "--layer-count",
        str(args.layer_count),
        "--context-length",
        str(args.context_length),
        "--model-size-mb",
        str(args.model_size_mb),
        "--reserved-kv-cache-mb",
        str(args.reserved_kv_cache_mb),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--llama-cli",
        args.llama_cli,
        "--llama-rpc-server",
        args.llama_rpc_server,
        "--prompt-placeholder",
        args.prompt_placeholder,
        "--endpoint-timeout-seconds",
        str(args.endpoint_timeout_seconds),
        "--real-timeout-seconds",
        str(args.real_timeout_seconds),
    ])
    for attr in [
        "model_metadata",
        "device_profile",
        "real_benchmark_report",
        "real_run_report",
        "baseline_digest",
        "rpc_endpoint",
    ]:
        setattr(inference_args, attr, getattr(args, attr))
    inference_args.start_workers = bool(args.start_workers)
    inference_report = inference_pack.build_report(inference_args)
    deployment_runbook = handoff.build_deployment_runbook(output_dir=output_dir, inference_report=inference_report)
    next_layer_contract = handoff.build_next_layer_contract(inference_report=inference_report)
    adapter_conformance = handoff.build_adapter_conformance(inference_report=inference_report)
    test_gate_summary = handoff.build_test_gate_summary(mode=args.mode, full_pytest=args.full_pytest)
    report = handoff.build_handoff_report(
        output_dir=output_dir,
        mode=args.mode,
        inference_report=inference_report,
        deployment_runbook=deployment_runbook,
        next_layer_contract=next_layer_contract,
        adapter_conformance=adapter_conformance,
        test_gate_summary=test_gate_summary,
    )

    write_json(output_dir / "deployment_runbook.json", deployment_runbook)
    write_json(output_dir / "next_layer_contract.json", next_layer_contract)
    write_json(output_dir / "adapter_conformance.json", adapter_conformance)
    write_json(output_dir / "test_gate_summary.json", test_gate_summary)

    artifacts = {
        "summary_json": handoff.artifact_entry(output_dir / "core_technology_handoff_rc.json", output_dir, kind="core_technology_handoff_rc", schema=handoff.HANDOFF_SCHEMA, ok=report.get("ok")),
        "summary_markdown": handoff.artifact_entry(output_dir / "core_technology_handoff_rc.md", output_dir, kind="core_technology_handoff_rc_markdown"),
        "support_bundle_json": handoff.artifact_entry(output_dir / "support_bundle.json", output_dir, kind="core_technology_handoff_rc_support_bundle", schema=handoff.HANDOFF_SUPPORT_BUNDLE_SCHEMA),
        "inference_rc_json": handoff.artifact_entry(inference_output_dir / "core_technology_inference_rc.json", output_dir, kind="core_technology_inference_rc", schema=inference_rc.RC_SCHEMA, ok=inference_report.get("ok")),
        "deployment_runbook_json": handoff.artifact_entry(output_dir / "deployment_runbook.json", output_dir, kind="core_technology_deployment_runbook", schema=handoff.DEPLOYMENT_RUNBOOK_SCHEMA),
        "next_layer_contract_json": handoff.artifact_entry(output_dir / "next_layer_contract.json", output_dir, kind="core_technology_next_layer_contract", schema=handoff.NEXT_LAYER_CONTRACT_SCHEMA),
        "adapter_conformance_json": handoff.artifact_entry(output_dir / "adapter_conformance.json", output_dir, kind="core_technology_adapter_conformance", schema=handoff.ADAPTER_CONFORMANCE_SCHEMA),
        "test_gate_summary_json": handoff.artifact_entry(output_dir / "test_gate_summary.json", output_dir, kind="core_technology_test_gate_summary", schema=handoff.TEST_GATE_SCHEMA),
    }
    report["artifacts"] = artifacts
    (output_dir / "core_technology_handoff_rc.md").write_text(handoff.render_markdown(report), encoding="utf-8")
    artifacts["summary_markdown"]["present"] = True
    artifacts["summary_json"]["present"] = True
    artifacts["support_bundle_json"]["present"] = True
    artifacts["inference_rc_json"]["present"] = (inference_output_dir / "core_technology_inference_rc.json").is_file()
    report["artifact_summary"] = handoff.artifact_summary(artifacts)
    write_json(output_dir / "support_bundle.json", handoff.build_support_bundle(report))
    write_json(output_dir / "core_technology_handoff_rc.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CrowdTensor core technology Handoff RC evidence.")
    parser.add_argument("--mode", choices=["plan", "fixture", "real"], default="fixture")
    parser.add_argument("--output-dir", default="dist/core-technology-handoff-rc")
    parser.add_argument("--runtime-backend", default=inference_rc.SUPPORTED_RUNTIME)
    parser.add_argument("--model-id", default="gguf-7b-alpha-fixture")
    parser.add_argument("--model-path", default="models/gguf-7b-alpha.Q4_K_M.gguf")
    parser.add_argument("--quantization", default="Q4_K_M")
    parser.add_argument("--layer-count", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--model-size-mb", type=int, default=7168)
    parser.add_argument("--reserved-kv-cache-mb", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=inference_rc.DEFAULT_RC_MAX_NEW_TOKENS)
    parser.add_argument("--model-metadata", default="")
    parser.add_argument("--device-profile", default="")
    parser.add_argument("--real-benchmark-report", default="")
    parser.add_argument("--real-run-report", default="")
    parser.add_argument("--baseline-digest", default="")
    parser.add_argument("--llama-cli", default="llama-cli")
    parser.add_argument("--llama-rpc-server", default="rpc-server")
    parser.add_argument("--prompt-placeholder", default="PROMPT_FILE")
    parser.add_argument("--rpc-endpoint", default="")
    parser.add_argument("--endpoint-timeout-seconds", type=float, default=0.25)
    parser.add_argument("--real-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--start-workers", action="store_true")
    parser.add_argument("--full-pytest", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.layer_count < 1:
        raise SystemExit("--layer-count must be positive")
    if args.context_length < 1:
        raise SystemExit("--context-length must be positive")
    if args.model_size_mb < 1:
        raise SystemExit("--model-size-mb must be positive")
    if args.reserved_kv_cache_mb < 0:
        raise SystemExit("--reserved-kv-cache-mb must be non-negative")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")
    if args.mode == "real" and args.max_new_tokens > inference_rc.MAX_REAL_RUN_TOKENS:
        raise SystemExit(f"--max-new-tokens must be <= {inference_rc.MAX_REAL_RUN_TOKENS} in real mode")
    if args.endpoint_timeout_seconds <= 0:
        raise SystemExit("--endpoint-timeout-seconds must be positive")
    if args.real_timeout_seconds <= 0:
        raise SystemExit("--real-timeout-seconds must be positive")
    if args.mode == "real" and args.real_timeout_seconds > inference_rc.MAX_REAL_RUN_TIMEOUT_SECONDS:
        raise SystemExit(f"--real-timeout-seconds must be <= {inference_rc.MAX_REAL_RUN_TIMEOUT_SECONDS} in real mode")
    for attr in ["real_benchmark_report", "real_run_report"]:
        value = getattr(args, attr)
        if value and not Path(value).is_file():
            raise SystemExit(f"--{attr.replace('_', '-')} must point to an existing JSON file")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(handoff.render_markdown(report))


if __name__ == "__main__":
    main()
