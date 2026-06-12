#!/usr/bin/env python3
"""Build the CrowdTensor core technology Inference RC evidence pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor import large_model_inference_rc as rc  # noqa: E402
from crowdtensor import large_model_shard as alpha  # noqa: E402
from scripts import large_model_shard_alpha_pack as alpha_pack  # noqa: E402


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_json_or_file(value: str, *, expect_list: bool = False) -> Any:
    try:
        return rc.read_json_value(value, expect_list=expect_list)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_new_tokens > rc.MAX_REAL_RUN_TOKENS and args.mode == "real":
        raise SystemExit(f"--max-new-tokens must be <= {rc.MAX_REAL_RUN_TOKENS} in real mode")
    metadata = parse_json_or_file(args.model_metadata) if args.model_metadata else {}
    raw_devices = parse_json_or_file(args.device_profile, expect_list=True) if args.device_profile else None
    model_manifest = alpha.build_model_manifest(
        model_id=args.model_id,
        model_path=args.model_path,
        quantization=args.quantization,
        layer_count=args.layer_count,
        context_length=args.context_length,
        model_size_mb=args.model_size_mb,
        metadata=metadata,
    )
    device_profile = rc.build_device_profile_v2(
        raw_devices=raw_devices,
        rpc_endpoints=[args.rpc_endpoint] if args.rpc_endpoint else None,
    )
    alpha_devices = rc.device_profile_to_alpha_devices(device_profile)
    adapter = alpha.build_llama_cpp_rpc_adapter(
        model_manifest=model_manifest,
        devices=alpha_devices,
        llama_cli=args.llama_cli,
        rpc_server=args.llama_rpc_server,
        prompt_placeholder=args.prompt_placeholder,
        max_new_tokens=min(args.max_new_tokens, rc.MAX_REAL_RUN_TOKENS),
        real_runtime_verified=False,
    )
    alpha_args = alpha_pack.parse_args([
        "--output-dir",
        str(output_dir / "alpha"),
        "--model-id",
        str(args.model_id),
        "--model-path",
        str(args.model_path),
        "--quantization",
        str(args.quantization),
        "--layer-count",
        str(args.layer_count),
        "--context-length",
        str(args.context_length),
        "--model-size-mb",
        str(args.model_size_mb),
        "--reserved-kv-cache-mb",
        str(args.reserved_kv_cache_mb),
        "--max-new-tokens",
        str(min(args.max_new_tokens, rc.MAX_REAL_RUN_TOKENS)),
        "--llama-cli",
        str(args.llama_cli),
        "--llama-rpc-server",
        str(args.llama_rpc_server),
        "--prompt-placeholder",
        str(args.prompt_placeholder),
    ])
    if args.model_metadata:
        alpha_args.model_metadata = args.model_metadata
    if args.device_profile:
        alpha_args.device_profile = args.device_profile
    if args.real_benchmark_report:
        alpha_args.real_benchmark_report = args.real_benchmark_report
    alpha_report = alpha_pack.build_report(alpha_args)
    adapter_interface = rc.build_runtime_adapter_interface(args.runtime_backend)
    runtime_probe = rc.build_runtime_probe(
        adapter=adapter,
        model_manifest=model_manifest,
        llama_cli=args.llama_cli,
        llama_rpc_server=args.llama_rpc_server,
        endpoint_timeout_seconds=args.endpoint_timeout_seconds,
    )
    partition_v2 = rc.build_partition_manifest_v2(
        model_manifest=model_manifest,
        devices=alpha_devices,
        reserved_kv_cache_mb=args.reserved_kv_cache_mb,
    )
    runner_result = rc.build_runner_result(
        mode=args.mode,
        adapter=adapter,
        model_manifest=model_manifest,
        runtime_probe=runtime_probe,
        partition_manifest=partition_v2,
        max_new_tokens=args.max_new_tokens,
        timeout_seconds=args.real_timeout_seconds,
        start_workers=args.start_workers,
        real_run_report=args.real_run_report,
    )
    imported_benchmark = alpha.imported_real_benchmark(args.real_benchmark_report) if args.real_benchmark_report else None
    benchmark = rc.build_benchmark_v2(
        runner_result=runner_result,
        partition_manifest=partition_v2,
        device_profile=device_profile,
        imported_benchmark=imported_benchmark,
    )
    correctness = rc.build_correctness_summary(
        runner_result=runner_result,
        model_manifest=model_manifest,
        adapter=adapter,
        partition_manifest=partition_v2,
        baseline_digest=args.baseline_digest,
    )
    serving_hooks = rc.build_serving_hooks(
        runner_result=runner_result,
        partition_manifest=partition_v2,
        max_new_tokens=min(args.max_new_tokens, rc.MAX_REAL_RUN_TOKENS),
    )
    report = rc.build_rc_report(
        output_dir=output_dir,
        mode=args.mode,
        alpha_report=alpha_report,
        adapter_interface=adapter_interface,
        runtime_probe=runtime_probe,
        device_profile=device_profile,
        runtime_adapter=adapter,
        partition_manifest=partition_v2,
        runner_result=runner_result,
        benchmark=benchmark,
        correctness=correctness,
        serving_hooks=serving_hooks,
    )

    write_json(output_dir / "runtime_adapter_interface.json", adapter_interface)
    write_json(output_dir / "runtime_adapter_probe.json", runtime_probe)
    write_json(output_dir / "device_profile.json", device_profile)
    write_json(output_dir / "partition_manifest_v2.json", partition_v2)
    write_json(output_dir / "runner_result.json", runner_result)
    write_json(output_dir / "benchmark_v2.json", benchmark)
    write_json(output_dir / "correctness_summary.json", correctness)
    write_json(output_dir / "serving_hooks.json", serving_hooks)

    artifacts = {
        "summary_json": rc.artifact_entry(output_dir / "core_technology_inference_rc.json", output_dir, kind="core_technology_inference_rc", schema=rc.RC_SCHEMA, ok=report.get("ok")),
        "summary_markdown": rc.artifact_entry(output_dir / "core_technology_inference_rc.md", output_dir, kind="core_technology_inference_rc_markdown"),
        "support_bundle_json": rc.artifact_entry(output_dir / "support_bundle.json", output_dir, kind="core_technology_inference_rc_support_bundle", schema=rc.RC_SUPPORT_BUNDLE_SCHEMA),
        "runtime_adapter_interface_json": rc.artifact_entry(output_dir / "runtime_adapter_interface.json", output_dir, kind="large_model_runtime_adapter_interface", schema=rc.RUNTIME_ADAPTER_INTERFACE_SCHEMA),
        "runtime_adapter_probe_json": rc.artifact_entry(output_dir / "runtime_adapter_probe.json", output_dir, kind="large_model_runtime_adapter_probe", schema=rc.RUNTIME_ADAPTER_PROBE_SCHEMA),
        "device_profile_json": rc.artifact_entry(output_dir / "device_profile.json", output_dir, kind="large_model_device_profile", schema=rc.DEVICE_PROFILE_SCHEMA),
        "partition_manifest_v2_json": rc.artifact_entry(output_dir / "partition_manifest_v2.json", output_dir, kind="large_model_partition_manifest_v2", schema=rc.PARTITION_MANIFEST_SCHEMA),
        "runner_result_json": rc.artifact_entry(output_dir / "runner_result.json", output_dir, kind="large_model_runner_result", schema=rc.RUNNER_RESULT_SCHEMA),
        "benchmark_v2_json": rc.artifact_entry(output_dir / "benchmark_v2.json", output_dir, kind="large_model_benchmark_v2", schema=rc.BENCHMARK_SCHEMA),
        "correctness_summary_json": rc.artifact_entry(output_dir / "correctness_summary.json", output_dir, kind="large_model_correctness_summary", schema=rc.CORRECTNESS_SCHEMA),
        "serving_hooks_json": rc.artifact_entry(output_dir / "serving_hooks.json", output_dir, kind="large_model_serving_hooks", schema=rc.SERVING_HOOKS_SCHEMA),
    }
    report["artifacts"] = artifacts
    (output_dir / "core_technology_inference_rc.md").write_text(rc.render_markdown(report), encoding="utf-8")
    artifacts["summary_markdown"]["present"] = True
    artifacts["summary_json"]["present"] = True
    artifacts["support_bundle_json"]["present"] = True
    report["artifact_summary"] = rc.artifact_summary(artifacts)
    write_json(output_dir / "support_bundle.json", rc.build_support_bundle(report))
    write_json(output_dir / "core_technology_inference_rc.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CrowdTensor core technology Inference RC evidence.")
    parser.add_argument("--mode", choices=["plan", "fixture", "real"], default="fixture")
    parser.add_argument("--output-dir", default="dist/core-technology-inference-rc")
    parser.add_argument("--runtime-backend", default=rc.SUPPORTED_RUNTIME)
    parser.add_argument("--model-id", default=alpha.DEFAULT_MODEL_ID)
    parser.add_argument("--model-path", default=alpha.DEFAULT_MODEL_PATH)
    parser.add_argument("--quantization", default=alpha.DEFAULT_QUANTIZATION)
    parser.add_argument("--layer-count", type=int, default=alpha.DEFAULT_LAYER_COUNT)
    parser.add_argument("--context-length", type=int, default=alpha.DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--model-size-mb", type=int, default=alpha.DEFAULT_MODEL_SIZE_MB)
    parser.add_argument("--reserved-kv-cache-mb", type=int, default=alpha.DEFAULT_KV_CACHE_MB)
    parser.add_argument("--max-new-tokens", type=int, default=rc.DEFAULT_RC_MAX_NEW_TOKENS)
    parser.add_argument("--model-metadata", default="")
    parser.add_argument("--device-profile", default="")
    parser.add_argument("--real-benchmark-report", default="")
    parser.add_argument("--real-run-report", default="")
    parser.add_argument("--baseline-digest", default="")
    parser.add_argument("--llama-cli", default=alpha.DEFAULT_LLAMA_CLI)
    parser.add_argument("--llama-rpc-server", default=alpha.DEFAULT_LLAMA_RPC_SERVER)
    parser.add_argument("--prompt-placeholder", default=alpha.DEFAULT_PROMPT_PLACEHOLDER)
    parser.add_argument("--rpc-endpoint", default="")
    parser.add_argument("--endpoint-timeout-seconds", type=float, default=0.25)
    parser.add_argument("--real-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--start-workers", action="store_true")
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
    if args.mode == "real" and args.max_new_tokens > rc.MAX_REAL_RUN_TOKENS:
        raise SystemExit(f"--max-new-tokens must be <= {rc.MAX_REAL_RUN_TOKENS} in real mode")
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
        print(rc.render_markdown(report))


if __name__ == "__main__":
    main()
