#!/usr/bin/env python3
"""Build the Large-Model Shard Alpha evidence pack.

This is a core technology layer artifact.  It creates a llama.cpp RPC / GGUF
adapter plan, partition manifest, versioned workload contract, and benchmark
report.  CI runs the fixture-contract path; real 7B verification is represented
only when a real benchmark JSON is explicitly imported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor import large_model_shard as lms  # noqa: E402


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_json_or_file(value: str, *, expect_list: bool = False) -> Any:
    text = str(value or "").strip()
    if not text:
        return [] if expect_list else {}
    path = Path(text)
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
    else:
        loaded = json.loads(text)
    if expect_list and not isinstance(loaded, list):
        raise SystemExit("expected a JSON list")
    if not expect_list and not isinstance(loaded, dict):
        raise SystemExit("expected a JSON object")
    return loaded


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = parse_json_or_file(args.model_metadata) if args.model_metadata else {}
    raw_devices = parse_json_or_file(args.device_profile, expect_list=True) if args.device_profile else None
    model_manifest = lms.build_model_manifest(
        model_id=args.model_id,
        model_path=args.model_path,
        quantization=args.quantization,
        layer_count=args.layer_count,
        context_length=args.context_length,
        model_size_mb=args.model_size_mb,
        metadata=metadata,
    )
    devices = lms.normalize_device_profiles(raw_devices)
    real_benchmark = lms.imported_real_benchmark(args.real_benchmark_report) if args.real_benchmark_report else None
    adapter = lms.build_llama_cpp_rpc_adapter(
        model_manifest=model_manifest,
        devices=devices,
        llama_cli=args.llama_cli,
        rpc_server=args.llama_rpc_server,
        prompt_placeholder=args.prompt_placeholder,
        max_new_tokens=args.max_new_tokens,
        real_runtime_verified=bool(real_benchmark),
    )
    partition_manifest = lms.plan_partitions(
        model_manifest=model_manifest,
        devices=devices,
        reserved_kv_cache_mb=args.reserved_kv_cache_mb,
    )
    workload_contract = lms.build_workload_contract(
        model_manifest=model_manifest,
        adapter=adapter,
        partition_manifest=partition_manifest,
        max_new_tokens=args.max_new_tokens,
    )
    benchmark = lms.build_benchmark_report(
        model_manifest=model_manifest,
        adapter=adapter,
        partition_manifest=partition_manifest,
        real_benchmark=real_benchmark,
    )
    write_json(output_dir / "model_manifest.json", model_manifest)
    write_json(output_dir / "device_profiles.json", {"devices": devices})
    write_json(output_dir / "runtime_adapter.json", adapter)
    write_json(output_dir / "partition_manifest.json", partition_manifest)
    write_json(output_dir / "workload_contract.json", workload_contract)
    write_json(output_dir / "benchmark.json", benchmark)
    report = lms.build_alpha_report(
        output_dir=output_dir,
        model_manifest=model_manifest,
        devices=devices,
        adapter=adapter,
        partition_manifest=partition_manifest,
        workload_contract=workload_contract,
        benchmark=benchmark,
    )
    artifacts = {
        "summary_json": lms.artifact_entry(
            output_dir / "large_model_shard_alpha.json",
            output_dir,
            kind="large_model_shard_alpha",
            schema=lms.ALPHA_SCHEMA,
            ok=report.get("ok"),
        ),
        "summary_markdown": lms.artifact_entry(
            output_dir / "large_model_shard_alpha.md",
            output_dir,
            kind="large_model_shard_alpha_markdown",
        ),
        "model_manifest_json": lms.artifact_entry(
            output_dir / "model_manifest.json",
            output_dir,
            kind="large_model_manifest",
            schema=lms.MODEL_MANIFEST_SCHEMA,
        ),
        "device_profiles_json": lms.artifact_entry(
            output_dir / "device_profiles.json",
            output_dir,
            kind="large_model_device_profiles",
            schema=lms.DEVICE_PROFILE_SCHEMA,
        ),
        "runtime_adapter_json": lms.artifact_entry(
            output_dir / "runtime_adapter.json",
            output_dir,
            kind="large_model_runtime_adapter",
            schema=lms.RUNTIME_ADAPTER_SCHEMA,
        ),
        "partition_manifest_json": lms.artifact_entry(
            output_dir / "partition_manifest.json",
            output_dir,
            kind="large_model_partition_manifest",
            schema=lms.PARTITION_MANIFEST_SCHEMA,
        ),
        "workload_contract_json": lms.artifact_entry(
            output_dir / "workload_contract.json",
            output_dir,
            kind="large_model_workload_contract",
            schema=lms.WORKLOAD_CONTRACT_SCHEMA,
        ),
        "benchmark_json": lms.artifact_entry(
            output_dir / "benchmark.json",
            output_dir,
            kind="large_model_shard_benchmark",
            schema=lms.BENCHMARK_SCHEMA,
        ),
        "support_bundle_json": lms.artifact_entry(
            output_dir / "support_bundle.json",
            output_dir,
            kind="large_model_shard_alpha_support_bundle",
            schema=lms.SUPPORT_BUNDLE_SCHEMA,
        ),
    }
    report["artifacts"] = artifacts
    (output_dir / "large_model_shard_alpha.md").write_text(lms.render_markdown(report), encoding="utf-8")
    artifacts["summary_markdown"]["present"] = (output_dir / "large_model_shard_alpha.md").is_file()
    artifacts["support_bundle_json"]["present"] = True
    artifacts["summary_json"]["present"] = True
    report["artifact_summary"] = lms.artifact_summary(artifacts)
    support = lms.build_support_bundle(report)
    write_json(output_dir / "support_bundle.json", support)
    write_json(output_dir / "large_model_shard_alpha.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Large-Model Shard Alpha report.")
    parser.add_argument("--output-dir", default="dist/large-model-shard-alpha")
    parser.add_argument("--model-id", default=lms.DEFAULT_MODEL_ID)
    parser.add_argument("--model-path", default=lms.DEFAULT_MODEL_PATH)
    parser.add_argument("--quantization", default=lms.DEFAULT_QUANTIZATION)
    parser.add_argument("--layer-count", type=int, default=lms.DEFAULT_LAYER_COUNT)
    parser.add_argument("--context-length", type=int, default=lms.DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--model-size-mb", type=int, default=lms.DEFAULT_MODEL_SIZE_MB)
    parser.add_argument("--reserved-kv-cache-mb", type=int, default=lms.DEFAULT_KV_CACHE_MB)
    parser.add_argument("--max-new-tokens", type=int, default=lms.DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--model-metadata", default="", help="JSON object or path with model metadata")
    parser.add_argument("--device-profile", default="", help="JSON list/path of device profiles")
    parser.add_argument("--real-benchmark-report", default="", help="import real 7B benchmark metrics and mark real_runtime_verified")
    parser.add_argument("--llama-cli", default=lms.DEFAULT_LLAMA_CLI)
    parser.add_argument("--llama-rpc-server", default=lms.DEFAULT_LLAMA_RPC_SERVER)
    parser.add_argument("--prompt-placeholder", default=lms.DEFAULT_PROMPT_PLACEHOLDER)
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
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(lms.render_markdown(report))


if __name__ == "__main__":
    main()
