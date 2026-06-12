from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from crowdtensor import cli
from crowdtensor import large_model_shard as lms
from scripts import large_model_shard_alpha_check as check
from scripts import large_model_shard_alpha_pack as pack


class LargeModelShardAlphaTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_large_model_alpha_test_"))

    def test_default_adapter_planner_contract_and_benchmark_are_ready(self) -> None:
        model = lms.build_model_manifest()
        devices = lms.default_device_profiles()
        adapter = lms.build_llama_cpp_rpc_adapter(model_manifest=model, devices=devices)
        partition = lms.plan_partitions(model_manifest=model, devices=devices)
        contract = lms.build_workload_contract(
            model_manifest=model,
            adapter=adapter,
            partition_manifest=partition,
        )
        benchmark = lms.build_benchmark_report(
            model_manifest=model,
            adapter=adapter,
            partition_manifest=partition,
        )

        self.assertEqual(adapter["schema"], lms.RUNTIME_ADAPTER_SCHEMA)
        self.assertEqual(adapter["adapter_kind"], "llama_cpp_rpc")
        self.assertIn("--rpc", adapter["client_command"])
        self.assertTrue(adapter["controlled_network_only"])
        self.assertTrue(adapter["not_public_rpc_safe"])
        self.assertFalse(adapter["real_runtime_verified"])

        self.assertTrue(partition["runnable"])
        self.assertEqual(partition["assigned_layer_count"], 32)
        self.assertEqual(partition["unassigned_layer_count"], 0)
        self.assertEqual([item["layer_count"] for item in partition["assignments"]], [16, 16])
        self.assertIn("large_model_memory_budget_ready", partition["diagnosis_codes"])

        self.assertEqual(contract["schema"], lms.WORKLOAD_CONTRACT_SCHEMA)
        self.assertEqual(contract["serving_readiness"]["streaming_event_schema"], "large_model_stream_event_v1")
        self.assertEqual(contract["serving_readiness"]["bounded_batch_request_schema"], "large_model_batch_request_v1")
        self.assertEqual(contract["serving_readiness"]["cancellation_field"], "cancel_requested")
        self.assertFalse(contract["redaction_policy"]["raw_prompt_public"])
        self.assertFalse(contract["redaction_policy"]["generated_token_ids_public"])
        self.assertFalse(contract["redaction_policy"]["kv_cache_public"])

        self.assertEqual(benchmark["schema"], lms.BENCHMARK_SCHEMA)
        self.assertEqual(benchmark["measurement_kind"], "fixture-planning-estimate")
        self.assertFalse(benchmark["real_runtime_verified"])
        self.assertTrue(benchmark["fixture"])
        self.assertTrue(benchmark["comparison"]["compares_single_device_fallback"])
        self.assertTrue(benchmark["comparison"]["compares_sharded_adapter_path"])

    def test_public_endpoint_blocks_partition_readiness(self) -> None:
        model = lms.build_model_manifest()
        devices = lms.normalize_device_profiles([
            {
                "device_id": "public-node",
                "backend": "cuda",
                "rpc_endpoint": "http://8.8.8.8:50052",
                "usable_memory_mb": 12288,
            }
        ])

        partition = lms.plan_partitions(model_manifest=model, devices=devices)

        self.assertFalse(partition["runnable"])
        self.assertIn("large_model_rpc_endpoint_not_controlled", partition["blockers"])

    def test_pack_report_is_ci_safe_and_check_validates_it(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args(["--output-dir", str(output_dir)]))

        self.assertTrue(report["ok"])
        self.assertEqual(report["schema"], lms.ALPHA_SCHEMA)
        self.assertEqual(report["workload_contract"]["schema"], lms.WORKLOAD_CONTRACT_SCHEMA)
        self.assertEqual(report["evidence_scope"], "fixture-contract-plan")
        self.assertFalse(report["real_runtime_verified"])
        self.assertTrue(report["safety"]["public_artifact_safe"])
        self.assertEqual(lms.public_redaction_errors(report), [])
        self.assertEqual(report["artifact_summary"]["present_artifact_count"], report["artifact_summary"]["artifact_count"])
        check.validate_report(report)

        for name in [
            "large_model_shard_alpha.json",
            "large_model_shard_alpha.md",
            "runtime_adapter.json",
            "partition_manifest.json",
            "workload_contract.json",
            "benchmark.json",
            "support_bundle.json",
        ]:
            self.assertTrue((output_dir / name).is_file(), name)

    def test_real_benchmark_import_marks_real_runtime_verified(self) -> None:
        output_dir = self._tmp_dir()
        benchmark_path = output_dir / "real_benchmark.json"
        benchmark_path.write_text(
            json.dumps({
                "metrics": {
                    "ttft_ms": 123.0,
                    "tokens_per_second": 7.5,
                    "p50_latency_ms": 90.0,
                    "p95_latency_ms": 140.0,
                    "memory_peak_mb": 4096,
                    "network_bytes_per_token": 131072,
                    "cache_hits": 2,
                    "cache_misses": 1,
                }
            }),
            encoding="utf-8",
        )

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--real-benchmark-report",
            str(benchmark_path),
        ]))

        self.assertTrue(report["ok"])
        self.assertTrue(report["real_runtime_verified"])
        self.assertEqual(report["evidence_scope"], "real-runtime")
        self.assertEqual(report["benchmark"]["measurement_kind"], "real-runtime")
        self.assertEqual(report["benchmark"]["sharded_adapter_path"]["tokens_per_second"], 7.5)
        self.assertIn("large_model_7b_real_runtime_verified", report["diagnosis_codes"])
        self.assertNotIn("large_model_7b_real_runtime_deferred", report["diagnosis_codes"])

    def test_cli_wrapper_generates_ready_summary_without_overclaiming(self) -> None:
        output_dir = self._tmp_dir()
        args = cli.parse_args(["large-model-shard", "--output-dir", str(output_dir)])
        summary = cli.build_large_model_shard_alpha(args)

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["cli_schema"], "large_model_shard_alpha_cli_v1")
        self.assertEqual(summary["schema"], lms.ALPHA_SCHEMA)
        self.assertEqual(summary["evidence_scope"], "fixture-contract-plan")
        self.assertFalse(summary["real_runtime_verified"])
        self.assertEqual(summary["artifact_summary"]["present_artifact_count"], summary["artifact_summary"]["artifact_count"])
        self.assertTrue((output_dir / "large_model_shard_alpha_cli_summary.json").is_file())

        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            cli.print_large_model_shard_alpha(summary)
        output = rendered.getvalue()
        self.assertIn("CrowdTensor large-model shard Alpha", output)
        self.assertIn("real_runtime_verified=False", output)
        self.assertIn("llama_cpp_rpc", output)

    def test_cli_rejects_bad_large_model_args(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args(["large-model-shard", "--layer-count", "0"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["large-model-shard", "--reserved-kv-cache-mb", "-1"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["large-model-shard", "--real-benchmark-report", "/tmp/does-not-exist.json"])


if __name__ == "__main__":
    unittest.main()
