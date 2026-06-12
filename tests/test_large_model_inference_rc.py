from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from crowdtensor import cli
from crowdtensor import large_model_inference_rc as rc
from crowdtensor import large_model_shard as alpha
from scripts import large_model_inference_rc_check as check
from scripts import large_model_inference_rc_pack as pack


class LargeModelInferenceRcTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_large_model_rc_test_"))

    def _model_and_partition(self) -> tuple[dict, dict, dict, dict]:
        model = alpha.build_model_manifest()
        raw_devices = [
            {
                "device_id": "worker-a",
                "backend": "cuda",
                "rpc_endpoint": "http://127.0.0.1:50052",
                "usable_memory_mb": 12288,
                "vram_total_mb": 16384,
            },
            {
                "device_id": "worker-b",
                "backend": "cuda",
                "rpc_endpoint": "http://127.0.0.1:50053",
                "usable_memory_mb": 12288,
                "vram_total_mb": 16384,
            },
        ]
        profile = rc.build_device_profile_v2(raw_devices=raw_devices)
        devices = rc.device_profile_to_alpha_devices(profile)
        partition = rc.build_partition_manifest_v2(model_manifest=model, devices=devices)
        adapter = alpha.build_llama_cpp_rpc_adapter(model_manifest=model, devices=devices)
        return model, profile, partition, adapter

    def test_runtime_interface_probe_and_device_profile_contracts(self) -> None:
        model, profile, _partition, adapter = self._model_and_partition()
        interface = rc.build_runtime_adapter_interface("vllm")
        probe = rc.build_runtime_probe(
            adapter=adapter,
            model_manifest=model,
            llama_cli="crowdtensor-missing-llama-cli",
            llama_rpc_server="crowdtensor-missing-rpc-server",
            endpoint_timeout_seconds=0.01,
        )

        self.assertEqual(interface["schema"], rc.RUNTIME_ADAPTER_INTERFACE_SCHEMA)
        self.assertFalse(interface["selected_supported"])
        self.assertIn("unsupported_runtime_backend", interface["diagnosis_codes"])
        unsupported = [item for item in interface["descriptors"] if item["status"] == "unsupported"]
        self.assertGreaterEqual(len(unsupported), len(rc.UNSUPPORTED_RUNTIMES))

        self.assertEqual(probe["schema"], rc.RUNTIME_ADAPTER_PROBE_SCHEMA)
        self.assertFalse(probe["real_runtime_ready"])
        self.assertIn("large_model_runtime_binaries_missing", probe["diagnosis_codes"])
        self.assertIn("large_model_local_model_missing", probe["diagnosis_codes"])
        self.assertTrue(probe["controlled_lan_vpn_only"])
        self.assertFalse(probe["model_file"]["model_path_public"])

        self.assertEqual(profile["schema"], rc.DEVICE_PROFILE_SCHEMA)
        self.assertEqual(profile["source"], "json-import")
        self.assertEqual(len(profile["devices"]), 2)
        self.assertIn("large_model_device_profile_v2_ready", profile["diagnosis_codes"])

    def test_partition_runner_benchmark_correctness_and_serving_hooks(self) -> None:
        model, profile, partition, adapter = self._model_and_partition()
        runtime_probe = rc.build_runtime_probe(
            adapter=adapter,
            model_manifest=model,
            llama_cli="crowdtensor-missing-llama-cli",
            llama_rpc_server="crowdtensor-missing-rpc-server",
            endpoint_timeout_seconds=0.01,
        )
        runner = rc.build_runner_result(
            mode="fixture",
            adapter=adapter,
            model_manifest=model,
            runtime_probe=runtime_probe,
            partition_manifest=partition,
            max_new_tokens=8,
            timeout_seconds=120.0,
        )
        benchmark = rc.build_benchmark_v2(
            runner_result=runner,
            partition_manifest=partition,
            device_profile=profile,
            imported_benchmark={"real_runtime_verified": True, "metrics": {"tokens_per_second": 99.0}},
        )
        correctness = rc.build_correctness_summary(
            runner_result=runner,
            model_manifest=model,
            adapter=adapter,
            partition_manifest=partition,
            baseline_digest=runner["output_digest"],
        )
        serving = rc.build_serving_hooks(runner_result=runner, partition_manifest=partition, max_new_tokens=8)

        self.assertEqual(partition["schema"], rc.PARTITION_MANIFEST_SCHEMA)
        self.assertTrue(partition["runnable"])
        self.assertEqual(partition["tensor_split_plan"]["tensor_split"], [0.5, 0.5])
        self.assertIn("large_model_prefill_decode_memory_estimate_ready", partition["diagnosis_codes"])

        self.assertEqual(runner["schema"], rc.RUNNER_RESULT_SCHEMA)
        self.assertEqual(runner["runner_mode"], "fixture")
        self.assertFalse(runner["real_runtime_verified"])
        self.assertFalse(runner["output_text_public"])
        self.assertFalse(runner["generated_token_ids_public"])

        self.assertEqual(benchmark["schema"], rc.BENCHMARK_SCHEMA)
        self.assertFalse(benchmark["real_runtime_verified"])
        self.assertTrue(benchmark["imported_benchmark_used"])
        self.assertEqual(benchmark["tokens_per_second"], 99.0)
        self.assertTrue(benchmark["failure_diagnosis"]["imported_benchmark_without_runner_verification"])

        self.assertEqual(correctness["schema"], rc.CORRECTNESS_SCHEMA)
        self.assertTrue(correctness["baseline_comparison"]["match"])
        self.assertFalse(correctness["generated_token_ids_public"])

        self.assertEqual(serving["schema"], rc.SERVING_HOOKS_SCHEMA)
        self.assertEqual(serving["streaming_event_schema"], rc.STREAM_EVENT_SCHEMA)
        self.assertEqual(serving["bounded_batch_request_schema"], rc.BOUNDED_BATCH_SCHEMA)
        self.assertEqual(serving["health_aware_route_metadata_schema"], rc.ROUTE_HEALTH_SCHEMA)
        self.assertTrue(serving["streaming_event_emitter_ready"])
        self.assertTrue(serving["sample_stream_events"])
        self.assertEqual(serving["cancellation_field"], "cancel_requested")
        self.assertEqual(serving["timeout_field"], "timeout_seconds")

    def test_pack_report_is_ci_safe_and_check_validates_it(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args(["--output-dir", str(output_dir), "--mode", "fixture"]))

        self.assertTrue(report["ok"])
        self.assertEqual(report["schema"], rc.RC_SCHEMA)
        self.assertFalse(report["real_runtime_verified"])
        self.assertFalse(report["real_7b_runtime_verified"])
        self.assertIn("core_technology_real_7b_runtime_not_verified", report["diagnosis_codes"])
        self.assertIn("large_model_runtime_binaries_missing", report["blockers"])
        self.assertTrue(report["safety"]["public_artifact_safe"])
        self.assertEqual(rc.public_redaction_errors(report), [])
        self.assertEqual(report["artifact_summary"]["present_artifact_count"], report["artifact_summary"]["artifact_count"])
        check.validate_report(report)

        for name in [
            "core_technology_inference_rc.json",
            "core_technology_inference_rc.md",
            "runtime_adapter_interface.json",
            "runtime_adapter_probe.json",
            "device_profile.json",
            "partition_manifest_v2.json",
            "runner_result.json",
            "benchmark_v2.json",
            "correctness_summary.json",
            "serving_hooks.json",
            "support_bundle.json",
        ]:
            self.assertTrue((output_dir / name).is_file(), name)

    def test_real_run_import_marks_runner_and_rc_real_verified(self) -> None:
        output_dir = self._tmp_dir()
        real_run_path = output_dir / "real_run.json"
        real_run_path.write_text(
            json.dumps({
                "metrics": {
                    "ttft_ms": 120.0,
                    "tokens_per_second": 8.5,
                    "wall_time_seconds": 1.2,
                    "generated_token_count": 4,
                    "max_new_tokens": 4,
                    "output_digest": "sha256:" + "1" * 64,
                    "p50_latency_ms": 80.0,
                    "p95_latency_ms": 140.0,
                    "memory_peak_mb": 4096,
                    "network_bytes_per_token": 65536,
                    "cache_hits": 1,
                    "cache_misses": 3,
                }
            }),
            encoding="utf-8",
        )

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--mode",
            "fixture",
            "--real-run-report",
            str(real_run_path),
        ]))

        self.assertTrue(report["ok"])
        self.assertTrue(report["real_runtime_verified"])
        self.assertTrue(report["real_7b_runtime_verified"])
        self.assertEqual(report["runner_result"]["runner_mode"], "real-import")
        self.assertEqual(report["benchmark"]["measurement_kind"], "real-runtime")
        self.assertIn("core_technology_real_7b_runtime_verified", report["diagnosis_codes"])
        self.assertFalse(report["runner_result"]["output_text_public"])

    def test_cli_wrapper_generates_ready_rc_summary_without_real_overclaim(self) -> None:
        output_dir = self._tmp_dir()
        args = cli.parse_args(["large-model-shard-rc", "--output-dir", str(output_dir)])
        summary = cli.build_core_technology_inference_rc(args)

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["cli_schema"], "core_technology_inference_rc_cli_v1")
        self.assertEqual(summary["schema"], rc.RC_SCHEMA)
        self.assertFalse(summary["real_runtime_verified"])
        self.assertEqual(summary["artifact_summary"]["present_artifact_count"], summary["artifact_summary"]["artifact_count"])
        self.assertTrue((output_dir / "core_technology_inference_rc_cli_summary.json").is_file())

        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            cli.print_core_technology_inference_rc(summary)
        output = rendered.getvalue()
        self.assertIn("CrowdTensor core technology Inference RC", output)
        self.assertIn("real_7b=False", output)
        self.assertIn("serving_hooks", output)

    def test_cli_rejects_bad_rc_args(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args(["large-model-shard-rc", "--layer-count", "0"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["large-model-shard-rc", "--reserved-kv-cache-mb", "-1"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["large-model-shard-rc", "--mode", "real", "--max-new-tokens", "9"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["large-model-shard-rc", "--mode", "real", "--real-timeout-seconds", "1201"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["large-model-shard-rc", "--real-run-report", "/tmp/does-not-exist.json"])


if __name__ == "__main__":
    unittest.main()
