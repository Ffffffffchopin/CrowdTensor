from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from crowdtensor import cli
from scripts import large_model_kaggle_validation_check as check
from scripts import large_model_kaggle_validation_pack as pack


class LargeModelKaggleValidationTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_large_model_kaggle_test_"))

    def test_fixture_is_public_safe_without_real_overclaim(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args(["--mode", "fixture", "--output-dir", str(output_dir)]))

        self.assertFalse(report["ok"])
        self.assertEqual(report["schema"], pack.SCHEMA)
        self.assertFalse(report["real_runtime_verified"])
        self.assertFalse(report["real_7b_runtime_verified"])
        self.assertFalse(report["sharded_path_verified"])
        self.assertIn("large_model_kaggle_not_executed", report["blockers"])
        self.assertIn("large_model_sharded_runtime_path_not_verified", report["blockers"])
        self.assertTrue(report["safety"]["public_artifact_safe"])
        self.assertEqual(pack.public_redaction_errors(report), [])
        check.validate_report(report)

        for name in [
            "large_model_kaggle_validation.json",
            "large_model_kaggle_validation.md",
            "support_bundle.json",
            "large_model_kaggle_validation_run_normalized.json",
        ]:
            self.assertTrue((output_dir / name).is_file(), name)
        metadata = json.loads((output_dir / "kaggle-kernel" / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")

    def test_real_7b_import_builds_rc_and_handoff_chain(self) -> None:
        output_dir = self._tmp_dir()
        run_report = output_dir / "real_7b_run.json"
        run_report.write_text(
            json.dumps({
                "schema": pack.RUN_SCHEMA,
                "ok": True,
                "model": {
                    "model_id": "qwen2.5-7b-instruct-q2-k",
                    "parameter_count_b": 7.6,
                    "quantization": "Q2_K",
                    "model_size_mb": 2876,
                },
                "runtime": {
                    "backend": "llama_cpp_rpc",
                    "intended_backend": "llama_cpp_rpc",
                    "worker_count": 2,
                    "stage_count": 2,
                    "cuda_runtime_verified": True,
                    "sharded_path_verified": True,
                    "multi_worker_sharded_path_verified": True,
                },
                "hardware": {
                    "provider": "kaggle",
                    "gpu_count": 2,
                    "gpu_names": ["Tesla T4", "Tesla T4"],
                    "devices": [
                        {"name": "Tesla T4", "memory_total_mb": 15360, "memory_free_mb": 14000},
                        {"name": "Tesla T4", "memory_total_mb": 15360, "memory_free_mb": 14000},
                    ],
                    "kaggle_gpu_verified": True,
                },
                "validation": {
                    "real_runtime_verified": True,
                    "real_7b_runtime_verified": True,
                    "kaggle_gpu_verified": True,
                    "gpu_runtime_verified": True,
                    "sharded_path_verified": True,
                    "multi_worker_sharded_path_verified": True,
                    "core_validation_ready": True,
                    "scale_tier": "7b",
                },
                "metrics": {
                    "ttft_ms": 100.0,
                    "tokens_per_second": 12.0,
                    "wall_time_seconds": 1.0,
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "output_digest": "sha256:" + "4" * 64,
                },
                "diagnosis_codes": ["large_model_kaggle_real_runtime_verified"],
            }),
            encoding="utf-8",
        )

        report = pack.build_report(pack.parse_args([
            "--mode",
            "evidence-import",
            "--output-dir",
            str(output_dir / "imported-7b"),
            "--run-report",
            str(run_report),
        ]))

        self.assertTrue(report["ok"])
        self.assertTrue(report["real_runtime_verified"])
        self.assertTrue(report["real_7b_runtime_verified"])
        self.assertTrue(report["gpu_runtime_verified"])
        self.assertTrue(report["sharded_path_verified"])
        self.assertTrue(report["multi_worker_sharded_path_verified"])
        self.assertTrue(report["core_validation_ready"])
        self.assertEqual(report["model"]["model_id"], "qwen2.5-7b-instruct-q2-k")
        self.assertEqual(report["runtime"]["backend"], "llama_cpp_rpc")
        self.assertEqual(report["metrics"]["generated_token_count"], 8)
        self.assertTrue(report["inference_rc_report"]["real_runtime_verified"])
        self.assertTrue(report["inference_rc_report"]["real_7b_runtime_verified"])
        self.assertTrue(report["handoff_rc_report"]["real_runtime_verified"])
        check.validate_report(report)
        check.validate_report(report, require_real_7b=True, require_core_ready=True)

    def test_small_real_import_does_not_overclaim_7b(self) -> None:
        output_dir = self._tmp_dir()
        run_report = output_dir / "small_run.json"
        run_report.write_text(
            json.dumps({
                "schema": pack.RUN_SCHEMA,
                "ok": True,
                "model": {
                    "model_id": "qwen2.5-1.5b-instruct-q4-k-m",
                    "parameter_count_b": 1.5,
                    "quantization": "Q4_K_M",
                    "model_size_mb": 1066,
                    "layer_count": 28,
                },
                "runtime": {
                    "backend": "llama_cpp_rpc",
                    "intended_backend": "llama_cpp_rpc",
                    "worker_count": 1,
                    "stage_count": 1,
                    "cuda_runtime_verified": True,
                    "sharded_path_verified": True,
                    "multi_worker_sharded_path_verified": False,
                },
                "hardware": {
                    "provider": "kaggle",
                    "gpu_count": 1,
                    "gpu_names": ["Tesla P100-PCIE-16GB"],
                    "devices": [{"name": "Tesla P100-PCIE-16GB", "memory_total_mb": 16280, "memory_free_mb": 15000}],
                    "kaggle_gpu_verified": True,
                },
                "validation": {
                    "real_runtime_verified": True,
                    "real_7b_runtime_verified": False,
                    "kaggle_gpu_verified": True,
                    "gpu_runtime_verified": True,
                    "sharded_path_verified": True,
                    "multi_worker_sharded_path_verified": False,
                    "core_validation_ready": False,
                    "scale_tier": "small",
                },
                "metrics": {
                    "ttft_ms": 100.0,
                    "tokens_per_second": 12.0,
                    "wall_time_seconds": 1.0,
                    "generated_token_count": 4,
                    "max_new_tokens": 4,
                    "output_digest": "sha256:" + "3" * 64,
                },
                "diagnosis_codes": ["large_model_kaggle_real_runtime_verified"],
            }),
            encoding="utf-8",
        )

        report = pack.build_report(pack.parse_args([
            "--mode",
            "evidence-import",
            "--output-dir",
            str(output_dir / "imported"),
            "--run-report",
            str(run_report),
        ]))

        self.assertTrue(report["ok"])
        self.assertTrue(report["real_runtime_verified"])
        self.assertFalse(report["real_7b_runtime_verified"])
        self.assertTrue(report["gpu_runtime_verified"])
        self.assertTrue(report["sharded_path_verified"])
        self.assertFalse(report["core_validation_ready"])
        self.assertEqual(report["model"]["model_id"], "qwen2.5-1.5b-instruct-q4-k-m")
        self.assertEqual(report["runtime"]["backend"], "llama_cpp_rpc")
        self.assertEqual(report["metrics"]["generated_token_count"], 4)
        self.assertIn("large_model_7b_runtime_not_verified", report["diagnosis_codes"])
        self.assertTrue(report["inference_rc_report"]["real_runtime_verified"])
        self.assertFalse(report["inference_rc_report"]["real_7b_runtime_verified"])
        self.assertEqual(report["inference_rc_report"]["alpha_report"]["model_manifest"]["model_size_mb"], 1066)
        check.validate_report(report)

    def test_blocked_import_preserves_rpc_summary(self) -> None:
        output_dir = self._tmp_dir()
        run_report = output_dir / "blocked_run.json"
        run_report.write_text(
            json.dumps({
                "schema": pack.RUN_SCHEMA,
                "ok": False,
                "partial_stage": "large_model_kaggle_tier_run_start",
                "hardware": {
                    "provider": "kaggle",
                    "gpu_count": 2,
                    "gpu_names": ["Tesla T4", "Tesla T4"],
                    "devices": [
                        {"name": "Tesla T4", "memory_total_mb": 15360, "memory_free_mb": 14913},
                        {"name": "Tesla T4", "memory_total_mb": 15360, "memory_free_mb": 14913},
                    ],
                    "kaggle_gpu_verified": True,
                },
                "rpc": {
                    "ok": True,
                    "enabled": True,
                    "worker_count": 1,
                    "requested_worker_count": 1,
                    "rpc_worker_limit": 1,
                    "alive_count": 1,
                    "servers": [{"endpoint": "127.0.0.1:50052", "device_index": 0}],
                },
                "tier_results": [{
                    "schema": pack.RUN_SCHEMA,
                    "tier": "small",
                    "ok": False,
                    "runtime": {
                        "backend": "llama_cpp_rpc",
                        "intended_backend": "llama_cpp_rpc",
                        "runtime_path": "rpc",
                        "rpc_enabled": True,
                        "rpc_endpoints": ["127.0.0.1:50052"],
                        "worker_count": 1,
                        "stage_count": 1,
                    },
                    "runner_step": {
                        "telemetry": {
                            "samples": [{
                                "cgroup_memory": {
                                    "memory.current": 30_000_000_000,
                                    "memory.max": 32_000_000_000,
                                    "memory.peak": 30_500_000_000,
                                    "memory.events": {"oom": 0, "oom_kill": 0},
                                },
                                "gpu_memory": {
                                    "devices": [{
                                        "index": 0,
                                        "memory_total_mb": 15360,
                                        "memory_used_mb": 1024,
                                    }],
                                },
                                "disk": {"free_bytes": 2_000_000_000},
                                "processes": {
                                    "100": {"Name": "llama-cli", "VmRSS": "372000 kB", "VmHWM": "1506508 kB"},
                                },
                            }],
                        },
                    },
                    "diagnosis_codes": ["large_model_kaggle_tier_run_start"],
                }],
                "diagnosis_codes": ["large_model_kaggle_tier_run_start"],
                "blockers": ["large_model_kaggle_no_successful_real_run"],
            }),
            encoding="utf-8",
        )

        report = pack.build_report(pack.parse_args([
            "--mode",
            "evidence-import",
            "--output-dir",
            str(output_dir / "blocked-import"),
            "--run-report",
            str(run_report),
        ]))

        self.assertFalse(report["ok"])
        self.assertEqual(report["largest_successful_tier"], "")
        self.assertEqual(report["run_report"]["validation"]["attempted_scale_tier"], "small")
        self.assertEqual(report["rpc"]["worker_count"], 1)
        self.assertEqual(report["rpc"]["rpc_worker_limit"], 1)
        self.assertEqual(report["run_report"]["rpc"]["requested_worker_count"], 1)
        self.assertTrue(report["resource_pressure_summary"]["cgroup_memory_pressure"])
        self.assertTrue(report["resource_pressure_summary"]["gpu_memory_low_pressure"])
        self.assertIn("large_model_kaggle_cgroup_memory_pressure", report["diagnosis_codes"])
        self.assertIn("large_model_kaggle_container_memory_pressure_not_vram", report["blockers"])
        self.assertEqual(json.loads((output_dir / "blocked-import" / "support_bundle.json").read_text(encoding="utf-8"))["rpc"]["worker_count"], 1)
        check.validate_report(report)

    def test_cli_wrapper_and_bad_args(self) -> None:
        output_dir = self._tmp_dir()
        args = cli.parse_args(["large-model-kaggle-validate", "--mode", "package", "--output-dir", str(output_dir)])
        summary = cli.build_large_model_kaggle_validation(args)

        self.assertFalse(summary["ok"])
        self.assertEqual(summary["cli_schema"], "large_model_kaggle_validation_cli_v1")
        self.assertFalse(summary["real_7b_runtime_verified"])
        self.assertFalse(summary["core_validation_ready"])
        self.assertTrue((output_dir / "large_model_kaggle_validation_cli_summary.json").is_file())

        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            cli.print_large_model_kaggle_validation(summary)
        output = rendered.getvalue()
        self.assertIn("CrowdTensor large-model Kaggle validation", output)
        self.assertIn("real_7b=False", output)

        with self.assertRaises(SystemExit):
            cli.parse_args(["large-model-kaggle-validate", "--max-new-tokens", "9"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["large-model-kaggle-validate", "--cuda-build-jobs", "0"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["large-model-kaggle-validate", "--cuda-build-timeout-seconds", "0"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["large-model-kaggle-validate", "--rpc-worker-limit", "-1"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["large-model-kaggle-validate", "--mode", "evidence-import"])

    def test_hf_cuda_package_uses_redacted_inline_command(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "--mode",
            "package",
            "--output-dir",
            str(output_dir),
            "--tiers",
            "small",
            "--runtime-path",
            "hf-cuda",
            "--hf-cuda-install-compat",
        ])
        report = pack.build_report(args)

        self.assertFalse(report["core_validation_ready"])
        kernel = output_dir / "kaggle-kernel" / "kernel.py"
        self.assertTrue(kernel.is_file())
        source = kernel.read_text(encoding="utf-8")
        compile(source, str(kernel), "exec")
        self.assertIn('"hf-cuda"', source)
        self.assertIn("HF_CUDA_INSTALL_COMPAT = True", source)
        self.assertIn("<inline-python-redacted>", source)
        self.assertIn("sharded_path_verified = False", source)

    def test_source_cuda_package_uses_arch_and_no_vmm(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "--mode",
            "package",
            "--output-dir",
            str(output_dir),
            "--llama-build-mode",
            "source-cuda",
            "--runtime-path",
            "rpc",
            "--cuda-architectures",
            "60",
            "--cuda-build-jobs",
            "1",
            "--cuda-build-timeout-seconds",
            "5400",
            "--rpc-worker-limit",
            "1",
        ])
        pack.build_report(args)

        source = (output_dir / "kaggle-kernel" / "kernel.py").read_text(encoding="utf-8")
        compile(source, str(output_dir / "kaggle-kernel" / "kernel.py"), "exec")
        self.assertIn('CUDA_ARCHITECTURES = "60"', source)
        self.assertIn("CUDA_BUILD_JOBS = 1", source)
        self.assertIn("CUDA_BUILD_TIMEOUT_SECONDS = 5400", source)
        self.assertIn("RPC_WORKER_LIMIT = 1", source)
        self.assertIn('"rpc_worker_limit": int(RPC_WORKER_LIMIT or 0)', source)
        self.assertIn("CUDA_NO_VMM = True", source)
        self.assertIn('env["CUDA_VISIBLE_DEVICES"] = ""', source)
        self.assertIn("large_model_kaggle_tier_download_start", source)
        self.assertIn("large_model_kaggle_tier_run_start", source)
        self.assertIn("def update_tier_progress", source)
        self.assertIn("def resource_snapshot", source)
        self.assertIn("def run_monitored", source)
        self.assertIn("import shutil", source)
        self.assertIn("cgroup_memory_snapshot", source)
        self.assertIn("disk_snapshot", source)
        self.assertIn("compact_llama_runtime", source)
        self.assertIn("rpc-server-", source)
        self.assertIn("tier_before_run", source)
        self.assertIn("include_tail=False", source)
        self.assertIn("os.replace(tmp, target)", source)
        self.assertIn("-DGGML_RPC=ON", source)
        self.assertIn("-DGGML_CUDA_NO_VMM=ON", source)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=", source)
        self.assertIn("large_model_kaggle_llama_cpp_prepare_complete", source)

    def test_kaggle_auto_falls_back_to_full_output_when_report_missing(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            calls.append([str(item) for item in command])
            command_line = " ".join(str(item) for item in command)
            if "kernels push" in command_line:
                return pack.subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="https://www.kaggle.com/code/tester/ct-large-llm-fallback",
                    stderr="",
                )
            if "kernels status" in command_line:
                return pack.subprocess.CompletedProcess(command, 0, stdout="KernelWorkerStatus.ERROR", stderr="")
            if "kernels output" in command_line:
                return pack.subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            if "kernels delete" in command_line:
                return pack.subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            return pack.subprocess.CompletedProcess(command, 1, stdout="", stderr="unexpected")

        args = pack.parse_args([
            "--mode",
            "kaggle-auto",
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "tester",
            "--kaggle-status-timeout-seconds",
            "1",
            "--kaggle-status-poll-interval",
            "1",
        ])
        steps, _package, run_report_path = pack.run_kaggle_auto(args, output_dir=output_dir, runner=fake_runner)

        self.assertEqual(run_report_path, output_dir / "kaggle-output" / "large_model_kaggle_validation_run.json")
        self.assertIn("kaggle_kernel_output", [step["name"] for step in steps])
        self.assertIn("kaggle_kernel_output_full_fallback", [step["name"] for step in steps])
        output_calls = [call for call in calls if call[:3] == ["kaggle", "kernels", "output"]]
        self.assertEqual(len(output_calls), 2)
        self.assertIn("--file-pattern", output_calls[0])

    def test_kaggle_auto_invalid_run_report_becomes_blocked_report(self) -> None:
        output_dir = self._tmp_dir()

        def fake_runner(command, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            command_line = " ".join(str(item) for item in command)
            if "kernels push" in command_line:
                return pack.subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="https://www.kaggle.com/code/tester/ct-large-llm-invalid-json",
                    stderr="",
                )
            if "kernels status" in command_line:
                return pack.subprocess.CompletedProcess(command, 0, stdout="KernelWorkerStatus.ERROR", stderr="")
            if "kernels output" in command_line:
                output_path = Path(command[command.index("-p") + 1])
                output_path.mkdir(parents=True, exist_ok=True)
                (output_path / "large_model_kaggle_validation_run.json").write_text('{"schema": ', encoding="utf-8")
                return pack.subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            if "kernels delete" in command_line:
                return pack.subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            return pack.subprocess.CompletedProcess(command, 1, stdout="", stderr="unexpected")

        args = pack.parse_args([
            "--mode",
            "kaggle-auto",
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "tester",
            "--kaggle-status-timeout-seconds",
            "1",
            "--kaggle-status-poll-interval",
            "1",
        ])
        report = pack.build_report(args, runner=fake_runner)

        self.assertFalse(report["ok"])
        self.assertIn("large_model_kaggle_run_report_invalid_json", report["blockers"])
        self.assertEqual(report["run_report"]["run_report_load"]["reason"], "invalid_json")
        check.validate_report(report)


if __name__ == "__main__":
    unittest.main()
