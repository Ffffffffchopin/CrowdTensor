from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import deepseek_v4_flash_kaggle_llama_v4_build_preflight as probe


class DeepSeekV4FlashKaggleLlamaV4BuildPreflightTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="ct_dsv4_kaggle_llama_build_"))

    def test_build_package_renders_v4_fork_cuda_rpc_kernel(self) -> None:
        out = self._tmp_dir()
        args = probe.parse_args([
            "--output-dir",
            str(out),
            "--kaggle-owner",
            "tester",
            "--kernel-slug-prefix",
            "ct-dsv4-test",
            "--repo-url",
            "https://github.com/cchuter/llama.cpp.git",
            "--branch",
            "feat/v4-port-cuda",
            "--cuda-build-jobs",
            "1",
            "--patch-rpc-op-count-guard",
            "--export-runtime-tarball",
        ])
        package = probe.build_package(args, output_dir=out)

        kernel = package["kernel_dir"] / "kernel.py"
        source = kernel.read_text(encoding="utf-8")
        compile(source, str(kernel), "exec")
        self.assertIn("feat/v4-port-cuda", source)
        self.assertIn("-DGGML_CUDA=ON", source)
        self.assertIn("-DGGML_RPC=ON", source)
        self.assertIn("--target", source)
        self.assertIn("llama-cli", source)
        self.assertIn("rpc-server", source)
        self.assertIn("deepseek_v4_flash_kaggle_llama_v4_build_worker.json", source)
        self.assertIn("PATCH_RPC_OP_COUNT_GUARD = True", source)
        self.assertIn("rpc_op_count_guard_accepts_101_or_102", source)
        self.assertIn("EXPORT_RUNTIME_TARBALL = True", source)
        self.assertIn("deepseek-v4-flash-llama-v4-runtime.tar.gz", source)
        self.assertIn("runtime-bundle.json", source)
        self.assertEqual(package["metadata"]["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(package["metadata"]["enable_internet"], "true")

    def test_success_report_requires_worker_and_cleanup(self) -> None:
        out = self._tmp_dir()
        args = probe.parse_args(["--output-dir", str(out), "--kaggle-owner", "tester"])
        package = {"kernel_ref": "tester/kernel", "kernel_slug": "kernel"}
        worker = {
            "schema": probe.WORKER_SCHEMA,
            "ok": True,
            "stage": "final",
            "repo_url": "https://github.com/cchuter/llama.cpp.git",
            "branch": "feat/v4-port-cuda",
            "commit_hash_public": "a" * 40,
            "cuda_architectures": "75",
            "hardware": {"gpu_count": 2, "kaggle_gpu_verified": True},
            "llama_cli_present": True,
            "rpc_server_present": True,
            "llama_cli_supports_rpc": True,
            "llama_cli_supports_tensor_split": True,
            "runtime_tarball": {
                "ok": True,
                "tarball_name": "deepseek-v4-flash-llama-v4-runtime.tar.gz",
                "tarball_size_bytes": 123,
                "tarball_sha256": "sha256:" + "b" * 64,
            },
            "steps": {"cmake_configure": {"ok": True}, "cmake_build": {"ok": True}},
            "temp_cleanup": {"ok": True},
            "blockers": [],
            "diagnosis_codes": ["kaggle_llama_v4_rpc_build_ready"],
            "public_artifact_safe": True,
        }
        steps = [
            {"name": "kaggle_kernel_push", "ok": True},
            {"name": "kaggle_kernel_output", "ok": True},
            {"name": "kaggle_kernel_delete", "ok": True},
        ]
        report = probe.build_report(args, output_dir=out, package=package, steps=steps, worker_report=worker)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["llama_v4_runtime_build_ready"])
        self.assertEqual(report["blockers"], [])
        self.assertTrue(report["worker_summary"]["cmake_build_ok"])
        self.assertFalse(report["runtime_artifact"]["runtime_tarball_requested"])
        self.assertTrue(report["runtime_artifact"]["runtime_tarball_exported"])

    def test_export_runtime_tarball_request_requires_worker_tarball(self) -> None:
        out = self._tmp_dir()
        args = probe.parse_args([
            "--output-dir",
            str(out),
            "--kaggle-owner",
            "tester",
            "--export-runtime-tarball",
        ])
        package = {"kernel_ref": "tester/kernel", "kernel_slug": "kernel"}
        worker = {
            "schema": probe.WORKER_SCHEMA,
            "ok": True,
            "stage": "final",
            "hardware": {"gpu_count": 2, "kaggle_gpu_verified": True},
            "llama_cli_present": True,
            "rpc_server_present": True,
            "llama_cli_supports_rpc": True,
            "llama_cli_supports_tensor_split": True,
            "steps": {"cmake_configure": {"ok": True}, "cmake_build": {"ok": True}},
            "temp_cleanup": {"ok": True},
            "blockers": [],
            "diagnosis_codes": ["kaggle_llama_v4_rpc_build_ready"],
            "public_artifact_safe": True,
        }
        steps = [
            {"name": "kaggle_kernel_push", "ok": True},
            {"name": "kaggle_kernel_output", "ok": True},
            {"name": "kaggle_kernel_delete", "ok": True},
        ]
        report = probe.build_report(args, output_dir=out, package=package, steps=steps, worker_report=worker)

        self.assertFalse(report["ok"])
        self.assertFalse(report["llama_v4_runtime_build_ready"])
        self.assertTrue(report["runtime_artifact"]["runtime_tarball_requested"])
        self.assertFalse(report["runtime_artifact"]["runtime_tarball_exported"])
        self.assertIn("kaggle_llama_v4_runtime_tarball_export_missing", report["blockers"])

    def test_missing_worker_report_is_blocker_not_success(self) -> None:
        out = self._tmp_dir()
        args = probe.parse_args(["--output-dir", str(out), "--kaggle-owner", "tester"])
        package = {"kernel_ref": "tester/kernel", "kernel_slug": "kernel"}
        steps = [{"name": "kaggle_kernel_push", "ok": True}, {"name": "kaggle_kernel_delete", "ok": True}]
        report = probe.build_report(args, output_dir=out, package=package, steps=steps, worker_report={})

        self.assertFalse(report["ok"])
        self.assertFalse(report["llama_v4_runtime_build_ready"])
        self.assertIn("kaggle_llama_v4_worker_report_missing", report["blockers"])


if __name__ == "__main__":
    unittest.main()
