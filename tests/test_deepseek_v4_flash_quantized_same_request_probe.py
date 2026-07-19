from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import deepseek_v4_flash_quantized_same_request_probe as probe


class DeepSeekV4FlashQuantizedSameRequestProbeTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="ct_dsv4_same_request_"))

    def _source_report(self) -> dict:
        return {
            "schema": "deepseek_v4_flash_quantized_source_resolver_v1",
            "ok": True,
            "recommended_live_probe_candidate": {
                "candidate_id": "iq1-s-xl-gguf",
                "repo": "teamblobfish/DeepSeek-V4-Flash-GGUF",
                "quant": "IQ1_S-XL",
                "runtime_backend": "llama_cpp_v4_fork",
                "runtime_fork": "cchuter/llama.cpp@feat/v4-port-cuda",
                "total_size_gb": 61.540805,
                "split_file_count": 2,
                "files": [
                    {"path": "IQ1_S-XL/DeepSeek-V4-Flash-IQ1_S-XL-00001-of-00002.gguf", "size_bytes": 49952658368, "size_gb": 49.952658},
                    {"path": "IQ1_S-XL/DeepSeek-V4-Flash-IQ1_S-XL-00002-of-00002.gguf", "size_bytes": 11588146976, "size_gb": 11.588147},
                ],
                "blockers": ["candidate_exceeds_single_t4x2_memory_budget"],
            },
        }

    def test_preflight_renders_live_kaggle_kernel_without_success_overclaim(self) -> None:
        out = self._tmp_dir()
        source_path = out / "source.json"
        source_path.write_text(json.dumps(self._source_report()), encoding="utf-8")
        args = probe.parse_args([
            "--mode",
            "preflight",
            "--output-dir",
            str(out),
            "--source-resolver-report",
            str(source_path),
            "--kaggle-owner",
            "tester",
            "--patch-rpc-op-count-guard",
            "--keep-private-package",
        ])
        candidate = probe.candidate_from_source(probe.load_json(source_path), args)
        report = probe.run_preflight(args, output_dir=out, candidate=candidate)

        self.assertFalse(report["same_request_decode_verified"])
        self.assertFalse(report["live_run_performed"])
        self.assertEqual(report["failure_stage"], "same_request_live_probe_not_started")
        self.assertIn("deepseek_v4_flash_quantized_same_request_live_run_not_started", report["blockers"])
        kernel = out / "private-kaggle-kernel" / "kernel.py"
        source = kernel.read_text(encoding="utf-8")
        compile(source, str(kernel), "exec")
        self.assertIn("teamblobfish/DeepSeek-V4-Flash-GGUF", source)
        self.assertIn("DeepSeek-V4-Flash-IQ1_S-XL-00001-of-00002.gguf", source)
        self.assertIn("feat/v4-port-cuda", source)
        self.assertIn("-DGGML_RPC=ON", source)
        self.assertIn("--rpc", source)
        self.assertIn("COLAB_RPC_HOST", source)
        self.assertIn("rpc_op_count_guard_accepts_101_or_102", source)

    def test_preflight_renders_runtime_tarball_fast_path_for_colab_and_kaggle(self) -> None:
        out = self._tmp_dir()
        source_path = out / "source.json"
        source_path.write_text(json.dumps(self._source_report()), encoding="utf-8")
        args = probe.parse_args([
            "--mode",
            "preflight",
            "--output-dir",
            str(out),
            "--source-resolver-report",
            str(source_path),
            "--kaggle-owner",
            "tester",
            "--runtime-tarball-url",
            "https://example.test/runtime.tar.gz",
            "--runtime-tarball-sha256",
            "sha256:" + "a" * 64,
            "--keep-private-package",
        ])
        candidate = probe.candidate_from_source(probe.load_json(source_path), args)
        report = probe.run_preflight(args, output_dir=out, candidate=candidate)

        self.assertFalse(report["same_request_decode_verified"])
        self.assertTrue(report["runtime"]["runtime_tarball_requested"])
        colab = probe.render_colab_rpc_code(args)
        kaggle = (out / "private-kaggle-kernel" / "kernel.py").read_text(encoding="utf-8")
        compile(colab, "<colab-rpc-tarball>", "exec")
        compile(kaggle, "<kaggle-tarball>", "exec")
        self.assertIn("RUNTIME_TARBALL_URL", colab)
        self.assertIn("prepare_runtime_from_tarball", colab)
        self.assertIn("colab_runtime_tarball_prepare_failed", colab)
        self.assertIn("KEEPALIVE_AFTER_READY = True", colab)
        self.assertIn("ready_keepalive", colab)
        self.assertIn("rpc_hello_probe", colab)
        self.assertIn("colab_bore_rpc_hello_failed", colab)
        self.assertIn("RUNTIME_TARBALL_URL", kaggle)
        self.assertIn("prepare_runtime_from_tarball", kaggle)
        self.assertIn("kaggle_runtime_tarball_prepare_failed", kaggle)
        self.assertIn("colab_rpc_hello_pre_download", kaggle)
        self.assertIn("SKIP_MODEL_DOWNLOAD_ON_RPC_HELLO_FAILURE = True", kaggle)
        self.assertIn("colab_rpc_reachable_before_llama", kaggle)
        self.assertIn("colab_rpc_endpoint_lost_after_model_download", kaggle)
        self.assertIn("kaggle_local_rpc_hello_failed", kaggle)
        self.assertIn("runtime_source", kaggle)
        self.assertIn("CLIENT_CUDA_VISIBLE", kaggle)
        self.assertIn("INCLUDE_CPU_RPC_ENDPOINT = False", kaggle)
        self.assertIn("provider_stage_counts", kaggle)

    def test_colab_rpc_code_is_public_safe_and_starts_rpc_tunnel(self) -> None:
        args = probe.parse_args([
            "--mode",
            "preflight",
            "--kaggle-owner",
            "tester",
            "--patch-rpc-op-count-guard",
        ])
        source = probe.render_colab_rpc_code(args)

        compile(source, "<colab-rpc>", "exec")
        self.assertIn("bore", source)
        self.assertIn("rpc-server", source)
        self.assertIn("-DGGML_RPC=ON", source)
        self.assertIn("CT_DSV4_COLAB_RPC_WORKER", source)
        self.assertIn("remote_endpoint_public", source)
        self.assertNotIn("runtime_proxy_token", source)

    def test_runtime_tarball_path_and_url_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            probe.parse_args([
                "--runtime-tarball-path",
                "/tmp/runtime.tar.gz",
                "--runtime-tarball-url",
                "https://example.test/runtime.tar.gz",
            ])

    def test_colab_background_launch_and_poll_code_are_short_public_safe_cells(self) -> None:
        args = probe.parse_args([
            "--mode",
            "preflight",
            "--kaggle-owner",
            "tester",
            "--patch-rpc-op-count-guard",
        ])
        worker = probe.render_colab_rpc_code(args)
        launch = probe.render_colab_background_launch_code(worker)
        poll = probe.render_colab_background_poll_code(args)

        compile(launch, "<colab-launch>", "exec")
        compile(poll, "<colab-poll>", "exec")
        self.assertEqual(args.colab_build_mode, "background")
        self.assertIn("subprocess.Popen", launch)
        self.assertIn("worker.py", launch)
        self.assertIn("CT_DSV4_COLAB_RPC_BACKGROUND_LAUNCH", launch)
        self.assertIn("CT_DSV4_COLAB_RPC_WORKER", poll)
        self.assertIn("background_pid_alive", poll)
        self.assertIn("remote_endpoint_public", poll)
        self.assertNotIn("runtime_proxy_token", launch + poll)

    def test_success_gate_requires_all_three_providers_and_generated_token(self) -> None:
        out = self._tmp_dir()
        args = probe.parse_args(["--mode", "kaggle-auto", "--output-dir", str(out), "--kaggle-owner", "tester"])
        candidate = {
            "candidate_id": "iq1-s-xl-gguf",
            "repo": "teamblobfish/DeepSeek-V4-Flash-GGUF",
            "quant": "IQ1_S-XL",
            "total_size_gb": 61.540805,
            "split_file_count": 2,
            "files": [{"path": "a.gguf", "size_bytes": 1, "size_gb": 0.0}],
            "blockers": [],
        }
        colab = {"schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1", "ok": True, "remote_host": "bore.pub", "remote_port": 12345, "blockers": [], "public_artifact_safe": True}
        package = {"kernel_ref": "tester/kernel", "kernel_slug": "kernel"}
        steps = [{"name": "kaggle_kernel_push", "ok": True}, {"name": "kaggle_kernel_delete", "ok": True}]
        worker = {
            "schema": probe.WORKER_SCHEMA,
            "ok": True,
            "same_request_decode_verified": True,
            "generated_token_count": 1,
            "accepted_providers": ["kaggle_cuda", "colab_cuda", "cpu"],
            "provider_stage_counts": {"kaggle_cuda": 2, "colab_cuda": 1, "cpu": 1},
            "blockers": [],
            "diagnosis_codes": ["deepseek_v4_flash_quantized_same_request_decode_verified"],
            "llama_cli_present": True,
            "rpc_server_present": True,
            "public_artifact_safe": True,
        }
        report = probe.build_report(
            args,
            output_dir=out,
            candidate=candidate,
            colab_rpc=colab,
            package=package,
            steps=steps,
            worker_report=worker,
            live_run_performed=True,
        )

        self.assertTrue(report["same_request_decode_verified"])
        self.assertEqual(report["failure_stage"], "")
        self.assertEqual(set(report["accepted_providers"]), {"kaggle_cuda", "colab_cuda", "cpu"})
        self.assertFalse(report["runtime"]["include_cpu_rpc_endpoint"])
        self.assertEqual(report["runtime"]["client_cuda_visible"], "0,1")

        worker["accepted_providers"] = ["kaggle_cuda", "cpu"]
        report = probe.build_report(
            args,
            output_dir=out,
            candidate=candidate,
            colab_rpc=colab,
            package=package,
            steps=steps,
            worker_report=worker,
            live_run_performed=True,
        )
        self.assertFalse(report["same_request_decode_verified"])
        self.assertIn("deepseek_v4_flash_quantized_same_request_decode_not_verified", report["blockers"])

    def test_colab_authuser_fallback_tries_bounded_list_until_success(self) -> None:
        args = probe.parse_args([
            "--mode",
            "kaggle-auto",
            "--kaggle-owner",
            "tester",
            "--colab-authusers",
            "0,1",
        ])
        calls: list[str] = []

        def fake_background(inner_args):
            calls.append(str(inner_args.colab_authuser))
            if str(inner_args.colab_authuser) == "0":
                return {
                    "schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1",
                    "ok": False,
                    "duration_seconds": 1.0,
                    "remote_port": 0,
                    "manager": {"ok": False, "blocker": "colab_cuda_reacquire_failed", "public_artifact_safe": True},
                    "worker": {},
                    "blockers": ["colab_rpc_worker_background_launch_failed"],
                    "public_artifact_safe": True,
                }
            return {
                "schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1",
                "ok": True,
                "duration_seconds": 2.0,
                "remote_host": "bore.pub",
                "remote_port": 12345,
                "manager": {"ok": True, "blocker": "", "public_artifact_safe": True},
                "worker": {"ok": True, "public_artifact_safe": True},
                "blockers": [],
                "public_artifact_safe": True,
            }

        with mock.patch.object(probe, "start_colab_rpc_worker_background", side_effect=fake_background):
            result = probe.start_colab_rpc_worker(args)

        self.assertTrue(result["ok"], result)
        self.assertEqual(calls, ["0", "1"])
        self.assertEqual(result["authuser"], "1")
        self.assertEqual(result["authuser_fallback"]["attempt_count"], 2)
        self.assertEqual(result["authuser_fallback"]["selected_authuser"], "1")
        self.assertEqual([item["authuser"] for item in result["authuser_fallback"]["attempts"]], ["0", "1"])

    def test_colab_accelerator_authuser_fallback_tries_matrix_until_success(self) -> None:
        args = probe.parse_args([
            "--mode",
            "kaggle-auto",
            "--kaggle-owner",
            "tester",
            "--colab-accelerators",
            "T4,L4",
            "--colab-authusers",
            "0",
        ])
        calls: list[tuple[str, str]] = []

        def fake_background(inner_args):
            calls.append((str(inner_args.colab_accelerator), str(inner_args.colab_authuser)))
            if str(inner_args.colab_accelerator) == "T4":
                return {
                    "schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1",
                    "ok": False,
                    "duration_seconds": 1.0,
                    "remote_port": 0,
                    "manager": {"ok": False, "blocker": "colab_cuda_reacquire_failed", "public_artifact_safe": True},
                    "worker": {},
                    "blockers": ["colab_rpc_worker_background_launch_failed"],
                    "public_artifact_safe": True,
                }
            return {
                "schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1",
                "ok": True,
                "duration_seconds": 2.0,
                "remote_host": "bore.pub",
                "remote_port": 12345,
                "manager": {"ok": True, "blocker": "", "public_artifact_safe": True},
                "worker": {"ok": True, "public_artifact_safe": True},
                "blockers": [],
                "public_artifact_safe": True,
            }

        with mock.patch.object(probe, "start_colab_rpc_worker_background", side_effect=fake_background):
            result = probe.start_colab_rpc_worker(args)

        self.assertTrue(result["ok"], result)
        self.assertEqual(calls, [("T4", "0"), ("L4", "0")])
        self.assertEqual(result["accelerator"], "L4")
        self.assertEqual(result["authuser"], "0")
        self.assertEqual(result["colab_fallback"]["selected_accelerator"], "L4")
        self.assertEqual(result["colab_fallback"]["attempted_targets"], [{"accelerator": "T4", "authuser": "0"}, {"accelerator": "L4", "authuser": "0"}])

    def test_colab_authuser_fallback_reports_exhaustion_public_safe(self) -> None:
        args = probe.parse_args([
            "--mode",
            "kaggle-auto",
            "--kaggle-owner",
            "tester",
            "--colab-authusers",
            "0,1",
        ])

        def fake_background(inner_args):
            return {
                "schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1",
                "ok": False,
                "duration_seconds": 1.0,
                "remote_port": 0,
                "manager": {"ok": False, "blocker": "colab_cuda_reacquire_failed", "public_artifact_safe": True},
                "worker": {},
                "blockers": ["colab_rpc_worker_background_launch_failed"],
                "public_artifact_safe": True,
            }

        with mock.patch.object(probe, "start_colab_rpc_worker_background", side_effect=fake_background):
            result = probe.start_colab_rpc_worker(args)

        self.assertFalse(result["ok"], result)
        self.assertIn("colab_rpc_worker_fallback_exhausted", result["blockers"])
        self.assertIn("colab_rpc_worker_authuser_fallback_exhausted", result["blockers"])
        self.assertEqual(result["colab_fallback"]["attempted_targets"], [{"accelerator": "T4", "authuser": "0"}, {"accelerator": "T4", "authuser": "1"}])
        self.assertEqual(result["authuser_fallback"]["attempted_authusers"], ["0", "1"])
        self.assertEqual(result["authuser_fallback"]["selected_authuser"], "")
        self.assertTrue(result["public_artifact_safe"])


if __name__ == "__main__":
    unittest.main()
