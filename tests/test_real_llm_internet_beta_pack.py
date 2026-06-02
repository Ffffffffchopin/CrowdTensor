from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import real_llm_internet_beta_check as beta_check
from scripts import real_llm_internet_beta_pack as pack


class RealLlmInternetBetaPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_real_internet_beta_test_"))

    def _args(self, output_dir: Path) -> object:
        return pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--port",
            "9190",
            "--base-port",
            "9191",
            "--request-count",
            "2",
            "--kaggle-owner",
            "xuyuhaosuyi",
            "--kernel-slug-prefix",
            "crowdtensor-real-llm-beta-test",
        ])

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def _external_generation_payload(self, *, generated_tokens: int | None = 16) -> dict:
        generation = {
            "max_new_tokens": 16,
            "generated_text_hash": "sha256:external-16",
            "decoded_tokens_match": True,
            "multi_token_generation_ready": True,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        }
        if generated_tokens is not None:
            generation["generated_token_count"] = generated_tokens
        return {
            "schema": pack.SCHEMA,
            "ok": True,
            "mode": "kaggle-auto",
            "workload": {
                "workload_type": pack.WORKLOAD_TYPE,
                "hf_model_id": "sshleifer/tiny-gpt2",
                "max_new_tokens": 16,
                "request_count": 1,
            },
            "generation": generation,
            "diagnosis_codes": [
                "real_llm_internet_beta_ready",
                "external_runtime_verified",
                "generation_complete",
                "multi_token_generation_ready",
                "decoded_tokens_match",
                "distinct_stage_miners",
                "stage_assignment_valid",
                "kaggle_kernels_deleted",
            ],
        }

    def _external_requeue_payload(self) -> dict:
        payload = self._external_generation_payload(generated_tokens=8)
        payload["diagnosis_codes"].extend([
            "external_stage_requeue_ready",
            "live_stage0_requeue_ready",
            "live_requeue_victim_claim_observed",
            "live_requeue_victim_kernel_deleted",
            "live_requeue_lease_timeout_observed",
            "live_requeue_rescue_result_accepted",
        ])
        payload["live_requeue_summary"] = {
            "enabled": True,
            "failure_mode": "kill-stage0-after-claim",
            "target_stage": "stage0",
            "victim_miner_id": "internet-real-llm-beta-stage0-victim",
            "rescue_miner_id": "internet-real-llm-beta-stage0-rescue",
            "claim_observed": True,
            "victim_kernel_deleted": True,
            "lease_expired": "<redacted>",
            "rescued_result": True,
            "victim_result_accepted": False,
        }
        return payload

    def test_evidence_import_combines_token_target_generation_and_requeue(self) -> None:
        output_dir = self._tmp_dir()
        generation_path = output_dir / "sources" / "generation.json"
        requeue_path = output_dir / "sources" / "requeue.json"
        self._write_json(generation_path, self._external_generation_payload(generated_tokens=16))
        self._write_json(requeue_path, self._external_requeue_payload())

        args = pack.parse_args([
            "--mode",
            "evidence-import",
            "--output-dir",
            str(output_dir / "import"),
            "--generation-report",
            str(generation_path),
            "--requeue-report",
            str(requeue_path),
            "--max-new-tokens",
            "16",
        ])
        report = pack.build_report(args)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["runtime_classification"]["external_runtime_verified"])
        self.assertTrue(report["runtime_classification"]["stage_requeue_verified"])
        self.assertEqual(report["generation"]["generated_token_count"], 16)
        self.assertTrue(report["generation"]["token_target_ready"])
        self.assertIs(report["live_requeue_summary"]["lease_expired"], True)
        self.assertIn("real_llm_internet_beta_evidence_import_ready", report["diagnosis_codes"])
        self.assertIn("external_generated_token_target_ready", report["diagnosis_codes"])

    def test_evidence_import_blocks_when_generation_token_count_missing(self) -> None:
        output_dir = self._tmp_dir()
        generation_path = output_dir / "sources" / "generation.json"
        requeue_path = output_dir / "sources" / "requeue.json"
        self._write_json(generation_path, self._external_generation_payload(generated_tokens=None))
        self._write_json(requeue_path, self._external_requeue_payload())

        args = pack.parse_args([
            "--mode",
            "evidence-import",
            "--output-dir",
            str(output_dir / "import"),
            "--generation-report",
            str(generation_path),
            "--requeue-report",
            str(requeue_path),
            "--max-new-tokens",
            "16",
        ])
        report = pack.build_report(args)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["generation"]["generated_token_count"], 0)
        self.assertFalse(report["generation"]["token_target_ready"])
        self.assertIn("external_generated_token_target_missing", report["diagnosis_codes"])
        self.assertIn("external generated token target", report["not_completed"])

    def test_evidence_import_preserves_imported_cuda_generation_metadata(self) -> None:
        output_dir = self._tmp_dir()
        generation_path = output_dir / "sources" / "gpu-generation.json"
        requeue_path = output_dir / "sources" / "requeue.json"
        generation = self._external_generation_payload(generated_tokens=16)
        generation["schema"] = "public_swarm_gpu_inference_beta_v1"
        generation["workload"]["real_llm_backend"] = "hf_transformers_cuda"
        generation["workload"]["real_llm_partition_mode"] = "stage-local"
        generation["workload"]["torch_spec"] = "torch==2.7.1+cu118 torchvision==0.22.1+cu118"
        generation["workload"]["torch_index_url"] = "https://download.pytorch.org/whl/cu118"
        generation["workload"]["transformers_spec"] = "transformers==4.40.2"
        self._write_json(generation_path, generation)
        self._write_json(requeue_path, self._external_requeue_payload())

        args = pack.parse_args([
            "--mode",
            "evidence-import",
            "--output-dir",
            str(output_dir / "import"),
            "--generation-report",
            str(generation_path),
            "--requeue-report",
            str(requeue_path),
            "--max-new-tokens",
            "16",
        ])
        report = pack.build_report(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["workload"]["real_llm_backend"], "hf_transformers_cuda")
        self.assertEqual(report["workload"]["real_llm_partition_mode"], "stage_local")
        self.assertTrue(report["safety"]["gpu_backend_selected"])
        self.assertFalse(report["safety"]["cpu_only_workload"])
        self.assertFalse(report["safety"]["coordinator_cuda_runtime_required"])
        self.assertTrue(report["safety"]["miner_cuda_runtime_required"])
        self.assertEqual(report["workload"]["torch_spec"], "torch==2.7.1+cu118 torchvision==0.22.1+cu118")
        self.assertEqual(report["workload"]["torch_index_url"], "https://download.pytorch.org/whl/cu118")
        self.assertEqual(report["workload"]["transformers_spec"], "transformers==4.40.2")
        self.assertEqual(report["artifacts"]["generation_report"]["schema"], "public_swarm_gpu_inference_beta_v1")
        self.assertEqual(report["artifacts"]["requeue_report"]["schema"], pack.SCHEMA)

    def test_kaggle_auto_success_aggregates_alpha_external_and_cleanup(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(
            self._args(output_dir),
            runner=beta_check.fake_runner,
            popen_factory=beta_check.FakePopen,  # type: ignore[arg-type]
            ready_probe=beta_check.ready_probe,
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "real_llm_internet_beta_v1")
        self.assertTrue(report["runtime_classification"]["kaggle_auto"])
        self.assertTrue(report["runtime_classification"]["external_runtime_verified"])
        for code in [
            "real_llm_internet_beta_ready",
            "real_llm_internet_alpha_ready",
            "external_runtime_verified",
            "kaggle_kernels_deleted",
            "kaggle_real_llm_stage0_seen",
            "kaggle_real_llm_stage1_seen",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "token_rotation_required",
        ]:
            self.assertIn(code, report["diagnosis_codes"])
        self.assertEqual(set(report["kaggle_lifecycle"]["pushed_refs"]), {"stage0", "stage1"})
        self.assertEqual(
            report["kaggle_lifecycle"]["pushed_refs"]["stage0"],
            "xuyuhaosuyi/crowdtensor-real-llm-beta-check-stage0",
        )
        self.assertTrue(report["kaggle_lifecycle"]["kernels_deleted"])
        self.assertTrue(report["artifacts"]["external_remote_real_llm_sharded_beta_json"]["present"])

    def test_cleanup_failure_blocks_ready_claim(self) -> None:
        output_dir = self._tmp_dir()

        def failing_cleanup_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            if command[:3] == ["kaggle", "kernels", "delete"]:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="delete failed")
            return beta_check.fake_runner(command, **kwargs)

        report = pack.build_report(
            self._args(output_dir),
            runner=failing_cleanup_runner,
            popen_factory=beta_check.FakePopen,  # type: ignore[arg-type]
            ready_probe=beta_check.ready_probe,
        )

        self.assertFalse(report["ok"], report)
        self.assertIn("kaggle_cleanup_failed", report["diagnosis_codes"])
        self.assertIn("real_llm_internet_beta_blocked", report["diagnosis_codes"])
        self.assertNotIn("real_llm_internet_beta_ready", report["diagnosis_codes"])

    def test_external_alpha_failure_blocks_ready_claim(self) -> None:
        output_dir = self._tmp_dir()

        def external_failure_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            if "real_llm_internet_alpha_pack.py" in joined and beta_check.option_value(command, "--mode") == "external-existing":
                child_dir = Path(beta_check.option_value(command, "--output-dir"))
                child_dir.mkdir(parents=True, exist_ok=True)
                return beta_check.completed({
                    "schema": "real_llm_internet_alpha_v1",
                    "ok": False,
                    "mode": "external-existing",
                    "diagnosis_codes": ["real_llm_internet_alpha_blocked"],
                }, returncode=1)
            return beta_check.fake_runner(command, **kwargs)

        report = pack.build_report(
            self._args(output_dir),
            runner=external_failure_runner,
            popen_factory=beta_check.FakePopen,  # type: ignore[arg-type]
            ready_probe=beta_check.ready_probe,
        )

        self.assertFalse(report["ok"], report)
        self.assertIn("real_llm_internet_beta_blocked", report["diagnosis_codes"])
        self.assertNotIn("real_llm_internet_beta_ready", report["diagnosis_codes"])
        self.assertTrue(report["kaggle_lifecycle"]["kernels_deleted"])

    def test_report_redacts_private_env_and_token_fragments(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(
            self._args(output_dir),
            runner=beta_check.fake_runner,
            popen_factory=beta_check.FakePopen,  # type: ignore[arg-type]
            ready_probe=beta_check.ready_probe,
        )
        serialized = json.dumps(report, sort_keys=True)

        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertNotIn("stage0-secret", serialized)
        self.assertNotIn("stage1-secret", serialized)
        self.assertNotIn("SOURCE_TARBALL_B64", serialized)
        self.assertNotIn("MINER_ENV_TEXT", serialized)
        self.assertNotIn("miner.private.env", serialized)
        self.assertNotIn("operator.private.env", serialized)
        self.assertNotIn("miner_registry.json", serialized)

    def test_kaggle_auto_cuda_backend_uses_gpu_runtime_contract(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--port",
            "9320",
            "--base-port",
            "9321",
            "--request-count",
            "1",
            "--kaggle-owner",
            "xuyuhaosuyi",
            "--kernel-slug-prefix",
            "crowdtensor-real-llm-beta-gpu-test",
            "--real-llm-backend",
            "hf_transformers_cuda",
        ])

        report = pack.build_report(
            args,
            runner=beta_check.fake_runner,
            popen_factory=beta_check.FakePopen,  # type: ignore[arg-type]
            ready_probe=beta_check.ready_probe,
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["workload"]["real_llm_backend"], "hf_transformers_cuda")
        self.assertTrue(report["safety"]["gpu_backend_selected"])
        self.assertFalse(report["safety"]["coordinator_cuda_runtime_required"])
        self.assertEqual(report["workload"]["torch_spec"], "torch==2.7.1+cu118 torchvision==0.22.1+cu118")
        self.assertEqual(report["workload"]["torch_index_url"], "https://download.pytorch.org/whl/cu118")
        self.assertEqual(report["workload"]["transformers_spec"], "transformers==4.40.2")
        self.assertIn("public_swarm_gpu_beta_ready", report["diagnosis_codes"])
        self.assertIn("gpu_stage0_ready", report["diagnosis_codes"])
        self.assertIn("gpu_stage1_ready", report["diagnosis_codes"])

    def test_kaggle_auto_cuda_backend_passes_torch_wheel_pin_to_kaggle_package(self) -> None:
        output_dir = self._tmp_dir()
        captured: list[list[str]] = []

        def capturing_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            if "kaggle_real_llm_live_package.py" in " ".join(command):
                captured.append(command)
            return beta_check.fake_runner(command, **kwargs)

        args = pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--port",
            "9320",
            "--base-port",
            "9321",
            "--request-count",
            "1",
            "--kaggle-owner",
            "xuyuhaosuyi",
            "--kernel-slug-prefix",
            "crowdtensor-real-llm-beta-gpu-test",
            "--real-llm-backend",
            "hf_transformers_cuda",
        ])

        report = pack.build_report(
            args,
            runner=capturing_runner,
            popen_factory=beta_check.FakePopen,  # type: ignore[arg-type]
            ready_probe=beta_check.ready_probe,
        )

        self.assertTrue(report["ok"], report)
        self.assertTrue(captured)
        command = captured[0]
        self.assertIn("--torch-spec", command)
        self.assertEqual(command[command.index("--torch-spec") + 1], "torch==2.7.1+cu118 torchvision==0.22.1+cu118")
        self.assertIn("--torch-index-url", command)
        self.assertEqual(command[command.index("--torch-index-url") + 1], "https://download.pytorch.org/whl/cu118")
        self.assertIn("--transformers-spec", command)
        self.assertEqual(command[command.index("--transformers-spec") + 1], "transformers==4.40.2")

    def test_live_stage0_requeue_kills_victim_and_pushes_rescue(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--port",
            "9190",
            "--base-port",
            "9191",
            "--request-count",
            "1",
            "--kaggle-owner",
            "xuyuhaosuyi",
            "--kernel-slug-prefix",
            "crowdtensor-real-llm-beta-test",
            "--failure-mode",
            "kill-stage0-after-claim",
        ])
        state_probe = beta_check.FakeStateProbe(
            target_stage="stage0",
            victim_miner_id="internet-real-llm-beta-stage0-victim",
            rescue_miner_id="internet-real-llm-beta-stage0-rescue",
        )

        report = pack.build_report(
            args,
            runner=beta_check.fake_runner,
            popen_factory=beta_check.FakePopen,  # type: ignore[arg-type]
            ready_probe=beta_check.ready_probe,
            state_probe=state_probe,
        )

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["runtime_classification"]["stage_requeue_verified"])
        self.assertIn("external_stage_requeue_ready", report["diagnosis_codes"])
        self.assertIn("live_stage0_requeue_ready", report["diagnosis_codes"])
        requeue = report["live_requeue_summary"]
        self.assertTrue(requeue["claim_observed"])
        self.assertTrue(requeue["victim_kernel_deleted"])
        self.assertTrue(requeue["lease_expired"])
        self.assertTrue(requeue["rescued_result"])
        self.assertFalse(requeue["victim_result_accepted"])
        self.assertEqual(
            set(report["kaggle_lifecycle"]["pushed_refs"]),
            {"stage0-victim", "stage0-rescue", "stage1"},
        )

    def test_live_stage1_requeue_kills_victim_and_pushes_rescue(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--port",
            "9190",
            "--base-port",
            "9191",
            "--request-count",
            "1",
            "--kaggle-owner",
            "xuyuhaosuyi",
            "--kernel-slug-prefix",
            "crowdtensor-real-llm-beta-test",
            "--failure-mode",
            "kill-stage1-after-claim",
        ])
        state_probe = beta_check.FakeStateProbe(
            target_stage="stage1",
            victim_miner_id="internet-real-llm-beta-stage1-victim",
            rescue_miner_id="internet-real-llm-beta-stage1-rescue",
        )

        report = pack.build_report(
            args,
            runner=beta_check.fake_runner,
            popen_factory=beta_check.FakePopen,  # type: ignore[arg-type]
            ready_probe=beta_check.ready_probe,
            state_probe=state_probe,
        )

        self.assertTrue(report["ok"], report)
        self.assertIn("live_stage1_requeue_ready", report["diagnosis_codes"])
        self.assertEqual(
            set(report["kaggle_lifecycle"]["pushed_refs"]),
            {"stage0", "stage1-victim", "stage1-rescue"},
        )


if __name__ == "__main__":
    unittest.main()
