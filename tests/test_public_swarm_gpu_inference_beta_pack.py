from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import public_swarm_gpu_inference_beta_pack as pack


class PublicSwarmGpuInferenceBetaPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_public_swarm_gpu_beta_test_"))

    def _assert_output_scope(self, report: dict, output_dir: Path, mode: str) -> None:
        prompt_inline_modes = {"local-loopback", "verify"}
        prompt_inline = mode in prompt_inline_modes
        self.assertFalse(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertTrue(report["output_request"]["public_artifact_safe"])
        self.assertEqual(report["prompt_scope"]["source"], "prompt-texts" if prompt_inline else "none")
        self.assertEqual(report["prompt_scope"]["prompt_count"], len(pack.DEFAULT_PROMPTS) if prompt_inline else 0)
        self.assertIs(report["prompt_scope"]["inline_prompt_text"], prompt_inline)
        self.assertIs(report["prompt_scope"]["terminal_next_commands_local_private"], prompt_inline)
        self.assertIs(report["prompt_scope"]["terminal_logs_local_private"], prompt_inline)
        self.assertTrue(report["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertFalse(report["prompt_scope"]["prompt_file_path_public"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(report["prompt_scope"]["public_artifact_safe"])
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertFalse(report["answer_scope"]["visible_in_terminal"])
        self.assertFalse(report["answer_scope"]["terminal_only"])
        self.assertEqual(report["answer_scope"]["saved_json_display"], "hash-only")
        self.assertEqual(report["answer_scope"]["saved_markdown_display"], "hash-only")
        self.assertTrue(report["answer_scope"]["public_artifact_safe"])
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(report["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(report["shareable_summary"]["raw_generated_text_public"])
        self.assertFalse(report["shareable_summary"]["generated_token_ids_public"])
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertFalse(report["shareable_summary"]["local_answer_terminal_only"])
        markdown = (output_dir / f"public_swarm_gpu_inference_beta_{mode.replace('-', '_')}.md").read_text(encoding="utf-8")
        self.assertIn("## Output Scope", markdown)
        self.assertIn(f"prompt scope: `source={'prompt-texts' if prompt_inline else 'none'}", markdown)
        self.assertIn("- answer scope: `no-local-answer`", markdown)

    def test_local_smoke_is_ci_safe_without_gpu_claim(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args([
            "local-smoke",
            "--output-dir",
            str(output_dir),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "public_swarm_gpu_inference_beta_v1")
        self.assertEqual(report["beta"]["backend"], "hf_transformers_cuda")
        self.assertIn("public_swarm_gpu_beta_smoke_ready", report["diagnosis_codes"])
        self.assertIn("gpu_runtime_smoke_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_gpu_beta_ready", report["diagnosis_codes"])
        self._assert_output_scope(report, output_dir, "local-smoke")

    def test_local_loopback_wraps_cuda_real_llm_route(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_real_llm_sharded_beta_pack.py", command[1])
            self.assertIn("--real-llm-backend", command)
            self.assertEqual(command[command.index("--real-llm-backend") + 1], "hf_transformers_cuda")
            child_dir = Path(command[command.index("--output-dir") + 1])
            child_dir.mkdir(parents=True, exist_ok=True)
            (child_dir / "remote_real_llm_sharded_beta.json").write_text("{}", encoding="utf-8")
            (child_dir / "remote_real_llm_sharded_beta.md").write_text("# gpu\n", encoding="utf-8")
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({
                    "schema": "remote_real_llm_sharded_beta_v1",
                    "ok": True,
                    "mode": "remote-loopback",
                    "diagnosis_codes": [
                        "remote_real_llm_sharded_ready",
                        "remote_real_llm_sharded_loopback_ready",
                        "real_llm_sharded_ready",
                        "real_llm_artifact_ready",
                        "activation_transport_ready",
                        "baseline_match",
                        "decoded_tokens_match",
                        "distinct_stage_miners",
                        "stage_assignment_valid",
                        "cuda_runtime_available",
                        "hf_transformers_cuda_ready",
                        "stage_local_partition_ready",
                        "stage0_partition_loaded",
                        "stage1_partition_loaded",
                        "partition_parameter_split_valid",
                    ],
                    "payload_summaries": {
                        "remote_loopback_real_llm_sharded_inference": {
                            "artifact": {
                                "backend": "hf_transformers_cuda",
                                "model_id": "sshleifer/tiny-gpt2",
                                "partition_mode": "stage_local",
                                "stage_local_partition_ready": True,
                                "partition_parameter_split_valid": True,
                                "stage0_parameter_count": 512,
                                "stage1_parameter_count": 512,
                                "full_model_parameter_count": 2048,
                            },
                            "stage_assignment": {
                                "stage0_miner_id": "gpu-stage0",
                                "stage1_miner_id": "gpu-stage1",
                                "distinct_stage_miners": True,
                                "stage_assignment_valid": True,
                            },
                        }
                    },
                }) + "\n",
                stderr="",
            )

        report = pack.build_report(pack.parse_args([
            "local-loopback",
            "--output-dir",
            str(output_dir),
        ]), runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertIn("public_swarm_gpu_beta_ready", report["diagnosis_codes"])
        self.assertIn("gpu_stage0_ready", report["diagnosis_codes"])
        self.assertIn("gpu_stage1_ready", report["diagnosis_codes"])
        self._assert_output_scope(report, output_dir, "local-loopback")
        self.assertTrue(calls)

    def test_local_loopback_redacts_prompt_texts_from_failure_tails(self) -> None:
        output_dir = self._tmp_dir()
        prompts = "private gpu prompt one,private gpu prompt two"

        def failing_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            del command
            return subprocess.CompletedProcess(
                args=["cmd"],
                returncode=1,
                stdout="private gpu prompt one failed\n",
                stderr="private gpu prompt two failed\n",
            )

        report = pack.build_report(pack.parse_args([
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--prompt-texts",
            prompts,
        ]), runner=failing_runner)
        markdown = (output_dir / "public_swarm_gpu_inference_beta_local_loopback.md").read_text(encoding="utf-8")
        encoded = json.dumps({"report": report, "markdown": markdown}, sort_keys=True)

        self.assertFalse(report["ok"])
        self.assertNotIn("private gpu prompt one", encoded)
        self.assertNotIn("private gpu prompt two", encoded)
        self.assertIn("<redacted>", encoded)
        self.assertEqual(report["prompt_scope"]["source"], "prompt-texts")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 2)

    def test_kaggle_package_writes_private_gpu_runbook(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args([
            "kaggle-package",
            "--output-dir",
            str(output_dir),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertIn("kaggle_gpu_package_ready", report["diagnosis_codes"])
        self.assertTrue((output_dir / "kaggle-gpu-package" / "KAGGLE_GPU_RUNBOOK.md").is_file())
        self.assertTrue((output_dir / "kaggle-gpu-package" / "kaggle_gpu_stage_miner.py").is_file())
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("<stage0-token>", serialized)
        self.assertNotIn("<stage1-token>", serialized)
        self._assert_output_scope(report, output_dir, "kaggle-package")

    def test_evidence_import_requires_gpu_ready_codes(self) -> None:
        output_dir = self._tmp_dir()
        source = output_dir / "source.json"
        source.write_text(json.dumps({
            "schema": "public_swarm_gpu_inference_beta_v1",
            "ok": True,
            "diagnosis_codes": [
                "public_swarm_gpu_beta_ready",
                "hf_transformers_cuda_ready",
            ],
        }), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(source),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertIn("public_swarm_gpu_beta_evidence_import_ready", report["diagnosis_codes"])
        self.assertIn("external_gpu_runtime_verified", report["diagnosis_codes"])
        self._assert_output_scope(report, output_dir, "evidence-import")

    def test_kaggle_auto_wraps_cuda_internet_beta_and_requires_cleanup(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("real_llm_internet_beta_pack.py", command[1])
            self.assertIn("--real-llm-backend", command)
            self.assertEqual(command[command.index("--real-llm-backend") + 1], "hf_transformers_cuda")
            self.assertIn("--kaggle-owner", command)
            self.assertIn("--torch-spec", command)
            self.assertEqual(command[command.index("--torch-spec") + 1], "torch==2.7.1+cu118 torchvision==0.22.1+cu118")
            self.assertIn("--torch-index-url", command)
            self.assertEqual(command[command.index("--torch-index-url") + 1], "https://download.pytorch.org/whl/cu118")
            self.assertIn("--transformers-spec", command)
            self.assertEqual(command[command.index("--transformers-spec") + 1], "transformers==4.40.2")
            child_dir = Path(command[command.index("--output-dir") + 1])
            remote = child_dir / "external-alpha" / "live-rc" / "remote-real-llm-runtime"
            remote.mkdir(parents=True, exist_ok=True)
            (remote / "remote_real_llm_sharded_beta.json").write_text("{}", encoding="utf-8")
            (child_dir / "real_llm_internet_beta.json").write_text("{}", encoding="utf-8")
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({
                    "schema": "real_llm_internet_beta_v1",
                    "ok": True,
                    "mode": "kaggle-auto",
                    "workload": {"real_llm_backend": "hf_transformers_cuda"},
                    "runtime_classification": {"kaggle_auto": True, "external_runtime_verified": True},
                    "kaggle_lifecycle": {
                        "owner": "xuyuhaosuyi",
                        "kernel_slug_prefix": "crowdtensor-public-swarm-gpu-beta-test",
                        "expected_push_count": 2,
                        "pushed_refs": {
                            "stage0": "xuyuhaosuyi/crowdtensor-public-swarm-gpu-beta-test-stage0",
                            "stage1": "xuyuhaosuyi/crowdtensor-public-swarm-gpu-beta-test-stage1",
                        },
                        "kernels_deleted": True,
                        "cleanup_required": True,
                    },
                    "diagnosis_codes": [
                        "real_llm_internet_beta_ready",
                        "external_runtime_verified",
                        "cuda_runtime_available",
                        "hf_transformers_cuda_ready",
                        "decoded_tokens_match",
                        "distinct_stage_miners",
                        "stage_assignment_valid",
                        "stage_local_partition_ready",
                        "stage0_partition_loaded",
                        "stage1_partition_loaded",
                        "partition_parameter_split_valid",
                        "kaggle_kernels_deleted",
                        "token_rotation_required",
                    ],
                }) + "\n",
                stderr="",
            )

        report = pack.build_report(pack.parse_args([
            "kaggle-auto",
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "xuyuhaosuyi",
            "--kernel-slug-prefix",
            "crowdtensor-public-swarm-gpu-beta-test",
        ]), runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertIn("public_swarm_gpu_beta_kaggle_auto_ready", report["diagnosis_codes"])
        self.assertIn("external_gpu_runtime_verified", report["diagnosis_codes"])
        self.assertTrue(report["kaggle_lifecycle"]["kernels_deleted"])
        self.assertTrue(report["safety"]["token_rotation_required"])
        self.assertTrue(report["safety"]["cuda_torch_wheel_pinned"])
        self.assertEqual(report["beta"]["torch_spec"], "torch==2.7.1+cu118 torchvision==0.22.1+cu118")
        self.assertEqual(report["beta"]["transformers_spec"], "transformers==4.40.2")
        self._assert_output_scope(report, output_dir, "kaggle-auto")
        self.assertTrue(calls)


if __name__ == "__main__":
    unittest.main()
