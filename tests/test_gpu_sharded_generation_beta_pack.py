from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import gpu_sharded_generation_beta_pack as pack
from scripts import gpu_sharded_generation_beta_check as check


class GpuShardedGenerationBetaPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_gpu_generation_beta_test_"))

    def _assert_output_scope(self, report: dict, output_dir: Path, mode: str) -> None:
        self.assertFalse(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertTrue(report["output_request"]["public_artifact_safe"])
        self.assertEqual(report["prompt_scope"]["source"], "imported-or-built-in-validation-prompts")
        self.assertIsInstance(report["prompt_scope"]["prompt_count"], int)
        self.assertGreaterEqual(report["prompt_scope"]["prompt_count"], 0)
        self.assertFalse(report["prompt_scope"]["inline_prompt_text"])
        self.assertFalse(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertFalse(report["prompt_scope"]["terminal_logs_local_private"])
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
        markdown = (output_dir / f"gpu_sharded_generation_beta_{mode.replace('-', '_')}.md").read_text(encoding="utf-8")
        self.assertIn("## Output Scope", markdown)
        self.assertIn("prompt scope: `source=imported-or-built-in-validation-prompts", markdown)
        self.assertIn("- answer scope: `no-local-answer`", markdown)

    def test_evidence_import_requires_multi_token_generation_ready(self) -> None:
        output_dir = self._tmp_dir()
        source = output_dir / "source.json"
        source.write_text(json.dumps({
            "schema": "public_swarm_gpu_inference_beta_v1",
            "ok": True,
            "diagnosis_codes": [
                "public_swarm_gpu_beta_ready",
                "hf_transformers_cuda_ready",
                "stage0_partition_loaded",
                "stage1_partition_loaded",
                "partition_parameter_split_valid",
                "stage_local_partition_ready",
                "decoded_tokens_match",
                "distinct_stage_miners",
                "stage_assignment_valid",
                "multi_token_generation_ready",
            ],
            "payload_summaries": {
                "remote_real_llm_sharded_beta": {
                    "generation": {
                        "max_new_tokens": 4,
                        "generated_token_count": 4,
                        "generated_text_hash": "sha256:generation",
                        "multi_token_generation_ready": True,
                    }
                }
            },
        }), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(source),
            "--max-new-tokens",
            "4",
        ]))

        self.assertTrue(report["ok"], report)
        self.assertIn("gpu_sharded_generation_ready", report["diagnosis_codes"])
        self.assertEqual(report["generation"]["generated_token_count"], 4)
        self.assertTrue(report["generation"]["raw_generated_text_public"] is False)
        self._assert_output_scope(report, output_dir, "evidence-import")

    def test_local_loopback_wraps_public_gpu_pack_with_generation_limit(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_gpu_inference_beta_pack.py", command[1])
            self.assertIn("--max-new-tokens", command)
            self.assertEqual(command[command.index("--max-new-tokens") + 1], "3")
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({
                    "schema": "public_swarm_gpu_inference_beta_v1",
                    "ok": True,
                    "mode": "local-loopback",
                    "diagnosis_codes": [
                        "public_swarm_gpu_beta_ready",
                        "hf_transformers_cuda_ready",
                        "stage0_partition_loaded",
                        "stage1_partition_loaded",
                        "partition_parameter_split_valid",
                        "stage_local_partition_ready",
                        "decoded_tokens_match",
                        "distinct_stage_miners",
                        "stage_assignment_valid",
                        "multi_token_generation_ready",
                    ],
                    "payload_summaries": {
                        "remote_real_llm_sharded_beta": {
                            "generation": {
                                "max_new_tokens": 3,
                                "generated_token_count": 3,
                                "generated_text_hash": "sha256:generation",
                                "multi_token_generation_ready": True,
                            }
                        }
                    },
                }) + "\n",
                stderr="",
            )

        report = pack.build_report(pack.parse_args([
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "3",
        ]), runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertIn("gpu_loopback_generation_ready", report["diagnosis_codes"])
        self._assert_output_scope(report, output_dir, "local-loopback")
        self.assertTrue(calls)

    def test_report_blocks_plain_generated_text_leak(self) -> None:
        output_dir = self._tmp_dir()
        source = output_dir / "leaky.json"
        source.write_text(json.dumps({
            "schema": "public_swarm_gpu_inference_beta_v1",
            "ok": True,
            "diagnosis_codes": ["public_swarm_gpu_beta_ready", "hf_transformers_cuda_ready", "multi_token_generation_ready"],
            "generated_text": " leaked",
        }), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(source),
            "--max-new-tokens",
            "2",
        ]))

        self.assertFalse(report["ok"])
        self.assertIn("sensitive_output_detected", report["diagnosis_codes"])

    def test_finds_generation_summary_in_nested_kaggle_payload(self) -> None:
        output_dir = self._tmp_dir()
        source = output_dir / "nested.json"
        source.write_text(json.dumps({
            "schema": "public_swarm_gpu_inference_beta_v1",
            "ok": True,
            "diagnosis_codes": [
                "public_swarm_gpu_beta_ready",
                "public_swarm_gpu_beta_kaggle_auto_ready",
                "external_gpu_runtime_verified",
                "kaggle_kernels_deleted",
                "hf_transformers_cuda_ready",
                "stage0_partition_loaded",
                "stage1_partition_loaded",
                "partition_parameter_split_valid",
                "stage_local_partition_ready",
                "decoded_tokens_match",
                "distinct_stage_miners",
                "stage_assignment_valid",
                "multi_token_generation_ready",
            ],
            "payload_summaries": {
                "real_llm_internet_beta": {
                    "external_alpha": {
                        "live_rc": {
                            "remote_real_llm_beta": {
                                "generation": {
                                    "max_new_tokens": 5,
                                    "generated_token_count": 5,
                                    "generated_text_hash": "sha256:nested",
                                    "multi_token_generation_ready": True,
                                }
                            }
                        }
                    }
                }
            },
        }), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(source),
            "--max-new-tokens",
            "5",
        ]))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["generation"]["generated_token_count"], 5)
        self.assertEqual(report["generation"]["generated_text_hash"], "sha256:nested")
        self._assert_output_scope(report, output_dir, "evidence-import")

    def test_prefers_nested_generation_hash_over_partial_top_level_summary(self) -> None:
        output_dir = self._tmp_dir()
        source = output_dir / "partial.json"
        source.write_text(json.dumps({
            "schema": "public_swarm_gpu_inference_beta_v1",
            "ok": True,
            "diagnosis_codes": [
                "public_swarm_gpu_beta_ready",
                "public_swarm_gpu_beta_kaggle_auto_ready",
                "external_gpu_runtime_verified",
                "kaggle_kernels_deleted",
                "hf_transformers_cuda_ready",
                "stage0_partition_loaded",
                "stage1_partition_loaded",
                "partition_parameter_split_valid",
                "stage_local_partition_ready",
                "decoded_tokens_match",
                "distinct_stage_miners",
                "stage_assignment_valid",
                "multi_token_generation_ready",
            ],
            "generation": {
                "max_new_tokens": 6,
                "generated_token_count": 6,
                "generated_text_hash": None,
                "multi_token_generation_ready": True,
            },
            "payload_summaries": {
                "real_llm_internet_beta": {
                    "remote_existing": {
                        "generation": {
                            "max_new_tokens": 6,
                            "generated_token_count": 6,
                            "generated_text_hash": "sha256:complete",
                            "generated_text_redacted": True,
                            "multi_token_generation_ready": True,
                        }
                    }
                }
            },
        }), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(source),
            "--max-new-tokens",
            "6",
        ]))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["generation"]["generated_text_hash"], "sha256:complete")

    def test_check_script_accepts_synthetic_multitoken_report(self) -> None:
        output_dir = self._tmp_dir()

        result = check.run_check(check.parse_args([
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "4",
            "--include-wrapper-check",
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "gpu_sharded_generation_beta_check_v1")
        self.assertIn("gpu_sharded_generation_beta_check_ready", result["diagnosis_codes"])


if __name__ == "__main__":
    unittest.main()
