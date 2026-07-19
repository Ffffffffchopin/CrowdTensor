from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from crowdtensor import cli
from scripts import gpu_swarm_production_like_validation_check as check
from scripts import gpu_swarm_production_like_validation_pack as pack


class GpuSwarmProductionLikeValidationTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_gpu_swarm_prod_like_test_"))

    def _write_json(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _fixture_reports(self, base: Path) -> dict[str, Path]:
        usability = {
            "schema": "gpu_swarm_usability_alpha_v1",
            "ok": True,
            "gpu_swarm_usability_alpha_ready": True,
            "two_gpu_stage_route_ready": True,
            "inference_request_lifecycle_ready": True,
            "public_artifact_safe": True,
        }
        handoff = {
            "schema": "core_technology_handoff_rc_v1",
            "ok": True,
            "core_technology_large_model_alpha_ready": True,
            "safety": {"public_artifact_safe": True},
        }
        status = {
            "schema": "core_technology_validation_status_v1",
            "ok": True,
            "core_validation_ready": True,
            "largest_successful_tier": "14b",
            "handoff_stage_selective_evidence": {
                "seven_b_multi_token_verified": True,
                "seven_b_generated_token_count": 2,
                "seven_b_model_id": "Qwen/Qwen2.5-7B-Instruct",
                "fourteen_b_dual_kaggle_verified": True,
                "fourteen_b_generated_token_count": 1,
                "fourteen_b_model_id": "Qwen/Qwen2.5-14B-Instruct",
                "n_stage_partition_plan_ready": True,
                "stage_selective_performance_report_ready": True,
                "tokens_per_second_effective": 0.003783,
                "latency_effective_elapsed_seconds": 264.345504,
            },
            "seven_b_eight_b_evidence": {
                "real_7b_runtime_verified": True,
                "generated_token_count": 2,
                "memory_peak_mb": 14608,
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
            },
            "safety": {"public_artifact_safe": True},
        }
        control = {
            "schema": "control_user_alpha_v1",
            "ok": True,
            "control_layer_ready": True,
            "user_layer_ready": True,
            "session_lifecycle_ready": True,
            "public_artifact_safe": True,
        }
        generation = {
            "schema": "real_llm_internet_beta_v1",
            "ok": True,
            "generation": {
                "generated_token_count": 16,
                "max_new_tokens": 16,
            },
            "workload": {"hf_model_id": "sshleifer/tiny-gpt2"},
            "diagnosis_codes": [
                "external_gpu_runtime_verified",
                "cuda_runtime_available",
                "gpu_runtime_ready",
                "distinct_stage_miners",
                "stage_assignment_valid",
                "external_stage_requeue_ready",
                "kaggle_kernels_deleted",
            ],
            "live_requeue_summary": {
                "claim_observed": True,
                "victim_deleted": True,
                "rescue_result_accepted": True,
                "victim_result_accepted": False,
            },
            "safety": {"token_rotation_required": True},
        }
        batch_stream = {
            "schema": "public_real_llm_swarm_beta_v1",
            "ok": True,
            "beta": {
                "max_new_tokens": 16,
                "batch": {
                    "batch_generation_ready": True,
                    "request_count": 2,
                },
                "stream": {
                    "events": [{"type": "progress", "index": index} for index in range(16)],
                    "progress": {"max_new_tokens": 16},
                },
            },
        }
        return {
            "usability": self._write_json(base / "gpu_swarm_usability_alpha.json", usability),
            "handoff": self._write_json(base / "core_handoff.json", handoff),
            "status": self._write_json(base / "core_status.json", status),
            "control": self._write_json(base / "control_user_alpha.json", control),
            "generation": self._write_json(base / "real_llm_internet_beta.json", generation),
            "batch_stream": self._write_json(base / "public_real_llm_swarm_beta.json", batch_stream),
        }

    def _pack_args(self, reports: dict[str, Path], output_dir: Path, *extra: str) -> list[str]:
        return [
            "--output-dir",
            str(output_dir),
            "--usability-report",
            str(reports["usability"]),
            "--core-handoff-report",
            str(reports["handoff"]),
            "--core-status-report",
            str(reports["status"]),
            "--control-user-alpha-report",
            str(reports["control"]),
            "--gpu-generation-report",
            str(reports["generation"]),
            "--batch-stream-report",
            str(reports["batch_stream"]),
            *extra,
        ]

    def test_pack_builds_production_like_report_and_check_validates(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)
        output_dir = base / "prod-like"

        report = pack.build_report(pack.parse_args(self._pack_args(reports, output_dir)))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["gpu_swarm_production_like_validation_ready"])
        self.assertTrue(report["production_like_workload_ready"])
        self.assertTrue(report["larger_model_attempted"])
        self.assertEqual(report["largest_successful_model_tier"], "14b")
        self.assertEqual(report["largest_attempted_model_tier"], "32b")
        self.assertEqual(report["larger_model_blocked_reason"], "candidate_requires_more_vram_than_retained_two_gpu_profile")
        self.assertTrue(report["multi_token_decode_ready"])
        self.assertTrue(report["batch_or_multi_request_ready"])
        self.assertTrue(report["stage_requeue_or_failure_recovery_ready"])
        self.assertFalse(report["fresh_gpu_run_performed"])
        self.assertFalse(report["external_runtime_verified"])
        self.assertTrue(report["retained_evidence_imported"])
        self.assertEqual(check.validate_report(report), [])

    def test_larger_model_attempt_records_bounded_32b_blocker(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)
        report = pack.build_report(pack.parse_args(self._pack_args(reports, base / "blocker")))

        attempt = report["larger_model_attempt"]
        feasibility = attempt["feasibility"]
        hardware = attempt["hardware_profile"]
        memory = attempt["memory_estimate"]
        self.assertEqual(attempt["largest_attempted_model_tier"], "32b")
        self.assertFalse(feasibility["feasible_on_current_retained_profile"])
        self.assertLess(hardware["available_vram_per_gpu_mb"], memory["required_vram_mb_per_stage"])
        self.assertEqual(feasibility["max_fresh_model_attempts"], 2)
        self.assertEqual(feasibility["max_requeue_attempts"], 1)
        self.assertLessEqual(feasibility["single_attempt_timeout_minutes"], 60)

    def test_public_artifacts_are_redacted(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)
        output_dir = base / "redaction"
        report = pack.build_report(pack.parse_args(self._pack_args(reports, output_dir)))

        self.assertEqual(pack.public_redaction_errors(report), [])
        scanned = "\n".join(
            path.read_text(encoding="utf-8")
            for path in output_dir.rglob("*")
            if path.is_file()
        )
        for fragment in [
            "CROWDTENSOR_MINER_TOKEN=",
            "CROWDTENSOR_OBSERVER_TOKEN=",
            "CROWDTENSOR_ADMIN_TOKEN=",
            "operator.private.env",
            "miner.private.env",
            "miner_registry.json",
            "kernel.py",
            '"prompt":',
            '"generated_text":',
            '"generated_token_ids":',
            '"activation":',
            '"activations":',
            '"hidden_state":',
            '"logits":',
            '"kv_cache":',
            '"past_key_values":',
            '"lease_token":',
            '"idempotency_key":',
            "SOURCE_TARBALL_B64",
            "MINER_ENV_TEXT",
        ]:
            self.assertNotIn(fragment, scanned)

    def test_check_script_builds_and_validates_report(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)

        result = check.build_check(check.parse_args([
            "--output-dir",
            str(base / "check"),
            "--usability-report",
            str(reports["usability"]),
            "--core-handoff-report",
            str(reports["handoff"]),
            "--core-status-report",
            str(reports["status"]),
            "--control-user-alpha-report",
            str(reports["control"]),
            "--gpu-generation-report",
            str(reports["generation"]),
            "--batch-stream-report",
            str(reports["batch_stream"]),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["gpu_swarm_production_like_validation_ready"])
        self.assertEqual(result["errors"], [])

    def test_cli_wrapper_generates_scale_validation_summary(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)
        output_dir = base / "cli"

        args = cli.parse_args([
            "gpu-swarm",
            "validate-production-like",
            "--output-dir",
            str(output_dir),
            "--usability-report",
            str(reports["usability"]),
            "--core-handoff-report",
            str(reports["handoff"]),
            "--core-status-report",
            str(reports["status"]),
            "--control-user-alpha-report",
            str(reports["control"]),
            "--gpu-generation-report",
            str(reports["generation"]),
            "--batch-stream-report",
            str(reports["batch_stream"]),
            "--json",
        ])
        summary = cli.build_gpu_swarm_production_like_validation(args)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], pack.SCHEMA)
        self.assertEqual(summary["cli_schema"], "gpu_swarm_production_like_validation_cli_v1")
        self.assertEqual(summary["largest_successful_model_tier"], "14b")
        self.assertEqual(summary["largest_attempted_model_tier"], "32b")
        self.assertTrue((output_dir / "gpu_swarm_production_like_validation_cli_summary.json").is_file())

        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            cli.print_gpu_swarm_production_like_validation(summary)
        output = rendered.getvalue()
        self.assertIn("CrowdTensor GPU Swarm Production-Like Validation", output)
        self.assertIn("production_like=True", output)
        self.assertIn("largest_attempted=32b", output)
        self.assertIn("tokens=16/16", output)
        self.assertIn("available_mb_per_gpu=15360", output)

    def test_argument_validation_rejects_unbounded_or_missing_inputs(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)
        with self.assertRaises(SystemExit):
            pack.parse_args(["--max-fresh-model-attempts", "3"])
        with self.assertRaises(SystemExit):
            pack.parse_args(["--max-attempt-timeout-minutes", "61"])
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "gpu-swarm",
                "validate-production-like",
                "--usability-report",
                str(reports["usability"]),
                "--core-handoff-report",
                str(reports["handoff"]),
                "--core-status-report",
                str(reports["status"]),
                "--control-user-alpha-report",
                str(reports["control"]),
                "--gpu-generation-report",
                str(reports["generation"]),
                "--batch-stream-report",
                str(reports["batch_stream"]),
                "--max-fresh-model-attempts",
                "3",
            ])
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "gpu-swarm",
                "validate-production-like",
                "--usability-report",
                "/tmp/missing-gpu-swarm-usability-alpha.json",
            ])


if __name__ == "__main__":
    unittest.main()
