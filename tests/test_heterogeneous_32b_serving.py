from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor import cli
from scripts import heterogeneous_32b_serving_check as check
from scripts import heterogeneous_32b_serving_pack as pack


class Heterogeneous32BServingTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_heterogeneous_32b_serving_test_"))

    def test_fixture_serving_report_has_required_product_like_fields(self) -> None:
        output_dir = self._tmp_dir() / "serving"
        report = pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(output_dir),
                    "--serving-mode",
                    "fixture",
                    "--max-new-tokens",
                    "4",
                    "--failure-injection",
                    "tpu-timeout",
                ]
            )
        )

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["heterogeneous_32b_serving_ready"])
        self.assertTrue(report["production_like_serving_path_ready"])
        self.assertTrue(report["gpu_tpu_cpu_32b_same_request_source_verified"])
        self.assertTrue(report["multi_token_generation_ready"])
        self.assertTrue(report["streaming_response_contract_ready"])
        self.assertTrue(report["stage_local_kv_cache_ready"])
        self.assertTrue(report["latency_metrics_ready"])
        self.assertTrue(report["failure_requeue_ready"])
        self.assertFalse(report["live_external_runtime_verified"])
        self.assertFalse(report["live_external_multitoken_attempt"]["fresh_live_run_attempted"])
        self.assertEqual(
            report["live_external_multitoken_attempt"]["blocked_reason"],
            "fresh_kaggle_multitoken_live_run_not_attempted",
        )
        self.assertEqual(report["blocked_reason"], "")
        self.assertEqual(check.validate_report(report), [])

    def test_checker_builds_and_validates_fixture_report(self) -> None:
        output_dir = self._tmp_dir() / "check"
        result = check.build_check(
            check.parse_args(
                [
                    "--output-dir",
                    str(output_dir),
                    "--serving-mode",
                    "fixture",
                    "--max-new-tokens",
                    "4",
                ]
            )
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["heterogeneous_32b_serving_ready"])
        self.assertTrue(result["production_like_serving_path_ready"])

    def test_checker_rejects_less_than_four_token_claim(self) -> None:
        output_dir = self._tmp_dir() / "tokens"
        report = pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(output_dir),
                    "--serving-mode",
                    "fixture",
                    "--max-new-tokens",
                    "4",
                ]
            )
        )
        report["generated_token_count"] = 1
        report["multi_token_generation_ready"] = True

        errors = check.validate_report(report)

        self.assertIn("generated_token_count_below_four", errors)

    def test_checker_rejects_live_external_overclaim_without_external_mode(self) -> None:
        output_dir = self._tmp_dir() / "overclaim"
        report = pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(output_dir),
                    "--serving-mode",
                    "fixture",
                    "--max-new-tokens",
                    "4",
                ]
            )
        )
        report["live_external_runtime_verified"] = True

        errors = check.validate_report(report)

        self.assertIn("live_external_true_without_external_mode", errors)

    def test_external_live_blocker_is_explicit_but_deployment_engineering_stays_ready(self) -> None:
        base = self._tmp_dir()
        live_report = {
            "schema": pack.SCHEMA,
            "ok": True,
            "heterogeneous_32b_serving_ready": True,
            "production_like_serving_path_ready": True,
            "gpu_tpu_cpu_32b_same_request_source_verified": True,
            "multi_token_generation_ready": True,
            "live_external_runtime_verified": False,
            "generated_token_count": 0,
            "fallback_model_used": False,
            "blockers": ["kaggle_web_tpu_runtime_queued"],
            "public_artifact_safe": True,
        }
        live_path = base / "live-blocked.json"
        live_path.write_text(json.dumps(live_report, indent=2, sort_keys=True), encoding="utf-8")

        report = pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "serving"),
                    "--serving-mode",
                    "fixture",
                    "--live-run-mode",
                    "external",
                    "--live-serving-report",
                    str(live_path),
                    "--max-new-tokens",
                    "4",
                ]
            )
        )

        self.assertTrue(report["heterogeneous_32b_serving_ready"])
        self.assertTrue(report["production_like_serving_path_ready"])
        self.assertFalse(report["live_external_runtime_verified"])
        self.assertEqual(report["blocked_reason"], "kaggle_web_tpu_runtime_queued")
        self.assertTrue(report["blocker_report"]["deployment_engineering_complete"])
        self.assertEqual(check.validate_report(report), [])

    def test_external_runtime_bridge_attempt_blocker_is_carried(self) -> None:
        base = self._tmp_dir()
        bridge_attempt = {
            "schema": "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1",
            "ok": False,
            "generated_token_count": 0,
            "target_generated_token_count": 4,
            "accepted_stage_backends": ["cuda"],
            "stage_task_counts": {"stage0": 1, "stage1": 0, "stage2": 0},
            "blocked_reason": "same_request_runtime_bridge_not_verified",
            "blockers": ["same_request_runtime_bridge_not_verified"],
            "stage_reports": {
                "jax_tpu_stage": {
                    "ok": False,
                    "blockers": ["web_tpu_jupyter_proxy_not_found"],
                    "public_artifact_safe": True,
                    "jupyter_proxy_token_public": False,
                }
            },
            "cleanup": {
                "kaggle_gpu_kernel_deleted": True,
                "private_gpu_package_removed": True,
                "web_tpu_runtime_private_token_public": False,
            },
            "public_artifact_safe": True,
        }
        bridge_path = base / "bridge-attempt.json"
        bridge_path.write_text(json.dumps(bridge_attempt, indent=2, sort_keys=True), encoding="utf-8")

        report = pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "serving"),
                    "--serving-mode",
                    "fixture",
                    "--live-run-mode",
                    "external",
                    "--live-serving-report",
                    str(bridge_path),
                    "--max-new-tokens",
                    "4",
                ]
            )
        )

        self.assertTrue(report["heterogeneous_32b_serving_ready"])
        self.assertTrue(report["production_like_serving_path_ready"])
        self.assertFalse(report["live_external_runtime_verified"])
        self.assertEqual(report["blocked_reason"], "web_tpu_jupyter_proxy_not_found")
        self.assertTrue(report["live_external_multitoken_attempt"]["fresh_live_run_attempted"])
        self.assertTrue(report["live_external_multitoken_attempt"]["bridge_attempt_report"])
        self.assertIn("web_tpu_jupyter_proxy_not_found", report["live_external_multitoken_attempt"]["blockers"])
        self.assertEqual(report["live_external_multitoken_attempt"]["accepted_stage_backends"], ["cuda"])
        self.assertEqual(check.validate_report(report), [])

    def test_gpu_quota_blocker_takes_priority_over_downstream_stage_gaps(self) -> None:
        base = self._tmp_dir()
        bridge_attempt = {
            "schema": "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1",
            "ok": False,
            "generated_token_count": 0,
            "target_generated_token_count": 4,
            "accepted_stage_backends": [],
            "stage_task_counts": {"stage0": 0, "stage1": 0, "stage2": 0},
            "blockers": [
                "same_request_runtime_bridge_not_verified",
                "kaggle_gpu_batch_session_limit_reached",
                "cuda_stage_not_ready",
                "jax_tpu_stage_not_ready",
                "cpu_tail_not_ready",
            ],
            "cleanup": {
                "kaggle_gpu_kernel_created": False,
                "kaggle_gpu_kernel_deleted": True,
                "private_gpu_package_removed": True,
                "web_tpu_runtime_private_token_public": False,
            },
            "public_artifact_safe": True,
        }
        bridge_path = base / "bridge-quota-attempt.json"
        bridge_path.write_text(json.dumps(bridge_attempt, indent=2, sort_keys=True), encoding="utf-8")

        report = pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "serving"),
                    "--serving-mode",
                    "fixture",
                    "--live-run-mode",
                    "external",
                    "--live-serving-report",
                    str(bridge_path),
                    "--max-new-tokens",
                    "4",
                ]
            )
        )

        self.assertEqual(report["blocked_reason"], "kaggle_gpu_batch_session_limit_reached")
        self.assertEqual(
            report["live_external_multitoken_attempt"]["blocked_reason"],
            "kaggle_gpu_batch_session_limit_reached",
        )
        self.assertEqual(check.validate_report(report), [])

    def test_successful_runtime_bridge_counts_as_live_external_multitoken(self) -> None:
        base = self._tmp_dir()
        bridge_success = {
            "schema": "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1",
            "ok": True,
            "same_request_runtime_bridge_verified": True,
            "gpu_tpu_cpu_32b_same_request_verified": True,
            "same_request_32b_model_verified": True,
            "generated_token_count": 4,
            "target_generated_token_count": 4,
            "accepted_stage_backends": ["cpu", "cuda", "jax_tpu"],
            "stage_task_counts": {"stage0": 4, "stage1": 4, "stage2": 4},
            "cleanup": {
                "kaggle_gpu_kernel_created": True,
                "kaggle_gpu_kernel_deleted": True,
                "private_gpu_package_removed": True,
                "web_tpu_runtime_private_token_public": False,
            },
            "safety": {"public_artifact_safe": True},
        }
        bridge_path = base / "bridge-success.json"
        bridge_path.write_text(json.dumps(bridge_success, indent=2, sort_keys=True), encoding="utf-8")

        report = pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(base / "serving"),
                    "--serving-mode",
                    "fixture",
                    "--live-run-mode",
                    "external",
                    "--live-serving-report",
                    str(bridge_path),
                    "--max-new-tokens",
                    "4",
                ]
            )
        )

        self.assertTrue(report["live_external_runtime_verified"])
        self.assertEqual(report["blocked_reason"], "")
        self.assertFalse(report["blocker_report"]["blocked"])
        self.assertTrue(report["live_external_multitoken_attempt"]["live_external_runtime_verified"])
        self.assertEqual(report["live_external_multitoken_attempt"]["attempt_status"], "verified_live_external_runtime")
        self.assertEqual(check.validate_report(report), [])

    def test_public_artifacts_are_redacted(self) -> None:
        output_dir = self._tmp_dir() / "redaction"
        report = pack.build_report(
            pack.parse_args(
                [
                    "--output-dir",
                    str(output_dir),
                    "--serving-mode",
                    "fixture",
                    "--max-new-tokens",
                    "4",
                ]
            )
        )

        self.assertEqual(pack.public_redaction_errors(report), [])
        scanned = "\n".join(path.read_text(encoding="utf-8") for path in output_dir.rglob("*") if path.is_file())
        for fragment in [
            "KAGGLE_KEY=",
            "KAGGLE_USERNAME=",
            "HF_TOKEN=",
            "Bearer ",
            "jupyter-proxy",
            "token=",
            "kaggle-cookies",
            "kaggle-web-storage-state",
            '"prompt":',
            '"raw_prompt":',
            '"generated_text":',
            '"generated_token_ids":',
            '"activation":',
            '"hidden_state":',
            '"logits":',
            '"kv_cache":',
            '"past_key_values":',
            '"lease_token":',
            '"idempotency_key":',
        ]:
            self.assertNotIn(fragment, scanned)

    def test_cli_wraps_heterogeneous_32b_serving(self) -> None:
        output_dir = self._tmp_dir() / "cli"
        summary = cli.build_heterogeneous_32b_serving(
            cli.parse_args(
                [
                    "heterogeneous-32b-serving",
                    "--output-dir",
                    str(output_dir),
                    "--serving-mode",
                    "fixture",
                    "--max-new-tokens",
                    "4",
                ]
            )
        )

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "heterogeneous_32b_serving_cli_v1")
        self.assertTrue(summary["heterogeneous_32b_serving_ready"])
        self.assertTrue((output_dir / "live_external_multitoken_attempt.json").is_file())
        self.assertTrue((output_dir / "heterogeneous_32b_serving_cli_summary.json").is_file())

    def test_parse_rejects_live_external_without_report(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args(
                [
                    "--output-dir",
                    str(self._tmp_dir() / "bad"),
                    "--serving-mode",
                    "fixture",
                    "--live-run-mode",
                    "external",
                ]
            )


if __name__ == "__main__":
    unittest.main()
