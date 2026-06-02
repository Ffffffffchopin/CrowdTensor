from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import public_swarm_product_beta_check as check
from scripts import public_swarm_product_beta_pack as pack


class PublicSwarmProductBetaPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_product_beta_test_"))

    def test_check_builds_ready_local_loopback_product_beta(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "local-loopback",
            "--output-dir",
            str(self._tmp_dir()),
            "--max-new-tokens",
            "2",
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "public_swarm_product_beta_check_v1")

    def test_check_builds_ready_kaggle_package_product_beta(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "package",
            "--target",
            "kaggle",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_check_builds_ready_external_existing_product_beta(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "external-existing",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_external_existing_requires_auth(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args([
                "external-existing",
                "--coordinator-url",
                "http://127.0.0.1:9999",
            ])

    def test_rc_args_forward_bounded_prompt_batch(self) -> None:
        args = pack.parse_args([
            "local-loopback",
            "--prompt-texts",
            "first prompt,second prompt",
        ])
        rc_args = pack.rc_args(args, Path("/tmp/product-beta-rc"))

        self.assertEqual(rc_args.prompt_texts, "first prompt,second prompt")

    def test_rc_args_forward_stream_generation(self) -> None:
        args = pack.parse_args([
            "local-loopback",
            "--stream-generation",
        ])
        rc_args = pack.rc_args(args, Path("/tmp/product-beta-rc"))

        self.assertTrue(rc_args.stream_generation)

    def test_local_loopback_ready_when_user_path_and_split_ready_even_if_legacy_rc_blocked(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "16",
        ])
        rc_payload = {
            "schema": pack.RC_SCHEMA,
            "ok": False,
            "mode": "local-loopback",
            "rc": {
                "ready": False,
                "mode_ready": True,
                "product_beta_ready": False,
                "p2p_lite_route_ready": False,
                "cpu_fallback_ready": True,
                "workload_type": pack.WORKLOAD_TYPE,
                "max_new_tokens": 16,
            },
            "diagnosis_codes": [
                "public_swarm_inference_beta_rc_blocked",
                "p2p_lite_route_blocked",
                "p2p_lite_discovery_blocked",
                "serve_join_generate_loop_ready",
                "remote_generate_session_ready",
                "public_swarm_generate_ready",
                "cpu_fallback_ready",
                "local_cpu_inference_ready",
                "read_only_workload",
                "not_production",
            ],
        }
        split_payload = {
            "schema": pack.REMOTE_REAL_SCHEMA,
            "ok": True,
            "mode": "remote-loopback",
            "diagnosis_codes": [
                "remote_real_llm_sharded_ready",
                "remote_real_llm_sharded_loopback_ready",
                "decoded_tokens_match",
                "distinct_stage_miners",
                "stage_assignment_valid",
            ],
            "payload_summaries": {
                "remote_real_llm_sharded_beta": {
                    "session": {
                        "stage_count": 2,
                        "request_count": 1,
                        "model_id": "sshleifer/tiny-gpt2",
                    },
                    "stage_assignment": {
                        "stage0_miner_id": "stage0",
                        "stage1_miner_id": "stage1",
                        "distinct_stage_miners": True,
                        "stage_assignment_valid": True,
                    },
                }
            },
        }
        original_rc = pack.run_rc_core
        original_split = pack.run_split_validation

        def fake_rc(_args: object, *, output_dir: Path, runner: object) -> tuple[dict, dict]:
            del _args, output_dir, runner
            return {"name": "public_swarm_beta_rc_core", "ok": False, "payload_schema": pack.RC_SCHEMA}, rc_payload

        def fake_split(_args: object, *, output_dir: Path, runner: object) -> tuple[dict, dict]:
            del _args, output_dir, runner
            return {"name": "split", "ok": True, "payload_schema": pack.REMOTE_REAL_SCHEMA}, split_payload

        try:
            pack.run_rc_core = fake_rc
            pack.run_split_validation = fake_split
            report = pack.build_report(args, runner=subprocess.run)
        finally:
            pack.run_rc_core = original_rc
            pack.run_split_validation = original_split

        codes = set(report["diagnosis_codes"])
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["product_beta"]["mode_ready"])
        self.assertIn("public_swarm_product_beta_ready", codes)
        self.assertIn("serve_join_generate_loop_ready", codes)
        self.assertIn("public_swarm_generate_ready", codes)
        self.assertIn("decoded_tokens_match", codes)
        self.assertNotIn("p2p_lite_route_ready", codes)
        self.assertNotIn("p2p_lite_discovery_ready", codes)
        self.assertNotIn("p2p_lite_route_blocked", codes)
        self.assertNotIn("p2p_lite_discovery_blocked", codes)
        self.assertNotIn("public_swarm_inference_beta_rc_blocked", codes)
        self.assertNotIn("public_swarm_product_rc_blocked", codes)

    def test_report_preserves_child_batch_readiness(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "package",
            "--output-dir",
            str(output_dir),
            "--prompt-texts",
            "first prompt,second prompt",
        ])
        rc_payload = {
            "schema": pack.RC_SCHEMA,
            "ok": True,
            "mode": "package",
            "rc": {
                "ready": True,
                "batch": {
                    "enabled": True,
                    "request_count": 2,
                    "prompt_hashes": ["sha256:a", "sha256:b"],
                    "prompt_char_counts": [12, 13],
                    "results": [
                        {
                            "request_id": "req-1",
                            "prompt_hash": "sha256:a",
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g1",
                            "multi_token_generation_ready": True,
                        },
                        {
                            "request_id": "req-2",
                            "prompt_hash": "sha256:b",
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g2",
                            "multi_token_generation_ready": True,
                        },
                    ],
                    "batch_generation_ready": True,
                },
            },
            "diagnosis_codes": [
                "public_swarm_inference_beta_rc_ready",
                "public_swarm_beta_rc_package_ready",
                "miner_join_pack_ready",
                "private_artifacts_local_only",
                "public_swarm_generate_batch_ready",
            ],
        }
        original_rc = pack.run_rc_core
        original_split = pack.run_split_validation

        def fake_rc(_args: object, *, output_dir: Path, runner: object) -> tuple[dict, dict]:
            del _args, output_dir, runner
            return {"name": "public_swarm_beta_rc_core", "ok": True, "payload_schema": pack.RC_SCHEMA}, rc_payload

        def fake_split(_args: object, *, output_dir: Path, runner: object) -> tuple[dict, dict]:
            del _args, output_dir, runner
            return {"name": "split", "ok": True}, {}

        try:
            pack.run_rc_core = fake_rc
            pack.run_split_validation = fake_split
            report = pack.build_report(args, runner=subprocess.run)
        finally:
            pack.run_rc_core = original_rc
            pack.run_split_validation = original_split

        self.assertTrue(report["product_beta"]["batch"]["batch_generation_ready"])
        self.assertIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])

    def test_report_rejects_child_batch_ready_code_without_structured_batch_evidence(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "package",
            "--output-dir",
            str(output_dir),
            "--prompt-texts",
            "first prompt,second prompt",
        ])
        rc_payload = {
            "schema": pack.RC_SCHEMA,
            "ok": True,
            "mode": "package",
            "rc": {
                "ready": True,
                "batch": {
                    "enabled": True,
                    "request_count": 2,
                    "batch_generation_ready": False,
                },
            },
            "diagnosis_codes": [
                "public_swarm_inference_beta_rc_ready",
                "public_swarm_beta_rc_package_ready",
                "miner_join_pack_ready",
                "private_artifacts_local_only",
                "public_swarm_generate_batch_ready",
            ],
        }
        original_rc = pack.run_rc_core
        original_split = pack.run_split_validation

        def fake_rc(_args: object, *, output_dir: Path, runner: object) -> tuple[dict, dict]:
            del _args, output_dir, runner
            return {"name": "public_swarm_beta_rc_core", "ok": True, "payload_schema": pack.RC_SCHEMA}, rc_payload

        def fake_split(_args: object, *, output_dir: Path, runner: object) -> tuple[dict, dict]:
            del _args, output_dir, runner
            return {"name": "split", "ok": True}, {}

        try:
            pack.run_rc_core = fake_rc
            pack.run_split_validation = fake_split
            report = pack.build_report(args, runner=subprocess.run)
        finally:
            pack.run_rc_core = original_rc
            pack.run_split_validation = original_split

        self.assertTrue(report["ok"], report)
        self.assertFalse(report["product_beta"]["batch"]["batch_generation_ready"])
        self.assertNotIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])

    def test_report_rejects_child_batch_with_duplicate_request_identity(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "package",
            "--output-dir",
            str(output_dir),
            "--prompt-texts",
            "first prompt,second prompt",
        ])
        rc_payload = {
            "schema": pack.RC_SCHEMA,
            "ok": True,
            "mode": "package",
            "rc": {
                "ready": True,
                "batch": {
                    "enabled": True,
                    "request_count": 2,
                    "expected_request_count": 2,
                    "observed_request_count": 2,
                    "result_count": 2,
                    "results": [
                        {
                            "request_id": "req-1",
                            "prompt_hash": "sha256:a",
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g1",
                            "multi_token_generation_ready": True,
                        },
                        {
                            "request_id": "req-1",
                            "prompt_hash": "sha256:a",
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g1-dup",
                            "multi_token_generation_ready": True,
                        },
                    ],
                    "batch_generation_ready": True,
                },
            },
            "diagnosis_codes": [
                "public_swarm_inference_beta_rc_ready",
                "public_swarm_beta_rc_package_ready",
                "miner_join_pack_ready",
                "private_artifacts_local_only",
                "public_swarm_generate_batch_ready",
            ],
        }
        original_rc = pack.run_rc_core
        original_split = pack.run_split_validation

        def fake_rc(_args: object, *, output_dir: Path, runner: object) -> tuple[dict, dict]:
            del _args, output_dir, runner
            return {"name": "public_swarm_beta_rc_core", "ok": True, "payload_schema": pack.RC_SCHEMA}, rc_payload

        def fake_split(_args: object, *, output_dir: Path, runner: object) -> tuple[dict, dict]:
            del _args, output_dir, runner
            return {"name": "split", "ok": True}, {}

        try:
            pack.run_rc_core = fake_rc
            pack.run_split_validation = fake_split
            report = pack.build_report(args, runner=subprocess.run)
        finally:
            pack.run_rc_core = original_rc
            pack.run_split_validation = original_split

        self.assertTrue(report["ok"], report)
        self.assertFalse(report["product_beta"]["batch"]["batch_identity_ready"])
        self.assertFalse(report["product_beta"]["batch"]["batch_generation_ready"])
        self.assertNotIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])

    def test_report_preserves_child_stream_readiness(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "package",
            "--output-dir",
            str(output_dir),
            "--stream-generation",
        ])
        rc_payload = {
            "schema": pack.RC_SCHEMA,
            "ok": True,
            "mode": "package",
            "rc": {
                "ready": True,
                "stream": {
                    "enabled": True,
                    "requested": True,
                    "endpoint_ready": True,
                    "stream_generation_ready": True,
                    "progress": {
                        "stream_progress_complete": True,
                        "all_token_events_ready": True,
                        "monotonic_progress": True,
                        "expected_request_count": 1,
                        "observed_token_counts": [1],
                        "max_observed_token_count": 1,
                        "max_new_tokens": 1,
                    },
                },
            },
            "diagnosis_codes": [
                "public_swarm_inference_beta_rc_ready",
                "public_swarm_beta_rc_package_ready",
                "miner_join_pack_ready",
                "private_artifacts_local_only",
                "public_swarm_generate_stream_ready",
                "public_swarm_generate_stream_endpoint_ready",
            ],
        }
        original_rc = pack.run_rc_core
        original_split = pack.run_split_validation

        def fake_rc(_args: object, *, output_dir: Path, runner: object) -> tuple[dict, dict]:
            del _args, output_dir, runner
            return {"name": "public_swarm_beta_rc_core", "ok": True, "payload_schema": pack.RC_SCHEMA}, rc_payload

        def fake_split(_args: object, *, output_dir: Path, runner: object) -> tuple[dict, dict]:
            del _args, output_dir, runner
            return {"name": "split", "ok": True}, {}

        try:
            pack.run_rc_core = fake_rc
            pack.run_split_validation = fake_split
            report = pack.build_report(args, runner=subprocess.run)
        finally:
            pack.run_rc_core = original_rc
            pack.run_split_validation = original_split

        self.assertTrue(report["product_beta"]["stream"]["stream_generation_ready"])
        self.assertIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_endpoint_ready", report["diagnosis_codes"])

    def test_report_rejects_batch_stream_without_per_request_progress(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "package",
            "--output-dir",
            str(output_dir),
            "--stream-generation",
        ])
        rc_payload = {
            "schema": pack.RC_SCHEMA,
            "ok": True,
            "mode": "package",
            "rc": {
                "ready": True,
                "batch": {
                    "enabled": True,
                    "expected_request_count": 2,
                    "request_count": 2,
                    "results": [
                        {
                            "request_id": "req-1",
                            "prompt_hash": "sha256:a",
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g1",
                            "multi_token_generation_ready": True,
                        },
                        {
                            "request_id": "req-2",
                            "prompt_hash": "sha256:b",
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g2",
                            "multi_token_generation_ready": True,
                        },
                    ],
                    "batch_generation_ready": True,
                },
                "stream": {
                    "enabled": True,
                    "requested": True,
                    "endpoint_ready": True,
                    "stream_generation_ready": True,
                    "progress": {
                        "stream_progress_complete": True,
                        "all_token_events_ready": True,
                        "monotonic_progress": True,
                        "expected_request_count": 2,
                        "observed_token_counts": [1, 2],
                        "max_observed_token_count": 2,
                        "max_new_tokens": 2,
                    },
                },
            },
            "diagnosis_codes": [
                "public_swarm_inference_beta_rc_ready",
                "public_swarm_beta_rc_package_ready",
                "miner_join_pack_ready",
                "private_artifacts_local_only",
                "public_swarm_generate_batch_ready",
                "public_swarm_generate_stream_ready",
                "public_swarm_generate_stream_endpoint_ready",
            ],
        }
        original_rc = pack.run_rc_core
        original_split = pack.run_split_validation

        def fake_rc(_args: object, *, output_dir: Path, runner: object) -> tuple[dict, dict]:
            del _args, output_dir, runner
            return {"name": "public_swarm_beta_rc_core", "ok": True, "payload_schema": pack.RC_SCHEMA}, rc_payload

        def fake_split(_args: object, *, output_dir: Path, runner: object) -> tuple[dict, dict]:
            del _args, output_dir, runner
            return {"name": "split", "ok": True}, {}

        try:
            pack.run_rc_core = fake_rc
            pack.run_split_validation = fake_split
            report = pack.build_report(args, runner=subprocess.run)
        finally:
            pack.run_rc_core = original_rc
            pack.run_split_validation = original_split

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["product_beta"]["stream"]["stream_generation_ready"])
        self.assertNotIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_generate_stream_endpoint_ready", report["diagnosis_codes"])

    def test_prompt_batch_rejects_more_than_four_prompts(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args([
                "local-loopback",
                "--prompt-texts",
                "one,two,three,four,five",
            ])

    def test_report_redacts_secret_fragments(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.persist_report(
            {
                "schema": pack.SCHEMA,
                "ok": True,
                "mode": "package",
                "diagnosis_codes": ["public_swarm_product_beta_ready"],
                "product_beta": {"ready": True},
                "steps": [],
                "limitations": [],
                "safety": {},
                "secret": "admin-secret",
            },
            output_dir=output_dir,
            secret_values=["admin-secret"],
        )

        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("admin-secret", encoded)


if __name__ == "__main__":
    unittest.main()
