from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "product_swarm_mvp_check.py"
SPEC = importlib.util.spec_from_file_location("product_swarm_mvp_check", SCRIPT_PATH)
product_check = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(product_check)


class ProductSwarmMvpCheckTests(unittest.TestCase):
    def test_missing_hf_dependency_reports_degraded_ready_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = product_check.parse_args(["--output-dir", tmp])
            with patch.object(product_check, "missing_hf_dependencies", return_value=["transformers"]):
                report = product_check.build_report(args)

            self.assertTrue(report["ok"], report)
            self.assertTrue(report["degraded"])
            self.assertIn("product_swarm_mvp_degraded_ready", report["diagnosis_codes"])
            self.assertIn("hf_dependencies_missing", report["diagnosis_codes"])
            self.assertTrue((Path(tmp) / "product_swarm_mvp_check.json").is_file())

    def test_missing_hf_dependency_blocks_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = product_check.parse_args(["--output-dir", tmp, "--require-hf-runtime"])
            with patch.object(product_check, "missing_hf_dependencies", return_value=["transformers"]):
                report = product_check.build_report(args)

            self.assertFalse(report["ok"], report)
            self.assertIn("product_swarm_mvp_hf_runtime_missing", report["diagnosis_codes"])

    def test_finalize_report_accepts_real_generate_and_stage_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = product_check.parse_args(["--output-dir", tmp, "--max-new-tokens", "2"])
            payloads = {
                "generate": {
                    "ok": True,
                    "session": {"session_id": "real-session"},
                    "generation": {
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:generated",
                        "decoded_tokens_match": True,
                        "raw_generated_text_public": False,
                        "generated_token_ids_public": False,
                    },
                }
            }
            rows = [
                {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 0}},
                {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": 0}},
                {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 1}},
                {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": 1}},
            ]

            with patch.object(product_check, "request_json", return_value={"results": rows}):
                report = product_check.finalize_report(
                    args,
                    Path(tmp),
                    "http://127.0.0.1:9877",
                    [
                        {"name": "serve", "ok": True},
                        {"name": "join_stage0_step_0", "ok": True, "duration_seconds": 0.2},
                        {"name": "join_stage1_step_0", "ok": True, "duration_seconds": 0.3},
                        {"name": "join_stage0_step_1", "ok": True, "duration_seconds": 0.2},
                        {"name": "join_stage1_step_1", "ok": True, "duration_seconds": 0.3},
                        {"name": "generate", "ok": True},
                    ],
                    payloads,
                    {},
                )

            encoded = json.dumps(report, sort_keys=True)
            self.assertTrue(report["ok"], report)
            self.assertIn("product_swarm_mvp_ready", report["diagnosis_codes"])
            self.assertIn("stage_latency_ready", report["diagnosis_codes"])
            self.assertIn("throughput_summary_ready", report["diagnosis_codes"])
            self.assertIn("memory_or_vram_summary_ready", report["diagnosis_codes"])
            self.assertEqual(report["generation"]["generated_token_count"], 2)
            self.assertTrue(report["stage_assignment"]["distinct_stage_miners"])
            self.assertTrue(report["performance"]["stage_latency_ready"])
            self.assertTrue(report["performance"]["throughput_summary_ready"])
            self.assertTrue(report["runtime_resources"]["memory_or_vram_summary_ready"])
            self.assertNotIn('"generated_token_ids":', encoded)
            self.assertNotIn('"generated_text":', encoded)

    def test_finalize_report_accepts_batched_generate_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = product_check.parse_args([
                "--output-dir",
                tmp,
                "--prompt-texts",
                "first private prompt,second private prompt",
                "--max-new-tokens",
                "2",
            ])
            payloads = {
                "generate": {
                    "ok": True,
                    "session": {"session_id": "real-session"},
                    "generation": {
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:batch",
                        "decoded_tokens_match": True,
                        "request_count": 2,
                        "batch_generation_ready": True,
                        "raw_generated_text_public": False,
                        "generated_token_ids_public": False,
                        "results": [
                            {
                                "request_id": "req-1",
                                "prompt_hash": "sha256:p1",
                                "generated_token_count": 2,
                                "max_new_tokens": 2,
                                "generated_text_hash": "sha256:g1",
                                "decoded_tokens_match": True,
                                "multi_token_generation_ready": True,
                                "raw_generated_text_public": False,
                                "generated_token_ids_public": False,
                            },
                            {
                                "request_id": "req-2",
                                "prompt_hash": "sha256:p2",
                                "generated_token_count": 2,
                                "max_new_tokens": 2,
                                "generated_text_hash": "sha256:g2",
                                "decoded_tokens_match": True,
                                "multi_token_generation_ready": True,
                                "raw_generated_text_public": False,
                                "generated_token_ids_public": False,
                            },
                        ],
                    },
                }
            }
            rows = []
            for generation_step in range(2):
                rows.extend([
                    {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": generation_step, "request_count": 2}},
                    {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": generation_step, "request_count": 2}},
                ])

            with patch.object(product_check, "request_json", return_value={"results": rows}):
                report = product_check.finalize_report(
                    args,
                    Path(tmp),
                    "http://127.0.0.1:9877",
                    [
                        {"name": "serve", "ok": True},
                        {"name": "join_stage0_step_0", "ok": True, "duration_seconds": 0.2},
                        {"name": "join_stage1_step_0", "ok": True, "duration_seconds": 0.3},
                        {"name": "join_stage0_step_1", "ok": True, "duration_seconds": 0.2},
                        {"name": "join_stage1_step_1", "ok": True, "duration_seconds": 0.3},
                        {"name": "generate", "ok": True},
                    ],
                    payloads,
                    {},
                )

            encoded = json.dumps(report, sort_keys=True)
            self.assertTrue(report["ok"], report)
            self.assertIn("product_swarm_mvp_batch_ready", report["diagnosis_codes"])
            self.assertIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])
            self.assertTrue(report["batch"]["enabled"])
            self.assertTrue(report["batch"]["batch_generation_ready"])
            self.assertTrue(report["batch"]["batch_identity_ready"])
            self.assertEqual(report["batch"]["expected_request_count"], 2)
            self.assertEqual(report["batch"]["result_count"], 2)
            self.assertEqual(report["ledger"]["accepted_rows"], 4)
            self.assertNotIn("first private prompt", encoded)
            self.assertNotIn("second private prompt", encoded)
            self.assertNotIn('"generated_token_ids":', encoded)
            self.assertNotIn('"generated_text":', encoded)

    def test_finalize_report_blocks_batch_with_duplicate_request_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = product_check.parse_args([
                "--output-dir",
                tmp,
                "--max-new-tokens",
                "2",
                "--prompt-texts",
                "first private prompt,second private prompt",
            ])
            payloads = {
                "generate": {
                    "schema": "public_swarm_product_cli_v1",
                    "ok": True,
                    "generation": {
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "multi_token_generation_ready": True,
                        "batch_generation_ready": True,
                        "request_count": 2,
                        "results": [
                            {
                                "request_id": "req-1",
                                "prompt_hash": "sha256:p1",
                                "generated_token_count": 2,
                                "max_new_tokens": 2,
                                "generated_text_hash": "sha256:g1",
                                "decoded_tokens_match": True,
                                "multi_token_generation_ready": True,
                            },
                            {
                                "request_id": "req-1",
                                "prompt_hash": "sha256:p1",
                                "generated_token_count": 2,
                                "max_new_tokens": 2,
                                "generated_text_hash": "sha256:g1-dup",
                                "decoded_tokens_match": True,
                                "multi_token_generation_ready": True,
                            },
                        ],
                    },
                }
            }

            with patch.object(product_check, "request_json", return_value={"results": []}):
                report = product_check.finalize_report(
                    args,
                    Path(tmp),
                    "http://127.0.0.1:9877",
                    [{"name": "serve", "ok": True}, {"name": "generate", "ok": True}],
                    payloads,
                    {},
                )

            self.assertFalse(report["ok"], report)
            self.assertFalse(report["batch"]["batch_identity_ready"])
            self.assertFalse(report["batch"]["batch_generation_ready"])
            self.assertNotIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])

    def test_finalize_report_requires_requested_stream_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = product_check.parse_args([
                "--output-dir",
                tmp,
                "--max-new-tokens",
                "2",
                "--stream-generation",
            ])
            payloads = {
                "generate": {
                    "ok": True,
                    "session": {"session_id": "real-session"},
                    "diagnosis_codes": [
                        "public_swarm_generate_ready",
                        "public_swarm_generate_stream_ready",
                        "public_swarm_generate_stream_endpoint_ready",
                    ],
                    "generation": {
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:generated",
                        "decoded_tokens_match": True,
                        "raw_generated_text_public": False,
                        "generated_token_ids_public": False,
                    },
                    "stream": {
                        "enabled": True,
                        "event_count": 2,
                        "source": "admin-session-stream",
                        "endpoint_ready": True,
                        "progress": {
                            "stream_progress_complete": True,
                            "all_token_events_ready": True,
                            "monotonic_progress": True,
                            "observed_token_counts": [1, 2],
                            "max_observed_token_count": 2,
                            "max_new_tokens": 2,
                            "source": "admin-session-stream",
                        },
                        "events": [
                            {
                                "schema": "session_stream_event_v1",
                                "session_id": "real-session",
                                "task_id": "task-1",
                                "miner_id": "stage1",
                                "stage_id": 1,
                                "generated_token_count": 1,
                                "max_new_tokens": 2,
                                "generation_step": 0,
                                "generated_text_hash": "sha256:step1",
                                "raw_generated_text_public": False,
                                "generated_token_ids_public": False,
                            },
                            {
                                "schema": "session_stream_event_v1",
                                "session_id": "real-session",
                                "task_id": "task-2",
                                "miner_id": "stage1",
                                "stage_id": 1,
                                "generated_token_count": 2,
                                "max_new_tokens": 2,
                                "generation_step": 1,
                                "generated_text_hash": "sha256:step2",
                                "raw_generated_text_public": False,
                                "generated_token_ids_public": False,
                            },
                        ],
                        "raw_generated_text_public": False,
                        "generated_token_ids_public": False,
                    },
                }
            }
            rows = [
                {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 0}},
                {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": 0}},
                {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 1}},
                {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": 1}},
            ]

            with patch.object(product_check, "request_json", return_value={"results": rows}):
                report = product_check.finalize_report(
                    args,
                    Path(tmp),
                    "http://127.0.0.1:9877",
                    [
                        {"name": "serve", "ok": True},
                        {"name": "join_stage0_step_0", "ok": True, "duration_seconds": 0.2},
                        {"name": "join_stage1_step_0", "ok": True, "duration_seconds": 0.3},
                        {"name": "join_stage0_step_1", "ok": True, "duration_seconds": 0.2},
                        {"name": "join_stage1_step_1", "ok": True, "duration_seconds": 0.3},
                        {"name": "generate", "ok": True},
                    ],
                    payloads,
                    {},
                )

            encoded = json.dumps(report, sort_keys=True)
            self.assertTrue(report["ok"], report)
            self.assertTrue(report["stream"]["stream_generation_ready"])
            self.assertTrue(report["stream"]["endpoint_ready"])
            self.assertEqual(report["stream"]["event_count"], 2)
            self.assertIn("product_swarm_mvp_stream_ready", report["diagnosis_codes"])
            self.assertIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
            self.assertIn("public_swarm_generate_stream_endpoint_ready", report["diagnosis_codes"])
            self.assertNotIn('"generated_token_ids":', encoded)
            self.assertNotIn('"generated_text":', encoded)

    def test_finalize_report_blocks_incomplete_requested_stream_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = product_check.parse_args([
                "--output-dir",
                tmp,
                "--max-new-tokens",
                "2",
                "--stream-generation",
            ])
            payloads = {
                "generate": {
                    "ok": True,
                    "session": {"session_id": "real-session"},
                    "generation": {
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:generated",
                        "decoded_tokens_match": True,
                        "raw_generated_text_public": False,
                        "generated_token_ids_public": False,
                    },
                    "stream": {
                        "enabled": True,
                        "event_count": 1,
                        "source": "admin-session-stream",
                        "endpoint_ready": True,
                        "progress": {
                            "stream_progress_complete": False,
                            "observed_token_counts": [1],
                            "max_observed_token_count": 1,
                            "max_new_tokens": 2,
                        },
                        "events": [],
                    },
                }
            }
            rows = [
                {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 0}},
                {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": 0}},
                {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 1}},
                {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": 1}},
            ]

            with patch.object(product_check, "request_json", return_value={"results": rows}):
                report = product_check.finalize_report(
                    args,
                    Path(tmp),
                    "http://127.0.0.1:9877",
                    [{"name": "serve", "ok": True}, {"name": "generate", "ok": True}],
                    payloads,
                    {},
                )

            self.assertFalse(report["ok"], report)
            self.assertIn("stream_generation_not_ready", report["diagnosis_codes"])
            self.assertFalse(report["stream"]["stream_generation_ready"])

    def test_finalize_report_blocks_batch_stream_without_per_request_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = product_check.parse_args([
                "--output-dir",
                tmp,
                "--prompt-texts",
                "first private prompt,second private prompt",
                "--max-new-tokens",
                "2",
                "--stream-generation",
            ])
            payloads = {
                "generate": {
                    "ok": True,
                    "session": {"session_id": "real-session"},
                    "diagnosis_codes": [
                        "public_swarm_generate_ready",
                        "public_swarm_generate_batch_ready",
                        "public_swarm_generate_stream_ready",
                    ],
                    "generation": {
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:batch",
                        "decoded_tokens_match": True,
                        "request_count": 2,
                        "batch_generation_ready": True,
                        "raw_generated_text_public": False,
                        "generated_token_ids_public": False,
                    },
                    "stream": {
                        "enabled": True,
                        "event_count": 4,
                        "source": "admin-session-stream",
                        "endpoint_ready": True,
                        "progress": {
                            "stream_progress_complete": True,
                            "all_token_events_ready": True,
                            "monotonic_progress": True,
                            "expected_request_count": 2,
                            "observed_token_counts": [1, 2],
                            "max_observed_token_count": 2,
                            "max_new_tokens": 2,
                        },
                        "events": [],
                    },
                }
            }
            rows = []
            for generation_step in range(2):
                rows.extend([
                    {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": generation_step, "request_count": 2}},
                    {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": generation_step, "request_count": 2}},
                ])

            with patch.object(product_check, "request_json", return_value={"results": rows}):
                report = product_check.finalize_report(
                    args,
                    Path(tmp),
                    "http://127.0.0.1:9877",
                    [{"name": "serve", "ok": True}, {"name": "generate", "ok": True}],
                    payloads,
                    {},
                )

            encoded = json.dumps(report, sort_keys=True)
            self.assertFalse(report["ok"], report)
            self.assertFalse(report["stream"]["stream_generation_ready"])
            self.assertIn("stream_generation_not_ready", report["diagnosis_codes"])
            self.assertNotIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
            self.assertNotIn("first private prompt", encoded)
            self.assertNotIn("second private prompt", encoded)

    def test_parse_args_rejects_unbounded_prompt_batches(self) -> None:
        with self.assertRaises(SystemExit):
            product_check.parse_args([
                "--prompt-texts",
                "one,two,three,four,five",
            ])

    def test_parse_args_accepts_prompt_texts_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = Path(tmp) / "prompts.txt"
            prompts = ["one prompt, with comma", "second prompt"]
            prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")

            args = product_check.parse_args([
                "--prompt-texts-file",
                str(prompt_file),
            ])

        self.assertEqual(args.prompt_texts_file, str(prompt_file))
        self.assertEqual(args.prompt_texts_list, prompts)

    def test_parse_args_rejects_prompt_texts_file_over_batch_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = Path(tmp) / "too-many-prompts.txt"
            prompt_file.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                product_check.parse_args(["--prompt-texts-file", str(prompt_file)])

        self.assertEqual(str(raised.exception), "prompt_texts_file must contain at most 4 prompts")

    def test_parse_args_rejects_prompt_texts_file_long_line_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = Path(tmp) / "long-prompts.txt"
            private_prompt = "x" * 257
            prompt_file.write_text(f"one\n{private_prompt}\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                product_check.parse_args(["--prompt-texts-file", str(prompt_file)])

        error = str(raised.exception)
        self.assertEqual(error, "prompt_texts_file line 2 must be at most 256 characters")
        self.assertNotIn(private_prompt, error)

    def test_parse_args_rejects_prompt_texts_and_file_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = Path(tmp) / "prompts.txt"
            prompt_file.write_text("one prompt\n", encoding="utf-8")
            with self.assertRaises(SystemExit) as raised:
                product_check.parse_args([
                    "--prompt-texts",
                    "one,two",
                    "--prompt-texts-file",
                    str(prompt_file),
                ])

        self.assertEqual(str(raised.exception), "product_swarm_mvp accepts either --prompt-texts or --prompt-texts-file, not both")


if __name__ == "__main__":
    unittest.main()
