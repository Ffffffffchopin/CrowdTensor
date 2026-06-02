from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import persistent_real_llm_kv_cache_check as kv_check


class PersistentRealLlmKvCacheCheckTests(unittest.TestCase):
    def test_missing_hf_dependency_degrades_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = kv_check.parse_args(["--output-dir", tmp])

            report = kv_check.degraded_report(args, Path(tmp), ["transformers"])

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["degraded"])
        self.assertIn("persistent_kv_cache_degraded_ready", report["diagnosis_codes"])
        self.assertIn("hf_dependencies_missing", report["diagnosis_codes"])

    def test_missing_hf_dependency_blocks_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = kv_check.parse_args(["--output-dir", tmp, "--require-hf-runtime"])

            report = kv_check.degraded_report(args, Path(tmp), ["transformers"])

        self.assertFalse(report["ok"], report)
        self.assertIn("persistent_kv_cache_hf_runtime_missing", report["diagnosis_codes"])

    def test_stage0_kv_rows_extracts_safe_cache_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            events = [
                {
                    "type": "task_completed",
                    "miner_id": "stage0",
                    "validation": {"stage_id": 0, "generation_step": 0},
                    "activation_results": [
                        {
                            "activation_hash": "sha256:first",
                            "kv_cache_schema": "real_llm_stage0_kv_cache_v1",
                            "kv_cache_ready": True,
                            "kv_cache_hit": False,
                            "kv_cache_tokens_before": 0,
                            "kv_cache_tokens_after": 7,
                            "generated_prefix_token_count": 0,
                            "hidden_state": [[0.1]],
                            "input_ids": [1, 2],
                        }
                    ],
                },
                {
                    "type": "task_completed",
                    "miner_id": "stage0",
                    "validation": {"stage_id": 0, "generation_step": 1},
                    "activation_results": [
                        {
                            "activation_hash": "sha256:second",
                            "kv_cache_schema": "real_llm_stage0_kv_cache_v1",
                            "kv_cache_ready": True,
                            "kv_cache_hit": True,
                            "kv_cache_tokens_before": 7,
                            "kv_cache_tokens_after": 8,
                            "generated_prefix_token_count": 1,
                        }
                    ],
                },
                {
                    "type": "task_completed",
                    "miner_id": "stage1",
                    "validation": {"stage_id": 1, "generation_step": 1},
                    "activation_results": [],
                },
            ]
            (state / "tasks.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            rows = kv_check.stage0_kv_rows(state)

        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]["kv_cache_hit"])
        self.assertTrue(rows[1]["kv_cache_hit"])
        self.assertEqual(rows[1]["kv_cache_schema"], "real_llm_stage0_kv_cache_v1")
        self.assertEqual(rows[1]["stage_id"], 0)
        self.assertEqual(rows[1]["kv_cache_tokens_before"], 7)
        self.assertEqual(rows[1]["kv_cache_tokens_after"], 8)
        self.assertNotIn("hidden_state", json.dumps(rows, sort_keys=True))
        self.assertNotIn("input_ids", json.dumps(rows, sort_keys=True))

    def test_stage1_kv_rows_extracts_safe_cache_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            events = [
                {
                    "type": "task_completed",
                    "miner_id": "stage1",
                    "validation": {"stage_id": 1, "generation_step": 0},
                    "inference_results": [
                        {
                            "activation_hash": "sha256:first",
                            "output_hash": "sha256:out-first",
                            "kv_cache_schema": "real_llm_stage1_kv_cache_v1",
                            "kv_cache_stage": "stage1_suffix",
                            "kv_cache_ready": True,
                            "kv_cache_hit": False,
                            "kv_cache_tokens_before": 0,
                            "kv_cache_tokens_after": 7,
                            "generated_token_count": 1,
                            "generated_token_ids": [1],
                            "generated_text": "raw",
                        }
                    ],
                },
                {
                    "type": "task_completed",
                    "miner_id": "stage1",
                    "validation": {"stage_id": 1, "generation_step": 1},
                    "inference_results": [],
                    "sharded_inference_result": {
                        "inference_result": {
                            "activation_hash": "sha256:second",
                            "output_hash": "sha256:out-second",
                            "kv_cache_schema": "real_llm_stage1_kv_cache_v1",
                            "kv_cache_stage": "stage1_suffix",
                            "kv_cache_ready": True,
                            "kv_cache_hit": True,
                            "kv_cache_tokens_before": 7,
                            "kv_cache_tokens_after": 8,
                            "generated_token_count": 2,
                            "next_token_text": "raw",
                        }
                    },
                },
                {
                    "type": "task_completed",
                    "miner_id": "stage0",
                    "validation": {"stage_id": 0, "generation_step": 1},
                    "activation_results": [],
                },
            ]
            (state / "tasks.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            rows = kv_check.stage1_kv_rows(state)
            summary = kv_check.kv_cache_summary(
                schema="real_llm_stage1_kv_cache_v1",
                stage="stage1_suffix",
                expected_hits=1,
                rows=rows,
            )

        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]["kv_cache_hit"])
        self.assertTrue(rows[1]["kv_cache_hit"])
        self.assertEqual(rows[1]["kv_cache_schema"], "real_llm_stage1_kv_cache_v1")
        self.assertEqual(rows[1]["stage_id"], 1)
        self.assertEqual(summary["ready_count"], 2)
        self.assertEqual(summary["hit_count"], 1)
        for row in summary["rows"]:
            self.assertNotIn("activation_hash", row)
            self.assertNotIn("output_hash", row)
        encoded = json.dumps(summary, sort_keys=True)
        self.assertNotIn("generated_token_ids", encoded)
        self.assertNotIn("generated_text", encoded)
        self.assertNotIn("next_token_text", encoded)
        self.assertNotIn("sha256:second", encoded)
        self.assertNotIn("sha256:out-second", encoded)

    def test_public_report_safety_allows_negative_boolean_field_names(self) -> None:
        report = {
            "schema": kv_check.SCHEMA,
            "ok": True,
            "kv_cache": {
                "stage0": {
                    "raw_hidden_state_public": False,
                    "raw_input_ids_public": False,
                    "rows": [
                        {
                            "kv_cache_ready": True,
                            "kv_cache_hit": True,
                        }
                    ],
                },
                "stage1": {
                    "output_hashes_public": False,
                    "rows": [
                        {
                            "kv_cache_ready": True,
                            "kv_cache_hit": True,
                        }
                    ],
                },
            },
            "safety": {
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            },
        }

        self.assertEqual(kv_check.validate_public_report(report), [])


if __name__ == "__main__":
    unittest.main()
