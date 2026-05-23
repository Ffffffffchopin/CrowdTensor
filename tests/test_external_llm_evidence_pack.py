from __future__ import annotations

import argparse
import json
from pathlib import Path
import importlib.util
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "external_llm_evidence_pack.py"
SPEC = importlib.util.spec_from_file_location("external_llm_evidence_pack", SCRIPT_PATH)
external_llm_evidence_pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(external_llm_evidence_pack)


class ExternalLLMEvidencePackTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            miner_id="external-llm-evidence-miner",
            request_count=3,
            adapter_kind="http_openai_chat",
            llm_runtime_model_id="local-model",
            llm_runtime_url="http://127.0.0.1:11434/v1/chat/completions",
            llm_runtime_api_key="secret-api-key",
        )

    def _state(self) -> dict:
        return {
            "accepted_results": 1,
            "model_updates": 0,
            "model": {"global_step": 0},
            "tasks": [
                {
                    "task_id": "task-external",
                    "status": "completed",
                    "workload_type": "external_llm_infer",
                    "validation": {
                        "code": "ok",
                        "request_count": 3,
                        "completion_count": 3,
                        "output_chars": 128,
                        "adapter_kind": "http_openai_chat",
                        "model_id": "local-model",
                        "output_preview": "<redacted>",
                    },
                    "metrics": {
                        "request_count": 3,
                        "completion_count": 3,
                        "output_chars": 128,
                        "elapsed_ms": 12.5,
                        "requests_per_second": 240.0,
                    },
                }
            ],
            "miner_profiles": {
                "external-llm-evidence-miner": {
                    "last_capabilities": {
                        "supported_workloads": ["external_llm_infer"],
                        "external_llm_runtime": {
                            "adapter_kind": "http_openai_chat",
                            "model_id": "local-model",
                        },
                    },
                },
            },
        }

    def _ledger(self) -> dict:
        return {
            "results": [
                {
                    "task_id": "task-external",
                    "status": "completed",
                    "workload_type": "external_llm_infer",
                    "model_updated": False,
                    "model_bundle_updated": False,
                    "validation": {
                        "code": "ok",
                        "request_count": 3,
                        "completion_count": 3,
                        "output_chars": 128,
                        "adapter_kind": "http_openai_chat",
                        "model_id": "local-model",
                        "output_preview": "<redacted>",
                    },
                    "session_metrics": {
                        "request_count": 3,
                        "completion_count": 3,
                        "output_chars": 128,
                        "elapsed_ms": 12.5,
                        "requests_per_second": 240.0,
                        "adapter_kind": "http_openai_chat",
                        "model_id": "local-model",
                    },
                }
            ],
        }

    def test_build_evidence_summarizes_external_llm_without_raw_payloads(self) -> None:
        report = external_llm_evidence_pack.build_evidence(
            args=self._args(),
            state=self._state(),
            ledger=self._ledger(),
            miner_summary={"accepted_tasks": 1, "rejected_tasks": 0, "request_retries": 0},
            generated_at="2026-05-23T00:00:00+00:00",
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "external_llm_evidence_v1")
        self.assertEqual(report["adapter"]["kind"], "http_openai_chat")
        self.assertEqual(report["summary"]["completion_count"], 3)
        self.assertTrue(report["safety"]["read_only"])
        self.assertTrue(report["safety"]["redaction_ok"])
        self.assertIn("external_llm_evidence_ready", report["diagnosis_codes"])
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("secret-api-key", encoded)
        self.assertNotIn("http://127.0.0.1:11434", encoded)
        self.assertNotIn("output_text", encoded)
        self.assertNotIn("external_llm_results", encoded)

    def test_build_evidence_rejects_missing_completion(self) -> None:
        ledger = self._ledger()
        ledger["results"][0]["validation"]["completion_count"] = 1
        report = external_llm_evidence_pack.build_evidence(
            args=self._args(),
            state=self._state(),
            ledger=ledger,
            miner_summary={"accepted_tasks": 1, "rejected_tasks": 0, "request_retries": 0},
            generated_at="2026-05-23T00:00:00+00:00",
        )

        self.assertFalse(report["ok"])
        self.assertIn("external_llm_request_count_mismatch", report["diagnosis_codes"])


if __name__ == "__main__":
    unittest.main()
