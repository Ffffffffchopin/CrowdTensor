from __future__ import annotations

import unittest
from unittest.mock import patch

from crowdtensor.external_llm import (
    SCHEMA_VERSION,
    external_llm_inference_spec_for,
    parse_openai_chat_completion,
    run_external_llm_inference,
    run_mock_external_llm_inference,
    validate_external_llm_inference,
)


class ExternalLLMTests(unittest.TestCase):
    def test_mock_runtime_matches_claim_contract(self) -> None:
        spec = external_llm_inference_spec_for("task-a", "miner-a", request_count=3)

        result = run_mock_external_llm_inference(spec)
        validation = validate_external_llm_inference(
            result["external_llm_result"],
            external_llm_results=result["external_llm_results"],
            expected_requests=spec["requests"],
        )

        self.assertEqual(spec["schema_version"], SCHEMA_VERSION)
        self.assertTrue(validation["accepted"])
        self.assertEqual(validation["request_count"], 3)
        self.assertEqual(validation["completion_count"], 3)
        self.assertEqual(validation["adapter_kind"], "mock")
        self.assertGreater(validation["output_chars"], 0)
        self.assertIn("mock completion", validation["output_preview"])

    def test_validation_rejects_wrong_prompt_hash(self) -> None:
        spec = external_llm_inference_spec_for("task-b", "miner-b", request_count=1)
        result = run_mock_external_llm_inference(spec)
        bad = dict(result["external_llm_result"])
        bad["prompt_hash"] = "sha256:wrong"

        validation = validate_external_llm_inference(
            bad,
            external_llm_results=[bad],
            expected_requests=spec["requests"],
        )

        self.assertFalse(validation["accepted"])
        self.assertEqual(validation["code"], "external_llm_prompt_hash_mismatch")

    def test_validation_rejects_empty_output(self) -> None:
        spec = external_llm_inference_spec_for("task-c", "miner-c", request_count=1)
        result = run_mock_external_llm_inference(spec)
        bad = dict(result["external_llm_result"])
        bad["output_text"] = " "

        validation = validate_external_llm_inference(
            bad,
            external_llm_results=[bad],
            expected_requests=spec["requests"],
        )

        self.assertFalse(validation["accepted"])
        self.assertEqual(validation["code"], "external_llm_empty_output")

    def test_parse_openai_chat_completion_accepts_message_content(self) -> None:
        text = parse_openai_chat_completion({
            "choices": [{"message": {"role": "assistant", "content": "hello from http"}}],
        })

        self.assertEqual(text, "hello from http")

    def test_parse_openai_chat_completion_accepts_legacy_text(self) -> None:
        text = parse_openai_chat_completion({"choices": [{"text": "legacy completion"}]})

        self.assertEqual(text, "legacy completion")

    def test_parse_openai_chat_completion_rejects_malformed_payload(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing choices"):
            parse_openai_chat_completion({"not_choices": []})
        with self.assertRaisesRegex(RuntimeError, "empty completion"):
            parse_openai_chat_completion({"choices": [{"message": {"content": ""}}]})

    def test_http_runtime_uses_openai_chat_adapter(self) -> None:
        spec = external_llm_inference_spec_for("task-http", "miner-http", request_count=1)
        with patch("crowdtensor.external_llm.run_external_llm_http", return_value="http completion") as mock_http:
            result = run_external_llm_inference(
                spec,
                adapter_kind="http_openai_chat",
                model_id="local-model",
                runtime_url="http://127.0.0.1:11434/v1/chat/completions",
                api_key="secret",
                timeout=3.0,
            )

        mock_http.assert_called_once()
        self.assertEqual(result["adapter_kind"], "http_openai_chat")
        self.assertEqual(result["model_id"], "local-model")
        self.assertEqual(result["external_llm_result"]["output_text"], "http completion")


if __name__ == "__main__":
    unittest.main()
