from __future__ import annotations

import unittest

from crowdtensor import session_protocol as protocol


class SessionProtocolTests(unittest.TestCase):
    def test_build_session_request_redacts_prompt_and_sets_cuda_caps(self) -> None:
        request = protocol.build_session_request(
            prompt_text="CrowdTensor routes home GPUs",
            backend="cuda",
            stage_mode="split",
            max_new_tokens=8,
            route_source="peer-bootstrap",
        )

        self.assertEqual(request["schema"], "session_protocol_v1")
        self.assertEqual(request["backend"], "cuda")
        self.assertEqual(request["prompt_chars"], len("CrowdTensor routes home GPUs"))
        self.assertIn("prompt_hash", request)
        encoded = str(request)
        self.assertNotIn("CrowdTensor routes home GPUs", encoded)
        self.assertEqual(
            request["route_requirements"]["required_capabilities"],
            ["real_llm_sharded_cuda_stage0", "real_llm_sharded_cuda_stage1"],
        )

    def test_build_session_request_supports_bounded_prompt_batches(self) -> None:
        request = protocol.build_session_request(
            prompt_text="first prompt",
            prompt_texts=["first prompt", "second prompt"],
            backend="cpu",
            hf_model_id="distilgpt2",
            max_new_tokens=4,
        )
        payload = protocol.coordinator_payload_for_request(
            request,
            prompt_text="first prompt",
            prompt_texts=["first prompt", "second prompt"],
        )
        encoded = str(request)

        self.assertEqual(request["request_count"], 2)
        self.assertEqual(request["hf_model_id"], "distilgpt2")
        self.assertTrue(request["batch"]["enabled"])
        self.assertEqual(len(request["prompt_hashes"]), 2)
        self.assertEqual(request["prompt_char_counts"], [12, 13])
        self.assertNotIn("first prompt", encoded)
        self.assertNotIn("second prompt", encoded)
        self.assertEqual(payload["request_count"], 2)
        self.assertEqual(payload["hf_model_id"], "distilgpt2")
        self.assertEqual(payload["prompt"], "first prompt")
        self.assertEqual(payload["prompt_texts"], ["first prompt", "second prompt"])

    def test_build_session_request_rejects_unbounded_prompt_batches(self) -> None:
        with self.assertRaises(ValueError):
            protocol.build_session_request(
                prompt_text="first",
                prompt_texts=["a", "b", "c", "d", "e"],
            )

    def test_route_decision_matches_peer_catalog(self) -> None:
        request = protocol.build_session_request(
            prompt_text="hello",
            backend="cpu",
            stage_mode="split",
        )
        route = protocol.build_route_decision(
            request,
            peer_catalog=[
                {
                    "peer_id": "coordinator",
                    "role": "coordinator",
                    "urls": {"coordinator": "http://coordinator"},
                    "capabilities": {},
                },
                {
                    "peer_id": "stage0",
                    "role": "miner",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                },
                {
                    "peer_id": "stage1",
                    "role": "miner",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]},
                },
            ],
        )

        self.assertTrue(route["usable_now"], route)
        self.assertEqual(route["coordinator_url"], "http://coordinator")
        self.assertEqual(route["matched_capabilities"]["real_llm_sharded_stage0"], "stage0")
        self.assertEqual(route["matched_capabilities"]["real_llm_sharded_stage1"], "stage1")

    def test_route_decision_filters_stage_peers_by_model_id(self) -> None:
        request = protocol.build_session_request(
            prompt_text="hello",
            backend="cpu",
            hf_model_id="distilgpt2",
            stage_mode="split",
        )
        route = protocol.build_route_decision(
            request,
            peer_catalog=[
                {
                    "peer_id": "coordinator",
                    "role": "coordinator",
                    "urls": {"coordinator": "http://coordinator"},
                    "capabilities": {},
                },
                {
                    "peer_id": "stage0-tiny",
                    "role": "miner",
                    "capabilities": {
                        "hf_model_id": "sshleifer/tiny-gpt2",
                        "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"],
                    },
                },
                {
                    "peer_id": "stage0-distil",
                    "role": "miner",
                    "capabilities": {
                        "hf_model_id": "distilgpt2",
                        "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"],
                    },
                },
                {
                    "peer_id": "stage1-tiny",
                    "role": "miner",
                    "capabilities": {
                        "hf_model_id": "sshleifer/tiny-gpt2",
                        "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"],
                    },
                },
            ],
        )

        self.assertFalse(route["usable_now"], route)
        self.assertEqual(route["matched_capabilities"]["real_llm_sharded_stage0"], "stage0-distil")
        self.assertIn("real_llm_sharded_stage1", route["missing_capabilities"])
        self.assertEqual(route["model_filter"]["expected_hf_model_id"], "distilgpt2")
        self.assertIn("stage1-tiny", route["model_filter"]["mismatched_capabilities"]["real_llm_sharded_stage1"])
        self.assertIn("session_route_model_filter_ready", route["diagnosis_codes"])
        self.assertIn("session_route_model_mismatch", route["diagnosis_codes"])

    def test_route_decision_filters_coordinator_by_model_and_backend(self) -> None:
        request = protocol.build_session_request(
            prompt_text="hello",
            backend="cuda",
            hf_model_id="distilgpt2",
            stage_mode="split",
        )
        route = protocol.build_route_decision(
            request,
            peer_catalog=[
                {
                    "peer_id": "coord-tiny-cuda",
                    "role": "coordinator",
                    "urls": {"coordinator": "http://tiny-cuda"},
                    "capabilities": {"backend": "cuda", "hf_model_id": "sshleifer/tiny-gpt2"},
                },
                {
                    "peer_id": "coord-distil-cpu",
                    "role": "coordinator",
                    "urls": {"coordinator": "http://distil-cpu"},
                    "capabilities": {"backend": "cpu", "hf_model_id": "distilgpt2"},
                },
                {
                    "peer_id": "coord-distil-cuda",
                    "role": "coordinator",
                    "urls": {"coordinator": "http://distil-cuda"},
                    "capabilities": {"backend": "cuda", "hf_model_id": "distilgpt2"},
                },
                {
                    "peer_id": "stage0-distil-cuda",
                    "role": "miner",
                    "capabilities": {
                        "backend": "cuda",
                        "hf_model_id": "distilgpt2",
                        "real_llm_sharded_stage_capabilities": ["real_llm_sharded_cuda_stage0"],
                    },
                },
                {
                    "peer_id": "stage1-distil-cuda",
                    "role": "miner",
                    "capabilities": {
                        "backend": "cuda",
                        "hf_model_id": "distilgpt2",
                        "real_llm_sharded_stage_capabilities": ["real_llm_sharded_cuda_stage1"],
                    },
                },
            ],
        )

        self.assertTrue(route["usable_now"], route)
        self.assertEqual(route["coordinator_url"], "http://distil-cuda")
        self.assertEqual(route["coordinator_filter"]["mismatched_peers"], ["coord-tiny-cuda", "coord-distil-cpu"])
        self.assertIn("session_route_coordinator_filter_ready", route["diagnosis_codes"])
        self.assertIn("session_route_backend_filter_ready", route["diagnosis_codes"])

    def test_public_safety_blocks_raw_generation(self) -> None:
        with self.assertRaises(ValueError):
            protocol.assert_public_safe({"generated_text": "raw"})

    def test_safe_generation_summary_finds_nested_hash(self) -> None:
        summary = protocol.safe_generation_summary(
            {
                "payload": {
                    "generation": {
                        "max_new_tokens": 3,
                        "generated_token_count": 3,
                        "generated_text_hash": "sha256:x",
                    }
                }
            },
            max_new_tokens=3,
        )

        self.assertEqual(summary["schema"], "session_result_summary_v1")
        self.assertEqual(summary["generated_token_count"], 3)
        self.assertEqual(summary["generated_text_hash"], "sha256:x")
        self.assertTrue(summary["multi_token_generation_ready"])

    def test_safe_generation_summary_reports_batch_rows_without_raw_text(self) -> None:
        summary = protocol.safe_generation_summary(
            {
                "validation": {
                    "request_count": 2,
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "generated_text_hash": "sha256:batch",
                    "decoded_tokens_match": True,
                    "inference_results": [
                        {
                            "request_id": "req-1",
                            "prompt_hash": "sha256:p1",
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g1",
                            "generated_text": "raw one",
                            "generated_token_ids": [1, 2],
                            "decoded_tokens_match": True,
                        },
                        {
                            "request_id": "req-2",
                            "prompt_hash": "sha256:p2",
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g2",
                            "generated_text": "raw two",
                            "generated_token_ids": [3, 4],
                            "decoded_tokens_match": True,
                        },
                    ],
                }
            },
            max_new_tokens=2,
        )
        encoded = str(summary)

        self.assertTrue(summary["batch_generation_ready"])
        self.assertEqual(summary["request_count"], 2)
        self.assertEqual(summary["expected_request_count"], 2)
        self.assertEqual(summary["observed_request_count"], 2)
        self.assertTrue(summary["batch_identity_ready"])
        self.assertEqual([row["generated_text_hash"] for row in summary["results"]], ["sha256:g1", "sha256:g2"])
        self.assertNotIn("raw one", encoded)
        self.assertNotIn("raw two", encoded)
        self.assertNotIn('"generated_token_ids":', encoded)

    def test_safe_generation_summary_blocks_partial_batch_when_request_missing(self) -> None:
        summary = protocol.safe_generation_summary(
            {
                "validation": {
                    "request_count": 2,
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "generated_text_hash": "sha256:partial",
                    "decoded_tokens_match": True,
                    "inference_results": [
                        {
                            "request_id": "req-1",
                            "prompt_hash": "sha256:p1",
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g1",
                            "decoded_tokens_match": True,
                        }
                    ],
                }
            },
            max_new_tokens=2,
        )

        self.assertFalse(summary["batch_generation_ready"])
        self.assertEqual(summary["request_count"], 1)
        self.assertEqual(summary["expected_request_count"], 2)
        self.assertEqual(summary["observed_request_count"], 1)
        self.assertTrue(summary["results"][0]["multi_token_generation_ready"])

    def test_safe_generation_summary_blocks_batch_with_duplicate_request_identity(self) -> None:
        summary = protocol.safe_generation_summary(
            {
                "validation": {
                    "request_count": 2,
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "generated_text_hash": "sha256:duplicate",
                    "decoded_tokens_match": True,
                    "inference_results": [
                        {
                            "request_id": "req-1",
                            "prompt_hash": "sha256:p1",
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g1",
                            "decoded_tokens_match": True,
                        },
                        {
                            "request_id": "req-1",
                            "prompt_hash": "sha256:p1",
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g1-duplicate",
                            "decoded_tokens_match": True,
                        },
                    ],
                }
            },
            max_new_tokens=2,
        )

        self.assertFalse(summary["batch_generation_ready"])
        self.assertFalse(summary["batch_identity_ready"])
        self.assertEqual(summary["observed_request_count"], 2)

    def test_safe_generation_summary_blocks_batch_with_missing_request_identity(self) -> None:
        summary = protocol.safe_generation_summary(
            {
                "validation": {
                    "request_count": 2,
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "generated_text_hash": "sha256:missing-identity",
                    "decoded_tokens_match": True,
                    "inference_results": [
                        {
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g1",
                            "decoded_tokens_match": True,
                        },
                        {
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g2",
                            "decoded_tokens_match": True,
                        },
                    ],
                }
            },
            max_new_tokens=2,
        )

        self.assertFalse(summary["batch_generation_ready"])
        self.assertFalse(summary["batch_identity_ready"])
        self.assertEqual(summary["observed_request_count"], 2)

    def test_safe_generation_summary_blocks_batch_without_per_request_results(self) -> None:
        summary = protocol.safe_generation_summary(
            {
                "validation": {
                    "request_count": 2,
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "generated_text_hash": "sha256:aggregate-only",
                    "decoded_tokens_match": True,
                }
            },
            max_new_tokens=2,
        )

        self.assertTrue(summary["multi_token_generation_ready"])
        self.assertFalse(summary["batch_generation_ready"])
        self.assertEqual(summary["request_count"], 2)
        self.assertEqual(summary["expected_request_count"], 2)
        self.assertEqual(summary["observed_request_count"], 0)
        self.assertFalse(summary["batch_identity_ready"])
        self.assertEqual(summary["results"], [])

    def test_safe_stream_event_redacts_raw_generation_payload(self) -> None:
        event = protocol.safe_stream_event(
            {
                "task_id": "stage1-step0",
                "session_id": "session-1",
                "miner_id": "stage1-miner",
                "validation": {
                    "session_id": "session-1",
                    "stage_id": 1,
                    "request_id": "req-1",
                    "prompt_hash": "sha256:p1",
                    "generation_step": 0,
                    "generated_token_count": 1,
                    "max_new_tokens": 3,
                    "generated_text_hash": "sha256:step",
                    "generated_text": " raw token text",
                    "generated_token_ids": [42],
                    "decoded_tokens_match": True,
                },
            },
            max_new_tokens=3,
            observed_at=12.5,
        )
        encoded = str(event)

        self.assertEqual(event["schema"], "session_stream_event_v1")
        self.assertEqual(event["generated_token_count"], 1)
        self.assertEqual(event["request_id"], "req-1")
        self.assertEqual(event["prompt_hash"], "sha256:p1")
        self.assertEqual(event["generation_step"], 0)
        self.assertEqual(event["generated_text_hash"], "sha256:step")
        self.assertEqual(event["observed_at"], 12.5)
        self.assertFalse(event["raw_generated_text_public"])
        self.assertFalse(event["generated_token_ids_public"])
        self.assertNotIn("raw token text", encoded)
        self.assertNotIn('"generated_token_ids":', encoded)

    def test_safe_stream_events_expand_batch_results_per_request(self) -> None:
        events = protocol.safe_stream_events(
            {
                "task_id": "stage1-step0",
                "session_id": "session-1",
                "miner_id": "stage1-miner",
                "validation": {
                    "session_id": "session-1",
                    "stage_id": 1,
                    "generation_step": 0,
                    "generated_token_count": 1,
                    "max_new_tokens": 2,
                    "generated_text_hash": "sha256:aggregate",
                    "inference_results": [
                        {
                            "request_id": "req-1",
                            "prompt_hash": "sha256:p1",
                            "generation_step": 0,
                            "generated_token_count": 1,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:r1",
                            "generated_text": " raw one",
                            "generated_token_ids": [1],
                            "decoded_tokens_match": True,
                        },
                        {
                            "request_id": "req-2",
                            "prompt_hash": "sha256:p2",
                            "generation_step": 0,
                            "generated_token_count": 1,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:r2",
                            "generated_text": " raw two",
                            "generated_token_ids": [2],
                            "decoded_tokens_match": True,
                        },
                    ],
                },
            },
            max_new_tokens=2,
            observed_at=12.5,
        )
        encoded = str(events)

        self.assertEqual(
            [(event["request_id"], event["generated_token_count"], event["generated_text_hash"]) for event in events],
            [("req-1", 1, "sha256:r1"), ("req-2", 1, "sha256:r2")],
        )
        self.assertNotIn("raw one", encoded)
        self.assertNotIn("raw two", encoded)
        self.assertNotIn('"generated_token_ids":', encoded)


if __name__ == "__main__":
    unittest.main()
