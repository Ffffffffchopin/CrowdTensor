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


if __name__ == "__main__":
    unittest.main()
