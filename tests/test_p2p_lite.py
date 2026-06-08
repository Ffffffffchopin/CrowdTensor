from __future__ import annotations

import unittest

from crowdtensor.p2p_lite import PeerCatalog, peer_leak_paths, sanitize_peer, sign_peer_announcement, stable_peer_secret
from crowdtensor.session_protocol import build_session_request
from scripts import p2p_lite_discovery_check


class P2PLiteTests(unittest.TestCase):
    def test_catalog_resolves_coordinator_and_stage_peers(self) -> None:
        catalog = PeerCatalog(swarm_id="unit", ttl_seconds=10.0)
        now = 1000.0
        catalog.announce(
            {
                "peer_id": "coordinator",
                "role": "coordinator",
                "urls": {"coordinator": "http://127.0.0.1:8787"},
                "last_seen": now,
            },
            now=now,
        )
        catalog.announce(
            {
                "peer_id": "stage0",
                "role": "miner",
                "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                "last_seen": now,
            },
            now=now,
        )
        catalog.announce(
            {
                "peer_id": "stage1",
                "role": "miner",
                "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]},
                "last_seen": now,
            },
            now=now,
        )

        request = build_session_request(prompt_text="hello", backend="cpu")
        resolved = catalog.resolve(request, now=now)

        self.assertTrue(resolved["ok"], resolved)
        self.assertEqual(resolved["route"]["coordinator_url"], "http://127.0.0.1:8787")
        self.assertEqual(len(catalog.peers(now=now)), 3)
        self.assertEqual(catalog.prune(now=now + 11.0), 3)
        self.assertEqual(catalog.peers(now=now + 11.0), [])

    def test_catalog_route_filters_stage_peers_by_model_id(self) -> None:
        catalog = PeerCatalog(swarm_id="unit", ttl_seconds=10.0)
        now = 1000.0
        for peer in [
            {
                "peer_id": "coordinator",
                "role": "coordinator",
                "urls": {"coordinator": "http://127.0.0.1:8787"},
                "last_seen": now,
            },
            {
                "peer_id": "stage0-tiny",
                "role": "miner",
                "capabilities": {
                    "hf_model_id": "sshleifer/tiny-gpt2",
                    "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"],
                },
                "last_seen": now,
            },
            {
                "peer_id": "stage0-distil",
                "role": "miner",
                "capabilities": {
                    "hf_model_id": "distilgpt2",
                    "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"],
                },
                "last_seen": now,
            },
            {
                "peer_id": "stage1-tiny",
                "role": "miner",
                "capabilities": {
                    "hf_model_id": "sshleifer/tiny-gpt2",
                    "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"],
                },
                "last_seen": now,
            },
        ]:
            catalog.announce(peer, now=now)

        request = build_session_request(prompt_text="hello", backend="cpu", hf_model_id="distilgpt2")
        resolved = catalog.resolve(request, now=now)

        self.assertFalse(resolved["ok"], resolved)
        self.assertEqual(resolved["route"]["matched_capabilities"]["real_llm_sharded_stage0"], "stage0-distil")
        self.assertIn("real_llm_sharded_stage1", resolved["route"]["missing_capabilities"])
        self.assertIn("session_route_model_mismatch", resolved["diagnosis_codes"])

    def test_catalog_route_filters_coordinator_by_model_id(self) -> None:
        catalog = PeerCatalog(swarm_id="unit", ttl_seconds=10.0)
        now = 1000.0
        for peer in [
            {
                "peer_id": "coord-tiny",
                "role": "coordinator",
                "urls": {"coordinator": "http://tiny.example:8787"},
                "capabilities": {"backend": "cpu", "hf_model_id": "sshleifer/tiny-gpt2"},
                "last_seen": now,
            },
            {
                "peer_id": "coord-distil",
                "role": "coordinator",
                "urls": {"coordinator": "http://distil.example:8787"},
                "capabilities": {"backend": "cpu", "hf_model_id": "distilgpt2"},
                "last_seen": now,
            },
            {
                "peer_id": "stage0-distil",
                "role": "miner",
                "capabilities": {
                    "backend": "cpu",
                    "hf_model_id": "distilgpt2",
                    "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"],
                },
                "last_seen": now,
            },
            {
                "peer_id": "stage1-distil",
                "role": "miner",
                "capabilities": {
                    "backend": "cpu",
                    "hf_model_id": "distilgpt2",
                    "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"],
                },
                "last_seen": now,
            },
        ]:
            catalog.announce(peer, now=now)

        request = build_session_request(prompt_text="hello", backend="cpu", hf_model_id="distilgpt2")
        resolved = catalog.resolve(request, now=now)

        self.assertTrue(resolved["ok"], resolved)
        self.assertEqual(resolved["route"]["coordinator_url"], "http://distil.example:8787")
        self.assertEqual(resolved["route"]["coordinator_filter"]["mismatched_peers"], ["coord-tiny"])
        self.assertIn("session_route_coordinator_filter_ready", resolved["diagnosis_codes"])

    def test_sanitize_peer_rejects_secret_metadata(self) -> None:
        with self.assertRaises(ValueError):
            sanitize_peer({"peer_id": "x", "capabilities": {"miner_token": "secret"}})

    def test_peer_leak_paths_detects_prompt_fields(self) -> None:
        self.assertTrue(peer_leak_paths({"prompt": "raw"}))

    def test_signed_peer_announcement_is_verified_and_scored(self) -> None:
        secret = stable_peer_secret("unit")
        catalog = PeerCatalog(swarm_id="unit", ttl_seconds=10.0, peer_secret=secret, require_signed=True)
        now = 1000.0
        peer = sanitize_peer(
            {
                "peer_id": "stage0",
                "role": "miner",
                "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                "last_seen": now,
            },
            now=now,
        )
        signed = sign_peer_announcement(peer, secret, signed_at=now)

        announced = catalog.announce(signed, now=now)
        payload = catalog.catalog_payload(now=now)

        self.assertTrue(announced["identity_verified"], announced)
        self.assertGreaterEqual(announced["health_score"], 90)
        self.assertEqual(payload["registry"]["signed_peer_count"], 1)

    def test_signed_peer_announcement_rejects_tampering(self) -> None:
        secret = stable_peer_secret("unit")
        catalog = PeerCatalog(swarm_id="unit", peer_secret=secret, require_signed=True)
        now = 1000.0
        signed = sign_peer_announcement(
            sanitize_peer(
                {
                    "peer_id": "stage0",
                    "role": "miner",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                    "last_seen": now,
                },
                now=now,
            ),
            secret,
            signed_at=now,
        )
        signed["capabilities"]["real_llm_sharded_stage_capabilities"] = ["real_llm_sharded_stage1"]

        with self.assertRaises(ValueError):
            catalog.announce(signed, now=now)

    def test_discovery_check_reports_prompt_and_answer_scope(self) -> None:
        result = p2p_lite_discovery_check.run_check(p2p_lite_discovery_check.parse_args([]))

        self.assertEqual(result["schema"], "p2p_lite_discovery_check_v1")
        self.assertFalse(result["output_request"]["include_output"])
        self.assertFalse(result["output_request"]["raw_prompt_public"])
        self.assertEqual(result["prompt_scope"]["source"], "route-check-placeholder")
        self.assertEqual(result["prompt_scope"]["prompt_count"], 1)
        self.assertFalse(result["prompt_scope"]["inline_prompt_text"])
        self.assertFalse(result["prompt_scope"]["raw_prompt_public"])
        self.assertEqual(result["answer_scope"]["scope_state"], "no-local-answer")
        self.assertFalse(result["answer_scope"]["visible_in_terminal"])
        self.assertTrue(result["shareable_summary"]["public_artifact_safe"])
        self.assertTrue(result["cpu_route"]["ok"])
        self.assertTrue(result["cuda_route"]["ok"])
        self.assertEqual(
            result["cuda_route"]["route"]["coordinator_filter"]["compatible_candidate_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
