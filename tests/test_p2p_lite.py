from __future__ import annotations

import unittest

from crowdtensor.p2p_lite import PeerCatalog, peer_leak_paths, sanitize_peer
from crowdtensor.session_protocol import build_session_request


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

    def test_sanitize_peer_rejects_secret_metadata(self) -> None:
        with self.assertRaises(ValueError):
            sanitize_peer({"peer_id": "x", "capabilities": {"miner_token": "secret"}})

    def test_peer_leak_paths_detects_prompt_fields(self) -> None:
        self.assertTrue(peer_leak_paths({"prompt": "raw"}))


if __name__ == "__main__":
    unittest.main()
