from __future__ import annotations

import time
import unittest

from crowdtensor.p2p_lite import sanitize_peer, stable_peer_secret
from crowdtensor.real_p2p import (
    ProviderRecordStore,
    build_provider_record,
    create_app,
    nat_relay_diagnostics,
    peer_leak_paths,
    provider_peer_scoring,
)
from crowdtensor.session_protocol import build_session_request


class RealP2PTests(unittest.TestCase):
    def test_signed_provider_records_resolve_route_and_prune(self) -> None:
        secret = stable_peer_secret("real-p2p-unit")
        store = ProviderRecordStore(swarm_id="unit", ttl_seconds=10.0, record_secret=secret, require_signed=True)
        now = 1000.0
        for peer in [
            {
                "swarm_id": "unit",
                "peer_id": "coord",
                "role": "coordinator",
                "urls": {"coordinator": "http://127.0.0.1:8787"},
                "capabilities": {"backend": "cpu"},
                "last_seen": now,
            },
            {
                "swarm_id": "unit",
                "peer_id": "stage0",
                "role": "miner",
                "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                "last_seen": now,
            },
            {
                "swarm_id": "unit",
                "peer_id": "stage1",
                "role": "miner",
                "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]},
                "last_seen": now,
            },
        ]:
            record = build_provider_record(sanitize_peer(peer, now=now), secret, signed_at=now, now=now)
            store.announce(record, now=now)

        catalog = store.catalog_payload(now=now)
        route = store.route_lookup(build_session_request(prompt_text="hello", backend="cpu"), now=now)

        self.assertTrue(route["ok"], route)
        self.assertEqual(catalog["schema"], "real_p2p_provider_catalog_v1")
        self.assertEqual(catalog["registry"]["signed_provider_record_count"], 3)
        self.assertIn("replaceable_discovery_backend_ready", catalog["diagnosis_codes"])
        self.assertEqual(route["route"]["coordinator_url"], "http://127.0.0.1:8787")
        self.assertEqual(store.prune(now=now + 11.0), 3)

    def test_requires_signed_provider_records(self) -> None:
        secret = stable_peer_secret("real-p2p-unit")
        store = ProviderRecordStore(swarm_id="unit", record_secret=secret, require_signed=True)

        with self.assertRaises(ValueError):
            store.announce({"peer_id": "stage0", "role": "miner"})

    def test_rejects_cross_swarm_provider_records(self) -> None:
        secret = stable_peer_secret("real-p2p-unit")
        store = ProviderRecordStore(swarm_id="unit", record_secret=secret, require_signed=True)
        now = 1000.0
        record = build_provider_record(
            sanitize_peer(
                {
                    "swarm_id": "other-swarm",
                    "peer_id": "stage0",
                    "role": "miner",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                    "last_seen": now,
                },
                now=now,
            ),
            secret,
            signed_at=now,
            now=now,
        )

        with self.assertRaises(ValueError):
            store.announce(record, now=now)
        self.assertEqual(store.provider_records(now=now), [])

    def test_provider_record_rejects_secret_metadata(self) -> None:
        with self.assertRaises(ValueError):
            build_provider_record({"peer_id": "x", "capabilities": {"miner_token": "secret"}})
        self.assertTrue(peer_leak_paths({"prompt_text": "raw prompt"}))

    def test_nat_relay_diagnostics_are_explicit_boundaries(self) -> None:
        payload = nat_relay_diagnostics(
            bind_host="0.0.0.0",
            public_host="example.com",
            listen_port=8888,
            bootstrap_urls=["http://bootstrap:8888"],
        )

        self.assertTrue(payload["ok"])
        self.assertFalse(payload["nat_traversal_ready"])
        self.assertFalse(payload["relay_ready"])
        self.assertIn("bootstrap_peer_configured", payload["diagnosis_codes"])

    def test_peer_scoring_prioritizes_healthy_successful_provider(self) -> None:
        secret = stable_peer_secret("real-p2p-scoring")
        store = ProviderRecordStore(swarm_id="unit", ttl_seconds=30.0, record_secret=secret, require_signed=True)
        now = 2000.0
        for peer in [
            {
                "swarm_id": "unit",
                "peer_id": "coord",
                "role": "coordinator",
                "urls": {"coordinator": "http://127.0.0.1:8787"},
                "last_seen": now,
            },
            {
                "swarm_id": "unit",
                "peer_id": "stage0-bad",
                "role": "miner",
                "capabilities": {
                    "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"],
                    "failed_result_count": 4,
                    "stale_result_count": 2,
                },
                "last_seen": now,
            },
            {
                "swarm_id": "unit",
                "peer_id": "stage0-good",
                "role": "miner",
                "capabilities": {
                    "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"],
                    "accepted_result_count": 8,
                },
                "last_seen": now,
            },
            {
                "swarm_id": "unit",
                "peer_id": "stage1",
                "role": "miner",
                "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]},
                "last_seen": now,
            },
        ]:
            store.announce(build_provider_record(sanitize_peer(peer, now=now), secret, signed_at=now, now=now), now=now)

        catalog = store.catalog_payload(now=now)
        route = store.route_lookup(build_session_request(prompt_text="hello", backend="cpu"), now=now)

        self.assertIn("peer_scoring_ready", catalog["diagnosis_codes"])
        self.assertEqual(catalog["peer_scoring"]["schema"], "real_p2p_peer_scoring_v1")
        self.assertGreater(
            catalog["peer_scoring"]["scores"]["stage0-good"]["score"],
            catalog["peer_scoring"]["scores"]["stage0-bad"]["score"],
        )
        self.assertEqual(route["route"]["matched_capabilities"]["real_llm_sharded_stage0"], "stage0-good")
        self.assertIn("peer_scoring_ready", provider_peer_scoring({"peer_id": "x"})["diagnosis_codes"])

    def test_route_lookup_filters_stage_providers_by_model_id(self) -> None:
        secret = stable_peer_secret("real-p2p-model-filter")
        store = ProviderRecordStore(swarm_id="unit", ttl_seconds=30.0, record_secret=secret, require_signed=True)
        now = 2500.0
        for peer in [
            {
                "swarm_id": "unit",
                "peer_id": "coord",
                "role": "coordinator",
                "urls": {"coordinator": "http://127.0.0.1:8787"},
                "last_seen": now,
            },
            {
                "swarm_id": "unit",
                "peer_id": "stage0-tiny",
                "role": "miner",
                "capabilities": {
                    "hf_model_id": "sshleifer/tiny-gpt2",
                    "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"],
                },
                "last_seen": now,
            },
            {
                "swarm_id": "unit",
                "peer_id": "stage0-distil",
                "role": "miner",
                "capabilities": {
                    "hf_model_id": "distilgpt2",
                    "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"],
                },
                "last_seen": now,
            },
            {
                "swarm_id": "unit",
                "peer_id": "stage1-tiny",
                "role": "miner",
                "capabilities": {
                    "hf_model_id": "sshleifer/tiny-gpt2",
                    "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"],
                },
                "last_seen": now,
            },
        ]:
            store.announce(build_provider_record(sanitize_peer(peer, now=now), secret, signed_at=now, now=now), now=now)

        route = store.route_lookup(
            build_session_request(prompt_text="hello", backend="cpu", hf_model_id="distilgpt2"),
            now=now,
        )

        self.assertFalse(route["ok"], route)
        self.assertEqual(route["route"]["matched_capabilities"]["real_llm_sharded_stage0"], "stage0-distil")
        self.assertIn("real_llm_sharded_stage1", route["route"]["missing_capabilities"])
        self.assertIn("session_route_model_mismatch", route["diagnosis_codes"])

    def test_route_lookup_filters_coordinator_provider_by_model_id(self) -> None:
        secret = stable_peer_secret("real-p2p-coordinator-filter")
        store = ProviderRecordStore(swarm_id="unit", ttl_seconds=30.0, record_secret=secret, require_signed=True)
        now = 2600.0
        for peer in [
            {
                "swarm_id": "unit",
                "peer_id": "coord-tiny",
                "role": "coordinator",
                "urls": {"coordinator": "http://tiny.example:8787"},
                "capabilities": {"backend": "cpu", "hf_model_id": "sshleifer/tiny-gpt2"},
                "last_seen": now,
            },
            {
                "swarm_id": "unit",
                "peer_id": "coord-distil",
                "role": "coordinator",
                "urls": {"coordinator": "http://distil.example:8787"},
                "capabilities": {"backend": "cpu", "hf_model_id": "distilgpt2"},
                "last_seen": now,
            },
            {
                "swarm_id": "unit",
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
                "swarm_id": "unit",
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
            store.announce(build_provider_record(sanitize_peer(peer, now=now), secret, signed_at=now, now=now), now=now)

        route = store.route_lookup(
            build_session_request(prompt_text="hello", backend="cpu", hf_model_id="distilgpt2"),
            now=now,
        )

        self.assertTrue(route["ok"], route)
        self.assertEqual(route["route"]["coordinator_url"], "http://distil.example:8787")
        self.assertEqual(route["route"]["coordinator_filter"]["mismatched_peers"], ["coord-tiny"])
        self.assertIn("session_route_coordinator_filter_ready", route["diagnosis_codes"])

    def test_route_lookup_syncs_bootstrap_records_without_catalog_preheat(self) -> None:
        from fastapi.testclient import TestClient
        from unittest.mock import patch

        secret = stable_peer_secret("real-p2p-bootstrap-route")
        now = time.time()
        bootstrap_records = []
        for peer in [
            {
                "swarm_id": "unit",
                "peer_id": "coord",
                "role": "coordinator",
                "urls": {"coordinator": "http://127.0.0.1:8787"},
                "capabilities": {"backend": "cpu"},
                "last_seen": now,
            },
            {
                "swarm_id": "unit",
                "peer_id": "stage0",
                "role": "miner",
                "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                "last_seen": now,
            },
            {
                "swarm_id": "unit",
                "peer_id": "stage1",
                "role": "miner",
                "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]},
                "last_seen": now,
            },
        ]:
            bootstrap_records.append(
                build_provider_record(sanitize_peer(peer, now=now), secret, signed_at=now, now=now)
            )

        store = ProviderRecordStore(swarm_id="unit", ttl_seconds=60.0, record_secret=secret, require_signed=True)
        app = create_app(store=store, bootstrap_urls=["http://bootstrap.example"], listen_port=9999)

        with patch("crowdtensor.real_p2p.post_provider_record", return_value={"ok": True}), patch(
            "crowdtensor.real_p2p.fetch_provider_catalog",
            return_value={"schema": "real_p2p_provider_catalog_v1", "providers": bootstrap_records},
        ):
            response = TestClient(app).post(
                "/real-p2p/route",
                json={"session_request": build_session_request(prompt_text="hello", backend="cpu")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["route"]["coordinator_url"], "http://127.0.0.1:8787")
        self.assertEqual(payload["route"]["matched_capabilities"]["real_llm_sharded_stage0"], "stage0")
        self.assertEqual(payload["route"]["matched_capabilities"]["real_llm_sharded_stage1"], "stage1")
        self.assertEqual(payload["provider_count"], 3)
        self.assertEqual(payload["bootstrap_sync"]["merged_provider_count"], 3)
        self.assertIn("real_p2p_route_bootstrap_sync_ready", payload["diagnosis_codes"])
        self.assertIn("real_p2p_bootstrap_records_merged", payload["diagnosis_codes"])

    def test_route_bootstrap_sync_ignores_cross_swarm_records(self) -> None:
        from fastapi.testclient import TestClient
        from unittest.mock import patch

        secret = stable_peer_secret("real-p2p-bootstrap-route")
        now = time.time()
        records = [
            build_provider_record(
                sanitize_peer(
                    {
                        "swarm_id": "unit",
                        "peer_id": "coord",
                        "role": "coordinator",
                        "urls": {"coordinator": "http://127.0.0.1:8787"},
                        "last_seen": now,
                    },
                    now=now,
                ),
                secret,
                signed_at=now,
                now=now,
            ),
            build_provider_record(
                sanitize_peer(
                    {
                        "swarm_id": "unit",
                        "peer_id": "stage0",
                        "role": "miner",
                        "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                        "last_seen": now,
                    },
                    now=now,
                ),
                secret,
                signed_at=now,
                now=now,
            ),
            build_provider_record(
                sanitize_peer(
                    {
                        "swarm_id": "other-swarm",
                        "peer_id": "stage1-other",
                        "role": "miner",
                        "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]},
                        "last_seen": now,
                    },
                    now=now,
                ),
                secret,
                signed_at=now,
                now=now,
            ),
        ]
        store = ProviderRecordStore(swarm_id="unit", ttl_seconds=60.0, record_secret=secret, require_signed=True)
        app = create_app(store=store, bootstrap_urls=["http://bootstrap.example"], listen_port=9999)

        with patch("crowdtensor.real_p2p.post_provider_record", return_value={"ok": True}), patch(
            "crowdtensor.real_p2p.fetch_provider_catalog",
            return_value={"schema": "real_p2p_provider_catalog_v1", "providers": records},
        ):
            response = TestClient(app).post(
                "/real-p2p/route",
                json={"session_request": build_session_request(prompt_text="hello", backend="cpu")},
            )

        payload = response.json()
        self.assertFalse(payload["ok"], payload)
        self.assertEqual(payload["provider_count"], 2)
        self.assertEqual(payload["bootstrap_sync"]["merged_provider_count"], 2)
        self.assertEqual(payload["bootstrap_sync"]["rejected_provider_count"], 1)
        self.assertIn("real_llm_sharded_stage1", payload["route"]["missing_capabilities"])
        self.assertNotIn("stage1-other", [peer["peer_id"] for peer in store.peers(now=now)])
        self.assertIn("real_p2p_swarm_isolation_ready", payload["diagnosis_codes"])
        self.assertIn("real_p2p_bootstrap_records_rejected", payload["diagnosis_codes"])


if __name__ == "__main__":
    unittest.main()
