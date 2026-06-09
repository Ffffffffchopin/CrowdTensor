from __future__ import annotations

import importlib.util
import base64
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "create_miner_invite.py"
SPEC = importlib.util.spec_from_file_location("create_miner_invite", SCRIPT_PATH)
create_miner_invite = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(create_miner_invite)


class MinerInviteTests(unittest.TestCase):
    def test_invite_writes_hashed_registry_and_returns_plaintext_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "miners.json"

            invite = create_miner_invite.create_invite(
                registry_path=registry,
                miner_id="remote-a",
                coordinator_url="https://coordinator.example/",
                label="alpha",
                token="plain-token",
            )

            payload = json.loads(registry.read_text(encoding="utf-8"))
            entry = payload["miners"][0]
            self.assertEqual(entry["miner_id"], "remote-a")
            self.assertEqual(entry["label"], "alpha")
            self.assertTrue(entry["token"].startswith("sha256:"))
            self.assertNotIn("plain-token", registry.read_text(encoding="utf-8"))
            self.assertEqual(invite["env"]["CROWDTENSOR_MINER_TOKEN"], "plain-token")
            self.assertIn("CROWDTENSOR_MINER_TOKEN=plain-token", invite["run_command"])
            self.assertIn("--coordinator https://coordinator.example", invite["run_command"])
            self.assertEqual(invite["join_invite"]["schema"], "crowdtensor_miner_join_invite_v1")
            self.assertEqual(invite["join_invite"]["stage"], "both")
            self.assertEqual(invite["join_invite"]["backend"], "cpu")
            self.assertIn("crowdtensor join", invite["product_join_command"])
            self.assertIn("--coordinator-url https://coordinator.example", invite["product_join_command"])
            self.assertIn("--stage both", invite["product_join_command"])
            decoded = json.loads(base64.urlsafe_b64decode(invite["join_invite_code"]).decode("utf-8"))
            self.assertEqual(decoded["miner_token"], "plain-token")
            self.assertEqual(decoded["token_hash"], invite["token_hash"])

    def test_invite_records_product_policy_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "miners.json"

            invite = create_miner_invite.create_invite(
                registry_path=registry,
                miner_id="gpu-stage0",
                coordinator_url="https://coordinator.example",
                token="plain-token",
                stage="stage0",
                backend="cuda",
                hf_model_id="sshleifer/tiny-gpt2",
                max_tasks=4,
                max_runtime_seconds=120,
                trust_tier="probation",
                quota_task_limit=25,
                claim_rate_limit=2,
                claim_rate_window_seconds=60,
                reward_account="acct_123",
                invite_file=Path(tmp) / "gpu-stage0.invite.json",
            )

            payload = json.loads(registry.read_text(encoding="utf-8"))
            policy = payload["miners"][0]["join_policy"]
            self.assertEqual(policy["schema"], "crowdtensor_miner_join_policy_v1")
            self.assertEqual(policy["stage"], "stage0")
            self.assertEqual(policy["backend"], "cuda")
            self.assertEqual(policy["max_tasks"], 4)
            self.assertEqual(policy["max_runtime_seconds"], 120.0)
            self.assertEqual(policy["trust_tier"], "probation")
            self.assertEqual(policy["quota_task_limit"], 25)
            self.assertEqual(policy["claim_rate_limit"], 2)
            self.assertEqual(policy["claim_rate_window_seconds"], 60.0)
            self.assertEqual(policy["reward_account"], "acct_123")
            self.assertIn("--backend cuda", invite["product_join_command"])
            self.assertIn("--max-tasks 4", invite["product_join_command"])
            self.assertNotIn("plain-token", registry.read_text(encoding="utf-8"))
            invite_file = Path(invite["invite_file"])
            self.assertTrue(invite_file.exists())
            invite_payload = json.loads(invite_file.read_text(encoding="utf-8"))
            self.assertEqual(invite_payload["schema"], "crowdtensor_miner_join_invite_v1")
            self.assertEqual(invite_payload["miner_token"], "plain-token")
            self.assertEqual(invite_payload["policy"]["trust_tier"], "probation")
            self.assertEqual(invite_payload["policy"]["claim_rate_limit"], 2)

    def test_invite_requires_complete_claim_rate_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "miners.json"

            with self.assertRaises(ValueError):
                create_miner_invite.create_invite(
                    registry_path=registry,
                    miner_id="rate-a",
                    coordinator_url="https://coordinator.example",
                    token="plain-token",
                    claim_rate_limit=1,
                )
            with self.assertRaises(ValueError):
                create_miner_invite.create_invite(
                    registry_path=registry,
                    miner_id="rate-a",
                    coordinator_url="https://coordinator.example",
                    token="plain-token",
                    claim_rate_window_seconds=60,
                )

    def test_duplicate_requires_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "miners.json"
            create_miner_invite.create_invite(
                registry_path=registry,
                miner_id="remote-a",
                coordinator_url="http://127.0.0.1:8787",
                token="alpha-private-1",
            )
            with self.assertRaises(ValueError):
                create_miner_invite.create_invite(
                    registry_path=registry,
                    miner_id="remote-a",
                    coordinator_url="http://127.0.0.1:8787",
                    token="bravo-private-2",
                )

            replaced = create_miner_invite.create_invite(
                registry_path=registry,
                miner_id="remote-a",
                coordinator_url="http://127.0.0.1:8787",
                token="bravo-private-2",
                replace=True,
            )

            payload = json.loads(registry.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["miners"]), 1)
            self.assertEqual(replaced["env"]["CROWDTENSOR_MINER_TOKEN"], "bravo-private-2")
            self.assertNotIn("alpha-private-1", registry.read_text(encoding="utf-8"))
            self.assertNotIn("bravo-private-2", registry.read_text(encoding="utf-8"))

    def test_invalid_registry_shape_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "miners.json"
            registry.write_text(json.dumps({"miners": {}}), encoding="utf-8")

            with self.assertRaises(ValueError):
                create_miner_invite.create_invite(
                    registry_path=registry,
                    miner_id="remote-a",
                    coordinator_url="http://127.0.0.1:8787",
                    token="plain",
                )


if __name__ == "__main__":
    unittest.main()
