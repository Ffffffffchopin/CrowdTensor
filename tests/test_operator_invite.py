from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "create_operator_invite.py"
SPEC = importlib.util.spec_from_file_location("create_operator_invite", SCRIPT_PATH)
create_operator_invite_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(create_operator_invite_script)

from crowdtensor.operator_invite import create_operator_invite  # noqa: E402


class OperatorInviteTests(unittest.TestCase):
    def test_operator_invite_writes_hashed_registry_and_private_invite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "operator_registry.json"
            invite_file = Path(tmp) / "generate-desk.operator.invite.json"

            invite = create_operator_invite(
                registry_path=registry,
                operator_id="generate-desk",
                roles=["admin"],
                label="bounded generator",
                token="operator-secret",
                allowed_workloads=["real-llm-sharded"],
                max_request_count=2,
                max_new_tokens=8,
                max_active_sessions=4,
                max_total_sessions=100,
                rate_limit=30,
                rate_window_seconds=60,
                invite_file=invite_file,
            )

            payload = json.loads(registry.read_text(encoding="utf-8"))
            entry = payload["operators"][0]
            self.assertEqual(entry["operator_id"], "generate-desk")
            self.assertEqual(entry["roles"], ["admin"])
            self.assertEqual(entry["label"], "bounded generator")
            self.assertTrue(entry["token"].startswith("sha256:"))
            self.assertEqual(entry["session_policy"]["allowed_workloads"], ["real_llm_sharded_infer"])
            self.assertEqual(entry["session_policy"]["max_request_count"], 2)
            self.assertEqual(entry["session_policy"]["max_new_tokens"], 8)
            self.assertEqual(entry["session_policy"]["max_total_sessions"], 100)
            self.assertNotIn("operator-secret", registry.read_text(encoding="utf-8"))
            self.assertEqual(invite["env"]["CROWDTENSOR_ADMIN_TOKEN"], "operator-secret")
            self.assertEqual(invite["operator_invite"]["operator_token"], "operator-secret")
            self.assertTrue(invite_file.exists())
            invite_payload = json.loads(invite_file.read_text(encoding="utf-8"))
            self.assertEqual(invite_payload["schema"], "crowdtensor_operator_invite_v1")
            self.assertEqual(invite_payload["operator_token"], "operator-secret")
            decoded = json.loads(base64.urlsafe_b64decode(invite["operator_invite_code"]).decode("utf-8"))
            self.assertEqual(decoded["operator_token"], "operator-secret")

    def test_operator_invite_duplicate_requires_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "operator_registry.json"
            create_operator_invite(
                registry_path=registry,
                operator_id="auditor-a",
                roles=["auditor"],
                token="first-secret",
            )

            with self.assertRaises(ValueError):
                create_operator_invite(
                    registry_path=registry,
                    operator_id="auditor-a",
                    roles=["auditor"],
                    token="second-secret",
                )

            replaced = create_operator_invite(
                registry_path=registry,
                operator_id="auditor-a",
                roles=["accounting"],
                token="second-secret",
                replace=True,
            )

            payload = json.loads(registry.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["operators"]), 1)
            self.assertEqual(payload["operators"][0]["roles"], ["accounting"])
            self.assertEqual(replaced["env"]["CROWDTENSOR_ADMIN_TOKEN"], "second-secret")
            self.assertNotIn("first-secret", registry.read_text(encoding="utf-8"))
            self.assertNotIn("second-secret", registry.read_text(encoding="utf-8"))

    def test_operator_invite_validates_roles_and_rate_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "operator_registry.json"
            with self.assertRaises(ValueError):
                create_operator_invite(
                    registry_path=registry,
                    operator_id="bad-role",
                    roles=["root"],
                    token="secret",
                )
            with self.assertRaises(ValueError):
                create_operator_invite(
                    registry_path=registry,
                    operator_id="bad-rate",
                    roles=["admin"],
                    token="secret",
                    rate_limit=1,
                )


if __name__ == "__main__":
    unittest.main()
