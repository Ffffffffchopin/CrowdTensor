from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
from pathlib import Path
import tempfile
import unittest

from crowdtensor.auth import token_matches


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "remote_demo_runbook_pack.py"
SPEC = importlib.util.spec_from_file_location("remote_demo_runbook_pack", SCRIPT_PATH)
remote_demo_runbook_pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(remote_demo_runbook_pack)


class RemoteDemoRunbookPackTests(unittest.TestCase):
    def _args(self, tmp: Path) -> argparse.Namespace:
        return argparse.Namespace(
            output_dir=str(tmp),
            registry=str(tmp / "miner_registry.json"),
            operator_env=str(tmp / "operator.private.env"),
            miner_env=str(tmp / "miner.private.env"),
            coordinator_url="https://coordinator.example",
            bind_host="0.0.0.0",
            port=8787,
            state_dir="state",
            miner_id="remote-a",
            label="remote demo miner",
            request_count=4,
            scenario_id="route-baseline",
            backlog=1,
            max_tasks=1,
            lease_seconds=15.0,
            compute_seconds=0.2,
            heartbeat_interval=0.1,
            max_request_attempts=5,
            miner_token="miner-secret",
            observer_token="observer-secret",
            admin_token="admin-secret",
            replace=True,
            json_out=str(tmp / "remote_demo_runbook.json"),
            markdown_out=str(tmp / "remote_demo_runbook.md"),
        )

    def test_build_from_args_writes_redacted_public_runbook_and_private_envs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            args = self._args(tmp)

            payload = remote_demo_runbook_pack.build_from_args(args)
            remote_demo_runbook_pack.write_json(payload, args.json_out)
            remote_demo_runbook_pack.write_markdown(payload, args.markdown_out)

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["schema"], "remote_demo_runbook_v1")
            self.assertEqual(payload["demo"]["route"], "remote_python_model_bundle_infer")
            self.assertEqual(payload["demo"]["workload_type"], "model_bundle_infer")
            self.assertEqual(payload["demo"]["scenario_id"], "route-baseline")
            self.assertTrue(payload["safety"]["registry_hashed"])
            self.assertIn("--task-lane python-cli:cpu:1:model_bundle_infer", payload["commands"]["start_coordinator"])
            self.assertIn("--backlog 0", payload["commands"]["start_coordinator"])
            self.assertIn("remote_compute_evidence_pack.py --mode collect", payload["commands"]["collect_remote_evidence"])
            self.assertIn("--scenario-id route-baseline", payload["commands"]["collect_remote_evidence"])
            self.assertIn("--coordinator https://coordinator.example", payload["commands"]["collect_support_bundle"])
            self.assertEqual(payload["miner_join_pack"]["schema"], "miner_join_pack_v1")
            self.assertTrue(payload["miner_join_pack"]["ready"])
            self.assertIn("bash miner_join.sh", payload["miner_join_pack"]["recommended_command"])

            registry = json.loads((tmp / "miner_registry.json").read_text(encoding="utf-8"))
            token_hash = registry["miners"][0]["token"]
            self.assertTrue(token_hash.startswith("sha256:"))
            self.assertTrue(token_matches("miner-secret", token_hash))

            operator_env = (tmp / "operator.private.env").read_text(encoding="utf-8")
            miner_env = (tmp / "miner.private.env").read_text(encoding="utf-8")
            self.assertIn("export CROWDTENSOR_ADMIN_TOKEN='admin-secret'", operator_env)
            self.assertIn("export CROWDTENSOR_OBSERVER_TOKEN='observer-secret'", operator_env)
            self.assertNotIn("miner-secret", operator_env)
            self.assertIn("export CROWDTENSOR_MINER_TOKEN='miner-secret'", miner_env)
            self.assertNotIn("admin-secret", miner_env)
            self.assertEqual(stat.S_IMODE(os.stat(tmp / "operator.private.env").st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(os.stat(tmp / "miner.private.env").st_mode), 0o600)
            self.assertTrue((tmp / "miner_join.sh").is_file())
            self.assertTrue((tmp / "MINER_JOIN.md").is_file())

            public_text = json.dumps(payload, sort_keys=True)
            public_text += (tmp / "remote_demo_runbook.md").read_text(encoding="utf-8")
            public_text += (tmp / "MINER_JOIN.md").read_text(encoding="utf-8")
            self.assertNotIn("miner-secret", public_text)
            self.assertNotIn("observer-secret", public_text)
            self.assertNotIn("admin-secret", public_text)
            self.assertNotIn("CROWDTENSOR_MINER_TOKEN=miner-secret", public_text)

    def test_render_markdown_includes_public_commands_and_safety(self) -> None:
        payload = {
            "schema": "remote_demo_runbook_v1",
            "generated_at": "2026-05-22T00:00:00+00:00",
            "ok": True,
            "demo": {
                "coordinator_url": "https://coordinator.example",
                "workload_type": "model_bundle_infer",
                "route": "remote_python_model_bundle_infer",
                "request_count": 4,
                "scenario_schema": "model_bundle_inference_scenario_v1",
                "scenario_id": "route-baseline",
            },
            "files": {
                "registry": "state/miner_registry.json",
                "operator_private_env": "operator.private.env",
                "miner_private_env": "miner.private.env",
                "miner_join_script": "miner_join.sh",
                "miner_join_runbook": "MINER_JOIN.md",
            },
            "tokens": {
                "observer_hash_prefix": "sha256",
                "admin_hash_prefix": "sha256",
            },
            "commands": {
                "start_coordinator": "crowdtensord --task-lane python-cli:cpu:1:model_bundle_infer",
            },
            "miner_join_pack": {
                "schema": "miner_join_pack_v1",
                "recommended_command": "bash miner_join.sh",
            },
            "safety": {
                "registry_hashed": True,
                "public_artifact_redacted": True,
                "private_env_files": True,
                "requires_tls_or_vpn": True,
            },
            "limitations": ["Controlled demo"],
        }

        markdown = remote_demo_runbook_pack.render_markdown(payload)

        self.assertIn("# CrowdTensor Remote Demo Runbook", markdown)
        self.assertIn("remote_python_model_bundle_infer", markdown)
        self.assertIn("Scenario: `route-baseline`", markdown)
        self.assertIn("start_coordinator", markdown)
        self.assertIn("Registry hashed", markdown)
        self.assertIn("Miner Join Pack", markdown)


if __name__ == "__main__":
    unittest.main()
