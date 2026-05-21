from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from crowdtensor.auth import hash_token


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "security_preflight.py"
SPEC = importlib.util.spec_from_file_location("security_preflight", SCRIPT_PATH)
security_preflight = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(security_preflight)


def preflight_args(**overrides):
    values = {
        "host": "127.0.0.1",
        "miner_token": "",
        "observer_token": "",
        "admin_token": "",
        "miner_token_registry": "",
        "cors_origins": [],
        "strict": False,
        "json": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def check_by_id(report: dict, check_id: str) -> dict:
    return next(check for check in report["checks"] if check["id"] == check_id)


class SecurityPreflightTests(unittest.TestCase):
    def test_local_default_warns_but_passes(self) -> None:
        report = security_preflight.run_preflight(preflight_args())

        self.assertTrue(report["ok"], report)
        self.assertFalse(report["summary"]["remote_bind"])
        self.assertEqual(report["summary"]["errors"], 0)
        self.assertGreaterEqual(report["summary"]["warnings"], 1)

    def test_remote_bind_requires_miner_observer_and_admin_auth(self) -> None:
        report = security_preflight.run_preflight(preflight_args(host="0.0.0.0"))

        self.assertFalse(report["ok"])
        self.assertTrue(report["summary"]["remote_bind"])
        self.assertEqual(report["summary"]["errors"], 3)
        self.assertFalse(check_by_id(report, "miner_token")["ok"])
        self.assertFalse(check_by_id(report, "observer_token")["ok"])
        self.assertFalse(check_by_id(report, "admin_token")["ok"])

    def test_remote_bind_rejects_local_demo_tokens(self) -> None:
        report = security_preflight.run_preflight(preflight_args(
            host="0.0.0.0",
            miner_token="local-miner",
            observer_token="local-observer",
            admin_token="local-admin",
        ))

        self.assertFalse(report["ok"])
        self.assertEqual(report["summary"]["errors"], 3)
        self.assertIn("local demo value", check_by_id(report, "miner_token")["message"])
        self.assertIn("local demo value", check_by_id(report, "observer_token")["message"])
        self.assertIn("local demo value", check_by_id(report, "admin_token")["message"])

    def test_remote_bind_accepts_hashed_shared_tokens_and_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "miners.json"
            registry.write_text(
                json.dumps({
                    "miners": [
                        {"miner_id": "remote-a", "token": hash_token("remote-a-token"), "enabled": True},
                    ],
                }),
                encoding="utf-8",
            )

            report = security_preflight.run_preflight(preflight_args(
                host="0.0.0.0",
                observer_token=hash_token("observer"),
                admin_token=hash_token("admin"),
                miner_token_registry=str(registry),
            ))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["summary"]["errors"], 0)
        self.assertEqual(check_by_id(report, "miner_token_registry")["severity"], "info")

    def test_registry_plaintext_token_is_high_risk(self) -> None:
        plaintext_token = "plain-token-value"
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "miners.json"
            registry.write_text(
                json.dumps({"miners": [{"miner_id": "remote-a", "token": plaintext_token, "enabled": True}]}),
                encoding="utf-8",
            )

            report = security_preflight.run_preflight(preflight_args(
                host="0.0.0.0",
                miner_token=hash_token("shared-miner"),
                observer_token=hash_token("observer"),
                admin_token=hash_token("admin"),
                miner_token_registry=str(registry),
            ))

        self.assertFalse(report["ok"])
        registry_check = check_by_id(report, "miner_token_registry")
        self.assertEqual(registry_check["severity"], "error")
        self.assertIn("plaintext", registry_check["message"])
        self.assertNotIn(plaintext_token, json.dumps(report, sort_keys=True))

    def test_registry_shape_errors_are_high_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "miners.json"
            registry.write_text(json.dumps({"miners": {}}), encoding="utf-8")

            report = security_preflight.run_preflight(preflight_args(miner_token_registry=str(registry)))

        self.assertFalse(report["ok"])
        self.assertIn("miners list", check_by_id(report, "miner_token_registry")["message"])

    def test_strict_treats_warnings_as_failures(self) -> None:
        report = security_preflight.run_preflight(preflight_args(
            miner_token="plain-miner",
            observer_token="plain-observer",
            admin_token="plain-admin",
            strict=True,
        ))

        self.assertFalse(report["ok"])
        self.assertEqual(report["summary"]["errors"], 0)
        self.assertGreaterEqual(report["summary"]["warnings"], 1)

    def test_cli_json_exit_codes_and_no_secret_leak(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--host", "0.0.0.0", "--json"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["ok"])
        self.assertNotIn("local-miner", completed.stdout)


if __name__ == "__main__":
    unittest.main()
