from __future__ import annotations

import importlib.util
import json
import errno
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from crowdtensor.auth import hash_token


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "doctor.py"
SPEC = importlib.util.spec_from_file_location("doctor", SCRIPT_PATH)
doctor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(doctor)


def check_by_id(report: dict, check_id: str) -> dict:
    return next(check for check in report["checks"] if check["id"] == check_id)


class DoctorTests(unittest.TestCase):
    def test_default_local_doctor_passes(self) -> None:
        tmp_root = Path(self._tmp_dir())
        args = doctor.parse_args([
            "--root",
            str(ROOT),
            "--state-dir",
            str(tmp_root / "state"),
            "--port",
            "0",
        ])

        report = doctor.build_report(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["summary"]["errors"], 0)
        self.assertTrue(check_by_id(report, "python_version")["ok"])
        self.assertTrue(check_by_id(report, "state_dir")["ok"])
        self.assertIn("runtime_matrix", report)
        self.assertGreaterEqual(report["summary"]["runtime_matrix_available"], 1)
        self.assertIn("cpu_baseline_ready", report["summary"]["runtime_matrix_diagnosis_codes"])
        self.assertIn("cpu_baseline_ready", report["runtime_matrix"]["diagnosis_summary"]["codes"])
        self.assertIn("model_bundle_infer", report["runtime_matrix"]["summary"]["available_workloads"])

    def test_strict_blocks_warnings(self) -> None:
        tmp_root = Path(self._tmp_dir())
        args = doctor.parse_args([
            "--root",
            str(ROOT),
            "--state-dir",
            str(tmp_root / "state"),
            "--port",
            "0",
            "--browser",
            "--browser-path",
            str(tmp_root / "missing-browser"),
            "--strict",
        ])
        with patch.object(doctor, "module_available", side_effect=lambda name: False if name == "playwright" else True):
            report = doctor.build_report(args)

        self.assertFalse(report["ok"])
        self.assertGreaterEqual(report["summary"]["warnings"], 1)
        self.assertEqual(check_by_id(report, "browser_playwright")["severity"], "warning")

    def test_port_in_use_is_error(self) -> None:
        tmp_root = Path(self._tmp_dir())
        args = doctor.parse_args([
            "--root",
            str(ROOT),
            "--state-dir",
            str(tmp_root / "state"),
            "--port",
            "8787",
        ])

        with patch.object(doctor.socket, "socket", side_effect=OSError(errno.EADDRINUSE, "Address already in use")):
            report = doctor.build_report(args)

        self.assertFalse(report["ok"])
        self.assertFalse(check_by_id(report, "port_bind")["ok"])

    def test_missing_dependency_is_error(self) -> None:
        tmp_root = Path(self._tmp_dir())
        args = doctor.parse_args([
            "--root",
            str(ROOT),
            "--state-dir",
            str(tmp_root / "state"),
            "--port",
            "0",
        ])
        with patch.object(doctor, "module_available", side_effect=lambda name: False if name == "fastapi" else True):
            report = doctor.build_report(args)

        self.assertFalse(report["ok"])
        self.assertFalse(check_by_id(report, "dependency_fastapi")["ok"])

    def test_remote_demo_uses_security_preflight_without_secret_leak(self) -> None:
        tmp_root = Path(self._tmp_dir())
        registry = tmp_root / "miners.json"
        registry.write_text(
            json.dumps({"miners": [{"miner_id": "remote-a", "token": hash_token("remote-token"), "enabled": True}]}),
            encoding="utf-8",
        )
        args = doctor.parse_args([
            "--root",
            str(ROOT),
            "--state-dir",
            str(tmp_root / "state"),
            "--port",
            "0",
            "--remote-demo",
            "--host",
            "0.0.0.0",
            "--miner-token-registry",
            str(registry),
            "--observer-token",
            hash_token("observer"),
            "--admin-token",
            hash_token("admin"),
        ])

        report = doctor.build_report(args)

        self.assertTrue(report["ok"], report)
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("remote-token", serialized)
        self.assertIn("security_miner_token_registry", serialized)

    def test_cli_json_outputs_machine_readable_report(self) -> None:
        tmp_root = Path(self._tmp_dir())
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--root",
                str(ROOT),
                "--state-dir",
                str(tmp_root / "state"),
                "--port",
                "0",
                "--json",
            ],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["ok"], payload)

    def _tmp_dir(self) -> str:
        path = Path(self.id().replace(".", "_").replace("/", "_"))
        tmp_root = Path("/tmp") / f"crowdtensor_doctor_{path.name}"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(tmp_root, ignore_errors=True))
        return str(tmp_root)


if __name__ == "__main__":
    unittest.main()
