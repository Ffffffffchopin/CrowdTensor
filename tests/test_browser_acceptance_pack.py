from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "browser_acceptance_pack.py"
SPEC = importlib.util.spec_from_file_location("browser_acceptance_pack", SCRIPT_PATH)
browser_acceptance_pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(browser_acceptance_pack)


def browser_args(**overrides):
    values = {
        "host": "127.0.0.1",
        "base_port": 9310,
        "state_root": "",
        "browser": "",
        "headful": False,
        "timeout": 20.0,
        "check_timeout": 120.0,
        "report": "",
        "allow_skip": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class BrowserAcceptancePackTests(unittest.TestCase):
    def test_dependency_skip_reason_prefers_playwright_then_browser(self) -> None:
        with patch.object(browser_acceptance_pack, "playwright_available", return_value=False):
            self.assertEqual(
                browser_acceptance_pack.dependency_skip_reason("/usr/bin/chromium"),
                "Playwright Python is not installed",
            )
        with patch.object(browser_acceptance_pack, "playwright_available", return_value=True):
            self.assertEqual(
                browser_acceptance_pack.dependency_skip_reason(""),
                "Chromium-compatible browser was not found",
            )
            self.assertEqual(browser_acceptance_pack.dependency_skip_reason("/usr/bin/chromium"), "")

    def test_browser_checks_assign_core_three_ports_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checks = browser_acceptance_pack.browser_checks(
                browser_args(browser="/usr/bin/chromium", headful=True, timeout=7.5),
                Path(tmp),
                "/usr/bin/chromium",
            )

        self.assertEqual([check["name"] for check in checks], ["webrtc", "runtime_contract", "browser_miner"])
        self.assertEqual([check["port"] for check in checks], [9310, 9311, 9313])
        self.assertEqual([check.get("web_port") for check in checks], [None, 9312, 9314])
        self.assertIn("webrtc_smoke.py", checks[0]["command"][1])
        self.assertIn("runtime_contract_check.py", checks[1]["command"][1])
        self.assertIn("browser_miner_smoke.py", checks[2]["command"][1])
        for check in checks:
            self.assertIn("--browser", check["command"])
            self.assertIn("/usr/bin/chromium", check["command"])
            self.assertIn("--headful", check["command"])

    def test_build_report_marks_skipped_as_ok(self) -> None:
        report = browser_acceptance_pack.build_report(
            started_at="2026-05-21T00:00:00+00:00",
            finished_at="2026-05-21T00:00:01+00:00",
            duration_seconds=1.2345,
            checks=[],
            browser="",
            skipped=True,
            skip_reason="Chromium-compatible browser was not found",
        )

        self.assertTrue(report["ok"])
        self.assertTrue(report["skipped"])
        self.assertEqual(report["duration_seconds"], 1.234)

    def test_build_report_aggregates_check_results(self) -> None:
        passing = browser_acceptance_pack.build_report(
            started_at="2026-05-21T00:00:00+00:00",
            finished_at="2026-05-21T00:00:01+00:00",
            duration_seconds=1.0,
            checks=[{"name": "webrtc", "ok": True}, {"name": "browser_miner", "ok": True}],
            browser="/usr/bin/chromium",
        )
        failing = browser_acceptance_pack.build_report(
            started_at="2026-05-21T00:00:00+00:00",
            finished_at="2026-05-21T00:00:01+00:00",
            duration_seconds=1.0,
            checks=[{"name": "webrtc", "ok": True}, {"name": "browser_miner", "ok": False}],
            browser="/usr/bin/chromium",
        )

        self.assertTrue(passing["ok"])
        self.assertFalse(failing["ok"])

    def test_write_report_outputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "browser-report.json"
            browser_acceptance_pack.write_report({"ok": True, "checks": []}, str(path))
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(payload["ok"])

    def test_run_check_captures_failure_tail(self) -> None:
        check = {
            "name": "failing",
            "port": 9999,
            "state_dir": "/tmp/crowdtensor_browser_acceptance_unit",
            "command": [
                "python3",
                "-c",
                "import sys; print('out'); print('err', file=sys.stderr); raise SystemExit(3)",
            ],
        }

        result = browser_acceptance_pack.run_check(check, timeout_seconds=5)

        self.assertFalse(result["ok"])
        self.assertEqual(result["returncode"], 3)
        self.assertIn("out", result["stdout"])
        self.assertIn("err", result["stderr"])


if __name__ == "__main__":
    unittest.main()
