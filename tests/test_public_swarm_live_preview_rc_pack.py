from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import public_swarm_live_preview_rc_check as check
from scripts import public_swarm_live_preview_rc_pack as pack


class PublicSwarmLivePreviewRCPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_live_preview_test_"))

    def test_check_builds_ready_local_smoke(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "local-smoke",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "public_swarm_live_preview_rc_check_v1")

    def test_check_builds_ready_package(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "package",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_check_builds_ready_live_kaggle_contract(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "live-kaggle",
            "--failure-mode",
            "kill-stage0-after-claim",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_check_builds_ready_evidence_import(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "evidence-import",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_live_kaggle_requires_kaggle_owner(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args([
                "live-kaggle",
                "--output-dir",
                str(self._tmp_dir()),
                "--kaggle-owner",
                "",
            ])

    def test_report_redacts_sensitive_fragments(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.persist_report(
            {
                "schema": pack.SCHEMA,
                "ok": True,
                "mode": "local-smoke",
                "diagnosis_codes": ["public_swarm_live_preview_rc_ready"],
                "live_preview": {"ready": True},
                "steps": [],
                "safety": {},
                "limitations": [],
                "secret": "CROWDTENSOR_ADMIN_TOKEN=admin-secret",
                "artifacts": pack.base_artifacts(output_dir, ok=True),
            },
            output_dir=output_dir,
        )

        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("CROWDTENSOR_ADMIN_TOKEN=admin-secret", encoded)
        self.assertTrue(report["ok"])


if __name__ == "__main__":
    unittest.main()
