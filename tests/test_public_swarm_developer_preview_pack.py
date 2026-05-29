from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import public_swarm_developer_preview_check as check
from scripts import public_swarm_developer_preview_pack as pack


class PublicSwarmDeveloperPreviewPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_preview_test_"))

    def test_check_builds_ready_local_preview(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "local",
            "--output-dir",
            str(self._tmp_dir()),
            "--max-new-tokens",
            "2",
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "public_swarm_developer_preview_check_v1")

    def test_check_builds_ready_kaggle_package_preview(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "package",
            "--target",
            "kaggle",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_check_builds_ready_external_existing_preview(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "external-existing",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_check_builds_ready_evidence_import_preview(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "evidence-import",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_external_existing_requires_auth(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args([
                "external-existing",
                "--coordinator-url",
                "http://127.0.0.1:9999",
            ])

    def test_report_redacts_secret_fragments(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.persist_report(
            {
                "schema": pack.SCHEMA,
                "ok": True,
                "mode": "package",
                "diagnosis_codes": ["developer_preview_ready"],
                "developer_preview": {"ready": True},
                "steps": [],
                "limitations": [],
                "safety": {},
                "secret": "admin-secret",
            },
            output_dir=output_dir,
            secret_values=["admin-secret"],
        )

        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("admin-secret", encoded)


if __name__ == "__main__":
    unittest.main()
