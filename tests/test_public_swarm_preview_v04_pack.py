from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import public_swarm_preview_v04_check as check
from scripts import public_swarm_preview_v04_pack as pack


class PublicSwarmPreviewV04PackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_preview_v04_test_"))

    def test_check_builds_ready_evidence_import_with_optional_model(self) -> None:
        output_dir = self._tmp_dir()
        result = check.run_check(check.parse_args([
            "--mode",
            "evidence-import",
            "--require-optional-model-ready",
            "--output-dir",
            str(output_dir),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "public_swarm_preview_v04_check_v1")
        preview = json.loads((output_dir / "preview-v04" / "public_swarm_preview_v04.json").read_text(encoding="utf-8"))
        artifacts = preview["artifacts"]
        self.assertTrue(artifacts["product_swarm_mvp_source_json"]["present"])
        self.assertTrue(artifacts["optional_model_mvp_source_json"]["present"])

    def test_check_builds_ready_package(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "package",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_missing_optional_model_blocks_when_required(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args([
                "evidence-import",
                "--require-optional-model-ready",
            ])

    def test_report_redacts_sensitive_fragments(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.persist_report(
            {
                "schema": pack.SCHEMA,
                "ok": True,
                "mode": pack.MODE_EVIDENCE_IMPORT,
                "preview": {"ready": True},
                "performance": {},
                "steps": [],
                "payload_summaries": {},
                "diagnosis_codes": ["public_swarm_preview_v04_ready"],
                "artifacts": pack.base_artifacts(output_dir, ok=True),
                "safety": {},
                "limitations": [],
                "secret": "CROWDTENSOR_ADMIN_TOKEN=admin-secret",
            },
            output_dir=output_dir,
        )

        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("CROWDTENSOR_ADMIN_TOKEN=admin-secret", encoded)
        self.assertTrue(report["ok"])
        self.assertNotIn("sensitive_output_detected", report["diagnosis_codes"])


if __name__ == "__main__":
    unittest.main()
