from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import public_swarm_operator_preview_check as check
from scripts import public_swarm_operator_preview_pack as pack


class PublicSwarmOperatorPreviewPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_operator_preview_test_"))

    def test_check_builds_ready_local_smoke(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "local-smoke",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "public_swarm_operator_preview_check_v1")

    def test_check_builds_ready_package(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "package",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_check_builds_ready_live_kaggle(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "live-kaggle",
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

    def test_import_dual_live_evidence_requires_both_stage_reports(self) -> None:
        output_dir = self._tmp_dir()
        stage0 = output_dir / "stage0.json"
        stage1 = output_dir / "stage1.json"
        check.write_json(stage0, check.fake_live_payload(mode="live-kaggle", stage="stage0"))
        check.write_json(stage1, check.fake_live_payload(mode="live-kaggle", stage="stage1"))
        args = pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "preview"),
            "--live-stage0-report",
            str(stage0),
            "--live-stage1-report",
            str(stage1),
            "--release-readiness-report",
            str(output_dir / "missing-release.json"),
            "--developer-preview-report",
            str(output_dir / "missing-developer.json"),
        ])

        summary, payload = pack.import_dual_live_evidence(args)

        self.assertTrue(summary["ok"], summary)
        self.assertTrue(payload["live_preview"]["stage0_live_requeue_ready"])
        self.assertTrue(payload["live_preview"]["stage1_live_requeue_ready"])

    def test_live_kaggle_falls_back_to_retained_evidence_when_external_runtime_blocked(self) -> None:
        output_dir = self._tmp_dir()
        gpu_report = output_dir / "gpu.json"
        developer_report = output_dir / "developer.json"
        release_report = output_dir / "release.json"
        stage0 = output_dir / "stage0.json"
        stage1 = output_dir / "stage1.json"
        check.fake_gpu_report(gpu_report)
        check.write_json(developer_report, check.fake_developer_payload("local"))
        check.write_json(release_report, check.fake_release_payload())
        check.write_json(stage0, check.fake_live_payload(mode="live-kaggle", stage="stage0"))
        check.write_json(stage1, check.fake_live_payload(mode="live-kaggle", stage="stage1"))

        def blocked_live_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            if "public_swarm_live_preview_rc_pack.py" in joined and "live-kaggle" in command:
                return check.completed({
                    "schema": pack.LIVE_PREVIEW_SCHEMA,
                    "ok": False,
                    "mode": "live-kaggle",
                    "diagnosis_codes": ["hf_dependencies_missing"],
                }, returncode=1)
            return check.fake_runner(command)

        args = pack.parse_args([
            "live-kaggle",
            "--output-dir",
            str(output_dir / "preview"),
            "--gpu-report",
            str(gpu_report),
            "--developer-preview-report",
            str(developer_report),
            "--release-readiness-report",
            str(release_report),
            "--live-stage0-report",
            str(stage0),
            "--live-stage1-report",
            str(stage1),
            "--kaggle-owner",
            "owner",
            "--allow-dirty-release",
            "--json",
        ])

        report = pack.build_report(args, runner=blocked_live_runner)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["operator_preview"]["external_runtime_blocked"])
        self.assertIn("external_runtime_blocked", report["diagnosis_codes"])
        retained = report["payload_summaries"]["retained_live_evidence"]
        self.assertTrue(retained["ok"], retained)
        live_attempt = report["payload_summaries"]["live_attempt"]
        self.assertFalse(live_attempt["ok"], live_attempt)

    def test_local_smoke_degrades_to_cpu_fallback_when_hf_is_missing(self) -> None:
        output_dir = self._tmp_dir()
        gpu_report = output_dir / "gpu.json"
        release_report = output_dir / "release.json"
        check.fake_gpu_report(gpu_report)
        check.write_json(release_report, check.fake_release_payload())

        def degraded_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            if "public_swarm_developer_preview_pack.py" in joined:
                payload = check.fake_developer_payload("local")
                payload["ok"] = False
                payload["developer_preview"]["ready"] = False
                payload["developer_preview"]["product_beta_ready"] = False
                payload["developer_preview"]["support_bundle_ready"] = False
                payload["diagnosis_codes"] = [
                    code for code in payload["diagnosis_codes"] if code != "serve_join_generate_ready"
                ]
                payload["diagnosis_codes"].extend([
                    "developer_preview_blocked",
                    "hf_dependencies_missing",
                    "public_swarm_product_beta_blocked",
                ])
                return check.completed(payload, returncode=1)
            return check.fake_runner(command)

        args = pack.parse_args([
            "local-smoke",
            "--output-dir",
            str(output_dir / "preview"),
            "--gpu-report",
            str(gpu_report),
            "--release-readiness-report",
            str(release_report),
            "--allow-dirty-release",
            "--json",
        ])

        report = pack.build_report(args, runner=degraded_runner)

        self.assertTrue(report["ok"], report)
        preview = report["operator_preview"]
        self.assertFalse(preview["developer_preview_ready"])
        self.assertTrue(preview["developer_preview_degraded"])
        self.assertTrue(preview["cpu_fallback_ready"])
        self.assertTrue(preview["cpu_fallback_user_path_ready"])
        self.assertTrue(preview["user_path_ready"])
        self.assertFalse(preview["serve_join_generate_ready"])
        self.assertIn("operator_preview_cpu_fallback_user_path_ready", report["diagnosis_codes"])
        self.assertIn("developer_preview_degraded", report["diagnosis_codes"])
        self.assertIn("hf_dependencies_missing", report["diagnosis_codes"])

    def test_evidence_import_synthesizes_developer_summary_from_retained_live(self) -> None:
        output_dir = self._tmp_dir()
        gpu_report = output_dir / "gpu.json"
        release_report = output_dir / "release.json"
        missing_developer = output_dir / "missing-developer.json"
        stage0 = output_dir / "stage0.json"
        stage1 = output_dir / "stage1.json"
        check.fake_gpu_report(gpu_report)
        check.write_json(release_report, check.fake_release_payload())
        check.write_json(stage0, check.fake_live_payload(mode="live-kaggle", stage="stage0"))
        check.write_json(stage1, check.fake_live_payload(mode="live-kaggle", stage="stage1"))

        args = pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "preview"),
            "--gpu-report",
            str(gpu_report),
            "--developer-preview-report",
            str(missing_developer),
            "--live-stage0-report",
            str(stage0),
            "--live-stage1-report",
            str(stage1),
            "--release-readiness-report",
            str(release_report),
            "--allow-dirty-release",
            "--json",
        ])

        report = pack.build_report(args)

        self.assertTrue(report["ok"], report)
        preview = report["operator_preview"]
        self.assertTrue(preview["developer_preview_ready"])
        self.assertTrue(preview["retained_evidence_ready"])
        self.assertTrue(preview["retained_evidence_user_path_ready"])
        self.assertTrue(preview["user_path_ready"])
        self.assertIn("operator_preview_retained_evidence_ready", report["diagnosis_codes"])
        self.assertIn("developer_preview_retained_evidence_ready", report["diagnosis_codes"])

    def test_report_redacts_sensitive_fragments(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.persist_report(
            {
                "schema": pack.SCHEMA,
                "ok": True,
                "mode": "local-smoke",
                "diagnosis_codes": ["public_swarm_operator_preview_ready"],
                "operator_preview": {"ready": True},
                "steps": [],
                "payload_summaries": {},
                "safety": {},
                "limitations": [],
                "secret": "CROWDTENSOR_ADMIN_TOKEN=admin-secret",
                "artifacts": pack.base_artifacts(output_dir, ok=True),
                "gpu_report_path": str(output_dir / "gpu.json"),
            },
            output_dir=output_dir,
        )

        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("CROWDTENSOR_ADMIN_TOKEN=admin-secret", encoded)
        self.assertTrue(report["ok"])


if __name__ == "__main__":
    unittest.main()
