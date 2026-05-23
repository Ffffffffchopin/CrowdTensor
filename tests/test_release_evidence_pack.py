from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "release_evidence_pack.py"
SPEC = importlib.util.spec_from_file_location("release_evidence_pack", SCRIPT_PATH)
release_evidence_pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(release_evidence_pack)


def passing_runtime_report(path: Path) -> None:
    path.write_text(
        json.dumps({
            "ok": True,
            "started_at": "2026-05-21T00:00:00+00:00",
            "finished_at": "2026-05-21T00:00:01+00:00",
            "duration_seconds": 1.0,
            "checks": [
                {"name": "readiness", "ok": True},
                {"name": "api_contract", "ok": True},
            ],
        }),
        encoding="utf-8",
    )


def skipped_browser_report(path: Path) -> None:
    path.write_text(
        json.dumps({
            "ok": True,
            "skipped": True,
            "skip_reason": "Chromium-compatible browser was not found",
            "duration_seconds": 0.0,
            "checks": [],
        }),
        encoding="utf-8",
    )


def diagnostic_runtime_report(path: Path) -> None:
    path.write_text(
        json.dumps({
            "ok": True,
            "duration_seconds": 1.0,
            "diagnosis_summary": {
                "codes": ["home_compute_ready"],
                "by_check": {"home_compute_demo": ["home_compute_ready"]},
                "failed_checks": [],
            },
            "checks": [
                {"name": "home_compute_demo", "ok": True, "diagnosis_codes": ["home_compute_ready"]},
            ],
        }),
        encoding="utf-8",
    )


def base_args(tmp_root: Path, runtime_report: Path) -> object:
    return release_evidence_pack.parse_args([
        "--root",
        str(ROOT),
        "--runtime-report",
        str(runtime_report),
        "--json-out",
        str(tmp_root / "release-evidence.json"),
        "--allow-dirty",
    ])


class ReleaseEvidencePackTests(unittest.TestCase):
    def test_passing_runtime_report_is_ready(self) -> None:
        tmp_root = Path(self._tmp_dir())
        runtime_report = tmp_root / "runtime.json"
        browser_report = tmp_root / "browser.json"
        passing_runtime_report(runtime_report)
        skipped_browser_report(browser_report)
        args = release_evidence_pack.parse_args([
            "--root",
            str(ROOT),
            "--runtime-report",
            str(runtime_report),
            "--browser-report",
            str(browser_report),
            "--json-out",
            str(tmp_root / "release-evidence.json"),
            "--allow-dirty",
        ])

        evidence = release_evidence_pack.build_evidence(
            args,
            git_info={"available": True, "commit": "abc", "branch": "main", "dirty": False, "status": []},
        )

        self.assertTrue(evidence["release_status"]["ready"], evidence)
        self.assertEqual(evidence["reports"]["runtime"]["checks_total"], 2)
        self.assertTrue(evidence["reports"]["browser"]["skipped"])

    def test_runtime_report_preserves_diagnosis_summary(self) -> None:
        tmp_root = Path(self._tmp_dir())
        runtime_report = tmp_root / "runtime.json"
        diagnostic_runtime_report(runtime_report)

        report = release_evidence_pack.load_acceptance_report(
            str(runtime_report),
            name="runtime",
            required=True,
        )

        self.assertEqual(report["diagnosis_codes"], ["home_compute_ready"])
        self.assertEqual(report["diagnosis_by_check"], {"home_compute_demo": ["home_compute_ready"]})
        self.assertEqual(report["failed_checks"], [])

    def test_runtime_report_preserves_direct_diagnosis_rows(self) -> None:
        tmp_root = Path(self._tmp_dir())
        runtime_report = tmp_root / "runtime.json"
        runtime_report.write_text(
            json.dumps({
                "ok": True,
                "diagnosis_codes": ["home_compute_ready"],
                "diagnosis_by_check": {"home_compute_demo": ["home_compute_ready"]},
                "failed_checks": [],
                "checks": [],
            }),
            encoding="utf-8",
        )

        report = release_evidence_pack.load_acceptance_report(
            str(runtime_report),
            name="runtime",
            required=True,
        )

        self.assertEqual(report["diagnosis_codes"], ["home_compute_ready"])
        self.assertEqual(report["diagnosis_by_check"], {"home_compute_demo": ["home_compute_ready"]})

    def test_remote_report_preserves_safe_observability_summary(self) -> None:
        tmp_root = Path(self._tmp_dir())
        remote_report = tmp_root / "remote.json"
        remote_report.write_text(
            json.dumps({
                "ok": True,
                "diagnosis_codes": ["acceptance_ready"],
                "observability_summary": {
                    "schema": "remote_demo_observability_v1",
                    "route": "remote_python_model_bundle_infer",
                    "miner_id": "remote-a",
                    "availability": {
                        "health_ok": True,
                        "state_ok": True,
                        "admin_results_ok": True,
                        "elapsed_seconds": 1.5,
                    },
                    "work_queue": {
                        "accepted_results": 1,
                        "task_id": "task-1",
                    },
                    "miner": {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "accepted": 1,
                        "rejected": 0,
                        "supported_workloads": ["model_bundle_infer"],
                    },
                    "inference": {
                        "request_count": 4,
                        "request_trace_count": 4,
                        "requests_per_second": 1200.0,
                    },
                    "artifacts": {
                        "evidence_ok": True,
                        "support_bundle_ok": True,
                        "evidence_path": "/tmp/secret-local-path/evidence.json",
                        "support_bundle_path": "/tmp/secret-local-path/support.json",
                        "evidence_observability_schema": "remote_compute_observability_v1",
                    },
                    "diagnosis_codes": ["acceptance_ready"],
                    "token": "secret-token",
                },
            }),
            encoding="utf-8",
        )

        report = release_evidence_pack.load_acceptance_report(
            str(remote_report),
            name="remote",
            required=False,
        )

        observed = report["observability_summaries"][0]
        self.assertEqual(observed["schema"], "remote_demo_observability_v1")
        self.assertEqual(observed["route"], "remote_python_model_bundle_infer")
        self.assertEqual(observed["inference"]["request_count"], 4)
        self.assertEqual(observed["inference"]["requests_per_second"], 1200.0)
        self.assertEqual(observed["artifacts"]["evidence_observability_schema"], "remote_compute_observability_v1")
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("secret-token", encoded)
        self.assertNotIn("secret-local-path", encoded)

    def test_markdown_renders_observability_summaries(self) -> None:
        payload = {
            "release_status": {"status": "ready", "blocking_reasons": []},
            "git": {"commit": "abc", "branch": "main"},
            "project": {"name": "crowdtensord", "version": "0.1.0a0"},
            "generated_at": "2026-05-22T00:00:00+00:00",
            "checks": {"release_gate": {"ok": True, "total": 1, "failed": []}},
            "reports": {
                "remote": {
                    "present": True,
                    "ok": True,
                    "checks_total": 0,
                    "observability_summaries": [
                        {
                            "schema": "remote_compute_observability_v1",
                            "route": {"name": "remote_python_model_bundle_infer"},
                            "inference": {"request_count": 4, "requests_per_second": 1200.0},
                        },
                    ],
                },
            },
        }

        markdown = release_evidence_pack.render_markdown(payload)

        self.assertIn("remote_compute_observability_v1", markdown)
        self.assertIn("requests=4", markdown)

    def test_nested_release_report_diagnosis_rows_are_aggregated(self) -> None:
        summary = release_evidence_pack.summarize_diagnosis({
            "reports": {
                "runtime": {
                    "diagnosis_codes": ["home_compute_ready"],
                    "diagnosis_by_check": {"home_compute_demo": ["home_compute_ready"]},
                    "failed_checks": [],
                },
            },
        })

        self.assertEqual(summary["codes"], ["home_compute_ready"])
        self.assertEqual(summary["by_check"], {"runtime.home_compute_demo": ["home_compute_ready"]})

    def test_missing_runtime_report_blocks_release(self) -> None:
        tmp_root = Path(self._tmp_dir())
        args = base_args(tmp_root, tmp_root / "missing.json")

        evidence = release_evidence_pack.build_evidence(
            args,
            git_info={"available": True, "commit": "abc", "branch": "main", "dirty": False, "status": []},
        )

        self.assertFalse(evidence["release_status"]["ready"])
        self.assertTrue(any("runtime acceptance report is missing" in reason for reason in evidence["release_status"]["blocking_reasons"]))

    def test_failed_runtime_report_blocks_release(self) -> None:
        tmp_root = Path(self._tmp_dir())
        runtime_report = tmp_root / "runtime.json"
        runtime_report.write_text(json.dumps({"ok": False, "checks": [{"name": "api_contract", "ok": False}]}), encoding="utf-8")
        args = base_args(tmp_root, runtime_report)

        evidence = release_evidence_pack.build_evidence(
            args,
            git_info={"available": True, "commit": "abc", "branch": "main", "dirty": False, "status": []},
        )

        self.assertFalse(evidence["release_status"]["ready"])
        self.assertIn("api_contract", evidence["reports"]["runtime"]["checks_failed"])

    def test_strict_optional_requires_browser_and_remote_reports(self) -> None:
        tmp_root = Path(self._tmp_dir())
        runtime_report = tmp_root / "runtime.json"
        passing_runtime_report(runtime_report)
        args = release_evidence_pack.parse_args([
            "--root",
            str(ROOT),
            "--runtime-report",
            str(runtime_report),
            "--strict-optional",
            "--json-out",
            str(tmp_root / "release-evidence.json"),
            "--allow-dirty",
        ])

        evidence = release_evidence_pack.build_evidence(
            args,
            git_info={"available": True, "commit": "abc", "branch": "main", "dirty": False, "status": []},
        )

        self.assertFalse(evidence["release_status"]["ready"])
        self.assertTrue(any("browser acceptance report is missing" in reason for reason in evidence["release_status"]["blocking_reasons"]))
        self.assertTrue(any("remote acceptance report is missing" in reason for reason in evidence["release_status"]["blocking_reasons"]))

    def test_allow_missing_optional_overrides_strict_optional(self) -> None:
        tmp_root = Path(self._tmp_dir())
        runtime_report = tmp_root / "runtime.json"
        passing_runtime_report(runtime_report)
        args = release_evidence_pack.parse_args([
            "--root",
            str(ROOT),
            "--runtime-report",
            str(runtime_report),
            "--strict-optional",
            "--allow-missing-optional",
            "--json-out",
            str(tmp_root / "release-evidence.json"),
            "--allow-dirty",
        ])

        evidence = release_evidence_pack.build_evidence(
            args,
            git_info={"available": True, "commit": "abc", "branch": "main", "dirty": False, "status": []},
        )

        self.assertTrue(evidence["release_status"]["ready"], evidence)
        self.assertFalse(evidence["release_status"]["strict_optional"])
        self.assertTrue(evidence["release_status"]["allow_missing_optional"])

    def test_dirty_worktree_blocks_unless_allowed(self) -> None:
        tmp_root = Path(self._tmp_dir())
        runtime_report = tmp_root / "runtime.json"
        passing_runtime_report(runtime_report)
        args = release_evidence_pack.parse_args([
            "--root",
            str(ROOT),
            "--runtime-report",
            str(runtime_report),
            "--json-out",
            str(tmp_root / "release-evidence.json"),
        ])
        dirty_git = {"available": True, "commit": "abc", "branch": "main", "dirty": True, "status": [" M README.md"]}

        blocked = release_evidence_pack.build_evidence(args, git_info=dirty_git)
        args.allow_dirty = True
        allowed = release_evidence_pack.build_evidence(args, git_info=dirty_git)

        self.assertFalse(blocked["release_status"]["ready"])
        self.assertTrue(allowed["release_status"]["ready"], allowed)

    def test_markdown_mentions_blocking_reasons(self) -> None:
        payload = {
            "release_status": {"status": "blocked", "blocking_reasons": ["runtime acceptance report is missing"]},
            "project": {"name": "crowdtensord", "version": "0.1.0a0"},
            "git": {"commit": "abc", "branch": "main"},
            "generated_at": "2026-05-21T00:00:00+00:00",
            "checks": {"release_gate": {"ok": True, "total": 1, "failed": []}},
            "reports": {"runtime": {"present": False, "ok": False, "checks_total": 0}},
        }

        markdown = release_evidence_pack.render_markdown(payload)

        self.assertIn("runtime acceptance report is missing", markdown)

    def test_markdown_mentions_diagnosis_codes(self) -> None:
        payload = {
            "release_status": {"status": "ready", "blocking_reasons": []},
            "project": {"name": "crowdtensord", "version": "0.1.0a0"},
            "git": {"commit": "abc", "branch": "main"},
            "generated_at": "2026-05-21T00:00:00+00:00",
            "checks": {"release_gate": {"ok": True, "total": 1, "failed": []}},
            "reports": {
                "runtime": {
                    "present": True,
                    "ok": True,
                    "checks_total": 1,
                    "diagnosis_codes": ["home_compute_ready"],
                },
            },
        }

        markdown = release_evidence_pack.render_markdown(payload)

        self.assertIn("home_compute_ready", markdown)

    def _tmp_dir(self) -> str:
        path = Path(self.id().replace(".", "_").replace("/", "_"))
        tmp_root = Path("/tmp") / f"crowdtensor_release_evidence_{path.name}"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(tmp_root, ignore_errors=True))
        return str(tmp_root)


if __name__ == "__main__":
    unittest.main()
