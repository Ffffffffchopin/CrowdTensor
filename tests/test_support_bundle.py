from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "support_bundle.py"
SPEC = importlib.util.spec_from_file_location("support_bundle", SCRIPT_PATH)
support_bundle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(support_bundle)


class FakeHeaders:
    def __init__(self, content_type: str = "application/json") -> None:
        self.content_type = content_type

    def get(self, key: str, default: str = "") -> str:
        if key.lower() == "content-type":
            return self.content_type
        return default


class FakeResponse:
    def __init__(self, body: str, *, status: int = 200, content_type: str = "application/json") -> None:
        self.status = status
        self.body = body.encode("utf-8")
        self.headers = FakeHeaders(content_type)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def fake_urlopen(request, timeout=5.0):  # noqa: ANN001
    url = request.full_url
    if url.endswith("/health"):
        return FakeResponse(json.dumps({"ok": True, "service": "crowdtensord"}))
    if url.endswith("/version"):
        return FakeResponse(json.dumps({"version": "0.1.0a0"}))
    if url.endswith("/ready"):
        return FakeResponse(json.dumps({"ok": True, "task_counts": {"queued": 1}}))
    if url.endswith("/metrics"):
        return FakeResponse("crowdtensord_event_index 1\n", content_type="text/plain")
    if url.endswith("/state"):
        return FakeResponse(json.dumps({
            "event_index": 3,
            "task_counts": {"queued": 1},
            "accepted_results": 1,
            "rejected_results": 0,
            "model": {
                "global_step": 1,
                "optimizer_step": 1,
                "weights": [1, 2, 3],
            },
            "tasks": [{"lease_token": "secret-lease"}],
            "miner_profiles": {"miner": {"last_capabilities": {}}},
            "miner_workload_scores": {"miner": {}},
        }))
    if "/admin/results" in url:
        return FakeResponse(json.dumps({
            "results": [
                {
                    "task_id": "task-1",
                    "status": "accepted",
                    "lease_token": "secret",
                    "local_delta": [1.0],
                },
            ],
        }))
    raise AssertionError(f"unexpected URL {url}")


class SupportBundleTests(unittest.TestCase):
    def test_offline_bundle_contains_core_sections(self) -> None:
        tmp_root = Path(self._tmp_dir())
        args = support_bundle.parse_args([
            "--root",
            str(ROOT),
            "--state-dir",
            str(tmp_root / "state"),
            "--port",
            "0",
        ])

        bundle = support_bundle.build_bundle(args)

        self.assertIn("doctor", bundle)
        self.assertIn("release_gate", bundle)
        self.assertIn("git", bundle)
        self.assertFalse(bundle["online"]["enabled"])
        self.assertTrue(bundle["release_gate"]["ok"], bundle)

    def test_sanitize_redacts_sensitive_fields(self) -> None:
        payload = {
            "token": "secret",
            "nested": {
                "lease_token": "lease",
                "token_rotation_required": True,
                "weights": [1, 2],
                "safe": "value",
                "local_delta": [0.1],
                "idempotency_key": "idem",
            },
        }

        sanitized = support_bundle.sanitize(payload)

        self.assertEqual(sanitized["token"], "<redacted>")
        self.assertEqual(sanitized["nested"]["lease_token"], "<redacted>")
        self.assertIs(sanitized["nested"]["token_rotation_required"], True)
        self.assertEqual(sanitized["nested"]["weights"], "<redacted>")
        self.assertEqual(sanitized["nested"]["local_delta"], "<redacted>")
        self.assertEqual(sanitized["nested"]["safe"], "value")

    def test_broken_json_report_fails_cli(self) -> None:
        tmp_root = Path(self._tmp_dir())
        bad_report = tmp_root / "bad.json"
        bad_report.write_text("{bad json", encoding="utf-8")

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--root",
                str(ROOT),
                "--state-dir",
                str(tmp_root / "state"),
                "--runtime-report",
                str(bad_report),
            ],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("not valid JSON", completed.stderr)

    def test_online_collection_summarizes_and_redacts_state(self) -> None:
        tmp_root = Path(self._tmp_dir())
        args = support_bundle.parse_args([
            "--root",
            str(ROOT),
            "--state-dir",
            str(tmp_root / "state"),
            "--port",
            "0",
            "--coordinator",
            "http://127.0.0.1:8787",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
        ])

        with patch.object(support_bundle, "urlopen", side_effect=fake_urlopen):
            bundle = support_bundle.build_bundle(args)

        online = bundle["online"]
        self.assertTrue(online["enabled"])
        self.assertEqual(online["metrics"]["line_count"], 1)
        self.assertEqual(online["state"]["event_index"], 3)
        serialized = json.dumps(bundle, sort_keys=True)
        self.assertNotIn("secret-lease", serialized)
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertIn('"local_delta": "<redacted>"', serialized)

    def test_load_json_report_preserves_diagnosis_summary(self) -> None:
        tmp_root = Path(self._tmp_dir())
        report_path = tmp_root / "runtime.json"
        report_path.write_text(
            json.dumps({
                "ok": True,
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

        report = support_bundle.load_json_report(str(report_path), name="runtime")

        self.assertEqual(report["diagnosis_codes"], ["home_compute_ready"])
        self.assertEqual(report["diagnosis_by_check"], {"home_compute_demo": ["home_compute_ready"]})
        self.assertEqual(report["failed_checks"], [])

    def test_load_json_report_understands_release_evidence_status_and_diagnosis(self) -> None:
        tmp_root = Path(self._tmp_dir())
        report_path = tmp_root / "release-evidence.json"
        report_path.write_text(
            json.dumps({
                "release_status": {"ready": True, "status": "ready", "blocking_reasons": []},
                "reports": {
                    "runtime": {
                        "diagnosis_codes": ["home_compute_ready"],
                        "diagnosis_by_check": {"home_compute_demo": ["home_compute_ready"]},
                        "failed_checks": [],
                    },
                },
                "checks": {},
            }),
            encoding="utf-8",
        )

        report = support_bundle.load_json_report(str(report_path), name="release_evidence")

        self.assertTrue(report["ok"])
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["diagnosis_codes"], ["home_compute_ready"])
        self.assertEqual(report["diagnosis_by_check"], {"runtime.home_compute_demo": ["home_compute_ready"]})

    def test_load_json_report_preserves_safe_observability_summary(self) -> None:
        tmp_root = Path(self._tmp_dir())
        report_path = tmp_root / "remote.json"
        report_path.write_text(
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

        report = support_bundle.load_json_report(str(report_path), name="remote")

        observed = report["observability_summaries"][0]
        self.assertEqual(observed["schema"], "remote_demo_observability_v1")
        self.assertEqual(observed["route"], "remote_python_model_bundle_infer")
        self.assertTrue(observed["availability"]["health_ok"])
        self.assertEqual(observed["inference"]["request_count"], 4)
        self.assertEqual(observed["inference"]["requests_per_second"], 1200.0)
        self.assertEqual(observed["artifacts"]["evidence_observability_schema"], "remote_compute_observability_v1")
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("secret-token", encoded)
        self.assertNotIn("secret-local-path", encoded)

    def test_markdown_output_summarizes_reports(self) -> None:
        payload = {
            "generated_at": "2026-05-21T00:00:00+00:00",
            "project": {"name": "crowdtensord", "version": "0.1.0a0"},
            "git": {"commit": "abc"},
            "doctor": {"ok": True},
            "release_gate": {"ok": True},
            "online": {"enabled": False},
            "reports": {
                "runtime": {
                    "present": True,
                    "ok": True,
                    "checks_total": 2,
                    "diagnosis_codes": ["home_compute_ready"],
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

        markdown = support_bundle.render_markdown(payload)

        self.assertIn("CrowdTensorD Support Bundle", markdown)
        self.assertIn("runtime", markdown)
        self.assertIn("home_compute_ready", markdown)
        self.assertIn("remote_compute_observability_v1", markdown)
        self.assertIn("requests=4", markdown)

    def _tmp_dir(self) -> str:
        path = Path(self.id().replace(".", "_").replace("/", "_"))
        tmp_root = Path("/tmp") / f"crowdtensor_support_bundle_{path.name}"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(tmp_root, ignore_errors=True))
        return str(tmp_root)


if __name__ == "__main__":
    unittest.main()
