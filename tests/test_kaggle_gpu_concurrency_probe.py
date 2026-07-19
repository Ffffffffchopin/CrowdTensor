from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import kaggle_gpu_concurrency_check as check
from scripts import kaggle_gpu_concurrency_probe as probe


class KaggleGpuConcurrencyProbeTests(unittest.TestCase):
    def _tmp_report(self, payload: dict) -> Path:
        path = Path(tempfile.mkdtemp(prefix="ct_gpu_conc_")) / "report.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _success_report(self) -> dict:
        return {
            "schema": probe.SCHEMA,
            "evidence_ready": True,
            "ok": True,
            "simultaneous_t4x2_verified": True,
            "requested_kernel_count": 2,
            "accepted_submission_count": 2,
            "accelerator": "NvidiaTeslaT4",
            "max_observed_running_count": 2,
            "worker_runtime_overlap_verified": True,
            "worker_reports": [
                {
                    "schema": "kaggle_gpu_concurrency_worker_report_v1",
                    "ok": True,
                    "public_artifact_safe": True,
                    "raw_gpu_names_public": False,
                    "cuda_available": True,
                    "cuda_device_count": 2,
                    "started_at": "2026-06-29T00:00:00+00:00",
                    "finished_at": "2026-06-29T00:04:00+00:00",
                },
                {
                    "schema": "kaggle_gpu_concurrency_worker_report_v1",
                    "ok": True,
                    "public_artifact_safe": True,
                    "raw_gpu_names_public": False,
                    "cuda_available": True,
                    "cuda_device_count": 2,
                    "started_at": "2026-06-29T00:01:00+00:00",
                    "finished_at": "2026-06-29T00:05:00+00:00",
                },
            ],
            "cleanup": {"attempted": True, "deleted_refs": ["a/b", "a/c"], "failed_delete_refs": []},
            "private_kernel_payloads_removed": True,
            "public_artifact_safe": True,
            "blockers": [],
            "diagnosis_codes": ["two_kaggle_t4x2_kernels_verified"],
            "safety": {
                "raw_gpu_names_public": False,
                "credentials_public": False,
                "cookies_public": False,
                "runtime_proxy_public": False,
                "activation_public": False,
                "hidden_state_public": False,
                "logits_public": False,
                "kv_cache_public": False,
            },
        }

    def test_worker_reports_overlap(self) -> None:
        self.assertTrue(probe.worker_reports_overlap(self._success_report()["worker_reports"]))
        self.assertFalse(
            probe.worker_reports_overlap(
                [
                    {"started_at": "2026-06-29T00:00:00+00:00", "finished_at": "2026-06-29T00:01:00+00:00"},
                    {"started_at": "2026-06-29T00:02:00+00:00", "finished_at": "2026-06-29T00:03:00+00:00"},
                ]
            )
        )

    def test_checker_accepts_success_report(self) -> None:
        self.assertEqual(check.validate_report(self._success_report()), [])

    def test_checker_accepts_blocker_report(self) -> None:
        report = self._success_report()
        report["ok"] = False
        report["simultaneous_t4x2_verified"] = False
        report["accepted_submission_count"] = 1
        report["max_observed_running_count"] = 1
        report["worker_reports"] = report["worker_reports"][:1]
        report["blockers"] = ["kaggle_gpu_quota_or_session_limit"]
        self.assertEqual(check.validate_report(report), [])

    def test_checker_rejects_success_without_overlap(self) -> None:
        report = self._success_report()
        report["max_observed_running_count"] = 1
        report["worker_runtime_overlap_verified"] = False
        errors = check.validate_report(report)
        self.assertIn("success_without_overlap_evidence", errors)

    def test_push_rejection_diagnosis_marks_session_limit(self) -> None:
        codes = probe.diagnose_push_rejection("Maximum GPU session limit reached")
        self.assertIn("kaggle_gpu_quota_or_session_limit", codes)


if __name__ == "__main__":
    unittest.main()
