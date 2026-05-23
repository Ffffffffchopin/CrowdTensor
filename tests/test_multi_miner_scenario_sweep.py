from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "multi_miner_scenario_sweep.py"
SPEC = importlib.util.spec_from_file_location("multi_miner_scenario_sweep", SCRIPT_PATH)
multi_miner_scenario_sweep = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(multi_miner_scenario_sweep)


class MultiMinerScenarioSweepTests(unittest.TestCase):
    def _state(self) -> dict:
        tasks = []
        profiles = {}
        for index, scenario_id in enumerate(["route-baseline", "gradient-safety", "mixed-prompts"], start=1):
            miner_id = f"sweep-miner-{index}"
            task_id = f"task-{index}"
            profiles[miner_id] = {
                "runtime": "python-cli",
                "backend": "cpu",
                "accepted": 1,
                "rejected": 0,
                "last_capabilities": {"supported_workloads": ["model_bundle_infer"]},
            }
            tasks.append({
                "task_id": task_id,
                "status": "completed",
                "miner_id": miner_id,
                "workload_type": "model_bundle_infer",
                "validation": {
                    "code": "ok",
                    "request_count": 4,
                    "scenario_schema": "model_bundle_inference_scenario_v1",
                    "scenario_id": scenario_id,
                    "scenario_description": f"{scenario_id} scenario",
                    "scenario_request_count": 8,
                    "request_trace_count": 4,
                    "accuracy": 0.25,
                },
                "metrics": {
                    "request_count": 4,
                    "elapsed_ms": 1.5,
                    "requests_per_second": 2500.0,
                },
            })
        return {
            "accepted_results": 3,
            "rejected_results": 0,
            "task_counts": {"completed": 3, "queued": 0, "leased": 0, "rejected": 0},
            "model_updates": 0,
            "model": {
                "global_step": 0,
                "model_bundle": {"version": 0, "optimizer_step": 0},
            },
            "miner_profiles": profiles,
            "tasks": tasks,
        }

    def _state_with_rescue(self) -> dict:
        state = self._state()
        state["miner_profiles"]["sweep-miner-rescue"] = {
            "runtime": "python-cli",
            "backend": "cpu",
            "accepted": 1,
            "rejected": 0,
            "last_capabilities": {"supported_workloads": ["model_bundle_infer"]},
        }
        for task in state["tasks"]:
            if task["task_id"] == "task-1":
                task["miner_id"] = "sweep-miner-rescue"
                task["attempt"] = 2
        return state

    def _sessions(self) -> list[dict]:
        sessions = []
        for index, scenario_id in enumerate(["route-baseline", "gradient-safety", "mixed-prompts"], start=1):
            miner_id = f"sweep-miner-{index}"
            task_id = f"task-{index}"
            sessions.append({
                "miner_id": miner_id,
                "miner_summary": {"accepted_tasks": 1, "rejected_tasks": 0, "request_retries": 0},
                "session": {
                    "schema": "inference_session_request_v1",
                    "task_id": task_id,
                    "request_count": 4,
                    "scenario_id": scenario_id,
                    "workload_type": "model_bundle_infer",
                },
                "ledger": {
                    "results": [
                        {
                            "task_id": task_id,
                            "status": "completed",
                            "miner_id": miner_id,
                            "workload_type": "model_bundle_infer",
                            "model_updated": False,
                            "model_bundle_updated": False,
                            "validation": {
                                "code": "ok",
                                "request_count": 4,
                                "scenario_schema": "model_bundle_inference_scenario_v1",
                                "scenario_id": scenario_id,
                                "scenario_description": f"{scenario_id} scenario",
                                "scenario_request_count": 8,
                                "request_trace_count": 4,
                                "accuracy": 0.25,
                            },
                            "session_metrics": {
                                "request_count": 4,
                                "elapsed_ms": 1.5,
                                "requests_per_second": 2500.0,
                            },
                        },
                    ],
                },
            })
        return sessions

    def test_build_report_summarizes_multi_miner_sweep(self) -> None:
        report = multi_miner_scenario_sweep.build_report(
            base_url="http://127.0.0.1:8916",
            state=self._state(),
            sessions=self._sessions(),
            invites=[
                {"miner_id": "sweep-miner-1", "token_hash": "sha256:a"},
                {"miner_id": "sweep-miner-2", "token_hash": "sha256:b"},
                {"miner_id": "sweep-miner-3", "token_hash": "sha256:c"},
            ],
            request_count=4,
            generated_at="2026-05-23T00:00:00+00:00",
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["schema"], "multi_miner_scenario_sweep_v1")
        self.assertEqual(report["execution_mode"], "sequential")
        self.assertEqual(report["workload"]["scenario_ids"], ["route-baseline", "gradient-safety", "mixed-prompts"])
        self.assertEqual(report["distribution"]["distinct_miner_count"], 3)
        self.assertTrue(report["distribution"]["all_expected_miners_seen"])
        self.assertTrue(report["lease_summary"]["all_task_ids_unique"])
        self.assertTrue(report["lease_summary"]["one_result_per_task"])
        self.assertEqual(report["lease_summary"]["accepted_ledger_rows"], 3)
        self.assertTrue(report["process_summary"]["all_processes_ok"])
        self.assertEqual(report["observability_summary"]["schema"], "multi_miner_scenario_sweep_observability_v1")
        self.assertEqual(report["observability_summary"]["completed_sessions"], 3)
        self.assertIn("multi_miner_sweep_ready", report["diagnosis_codes"])
        self.assertTrue(report["safety"]["read_only"])
        self.assertTrue(report["safety"]["redaction_ok"])
        self.assertTrue(report["safety"]["registry_hashed"])
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("inference_result", encoded)
        self.assertNotIn("inference_results", encoded)
        self.assertNotIn("multi-miner-sweep-token", encoded)

    def test_build_report_rejects_scenario_mismatch(self) -> None:
        sessions = self._sessions()
        sessions[0]["ledger"]["results"][0]["validation"]["scenario_id"] = "mixed-prompts"
        report = multi_miner_scenario_sweep.build_report(
            base_url="http://127.0.0.1:8916",
            state=self._state(),
            sessions=sessions,
            invites=[
                {"miner_id": "sweep-miner-1", "token_hash": "sha256:a"},
                {"miner_id": "sweep-miner-2", "token_hash": "sha256:b"},
                {"miner_id": "sweep-miner-3", "token_hash": "sha256:c"},
            ],
            request_count=4,
            generated_at="2026-05-23T00:00:00+00:00",
        )

        self.assertFalse(report["ok"])
        self.assertIn("scenario_mismatch", report["diagnosis_codes"])
        self.assertFalse(report["sessions"][0]["scenario_matches"])

    def test_build_report_requires_all_expected_miners(self) -> None:
        sessions = self._sessions()
        sessions[2]["ledger"]["results"][0]["miner_id"] = "sweep-miner-2"
        report = multi_miner_scenario_sweep.build_report(
            base_url="http://127.0.0.1:8916",
            state=self._state(),
            sessions=sessions,
            invites=[
                {"miner_id": "sweep-miner-1", "token_hash": "sha256:a"},
                {"miner_id": "sweep-miner-2", "token_hash": "sha256:b"},
                {"miner_id": "sweep-miner-3", "token_hash": "sha256:c"},
            ],
            request_count=4,
            generated_at="2026-05-23T00:00:00+00:00",
        )

        self.assertFalse(report["ok"])
        self.assertIn("miner_distribution_failed", report["diagnosis_codes"])
        self.assertFalse(report["distribution"]["all_expected_miners_seen"])

    def test_build_report_summarizes_concurrent_lease_race(self) -> None:
        report = multi_miner_scenario_sweep.build_report(
            base_url="http://127.0.0.1:8916",
            state=self._state(),
            sessions=self._sessions(),
            invites=[
                {"miner_id": "sweep-miner-1", "token_hash": "sha256:a"},
                {"miner_id": "sweep-miner-2", "token_hash": "sha256:b"},
                {"miner_id": "sweep-miner-3", "token_hash": "sha256:c"},
            ],
            request_count=4,
            execution_mode="concurrent",
            process_summaries=[
                {
                    "miner_id": "sweep-miner-1",
                    "ok": True,
                    "returncode": 0,
                    "timed_out": False,
                    "miner_summary": {"accepted_tasks": 1, "rejected_tasks": 0, "request_retries": 0},
                },
                {
                    "miner_id": "sweep-miner-2",
                    "ok": True,
                    "returncode": 0,
                    "timed_out": False,
                    "miner_summary": {"accepted_tasks": 1, "rejected_tasks": 0, "request_retries": 0},
                },
                {
                    "miner_id": "sweep-miner-3",
                    "ok": True,
                    "returncode": 0,
                    "timed_out": False,
                    "miner_summary": {"accepted_tasks": 1, "rejected_tasks": 0, "request_retries": 0},
                },
            ],
            generated_at="2026-05-23T00:00:00+00:00",
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["execution_mode"], "concurrent")
        self.assertEqual(report["process_summary"]["started"], 3)
        self.assertEqual(report["process_summary"]["ok"], 3)
        self.assertTrue(report["process_summary"]["all_processes_ok"])
        self.assertTrue(report["lease_summary"]["all_task_ids_unique"])
        self.assertTrue(report["lease_summary"]["one_result_per_task"])
        self.assertTrue(report["lease_summary"]["no_queued_or_leased_remaining"])
        self.assertIn("multi_miner_concurrent_ready", report["diagnosis_codes"])

    def test_build_report_rejects_duplicate_or_missing_lease_results(self) -> None:
        sessions = self._sessions()
        sessions[1]["ledger"]["results"] = []
        report = multi_miner_scenario_sweep.build_report(
            base_url="http://127.0.0.1:8916",
            state=self._state(),
            sessions=sessions,
            invites=[
                {"miner_id": "sweep-miner-1", "token_hash": "sha256:a"},
                {"miner_id": "sweep-miner-2", "token_hash": "sha256:b"},
                {"miner_id": "sweep-miner-3", "token_hash": "sha256:c"},
            ],
            request_count=4,
            execution_mode="concurrent",
            process_summaries=[
                {"miner_id": "sweep-miner-1", "ok": True, "returncode": 0},
                {"miner_id": "sweep-miner-2", "ok": True, "returncode": 0},
                {"miner_id": "sweep-miner-3", "ok": True, "returncode": 0},
            ],
            generated_at="2026-05-23T00:00:00+00:00",
        )

        self.assertFalse(report["ok"])
        self.assertFalse(report["lease_summary"]["one_result_per_task"])
        self.assertIn("lease_race_failed", report["diagnosis_codes"])

    def test_build_report_summarizes_kill_after_claim_requeue(self) -> None:
        sessions = self._sessions()
        sessions[0]["miner_id"] = "sweep-miner-rescue"
        sessions[0]["ledger"]["results"][0]["miner_id"] = "sweep-miner-rescue"
        report = multi_miner_scenario_sweep.build_report(
            base_url="http://127.0.0.1:8916",
            state=self._state_with_rescue(),
            sessions=sessions,
            invites=[
                {"miner_id": "sweep-miner-1", "token_hash": "sha256:a"},
                {"miner_id": "sweep-miner-2", "token_hash": "sha256:b"},
                {"miner_id": "sweep-miner-3", "token_hash": "sha256:c"},
                {"miner_id": "sweep-miner-rescue", "token_hash": "sha256:r"},
            ],
            request_count=4,
            execution_mode="concurrent",
            failure_mode="kill-after-claim",
            process_summaries=[
                {
                    "miner_id": "sweep-miner-1",
                    "ok": False,
                    "returncode": -15,
                    "expected_failure": True,
                    "killed_after_claim": True,
                    "claimed_task_id": "task-1",
                },
                {"miner_id": "sweep-miner-rescue", "ok": True, "returncode": 0},
                {"miner_id": "sweep-miner-2", "ok": True, "returncode": 0},
                {"miner_id": "sweep-miner-3", "ok": True, "returncode": 0},
            ],
            requeue_summary={
                "enabled": True,
                "failure_mode": "kill-after-claim",
                "victim_miner_id": "sweep-miner-1",
                "rescue_miner_id": "sweep-miner-rescue",
                "requeued_task_id": "task-1",
                "lease_expired": True,
                "rescued_result": True,
                "victim_result_accepted": False,
            },
            generated_at="2026-05-23T00:00:00+00:00",
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["failure_mode"], "kill-after-claim")
        self.assertEqual(report["process_summary"]["expected_failures"], 1)
        self.assertTrue(report["process_summary"]["all_processes_ok"])
        self.assertTrue(report["requeue_summary"]["lease_expired"])
        self.assertTrue(report["requeue_summary"]["rescued_result"])
        self.assertFalse(report["requeue_summary"]["victim_result_accepted"])
        self.assertEqual(
            report["distribution"]["expected_miner_ids"],
            ["sweep-miner-2", "sweep-miner-3", "sweep-miner-rescue"],
        )
        self.assertIn("multi_miner_requeue_ready", report["diagnosis_codes"])

    def test_build_report_rejects_missing_requeue_observation(self) -> None:
        sessions = self._sessions()
        sessions[0]["miner_id"] = "sweep-miner-rescue"
        sessions[0]["ledger"]["results"][0]["miner_id"] = "sweep-miner-rescue"
        report = multi_miner_scenario_sweep.build_report(
            base_url="http://127.0.0.1:8916",
            state=self._state_with_rescue(),
            sessions=sessions,
            invites=[
                {"miner_id": "sweep-miner-1", "token_hash": "sha256:a"},
                {"miner_id": "sweep-miner-2", "token_hash": "sha256:b"},
                {"miner_id": "sweep-miner-3", "token_hash": "sha256:c"},
                {"miner_id": "sweep-miner-rescue", "token_hash": "sha256:r"},
            ],
            request_count=4,
            execution_mode="concurrent",
            failure_mode="kill-after-claim",
            process_summaries=[
                {"miner_id": "sweep-miner-1", "ok": False, "returncode": -15, "expected_failure": True},
                {"miner_id": "sweep-miner-rescue", "ok": True, "returncode": 0},
                {"miner_id": "sweep-miner-2", "ok": True, "returncode": 0},
                {"miner_id": "sweep-miner-3", "ok": True, "returncode": 0},
            ],
            requeue_summary={
                "enabled": True,
                "failure_mode": "kill-after-claim",
                "victim_miner_id": "sweep-miner-1",
                "rescue_miner_id": "sweep-miner-rescue",
                "requeued_task_id": "task-1",
                "lease_expired": False,
                "rescued_result": True,
                "victim_result_accepted": False,
            },
            generated_at="2026-05-23T00:00:00+00:00",
        )

        self.assertFalse(report["ok"])
        self.assertIn("requeue_not_observed", report["diagnosis_codes"])

    def test_parse_scenario_ids_rejects_unknown_and_singleton(self) -> None:
        self.assertEqual(
            multi_miner_scenario_sweep.parse_scenario_ids("route-baseline,gradient-safety"),
            ["route-baseline", "gradient-safety"],
        )
        with self.assertRaises(ValueError):
            multi_miner_scenario_sweep.parse_scenario_ids("route-baseline")
        with self.assertRaises(ValueError):
            multi_miner_scenario_sweep.parse_scenario_ids("unknown")


if __name__ == "__main__":
    unittest.main()
