from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "remote_demo_acceptance_pack.py"
SPEC = importlib.util.spec_from_file_location("remote_demo_acceptance_pack", SCRIPT_PATH)
remote_demo_acceptance_pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(remote_demo_acceptance_pack)


class RemoteDemoAcceptancePackTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            coordinator_url="https://coordinator.example",
            miner_id="remote-a",
            observer_token="observer-secret",
            admin_token="admin-secret",
            request_count=4,
            timeout_seconds=120.0,
            poll_interval=2.0,
            http_timeout=5.0,
            artifact_timeout=60.0,
            admin_results_limit=10,
            scenario_id="route-baseline",
            create_session=False,
            session_task_id="",
            output_dir="dist/remote-demo-acceptance",
            json_out="",
            markdown_out="",
        )

    def _state(
        self,
        *,
        miner_id: str = "remote-a",
        workloads: list[str] | None = None,
        validation: dict | None = None,
        include_task: bool = True,
        include_lane: bool = True,
    ) -> dict:
        validation = validation or {
            "code": "ok",
            "request_count": 4,
            "scenario_schema": "model_bundle_inference_scenario_v1",
            "scenario_id": "route-baseline",
            "scenario_description": "Fixed CPU read-only route prompts from the built-in bundle corpus.",
            "scenario_request_count": 8,
            "request_trace_count": 4,
            "accuracy": 0.25,
        }
        state = {
            "accepted_results": 1,
            "task_counts": {"completed": 1, "queued": 1, "leased": 0, "rejected": 0},
            "miner_profiles": {
                miner_id: {
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "accepted": 1,
                    "rejected": 0,
                    "last_capabilities": {
                        "supported_workloads": ["model_bundle_infer"] if workloads is None else workloads,
                    },
                },
            },
            "tasks": [],
        }
        if include_lane:
            state["task_lanes"] = [
                {
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "protocol_version": "runtime_contract_v1",
                    "workload_type": "model_bundle_infer",
                    "count": 1,
                }
            ]
        if include_task:
            state["tasks"].append(
                {
                    "task_id": "task-1",
                    "status": "completed",
                    "miner_id": miner_id,
                    "workload_type": "model_bundle_infer",
                    "validation": validation,
                },
            )
        return state

    def _results(
        self,
        *,
        miner_id: str = "remote-a",
        task_id: str = "task-1",
        validation: dict | None = None,
        include_result: bool = True,
    ) -> dict:
        if not include_result:
            return {"results": []}
        validation = validation or {
            "code": "ok",
            "request_count": 4,
            "scenario_schema": "model_bundle_inference_scenario_v1",
            "scenario_id": "route-baseline",
            "scenario_description": "Fixed CPU read-only route prompts from the built-in bundle corpus.",
            "scenario_request_count": 8,
            "request_trace_count": 4,
            "accuracy": 0.25,
        }
        return {
            "results": [
                {
                    "task_id": task_id,
                    "status": "completed",
                    "miner_id": miner_id,
                    "workload_type": "model_bundle_infer",
                    "validation": validation,
                },
            ],
        }

    def _ready(self, *, include_lane: bool = True) -> dict:
        return {
            "ok": True,
            "task_counts": {"completed": 1, "queued": 1, "leased": 0, "rejected": 0},
            "task_lanes": [
                {
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "protocol_version": "runtime_contract_v1",
                    "workload_type": "model_bundle_infer",
                    "count": 1,
                }
            ] if include_lane else [],
        }

    def _status(
        self,
        *,
        state: dict | None = None,
        results: dict | None = None,
        ready: dict | None = None,
        observations: dict | None = None,
        request_count: int = 4,
    ) -> dict:
        state = self._state() if state is None else state
        results = self._results() if results is None else results
        ready = self._ready() if ready is None else ready
        return {
            "health": {"ok": True, "service": "crowdtensord-coordinator"},
            "ready": ready,
            "state": state,
            "results": results,
            "observations": observations or {
                "health": {"endpoint": "health", "ok": True, "status": 200},
                "ready": {"endpoint": "ready", "ok": True, "status": 200},
                "state": {"endpoint": "state", "ok": True, "status": 200},
                "admin_results": {"endpoint": "admin_results", "ok": True, "status": 200},
            },
            "summary": remote_demo_acceptance_pack.readiness_summary(
                state=state,
                results=results,
                miner_id="remote-a",
                request_count=request_count,
                scenario_id="route-baseline",
            ),
        }

    def _wait(self, *, status: dict | None = None, ok: bool = True) -> dict:
        return {
            "ok": ok,
            "attempts": 2,
            "elapsed_seconds": 1.5,
            "status": self._status() if status is None else status,
            "errors": [],
        }

    def _artifacts(self, *, evidence_ok: bool = True, support_ok: bool = True) -> dict:
        return {
            "evidence": {
                "ok": evidence_ok,
                "path": "/tmp/evidence.json",
                "markdown_path": "/tmp/evidence.md",
                "summary": {
                    "schema": "remote_compute_evidence_v1",
                    "ok": evidence_ok,
                    "route": "remote_python_model_bundle_infer",
                    "route_confidence": "ready",
                    "route_usable_now": True,
                    "request_count": 4,
                    "scenario_schema": "model_bundle_inference_scenario_v1",
                    "scenario_id": "route-baseline",
                    "scenario_matches": True,
                    "request_trace_count": 4,
                    "accuracy": 0.25,
                    "requests_per_second": 1200.0,
                    "read_only": True,
                    "redaction_ok": True,
                    "observability_schema": "remote_compute_observability_v1",
                },
            },
            "support_bundle": {
                "ok": support_ok,
                "path": "/tmp/support.json",
                "markdown_path": "/tmp/support.md",
                "summary": {
                    "online_enabled": True,
                    "health_ok": True,
                    "state_ok": True,
                    "admin_results_ok": True,
                },
            },
        }

    def test_readiness_summary_requires_profile_result_validation_and_request_count(self) -> None:
        summary = remote_demo_acceptance_pack.readiness_summary(
            state=self._state(),
            results=self._results(),
            miner_id="remote-a",
            request_count=4,
            scenario_id="route-baseline",
        )

        self.assertTrue(summary["ready"])
        self.assertEqual(summary["route"], "remote_python_model_bundle_infer")
        self.assertIn("runtime:python-cli", summary["matched_capabilities"])
        self.assertIn("accepted_result", summary["matched_capabilities"])
        self.assertIn("scenario_id", summary["matched_capabilities"])
        self.assertEqual(summary["inference"]["scenario_id"], "route-baseline")
        self.assertTrue(summary["inference"]["scenario_matches"])
        self.assertFalse(summary["missing_capabilities"])

        failed = remote_demo_acceptance_pack.readiness_summary(
            state=self._state(),
            results={"results": []},
            miner_id="remote-a",
            request_count=4,
            scenario_id="route-baseline",
        )
        self.assertFalse(failed["ready"])
        self.assertIn("accepted_result", failed["missing_capabilities"])

    def test_build_report_summarizes_artifacts_without_secrets(self) -> None:
        args = self._args()
        wait = self._wait()
        artifacts = self._artifacts()

        report = remote_demo_acceptance_pack.build_report(
            args=args,
            wait=wait,
            artifacts=artifacts,
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["schema"], "remote_demo_acceptance_v1")
        self.assertEqual(report["route"], "remote_python_model_bundle_infer")
        self.assertEqual(report["diagnosis"]["primary_code"], "acceptance_ready")
        self.assertEqual(report["diagnosis_codes"], ["acceptance_ready"])
        self.assertEqual(report["evidence_summary"]["schema"], "remote_compute_evidence_v1")
        self.assertEqual(report["observability_summary"]["schema"], "remote_demo_observability_v1")
        self.assertEqual(report["observability_summary"]["availability"]["attempts"], 2)
        self.assertTrue(report["observability_summary"]["availability"]["health_ok"])
        self.assertEqual(report["observability_summary"]["work_queue"]["accepted_results"], 1)
        self.assertEqual(report["observability_summary"]["miner"]["runtime"], "python-cli")
        self.assertEqual(report["observability_summary"]["inference"]["request_count"], 4)
        self.assertEqual(report["scenario"]["scenario_id"], "route-baseline")
        self.assertEqual(report["evidence_summary"]["scenario_id"], "route-baseline")
        self.assertTrue(report["evidence_summary"]["scenario_matches"])
        self.assertEqual(report["observability_summary"]["inference"]["scenario_id"], "route-baseline")
        self.assertTrue(report["observability_summary"]["inference"]["scenario_matches"])
        self.assertEqual(report["observability_summary"]["inference"]["requests_per_second"], 1200.0)
        self.assertFalse(report["session_request"]["created"])
        self.assertEqual(
            report["observability_summary"]["artifacts"]["evidence_observability_schema"],
            "remote_compute_observability_v1",
        )
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("observer-secret", encoded)
        self.assertNotIn("admin-secret", encoded)

    def test_build_report_summarizes_created_session_request(self) -> None:
        args = self._args()
        args.create_session = True
        args.session_task_id = "task-created"
        state = self._state()
        state["tasks"][0]["task_id"] = "task-created"
        status = self._status(
            state=state,
            results=self._results(task_id="task-created"),
        )
        status["summary"] = remote_demo_acceptance_pack.readiness_summary(
            state=state,
            results=status["results"],
            miner_id="remote-a",
            request_count=4,
            scenario_id="route-baseline",
            task_id="task-created",
        )
        wait = self._wait(status=status)
        wait["session_create"] = {
            "ok": True,
            "session": {
                "created": True,
                "schema": "inference_session_request_v1",
                "task_id": "task-created",
                "request_count": 4,
                "scenario_schema": "model_bundle_inference_scenario_v1",
                "scenario_id": "route-baseline",
                "scenario_description": "Fixed CPU read-only route prompts from the built-in bundle corpus.",
                "scenario_request_count": 8,
                "workload_type": "model_bundle_infer",
                "status": "queued",
            },
        }

        report = remote_demo_acceptance_pack.build_report(
            args=args,
            wait=wait,
            artifacts=self._artifacts(),
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertTrue(report["ok"])
        self.assertTrue(report["session_request"]["created"])
        self.assertEqual(report["session_request"]["task_id"], "task-created")
        self.assertEqual(report["session_request"]["scenario_id"], "route-baseline")
        self.assertEqual(report["observability_summary"]["session_request"]["task_id"], "task-created")
        self.assertEqual(report["observability_summary"]["work_queue"]["expected_task_id"], "task-created")

    def test_create_session_mode_ignores_old_result_without_matching_task_id(self) -> None:
        state = self._state()
        state["tasks"][0]["task_id"] = "old-task"

        summary = remote_demo_acceptance_pack.readiness_summary(
            state=state,
            results=self._results(task_id="old-task"),
            miner_id="remote-a",
            request_count=4,
            scenario_id="route-baseline",
            task_id="new-task",
        )

        self.assertFalse(summary["ready"])
        self.assertEqual(summary["expected_task_id"], "new-task")
        self.assertIn("accepted_result", summary["missing_capabilities"])

    def test_readiness_summary_rejects_scenario_mismatch(self) -> None:
        validation = {
            "code": "ok",
            "request_count": 4,
            "scenario_schema": "model_bundle_inference_scenario_v1",
            "scenario_id": "mixed-prompts",
            "request_trace_count": 4,
            "accuracy": 0.25,
        }

        summary = remote_demo_acceptance_pack.readiness_summary(
            state=self._state(validation=validation),
            results=self._results(validation=validation),
            miner_id="remote-a",
            request_count=4,
            scenario_id="route-baseline",
        )

        self.assertFalse(summary["ready"])
        self.assertIn("scenario_id", summary["missing_capabilities"])
        self.assertEqual(summary["inference"]["scenario_id"], "mixed-prompts")
        self.assertFalse(summary["inference"]["scenario_matches"])

    def test_diagnosis_classifies_session_create_failure(self) -> None:
        args = self._args()
        args.create_session = True
        wait = {
            "ok": False,
            "attempts": 0,
            "elapsed_seconds": 0.0,
            "status": {},
            "errors": [],
            "session_create": {
                "ok": False,
                "observation": {
                    "endpoint": "admin_inference_session",
                    "ok": False,
                    "status": 403,
                    "detail": "invalid admin token",
                },
            },
        }

        report = remote_demo_acceptance_pack.build_report(
            args=args,
            wait=wait,
            artifacts=None,
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["diagnosis"]["primary_code"], "session_create_failed")
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("admin-secret", encoded)

    def test_diagnosis_classifies_coordinator_unreachable(self) -> None:
        args = self._args()
        status = self._status(
            state={},
            results={},
            ready={},
            observations={
                "health": {
                    "endpoint": "health",
                    "ok": False,
                    "status": None,
                    "error": "URLError",
                    "detail": "connection refused",
                }
            },
        )

        report = remote_demo_acceptance_pack.build_report(
            args=args,
            wait=self._wait(status=status, ok=False),
            artifacts=None,
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["diagnosis"]["primary_code"], "coordinator_unreachable")
        self.assertEqual(report["observability_summary"]["schema"], "remote_demo_observability_v1")
        self.assertFalse(report["observability_summary"]["availability"]["health_ok"])

    def test_diagnosis_classifies_auth_failures(self) -> None:
        args = self._args()
        status = self._status(
            state={},
            results={},
            observations={
                "health": {"endpoint": "health", "ok": True, "status": 200},
                "ready": {"endpoint": "ready", "ok": True, "status": 200},
                "state": {"endpoint": "state", "ok": False, "status": 403, "detail": "invalid observer token"},
                "admin_results": {"endpoint": "admin_results", "ok": False, "status": 403, "detail": "invalid admin token"},
            },
        )

        report = remote_demo_acceptance_pack.build_report(
            args=args,
            wait=self._wait(status=status, ok=False),
            artifacts=None,
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertEqual(report["diagnosis"]["primary_code"], "observer_auth_failed")
        self.assertIn("admin_auth_failed", report["diagnosis_codes"])
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("observer-secret", encoded)
        self.assertNotIn("admin-secret", encoded)

    def test_diagnosis_classifies_missing_miner_and_lane(self) -> None:
        args = self._args()
        state = self._state(miner_id="other-miner", include_task=False, include_lane=False)
        status = self._status(state=state, results={"results": []}, ready=self._ready(include_lane=False))

        report = remote_demo_acceptance_pack.build_report(
            args=args,
            wait=self._wait(status=status, ok=False),
            artifacts=None,
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertEqual(report["diagnosis"]["primary_code"], "miner_not_seen")
        self.assertIn("task_lane_missing", report["diagnosis_codes"])

    def test_diagnosis_classifies_workload_and_result_gaps(self) -> None:
        args = self._args()
        state = self._state(workloads=["diloco_train"], include_task=False, include_lane=True)
        status = self._status(state=state, results={"results": []}, ready=self._ready(include_lane=True))

        report = remote_demo_acceptance_pack.build_report(
            args=args,
            wait=self._wait(status=status, ok=False),
            artifacts=None,
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertEqual(report["diagnosis"]["primary_code"], "workload_not_advertised")
        self.assertIn("no_accepted_result", report["diagnosis_codes"])

    def test_diagnosis_classifies_validation_failure(self) -> None:
        args = self._args()
        validation = {"code": "accuracy_below_threshold", "request_count": 4}
        state = self._state(validation=validation)
        results = self._results(validation=validation)
        status = self._status(state=state, results=results)

        report = remote_demo_acceptance_pack.build_report(
            args=args,
            wait=self._wait(status=status, ok=False),
            artifacts=None,
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertEqual(report["diagnosis"]["primary_code"], "validation_failed")

    def test_diagnosis_classifies_request_count_mismatch(self) -> None:
        args = self._args()
        validation = {
            "code": "ok",
            "request_count": 2,
            "scenario_schema": "model_bundle_inference_scenario_v1",
            "scenario_id": "route-baseline",
        }
        state = self._state(validation=validation)
        results = self._results(validation=validation)
        status = self._status(state=state, results=results)

        report = remote_demo_acceptance_pack.build_report(
            args=args,
            wait=self._wait(status=status, ok=False),
            artifacts=None,
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertEqual(report["diagnosis"]["primary_code"], "request_count_mismatch")

    def test_diagnosis_classifies_artifact_collection_failure(self) -> None:
        args = self._args()

        report = remote_demo_acceptance_pack.build_report(
            args=args,
            wait=self._wait(ok=True),
            artifacts=self._artifacts(evidence_ok=False),
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["diagnosis"]["primary_code"], "artifact_collection_failed")

    def test_markdown_renders_acceptance_sections(self) -> None:
        report = {
            "generated_at": "2026-05-22T00:00:00+00:00",
            "ok": True,
            "coordinator_url": "https://coordinator.example",
            "miner_id": "remote-a",
            "route": "remote_python_model_bundle_infer",
            "scenario": {
                "scenario_schema": "model_bundle_inference_scenario_v1",
                "scenario_id": "route-baseline",
            },
            "wait_summary": {"ok": True, "attempts": 1, "elapsed_seconds": 0.1},
            "diagnosis": {
                "primary_code": "acceptance_ready",
                "severity": "info",
                "summary": "Remote demo acceptance completed.",
                "next_steps": ["Share the generated report."],
            },
            "diagnosis_codes": ["acceptance_ready"],
            "evidence_summary": {
                "schema": "remote_compute_evidence_v1",
                "ok": True,
                "scenario_id": "route-baseline",
                "scenario_matches": True,
            },
            "support_bundle_summary": {"online_enabled": True},
            "observability_summary": {
                "schema": "remote_demo_observability_v1",
                "availability": {
                    "health_ok": True,
                    "state_ok": True,
                    "admin_results_ok": True,
                },
                "work_queue": {"accepted_results": 1},
                "miner": {"runtime": "python-cli", "backend": "cpu"},
                "inference": {
                    "request_count": 4,
                    "scenario_id": "route-baseline",
                    "scenario_matches": True,
                    "request_trace_count": 4,
                    "requests_per_second": 1200.0,
                },
                "artifacts": {"evidence_ok": True, "support_bundle_ok": True},
            },
            "artifacts": {"evidence_json": "/tmp/evidence.json"},
            "session_request": {
                "created": True,
                "schema": "inference_session_request_v1",
                "task_id": "task-1",
                "request_count": 4,
                "scenario_schema": "model_bundle_inference_scenario_v1",
                "scenario_id": "route-baseline",
                "workload_type": "model_bundle_infer",
            },
            "limitations": ["Controlled demo"],
        }

        markdown = remote_demo_acceptance_pack.render_markdown(report)

        self.assertIn("# CrowdTensor Remote Demo Acceptance", markdown)
        self.assertIn("remote_python_model_bundle_infer", markdown)
        self.assertIn("Scenario: `route-baseline`", markdown)
        self.assertIn("## Diagnosis", markdown)
        self.assertIn("## Session Request", markdown)
        self.assertIn("inference_session_request_v1", markdown)
        self.assertIn("acceptance_ready", markdown)
        self.assertIn("## Observability", markdown)
        self.assertIn("remote_demo_observability_v1", markdown)
        self.assertIn("Support Bundle", markdown)


if __name__ == "__main__":
    unittest.main()
