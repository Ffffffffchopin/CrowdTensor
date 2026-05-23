from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "demo_manifest_pack.py"
SPEC = importlib.util.spec_from_file_location("demo_manifest_pack", SCRIPT_PATH)
demo_manifest_pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(demo_manifest_pack)


class DemoManifestPackTests(unittest.TestCase):
    def _runtime_matrix(self) -> dict:
        return {
            "ok": True,
            "host_profile": {
                "python": "3.12.0",
                "os": "Linux",
                "machine": "x86_64",
                "cpu_count": 8,
            },
            "summary": {
                "available_workloads": ["model_bundle_infer"],
                "blocked_workloads": [],
            },
            "hardware_targets": [
                {
                    "name": "cpu_baseline",
                    "status": "available",
                    "usable_now": True,
                    "operator_action": "run_now",
                    "diagnosis_codes": ["cpu_baseline_ready"],
                },
            ],
            "recommended_routes": [
                {
                    "name": "local_cpu_model_bundle_infer",
                    "target": "cpu_baseline",
                    "workload": "model_bundle_infer",
                    "status": "available",
                    "usable_now": True,
                    "confidence": "ready",
                    "operator_action": "run_now",
                    "diagnosis_codes": ["cpu_baseline_ready"],
                },
            ],
            "diagnosis_summary": {"codes": ["cpu_baseline_ready"]},
            "hardware_diagnosis_summary": {"codes": ["cpu_baseline_ready"]},
        }

    def _remote_evidence(self) -> dict:
        return {
            "schema": "remote_compute_evidence_v1",
            "mode": "local-loopback",
            "ok": True,
            "route_decision": {
                "name": "remote_python_model_bundle_infer",
                "target": "remote_python_miner",
                "workload": "model_bundle_infer",
                "confidence": "ready",
                "usable_now": True,
                "matched_capabilities": ["runtime:python-cli", "backend:cpu"],
                "missing_capabilities": [],
            },
            "miner": {
                "miner_id": "remote-a",
                "profile": {
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "accepted": 1,
                    "rejected": 0,
                },
            },
            "inference_summary": {
                "ok": True,
                "workload_type": "model_bundle_infer",
                "request_count": 4,
                "expected_request_count": 4,
                "request_trace_count": 4,
                "accuracy": 0.25,
                "elapsed_ms": 2.0,
                "requests_per_second": 2000.0,
            },
            "observability_summary": {
                "schema": "remote_compute_observability_v1",
                "route": {
                    "name": "remote_python_model_bundle_infer",
                    "confidence": "ready",
                    "usable_now": True,
                    "matched_capabilities": ["runtime:python-cli"],
                    "missing_capabilities": [],
                },
                "miner": {
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "accepted": 1,
                    "rejected": 0,
                },
                "work_queue": {
                    "task_counts": {"completed": 1},
                    "accepted_results": 1,
                    "rejected_results": 0,
                    "ledger_rows": 1,
                },
                "inference": {
                    "ok": True,
                    "request_count": 4,
                    "expected_request_count": 4,
                    "request_trace_count": 4,
                    "accuracy": 0.25,
                    "elapsed_ms": 2.0,
                    "requests_per_second": 2000.0,
                },
                "safety": {
                    "read_only": True,
                    "redaction_ok": True,
                    "registry_hashed": True,
                    "raw_payloads_exposed": False,
                },
            },
            "safety": {
                "read_only": True,
                "redaction_ok": True,
                "registry_hashed": True,
                "raw_payloads_exposed": False,
            },
        }

    def _support_bundle(self) -> dict:
        return {
            "doctor": {"ok": True},
            "release_gate": {"ok": True},
            "online": {"enabled": False},
            "reports": {
                "remote": {
                    "present": True,
                    "ok": True,
                    "diagnosis_codes": ["acceptance_ready"],
                    "observability_summaries": [
                        {
                            "schema": "remote_compute_observability_v1",
                            "route": {"name": "remote_python_model_bundle_infer"},
                            "inference": {"request_count": 4, "requests_per_second": 2000.0},
                        },
                    ],
                },
            },
        }

    def test_build_manifest_summarizes_safe_artifacts(self) -> None:
        output_dir = Path(self._tmp_dir())
        runtime_json = output_dir / "runtime_matrix.json"
        remote_json = output_dir / "remote_compute_evidence.json"
        support_json = output_dir / "support_bundle.json"
        manifest_md = output_dir / "demo_manifest.md"
        for path in [runtime_json, remote_json, support_json, manifest_md]:
            path.write_text("{}", encoding="utf-8")
        artifacts = {
            "runtime_matrix": demo_manifest_pack.artifact_entry(
                output_dir=output_dir,
                path=runtime_json,
                kind="runtime_matrix",
                ok=True,
            ),
            "remote_compute_evidence_json": demo_manifest_pack.artifact_entry(
                output_dir=output_dir,
                path=remote_json,
                kind="remote_compute_evidence",
                schema="remote_compute_evidence_v1",
                ok=True,
            ),
            "support_bundle_json": demo_manifest_pack.artifact_entry(
                output_dir=output_dir,
                path=support_json,
                kind="support_bundle",
                ok=True,
            ),
            "demo_manifest_markdown": demo_manifest_pack.artifact_entry(
                output_dir=output_dir,
                path=manifest_md,
                kind="demo_manifest_markdown",
            ),
        }

        manifest = demo_manifest_pack.build_manifest(
            output_dir=output_dir,
            mode="local-loopback",
            request_count=4,
            runtime=self._runtime_matrix(),
            remote_evidence=self._remote_evidence(),
            support=self._support_bundle(),
            artifacts=artifacts,
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertTrue(manifest["ok"])
        self.assertEqual(manifest["schema"], "demo_manifest_v1")
        self.assertEqual(
            manifest["summaries"]["remote_compute_evidence"]["route"]["name"],
            "remote_python_model_bundle_infer",
        )
        self.assertEqual(
            manifest["summaries"]["remote_compute_evidence"]["observability"]["schema"],
            "remote_compute_observability_v1",
        )
        self.assertEqual(
            manifest["summaries"]["support_bundle"]["remote_report"]["observability_summaries"][0]["schema"],
            "remote_compute_observability_v1",
        )
        for artifact in manifest["artifacts"].values():
            self.assertFalse(Path(artifact["path"]).is_absolute())
        encoded = json.dumps(manifest, sort_keys=True)
        self.assertNotIn("lease_token", encoded)
        self.assertNotIn("idempotency_key", encoded)

    def test_markdown_renders_manifest_sections(self) -> None:
        output_dir = Path(self._tmp_dir())
        manifest = demo_manifest_pack.build_manifest(
            output_dir=output_dir,
            mode="local-loopback",
            request_count=4,
            runtime=self._runtime_matrix(),
            remote_evidence=self._remote_evidence(),
            support=self._support_bundle(),
            artifacts={
                "runtime_matrix": {
                    "kind": "runtime_matrix",
                    "path": "runtime_matrix.json",
                    "present": True,
                    "ok": True,
                },
            },
            generated_at="2026-05-22T00:00:00+00:00",
        )

        markdown = demo_manifest_pack.render_markdown(manifest)

        self.assertIn("# CrowdTensor Demo Manifest", markdown)
        self.assertIn("demo_manifest_v1", markdown)
        self.assertIn("remote_python_model_bundle_infer", markdown)
        self.assertIn("remote_compute_observability_v1", markdown)
        self.assertIn("runtime_matrix.json", markdown)

    def test_manifest_marks_secret_like_payload_unsafe(self) -> None:
        remote = self._remote_evidence()
        remote["miner"]["miner_id"] = "demo-manifest-token"

        manifest = demo_manifest_pack.build_manifest(
            output_dir=Path(self._tmp_dir()),
            mode="local-loopback",
            request_count=4,
            runtime=self._runtime_matrix(),
            remote_evidence=remote,
            support=self._support_bundle(),
            artifacts={},
            generated_at="2026-05-22T00:00:00+00:00",
        )

        self.assertFalse(manifest["ok"])
        self.assertIn("safety_error", manifest)

    def _tmp_dir(self) -> str:
        path = Path(self.id().replace(".", "_").replace("/", "_"))
        tmp_root = Path("/tmp") / f"crowdtensor_demo_manifest_{path.name}"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(tmp_root, ignore_errors=True))
        return str(tmp_root)


if __name__ == "__main__":
    unittest.main()
