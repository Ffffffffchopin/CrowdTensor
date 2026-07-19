from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from crowdtensor import cli
from scripts import gpu_swarm_usability_alpha_check as check
from scripts import gpu_swarm_usability_alpha_pack as pack


class GpuSwarmUsabilityAlphaTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_gpu_swarm_alpha_test_"))

    def _write_json(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _evidence_reports(self, base: Path) -> tuple[Path, Path, Path]:
        handoff = {
            "schema": "core_technology_handoff_rc_v1",
            "ok": True,
            "core_technology_large_model_alpha_ready": True,
            "safety": {"public_artifact_safe": True},
        }
        status = {
            "schema": "core_technology_validation_status_v1",
            "ok": True,
            "core_validation_ready": True,
            "largest_successful_tier": "14b",
            "safety": {"public_artifact_safe": True},
        }
        control = {
            "schema": "control_user_alpha_v1",
            "ok": True,
            "control_layer_ready": True,
            "user_layer_ready": True,
            "session_lifecycle_ready": True,
            "model_catalog": {
                "schema": "control_user_model_catalog_v1",
                "model_catalog_ready": True,
                "default_model_id": "Qwen/Qwen2.5-14B-Instruct",
                "capabilities": {
                    "large_model_stage_selective_ready": True,
                    "n_stage_partition_plan_ready": True,
                    "stage_selective_performance_report_ready": True,
                    "stage_weight_download_scope_ready": True,
                },
                "models": [
                    {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "model_family": "7b",
                        "backend": "hf_transformers_cuda",
                        "execution_mode": "stage_selective_hf",
                        "partition_mode": "stage_local",
                        "stage_count_live": 2,
                        "target_stage_count": 4,
                        "live_verified": True,
                        "multi_token_verified": True,
                        "verified_token_count": 2,
                        "n_stage_plan_ready": True,
                        "public_artifact_safe": True,
                    },
                    {
                        "model_id": "Qwen/Qwen2.5-14B-Instruct",
                        "model_family": "14b",
                        "backend": "hf_transformers_cuda",
                        "execution_mode": "stage_selective_hf",
                        "partition_mode": "stage_local",
                        "stage_count_live": 2,
                        "target_stage_count": 4,
                        "live_verified": True,
                        "dual_kaggle_verified": True,
                        "verified_token_count": 1,
                        "n_stage_plan_ready": True,
                        "public_artifact_safe": True,
                    },
                ],
                "boundaries": list(pack.BOUNDARIES),
            },
        }
        return (
            self._write_json(base / "core_handoff.json", handoff),
            self._write_json(base / "core_status.json", status),
            self._write_json(base / "control_user_alpha.json", control),
        )

    def _pack_args(self, base: Path, output_dir: Path, *extra: str) -> list[str]:
        handoff, status, control = self._evidence_reports(base)
        return [
            "--output-dir",
            str(output_dir),
            "--core-handoff-report",
            str(handoff),
            "--core-status-report",
            str(status),
            "--control-user-alpha-report",
            str(control),
            *extra,
        ]

    def test_pack_builds_user_flow_join_packs_and_check_validates(self) -> None:
        base = self._tmp_dir()
        output_dir = base / "gpu-alpha"

        report = pack.build_report(pack.parse_args(self._pack_args(base, output_dir)))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["gpu_swarm_usability_alpha_ready"])
        self.assertTrue(report["user_gpu_swarm_entrypoint_ready"])
        self.assertTrue(report["gpu_miner_join_pack_ready"])
        self.assertTrue(report["coordinator_workflow_ready"])
        self.assertTrue(report["two_gpu_stage_route_ready"])
        self.assertTrue(report["inference_request_lifecycle_ready"])
        self.assertTrue(report["model_catalog_imported"])
        self.assertTrue(report["control_user_alpha_imported"])
        self.assertTrue(report["core_handoff_imported"])
        self.assertTrue(report["public_artifact_safe"])
        self.assertEqual(report["execution_mode"], "evidence-import")
        self.assertFalse(report["external_runtime_verified"])
        self.assertEqual(check.validate_report(report), [])

        stages = {item["stage"]: item for item in report["miner_join_packs"]["stages"]}
        self.assertEqual(set(stages), {"stage0", "stage1"})
        self.assertEqual(stages["stage0"]["required_capability"], "real_llm_sharded_cuda_stage0")
        self.assertEqual(stages["stage1"]["required_capability"], "real_llm_sharded_cuda_stage1")
        self.assertEqual(stages["stage0"]["backend"], "hf_transformers_cuda")
        self.assertIn(pack.SAFE_MINER_TOKEN_ENV, stages["stage0"]["command_template"])
        self.assertNotIn("'${", stages["stage0"]["command_template"])

        events = [item["event"] for item in report["inference_lifecycle"]["events"]]
        self.assertEqual(["prepare", "coordinator_plan", "miner_join_plan", "infer_request", "status", "collect"], events)
        labels = {item["label"] for item in report["user_workflow"]["next_commands"]}
        self.assertTrue({"prepare", "coordinator", "miner-stage0", "miner-stage1", "infer", "status", "collect"} <= labels)

        for path in [
            output_dir / "gpu_swarm_usability_alpha.json",
            output_dir / "GPU_SWARM_ALPHA.md",
            output_dir / "support_bundle.json",
            output_dir / "stage-stage0" / "miner_join.sh",
            output_dir / "stage-stage1" / "miner_join.sh",
            output_dir / "stage-stage0" / "miner.env.template",
            output_dir / "stage-stage1" / "miner.env.template",
            output_dir / "start_coordinator.sh",
        ]:
            self.assertTrue(path.is_file(), path)

    def test_external_existing_mode_is_honest_when_runtime_not_verified(self) -> None:
        base = self._tmp_dir()
        output_dir = base / "external-existing"

        report = pack.build_report(pack.parse_args(self._pack_args(
            base,
            output_dir,
            "--execution-mode",
            "external-existing",
        )))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["execution_mode"], "external-existing")
        self.assertFalse(report["external_runtime_verified"])
        self.assertFalse(report["mode_truth"]["fresh_gpu_run_performed"])
        self.assertEqual(report["inference_lifecycle"]["status"]["state"], "requires_external_runtime")
        self.assertIn("gpu_swarm_external_runtime_not_verified", report["diagnosis_codes"])
        self.assertEqual(check.validate_report(report), [])

    def test_miner_action_preserves_selected_stage_and_safe_join_script(self) -> None:
        base = self._tmp_dir()
        output_dir = base / "miner-stage1"

        report = pack.build_report(pack.parse_args(self._pack_args(
            base,
            output_dir,
            "--action",
            "miner",
            "--stage",
            "stage1",
        )))

        self.assertEqual(report["action"], "miner")
        self.assertEqual(report["selected_stage"], "stage1")
        self.assertIn("gpu_swarm_miner_ready", report["diagnosis_codes"])
        join_script = (output_dir / "stage-stage1" / "miner_join.sh").read_text(encoding="utf-8")
        self.assertIn(pack.SAFE_MINER_TOKEN_ENV, join_script)
        self.assertIn("--real-llm-stage-role stage1", join_script)
        self.assertNotIn("CROWDTENSOR_MINER_TOKEN", join_script)
        self.assertNotIn("operator.private.env", join_script)

    def test_public_artifacts_are_redacted(self) -> None:
        base = self._tmp_dir()
        output_dir = base / "redaction"
        report = pack.build_report(pack.parse_args(self._pack_args(base, output_dir)))

        self.assertEqual(pack.public_redaction_errors(report), [])
        scanned = "\n".join(
            path.read_text(encoding="utf-8")
            for path in output_dir.rglob("*")
            if path.is_file()
        )
        for fragment in [
            "CROWDTENSOR_MINER_TOKEN",
            "CROWDTENSOR_OBSERVER_TOKEN",
            "CROWDTENSOR_ADMIN_TOKEN",
            "operator.private.env",
            "miner.private.env",
            "miner_registry.json",
            "kernel.py",
            '"generated_text":',
            '"generated_token_ids":',
            '"activation":',
        ]:
            self.assertNotIn(fragment, scanned)

    def test_check_script_builds_and_validates_report(self) -> None:
        base = self._tmp_dir()
        handoff, status, control = self._evidence_reports(base)

        result = check.build_check(check.parse_args([
            "--output-dir",
            str(base / "check-alpha"),
            "--core-handoff-report",
            str(handoff),
            "--core-status-report",
            str(status),
            "--control-user-alpha-report",
            str(control),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["gpu_swarm_usability_alpha_ready"])
        self.assertEqual(result["errors"], [])

    def test_cli_wrapper_generates_one_command_smoke_summary(self) -> None:
        base = self._tmp_dir()
        handoff, status, control = self._evidence_reports(base)
        output_dir = base / "cli-alpha"

        args = cli.parse_args([
            "gpu-swarm",
            "smoke",
            "--output-dir",
            str(output_dir),
            "--core-handoff-report",
            str(handoff),
            "--core-status-report",
            str(status),
            "--control-user-alpha-report",
            str(control),
            "--json",
        ])
        summary = cli.build_gpu_swarm_usability_alpha(args)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "gpu_swarm_usability_alpha_cli_v1")
        self.assertEqual(summary["schema"], pack.SCHEMA)
        self.assertTrue(summary["two_gpu_stage_route_ready"])
        self.assertTrue(summary["public_artifact_safe"])
        self.assertTrue((output_dir / "gpu_swarm_usability_alpha_cli_summary.json").is_file())

        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            cli.print_gpu_swarm_usability_alpha(summary)
        output = rendered.getvalue()
        self.assertIn("CrowdTensor GPU Swarm Usability Alpha", output)
        self.assertIn("join_pack=True", output)
        self.assertIn("miner: stage=stage0", output)

    def test_argument_validation_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args(["--max-new-tokens", "0"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["gpu-swarm", "smoke", "--port", "0"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["gpu-swarm", "miner"])
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "gpu-swarm",
                "smoke",
                "--core-handoff-report",
                "/tmp/does-not-exist-gpu-swarm-alpha.json",
            ])


if __name__ == "__main__":
    unittest.main()
