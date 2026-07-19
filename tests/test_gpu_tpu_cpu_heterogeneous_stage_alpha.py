from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gpu_tpu_cpu_heterogeneous_stage_alpha_check as check
from scripts import gpu_tpu_cpu_heterogeneous_stage_alpha_pack as pack
from crowdtensor import cli


class GpuTpuCpuHeterogeneousStageAlphaTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_gpu_tpu_cpu_alpha_test_"))

    def _write_json(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _fixture_reports(self, base: Path) -> dict[str, Path]:
        return {
            "tpu": self._write_json(base / "tpu.json", pack.fixture_tpu_report()),
            "gpu_full": self._write_json(base / "gpu_full.json", pack.fixture_gpu_full_report()),
            "gpu_awq": self._write_json(base / "gpu_awq.json", pack.fixture_gpu_awq_report()),
            "cpu": self._write_json(base / "cpu.json", pack.fixture_cpu_report()),
        }

    def _pack_args(self, reports: dict[str, Path], output_dir: Path, *extra: str) -> list[str]:
        return [
            "--output-dir",
            str(output_dir),
            "--execution-mode",
            "evidence-import",
            "--tpu-real-llm-report",
            str(reports["tpu"]),
            "--gpu-full-32b-report",
            str(reports["gpu_full"]),
            "--gpu-awq-32b-report",
            str(reports["gpu_awq"]),
            "--cpu-real-llm-report",
            str(reports["cpu"]),
            "--local-e2e-mode",
            "fixture",
            "--bridge-mode",
            "fixture",
            *extra,
        ]

    def test_pack_builds_alpha_report_and_check_validates(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)
        output_dir = base / "alpha"

        report = pack.build_report(pack.parse_args(self._pack_args(reports, output_dir)))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["gpu_tpu_cpu_heterogeneous_stage_alpha_ready"])
        self.assertTrue(report["small_medium_real_model_end_to_end_ready"])
        self.assertTrue(report["gpu_tpu_cpu_32b_feasibility_report_ready"])
        self.assertTrue(report["next_rc_boundary_ready"])
        self.assertTrue(report["local_three_stage_real_model_e2e_ready"])
        self.assertTrue(report["torch_jax_torch_bridge_ready"])
        self.assertFalse(report["same_request_live_heterogeneous_verified"])
        self.assertFalse(report["live_tpu_stage_miner_integrated"])
        self.assertEqual(check.validate_report(report), [])

    def test_stage_contract_requires_cuda_tpu_and_cpu_without_live_overclaim(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)
        report = pack.build_report(pack.parse_args(self._pack_args(reports, base / "contract")))

        contract = report["stage_contract_smoke"]
        self.assertEqual(contract["execution_kind"], "logical_contract_with_retained_backend_evidence")
        self.assertFalse(contract["same_request_live_heterogeneous_verified"])
        self.assertFalse(contract["live_tpu_stage_miner_integrated"])
        backends = {stage["backend"] for stage in contract["stage_plan"]}
        self.assertEqual(backends, {"cuda", "jax_tpu", "cpu"})
        self.assertEqual(len(contract["handoffs"]), 3)
        self.assertTrue(all(item["activation_hash"].startswith("sha256:") for item in contract["handoffs"]))
        self.assertTrue(all(item["activation_payload_public"] is False for item in contract["handoffs"]))

    def test_local_three_stage_real_model_e2e_runner_executes_tiny_gpt2(self) -> None:
        args = pack.parse_args([
            "--output-dir",
            str(self._tmp_dir() / "real-local-run"),
            "--execution-mode",
            "fixture",
            "--local-e2e-mode",
            "run",
            "--local-e2e-model-id",
            "hf-internal-testing/tiny-random-gpt2",
            "--small-medium-min-parameter-count",
            "1",
        ])

        result = pack.build_local_three_stage_e2e(args)

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["three_stage_real_model_e2e_ready"])
        self.assertTrue(result["real_hf_model_loaded"])
        self.assertTrue(result["real_model_forward_executed"])
        self.assertTrue(result["baseline_match"])
        self.assertEqual(result["stage_count"], 3)
        self.assertEqual({stage["target_backend_family"] for stage in result["stage_plan"]}, {"gpu", "tpu", "cpu"})
        self.assertTrue(all(item["activation_hash"].startswith("sha256:") for item in result["activation_handoffs"]))
        self.assertFalse(result["generated_token_ids_public"])

    def test_torch_jax_bridge_probe_blocks_cleanly_without_jax_or_runs(self) -> None:
        args = pack.parse_args([
            "--output-dir",
            str(self._tmp_dir() / "bridge"),
            "--execution-mode",
            "fixture",
            "--bridge-mode",
            "run",
            "--bridge-model-id",
            "hf-internal-testing/tiny-random-gpt2",
        ])

        result = pack.build_torch_jax_bridge_probe(args)

        self.assertEqual(result["schema"], pack.TORCH_JAX_BRIDGE_SCHEMA)
        self.assertFalse(result["activation_public"])
        self.assertFalse(result["generated_token_ids_public"])
        if result["ok"]:
            self.assertTrue(result["bridge_ready"])
            self.assertTrue(result["baseline_match"])
            self.assertEqual(result["generated_token_count"], 1)
        else:
            self.assertFalse(result["bridge_ready"])
            self.assertTrue(result["blockers"])

    def test_32b_feasibility_marks_rc_ready_but_not_current_same_request_success(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)
        report = pack.build_report(pack.parse_args(self._pack_args(reports, base / "feasibility")))

        feasibility = report["heterogeneous_32b_feasibility"]
        self.assertTrue(feasibility["gpu_tpu_cpu_32b_feasibility_report_ready"])
        self.assertTrue(feasibility["ready_for_bounded_rc"])
        self.assertEqual(feasibility["verdict"], "ready_for_bounded_rc_not_yet_live_verified")
        self.assertFalse(feasibility["gpu_tpu_cpu_32b_same_request_feasible_now"])
        self.assertFalse(feasibility["same_request_live_heterogeneous_verified"])
        self.assertFalse(feasibility["tpu_32b_runtime_adapter_ready"])
        self.assertTrue(feasibility["local_three_stage_real_model_e2e_ready"])
        blockers = feasibility["blockers"]
        self.assertTrue(blockers["same_request_live_gpu_tpu_cpu_not_verified"])
        self.assertTrue(blockers["tpu_qwen_llama_stage_runtime_missing"])

    def test_public_artifacts_are_redacted(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)
        output_dir = base / "redaction"
        report = pack.build_report(pack.parse_args(self._pack_args(reports, output_dir)))

        self.assertEqual(pack.public_redaction_errors(report), [])
        scanned = "\n".join(
            path.read_text(encoding="utf-8")
            for path in output_dir.rglob("*")
            if path.is_file()
        )
        for fragment in [
            "KAGGLE_KEY=",
            "KAGGLE_USERNAME=",
            "CROWDTENSOR_MINER_TOKEN=",
            "CROWDTENSOR_OBSERVER_TOKEN=",
            "CROWDTENSOR_ADMIN_TOKEN=",
            "HF_TOKEN=",
            "Bearer ",
            "kaggle-cookies.json",
            "kaggle-web-storage-state.json",
            "operator.private.env",
            "miner.private.env",
            "miner_registry.json",
            "kernel.py",
            '"prompt":',
            '"generated_text":',
            '"generated_token_ids":',
            '"activation":',
            '"activations":',
            '"hidden_state":',
            '"logits":',
            '"kv_cache":',
            '"past_key_values":',
            '"lease_token":',
            '"idempotency_key":',
        ]:
            self.assertNotIn(fragment, scanned)

    def test_check_script_builds_and_validates_report(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)

        result = check.build_check(check.parse_args([
            "--output-dir",
            str(base / "check"),
            "--execution-mode",
            "evidence-import",
            "--tpu-real-llm-report",
            str(reports["tpu"]),
            "--gpu-full-32b-report",
            str(reports["gpu_full"]),
            "--gpu-awq-32b-report",
            str(reports["gpu_awq"]),
            "--cpu-real-llm-report",
            str(reports["cpu"]),
            "--local-e2e-mode",
            "fixture",
            "--bridge-mode",
            "fixture",
        ]))

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["gpu_tpu_cpu_heterogeneous_stage_alpha_ready"])
        self.assertTrue(result["small_medium_real_model_end_to_end_ready"])
        self.assertTrue(result["local_three_stage_real_model_e2e_ready"])
        self.assertFalse(result["same_request_live_heterogeneous_verified"])
        self.assertEqual(result["errors"], [])

    def test_check_rejects_overclaimed_live_heterogeneous_request(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)
        report = pack.build_report(pack.parse_args(self._pack_args(reports, base / "overclaim")))
        report["same_request_live_heterogeneous_verified"] = True
        report["stage_contract_smoke"]["same_request_live_heterogeneous_verified"] = True
        report["heterogeneous_32b_feasibility"]["same_request_live_heterogeneous_verified"] = True

        errors = check.validate_report(report)

        self.assertIn("same_request_live_heterogeneous_verified_must_remain_false_for_alpha", errors)
        self.assertIn("stage_contract_overclaims_live_request", errors)
        self.assertIn("feasibility_overclaims_same_request", errors)

    def test_fixture_mode_without_external_files_is_valid(self) -> None:
        output_dir = self._tmp_dir() / "fixture"

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--execution-mode",
            "fixture",
            "--local-e2e-mode",
            "fixture",
            "--bridge-mode",
            "fixture",
        ]))

        self.assertTrue(report["ok"], report)
        self.assertEqual(check.validate_report(report), [])

    def test_main_json_prints_report(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)

        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as raised:
            with contextlib.redirect_stdout(stdout):
                pack.main([
                    "--output-dir",
                    str(base / "main"),
                    "--execution-mode",
                    "evidence-import",
                    "--tpu-real-llm-report",
                    str(reports["tpu"]),
                    "--gpu-full-32b-report",
                    str(reports["gpu_full"]),
                    "--gpu-awq-32b-report",
                    str(reports["gpu_awq"]),
                    "--cpu-real-llm-report",
                    str(reports["cpu"]),
                    "--local-e2e-mode",
                    "fixture",
                    "--bridge-mode",
                    "fixture",
                    "--json",
                ])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["same_request_live_heterogeneous_verified"])

    def test_cli_wraps_heterogeneous_stage_alpha(self) -> None:
        base = self._tmp_dir()
        reports = self._fixture_reports(base)
        output_dir = base / "cli"

        with self.assertRaises(SystemExit) as raised:
            cli.main([
                "heterogeneous-stage-alpha",
                "--output-dir",
                str(output_dir),
                "--execution-mode",
                "evidence-import",
                "--tpu-real-llm-report",
                str(reports["tpu"]),
                "--gpu-full-32b-report",
                str(reports["gpu_full"]),
                "--gpu-awq-32b-report",
                str(reports["gpu_awq"]),
                "--cpu-real-llm-report",
                str(reports["cpu"]),
                "--local-e2e-mode",
                "fixture",
                "--bridge-mode",
                "fixture",
                "--json",
            ])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads((output_dir / "gpu_tpu_cpu_heterogeneous_stage_alpha_cli_summary.json").read_text(encoding="utf-8"))
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["cli_schema"], "gpu_tpu_cpu_heterogeneous_stage_alpha_cli_v1")
        self.assertTrue(payload["gpu_tpu_cpu_heterogeneous_stage_alpha_ready"])
        self.assertTrue(payload["local_three_stage_real_model_e2e_ready"])
        self.assertTrue(payload["torch_jax_torch_bridge_ready"])
        self.assertFalse(payload["same_request_live_heterogeneous_verified"])


if __name__ == "__main__":
    unittest.main()
