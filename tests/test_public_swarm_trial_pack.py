from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import scripts.public_swarm_trial_pack as pack


def completed(payload: dict, *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class PublicSwarmTrialPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_swarm_trial_test_"))

    def _base_args(self, output_dir: Path, *extra: str) -> list[str]:
        gpu_report = output_dir / "gpu_report.json"
        write_json(gpu_report, self._gpu_payload())
        stage0 = output_dir / "stage0.json"
        stage1 = output_dir / "stage1.json"
        write_json(stage0, self._operator_payload("live-kaggle"))
        write_json(stage1, self._operator_payload("live-kaggle"))
        return [
            "--output-dir",
            str(output_dir / "trial"),
            "--gpu-report",
            str(gpu_report),
            "--live-stage0-report",
            str(stage0),
            "--live-stage1-report",
            str(stage1),
            "--release-readiness-report",
            str(output_dir / "missing_release.json"),
            "--kaggle-owner",
            "owner",
            "--timeout-seconds",
            "1",
            "--remote-timeout-seconds",
            "1",
            "--cpu-timeout-seconds",
            "1",
            "--release-timeout-seconds",
            "1",
            *extra,
        ]

    def _product_payload(self, *, ok: bool = True, hf_missing: bool = False) -> dict:
        codes = [
            "cpu_fallback_ready",
            "local_cpu_inference_ready",
            "read_only_workload",
            "not_production",
            "not_p2p",
            "not_large_model_serving",
        ]
        if ok:
            codes.extend([
                "public_swarm_product_beta_ready",
                "public_swarm_product_beta_user_path_ready",
                "serve_ready",
                "stage0_join_ready",
                "stage1_join_ready",
                "generate_ready",
                "public_swarm_generate_ready",
                "private_artifacts_cleaned",
            ])
        if hf_missing:
            codes.append("hf_dependencies_missing")
        return {
            "schema": pack.PRODUCT_BETA_SCHEMA,
            "ok": ok,
            "mode": "local-loopback",
            "product_beta": {"ready": ok, "cpu_fallback_ready": True, "workload_type": pack.WORKLOAD_TYPE},
            "generation": {"generated_token_count": 2},
            "diagnosis_codes": codes,
        }

    def _operator_payload(self, mode: str = "local-smoke", *, degraded: bool = False) -> dict:
        codes = [
            "public_swarm_operator_preview_ready",
            "operator_preview_user_path_ready",
            "cpu_fallback_ready",
            "live_preview_ready",
            "release_readiness_ready",
            "support_bundle_ready",
            "read_only_workload",
            "not_production",
            "not_p2p",
            "not_large_model_serving",
        ]
        if degraded:
            codes.extend(["developer_preview_degraded", "operator_preview_cpu_fallback_user_path_ready"])
        elif mode == "evidence-import":
            codes.extend(["operator_preview_evidence_import_ready", "operator_preview_retained_evidence_ready", "token_rotation_required"])
        elif mode == "package":
            codes.extend(["operator_preview_package_ready", "miner_join_pack_ready", "private_artifacts_local_only"])
        elif mode == "live-kaggle":
            codes.extend(["operator_preview_live_kaggle_ready", "external_runtime_verified", "kaggle_kernels_deleted", "token_rotation_required"])
        else:
            codes.extend(["operator_preview_local_smoke_ready", "serve_join_generate_ready", "private_artifacts_cleaned"])
        return {
            "schema": pack.OPERATOR_PREVIEW_SCHEMA,
            "ok": True,
            "mode": mode,
            "operator_preview": {
                "ready": True,
                "cpu_fallback_ready": True,
                "live_preview_ready": True,
                "release_readiness_ready": True,
                "support_bundle_ready": True,
                "user_path_ready": True,
                "degraded": degraded,
            },
            "diagnosis_codes": codes,
        }

    def _gpu_payload(self) -> dict:
        return {
            "schema": pack.GPU_GENERATION_SCHEMA,
            "ok": True,
            "mode": "evidence-import",
            "generation": {"generated_token_count": 16, "multi_token_generation_ready": True, "raw_generated_text_public": False},
            "diagnosis_codes": [
                "gpu_sharded_generation_ready",
                "multi_token_generation_ready",
                "gpu_generation_evidence_import_ready",
                "external_gpu_runtime_verified",
                "token_rotation_required",
                "read_only_workload",
                "not_production",
                "not_p2p",
            ],
        }

    def test_local_loopback_reports_real_serve_join_generate_ready(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            if "public_swarm_product_beta_pack.py" in command[1]:
                return completed(self._product_payload())
            if "public_swarm_operator_preview_pack.py" in command[1]:
                return completed(self._operator_payload("local-smoke"))
            if "gpu_sharded_generation_beta_pack.py" in command[1]:
                return completed(self._gpu_payload())
            raise AssertionError(command)

        args = pack.parse_args(["local-loopback", *self._base_args(output_dir)])
        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["trial"]["serve_join_generate_trial_ready"])
        self.assertTrue(report["trial"]["stage0_join_ready"])
        self.assertTrue(report["trial"]["stage1_join_ready"])
        self.assertTrue(report["trial"]["generated_token_count_ready"])
        self.assertIn("serve_join_generate_trial_ready", report["diagnosis_codes"])
        self.assertTrue(calls)
        saved = json.loads((output_dir / "trial" / "public_swarm_trial.json").read_text(encoding="utf-8"))
        self.assertFalse(saved["output_request"]["include_output"])
        self.assertFalse(saved["output_request"]["raw_prompt_public"])
        self.assertFalse(saved["output_request"]["raw_generated_text_public"])
        self.assertFalse(saved["output_request"]["generated_token_ids_public"])
        self.assertTrue(saved["output_request"]["public_artifact_safe"])
        self.assertEqual(saved["answer_scope"]["scope_state"], "no-local-answer")
        self.assertFalse(saved["answer_scope"]["visible_in_terminal"])
        self.assertFalse(saved["answer_scope"]["terminal_only"])
        self.assertEqual(saved["answer_scope"]["saved_json_display"], "hash-only")
        self.assertEqual(saved["answer_scope"]["saved_markdown_display"], "hash-only")
        self.assertTrue(saved["answer_scope"]["public_artifact_safe"])
        self.assertTrue(saved["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(saved["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(saved["shareable_summary"]["raw_generated_text_public"])
        self.assertFalse(saved["shareable_summary"]["generated_token_ids_public"])
        self.assertEqual(saved["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertFalse(saved["shareable_summary"]["local_answer_terminal_only"])
        markdown = (output_dir / "trial" / "public_swarm_trial.md").read_text(encoding="utf-8")
        self.assertIn("## Output Scope", markdown)
        self.assertIn("- answer scope: `no-local-answer`", markdown)
        self.assertIn(
            "- shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False answer_scope_state=no-local-answer local_answer_terminal_only=False`",
            markdown,
        )
        support = json.loads((output_dir / "trial" / "support_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(support["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(support["shareable_summary"]["answer_scope_state"], "no-local-answer")

    def test_local_loopback_degrades_to_cpu_fallback_when_hf_missing(self) -> None:
        output_dir = self._tmp_dir()

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if "public_swarm_product_beta_pack.py" in command[1]:
                return completed(self._product_payload(ok=False, hf_missing=True), returncode=1)
            if "public_swarm_operator_preview_pack.py" in command[1]:
                return completed(self._operator_payload("local-smoke", degraded=True))
            if "gpu_sharded_generation_beta_pack.py" in command[1]:
                return completed(self._gpu_payload())
            raise AssertionError(command)

        args = pack.parse_args(["local-loopback", *self._base_args(output_dir)])
        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertFalse(report["trial"]["serve_join_generate_trial_ready"])
        self.assertTrue(report["trial"]["degraded_cpu_fallback_ready"])
        self.assertIn("hf_dependencies_missing", report["diagnosis_codes"])
        self.assertIn("swarm_trial_degraded_cpu_fallback_ready", report["diagnosis_codes"])

    def test_evidence_import_uses_retained_reports_without_sensitive_fragments(self) -> None:
        output_dir = self._tmp_dir()
        product_report = output_dir / "product.json"
        operator_report = output_dir / "operator.json"
        write_json(product_report, self._product_payload())
        write_json(operator_report, self._operator_payload("evidence-import"))
        args = pack.parse_args([
            "evidence-import",
            *self._base_args(output_dir, "--product-beta-report", str(product_report), "--operator-preview-report", str(operator_report)),
        ])

        report = pack.build_report(args)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertIn("public_swarm_trial_evidence_import_ready", report["diagnosis_codes"])
        self.assertIn("gpu_generation_evidence_import_ready", report["diagnosis_codes"])
        for fragment in pack.SECRET_FRAGMENTS:
            self.assertNotIn(fragment, encoded)
        self.assertFalse(report["output_request"]["include_output"])
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")


if __name__ == "__main__":
    unittest.main()
