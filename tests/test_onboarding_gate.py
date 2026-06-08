from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "onboarding_gate.py"
SPEC = importlib.util.spec_from_file_location("onboarding_gate", SCRIPT_PATH)
onboarding_gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(onboarding_gate)


def args_for(tmp_root: Path, **overrides: object):
    args = onboarding_gate.parse_args([
        "--quick",
        "--output-dir",
        str(tmp_root / "out"),
        "--json-out",
        str(tmp_root / "onboarding.json"),
        "--base-port",
        "19400",
    ])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class FakeRunner:
    def __init__(self, *, fail_step: str = "", stdout_fragment: str = "") -> None:
        self.commands: list[list[str]] = []
        self.fail_step = fail_step
        self.stdout_fragment = stdout_fragment

    def __call__(self, command, **kwargs):  # type: ignore[no-untyped-def]
        command = [str(part) for part in command]
        self.commands.append(command)
        step = self.step_name(command)
        if self.fail_step and step == self.fail_step:
            return subprocess.CompletedProcess(
                command,
                2,
                stdout=self.stdout_fragment or "failed\n",
                stderr="boom CROWDTENSOR_ADMIN_TOKEN\n",
            )
        return subprocess.CompletedProcess(command, 0, stdout=self.stdout_for(step, command), stderr="")

    @staticmethod
    def step_name(command: list[str]) -> str:
        joined = " ".join(command)
        if "-m venv" in joined:
            return "create_venv"
        if "-m pip install" in joined:
            return "install_package"
        if command[-1:] == ["--help"] and "crowdtensor-miner" in command[0]:
            return "crowdtensor_miner_help"
        if command[-1:] == ["--help"] and "crowdtensord" in command[0]:
            return "crowdtensord_help"
        if command[-1:] == ["--help"]:
            return "crowdtensor_help"
        if any("user_friendly_inference_frontdoor_check.py" in part for part in command):
            return "user_friendly_inference_frontdoor"
        if "infer" in command and "--shareable-terminal" in command:
            return "user_infer_smoke"
        if "local-proof" in command:
            return "local_proof"
        if "home-infer" in command:
            return "home_infer"
        if "llm-infer" in command:
            return "llm_infer_mock"
        if "cpu-infer" in command:
            return "cpu_infer_beta"
        if "release-ready" in command:
            return "release_ready_smoke"
        return "unknown"

    @staticmethod
    def output_dir_after(command: list[str], flag: str = "--output-dir") -> Path | None:
        if flag not in command:
            return None
        index = command.index(flag)
        return Path(command[index + 1])

    def stdout_for(self, step: str, command: list[str]) -> str:
        if step.endswith("_help") or step in {"create_venv", "install_package"}:
            return "usage: ok\n"
        output_dir = self.output_dir_after(command)
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        if step == "user_friendly_inference_frontdoor":
            (output_dir / "user_friendly_inference_frontdoor_check.json").write_text("{}", encoding="utf-8")
            return json.dumps({
                "schema": "user_friendly_inference_frontdoor_check_v1",
                "ok": True,
                "diagnosis_codes": ["user_friendly_inference_frontdoor_check_ready"],
            })
        if step == "user_infer_smoke":
            report = {
                "schema": "crowdtensor_infer_cli_v1",
                "ok": True,
                "diagnosis_codes": ["crowdtensor_infer_ready", "user_friendly_infer_ready"],
                "inference_verdict": {
                    "schema": "crowdtensor_inference_verdict_v1",
                    "state": "completed",
                    "completed": True,
                    "answer_scope_state": "shareable-terminal-redacted",
                    "answer_visible_in_terminal": False,
                    "evidence_level": "local-cpu-loopback",
                    "executed_where": "local-cpu",
                    "gpu_state": "local-cpu-only",
                    "fresh_kaggle_gpu_verified": False,
                    "public_artifact_safe": True,
                },
                "answer_scope": {
                    "scope_state": "shareable-terminal-redacted",
                    "visible_in_terminal": False,
                    "public_artifact_safe": True,
                },
                "output_display": {
                    "terminal_display": "shareable-terminal-redacted",
                    "terminal_text_available": False,
                    "saved_artifact_display": "hash-only",
                    "public_artifact_safe": True,
                },
                "gpu_status": {
                    "state": "local-cpu-only",
                    "fresh_kaggle_gpu_attempted": False,
                    "fresh_kaggle_gpu_verified": False,
                },
                "shareable_terminal": {
                    "enabled": True,
                    "prompt_sources_redacted": True,
                    "answer_text_redacted": True,
                    "public_artifact_safe": True,
                },
                "shareable_summary": {
                    "raw_prompt_public": False,
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                },
                "safety": {
                    "raw_prompt_public": False,
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                },
            }
            (output_dir / "infer_summary.json").write_text(json.dumps(report), encoding="utf-8")
            (output_dir / "infer_summary.md").write_text(
                "- Verdict: `answer=shareable-terminal-redacted fresh_kaggle_gpu=False`\n",
                encoding="utf-8",
            )
            return "CrowdTensor infer\n  verdict: state=completed answer=shareable-terminal-redacted\n"
        if step == "local_proof":
            (output_dir / "local_proof_summary.json").write_text("{}", encoding="utf-8")
            return json.dumps({"schema": "local_proof_summary_v1", "ok": True, "diagnosis_codes": ["home_compute_ready"]})
        if step == "home_infer":
            (output_dir / "home_inference_cli_summary.json").write_text("{}", encoding="utf-8")
            return json.dumps({"schema": "home_inference_cli_v1", "ok": True, "diagnosis_codes": ["home_compute_ready"]})
        if step == "llm_infer_mock":
            (output_dir / "llm_inference_cli_summary.json").write_text("{}", encoding="utf-8")
            return json.dumps({"schema": "llm_inference_cli_v1", "ok": True, "diagnosis_codes": ["external_llm_evidence_ready"]})
        if step == "cpu_infer_beta":
            (output_dir / "cpu_inference_beta.json").write_text("{}", encoding="utf-8")
            return json.dumps({"schema": "cpu_inference_beta_v1", "ok": True, "diagnosis_codes": ["cpu_inference_beta_ready"]})
        if step == "release_ready_smoke":
            (output_dir / "release_readiness.json").write_text("{}", encoding="utf-8")
            return json.dumps({
                "schema": "release_readiness_v1",
                "ok": True,
                "release_status": {"diagnosis_codes": ["release_ready"]},
            })
        raise AssertionError(f"unexpected step {step}")


class OnboardingGateTests(unittest.TestCase):
    def test_onboarding_gate_runs_clean_venv_install_and_user_entrypoints(self) -> None:
        tmp_root = Path(self._tmp_dir())
        runner = FakeRunner()

        summary = onboarding_gate.build_onboarding_gate(args_for(tmp_root), runner=runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "onboarding_gate_v1")
        self.assertIn("onboarding_ready", summary["diagnosis_codes"])
        joined_commands = [" ".join(command) for command in runner.commands]
        self.assertTrue(any("-m venv" in command for command in joined_commands))
        self.assertTrue(any("-m pip install -e .[dev,hf]" in command for command in joined_commands))
        for fragment in [
            "crowdtensor --help",
            "crowdtensord --help",
            "crowdtensor-miner --help",
            "user_friendly_inference_frontdoor_check.py",
            "infer --prompt-stdin --shareable-terminal",
            "local-proof",
            "home-infer",
            "llm-infer --mock",
            "cpu-infer --mode local",
            "release-ready --allow-dirty",
        ]:
            self.assertTrue(any(fragment in command for command in joined_commands), fragment)
        self.assertEqual(summary["install"]["extras"], ["dev", "hf"])
        self.assertEqual(summary["install"]["command"], "python -m pip install -e .[dev,hf]")
        self.assertEqual(summary["request_count"], 1)
        self.assertEqual(summary["external_llm_request_count"], 1)
        self.assertTrue(summary["venv"]["removed"])
        self.assertTrue((tmp_root / "onboarding.json").is_file())
        persisted = json.loads((tmp_root / "onboarding.json").read_text(encoding="utf-8"))
        self.assertTrue(persisted["artifacts"]["onboarding_gate_json"]["present"])
        self.assertEqual(persisted["artifacts"]["onboarding_gate_json"]["schema"], "onboarding_gate_v1")
        self.assertEqual(
            summary["payload_summaries"]["user_friendly_inference_frontdoor"]["schema"],
            "user_friendly_inference_frontdoor_check_v1",
        )
        self.assertEqual(
            summary["payload_summaries"]["user_infer_smoke"]["schema"],
            "user_infer_smoke_validation_v1",
        )
        self.assertTrue(summary["artifacts"]["user_friendly_inference_frontdoor_check"]["present"])
        self.assertTrue(summary["artifacts"]["user_infer_smoke_summary"]["present"])
        user_infer_step = next(step for step in summary["steps"] if step["name"] == "user_infer_smoke")
        self.assertTrue(user_infer_step["stdin_provided"])
        self.assertTrue(user_infer_step["stdin_redacted"])
        self.assertTrue(user_infer_step["validation_ok"])
        self.assertEqual(user_infer_step["validation"]["gpu_state"], "local-cpu-only")
        self.assertFalse(user_infer_step["validation"]["fresh_kaggle_gpu_verified"])
        self.assertEqual(
            summary["payload_summaries"]["cpu_infer_beta"]["schema"],
            "cpu_inference_beta_v1",
        )

    def test_install_failure_short_circuits_runtime_steps(self) -> None:
        tmp_root = Path(self._tmp_dir())
        runner = FakeRunner(fail_step="install_package")

        summary = onboarding_gate.build_onboarding_gate(args_for(tmp_root), runner=runner)

        self.assertFalse(summary["ok"])
        self.assertIn("install_failed", summary["diagnosis_codes"])
        names = [step["name"] for step in summary["steps"]]
        self.assertEqual(names[:3], ["create_venv", "install_package", "crowdtensor_help"])
        skipped = [step for step in summary["steps"] if step.get("skipped")]
        self.assertTrue(skipped)
        self.assertEqual(len(runner.commands), 2)

    def test_payload_not_ok_adds_step_specific_diagnosis(self) -> None:
        tmp_root = Path(self._tmp_dir())

        class BadHomeRunner(FakeRunner):
            def stdout_for(self, step: str, command: list[str]) -> str:
                if step == "home_infer":
                    output_dir = self.output_dir_after(command)
                    assert output_dir is not None
                    output_dir.mkdir(parents=True, exist_ok=True)
                    return json.dumps({"schema": "home_inference_cli_v1", "ok": False, "diagnosis_codes": ["home_blocked"]})
                return super().stdout_for(step, command)

        summary = onboarding_gate.build_onboarding_gate(args_for(tmp_root), runner=BadHomeRunner())

        self.assertFalse(summary["ok"])
        self.assertIn("home_infer_failed", summary["diagnosis_codes"])
        home_step = next(step for step in summary["steps"] if step["name"] == "home_infer")
        self.assertEqual(home_step["error"], "payload_not_ok")
        self.assertIn("home_blocked", home_step["diagnosis_codes"])

    def test_user_infer_smoke_validation_blocks_artifact_regression(self) -> None:
        tmp_root = Path(self._tmp_dir())

        class BadInferRunner(FakeRunner):
            def stdout_for(self, step: str, command: list[str]) -> str:
                if step == "user_infer_smoke":
                    output_dir = self.output_dir_after(command)
                    assert output_dir is not None
                    output_dir.mkdir(parents=True, exist_ok=True)
                    report = {
                        "schema": "crowdtensor_infer_cli_v1",
                        "ok": True,
                        "diagnosis_codes": ["crowdtensor_infer_ready", "user_friendly_infer_ready"],
                        "inference_verdict": {
                            "schema": "crowdtensor_inference_verdict_v1",
                            "state": "completed",
                            "completed": True,
                            "answer_scope_state": "terminal-visible",
                            "answer_visible_in_terminal": True,
                            "evidence_level": "local-cpu-loopback",
                            "executed_where": "local-cpu",
                            "gpu_state": "fresh-kaggle-gpu-verified",
                            "fresh_kaggle_gpu_verified": True,
                            "public_artifact_safe": True,
                        },
                        "answer_scope": {
                            "scope_state": "terminal-visible",
                            "visible_in_terminal": True,
                            "public_artifact_safe": True,
                        },
                        "output_display": {
                            "terminal_display": "local-private",
                            "terminal_text_available": True,
                            "saved_artifact_display": "hash-only",
                            "public_artifact_safe": True,
                        },
                        "gpu_status": {
                            "state": "fresh-kaggle-gpu-verified",
                            "fresh_kaggle_gpu_attempted": True,
                            "fresh_kaggle_gpu_verified": True,
                        },
                        "shareable_terminal": {
                            "enabled": False,
                            "prompt_sources_redacted": False,
                            "answer_text_redacted": False,
                            "public_artifact_safe": False,
                        },
                        "shareable_summary": {
                            "raw_prompt_public": False,
                            "raw_generated_text_public": False,
                            "generated_token_ids_public": False,
                        },
                        "safety": {
                            "raw_prompt_public": False,
                            "raw_generated_text_public": False,
                            "generated_token_ids_public": False,
                        },
                    }
                    (output_dir / "infer_summary.json").write_text(json.dumps(report), encoding="utf-8")
                    (output_dir / "infer_summary.md").write_text(
                        onboarding_gate.USER_INFER_SMOKE_PROMPT,
                        encoding="utf-8",
                    )
                    return "CrowdTensor infer\n"
                return super().stdout_for(step, command)

        summary = onboarding_gate.build_onboarding_gate(args_for(tmp_root), runner=BadInferRunner())

        self.assertFalse(summary["ok"])
        self.assertIn("user_infer_smoke_failed", summary["diagnosis_codes"])
        user_infer_step = next(step for step in summary["steps"] if step["name"] == "user_infer_smoke")
        self.assertEqual(user_infer_step["error"], "validation_not_ok")
        self.assertFalse(user_infer_step["validation_ok"])
        self.assertIn("verdict_answer_scope_mismatch", user_infer_step["validation"]["errors"])
        self.assertIn("verdict_fresh_kaggle_gpu_mismatch", user_infer_step["validation"]["errors"])
        self.assertTrue(any(error.startswith("artifact_leaked_") for error in user_infer_step["validation"]["errors"]))

    def test_secret_like_failure_output_is_redacted(self) -> None:
        tmp_root = Path(self._tmp_dir())
        runner = FakeRunner(fail_step="crowdtensor_help", stdout_fragment="CROWDTENSOR_ADMIN_TOKEN leaked\n")

        summary = onboarding_gate.build_onboarding_gate(args_for(tmp_root), runner=runner)

        encoded = json.dumps(summary, sort_keys=True)
        self.assertNotIn("CROWDTENSOR_ADMIN_TOKEN", encoded)
        self.assertIn("<redacted>", encoded)

    def _tmp_dir(self) -> str:
        path = Path(self.id().replace(".", "_").replace("/", "_"))
        tmp_root = Path("/tmp") / f"crowdtensor_onboarding_gate_{path.name}"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(tmp_root, ignore_errors=True))
        return str(tmp_root)


if __name__ == "__main__":
    unittest.main()
