from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import public_swarm_inference_beta_rc_check as check
from scripts import public_swarm_inference_beta_rc_pack as pack


class PublicSwarmInferenceBetaRcPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_public_swarm_beta_rc_test_"))

    def test_check_builds_ready_local_loopback_rc_with_fake_runner(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "local-loopback",
            "--output-dir",
            str(self._tmp_dir()),
            "--max-new-tokens",
            "2",
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "public_swarm_inference_beta_rc_check_v1")

    def test_check_builds_ready_kaggle_package_rc_with_fake_runner(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "package",
            "--target",
            "kaggle",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_external_existing_requires_auth(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args([
                "external-existing",
                "--coordinator-url",
                "http://127.0.0.1:9999",
            ])

    def test_common_report_blocks_when_generate_loop_missing(self) -> None:
        output_dir = self._tmp_dir()

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            return check.fake_runner(command, **_)

        args = pack.parse_args([
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "2",
        ])
        product_step, product_payload = pack.run_product_beta(args, output_dir=output_dir / "product-beta", runner=fake_runner)
        p2p_step, p2p_payload = pack.run_p2p_route(args, output_dir=output_dir / "p2p-lite", runner=fake_runner)
        cpu_step, cpu_payload = pack.run_cpu_fallback(args, output_dir=output_dir / "cpu-fallback", runner=fake_runner)

        report = pack.build_common_report(
            args,
            output_dir=output_dir,
            product_step=product_step,
            product_payload=product_payload,
            p2p_step=p2p_step,
            p2p_payload=p2p_payload,
            cpu_step=cpu_step,
            cpu_payload=cpu_payload,
            mode_body={"ok": False, "diagnosis_codes": ["generation_timeout"]},
            mode_steps=[{"name": "serve_join_generate_loop", "ok": False}],
        )

        self.assertFalse(report["ok"], report)
        self.assertIn("public_swarm_inference_beta_rc_blocked", report["diagnosis_codes"])
        self.assertIn("generation_timeout", report["diagnosis_codes"])

    def test_common_report_rejects_batch_stream_ready_codes_without_per_request_stream_evidence(self) -> None:
        output_dir = self._tmp_dir()

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            return check.fake_runner(command, **_)

        args = pack.parse_args([
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--prompt-texts",
            "first prompt,second prompt",
            "--stream-generation",
            "--max-new-tokens",
            "2",
        ])
        product_step, product_payload = pack.run_product_beta(args, output_dir=output_dir / "product-beta", runner=fake_runner)
        p2p_step, p2p_payload = pack.run_p2p_route(args, output_dir=output_dir / "p2p-lite", runner=fake_runner)
        cpu_step, cpu_payload = pack.run_cpu_fallback(args, output_dir=output_dir / "cpu-fallback", runner=fake_runner)

        report = pack.build_common_report(
            args,
            output_dir=output_dir,
            product_step=product_step,
            product_payload=product_payload,
            p2p_step=p2p_step,
            p2p_payload=p2p_payload,
            cpu_step=cpu_step,
            cpu_payload=cpu_payload,
            mode_body={
                "ok": True,
                "diagnosis_codes": [
                    "serve_join_generate_loop_ready",
                    "remote_generate_session_ready",
                    "public_swarm_generate_ready",
                    "public_swarm_generate_batch_ready",
                    "public_swarm_generate_stream_ready",
                    "public_swarm_generate_stream_endpoint_ready",
                ],
                "generation": {
                    "batch": {
                        "enabled": True,
                        "request_count": 2,
                        "results": [
                            {
                                "request_id": "req-1",
                                "prompt_hash": "sha256:a",
                                "generated_token_count": 2,
                                "max_new_tokens": 2,
                                "generated_text_hash": "sha256:g1",
                                "multi_token_generation_ready": True,
                            },
                            {
                                "request_id": "req-2",
                                "prompt_hash": "sha256:b",
                                "generated_token_count": 2,
                                "max_new_tokens": 2,
                                "generated_text_hash": "sha256:g2",
                                "multi_token_generation_ready": True,
                            },
                        ],
                        "batch_generation_ready": True,
                    },
                    "stream": {
                        "enabled": True,
                        "requested": True,
                        "endpoint_ready": True,
                        "stream_generation_ready": True,
                        "progress": {
                            "stream_progress_complete": True,
                            "monotonic_progress": True,
                            "expected_request_count": 2,
                            "per_request_progress_complete": False,
                            "per_request_monotonic_progress": False,
                        },
                    },
                },
            },
            mode_steps=[{"name": "serve_join_generate_loop", "ok": True}],
        )

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["rc"]["mode_ready"])
        self.assertTrue(report["rc"]["batch"]["batch_generation_ready"])
        self.assertFalse(report["rc"]["stream"]["stream_generation_ready"])
        self.assertIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_generate_stream_endpoint_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_inference_beta_rc_blocked", report["diagnosis_codes"])

    def test_generate_existing_forwards_bounded_prompt_batch(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []
        args = pack.parse_args([
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            "http://127.0.0.1:9999",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--prompt-texts",
            "first prompt,second prompt",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({
                    "schema": pack.PRODUCT_CLI_SCHEMA,
                    "ok": True,
                    "batch": {
                        "enabled": True,
                        "request_count": 2,
                        "prompt_hashes": ["sha256:a", "sha256:b"],
                        "prompt_char_counts": [12, 13],
                        "results": [
                            {
                                "request_id": "req-1",
                                "prompt_hash": "sha256:a",
                                "generated_token_count": 2,
                                "max_new_tokens": 2,
                                "generated_text_hash": "sha256:g1",
                                "multi_token_generation_ready": True,
                            },
                            {
                                "request_id": "req-2",
                                "prompt_hash": "sha256:b",
                                "generated_token_count": 2,
                                "max_new_tokens": 2,
                                "generated_text_hash": "sha256:g2",
                                "multi_token_generation_ready": True,
                            },
                        ],
                        "batch_generation_ready": True,
                    },
                    "diagnosis_codes": ["public_swarm_generate_ready", "public_swarm_generate_batch_ready"],
                }) + "\n",
                stderr="",
            )

        step, payload = pack.run_existing_generate(args, output_dir=output_dir, runner=fake_runner)

        self.assertTrue(step["ok"], step)
        self.assertTrue(pack.safe_batch_summary(payload)["batch_generation_ready"])
        self.assertTrue(calls)

    def test_generate_existing_rejects_batch_with_duplicate_request_identity(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            "http://127.0.0.1:9999",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--prompt-texts",
            "first prompt,second prompt",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({
                    "schema": pack.PRODUCT_CLI_SCHEMA,
                    "ok": True,
                    "batch": {
                        "enabled": True,
                        "request_count": 2,
                        "results": [
                            {
                                "request_id": "req-1",
                                "prompt_hash": "sha256:a",
                                "generated_token_count": 2,
                                "max_new_tokens": 2,
                                "generated_text_hash": "sha256:g1",
                                "multi_token_generation_ready": True,
                            },
                            {
                                "request_id": "req-1",
                                "prompt_hash": "sha256:a",
                                "generated_token_count": 2,
                                "max_new_tokens": 2,
                                "generated_text_hash": "sha256:g1-dup",
                                "multi_token_generation_ready": True,
                            },
                        ],
                        "batch_generation_ready": True,
                    },
                    "diagnosis_codes": ["public_swarm_generate_ready", "public_swarm_generate_batch_ready"],
                }) + "\n",
                stderr="",
            )

        step, payload = pack.run_existing_generate(args, output_dir=output_dir, runner=fake_runner)
        batch = pack.safe_batch_summary(payload)

        self.assertTrue(step["ok"], step)
        self.assertFalse(batch["batch_identity_ready"])
        self.assertFalse(batch["batch_generation_ready"])

    def test_generate_existing_forwards_and_preserves_stream_generation(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []
        args = pack.parse_args([
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            "http://127.0.0.1:9999",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--stream-generation",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("--stream", command)
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({
                    "schema": pack.PRODUCT_CLI_SCHEMA,
                    "ok": True,
                    "stream": {
                        "enabled": True,
                        "requested": True,
                        "event_count": 2,
                        "source": "admin-session-stream",
                        "endpoint_ready": True,
                        "progress": {
                            "stream_progress_complete": True,
                            "all_token_events_ready": True,
                            "monotonic_progress": True,
                            "observed_token_counts": [1, 2],
                            "max_observed_token_count": 2,
                            "max_new_tokens": 2,
                            "source": "admin-session-stream",
                        },
                        "events": [
                            {
                                "schema": "session_stream_event_v1",
                                "session_id": "session-1",
                                "task_id": "task-1",
                                "miner_id": "stage1",
                                "stage_id": 1,
                                "generated_token_count": 2,
                                "max_new_tokens": 2,
                                "generation_step": 1,
                                "generated_text_hash": "sha256:stream",
                                "decoded_tokens_match": True,
                                "raw_generated_text_public": False,
                                "generated_token_ids_public": False,
                            }
                        ],
                        "stream_generation_ready": True,
                    },
                    "diagnosis_codes": [
                        "public_swarm_generate_ready",
                        "public_swarm_generate_stream_ready",
                        "public_swarm_generate_stream_endpoint_ready",
                    ],
                }) + "\n",
                stderr="",
            )

        step, payload = pack.run_existing_generate(args, output_dir=output_dir, runner=fake_runner)

        self.assertTrue(step["ok"], step)
        self.assertTrue(pack.safe_stream_summary(payload)["stream_generation_ready"])
        self.assertTrue(calls)

    def test_common_report_preserves_stream_readiness(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--stream-generation",
        ])
        report = pack.build_common_report(
            args,
            output_dir=output_dir,
            product_step={"name": "product", "ok": True},
            product_payload={
                "schema": pack.BETA_SCHEMA,
                "ok": True,
                "diagnosis_codes": ["public_swarm_product_beta_ready"],
            },
            p2p_step={"name": "p2p", "ok": True},
            p2p_payload={
                "schema": pack.P2P_CHECK_SCHEMA,
                "ok": True,
                "diagnosis_codes": ["p2p_lite_discovery_ready"],
            },
            cpu_step={"name": "cpu", "ok": True},
            cpu_payload={
                "schema": pack.CPU_BETA_SCHEMA,
                "ok": True,
                "diagnosis_codes": ["cpu_inference_beta_ready"],
            },
            mode_body={
                "ok": True,
                "generation": {
                    "stream": {
                        "enabled": True,
                        "requested": True,
                        "endpoint_ready": True,
                        "stream_generation_ready": True,
                        "progress": {
                            "stream_progress_complete": True,
                            "all_token_events_ready": True,
                            "monotonic_progress": True,
                            "observed_token_counts": [1, 2],
                            "max_observed_token_count": 2,
                            "max_new_tokens": 2,
                        },
                    }
                },
                "diagnosis_codes": ["public_swarm_generate_ready"],
            },
            mode_steps=[{"name": "serve_join_generate_loop", "ok": True}],
        )

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["rc"]["stream"]["stream_generation_ready"])
        self.assertIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_endpoint_ready", report["diagnosis_codes"])

    def test_local_generate_loop_uses_generation_steps_as_stage_max_tasks_for_batch(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "3",
            "--prompt-texts",
            "first prompt,second prompt",
            "--stream-generation",
        ])
        popen_commands: list[list[str]] = []

        class FakeProc:
            def __init__(self, command: list[str]) -> None:
                self.command = command
                self.returncode = None
                self.stdout = None
                self.stderr = None

            def poll(self) -> int | None:
                return self.returncode

            def wait(self, timeout: float | None = None) -> int:
                self.returncode = 0
                return 0

            def terminate(self) -> None:
                self.returncode = 0

            def kill(self) -> None:
                self.returncode = -9

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                self.returncode = 0 if self.returncode is None else self.returncode
                return "", ""

        def fake_popen(command: list[str], **_: object) -> FakeProc:
            popen_commands.append(command)
            return FakeProc(command)

        generate_payload = {
            "schema": pack.PRODUCT_CLI_SCHEMA,
            "ok": True,
            "generation": {
                "generated_token_count": 3,
                "max_new_tokens": 3,
                "multi_token_generation_ready": True,
                "batch_generation_ready": True,
                "request_count": 2,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            },
            "batch": {
                "enabled": True,
                "expected_request_count": 2,
                "request_count": 2,
                "results": [
                    {
                        "request_id": "req-1",
                        "prompt_hash": "sha256:p1",
                        "generated_token_count": 3,
                        "max_new_tokens": 3,
                        "generated_text_hash": "sha256:g1",
                        "multi_token_generation_ready": True,
                    },
                    {
                        "request_id": "req-2",
                        "prompt_hash": "sha256:p2",
                        "generated_token_count": 3,
                        "max_new_tokens": 3,
                        "generated_text_hash": "sha256:g2",
                        "multi_token_generation_ready": True,
                    },
                ],
                "batch_generation_ready": True,
            },
            "stream": {
                "enabled": True,
                "requested": True,
                "endpoint_ready": True,
                "stream_generation_ready": True,
                "event_count": 6,
                "progress": {
                    "stream_progress_complete": True,
                    "all_token_events_ready": True,
                    "monotonic_progress": True,
                    "expected_request_count": 2,
                    "per_request_progress": [
                        {
                            "request_key": "req-1",
                            "request_id": "req-1",
                            "prompt_hash": "sha256:p1",
                            "event_count": 3,
                            "observed_token_counts": [1, 2, 3],
                            "max_observed_token_count": 3,
                            "target_token_count": 3,
                            "monotonic_progress": True,
                            "stream_progress_complete": True,
                        },
                        {
                            "request_key": "req-2",
                            "request_id": "req-2",
                            "prompt_hash": "sha256:p2",
                            "event_count": 3,
                            "observed_token_counts": [1, 2, 3],
                            "max_observed_token_count": 3,
                            "target_token_count": 3,
                            "monotonic_progress": True,
                            "stream_progress_complete": True,
                        },
                    ],
                    "per_request_progress_complete": True,
                    "per_request_monotonic_progress": True,
                    "observed_token_counts": [1, 2, 3],
                    "max_observed_token_count": 3,
                    "max_new_tokens": 3,
                },
            },
            "diagnosis_codes": [
                "public_swarm_generate_ready",
                "public_swarm_generate_batch_ready",
                "public_swarm_generate_stream_ready",
                "public_swarm_generate_stream_endpoint_ready",
            ],
        }

        with (
            patch.object(pack.subprocess, "Popen", side_effect=fake_popen),
            patch.object(pack.subprocess, "run", return_value=subprocess.CompletedProcess(
                args=["generate"],
                returncode=0,
                stdout=json.dumps(generate_payload) + "\n",
                stderr="",
            )),
            patch.object(pack, "wait_health", return_value={"ok": True}),
            patch.object(pack, "missing_hf_dependencies", return_value=[]),
        ):
            report = pack.run_product_generate_loop(args, output_dir=output_dir)

        self.assertTrue(report["ok"], report)
        join_commands = [command for command in popen_commands if "join" in command]
        self.assertEqual(len(join_commands), 2)
        for command in join_commands:
            self.assertEqual(command[command.index("--max-tasks") + 1], "3")
        self.assertIn("serve_join_generate_loop_ready", report["diagnosis_codes"])

    def test_prompt_batch_rejects_more_than_four_prompts(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args([
                "local-loopback",
                "--prompt-texts",
                "one,two,three,four,five",
            ])

    def test_report_redacts_secret_fragments(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args(["package", "--output-dir", str(output_dir)])
        report = pack.persist_report(
            {
                "schema": pack.SCHEMA,
                "ok": True,
                "mode": "package",
                "diagnosis_codes": ["public_swarm_inference_beta_rc_ready"],
                "rc": {"ready": True},
                "steps": [],
                "limitations": [],
                "safety": {},
                "secret": "admin-secret",
            },
            output_dir=output_dir,
            secret_values=["admin-secret"],
        )

        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("admin-secret", encoded)


if __name__ == "__main__":
    unittest.main()
