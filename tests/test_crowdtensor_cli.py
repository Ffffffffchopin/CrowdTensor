from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

from crowdtensor import cli


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


class CrowdTensorCliTests(unittest.TestCase):
    def _tmp_dir(self) -> str:
        return tempfile.mkdtemp(prefix="crowdtensor_cli_test_")

    def _cleanup_args(self, *extra: str) -> object:
        return cli.parse_args(["clean-artifacts", *extra])

    def test_wait_progress_text_adds_batch_request_progress_only_for_batches(self) -> None:
        single = {
            "poll_count": 2,
            "accepted_rows_seen": 1,
            "max_observed_token_count": 4,
            "target_token_count": 4,
            "expected_request_count": 1,
            "observed_request_count": 1,
            "batch_generation_ready": True,
            "ledger_endpoint_ready": True,
            "stream_endpoint_ready": False,
        }
        batch = dict(single, expected_request_count=2, observed_request_count=1, batch_generation_ready=False)
        with_error = dict(single, last_error_type="HTTPError", last_error_detail="must not leak")

        self.assertEqual(
            cli.wait_progress_text(single),
            "polls=2 accepted_rows=1 tokens=4/4 ledger=True stream=False",
        )
        self.assertEqual(
            cli.wait_progress_text(batch),
            "polls=2 accepted_rows=1 tokens=4/4 requests=1/2 batch_ready=False ledger=True stream=False",
        )
        self.assertEqual(
            cli.wait_progress_text(with_error),
            "polls=2 accepted_rows=1 tokens=4/4 ledger=True stream=False last_error=HTTPError",
        )
        self.assertNotIn("must not leak", cli.wait_progress_text(with_error))

    def test_infer_result_text_distinguishes_terminal_private_from_saved_artifacts(self) -> None:
        rendered = cli.infer_result_text({
            "status": "complete",
            "generated_token_count": 2,
            "max_new_tokens": 2,
            "output_count": 1,
            "display": "local-private",
            "generated_text_hash": "sha256:answer",
            "public_artifact_safe": False,
        })

        self.assertIn("display=local-private", rendered)
        self.assertIn("terminal_private=True", rendered)
        self.assertIn("saved_public_artifact_safe=True", rendered)
        self.assertNotIn("public_artifact_safe=False", rendered)

    def test_stream_progress_lines_renders_single_and_batch_progress(self) -> None:
        single = {
            "observed_token_counts": [1, 2],
            "max_observed_token_count": 2,
            "target_token_count": 2,
            "stream_progress_complete": True,
            "per_request_progress": [{"request_key": "req-single"}],
        }
        batch = {
            "target_token_count": 2,
            "expected_request_count": 3,
            "per_request_progress": [
                {
                    "request_id": "req-1",
                    "observed_token_counts": [1, 2],
                    "max_observed_token_count": 2,
                    "target_token_count": 2,
                    "stream_progress_complete": True,
                },
                {
                    "request_id": "req-2",
                    "observed_token_counts": [1],
                    "max_observed_token_count": 1,
                    "target_token_count": 2,
                    "stream_progress_complete": False,
                },
            ]
        }

        self.assertEqual(
            cli.stream_progress_lines(single),
            ["  stream_progress: request=req-single tokens=2/2 counts=[1, 2] complete=True"],
        )
        self.assertEqual(
            cli.stream_progress_lines(single, prefix="- Stream request", single_prefix="- Stream progress"),
            ["- Stream progress: request=req-single tokens=2/2 counts=[1, 2] complete=True"],
        )
        self.assertEqual(
            cli.stream_progress_lines(batch),
            [
                "  stream[1]: request=req-1 tokens=2/2 counts=[1, 2] complete=True missing=False",
                "  stream[2]: request=req-2 tokens=1/2 counts=[1] complete=False missing=False",
                "  stream[3]: request=missing tokens=0/2 counts=[] complete=False missing=True",
            ],
        )
        self.assertEqual(
            cli.stream_progress_lines(batch, prefix="- Stream request", single_prefix="- Stream progress"),
            [
                "- Stream request[1]: request=req-1 tokens=2/2 counts=[1, 2] complete=True missing=False",
                "- Stream request[2]: request=req-2 tokens=1/2 counts=[1] complete=False missing=False",
                "- Stream request[3]: request=missing tokens=0/2 counts=[] complete=False missing=True",
            ],
        )
        markdown = cli.render_infer_summary_markdown({
            "ok": True,
            "mode": "local",
            "stream": {
                "enabled": True,
                "ready": True,
                "event_count": 3,
                "source": "admin-session-stream",
                "progress": batch,
            },
            "diagnosis_codes": ["public_swarm_generate_ready"],
        })
        self.assertIn(
            "- Stream request[1]: request=req-1 tokens=2/2 counts=[1, 2] complete=True missing=False",
            markdown,
        )
        self.assertIn(
            "- Stream request[2]: request=req-2 tokens=1/2 counts=[1] complete=False missing=False",
            markdown,
        )
        self.assertIn(
            "- Stream request[3]: request=missing tokens=0/2 counts=[] complete=False missing=True",
            markdown,
        )

        self.assertEqual(
            cli.stream_progress_issue_summary(batch),
            "missing_requests=1/3 request[2]=req-2:1/2 request[3]=missing",
        )
        self.assertEqual(
            cli.infer_trace_request_lines(
                {
                    "request_trace": [
                        {
                            "request_id": "req-1",
                            "prompt_hash": "sha256:p1",
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g1",
                            "source": "generation-results",
                        },
                        {
                            "request_id": "",
                            "prompt_hash": "sha256:abcdef1234567890zz",
                            "generated_token_count": 1,
                            "max_new_tokens": 2,
                            "generated_text_hash": None,
                            "source": "stream-progress",
                        },
                    ]
                }
            ),
            [
                "  trace_request[1]: request=req-1 tokens=2/2 hash=sha256:g1 source=generation-results",
                "  trace_request[2]: request=sha256:abcdef1234567890z tokens=1/2 hash=none source=stream-progress",
            ],
        )
        self.assertEqual(
            cli.stream_request_label({"prompt_hash": "sha256:abcdef1234567890zz"}),
            "sha256:abcdef1234567890z",
        )
        self.assertEqual(cli.stream_request_label({"request_key": "<redacted>"}), "unknown")

    def test_prompt_scope_marks_file_and_stdin_as_shareable_log_friendly(self) -> None:
        file_scope = cli.prompt_scope_from_args(
            argparse.Namespace(
                prompt_texts_file="",
                prompt_file="/tmp/private-prompt.txt",
                prompt_stdin=False,
                prompt_texts="",
                prompt_text="",
            ),
            prompt_count=1,
        )
        stdin_scope = cli.prompt_scope_from_args(
            argparse.Namespace(
                prompt_texts_file="",
                prompt_file="",
                prompt_stdin=True,
                prompt_texts="",
                prompt_text="",
            ),
            prompt_count=1,
        )

        self.assertEqual(file_scope["source"], "prompt-file")
        self.assertFalse(file_scope["inline_prompt_text"])
        self.assertTrue(file_scope["terminal_next_commands_local_private"])
        self.assertTrue(file_scope["terminal_logs_local_private"])
        self.assertTrue(file_scope["terminal_local_paths"])
        self.assertTrue(file_scope["saved_artifacts_prompt_placeholders"])
        self.assertTrue(file_scope["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertFalse(file_scope["prompt_file_path_public"])
        self.assertFalse(file_scope["raw_prompt_public"])
        self.assertTrue(file_scope["public_artifact_safe"])
        self.assertEqual(stdin_scope["source"], "prompt-stdin")
        self.assertFalse(stdin_scope["terminal_local_paths"])
        self.assertTrue(stdin_scope["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertFalse(stdin_scope["prompt_file_path_public"])
        self.assertFalse(stdin_scope["raw_prompt_public"])

    def test_local_proof_success_summarizes_steps_and_artifacts(self) -> None:
        calls: list[list[str]] = []
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            if "doctor.py" in command[1]:
                return completed({"ok": True, "summary": {"errors": 0}})
            if "runtime_matrix.py" in command[1]:
                return completed({"ok": True, "diagnosis_summary": {"codes": ["cpu_baseline_ready"]}})
            if "home_compute_demo.py" in command[1]:
                return completed({"ok": True, "diagnosis_codes": ["home_compute_ready"]})
            if "demo_manifest_pack.py" in command[1]:
                (output_dir / "demo_manifest.json").write_text("{}", encoding="utf-8")
                (output_dir / "demo_manifest.md").write_text("# Demo\n", encoding="utf-8")
                return completed({"ok": True, "schema": "demo_manifest_v1"})
            raise AssertionError(command)

        args = cli.parse_args([
            "local-proof",
            "--output-dir",
            str(output_dir),
            "--base-port",
            "9000",
            "--request-count",
            "4",
        ])

        summary = cli.build_local_proof(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "local_proof_summary_v1")
        self.assertEqual([step["name"] for step in summary["steps"]], [
            "doctor",
            "runtime_matrix",
            "home_compute_demo",
            "demo_manifest",
        ])
        self.assertEqual(summary["diagnosis_codes"], ["cpu_baseline_ready", "home_compute_ready"])
        self.assertTrue(summary["artifacts"]["demo_manifest_json"]["present"])
        self.assertTrue((output_dir / "local_proof_summary.json").is_file())
        self.assertTrue(any("demo_manifest_pack.py" in command[1] for command in calls))

    def test_infer_help_shows_user_examples_and_boundaries(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            cli.parse_args(["infer", "--help"])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        self.assertIn("Run the shortest user-facing CrowdTensor inference path.", rendered)
        self.assertIn("examples:", rendered)
        self.assertIn('crowdtensor infer "your prompt" --max-new-tokens 8 --stream', rendered)
        self.assertIn("short safe stderr start hint", rendered)
        self.assertIn("--json keeps stdout machine-readable", rendered)
        self.assertIn("Start with the review/review_summary line and inspect_first", rendered)
        self.assertIn("first Markdown artifact to open", rendered)
        self.assertEqual(rendered.count("Start with the review/review_summary line"), 1)
        self.assertIn("status/user_status line then", rendered)
        self.assertIn("preflight-ready", rendered)
        self.assertIn("preflight-partial", rendered)
        self.assertIn("Reports include action, recommended_next, and next[...] lines", rendered)
        self.assertIn("auto-select an available loopback Coordinator port", rendered)
        self.assertIn("--coordinator-port only when you need a reproducible fixed local port", rendered)
        self.assertIn("ready_to_submit labels mean", rendered)
        self.assertIn("ready_to_submit.next_step is the script-friendly", rendered)
        self.assertIn("stage_preflight_unknown means rerun the stage preflight", rendered)
        self.assertIn("stage_preflight_not_checked means fix route/Coordinator, then rerun with observer token", rendered)
        self.assertIn("partial can submit but still needs", rendered)
        self.assertIn("printed", rendered)
        self.assertIn("follow-up preflight", rendered)
        self.assertIn("Use one prompt source at a time: positional prompt, --prompt-text/--prompt", rendered)
        self.assertIn("--prompt-file for a UTF-8 single prompt file", rendered)
        self.assertIn("--prompt-stdin for an explicit", rendered)
        self.assertIn("--prompt-texts-file for one prompt per line", rendered)
        self.assertIn("crowdtensor infer --prompt-file prompt.txt --max-new-tokens 8", rendered)
        self.assertIn('echo "your prompt" | crowdtensor infer --prompt-stdin --max-new-tokens 8', rendered)
        self.assertIn("crowdtensor infer --prompt-texts-file prompts.txt --max-new-tokens 8 --stream", rendered)
        self.assertIn("ambiguous mixes are rejected", rendered)
        self.assertIn("output_request.include_output", rendered)
        self.assertIn("output_request.raw_generated_text_public false", rendered)
        self.assertIn("Reports also include prompt_scope", rendered)
        self.assertIn("source, prompt_count", rendered)
        self.assertIn("terminal next commands", rendered)
        self.assertIn("terminal_local_paths", rendered)
        self.assertIn("saved artifacts use placeholders", rendered)
        self.assertIn("raw_prompt_public=false", rendered)
        self.assertIn("prompt_scope never stores raw prompt text", rendered)
        self.assertIn("Output Scope section", rendered)
        self.assertIn("output request note", rendered)
        self.assertIn("prompt scope note", rendered)
        self.assertIn("answer scope note", rendered)
        self.assertIn("evidence, hashes, counts, and diagnostics", rendered)
        self.assertIn("instead of raw prompts or answer transcripts", rendered)
        self.assertIn("The trace line summarizes session, request count, ledger rows, stream events", rendered)
        self.assertIn("The result line summarizes completion state", rendered)
        self.assertIn("display safety: local-private for terminal-only", rendered)
        self.assertIn("hash-only for redacted summaries", rendered)
        self.assertIn("hash-only-json for JSON stdout", rendered)
        self.assertIn("saved-terminal-redacted when saved artifacts record that terminal text", rendered)
        self.assertIn("was shown locally but removed from JSON/Markdown", rendered)
        self.assertIn("shareable-terminal-redacted", rendered)
        self.assertIn("means --shareable-terminal also hid that answer", rendered)
        self.assertIn("In non-JSON human output, answer_scope states where any answer text is visible", rendered)
        self.assertIn("when no local answer text is available", rendered)
        self.assertIn("still prints answer_scope=no-local-answer", rendered)
        self.assertIn("When local text is shown", rendered)
        self.assertIn("terminal prints it as answer: or answer[n]:", rendered)
        self.assertIn("answer_scope and local_output", rendered)
        self.assertIn("safety metadata with safe count/source fields", rendered)
        self.assertIn("answer_scope.scope_state uses stable values", rendered)
        self.assertIn("terminal-visible", rendered)
        self.assertIn("saved-terminal-redacted", rendered)
        self.assertIn("shareable-terminal-redacted", rendered)
        self.assertIn("json-suppressed", rendered)
        self.assertIn("no-local-answer", rendered)
        self.assertIn("answer_scope_note", rendered)
        self.assertIn("output_display_note", rendered)
        self.assertIn("answer-display and artifact-redaction policy", rendered)
        self.assertIn("JSON mode", rendered)
        self.assertIn("can still report completed generation through json-suppressed", rendered)
        self.assertIn("saved_redacted=True count=N", rendered)
        self.assertIn("that means output", rendered)
        self.assertIn("exists, but the raw answer is intentionally hidden", rendered)
        self.assertIn("local_output metadata", rendered)
        self.assertIn("safe count/source fields", rendered)
        self.assertIn("local-private-task-state", rendered)
        self.assertIn("coordinator-validation", rendered)
        self.assertIn("The output_display line makes the display policy explicit", rendered)
        self.assertIn("--include-output records that local display was requested", rendered)
        self.assertIn("it does not make raw generated text public", rendered)
        self.assertIn("The runtime_options line records safe wait/retry controls", rendered)
        self.assertIn("timeout_seconds", rendered)
        self.assertIn("poll_interval", rendered)
        self.assertIn("http_timeout", rendered)
        self.assertIn("admin_results_limit", rendered)
        self.assertIn("Timeout retry commands", rendered)
        self.assertIn("preserve non-default poll/http/result-limit values", rendered)
        self.assertIn("only extending", rendered)
        self.assertIn("--timeout-seconds", rendered)
        self.assertIn("safe per-request ids or prompt hashes", rendered)
        self.assertIn("never exposes raw prompt text", rendered)
        self.assertIn("The issue line summarizes state, primary diagnosis, next step, safe progress", rendered)
        self.assertIn("redacted detail is available", rendered)
        self.assertIn("The artifacts line points to the first Markdown summary to inspect", rendered)
        self.assertIn("redacted JSON/Markdown artifact paths", rendered)
        self.assertIn("recommended command label", rendered)
        self.assertIn("attention warnings such as incomplete stream evidence", rendered)
        self.assertIn("inspect_first line", rendered)
        self.assertIn("Markdown summary to open first", rendered)
        self.assertIn("review_next line", rendered)
        self.assertIn("safe recommended command", rendered)
        self.assertIn("terminal output renders local prompt sources for copying", rendered)
        self.assertIn("using a pipe placeholder for --prompt-stdin", rendered)
        self.assertIn("saved Markdown command lines also use that stdin pipe placeholder", rendered)
        self.assertIn("inline prompt terminal next commands are local-private", rendered)
        self.assertIn("prompt-file and prompt-texts-file terminal next commands include local file paths", rendered)
        self.assertIn("mark terminal_local_paths=True", rendered)
        self.assertIn("use --prompt-stdin or --shareable-terminal when terminal logs need to be shareable", rendered)
        self.assertIn("pass --shareable-terminal to keep human output", rendered)
        self.assertIn("hide inline prompts, local prompt paths, and local answer text", rendered)
        self.assertIn("safe printf placeholder pipe", rendered)
        self.assertIn("copyable reruns", rendered)
        self.assertIn("JSON fields and saved Markdown prompt values keep prompt placeholders", rendered)
        self.assertIn("prompt_scope records that distinction without raw text", rendered)
        self.assertIn("Plain\n`infer --dry-run` defaults to that existing-swarm preflight", rendered)
        self.assertIn("check an existing route/session without submitting", rendered)
        self.assertIn("defaults to --mode existing when", rendered)
        self.assertIn("--skip-live-preflight", rendered)
        self.assertIn("existing dry-run only: skip Coordinator /ready", rendered)
        self.assertIn("optional single prompt text; mutually exclusive with", rendered)
        self.assertIn("single prompt text; mutually exclusive with other", rendered)
        self.assertIn("other prompt sources", rendered)
        self.assertIn("comma-separated bounded batch of up to 4 prompts", rendered)
        self.assertIn("request local human display of generated text", rendered)
        self.assertIn("not production", rendered)

    def test_generate_help_shows_user_examples_and_boundaries(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            cli.parse_args(["generate", "--help"])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        self.assertIn("Create a bounded CrowdTensor generation request", rendered)
        self.assertIn("examples:", rendered)
        self.assertIn('crowdtensor generate "your prompt"', rendered)
        self.assertIn("generate_summary.json/generate_summary.md", rendered)
        self.assertIn("--output-dir", rendered)
        self.assertIn("missing routes return startup guidance", rendered)
        self.assertIn("short safe stderr start hint", rendered)
        self.assertIn("--json keeps stdout machine-readable", rendered)
        self.assertIn("Start with the review/review_summary line and inspect_first", rendered)
        self.assertIn("first Markdown artifact to open", rendered)
        self.assertEqual(rendered.count("Start with the review/review_summary line"), 1)
        self.assertIn("status/user_status line then", rendered)
        self.assertIn("preflight-ready", rendered)
        self.assertIn("preflight-partial", rendered)
        self.assertIn("ready_to_submit labels mean", rendered)
        self.assertIn("ready_to_submit.next_step is the script-friendly", rendered)
        self.assertIn("stage_preflight_unknown", rendered)
        self.assertIn("stage_preflight_not_checked means fix route/Coordinator, then rerun with observer token", rendered)
        self.assertIn("printed", rendered)
        self.assertIn("follow-up preflight", rendered)
        self.assertIn("skipped is request-shape only", rendered)
        self.assertIn("Use one prompt source at a time: positional prompt, --prompt-text/--prompt", rendered)
        self.assertIn("--prompt-file for a UTF-8 single prompt file", rendered)
        self.assertIn("--prompt-stdin for an explicit", rendered)
        self.assertIn("--prompt-texts-file for one prompt per line", rendered)
        self.assertIn("crowdtensor generate --prompt-file prompt.txt", rendered)
        self.assertIn('echo "your prompt" | crowdtensor generate --prompt-stdin --coordinator-url http://127.0.0.1:8787 --dry-run', rendered)
        self.assertIn("crowdtensor generate --prompt-texts-file prompts.txt --coordinator-url http://127.0.0.1:8787 --dry-run", rendered)
        self.assertIn("ambiguous mixes are rejected", rendered)
        self.assertIn("output_request.include_output", rendered)
        self.assertIn("output_request.raw_generated_text_public false", rendered)
        self.assertIn("Reports also include prompt_scope", rendered)
        self.assertIn("source, prompt_count", rendered)
        self.assertIn("terminal next commands", rendered)
        self.assertIn("terminal_local_paths", rendered)
        self.assertIn("saved artifacts use placeholders", rendered)
        self.assertIn("raw_prompt_public=false", rendered)
        self.assertIn("prompt_scope never stores raw prompt text", rendered)
        self.assertIn("Output Scope section", rendered)
        self.assertIn("output request note", rendered)
        self.assertIn("prompt scope note", rendered)
        self.assertIn("answer scope note", rendered)
        self.assertIn("evidence, hashes, counts, and diagnostics", rendered)
        self.assertIn("instead of raw prompts or answer transcripts", rendered)
        self.assertIn("The trace line summarizes session, request count, ledger rows, stream events", rendered)
        self.assertIn("The result line summarizes completion state", rendered)
        self.assertIn("display safety: local-private for terminal-only", rendered)
        self.assertIn("hash-only for redacted summaries", rendered)
        self.assertIn("hash-only-json for JSON stdout", rendered)
        self.assertIn("saved-terminal-redacted when saved artifacts record that terminal text", rendered)
        self.assertIn("was shown locally but removed from JSON/Markdown", rendered)
        self.assertIn("shareable-terminal-redacted", rendered)
        self.assertIn("means --shareable-terminal also hid that answer", rendered)
        self.assertIn("In non-JSON human output, answer_scope states where any answer text is visible", rendered)
        self.assertIn("when no local answer text is available", rendered)
        self.assertIn("still prints answer_scope=no-local-answer", rendered)
        self.assertIn("When local text is shown", rendered)
        self.assertIn("terminal prints it as answer: or answer[n]:", rendered)
        self.assertIn("answer_scope and local_output", rendered)
        self.assertIn("safety metadata with safe count/source fields", rendered)
        self.assertIn("answer_scope.scope_state uses stable values", rendered)
        self.assertIn("terminal-visible", rendered)
        self.assertIn("saved-terminal-redacted", rendered)
        self.assertIn("shareable-terminal-redacted", rendered)
        self.assertIn("json-suppressed", rendered)
        self.assertIn("no-local-answer", rendered)
        self.assertIn("answer_scope_note", rendered)
        self.assertIn("output_display_note", rendered)
        self.assertIn("answer-display and artifact-redaction policy", rendered)
        self.assertIn("JSON mode", rendered)
        self.assertIn("can still report completed generation through json-suppressed", rendered)
        self.assertIn("saved_redacted=True count=N", rendered)
        self.assertIn("that means output", rendered)
        self.assertIn("exists, but the raw answer is intentionally hidden", rendered)
        self.assertIn("local_output metadata", rendered)
        self.assertIn("safe count/source fields", rendered)
        self.assertIn("local-private-task-state", rendered)
        self.assertIn("coordinator-validation", rendered)
        self.assertIn("The output_display line makes the display policy explicit", rendered)
        self.assertIn("--include-output records that local display was requested", rendered)
        self.assertIn("it does not make raw generated text public", rendered)
        self.assertIn("The runtime_options line records safe wait/retry controls", rendered)
        self.assertIn("timeout_seconds", rendered)
        self.assertIn("poll_interval", rendered)
        self.assertIn("http_timeout", rendered)
        self.assertIn("admin_results_limit", rendered)
        self.assertIn("Timeout retry commands", rendered)
        self.assertIn("preserve non-default poll/http/result-limit values", rendered)
        self.assertIn("only extending", rendered)
        self.assertIn("--timeout-seconds", rendered)
        self.assertIn("safe per-request ids or prompt hashes", rendered)
        self.assertIn("never exposes raw prompt text", rendered)
        self.assertIn("The issue line summarizes state, primary diagnosis, next step, safe progress", rendered)
        self.assertIn("The artifacts line points to the first Markdown summary to inspect", rendered)
        self.assertIn("redacted JSON/Markdown artifact paths", rendered)
        self.assertIn("recommended command label", rendered)
        self.assertIn("attention warnings such as incomplete stream evidence", rendered)
        self.assertIn("inspect_first line", rendered)
        self.assertIn("Markdown summary to open first", rendered)
        self.assertIn("review_next line", rendered)
        self.assertIn("safe recommended command", rendered)
        self.assertIn("terminal output renders local prompt sources for copying", rendered)
        self.assertIn("using a pipe placeholder for --prompt-stdin", rendered)
        self.assertIn("saved Markdown command lines also use that stdin pipe placeholder", rendered)
        self.assertIn("inline prompt terminal next commands are local-private", rendered)
        self.assertIn("prompt-file and prompt-texts-file terminal next commands include local file paths", rendered)
        self.assertIn("mark terminal_local_paths=True", rendered)
        self.assertIn("use --prompt-stdin or --shareable-terminal when terminal logs need to be shareable", rendered)
        self.assertIn("pass --shareable-terminal to keep human output", rendered)
        self.assertIn("hide inline prompts, local prompt paths, and local answer text", rendered)
        self.assertIn("safe printf placeholder pipe", rendered)
        self.assertIn("copyable reruns", rendered)
        self.assertIn("JSON fields and saved Markdown prompt values keep prompt placeholders", rendered)
        self.assertIn("prompt_scope records that distinction without raw text", rendered)
        self.assertIn("redacted detail is available", rendered)
        self.assertIn("check route/session readiness without submitting a", rendered)
        self.assertIn("generation task", rendered)
        self.assertIn("optional single prompt text; mutually exclusive with", rendered)
        self.assertIn("single prompt text; mutually exclusive with other", rendered)
        self.assertIn("other prompt sources", rendered)
        self.assertIn("comma-separated bounded batch of up to 4 prompts", rendered)
        self.assertIn("request local human display of generated text", rendered)
        self.assertIn("not production", rendered)

    def test_stage_preflight_missing_text_distinguishes_not_checked(self) -> None:
        self.assertEqual(cli.stage_preflight_missing_text({"checked": False}), "not_checked")
        self.assertEqual(cli.stage_preflight_missing_text({"checked": True}), "none")
        self.assertEqual(
            cli.stage_preflight_missing_text({
                "checked": True,
                "missing_capabilities": ["real_llm_sharded_stage1"],
            }),
            "real_llm_sharded_stage1",
        )
        self.assertEqual(
            cli.annotate_stage_preflight({"checked": False})["missing_summary"],
            "not_checked",
        )
        self.assertEqual(cli.stage_preflight_diagnosis_code({"checked": True, "ok": True}), "stage_preflight_ready")
        self.assertEqual(cli.stage_preflight_diagnosis_code({"checked": True, "ok": False}), "stage_preflight_failed")
        self.assertEqual(
            cli.stage_preflight_diagnosis_code({"checked": False, "reason": "coordinator_not_ready"}),
            "stage_preflight_not_checked",
        )
        self.assertEqual(
            cli.stage_preflight_diagnosis_code({"checked": False, "reason": "route_not_ready"}),
            "stage_preflight_not_checked",
        )
        self.assertEqual(
            cli.stage_preflight_diagnosis_code({"checked": False, "reason": "observer_token_missing"}),
            "stage_preflight_skipped",
        )
        self.assertEqual(cli.ready_to_submit_stage_text({"stage_verification": "ready"}), "ready")
        self.assertEqual(cli.ready_to_submit_stage_text({"stage_preflight_ok": False}), "failed")
        self.assertEqual(cli.ready_to_submit_stage_text({}), "not_checked")
        self.assertEqual(cli.ready_to_submit_warning_text({"warning_codes": []}), "none")
        self.assertEqual(
            cli.ready_to_submit_warning_text({"warning_codes": ["coordinator_preflight_skipped", "stage_preflight_skipped"]}),
            "coordinator_preflight_skipped,stage_preflight_skipped",
        )
        self.assertIn(
            "Coordinator live readiness was skipped",
            cli.attention_display_text("coordinator_preflight_skipped,stage_preflight_skipped"),
        )
        self.assertIn(
            "stage0/stage1 Miner readiness was skipped",
            cli.attention_display_text("coordinator_preflight_skipped,stage_preflight_skipped"),
        )
        self.assertIn(
            "stream progress is incomplete",
            cli.attention_display_text("request[2]=req-2:1/2"),
        )
        self.assertEqual(
            cli.next_reason_detail("confirm_live_preflight"),
            "Run live preflight before submitting because readiness was skipped.",
        )
        self.assertEqual(
            cli.next_reason_detail("rerun_inference"),
            "Rerun the inference request.",
        )
        self.assertEqual(cli.route_catalog_missing_text({"route_source": "coordinator-url"}), "not_used")
        self.assertEqual(
            cli.route_catalog_missing_text({
                "route_source": "p2p-discovery",
                "missing_capabilities": ["real_llm_sharded_stage1"],
            }),
            "real_llm_sharded_stage1",
        )
        self.assertEqual(
            cli.infer_route_distinct_stage_text({"distinct_stage_miners": False}, {"checked": False}),
            "not_checked",
        )
        self.assertEqual(
            cli.infer_route_distinct_stage_text({"distinct_stage_miners": False}, {"checked": True, "distinct_stage_miners": True}),
            "True",
        )
        self.assertEqual(
            cli.coordinator_ready_text({"ok": False, "error": "OSError"}),
            "not_ready service=none protocol=none error=OSError",
        )
        self.assertEqual(
            cli.coordinator_ready_text({"ok": None, "reason": "live_preflight_skipped"}),
            "not_checked service=none protocol=none reason=live_preflight_skipped",
        )
        self.assertEqual(
            cli.ready_to_submit_status_text({
                "ok": None,
                "readiness_label": "skipped",
                "fully_verified": False,
                "route_ready": True,
                "coordinator_ready": None,
                "stage_verification": "skipped",
                "next_step": "run_live_preflight",
                "warning_codes": ["coordinator_preflight_skipped", "stage_preflight_skipped"],
            }),
            "not_checked label=skipped fully_verified=False route=True coordinator=not_checked stage=skipped stage_verification=skipped next_step=run_live_preflight warnings=coordinator_preflight_skipped,stage_preflight_skipped",
        )
        self.assertEqual(
            cli.guarded_submit_label("submit inference", {"readiness_label": "blocked"}),
            "submit inference after checks pass",
        )
        self.assertEqual(
            cli.guarded_submit_label("submit inference", {"readiness_label": "skipped"}),
            "submit inference after live preflight",
        )
        self.assertEqual(
            cli.guarded_submit_label(
                "submit inference",
                {"readiness_label": "partial", "warning_codes": ["stage_preflight_skipped"]},
            ),
            "submit inference after stage preflight",
        )
        self.assertEqual(
            cli.guarded_submit_label(
                "submit inference",
                {"readiness_label": "partial", "warning_codes": ["stage_preflight_unknown"]},
            ),
            "submit inference after stage preflight",
        )
        self.assertEqual(
            cli.guarded_submit_label(
                "submit inference",
                {"readiness_label": "partial", "warning_codes": ["stage_preflight_not_checked"]},
            ),
            "submit inference after stage preflight",
        )
        self.assertEqual(
            cli.guarded_submit_label("submit inference", {"readiness_label": "verified"}),
            "submit inference",
        )
        self.assertEqual(
            cli.guarded_submit_label("submit inference", {"next_step": "run_stage_preflight"}),
            "submit inference after stage preflight",
        )
        self.assertEqual(
            cli.guarded_submit_label("submit inference", {"next_step": "submit_with_caution"}),
            "submit inference with caution",
        )
        self.assertEqual(
            cli._ready_to_submit_status(
                submit_ok=True,
                route_ready=True,
                coordinator_ok=True,
                coordinator_preflight_required=True,
                stage_preflight_ok=None,
                stage_preflight_required=False,
                source="unit",
            )["next_step"],
            "run_stage_preflight",
        )
        self.assertEqual(
            cli._ready_to_submit_status(
                submit_ok=True,
                route_ready=True,
                coordinator_ok=True,
                coordinator_preflight_required=True,
                stage_preflight_ok=None,
                stage_preflight_required=True,
                source="unit",
            )["next_step"],
            "run_stage_preflight",
        )
        unknown_stage = cli._ready_to_submit_status(
            submit_ok=True,
            route_ready=True,
            coordinator_ok=True,
            coordinator_preflight_required=True,
            stage_preflight_ok=None,
            stage_preflight_required=True,
            source="unit",
        )
        self.assertEqual(unknown_stage["stage_verification"], "unknown")
        self.assertEqual(unknown_stage["warning_codes"], ["stage_preflight_unknown"])
        caution_status = cli._ready_to_submit_status(
            submit_ok=True,
            route_ready=True,
            coordinator_ok=None,
            coordinator_preflight_required=False,
            stage_preflight_ok=True,
            stage_preflight_required=True,
            source="unit",
        )
        self.assertFalse(caution_status["fully_verified"])
        self.assertEqual(caution_status["readiness_label"], "partial")
        self.assertEqual(caution_status["warning_codes"], ["coordinator_preflight_skipped"])
        self.assertEqual(
            caution_status["readiness_summary"],
            "Request can be submitted, but Coordinator readiness is not fully verified.",
        )
        self.assertEqual(caution_status["next_step"], "submit_with_caution")
        self.assertEqual(
            cli._ready_to_submit_status(
                submit_ok=True,
                route_ready=True,
                coordinator_ok=None,
                coordinator_preflight_required=False,
                stage_preflight_ok=None,
                stage_preflight_required=False,
                source="unit",
            )["readiness_summary"],
            "Request can be submitted, but live readiness is not fully verified.",
        )
        for blocked_status in [
            cli._ready_to_submit_status(
                submit_ok=True,
                route_ready=False,
                coordinator_ok=None,
                coordinator_preflight_required=False,
                stage_preflight_ok=None,
                stage_preflight_required=False,
                source="unit",
            ),
            cli._ready_to_submit_status(
                submit_ok=True,
                route_ready=True,
                coordinator_ok=False,
                coordinator_preflight_required=True,
                stage_preflight_ok=None,
                stage_preflight_required=False,
                source="unit",
            ),
            cli._ready_to_submit_status(
                submit_ok=True,
                route_ready=True,
                coordinator_ok=True,
                coordinator_preflight_required=True,
                stage_preflight_ok=False,
                stage_preflight_required=True,
                source="unit",
            ),
        ]:
            self.assertFalse(blocked_status["fully_verified"])
            self.assertEqual(blocked_status["readiness_label"], "blocked")
            self.assertEqual(blocked_status["next_step"], "fix_blockers")
        self.assertIn(
            "stage0/stage1 were not fully verified",
            cli._ready_to_submit_action("Inference", {"next_step": "run_stage_preflight"}),
        )
        self.assertIn(
            "live readiness was skipped",
            cli._ready_to_submit_action("Generation", {"next_step": "run_live_preflight"}),
        )
        self.assertIn(
            "run the printed preflight next command",
            cli._ready_to_submit_action("Inference", {"next_step": "submit_with_caution"}),
        )
        self.assertIn(
            "Coordinator readiness was not fully verified",
            cli._ready_to_submit_action(
                "Inference",
                {
                    "next_step": "submit_with_caution",
                    "warning_codes": ["coordinator_preflight_skipped"],
                },
            ),
        )
        self.assertIn(
            "Coordinator readiness first",
            cli._infer_operator_action(
                argparse.Namespace(dry_run=True, infer_mode="existing", full_evidence=False),
                {"diagnosis_codes": ["stage_preflight_not_checked"]},
                ok=False,
            ),
        )
        self.assertIn(
            "Coordinator readiness first",
            cli._product_generate_operator_action({"ok": False, "diagnosis_codes": ["stage_preflight_not_checked"]}),
        )
        self.assertIn(
            "stage0/stage1 were not fully verified",
            cli._ready_to_submit_action(
                "Inference",
                {
                    "ok": True,
                    "fully_verified": False,
                    "warning_codes": ["stage_preflight_unknown"],
                },
            ),
        )
        self.assertIn(
            "stage0/stage1 were not fully verified",
            cli._ready_to_submit_action(
                "Inference",
                {
                    "ok": True,
                    "fully_verified": False,
                    "warning_codes": ["stage_preflight_not_checked"],
                },
            ),
        )

    def test_infer_prompt_redaction_values_include_batch_items_and_raw_batch(self) -> None:
        args = argparse.Namespace(prompt_text="", prompt_texts="first private prompt,second private prompt")

        values = cli._infer_prompt_redaction_values(args)

        self.assertIn("first private prompt", values)
        self.assertIn("second private prompt", values)
        self.assertIn("first private prompt,second private prompt", values)

    def test_serve_help_shows_inference_flow_examples(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            cli.parse_args(["serve", "--help"])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        self.assertIn("Start or print the Coordinator used by the product inference flow", rendered)
        self.assertIn("stage0 and one stage1 Miner", rendered)
        self.assertIn("generate --dry-run", rendered)
        self.assertIn("crowdtensor serve --profile cpu-real-llm", rendered)
        self.assertIn("Boundary: local/private Coordinator by default", rendered)

    def test_join_help_shows_stage_miner_examples(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            cli.parse_args(["join", "--help"])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        self.assertIn("Start or print a product Miner", rendered)
        self.assertIn("distinct stage0", rendered)
        self.assertIn("generate --dry-run", rendered)
        self.assertIn("--miner-id stage0-miner --stage stage0 --run", rendered)
        self.assertIn("not large-model serving", rendered)

    def test_user_docs_manual_demo_sets_tokens_before_submit(self) -> None:
        readme = (cli.ROOT / "README.md").read_text(encoding="utf-8")
        quickstart = (cli.ROOT / "docs" / "quickstart.md").read_text(encoding="utf-8")

        for rendered in [readme, quickstart]:
            self.assertIn("export CROWDTENSOR_ADMIN_TOKEN=local-admin", rendered)
            self.assertIn("export CROWDTENSOR_MINER_TOKEN=local-miner", rendered)
            self.assertIn("export CROWDTENSOR_OBSERVER_TOKEN=local-observer", rendered)
            self.assertIn('--observer-token "$CROWDTENSOR_OBSERVER_TOKEN"', rendered)
            self.assertIn("--dry-run", rendered)
            self.assertIn("Pick one prompt source per command", rendered)
            self.assertIn("--prompt-file prompt.txt", rendered)
            self.assertIn("--prompt-stdin", rendered)
            self.assertIn("--prompt-texts-file prompts.txt", rendered)
            self.assertIn("UTF-8 single", rendered)
            self.assertIn("Single prompts are capped at 256 characters", rendered)
            self.assertIn("batch files accept", rendered)
            self.assertIn("up to 4 non-empty prompt lines", rendered)
            self.assertIn("crowdtensor infer --prompt-file prompt.txt --max-new-tokens 8", rendered)
            self.assertIn('echo "your prompt" | crowdtensor infer --prompt-stdin --max-new-tokens 8', rendered)
            self.assertIn("crowdtensor infer --prompt-texts-file prompts.txt --max-new-tokens 8 --stream", rendered)
            self.assertIn("auto-selects an available loopback Coordinator port", rendered)
            self.assertIn("--coordinator-port", rendered)
            self.assertIn("fixed reproducible local port", rendered)
            self.assertIn("crowdtensor generate --prompt-file prompt.txt", rendered)
            self.assertIn('echo "your prompt" | crowdtensor generate --prompt-stdin', rendered)
            self.assertIn("crowdtensor generate --prompt-texts-file prompts.txt", rendered)
            self.assertIn("mixed prompt sources", rendered)
            self.assertIn("output_request.include_output", rendered)
            self.assertIn("output_request.raw_generated_text_public", rendered)
            self.assertIn("Markdown `Output Scope` section", rendered)
            self.assertIn("`output request note`", rendered)
            self.assertIn("`prompt scope note`", rendered)
            self.assertIn("`answer scope note`", rendered)
            self.assertIn("evidence, hashes, counts", rendered)
            self.assertIn("instead of raw prompts or answer transcripts", rendered)
            self.assertIn("`prompt_scope`", rendered)
            self.assertIn("machine-readable summary of the prompt", rendered)
            self.assertIn("`prompt-texts-file`", rendered)
            self.assertIn("`prompt_scope` does not contain", rendered)
            self.assertIn("raw prompt text", rendered)
            self.assertIn("generate_summary.json", rendered)
            self.assertIn("generate_summary.md", rendered)
            self.assertIn("Start by reading the `review` line", rendered)
            self.assertIn("`review_summary`", rendered)
            self.assertIn("Then use the `status` line", rendered)
            self.assertIn("`user_status`", rendered)
            self.assertIn("for detail", rendered)
            self.assertIn("`preflight-ready` means", rendered)
            self.assertIn("submit next", rendered)
            self.assertIn("`preflight-partial` means run the", rendered)
            self.assertIn("recommended check first", rendered)
            self.assertIn("crowdtensor public-real-llm-swarm-beta release", rendered)
            self.assertIn("crowdtensor public-real-llm-swarm-beta check", rendered)
            self.assertIn("--beta-report", rendered)
            self.assertIn("public_real_llm_swarm_beta_check.json", rendered)
            self.assertIn("public_real_llm_swarm_beta.md", rendered)
            self.assertIn("support_bundle.json", rendered)
            self.assertIn("Safe shareable files", rendered)
            self.assertIn("do not share", rendered)
            self.assertIn("generated token ids", rendered)
            self.assertIn("Not Completed", rendered)
            self.assertIn("printed `not_completed` lines", rendered)
            self.assertIn("KV-cache", rendered)
            self.assertIn("`recommended_next` plus `next[...]`", rendered)
            self.assertIn("`runtime_options` line", rendered)
            self.assertIn("`timeout_seconds`", rendered)
            self.assertIn("`poll_interval`", rendered)
            self.assertIn("`http_timeout`", rendered)
            self.assertIn("`admin_results_limit`", rendered)
            self.assertIn("preserve non-default", rendered)
            self.assertIn("only extending `--timeout-seconds`", rendered)
            self.assertIn("`trace`", rendered)
            self.assertIn("`result`", rendered)
            self.assertIn("completion state", rendered)
            self.assertIn("token count, output count", rendered)
            self.assertIn("display safety", rendered)
            self.assertIn("`local-private` for terminal-only generated text", rendered)
            self.assertIn("`hash-only` for redacted", rendered)
            self.assertIn("`hash-only-json` for JSON stdout", rendered)
            self.assertIn("terminal prints `answer_scope`", rendered)
            self.assertIn("display state is", rendered)
            self.assertIn("explicit: whether any answer text is visible", rendered)
            self.assertIn("`answer:`", rendered)
            self.assertIn("`answer[n]:`", rendered)
            self.assertIn("`answer_scope` and `local_output`", rendered)
            self.assertIn("safety metadata", rendered)
            self.assertIn("terminal still", rendered)
            self.assertIn("prints `answer_scope=no-local-answer`", rendered)
            self.assertIn("`answer_scope.scope_state` uses stable values", rendered)
            self.assertIn("`terminal-visible`", rendered)
            self.assertIn("`saved-terminal-redacted`", rendered)
            self.assertIn("`shareable-terminal-redacted`", rendered)
            self.assertIn("`json-suppressed`", rendered)
            self.assertIn("`no-local-answer`", rendered)
            self.assertIn("`answer_scope_note`", rendered)
            self.assertIn("`output_display_note`", rendered)
            self.assertIn("answer-display and", rendered)
            self.assertIn("artifact-redaction policy", rendered)
            self.assertIn("JSON mode can still", rendered)
            self.assertIn("`json-suppressed` plus redacted", rendered)
            self.assertIn("`saved_redacted=True count=N`", rendered)
            self.assertIn("output exists, but the raw answer is intentionally hidden", rendered)
            self.assertIn("Markdown `What To Do Next` and `Details`", rendered)
            self.assertIn("saved", rendered)
            self.assertIn("contain no generated text", rendered)
            self.assertIn("safe", rendered)
            self.assertIn("output `count` and `source` fields", rendered)
            self.assertIn("`local-private-task-state`", rendered)
            self.assertIn("`coordinator-validation`", rendered)
            self.assertIn("`issue`", rendered)
            self.assertIn("`issue_summary`", rendered)
            self.assertIn("`artifact_summary`", rendered)
            self.assertIn("`review_summary`", rendered)
            self.assertIn("current state, next step, first artifact", rendered)
            self.assertIn("`attention` value for warnings", rendered)
            self.assertIn("skipped preflights", rendered)
            self.assertIn("Markdown explains", rendered)
            self.assertIn("What To Do Next", rendered)
            self.assertIn("`inspect_first` line", rendered)
            self.assertIn("Markdown summary to open first", rendered)
            self.assertIn("`review_next` line", rendered)
            self.assertIn("safe recommended command", rendered)
            self.assertIn("human terminal output renders it", rendered)
            self.assertIn("with local prompt sources for copying", rendered)
            self.assertIn("`printf` pipe placeholder", rendered)
            self.assertIn("Saved Markdown command lines also use that stdin pipe", rendered)
            self.assertIn("copyable `printf`", rendered)
            self.assertIn("without expanding the real stdin prompt", rendered)
            self.assertIn("`shareable_terminal.enabled=True`", rendered)
            self.assertIn("`answer_scope.scope_state=shareable-terminal-redacted`", rendered)
            self.assertIn("JSON fields and saved Markdown prompt values keep prompt", rendered)
            self.assertIn("placeholders", rendered)
            self.assertIn("`prompt_scope` records that distinction without storing raw text", rendered)
            self.assertIn("local prompt file paths", rendered)
            self.assertIn("`terminal_local_paths=True`", rendered)
            self.assertIn("`--shareable-terminal`", rendered)
            self.assertIn("rerunning saved", rendered)
            self.assertIn("`--prompt-file`", rendered)
            self.assertIn("`--prompt-stdin`", rendered)
            self.assertIn("`--prompt-texts-file`", rendered)
            self.assertIn("when rerunning", rendered)
            self.assertIn("saved", rendered)
            self.assertIn("commands", rendered)
            self.assertIn("first Markdown summary to inspect", rendered)
            self.assertIn("accepted ledger rows", rendered)
            self.assertIn("primary diagnosis code", rendered)
            self.assertIn("safe progress text", rendered)
            self.assertIn("per-request ids or prompt hashes", rendered)
            self.assertIn("failure", rendered)
            self.assertIn("redacted", rendered)
            self.assertIn("prompt text", rendered)

    def test_public_real_llm_swarm_beta_docs_prefer_cli_check(self) -> None:
        readme = (cli.ROOT / "README.md").read_text(encoding="utf-8")
        quickstart = (cli.ROOT / "docs" / "quickstart.md").read_text(encoding="utf-8")
        operations = (cli.ROOT / "docs" / "operations.md").read_text(encoding="utf-8")
        memory = (cli.ROOT / "docs" / "project-memory.md").read_text(encoding="utf-8")

        for rendered in [readme, quickstart, operations, memory]:
            self.assertIn("crowdtensor public-real-llm-swarm-beta check", rendered)
            self.assertIn("--beta-report", rendered)
            self.assertIn("public_real_llm_swarm_beta_check", rendered)
            self.assertIn("public_real_llm_swarm_beta.json", rendered)
        self.assertIn("official user-facing validation entry", readme)
        self.assertIn("official user-facing validation entry", quickstart)
        self.assertIn("official user-facing validation wrapper", operations)
        self.assertIn("check_source: beta-report", operations)
        self.assertIn("check_source: beta-report", memory)
        self.assertIn("cli_mode: check", operations)
        self.assertIn("cli_mode: check", memory)
        self.assertIn("crowdtensor public-real-llm-swarm-beta check --hf-model-id distilgpt2 --beta-report", operations)
        self.assertIn("crowdtensor public-real-llm-swarm-beta check --hf-model-id distilgpt2 --beta-report", memory)
        self.assertNotIn("python scripts/public_real_llm_swarm_beta_check.py --json", operations)
        self.assertNotIn("python scripts/public_real_llm_swarm_beta_check.py --mode local-model-variant", memory)

    def test_runtime_matrix_block_skips_demo_and_manifest(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if "doctor.py" in command[1]:
                return completed({"ok": True})
            if "runtime_matrix.py" in command[1]:
                return completed({"ok": False, "diagnosis_summary": {"codes": ["runtime_matrix_blocked"]}}, returncode=1)
            raise AssertionError(f"unexpected command: {command}")

        args = cli.parse_args(["local-proof", "--output-dir", str(output_dir), "--base-port", "9001"])

        summary = cli.build_local_proof(args, runner=fake_runner)

        self.assertFalse(summary["ok"])
        self.assertIn("runtime_matrix_blocked", summary["errors"])
        skipped = [step for step in summary["steps"] if step.get("skipped")]
        self.assertEqual([step["name"] for step in skipped], ["home_compute_demo", "demo_manifest"])

    def test_summary_redacts_sensitive_payload_fragments(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if "doctor.py" in command[1]:
                return completed({"ok": True, "lease_token": "secret-lease"})
            if "runtime_matrix.py" in command[1]:
                return completed({"ok": True})
            if "home_compute_demo.py" in command[1]:
                return completed({"ok": True, "inference_results": [{"x": 1}]})
            if "demo_manifest_pack.py" in command[1]:
                return completed({"ok": True, "schema": "demo_manifest_v1"})
            raise AssertionError(command)

        args = cli.parse_args(["local-proof", "--output-dir", str(output_dir), "--base-port", "9002"])

        summary = cli.build_local_proof(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertNotIn("secret-lease", serialized)
        self.assertNotIn("inference_results", serialized)
        self.assertNotIn("lease_token", serialized)

    def test_main_json_outputs_summary_and_exit_zero(self) -> None:
        summary = {"schema": "local_proof_summary_v1", "ok": True}
        with patch.object(cli, "build_local_proof", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["local-proof", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "local_proof_summary_v1")

    def test_cleanup_dry_run_keeps_candidates(self) -> None:
        root = Path(self._tmp_dir())
        tmp_root = Path(self._tmp_dir())
        cache = root / "crowdtensor" / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "x.pyc").write_bytes(b"cache")
        proof = tmp_root / "crowdtensor_local_proof_old"
        proof.mkdir()
        (proof / "artifact.json").write_text("{}", encoding="utf-8")
        old_time = 1_700_000_000
        os.utime(proof, (old_time, old_time))

        args = self._cleanup_args("--json")
        report = cli.build_cleanup_report(args, root=root, tmp_root=tmp_root)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "cleanup_report_v1")
        self.assertEqual(report["mode"], "dry_run")
        self.assertTrue(cache.exists())
        self.assertTrue(proof.exists())
        actions = {candidate["action"] for candidate in report["candidates"]}
        self.assertIn("dry_run", actions)

    def test_cleanup_apply_deletes_cache_and_old_temp_dir(self) -> None:
        root = Path(self._tmp_dir())
        tmp_root = Path(self._tmp_dir())
        cache = root / "tests" / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "test.pyc").write_bytes(b"cache")
        proof = tmp_root / "crowdtensor_local_proof_old"
        proof.mkdir()
        (proof / "artifact.json").write_text("{}", encoding="utf-8")
        old_time = 1_700_000_000
        os.utime(proof, (old_time, old_time))

        args = self._cleanup_args("--apply", "--older-than-hours", "0", "--json")
        report = cli.build_cleanup_report(args, root=root, tmp_root=tmp_root)

        self.assertTrue(report["ok"], report)
        self.assertFalse(cache.exists())
        self.assertFalse(proof.exists())
        self.assertGreater(report["deleted_bytes"], 0)
        self.assertEqual({candidate["action"] for candidate in report["candidates"]}, {"deleted"})

    def test_cleanup_reports_require_explicit_include_reports(self) -> None:
        root = Path(self._tmp_dir())
        tmp_root = Path(self._tmp_dir())
        report_path = tmp_root / "crowdtensor_acceptance.json"
        report_path.write_text("{}", encoding="utf-8")
        old_time = 1_700_000_000
        os.utime(report_path, (old_time, old_time))

        default_report = cli.build_cleanup_report(
            self._cleanup_args("--apply", "--older-than-hours", "0", "--json"),
            root=root,
            tmp_root=tmp_root,
        )
        self.assertTrue(report_path.exists())
        self.assertEqual(default_report["candidates"][0]["skip_reason"], "requires_include_reports")

        include_report = cli.build_cleanup_report(
            self._cleanup_args("--apply", "--include-reports", "--older-than-hours", "0", "--json"),
            root=root,
            tmp_root=tmp_root,
        )
        self.assertFalse(report_path.exists())
        self.assertEqual(include_report["candidates"][0]["action"], "deleted")

    def test_cleanup_skips_protected_paths_and_symlinks(self) -> None:
        root = Path(self._tmp_dir())
        tmp_root = Path(self._tmp_dir())
        protected = root / "state" / "__pycache__"
        protected.mkdir(parents=True)
        (protected / "state.pyc").write_bytes(b"cache")
        target = tmp_root / "target"
        target.mkdir()
        link = tmp_root / "crowdtensor_local_proof_link"
        link.symlink_to(target, target_is_directory=True)

        args = self._cleanup_args("--apply", "--older-than-hours", "0", "--json")
        report = cli.build_cleanup_report(args, root=root, tmp_root=tmp_root)

        self.assertTrue(protected.exists())
        self.assertTrue(link.exists())
        skipped = {candidate["skip_reason"] for candidate in report["candidates"]}
        self.assertIn("protected_repo_path", skipped)
        self.assertIn("symlink", skipped)

    def test_main_cleanup_json_outputs_report(self) -> None:
        report = {"schema": "cleanup_report_v1", "ok": True}
        with patch.object(cli, "build_cleanup_report", return_value=report), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["clean-artifacts", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "cleanup_report_v1")

    def test_product_serve_redacts_tokens_in_command_report(self) -> None:
        args = cli.parse_args([
            "serve",
            "--admin-token",
            "admin-secret",
            "--miner-token",
            "miner-secret",
            "--json",
        ])

        report = cli.build_product_serve(args)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertNotIn("admin-secret", encoded)
        self.assertNotIn("miner-secret", encoded)
        self.assertIn("<redacted>", report["command"])
        self.assertIn("command_line", report)
        self.assertIn("--admin-token '<redacted>'", report["command_line"])
        self.assertNotIn("admin-secret", report["command_line"])
        self.assertNotIn("miner-secret", report["command_line"])
        self.assertIn("Rerun with --run", report["operator_action"])
        self.assertIn("generate --coordinator-url http://127.0.0.1:8787 --dry-run", report["operator_action"])
        self.assertNotIn("generate --p2p", report["operator_action"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            "crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}",
            next_lines,
        )
        self.assertNotIn(cli.DEFAULT_PRODUCT_GENERATE_PROMPT, encoded)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_serve(report)
        rendered = stdout.getvalue()
        self.assertIn("CrowdTensor serve", rendered)
        self.assertIn("  command: ", rendered)
        self.assertIn("--admin-token '<redacted>'", rendered)
        self.assertIn("  action: Rerun with --run", rendered)
        self.assertIn("  next[4] check generation route: crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}", rendered)
        self.assertIn("# requires CROWDTENSOR_OBSERVER_TOKEN", rendered)
        self.assertIn("# requires CROWDTENSOR_MINER_TOKEN", rendered)
        self.assertNotIn("admin-secret", rendered)
        self.assertNotIn("miner-secret", rendered)

    def test_product_serve_public_bind_action(self) -> None:
        args = cli.parse_args([
            "serve",
            "--bind-host",
            "0.0.0.0",
            "--public-host",
            "203.0.113.5",
            "--json",
        ])

        report = cli.build_product_serve(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("public_bind_requires_explicit_ack", report["diagnosis_codes"])
        self.assertIn("command_line", report)
        self.assertIn("trusted network boundary", report["operator_action"])

    def test_product_generate_dry_run_uses_session_protocol(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--output-dir",
            str(output_dir),
            "--prompt-text",
            "CrowdTensor prompt",
            "--backend",
            "cuda",
            "--hf-model-id",
            "distilgpt2",
            "--dry-run",
            "--skip-live-preflight",
            "--stream",
            "--include-output",
            "--json",
        ])

        report = cli.build_product_generate(args)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["session_request"]["schema"], "session_protocol_v1")
        self.assertEqual(report["session_request"]["hf_model_id"], "distilgpt2")
        self.assertTrue(report["stream"]["enabled"])
        self.assertTrue(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertTrue(report["output_request"]["public_artifact_safe"])
        self.assertEqual(report["saved_summary"]["path"], str(output_dir / "generate_summary.json"))
        self.assertEqual(report["saved_summary"]["markdown_path"], str(output_dir / "generate_summary.md"))
        self.assertTrue(report["artifacts"]["generate_summary"]["present"])
        self.assertTrue(report["artifacts"]["generate_summary_markdown"]["present"])
        self.assertEqual(report["artifact_summary"]["inspect_first"], str(output_dir / "generate_summary.md"))
        self.assertEqual(report["artifact_summary"]["summary_json"], str(output_dir / "generate_summary.json"))
        self.assertEqual(report["artifact_summary"]["summary_markdown"], str(output_dir / "generate_summary.md"))
        self.assertEqual(report["artifact_summary"]["artifact_count"], 2)
        self.assertEqual(report["artifact_summary"]["present_artifact_count"], 2)
        self.assertTrue(report["artifact_summary"]["public_artifact_safe"])
        self.assertEqual(report["review_summary"]["state"], "preflight-partial")
        self.assertEqual(report["review_summary"]["next_step"], "run_live_preflight")
        self.assertEqual(report["review_summary"]["inspect_first"], str(output_dir / "generate_summary.md"))
        self.assertEqual(report["review_summary"]["recommended_label"], "check generation route")
        self.assertEqual(report["review_summary"]["primary_code"], "coordinator_ready_preflight_skipped")
        self.assertEqual(report["review_summary"]["attention"], "coordinator_preflight_skipped,stage_preflight_skipped")
        self.assertIn("Coordinator live readiness was skipped", report["review_summary"]["attention_detail"])
        self.assertIn("stage0/stage1 Miner readiness was skipped", report["review_summary"]["attention_detail"])
        self.assertEqual(report["recommended_next_command"]["reason_detail"], "Run live preflight before submitting because readiness was skipped.")
        self.assertIn("<prompt>", report["review_summary"]["next_command"])
        self.assertIn("--dry-run", report["review_summary"]["next_command"])
        self.assertEqual(report["review_summary"]["requires_env"], ["CROWDTENSOR_OBSERVER_TOKEN"])
        self.assertTrue(report["review_summary"]["has_recommended_command"])
        self.assertTrue(report["review_summary"]["public_artifact_safe"])
        self.assertIsNone(report["trace"]["session_id"])
        self.assertEqual(report["trace"]["request_count"], 1)
        self.assertEqual(report["trace"]["accepted_rows_seen"], 0)
        self.assertEqual(report["trace"]["stream_event_count"], 0)
        self.assertEqual(report["trace"]["source"], "public_swarm_product_cli_v1")
        self.assertFalse(report["trace"]["raw_prompt_public"])
        self.assertFalse(report["trace"]["raw_generated_text_public"])
        self.assertFalse(report["trace"]["generated_token_ids_public"])
        self.assertTrue(report["trace"]["public_artifact_safe"])
        self.assertEqual(report["prompt_scope"]["source"], "prompt-text")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 1)
        self.assertTrue(report["prompt_scope"]["inline_prompt_text"])
        self.assertTrue(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertTrue(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertFalse(report["prompt_scope"]["terminal_local_paths"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_public_safe"])
        self.assertTrue(report["prompt_scope"]["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(report["prompt_scope"]["public_artifact_safe"])
        self.assertFalse(report["safety"]["raw_prompt_public"])
        self.assertFalse(report["safety"]["raw_generated_text_public"])
        self.assertFalse(report["safety"]["generated_token_ids_public"])
        self.assertTrue(report["safety"]["read_only_workload"])
        self.assertTrue(report["safety"]["coordinator_backed"])
        self.assertTrue(report["safety"]["not_production"])
        self.assertTrue(report["safety"]["not_large_model_serving"])
        self.assertTrue(report["safety"]["not_arbitrary_public_prompt_serving"])
        self.assertEqual(len(report["trace"]["request_trace"]), 1)
        self.assertEqual(report["trace"]["request_trace"][0]["source"], "session-request")
        self.assertTrue(report["trace"]["request_trace"][0]["prompt_hash"])
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(report["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(report["shareable_summary"]["raw_generated_text_public"])
        self.assertFalse(report["shareable_summary"]["generated_token_ids_public"])
        self.assertFalse(report["shareable_summary"]["local_output_display_only"])
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertFalse(report["shareable_summary"]["local_answer_terminal_only"])
        self.assertNotIn("local_output", report)
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(
            report["operator_action"],
            "Generation request shape is valid, but live readiness was skipped; rerun --dry-run without --skip-live-preflight before submitting.",
        )
        self.assertEqual(report["user_status"]["state"], "preflight-partial")
        self.assertEqual(report["user_status"]["next_step"], "run_live_preflight")
        self.assertEqual(report["user_status"]["recommended_label"], "check generation route")
        labels = [item["label"] for item in report["next_commands"]]
        self.assertIn("submit generation after live preflight", labels)
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            f"crowdtensor generate --max-new-tokens 16 --output-dir {output_dir} --coordinator-url http://127.0.0.1:8787 --backend cuda --hf-model-id distilgpt2 --prompt-text '<prompt>' --dry-run --observer-token ${{CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}} --stream --include-output",
            next_lines,
        )
        self.assertIn(
            f"crowdtensor generate --max-new-tokens 16 --output-dir {output_dir} --coordinator-url http://127.0.0.1:8787 --backend cuda --hf-model-id distilgpt2 --prompt-text '<prompt>' --stream --include-output",
            next_lines,
        )
        self.assertNotIn("CrowdTensor prompt", encoded)
        self.assertNotIn("CrowdTensor prompt", json.dumps(report["next_commands"], sort_keys=True))
        self.assertIn("prompt_hash", encoded)
        persisted = json.loads((output_dir / "generate_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["saved_summary"]["markdown_path"], str(output_dir / "generate_summary.md"))
        self.assertEqual(persisted["artifact_summary"]["inspect_first"], str(output_dir / "generate_summary.md"))
        self.assertTrue(persisted["artifact_summary"]["public_artifact_safe"])
        self.assertEqual(persisted["review_summary"]["inspect_first"], str(output_dir / "generate_summary.md"))
        self.assertEqual(persisted["review_summary"]["recommended_label"], "check generation route")
        self.assertIn("Coordinator live readiness was skipped", persisted["review_summary"]["attention_detail"])
        self.assertIn("stage0/stage1 Miner readiness was skipped", persisted["review_summary"]["attention_detail"])
        self.assertEqual(persisted["recommended_next_command"]["reason_detail"], "Run live preflight before submitting because readiness was skipped.")
        self.assertIn("<prompt>", persisted["review_summary"]["next_command"])
        self.assertNotIn("CrowdTensor prompt", persisted["review_summary"]["next_command"])
        self.assertTrue(persisted["review_summary"]["public_artifact_safe"])
        self.assertFalse(persisted["output_request"]["raw_prompt_public"])
        self.assertFalse(persisted["output_request"]["raw_generated_text_public"])
        self.assertFalse(persisted["output_request"]["generated_token_ids_public"])
        self.assertTrue(persisted["output_request"]["public_artifact_safe"])
        self.assertEqual(persisted["user_status"]["state"], "preflight-partial")
        self.assertEqual(persisted["user_status"]["next_step"], "run_live_preflight")
        self.assertEqual(persisted["trace"]["request_count"], 1)
        self.assertTrue(persisted["trace"]["request_trace"][0]["prompt_hash"])
        self.assertEqual(persisted["prompt_scope"]["source"], "prompt-text")
        self.assertTrue(persisted["prompt_scope"]["inline_prompt_text"])
        self.assertTrue(persisted["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertFalse(persisted["prompt_scope"]["terminal_local_paths"])
        self.assertTrue(persisted["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertFalse(persisted["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(persisted["prompt_scope"]["public_artifact_safe"])
        self.assertFalse(persisted["safety"]["raw_prompt_public"])
        self.assertFalse(persisted["safety"]["raw_generated_text_public"])
        self.assertFalse(persisted["safety"]["generated_token_ids_public"])
        self.assertTrue(persisted["safety"]["read_only_workload"])
        self.assertTrue(persisted["safety"]["coordinator_backed"])
        self.assertTrue(persisted["safety"]["not_production"])
        self.assertTrue(persisted["safety"]["not_large_model_serving"])
        self.assertTrue(persisted["safety"]["not_arbitrary_public_prompt_serving"])
        self.assertTrue(persisted["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(persisted["shareable_summary"]["raw_prompt_public"])
        self.assertEqual(persisted["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertFalse(persisted["shareable_summary"]["local_answer_terminal_only"])
        self.assertNotIn("CrowdTensor prompt", json.dumps(persisted, sort_keys=True))
        self.assertFalse(persisted["answer_scope"]["visible_in_terminal"])
        self.assertFalse(persisted["answer_scope"]["terminal_only"])
        self.assertEqual(persisted["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(persisted["answer_scope"]["summary"], cli.SAVED_NO_ANSWER_SCOPE_TEXT)
        markdown = (output_dir / "generate_summary.md").read_text(encoding="utf-8")
        self.assertIn("# CrowdTensor Generate Summary", markdown)
        self.assertIn("- OK: `True`", markdown)
        self.assertIn("- Dry run: `True`", markdown)
        self.assertLess(markdown.index("- Review: "), markdown.index("- OK: "))
        self.assertLess(markdown.index("- Review: "), markdown.index("- Status: "))
        self.assertIn(f"- Inspect first: `{output_dir / 'generate_summary.md'}`", markdown)
        self.assertLess(markdown.index("- Review next: "), markdown.index("- Inspect first: "))
        self.assertLess(markdown.index("- Inspect first: "), markdown.index("- Status: "))
        self.assertLess(markdown.index("- Status: "), markdown.index("- Issue: "))
        self.assertLess(markdown.index("- Issue: "), markdown.index("- OK: "))
        self.assertIn("## What To Do Next", markdown)
        self.assertIn("- State: `preflight-partial`", markdown)
        self.assertIn("- Next step: `run_live_preflight`", markdown)
        self.assertIn(
            "- Attention: `coordinator_preflight_skipped,stage_preflight_skipped - Coordinator live readiness was skipped; rerun the printed dry-run/live preflight before submitting; stage0/stage1 Miner readiness was skipped; rerun the printed stage preflight with an observer token.`",
            markdown,
        )
        self.assertIn("- Recommended: `check generation route` reason=`confirm_live_preflight`", markdown)
        self.assertIn("- Reason: Run live preflight before submitting because readiness was skipped.", markdown)
        self.assertIn("- Copy command: `crowdtensor generate --max-new-tokens 16", markdown)
        self.assertIn(
            "- Prompt input: saved Markdown keeps `<prompt>` placeholders; terminal `review_next` / `recommended_next` render safe local prompt sources for copy/paste when available, and saved commands should prefer `--prompt-file`, `--prompt-stdin`, or `--prompt-texts-file`.",
            markdown,
        )
        self.assertIn(
            "- Terminal prompt scope: human terminal `review_next`, `recommended_next`, and `next[...]` may render inline local prompts for copy/paste. Treat terminal logs as local-private; saved JSON/Markdown keep placeholders.",
            markdown,
        )
        self.assertIn("- Requires env: `CROWDTENSOR_OBSERVER_TOKEN`", markdown)
        self.assertIn("- Safety: saved Markdown keeps prompt placeholders and redacted generated output.", markdown)
        self.assertIn(f"- Safety: saved Markdown keeps prompt placeholders and redacted generated output. {cli.SAVED_NO_ANSWER_SCOPE_TEXT}", markdown)
        self.assertIn(f"- Answer scope note: {cli.SAVED_NO_ANSWER_SCOPE_TEXT}", markdown)
        self.assertIn(f"- Output display note: {cli.LOCAL_OUTPUT_DISPLAY_SCOPE_TEXT}", markdown)
        self.assertIn("- Answer scope: `state=no-local-answer ", markdown)
        self.assertNotIn("rerun without --json for local display", markdown)
        self.assertIn("## Details", markdown)
        self.assertIn(
            "- Status: `preflight-partial: Request shape is valid, but live readiness was skipped. next=run_live_preflight recommendation=check generation route public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            f"- Review: `state=preflight-partial next=run_live_preflight inspect={output_dir / 'generate_summary.md'} recommended=check generation route primary=coordinator_ready_preflight_skipped attention=coordinator_preflight_skipped,stage_preflight_skipped public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Review next: `label=check generation route reason=confirm_live_preflight command=crowdtensor generate --max-new-tokens 16",
            markdown,
        )
        self.assertIn(
            "- Trace: `session=none requests=1 ledger_rows=0 stream_events=0 source=public_swarm_product_cli_v1 public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False`",
            markdown,
        )
        self.assertIn(
            "- Prompt scope: `source=prompt-text count=1 inline_prompt_text=True terminal_next_commands_local_private=True terminal_local_paths=False saved_artifacts_prompt_placeholders=True prompt_file_path_public=False raw_prompt_public=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            f"- Artifacts: `inspect={output_dir / 'generate_summary.md'} json={output_dir / 'generate_summary.json'} markdown={output_dir / 'generate_summary.md'} present=2/2 public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Recommended next: `check generation route` reason=`confirm_live_preflight` command=`crowdtensor generate --max-new-tokens 16",
            markdown,
        )
        self.assertIn("requires=`CROWDTENSOR_OBSERVER_TOKEN`", markdown)
        self.assertIn("Generation request shape is valid", markdown)
        self.assertIn("Raw generated text and generated token ids are redacted", markdown)
        self.assertIn("## Artifacts", markdown)
        self.assertIn(
            "- `generate_summary`: path=`generate_summary.json` present=`True` kind=`crowdtensor_generate_summary`",
            markdown,
        )
        self.assertIn(
            "- `generate_summary_markdown`: path=`generate_summary.md` present=`True` kind=`crowdtensor_generate_summary_markdown`",
            markdown,
        )
        self.assertIn("`check generation route`", markdown)
        self.assertIn("Prompt placeholder `<prompt>` is redacted. To rerun safely", markdown)
        self.assertIn("replace the placeholder with `--prompt-file prompt.txt`", markdown)
        self.assertIn("`printf %s '<prompt>' | ... --prompt-stdin`", markdown)
        self.assertIn("do not paste private prompt text into saved commands", markdown)
        self.assertIn("Set required environment variables before running commands: `CROWDTENSOR_ADMIN_TOKEN, CROWDTENSOR_OBSERVER_TOKEN`.", markdown)
        self.assertNotIn("CrowdTensor prompt", markdown)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn(
            "  status: preflight-partial: Request shape is valid, but live readiness was skipped. next=run_live_preflight recommendation=check generation route public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            f"  review: state=preflight-partial next=run_live_preflight inspect={output_dir / 'generate_summary.md'} recommended=check generation route primary=coordinator_ready_preflight_skipped attention=coordinator_preflight_skipped,stage_preflight_skipped public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  review_next: label=check generation route reason=confirm_live_preflight command=crowdtensor generate --max-new-tokens 16",
            rendered,
        )
        self.assertIn(
            "  attention: coordinator_preflight_skipped,stage_preflight_skipped - Coordinator live readiness was skipped; rerun the printed dry-run/live preflight before submitting; stage0/stage1 Miner readiness was skipped; rerun the printed stage preflight with an observer token.",
            rendered,
        )
        self.assertIn("  action: Generation request shape is valid, but live readiness was skipped", rendered)
        self.assertEqual(rendered.count("  action: "), 1)
        self.assertIn(f"  inspect_first: {output_dir / 'generate_summary.md'}", rendered)
        self.assertLess(rendered.index("  review_next: "), rendered.index("  inspect_first: "))
        self.assertLess(rendered.index("  inspect_first: "), rendered.index("  attention: "))
        self.assertLess(rendered.index("  review: "), rendered.index("  status: "))
        self.assertLess(rendered.index("  attention: "), rendered.index("  action: "))
        self.assertLess(rendered.index("  action: "), rendered.index("  ok: "))
        self.assertLess(rendered.index("  ok: "), rendered.index("  diagnosis: "))
        self.assertIn(
            "  trace: session=none requests=1 ledger_rows=0 stream_events=0 source=public_swarm_product_cli_v1 public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            rendered,
        )
        self.assertIn("  stream: requested=True events=0 dry_run=True", rendered)
        self.assertIn(
            f"  saved_summary: {output_dir / 'generate_summary.json'} markdown={output_dir / 'generate_summary.md'} raw_generated_text_redacted=True public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            f"  artifacts: inspect={output_dir / 'generate_summary.md'} json={output_dir / 'generate_summary.json'} markdown={output_dir / 'generate_summary.md'} present=2/2 public_artifact_safe=True",
            rendered,
        )
        self.assertIn(f"  output_dir: {output_dir}", rendered)
        self.assertIn(
            "  recommended_reason: Run live preflight before submitting because readiness was skipped.",
            rendered,
        )
        self.assertNotIn("stream_events: None", rendered)

    def test_generate_shareable_terminal_persists_artifact_scope_for_prompt_file(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_dir = Path(self._tmp_dir())
        prompt_file = prompt_dir / "private_prompt.txt"
        prompt_file.write_text("CrowdTensor private file prompt", encoding="utf-8")
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            cli.main([
                "generate",
                "--prompt-file",
                str(prompt_file),
                "--shareable-terminal",
                "--coordinator-url",
                "http://127.0.0.1:8787",
                "--dry-run",
                "--skip-live-preflight",
                "--output-dir",
                str(output_dir),
            ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        progress = stderr.getvalue()
        self.assertNotIn(str(prompt_file), rendered)
        self.assertNotIn(str(prompt_file), progress)
        self.assertNotIn("prompt_scope:", rendered)
        self.assertIn("--prompt-file prompt.txt", rendered)
        self.assertIn(
            "  shareable_terminal: enabled=True prompt_sources_redacted=True answer_text_redacted=False public_artifact_safe=True",
            rendered,
        )
        persisted = json.loads((output_dir / "generate_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(
            persisted["shareable_terminal"],
            {
                "enabled": True,
                "prompt_sources_redacted": True,
                "answer_text_redacted": False,
                "public_artifact_safe": True,
            },
        )
        self.assertEqual(persisted["prompt_scope"]["source"], "prompt-file")
        self.assertTrue(persisted["prompt_scope"]["terminal_local_paths"])
        self.assertNotIn(str(prompt_file), json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "generate_summary.md").read_text(encoding="utf-8")
        self.assertIn(
            "- Shareable terminal: `enabled=True prompt_sources_redacted=True answer_text_redacted=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Prompt scope: `source=prompt-file count=1 inline_prompt_text=False terminal_next_commands_local_private=True terminal_local_paths=True",
            markdown,
        )
        self.assertNotIn(str(prompt_file), markdown)

    def test_generate_main_prints_copyable_local_prompt_without_persisting_it(self) -> None:
        prompt = "CrowdTensor prompt"

        def fake_build_product_generate(args: object) -> dict[str, object]:
            del args
            return {
                "schema": "public_swarm_product_cli_v1",
                "ok": True,
                "mode": "generate",
                "diagnosis_codes": ["generate_dry_run_ready"],
                "user_status": {
                    "state": "preflight-ready",
                    "next_step": "submit",
                    "headline": "Generation route is ready.",
                    "public_artifact_safe": True,
                },
                "answer_scope": {
                    "scope_state": "no-local-answer",
                    "terminal_only": False,
                    "visible_in_terminal": False,
                    "saved_json_display": "hash-only",
                    "saved_markdown_display": "hash-only",
                    "public_artifact_safe": True,
                },
                "runtime_options": {
                    "timeout_seconds": 420.0,
                    "poll_interval": 1.0,
                    "http_timeout": 30.0,
                    "admin_results_limit": 50,
                    "public_artifact_safe": True,
                },
                "route": {"route_source": "coordinator-url", "coordinator_url_present": True, "missing_capabilities": []},
                "next_commands": [
                    cli.command_entry(
                        "check generation route",
                        [
                            "crowdtensor",
                            "generate",
                            "--max-new-tokens",
                            "16",
                            "--coordinator-url",
                            "http://127.0.0.1:8787",
                            "--prompt-text",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--dry-run",
                        ],
                    )
                ],
                "recommended_next_command": {
                    **cli.command_entry(
                        "check generation route",
                        [
                            "crowdtensor",
                            "generate",
                            "--max-new-tokens",
                            "16",
                            "--coordinator-url",
                            "http://127.0.0.1:8787",
                            "--prompt-text",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--dry-run",
                        ],
                    ),
                    "reason": "verify_stage_miners",
                    "source_index": 1,
                },
                "review_summary": {
                    "state": "preflight-ready",
                    "next_step": "submit",
                    "recommended_label": "check generation route",
                    "recommended_reason": "verify_stage_miners",
                    "next_command": "crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --dry-run",
                    "public_artifact_safe": True,
                },
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli, "build_product_generate", side_effect=fake_build_product_generate):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                cli.main([
                    "generate",
                    "--coordinator-url",
                    "http://127.0.0.1:8787",
                    "--prompt-text",
                    prompt,
                    "--dry-run",
                ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        progress = stderr.getvalue()
        self.assertIn("checking route and stage readiness", progress)
        self.assertNotIn("request shape only", progress)
        self.assertIn("review, review_next", progress)
        self.assertIn("inspect_first", progress)
        self.assertIn("status/action", progress)
        self.assertIn("later lines include answer_scope", progress)
        self.assertIn("answer_scope_note", progress)
        self.assertIn("output_display_note", progress)
        self.assertIn("runtime_options", progress)
        self.assertIn("redacted JSON/Markdown artifacts", progress)
        self.assertNotIn(prompt, progress)
        self.assertIn(
            "prompt_scope: terminal_next_commands=local-private inline_prompt_text=True terminal_local_paths=False saved_artifacts=prompt-placeholders prefer_prompt_file_or_stdin_for_shareable_logs=True source=prompt-text prompt_file_path_public=False raw_prompt_public=False",
            rendered,
        )
        self.assertLess(rendered.index("  prompt_scope: "), rendered.index("  review_next: "))
        self.assertIn(
            f"review_next: label=check generation route reason=verify_stage_miners command=crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-text '{prompt}' --dry-run",
            rendered,
        )
        self.assertIn(
            f"recommended_next: check generation route reason=verify_stage_miners crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-text '{prompt}' --dry-run",
            rendered,
        )
        self.assertIn(f"next[1] check generation route: crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-text '{prompt}' --dry-run", rendered)
        self.assertNotIn(cli.INFER_PROMPT_PLACEHOLDER, rendered)
        self.assertLess(rendered.index("  status: "), rendered.index("  answer_scope: "))
        self.assertLess(rendered.index("  answer_scope: "), rendered.index("  runtime_options: "))

    def test_generate_shareable_terminal_hides_inline_prompt_and_answer(self) -> None:
        prompt = "CrowdTensor prompt"
        answer = "local generated answer"

        def fake_build_product_generate(args: object) -> dict[str, object]:
            self.assertTrue(getattr(args, "shareable_terminal"))
            return {
                "schema": "public_swarm_product_cli_v1",
                "ok": True,
                "mode": "generate",
                "json_mode": False,
                "diagnosis_codes": ["public_swarm_generate_ready"],
                "route": {"route_source": "coordinator-url", "coordinator_url_present": True, "missing_capabilities": []},
                "generation": {"generated_token_count": 2, "max_new_tokens": 2, "generated_text_hash": "sha256:generated"},
                "result": {
                    "status": "complete",
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "output_count": 1,
                    "display": "local-private",
                    "generated_text_hash": "sha256:generated",
                    "public_artifact_safe": False,
                },
                "output_display": {
                    "terminal_display": "local-private",
                    "terminal_text_available": True,
                    "saved_artifact_display": "hash-only",
                    "json_stdout_display": "hash-only-json",
                    "raw_generated_text_public": False,
                    "public_artifact_safe": True,
                },
                "answer_scope": {
                    "scope_state": "terminal-visible",
                    "terminal_only": True,
                    "visible_in_terminal": True,
                    "saved_json_display": "hash-only",
                    "saved_markdown_display": "hash-only",
                    "public_artifact_safe": True,
                },
                "local_output": {
                    "available": True,
                    "generated_text": answer,
                    "outputs": [{"generated_text": answer, "generated_token_count": 2}],
                    "output_count": 1,
                    "source": "coordinator-validation",
                    "display_only": True,
                    "public_artifact_safe": False,
                },
                "shareable_summary": {
                    "saved_artifacts_public_safe": True,
                    "raw_prompt_public": False,
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                    "local_output_display_only": True,
                    "answer_scope_state": "terminal-visible",
                    "local_answer_terminal_only": True,
                },
                "next_commands": [
                    cli.command_entry(
                        "rerun generation",
                        ["crowdtensor", "generate", "--prompt-text", cli.INFER_PROMPT_PLACEHOLDER],
                    )
                ],
                "recommended_next_command": {
                    **cli.command_entry(
                        "rerun generation",
                        ["crowdtensor", "generate", "--prompt-text", cli.INFER_PROMPT_PLACEHOLDER],
                    ),
                    "reason": "rerun_generation",
                },
                "review_summary": {
                    "state": "completed",
                    "next_step": "rerun_or_review_artifacts",
                    "recommended_label": "rerun generation",
                    "recommended_reason": "rerun_generation",
                    "next_command": "crowdtensor generate --prompt-text '<prompt>'",
                    "public_artifact_safe": True,
                },
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli, "build_product_generate", side_effect=fake_build_product_generate):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                cli.main([
                    "generate",
                    "--prompt-text",
                    prompt,
                    "--shareable-terminal",
                ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        self.assertNotIn(prompt, rendered)
        self.assertNotIn(answer, rendered)
        self.assertNotIn("  answer:", rendered)
        self.assertNotIn("prompt_scope:", rendered)
        self.assertIn("command=crowdtensor generate --prompt-text '<prompt>'", rendered)
        self.assertIn("next[1] rerun generation: crowdtensor generate --prompt-text '<prompt>'", rendered)
        self.assertIn(
            "  shareable_terminal: enabled=True prompt_sources_redacted=True answer_text_redacted=True public_artifact_safe=True",
            rendered,
        )
        self.assertIn("answer_scope: state=shareable-terminal-redacted", rendered)
        self.assertIn(f"answer_scope_note: {cli.SHAREABLE_TERMINAL_ANSWER_SCOPE_TEXT}", rendered)
        self.assertIn(f"output_display_note: {cli.SHAREABLE_TERMINAL_OUTPUT_DISPLAY_SCOPE_TEXT}", rendered)
        self.assertIn("local_output: available=False display_only=False public_artifact_safe=True", rendered)
        self.assertNotIn(prompt, stderr.getvalue())

    def test_generate_shareable_terminal_keeps_safe_stdin_next_command(self) -> None:
        prompt = "Prompt stdin private text"

        def fake_build_product_generate(args: object) -> dict[str, object]:
            self.assertTrue(getattr(args, "shareable_terminal"))
            self.assertTrue(getattr(args, "prompt_stdin"))
            self.assertEqual(getattr(args, "prompt_text"), prompt)
            return {
                "schema": "public_swarm_product_cli_v1",
                "ok": True,
                "mode": "generate",
                "json_mode": False,
                "diagnosis_codes": ["generate_dry_run_ready"],
                "route": {"route_source": "coordinator-url", "coordinator_url_present": True, "missing_capabilities": []},
                "generation": {"generated_token_count": 0, "max_new_tokens": 2, "generated_text_hash": ""},
                "result": {"status": "preflight-ready", "output_count": 0, "display": "hash-only", "public_artifact_safe": True},
                "local_output": {},
                "next_commands": [
                    cli.command_entry(
                        "check generation route",
                        ["crowdtensor", "generate", "--prompt-text", cli.INFER_PROMPT_PLACEHOLDER, "--dry-run"],
                    )
                ],
                "recommended_next_command": {
                    **cli.command_entry(
                        "check generation route",
                        ["crowdtensor", "generate", "--prompt-text", cli.INFER_PROMPT_PLACEHOLDER, "--dry-run"],
                    ),
                    "reason": "verify_stage_miners",
                },
                "review_summary": {
                    "state": "preflight-ready",
                    "next_step": "submit",
                    "recommended_label": "check generation route",
                    "recommended_reason": "verify_stage_miners",
                    "next_command": "crowdtensor generate --prompt-text '<prompt>' --dry-run",
                    "public_artifact_safe": True,
                },
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli.sys, "stdin", io.StringIO(prompt + "\n")):
            with patch.object(cli, "build_product_generate", side_effect=fake_build_product_generate):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    cli.main(["generate", "--prompt-stdin", "--dry-run", "--shareable-terminal"])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        self.assertIn("printf %s '<prompt>' | crowdtensor generate --prompt-stdin --dry-run", rendered)
        self.assertNotIn("--prompt-text '<prompt>'", rendered)
        self.assertNotIn(prompt, rendered)
        self.assertNotIn(prompt, stderr.getvalue())
        self.assertIn(
            "  shareable_terminal: enabled=True prompt_sources_redacted=True answer_text_redacted=False public_artifact_safe=True",
            rendered,
        )

    def test_generate_shareable_terminal_next_commands_preserve_shareable_mode(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "zeta-alpha-generate-needle-927"
        args = cli.parse_args([
            "generate",
            prompt,
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--dry-run",
            "--skip-live-preflight",
            "--shareable-terminal",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        with patch.object(
            cli,
            "request_json_url",
            side_effect=AssertionError("skip-live-preflight should not touch live Coordinator endpoints"),
        ):
            report = cli.build_product_generate(args)

        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertTrue(next_lines, report)
        self.assertTrue(all("--shareable-terminal" in line for line in next_lines), next_lines)
        self.assertIn("--shareable-terminal", report["recommended_next_command"]["command_line"])
        persisted = json.loads((output_dir / "generate_summary.json").read_text(encoding="utf-8"))
        persisted_lines = [item["command_line"] for item in persisted["next_commands"]]
        self.assertTrue(all("--shareable-terminal" in line for line in persisted_lines), persisted_lines)
        markdown = (output_dir / "generate_summary.md").read_text(encoding="utf-8")
        self.assertIn("--shareable-terminal", markdown)
        self.assertNotIn(prompt, json.dumps(report, sort_keys=True))
        self.assertNotIn(prompt, markdown)

    def test_generate_main_prints_copyable_batch_prompt_without_single_prompt_placeholder(self) -> None:
        prompts = "first private prompt,second private prompt"

        def fake_build_product_generate(args: object) -> dict[str, object]:
            del args
            return {
                "schema": "public_swarm_product_cli_v1",
                "ok": True,
                "mode": "generate",
                "diagnosis_codes": ["generate_dry_run_ready"],
                "route": {"route_source": "coordinator-url", "coordinator_url_present": True, "missing_capabilities": []},
                "next_commands": [
                    cli.command_entry(
                        "check generation route",
                        [
                            "crowdtensor",
                            "generate",
                            "--max-new-tokens",
                            "16",
                            "--coordinator-url",
                            "http://127.0.0.1:8787",
                            "--prompt-text",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--prompt-texts",
                            cli.INFER_BATCH_PROMPTS_PLACEHOLDER,
                            "--dry-run",
                        ],
                    )
                ],
                "recommended_next_command": {
                    **cli.command_entry(
                        "check generation route",
                        [
                            "crowdtensor",
                            "generate",
                            "--max-new-tokens",
                            "16",
                            "--coordinator-url",
                            "http://127.0.0.1:8787",
                            "--prompt-text",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--prompt-texts",
                            cli.INFER_BATCH_PROMPTS_PLACEHOLDER,
                            "--dry-run",
                        ],
                    ),
                    "reason": "verify_stage_miners",
                    "source_index": 1,
                },
                "review_summary": {
                    "state": "preflight-ready",
                    "next_step": "submit",
                    "recommended_label": "check generation route",
                    "recommended_reason": "verify_stage_miners",
                    "next_command": "crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --prompt-texts '<prompt-1>,<prompt-2>' --dry-run",
                    "public_artifact_safe": True,
                },
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli, "build_product_generate", side_effect=fake_build_product_generate):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                cli.main([
                    "generate",
                    "--coordinator-url",
                    "http://127.0.0.1:8787",
                    "--prompt-texts",
                    prompts,
                    "--dry-run",
                ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        self.assertIn("prompt_scope: terminal_next_commands=local-private inline_prompt_text=True", rendered)
        self.assertIn(
            f"review_next: label=check generation route reason=verify_stage_miners command=crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-texts '{prompts}' --dry-run",
            rendered,
        )
        self.assertIn(f"recommended_next: check generation route reason=verify_stage_miners crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-texts '{prompts}' --dry-run", rendered)
        self.assertIn(f"--prompt-texts '{prompts}' --dry-run", rendered)
        self.assertNotIn("--prompt-text '<prompt>'", rendered)
        self.assertNotIn(cli.INFER_PROMPT_PLACEHOLDER, rendered)

    def test_generate_main_prints_prompt_texts_file_without_expanding_prompts(self) -> None:
        prompts = ["first private prompt, with comma", "second private prompt"]
        prompt_file = Path(self._tmp_dir()) / "prompts.txt"
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")

        def fake_build_product_generate(args: object) -> dict[str, object]:
            self.assertEqual(getattr(args, "prompt_texts_file"), str(prompt_file))
            self.assertEqual(getattr(args, "prompt_texts_list"), prompts)
            return {
                "schema": "public_swarm_product_cli_v1",
                "ok": True,
                "mode": "generate",
                "diagnosis_codes": ["generate_dry_run_ready"],
                "route": {"route_source": "coordinator-url", "coordinator_url_present": True, "missing_capabilities": []},
                "next_commands": [
                    cli.command_entry(
                        "check generation route",
                        [
                            "crowdtensor",
                            "generate",
                            "--max-new-tokens",
                            "16",
                            "--coordinator-url",
                            "http://127.0.0.1:8787",
                            "--prompt-text",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--prompt-texts",
                            cli.INFER_BATCH_PROMPTS_PLACEHOLDER,
                            "--dry-run",
                        ],
                    )
                ],
                "recommended_next_command": {
                    **cli.command_entry(
                        "check generation route",
                        [
                            "crowdtensor",
                            "generate",
                            "--max-new-tokens",
                            "16",
                            "--coordinator-url",
                            "http://127.0.0.1:8787",
                            "--prompt-text",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--prompt-texts",
                            cli.INFER_BATCH_PROMPTS_PLACEHOLDER,
                            "--dry-run",
                        ],
                    ),
                    "reason": "verify_stage_miners",
                    "source_index": 1,
                },
                "review_summary": {
                    "state": "preflight-ready",
                    "next_step": "submit",
                    "recommended_label": "check generation route",
                    "recommended_reason": "verify_stage_miners",
                    "next_command": "crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --prompt-texts '<prompt-1>,<prompt-2>' --dry-run",
                    "public_artifact_safe": True,
                },
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli, "build_product_generate", side_effect=fake_build_product_generate):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                cli.main([
                    "generate",
                    "--coordinator-url",
                    "http://127.0.0.1:8787",
                    "--prompt-texts-file",
                    str(prompt_file),
                    "--dry-run",
                ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        progress = stderr.getvalue()
        self.assertIn(f"--prompt-texts-file {prompt_file}", rendered)
        self.assertIn(
            "prompt_scope: terminal_next_commands=local-private inline_prompt_text=False terminal_local_paths=True saved_artifacts=prompt-placeholders prefer_prompt_file_or_stdin_for_shareable_logs=True source=prompt-texts-file prompt_file_path_public=False raw_prompt_public=False",
            rendered,
        )
        self.assertNotIn("--prompt-text '<prompt>'", rendered)
        self.assertNotIn("--prompt-texts '<prompt-1>,<prompt-2>'", rendered)
        for prompt in prompts:
            self.assertNotIn(prompt, rendered)
            self.assertNotIn(prompt, progress)

    def test_generate_main_prints_prompt_file_without_expanding_prompt(self) -> None:
        prompt = "Prompt file private text"
        prompt_file = Path(self._tmp_dir()) / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        def fake_build_product_generate(args: object) -> dict[str, object]:
            self.assertEqual(getattr(args, "prompt_text"), prompt)
            self.assertEqual(getattr(args, "prompt_file"), str(prompt_file))
            return {
                "schema": "public_swarm_product_cli_v1",
                "ok": True,
                "mode": "generate",
                "diagnosis_codes": ["generate_dry_run_ready"],
                "route": {"route_source": "coordinator-url", "coordinator_url_present": True, "missing_capabilities": []},
                "next_commands": [
                    cli.command_entry(
                        "check generation route",
                        [
                            "crowdtensor",
                            "generate",
                            "--max-new-tokens",
                            "16",
                            "--coordinator-url",
                            "http://127.0.0.1:8787",
                            "--prompt-text",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--dry-run",
                        ],
                    )
                ],
                "recommended_next_command": {
                    **cli.command_entry(
                        "check generation route",
                        [
                            "crowdtensor",
                            "generate",
                            "--max-new-tokens",
                            "16",
                            "--coordinator-url",
                            "http://127.0.0.1:8787",
                            "--prompt-text",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--dry-run",
                        ],
                    ),
                    "reason": "verify_stage_miners",
                    "source_index": 1,
                },
                "review_summary": {
                    "state": "preflight-ready",
                    "next_step": "submit",
                    "recommended_label": "check generation route",
                    "recommended_reason": "verify_stage_miners",
                    "next_command": "crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --dry-run",
                    "public_artifact_safe": True,
                },
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli, "build_product_generate", side_effect=fake_build_product_generate):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                cli.main([
                    "generate",
                    "--coordinator-url",
                    "http://127.0.0.1:8787",
                    "--prompt-file",
                    str(prompt_file),
                    "--dry-run",
                ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        progress = stderr.getvalue()
        self.assertIn(f"--prompt-file {prompt_file}", rendered)
        self.assertIn(
            "prompt_scope: terminal_next_commands=local-private inline_prompt_text=False terminal_local_paths=True saved_artifacts=prompt-placeholders prefer_prompt_file_or_stdin_for_shareable_logs=True source=prompt-file prompt_file_path_public=False raw_prompt_public=False",
            rendered,
        )
        self.assertIn("terminal_local_paths=True", rendered)
        self.assertNotIn("--prompt-text '<prompt>'", rendered)
        self.assertNotIn(prompt, rendered)
        self.assertNotIn(prompt, progress)

    def test_generate_prompt_file_does_not_pollute_startup_commands(self) -> None:
        prompt_file = Path(self._tmp_dir()) / "prompt.txt"
        prompt_file.write_text("Prompt file private text", encoding="utf-8")
        report = {
            "local_prompt_file": str(prompt_file),
            "next_commands": [
                cli.command_entry("start Coordinator", ["crowdtensor", "serve", "--run"]),
                cli.command_entry("check generation route", ["crowdtensor", "generate", "--prompt-text", cli.INFER_PROMPT_PLACEHOLDER, "--dry-run"]),
            ],
        }

        self.assertEqual(
            cli.local_generate_command_line(report["next_commands"][0], report),
            "crowdtensor serve --run",
        )
        self.assertEqual(
            cli.local_generate_command_line(report["next_commands"][1], report),
            f"crowdtensor generate --prompt-file {prompt_file} --dry-run",
        )

    def test_generate_main_prints_prompt_stdin_without_expanding_prompt(self) -> None:
        prompt = "Prompt stdin private text"

        def fake_build_product_generate(args: object) -> dict[str, object]:
            self.assertEqual(getattr(args, "prompt_text"), prompt)
            self.assertTrue(getattr(args, "prompt_stdin"))
            return {
                "schema": "public_swarm_product_cli_v1",
                "ok": True,
                "mode": "generate",
                "diagnosis_codes": ["generate_dry_run_ready"],
                "route": {"route_source": "coordinator-url", "coordinator_url_present": True, "missing_capabilities": []},
                "next_commands": [
                    cli.command_entry(
                        "check generation route",
                        [
                            "crowdtensor",
                            "generate",
                            "--max-new-tokens",
                            "16",
                            "--coordinator-url",
                            "http://127.0.0.1:8787",
                            "--prompt-text",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--dry-run",
                        ],
                    )
                ],
                "recommended_next_command": {
                    **cli.command_entry(
                        "check generation route",
                        [
                            "crowdtensor",
                            "generate",
                            "--max-new-tokens",
                            "16",
                            "--coordinator-url",
                            "http://127.0.0.1:8787",
                            "--prompt-text",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--dry-run",
                        ],
                    ),
                    "reason": "verify_stage_miners",
                    "source_index": 1,
                },
                "review_summary": {
                    "state": "preflight-ready",
                    "next_step": "submit",
                    "recommended_label": "check generation route",
                    "recommended_reason": "verify_stage_miners",
                    "next_command": "crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --dry-run",
                    "public_artifact_safe": True,
                },
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli.sys, "stdin", io.StringIO(prompt + "\n")):
            with patch.object(cli, "build_product_generate", side_effect=fake_build_product_generate):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    cli.main([
                        "generate",
                        "--coordinator-url",
                        "http://127.0.0.1:8787",
                        "--prompt-stdin",
                        "--dry-run",
                    ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        progress = stderr.getvalue()
        self.assertIn("printf %s '<prompt>' | crowdtensor generate", rendered)
        self.assertIn("--prompt-stdin", rendered)
        self.assertNotIn("--prompt-text '<prompt>'", rendered)
        self.assertNotIn(prompt, rendered)
        self.assertNotIn(prompt, progress)

    def test_generate_prompt_stdin_does_not_pollute_startup_or_env_commands(self) -> None:
        report = {
            "local_prompt_stdin": True,
            "next_commands": [
                cli.command_entry("start Coordinator", ["crowdtensor", "serve", "--run"]),
                cli.command_entry("start stage0 Miner", ["crowdtensor", "join", "--stage", "stage0", "--run"]),
                cli.command_entry(
                    "submit generation",
                    ["crowdtensor", "generate", "--prompt-text", cli.INFER_PROMPT_PLACEHOLDER],
                    requires_env=["CROWDTENSOR_ADMIN_TOKEN"],
                ),
            ],
        }

        self.assertEqual(
            cli.local_generate_command_line(report["next_commands"][0], report),
            "crowdtensor serve --run",
        )
        self.assertEqual(
            cli.local_generate_command_line(report["next_commands"][1], report),
            "crowdtensor join --stage stage0 --run",
        )
        rendered_submit = cli.human_next_command_line(
            report["next_commands"][2],
            cli.local_generate_command_line(report["next_commands"][2], report),
        )
        self.assertEqual(
            rendered_submit,
            "printf %s '<prompt>' | CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor generate --prompt-stdin",
        )

    def test_generate_without_admin_token_prints_credential_start_hint(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "CrowdTensor prompt"

        def fake_build_product_generate(args: object) -> dict[str, object]:
            self.assertEqual(getattr(args, "admin_token"), "")
            return {
                "schema": "public_swarm_product_cli_v1",
                "ok": False,
                "mode": "generate",
                "operator_action": "Pass --admin-token or set CROWDTENSOR_ADMIN_TOKEN.",
                "output_dir": str(output_dir),
                "diagnosis_codes": ["admin_token_required"],
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli, "build_product_generate", side_effect=fake_build_product_generate):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                cli.main([
                    "generate",
                    prompt,
                    "--coordinator-url",
                    "http://127.0.0.1:8787",
                    "--output-dir",
                    str(output_dir),
                    "--max-new-tokens",
                    "2",
                ])

        self.assertEqual(raised.exception.code, 1)
        progress = stderr.getvalue()
        self.assertIn("checking credentials and request requirements", progress)
        self.assertNotIn("submitting a bounded generation request", progress)
        self.assertNotIn(prompt, progress)

    def test_generate_review_next_fallback_cleans_batch_prompt_conflict(self) -> None:
        prompts = "first private prompt,second private prompt"
        report = {
            "review_summary": {
                "recommended_label": "check generation route",
                "recommended_reason": "verify_stage_miners",
                "next_command": "crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --prompt-texts '<prompt-1>,<prompt-2>' --dry-run",
                "public_artifact_safe": True,
            },
            "local_prompt_texts": prompts,
        }

        summary = cli.display_review_summary(report, cli.local_generate_command_line)

        self.assertEqual(
            summary["next_command"],
            f"crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-texts '{prompts}' --dry-run",
        )
        self.assertNotIn("--prompt-text '<prompt>'", cli.review_next_command_text(summary))

    def test_generate_accepts_positional_prompt_like_infer(self) -> None:
        prompt = "CrowdTensor positional prompt"
        args = cli.parse_args([
            "generate",
            prompt,
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--json",
        ])

        self.assertEqual(args.prompt_text, prompt)
        report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("coordinator_route_missing", report["diagnosis_codes"])
        self.assertEqual(report["issue_summary"]["primary_code"], "coordinator_route_missing")
        self.assertEqual(report["review_summary"]["primary_code"], "coordinator_route_missing")
        self.assertEqual(report["session_request"]["prompt_chars"], len(prompt))
        self.assertNotIn(prompt, json.dumps(report, sort_keys=True))
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            "crowdtensor generate --max-new-tokens 2 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}",
            next_lines,
        )

    def test_generate_accepts_prompt_file_without_persisting_prompt_text(self) -> None:
        prompt = "Prompt file product request"
        prompt_file = Path(self._tmp_dir()) / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        args = cli.parse_args([
            "generate",
            "--prompt-file",
            str(prompt_file),
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--json",
        ])

        self.assertEqual(args.prompt_text, prompt)
        self.assertEqual(args.prompt_file, str(prompt_file))
        report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["session_request"]["prompt_chars"], len(prompt))
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn(prompt, encoded)
        self.assertNotIn(str(prompt_file), encoded)
        self.assertIn("prompt_hash", encoded)
        self.assertEqual(report["prompt_file"], cli.INFER_PROMPT_FILE_PLACEHOLDER)
        markdown = (Path(report["saved_summary"]["markdown_path"])).read_text(encoding="utf-8")
        self.assertIn("Prompt file placeholder `prompt.txt` is redacted", markdown)
        self.assertIn("Create `prompt.txt` with the local prompt", markdown)
        self.assertIn("replace it with a local prompt file path", markdown)

    def test_generate_accepts_prompt_stdin_without_persisting_prompt_text(self) -> None:
        prompt = "Prompt stdin product request"
        with patch.object(cli.sys, "stdin", io.StringIO(prompt + "\n")):
            args = cli.parse_args([
                "generate",
                "--prompt-stdin",
                "--admin-token",
                "admin-secret",
                "--max-new-tokens",
                "2",
                "--json",
            ])

        self.assertEqual(args.prompt_text, prompt)
        self.assertTrue(args.prompt_stdin)
        report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["session_request"]["prompt_chars"], len(prompt))
        self.assertNotIn(prompt, json.dumps(report, sort_keys=True))
        self.assertIn("prompt_hash", json.dumps(report, sort_keys=True))

    def test_generate_rejects_long_prompt_file_without_echoing_prompt(self) -> None:
        private_prompt = "x" * 257
        prompt_file = Path(self._tmp_dir()) / "long-prompt.txt"
        prompt_file.write_text(private_prompt, encoding="utf-8")
        with self.assertRaises(SystemExit) as raised:
            cli.parse_args(["generate", "--prompt-file", str(prompt_file)])

        error = str(raised.exception)
        self.assertEqual(error, "prompt_file must be at most 256 characters")
        self.assertNotIn(private_prompt, error)

    def test_generate_rejects_long_prompt_stdin_without_echoing_prompt(self) -> None:
        private_prompt = "x" * 257
        with patch.object(cli.sys, "stdin", io.StringIO(private_prompt)):
            with self.assertRaises(SystemExit) as raised:
                cli.parse_args(["generate", "--prompt-stdin"])

        error = str(raised.exception)
        self.assertEqual(error, "prompt_stdin must be at most 256 characters")
        self.assertNotIn(private_prompt, error)

    def test_generate_accepts_prompt_texts_file_without_persisting_prompt_text(self) -> None:
        prompts = ["Prompt file batch, with comma", "Second batch prompt"]
        prompt_file = Path(self._tmp_dir()) / "prompts.txt"
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        args = cli.parse_args([
            "generate",
            "--prompt-texts-file",
            str(prompt_file),
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--json",
        ])

        self.assertEqual(args.prompt_text, "")
        self.assertEqual(args.prompt_texts_file, str(prompt_file))
        self.assertEqual(args.prompt_texts_list, prompts)
        self.assertEqual(cli.prompt_list_from_args(args), prompts)
        report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["batch"]["request_count"], 2)
        self.assertEqual(report["session_request"]["prompt_char_counts"], [len(prompt) for prompt in prompts])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertTrue(any(f"--prompt-texts-file {cli.INFER_PROMPT_TEXTS_FILE_PLACEHOLDER}" in line for line in next_lines))
        self.assertFalse(any("--prompt-texts '<prompt-1>,<prompt-2>'" in line for line in next_lines))
        encoded = json.dumps(report, sort_keys=True)
        for prompt in prompts:
            self.assertNotIn(prompt, encoded)
        self.assertNotIn(str(prompt_file), encoded)
        self.assertIn("prompt_hashes", encoded)
        self.assertEqual(report["prompt_texts_file"], cli.INFER_PROMPT_TEXTS_FILE_PLACEHOLDER)
        markdown = (Path(report["saved_summary"]["markdown_path"])).read_text(encoding="utf-8")
        self.assertIn("Batch prompt file placeholder `prompts.txt` is redacted", markdown)
        self.assertIn("Create `prompts.txt` with one prompt per non-empty line", markdown)
        self.assertIn("replace it with a local batch prompt file path", markdown)

    def test_generate_prompt_file_next_commands_keep_file_source_without_prompt_text(self) -> None:
        prompt = "Generate file prompt stays private"
        prompt_file = Path(self._tmp_dir()) / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        args = cli.parse_args([
            "generate",
            "--prompt-file",
            str(prompt_file),
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--dry-run",
            "--skip-live-preflight",
            "--json",
        ])

        report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertTrue(any(f"--prompt-file {cli.INFER_PROMPT_FILE_PLACEHOLDER}" in line for line in next_lines))
        self.assertFalse(any("'<prompt>'" in line for line in next_lines))
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn(prompt, encoded)
        self.assertNotIn(str(prompt_file), encoded)
        self.assertEqual(report["prompt_file"], cli.INFER_PROMPT_FILE_PLACEHOLDER)

    def test_generate_prompt_stdin_next_commands_keep_stdin_source_without_prompt_text(self) -> None:
        prompt = "Generate stdin prompt stays private"
        output_dir = Path(self._tmp_dir())
        with patch.object(cli.sys, "stdin", io.StringIO(prompt + "\n")):
            args = cli.parse_args([
                "generate",
                "--prompt-stdin",
                "--coordinator-url",
                "http://127.0.0.1:8787",
                "--dry-run",
                "--skip-live-preflight",
                "--output-dir",
                str(output_dir),
                "--json",
            ])

        report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["prompt_stdin"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertTrue(any("--prompt-stdin" in line for line in next_lines))
        self.assertFalse(any("--prompt-text '<prompt>'" in line for line in next_lines))
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn(prompt, encoded)
        markdown = (output_dir / "generate_summary.md").read_text(encoding="utf-8")
        self.assertIn("this command reads stdin", markdown)
        self.assertIn("- Review next: `label=check generation route reason=confirm_live_preflight command=printf %s '<prompt>' | crowdtensor generate", markdown)
        self.assertIn("- Copy command: `printf %s '<prompt>' | crowdtensor generate", markdown)
        self.assertIn("1. `check generation route`: `printf %s '<prompt>' | crowdtensor generate", markdown)
        self.assertIn("2. `submit generation after live preflight`: `printf %s '<prompt>' | CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor generate", markdown)
        self.assertIn("printf %s '<prompt>' | crowdtensor generate", markdown)
        self.assertIn("Commands with `--prompt-stdin` read the prompt from stdin", markdown)
        self.assertIn(
            "- Terminal prompt scope: this stdin command is safe to copy from saved Markdown after replacing `<prompt>` locally; saved JSON/Markdown do not include raw prompt text.",
            markdown,
        )
        self.assertNotIn("may render inline local prompts for copy/paste", markdown)
        self.assertNotIn("printf %s '<prompt>' | printf %s '<prompt>'", markdown)
        self.assertNotIn(prompt, markdown)

    def test_generate_rejects_empty_prompt_stdin(self) -> None:
        with patch.object(cli.sys, "stdin", io.StringIO("")):
            with self.assertRaises(SystemExit) as raised:
                cli.parse_args(["generate", "--prompt-stdin"])

        self.assertEqual(str(raised.exception), "prompt_stdin is empty")

    def test_generate_rejects_empty_prompt_texts_file(self) -> None:
        prompt_file = Path(self._tmp_dir()) / "empty-prompts.txt"
        prompt_file.write_text("\n\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as raised:
            cli.parse_args(["generate", "--prompt-texts-file", str(prompt_file)])

        self.assertEqual(str(raised.exception), "prompt_texts_file is empty")

    def test_generate_rejects_long_prompt_texts_file_line_with_line_number(self) -> None:
        prompt_file = Path(self._tmp_dir()) / "long-prompts.txt"
        private_prompt = "x" * 257
        prompt_file.write_text(f"short prompt\n{private_prompt}\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as raised:
            cli.parse_args(["generate", "--prompt-texts-file", str(prompt_file)])

        error = str(raised.exception)
        self.assertEqual(error, "prompt_texts_file line 2 must be at most 256 characters")
        self.assertNotIn(private_prompt, error)

    def test_infer_accepts_prompt_text_alias_like_generate(self) -> None:
        prompt = "CrowdTensor flag prompt"
        args = cli.parse_args([
            "infer",
            "--prompt",
            prompt,
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--dry-run",
            "--json",
        ])

        self.assertEqual(args.prompt_text, prompt)
        self.assertEqual(args.prompt_text_arg, "")
        self.assertEqual(args.prompt_texts, "")

    def test_infer_accepts_prompt_file_without_persisting_prompt_text(self) -> None:
        prompt = "Infer prompt file request"
        prompt_file = Path(self._tmp_dir()) / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        args = cli.parse_args([
            "infer",
            "--prompt-file",
            str(prompt_file),
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--observer-token",
            "observer-secret",
            "--dry-run",
            "--json",
        ])

        self.assertEqual(args.prompt_text, prompt)
        self.assertEqual(args.prompt_file, str(prompt_file))
        with patch.object(cli, "request_json_url", return_value={
            "schema": "ready_v1",
            "service": "crowdtensord-coordinator",
            "protocol": "runtime_contract_v1",
        }):
            report = cli.build_infer(args)

        self.assertFalse(report["prompt"]["raw_prompt_public"])
        self.assertEqual(report["prompt"]["prompt_count"], 1)
        self.assertNotIn(prompt, json.dumps(report, sort_keys=True))

    def test_infer_accepts_prompt_stdin_without_persisting_prompt_text(self) -> None:
        prompt = "Infer prompt stdin request"
        with patch.object(cli.sys, "stdin", io.StringIO(prompt + "\n")):
            args = cli.parse_args([
                "infer",
                "--prompt-stdin",
                "--mode",
                "existing",
                "--coordinator-url",
                "http://127.0.0.1:8787",
                "--observer-token",
                "observer-secret",
                "--dry-run",
                "--json",
            ])

        self.assertEqual(args.prompt_text, prompt)
        self.assertTrue(args.prompt_stdin)
        with patch.object(cli, "request_json_url", return_value={
            "schema": "ready_v1",
            "service": "crowdtensord-coordinator",
            "protocol": "runtime_contract_v1",
        }):
            report = cli.build_infer(args)

        self.assertFalse(report["prompt"]["raw_prompt_public"])
        self.assertEqual(report["prompt"]["prompt_count"], 1)
        self.assertNotIn(prompt, json.dumps(report, sort_keys=True))

    def test_infer_rejects_long_prompt_file_without_echoing_prompt(self) -> None:
        private_prompt = "x" * 257
        prompt_file = Path(self._tmp_dir()) / "long-infer-prompt.txt"
        prompt_file.write_text(private_prompt, encoding="utf-8")
        with self.assertRaises(SystemExit) as raised:
            cli.parse_args(["infer", "--prompt-file", str(prompt_file)])

        error = str(raised.exception)
        self.assertEqual(error, "prompt_file must be at most 256 characters")
        self.assertNotIn(private_prompt, error)

    def test_infer_rejects_long_prompt_stdin_without_echoing_prompt(self) -> None:
        private_prompt = "x" * 257
        with patch.object(cli.sys, "stdin", io.StringIO(private_prompt)):
            with self.assertRaises(SystemExit) as raised:
                cli.parse_args(["infer", "--prompt-stdin"])

        error = str(raised.exception)
        self.assertEqual(error, "prompt_stdin must be at most 256 characters")
        self.assertNotIn(private_prompt, error)

    def test_infer_accepts_prompt_texts_file_without_persisting_prompt_text(self) -> None:
        prompts = ["Infer batch prompt, with comma", "Second infer prompt"]
        prompt_file = Path(self._tmp_dir()) / "infer-prompts.txt"
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        args = cli.parse_args([
            "infer",
            "--prompt-texts-file",
            str(prompt_file),
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--observer-token",
            "observer-secret",
            "--dry-run",
            "--json",
        ])

        self.assertEqual(args.prompt_texts_file, str(prompt_file))
        self.assertEqual(args.prompt_texts_list, prompts)
        with patch.object(cli, "request_json_url", return_value={
            "schema": "ready_v1",
            "service": "crowdtensord-coordinator",
            "protocol": "runtime_contract_v1",
        }):
            report = cli.build_infer(args)

        self.assertFalse(report["prompt"]["raw_prompt_public"])
        self.assertEqual(report["prompt"]["prompt_count"], 2)
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertTrue(any(f"--prompt-texts-file {cli.INFER_PROMPT_TEXTS_FILE_PLACEHOLDER}" in line for line in next_lines))
        self.assertFalse(any("--prompt-texts '<prompt-1>,<prompt-2>'" in line for line in next_lines))
        encoded = json.dumps(report, sort_keys=True)
        for prompt in prompts:
            self.assertNotIn(prompt, encoded)
        self.assertNotIn(str(prompt_file), encoded)

    def test_infer_prompt_file_next_commands_keep_file_source_without_prompt_text(self) -> None:
        prompt = "Infer file prompt stays private"
        prompt_file = Path(self._tmp_dir()) / "infer-prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        args = cli.parse_args([
            "infer",
            "--prompt-file",
            str(prompt_file),
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--dry-run",
            "--skip-live-preflight",
            "--json",
        ])

        report = cli.build_infer(args)

        self.assertTrue(report["ok"], report)
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertTrue(any(f"--prompt-file {cli.INFER_PROMPT_FILE_PLACEHOLDER}" in line for line in next_lines))
        self.assertFalse(any("infer '<prompt>'" in line for line in next_lines))
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn(prompt, encoded)
        self.assertNotIn(str(prompt_file), encoded)

    def test_infer_prompt_stdin_next_commands_keep_stdin_source_without_prompt_text(self) -> None:
        prompt = "Infer stdin prompt stays private"
        output_dir = Path(self._tmp_dir())
        with patch.object(cli.sys, "stdin", io.StringIO(prompt + "\n")):
            args = cli.parse_args([
                "infer",
                "--prompt-stdin",
                "--mode",
                "existing",
                "--coordinator-url",
                "http://127.0.0.1:8787",
                "--dry-run",
                "--skip-live-preflight",
                "--output-dir",
                str(output_dir),
                "--json",
            ])

        report = cli.build_infer(args)

        self.assertTrue(report["ok"], report)
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertTrue(any("--prompt-stdin" in line for line in next_lines))
        self.assertFalse(any("infer '<prompt>'" in line for line in next_lines))
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn(prompt, encoded)
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("this command reads stdin", markdown)
        self.assertIn("- Review next: `label=check existing swarm reason=confirm_live_preflight command=printf %s '<prompt>' | crowdtensor infer", markdown)
        self.assertIn("- Copy command: `printf %s '<prompt>' | crowdtensor infer", markdown)
        self.assertIn("1. `check existing swarm`: `printf %s '<prompt>' | crowdtensor infer", markdown)
        self.assertIn("2. `submit inference after live preflight`: `printf %s '<prompt>' | CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor infer", markdown)
        self.assertIn("printf %s '<prompt>' | crowdtensor infer", markdown)
        self.assertIn("Commands with `--prompt-stdin` read the prompt from stdin", markdown)
        self.assertIn(
            "- Terminal prompt scope: this stdin command is safe to copy from saved Markdown after replacing `<prompt>` locally; saved JSON/Markdown do not include raw prompt text.",
            markdown,
        )
        self.assertNotIn("may render inline local prompts for copy/paste", markdown)
        self.assertNotIn("printf %s '<prompt>' | printf %s '<prompt>'", markdown)
        self.assertNotIn(prompt, markdown)

    def test_generate_rejects_ambiguous_prompt_sources(self) -> None:
        prompt_file = Path(self._tmp_dir()) / "prompt.txt"
        prompt_file.write_text("file prompt", encoding="utf-8")
        prompts_file = Path(self._tmp_dir()) / "prompts.txt"
        prompts_file.write_text("first prompt\nsecond prompt\n", encoding="utf-8")
        cases = [
            ["generate", "positional prompt", "--prompt-text", "flag prompt"],
            ["generate", "positional prompt", "--prompt-texts", "first prompt,second prompt"],
            ["generate", "--prompt-text", "flag prompt", "--prompt-texts", "first prompt,second prompt"],
            ["generate", "--prompt-file", str(prompt_file), "--prompt-text", "flag prompt"],
            ["generate", "positional prompt", "--prompt-file", str(prompt_file)],
            ["generate", "--prompt-stdin", "--prompt-text", "flag prompt"],
            ["generate", "--prompt-stdin", "--prompt-file", str(prompt_file)],
            ["generate", "--prompt-stdin", "--prompt-texts", "first prompt,second prompt"],
            ["generate", "--prompt-texts-file", str(prompts_file), "--prompt-texts", "first prompt,second prompt"],
            ["generate", "--prompt-texts-file", str(prompts_file), "--prompt-file", str(prompt_file)],
        ]
        for argv in cases:
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as raised:
                cli.parse_args(argv)
            self.assertEqual(
                str(raised.exception),
                "generate accepts one prompt source: positional prompt, --prompt-text, --prompt-file, --prompt-stdin, --prompt-texts, or --prompt-texts-file",
            )

    def test_product_generate_dry_run_has_safe_default_prompt(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--dry-run",
            "--skip-live-preflight",
            "--json",
        ])

        report = cli.build_product_generate(args)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["session_request"]["prompt_chars"], len(cli.DEFAULT_PRODUCT_GENERATE_PROMPT))
        self.assertNotIn(cli.DEFAULT_PRODUCT_GENERATE_PROMPT, encoded)
        self.assertIn("prompt_hash", encoded)
        self.assertFalse(report["output_dir_explicit"])
        self.assertFalse(any("--output-dir" in item["command_line"] for item in report["next_commands"]))

    def test_product_generate_detects_output_dir_from_real_argv(self) -> None:
        output_dir = Path(self._tmp_dir())
        argv = [
            "crowdtensor",
            "generate",
            "--prompt-text",
            "CrowdTensor prompt",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--dry-run",
            "--skip-live-preflight",
            f"--output-dir={output_dir}",
        ]

        with patch.object(sys, "argv", argv):
            args = cli.parse_args()

        report = cli.build_product_generate(args)

        self.assertTrue(report["output_dir_explicit"])
        self.assertTrue(any(f"--output-dir {output_dir}" in item["command_line"] for item in report["next_commands"]))
        self.assertEqual(report["artifact_summary"]["inspect_first"], str(output_dir / "generate_summary.md"))

    def test_product_generate_dry_run_checks_coordinator_and_stage_preflight(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-text",
            "CrowdTensor prompt",
            "--observer-token",
            "observer-secret",
            "--dry-run",
            "--json",
        ])
        calls: list[tuple[str, str, str]] = []

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            observer_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del payload, admin_token, timeout
            calls.append((method, base_url, path))
            if path == "/ready":
                return {"schema": "ready_v1", "service": "crowdtensord", "protocol": "runtime_contract_v1"}
            if path == "/state":
                self.assertEqual(observer_token, "observer-secret")
                return {
                    "miner_profiles": {
                        "stage0": {
                            "last_capabilities": {
                                "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]
                            }
                        },
                        "stage1": {
                            "last_capabilities": {
                                "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]
                            }
                        },
                    }
                }
            raise AssertionError(path)

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["ready_to_submit"]["ok"])
        self.assertTrue(report["ready_to_submit"]["fully_verified"])
        self.assertEqual(report["ready_to_submit"]["readiness_label"], "verified")
        self.assertEqual(
            report["ready_to_submit"]["readiness_summary"],
            "Route, Coordinator, and distinct stage Miners are verified.",
        )
        self.assertEqual(report["ready_to_submit"]["next_step"], "submit")
        self.assertEqual(report["ready_to_submit"]["stage_verification"], "ready")
        self.assertEqual(report["ready_to_submit"]["warning_codes"], [])
        self.assertTrue(report["coordinator_ready"]["ok"])
        self.assertTrue(report["stage_preflight"]["ok"])
        self.assertEqual(report["stage_preflight"]["matched_miner_count"], 2)
        self.assertIn("coordinator_ready_preflight_ready", report["diagnosis_codes"])
        self.assertIn("stage_preflight_ready", report["diagnosis_codes"])
        self.assertIn(("GET", "http://127.0.0.1:8787", "/ready"), calls)
        self.assertIn(("GET", "http://127.0.0.1:8787", "/state"), calls)
        self.assertEqual(report["operator_action"], "Dry-run is verified; run the printed submit generation next command.")
        self.assertEqual(report["user_status"]["state"], "preflight-ready")
        self.assertEqual(report["user_status"]["headline"], "Preflight passed; submit generation next.")
        self.assertEqual(report["user_status"]["next_step"], "submit")
        self.assertEqual(report["user_status"]["recommended_label"], "submit generation")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn(
            "  status: preflight-ready: Preflight passed; submit generation next. next=submit recommendation=submit generation public_artifact_safe=True",
            rendered,
        )
        self.assertIn("  route: source=coordinator-url coordinator=True catalog_missing=not_used", rendered)
        self.assertIn("  coordinator_ready: ready service=crowdtensord protocol=runtime_contract_v1", rendered)
        self.assertIn("  stage_preflight: checked=True ok=True matched_miners=2 missing=none", rendered)
        self.assertIn("  ready_to_submit: ready label=verified fully_verified=True route=True coordinator=ready stage=ready stage_verification=ready next_step=submit warnings=none", rendered)
        self.assertIn("  readiness: Route, Coordinator, and distinct stage Miners are verified.", rendered)
        self.assertIn(
            "  recommended_next: submit generation reason=submit_verified_generation CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>'  # requires CROWDTENSOR_ADMIN_TOKEN",
            rendered,
        )
        self.assertIn("  next[1] check generation route: crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}", rendered)
        self.assertIn("# requires CROWDTENSOR_OBSERVER_TOKEN", rendered)
        self.assertIn("  next[2] submit generation: CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>'  # requires CROWDTENSOR_ADMIN_TOKEN", rendered)

    def test_product_generate_dry_run_without_observer_token_is_partial_stage_preflight(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-text",
            "CrowdTensor prompt",
            "--dry-run",
            "--json",
        ])
        calls: list[tuple[str, str, str]] = []

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            observer_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del payload, admin_token, observer_token, timeout
            calls.append((method, base_url, path))
            if path == "/ready":
                return {"schema": "ready_v1", "service": "crowdtensord", "protocol": "runtime_contract_v1"}
            raise AssertionError(path)

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(calls, [("GET", "http://127.0.0.1:8787", "/ready")])
        self.assertTrue(report["ready_to_submit"]["ok"])
        self.assertFalse(report["ready_to_submit"]["fully_verified"])
        self.assertEqual(report["ready_to_submit"]["readiness_label"], "partial")
        self.assertEqual(
            report["ready_to_submit"]["readiness_summary"],
            "Request can be submitted, but stage Miner readiness is not fully verified.",
        )
        self.assertEqual(report["ready_to_submit"]["next_step"], "run_stage_preflight")
        self.assertEqual(report["ready_to_submit"]["stage_verification"], "skipped")
        self.assertEqual(report["ready_to_submit"]["warning_codes"], ["stage_preflight_skipped"])
        self.assertFalse(report["ready_to_submit"]["stage_preflight_required"])
        self.assertFalse(report["stage_preflight"]["checked"])
        self.assertEqual(report["stage_preflight"]["reason"], "observer_token_missing")
        self.assertEqual(report["stage_preflight"]["source"], "not-checked")
        self.assertIn("coordinator_ready_preflight_ready", report["diagnosis_codes"])
        self.assertIn("stage_preflight_skipped", report["diagnosis_codes"])
        self.assertIn("generate_dry_run_partial", report["diagnosis_codes"])
        self.assertNotIn("generate_dry_run_ready", report["diagnosis_codes"])
        self.assertEqual(
            report["operator_action"],
            "Generation can be submitted, but stage0/stage1 were not fully verified; run the printed stage-preflight next command with --observer-token before the submit next command.",
        )
        self.assertEqual(report["user_status"]["state"], "preflight-partial")
        self.assertEqual(
            report["user_status"]["headline"],
            "Request can be submitted, but stage Miner readiness is not fully verified.",
        )
        self.assertEqual(report["user_status"]["next_step"], "run_stage_preflight")
        self.assertEqual(report["user_status"]["recommended_label"], "check generation route")
        labels = [item["label"] for item in report["next_commands"]]
        self.assertIn("check generation route", labels)
        self.assertIn("submit generation after stage preflight", labels)
        stage_check = next(item for item in report["next_commands"] if item["label"] == "check generation route")
        self.assertEqual(stage_check["requires_env"], ["CROWDTENSOR_OBSERVER_TOKEN"])
        submit = next(item for item in report["next_commands"] if item["label"] == "submit generation after stage preflight")
        self.assertEqual(submit["requires_env"], ["CROWDTENSOR_ADMIN_TOKEN"])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn(
            "  status: preflight-partial: Request can be submitted, but stage Miner readiness is not fully verified. next=run_stage_preflight recommendation=check generation route public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  stage_preflight: checked=False ok=None matched_miners=None missing=not_checked reason=observer_token_missing source=not-checked",
            rendered,
        )
        self.assertIn(
            "  ready_to_submit: ready label=partial fully_verified=False route=True coordinator=ready stage=skipped stage_verification=skipped next_step=run_stage_preflight warnings=stage_preflight_skipped",
            rendered,
        )
        self.assertIn("next[2] submit generation after stage preflight:", rendered)

    def test_product_generate_dry_run_can_skip_live_preflight_for_ci(self) -> None:
        argv = [
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-text",
            "CrowdTensor prompt",
            "--dry-run",
            "--skip-live-preflight",
            "--json",
        ]
        args = cli.parse_args(argv)

        with patch.object(
            cli,
            "request_json_url",
            side_effect=AssertionError("skip-live-preflight should not touch live Coordinator endpoints"),
        ):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertIsNone(report["ready_to_submit"]["ok"])
        self.assertFalse(report["ready_to_submit"]["fully_verified"])
        self.assertEqual(report["ready_to_submit"]["readiness_label"], "skipped")
        self.assertEqual(
            report["ready_to_submit"]["readiness_summary"],
            "Request shape is valid, but live readiness was skipped.",
        )
        self.assertEqual(report["ready_to_submit"]["next_step"], "run_live_preflight")
        self.assertEqual(report["ready_to_submit"]["stage_verification"], "skipped")
        self.assertEqual(
            report["ready_to_submit"]["warning_codes"],
            ["coordinator_preflight_skipped", "stage_preflight_skipped"],
        )
        self.assertFalse(report["ready_to_submit"]["coordinator_preflight_required"])
        self.assertEqual(report["coordinator_ready"]["reason"], "live_preflight_skipped")
        self.assertEqual(report["stage_preflight"]["reason"], "live_preflight_skipped")
        self.assertEqual(report["stage_preflight"]["missing_summary"], "not_checked")
        self.assertIn("coordinator_ready_preflight_skipped", report["diagnosis_codes"])
        self.assertIn("generate_request_shape_ready", report["diagnosis_codes"])
        self.assertNotIn("generate_dry_run_ready", report["diagnosis_codes"])
        self.assertIn("stage_preflight_skipped", report["diagnosis_codes"])
        self.assertEqual(
            report["operator_action"],
            "Generation request shape is valid, but live readiness was skipped; rerun --dry-run without --skip-live-preflight before submitting.",
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn("  coordinator_ready: not_checked service=none protocol=none reason=live_preflight_skipped", rendered)
        self.assertIn(
            "  ready_to_submit: not_checked label=skipped fully_verified=False route=True coordinator=not_checked stage=skipped stage_verification=skipped next_step=run_live_preflight warnings=coordinator_preflight_skipped,stage_preflight_skipped",
            rendered,
        )
        self.assertNotIn("ready_to_submit: None", rendered)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            cli.print_generate_start_hint(cli.parse_args([item for item in argv if item != "--json"]))
        progress = stderr.getvalue()
        self.assertIn("checking request shape only", progress)
        self.assertIn("live Coordinator and stage readiness are skipped", progress)
        self.assertNotIn("checking route and stage readiness before submitting work", progress)
        self.assertIn(
            "warnings=coordinator_preflight_skipped,stage_preflight_skipped",
            rendered,
        )

    def test_product_generate_stage_not_checked_includes_startup_next_commands(self) -> None:
        report = {
            "ok": False,
            "diagnosis_codes": ["stage_preflight_not_checked"],
            "route": {
                "route_source": "coordinator-url",
                "coordinator_url": "http://127.0.0.1:8787",
                "coordinator_url_present": True,
            },
            "session_request": {
                "backend": "cpu",
                "hf_model_id": "sshleifer/tiny-gpt2",
                "max_new_tokens": 4,
                "request_count": 1,
            },
            "ready_to_submit": {
                "readiness_label": "blocked",
                "next_step": "fix_blockers",
                "warning_codes": ["stage_preflight_not_checked"],
            },
            "stream": {"enabled": True},
            "output_request": {"include_output": True},
        }

        next_commands = cli._product_generate_next_commands(report)

        labels = [item["label"] for item in next_commands]
        self.assertEqual(labels[:3], ["start Coordinator", "start stage0 Miner", "start stage1 Miner"])
        self.assertIn("check generation route", labels)
        self.assertIn("submit generation after checks pass", labels)
        next_lines = [item["command_line"] for item in next_commands]
        self.assertIn(
            "crowdtensor serve --profile cpu-real-llm --bind-host 127.0.0.1 --public-host 127.0.0.1 --port 8787 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage0-miner --stage stage0 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage1-miner --stage stage1 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor generate --max-new-tokens 4 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN} --stream --include-output",
            next_lines,
        )
        self.assertIn(
            "crowdtensor generate --max-new-tokens 4 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --stream --include-output",
            next_lines,
        )
        submit = next(item for item in next_commands if item["label"] == "submit generation after checks pass")
        self.assertEqual(submit["requires_env"], ["CROWDTENSOR_ADMIN_TOKEN"])

    def test_product_generate_dry_run_ready_failure_includes_startup_next_commands(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8791",
            "--prompt-text",
            "CrowdTensor prompt",
            "--dry-run",
            "--json",
        ])

        with patch.object(cli, "request_json_url", side_effect=OSError("offline")):
            report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["coordinator_ready"]["error"], "OSError")
        self.assertEqual(
            report["ready_to_submit"]["warning_codes"],
            ["coordinator_not_ready", "stage_preflight_not_checked"],
        )
        self.assertIn("coordinator_ready_failed", report["diagnosis_codes"])
        self.assertIn("stage_preflight_not_checked", report["diagnosis_codes"])
        self.assertNotIn("stage_preflight_skipped", report["diagnosis_codes"])
        self.assertNotIn("generate_dry_run_ready", report["diagnosis_codes"])
        self.assertIn("Coordinator route exists", report["operator_action"])
        self.assertEqual(report["user_status"]["state"], "blocked")
        self.assertIn("Coordinator route exists", report["user_status"]["headline"])
        self.assertEqual(report["user_status"]["next_step"], "fix_blockers")
        self.assertEqual(report["user_status"]["recommended_label"], "start Coordinator")
        self.assertEqual(report["result"]["status"], "preflight")
        self.assertIsNone(report["result"]["generated_token_count"])
        self.assertIsNone(report["result"]["max_new_tokens"])
        self.assertEqual(report["result"]["output_count"], 0)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn(
            "  status: blocked: Coordinator route exists but /ready failed; start or restart the Coordinator and retry generate --dry-run. next=fix_blockers recommendation=start Coordinator public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  coordinator_ready: not_ready service=none protocol=none error=OSError",
            rendered,
        )
        self.assertIn("  result: status=preflight tokens=not-run outputs=0 display=hash-only-json hash=none public_artifact_safe=True", rendered)
        self.assertNotIn("tokens=0/16", rendered)
        markdown = cli.render_generate_summary_markdown(report)
        self.assertIn("- Result: `status=preflight tokens=not-run outputs=0 display=hash-only-json hash=none public_artifact_safe=True`", markdown)
        self.assertNotIn("tokens=0/16", markdown)
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            "crowdtensor serve --profile cpu-real-llm --bind-host 127.0.0.1 --public-host 127.0.0.1 --port 8791 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor join --coordinator-url http://127.0.0.1:8791 --miner-id stage0-miner --stage stage0 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor join --coordinator-url http://127.0.0.1:8791 --miner-id stage1-miner --stage stage1 --run",
            next_lines,
        )
        labels = [item["label"] for item in report["next_commands"]]
        self.assertIn("submit generation after checks pass", labels)

    def test_product_generate_low_loopback_port_suggests_safe_local_startup_port(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:9",
            "--prompt-text",
            "CrowdTensor prompt",
            "--dry-run",
            "--json",
        ])

        with patch.object(cli, "request_json_url", side_effect=OSError("offline")):
            report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("coordinator_ready_failed", report["diagnosis_codes"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            "crowdtensor serve --profile cpu-real-llm --bind-host 127.0.0.1 --public-host 127.0.0.1 --port 8787 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage0-miner --stage stage0 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}",
            next_lines,
        )
        self.assertFalse(any("--port 9 --run" in line for line in next_lines))
        self.assertFalse(any("--coordinator-url http://127.0.0.1:9" in line for line in next_lines))

    def test_product_generate_remote_ready_failure_keeps_remote_route_commands(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "https://coordinator.example:9443",
            "--prompt-text",
            "CrowdTensor prompt",
            "--dry-run",
            "--json",
        ])

        with patch.object(cli, "request_json_url", side_effect=OSError("offline")):
            report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("coordinator_ready_failed", report["diagnosis_codes"])
        self.assertIn("remote /ready failed", report["operator_action"])
        self.assertIn("remote Coordinator service", report["operator_action"])
        self.assertEqual(report["recommended_next_command"]["label"], "check generation route")
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertFalse(any(line.startswith("crowdtensor serve ") for line in next_lines))
        self.assertFalse(any(line.startswith("crowdtensor join ") for line in next_lines))
        self.assertIn(
            "crowdtensor generate --max-new-tokens 16 --coordinator-url https://coordinator.example:9443 --prompt-text '<prompt>' --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}",
            next_lines,
        )
        self.assertIn(
            "crowdtensor generate --max-new-tokens 16 --coordinator-url https://coordinator.example:9443 --prompt-text '<prompt>'",
            next_lines,
        )

    def test_p2pd_top_level_prints_daemon_command(self) -> None:
        args = cli.parse_args([
            "p2pd",
            "--port",
            "8789",
            "--peer-secret",
            "p2p-secret-value",
            "--require-signed",
            "--json",
        ])

        report = cli.build_p2pd_cli(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "p2pd_cli_v1")
        self.assertIn("p2p_lite_daemon.py", " ".join(report["command"]))
        self.assertNotIn("p2p-secret-value", json.dumps(report))
        self.assertIn("--<redacted>", report["command"])
        self.assertIn("p2p_signed_announce_required", report["diagnosis_codes"])
        self.assertIn("p2pd_command_ready", report["diagnosis_codes"])

    def test_p2p_daemon_top_level_prints_real_daemon_command(self) -> None:
        args = cli.parse_args([
            "p2p-daemon",
            "--port",
            "8889",
            "--record-secret",
            "p2p-secret-value",
            "--require-signed",
            "--json",
        ])

        report = cli.build_p2p_daemon_cli(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "p2p_daemon_cli_v1")
        self.assertIn("real_p2p_daemon.py", " ".join(report["command"]))
        self.assertNotIn("p2p-secret-value", json.dumps(report))
        self.assertIn("--<redacted>", report["command"])
        self.assertIn("real_p2p_provider_store_ready", report["diagnosis_codes"])
        self.assertIn("replaceable_discovery_backend_ready", report["diagnosis_codes"])

    def test_product_serve_p2p_announces_coordinator(self) -> None:
        args = cli.parse_args([
            "serve",
            "--p2p",
            "--peer-bootstrap",
            "http://127.0.0.1:8788",
            "--public-host",
            "coord.example",
            "--peer-secret",
            "p2p-secret-value",
            "--json",
        ])

        with patch.object(cli, "post_announce", return_value={"ok": True, "schema": "p2p_lite_announce_v1"}) as announced:
            report = cli.build_product_serve(args)

        self.assertTrue(report["ok"], report)
        self.assertIn("p2p_coordinator_announce_ready", report["diagnosis_codes"])
        self.assertIn("generate --p2p --dry-run", report["operator_action"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            "crowdtensor generate --max-new-tokens 16 --p2p --peer-bootstrap http://127.0.0.1:8788 --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}",
            next_lines,
        )
        self.assertTrue(any("CROWDTENSOR_P2P_PEER_SECRET" in item.get("requires_env", []) for item in report["next_commands"]))
        peer = announced.call_args.args[1]
        self.assertEqual(peer["role"], "coordinator")
        self.assertEqual(peer["urls"]["coordinator"], "http://coord.example:8787")
        self.assertEqual(peer["peer_signature"]["algorithm"], "hmac-sha256")
        self.assertNotIn("p2p-secret-value", json.dumps(report))

    def test_product_serve_real_p2p_announces_provider_record(self) -> None:
        args = cli.parse_args([
            "serve",
            "--p2p",
            "--p2p-backend",
            "real",
            "--peer-bootstrap",
            "http://127.0.0.1:8888",
            "--public-host",
            "coord.example",
            "--peer-secret",
            "p2p-secret-value",
            "--json",
        ])

        with patch.object(cli, "post_provider_record", return_value={"ok": True, "schema": "real_p2p_announce_v1"}) as announced:
            report = cli.build_product_serve(args)

        self.assertTrue(report["ok"], report)
        self.assertIn("real_p2p_coordinator_announce_ready", report["diagnosis_codes"])
        self.assertIn("generate --p2p --dry-run", report["operator_action"])
        record = announced.call_args.args[1]
        self.assertEqual(record["schema"], "real_p2p_provider_record_v1")
        self.assertEqual(record["provider"]["role"], "coordinator")
        self.assertEqual(record["provider"]["urls"]["coordinator"], "http://coord.example:8787")
        self.assertEqual(record["provider"]["peer_signature"]["algorithm"], "hmac-sha256")
        self.assertEqual(report["p2p"]["backend"], "real")
        self.assertNotIn("p2p-secret-value", json.dumps(report))

    def test_product_join_p2p_discovers_and_announces_stage_capability(self) -> None:
        args = cli.parse_args([
            "join",
            "--p2p",
            "--miner-id",
            "stage0-miner",
            "--stage",
            "stage0",
            "--peer-secret",
            "p2p-secret-value",
            "--json",
        ])

        catalog = {
            "peers": [
                {
                    "role": "coordinator",
                    "peer_id": "coord",
                    "urls": {"coordinator": "http://127.0.0.1:8787"},
                    "capabilities": {"backend": "cpu"},
                }
            ]
        }
        with patch.object(cli, "fetch_peer_catalog", return_value=catalog), patch.object(
            cli,
            "post_announce",
            return_value={"ok": True, "schema": "p2p_lite_announce_v1"},
        ) as announced:
            report = cli.build_product_join(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["coordinator_url"], "http://127.0.0.1:8787")
        self.assertIn("p2p_stage_miner_announce_ready", report["diagnosis_codes"])
        self.assertIn("Rerun with --run", report["operator_action"])
        self.assertIn("generate --p2p --dry-run", report["operator_action"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            "crowdtensor generate --max-new-tokens 16 --p2p --peer-bootstrap http://127.0.0.1:8788 --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}",
            next_lines,
        )
        peer = announced.call_args.args[1]
        self.assertEqual(peer["role"], "miner")
        self.assertIn("real_llm_sharded_stage0", peer["capabilities"]["real_llm_sharded_stage_capabilities"])
        self.assertEqual(peer["peer_signature"]["algorithm"], "hmac-sha256")
        self.assertNotIn("p2p-secret-value", json.dumps(report))

    def test_product_join_real_p2p_discovers_and_announces_stage_provider(self) -> None:
        args = cli.parse_args([
            "join",
            "--p2p",
            "--p2p-backend",
            "real",
            "--peer-bootstrap",
            "http://127.0.0.1:8888",
            "--miner-id",
            "stage0-miner",
            "--stage",
            "stage0",
            "--peer-secret",
            "p2p-secret-value",
            "--json",
        ])

        catalog = {
            "schema": "real_p2p_provider_catalog_v1",
            "peers": [
                {
                    "role": "coordinator",
                    "peer_id": "coord",
                    "urls": {"coordinator": "http://127.0.0.1:8787"},
                    "capabilities": {"backend": "cpu"},
                }
            ],
        }
        with patch.object(cli, "fetch_provider_catalog", return_value=catalog), patch.object(
            cli,
            "post_provider_record",
            return_value={"ok": True, "schema": "real_p2p_announce_v1"},
        ) as announced:
            report = cli.build_product_join(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["coordinator_url"], "http://127.0.0.1:8787")
        self.assertIn("real_p2p_stage_miner_announce_ready", report["diagnosis_codes"])
        self.assertIn("Rerun with --run", report["operator_action"])
        record = announced.call_args.args[1]
        self.assertEqual(record["schema"], "real_p2p_provider_record_v1")
        self.assertEqual(record["provider"]["role"], "miner")
        self.assertIn("real_llm_sharded_stage0", record["stage_capabilities"])
        self.assertEqual(record["provider"]["peer_signature"]["algorithm"], "hmac-sha256")
        self.assertNotIn("p2p-secret-value", json.dumps(report))

    def test_product_join_forwards_compute_seconds_to_miner(self) -> None:
        args = cli.parse_args([
            "join",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--miner-id",
            "slow-stage0",
            "--stage",
            "stage0",
            "--compute-seconds",
            "12.5",
            "--max-runtime-seconds",
            "30",
            "--max-request-attempts",
            "9",
            "--json",
        ])

        report = cli.build_product_join(args)

        self.assertTrue(report["ok"], report)
        self.assertIn("Rerun with --run", report["operator_action"])
        command = report["command"]
        self.assertIn("--compute-seconds", command)
        self.assertEqual(command[command.index("--compute-seconds") + 1], "12.5")
        self.assertIn("--max-runtime-seconds", command)
        self.assertEqual(command[command.index("--max-runtime-seconds") + 1], "30.0")
        self.assertIn("--max-request-attempts", command)
        self.assertEqual(command[command.index("--max-request-attempts") + 1], "9")
        self.assertIn("command_line", report)
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            "crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}",
            next_lines,
        )

    def test_product_join_missing_route_action(self) -> None:
        args = cli.parse_args([
            "join",
            "--peer-bootstrap",
            "http://127.0.0.1:8788",
            "--miner-id",
            "stage0-miner",
            "--stage",
            "stage0",
            "--json",
        ])

        with patch.object(cli, "fetch_peer_catalog", return_value={"peers": []}):
            report = cli.build_product_join(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("coordinator_route_missing", report["diagnosis_codes"])
        self.assertIn("Start the Coordinator", report["operator_action"])

    def test_product_join_p2p_discovery_unreachable_returns_actionable_report(self) -> None:
        args = cli.parse_args([
            "join",
            "--p2p",
            "--peer-bootstrap",
            "http://127.0.0.1:8799",
            "--miner-id",
            "stage0-miner",
            "--stage",
            "stage0",
            "--json",
        ])

        with patch.object(cli, "fetch_peer_catalog", side_effect=OSError("offline")):
            report = cli.build_product_join(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("p2p_discovery_unreachable", report["diagnosis_codes"])
        self.assertIn("coordinator_route_missing", report["diagnosis_codes"])
        self.assertEqual(report["p2p"]["discovery"]["error"], "OSError")
        self.assertIn("P2P discovery daemon", report["operator_action"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn("crowdtensor p2pd --port 8799 --run", next_lines)
        self.assertEqual(
            next_lines.count(
                "crowdtensor join --p2p --peer-bootstrap http://127.0.0.1:8799 --miner-id stage0-miner --stage stage0 --run"
            ),
            1,
        )

    def test_product_join_real_p2p_discovery_unreachable_suggests_real_daemon(self) -> None:
        args = cli.parse_args([
            "join",
            "--p2p",
            "--p2p-backend",
            "real",
            "--peer-bootstrap",
            "http://127.0.0.1:8899",
            "--miner-id",
            "stage0-miner",
            "--stage",
            "stage0",
            "--peer-secret",
            "p2p-secret-value",
            "--json",
        ])

        with patch.object(cli, "fetch_provider_catalog", side_effect=OSError("offline")):
            report = cli.build_product_join(args)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["p2p"]["backend"], "real")
        self.assertIn("p2p_discovery_unreachable", report["diagnosis_codes"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn("crowdtensor p2p-daemon --port 8899 --run", next_lines)
        self.assertFalse(any(line.startswith("crowdtensor p2pd ") for line in next_lines))
        self.assertNotIn("p2p-secret-value", json.dumps(report, sort_keys=True))

    def test_product_join_missing_p2p_route_preserves_secret_env_requirements(self) -> None:
        args = cli.parse_args([
            "join",
            "--p2p",
            "--peer-bootstrap",
            "http://127.0.0.1:8799",
            "--miner-id",
            "stage0-miner",
            "--stage",
            "stage0",
            "--miner-token",
            "miner-secret",
            "--peer-secret",
            "p2p-secret-value",
            "--json",
        ])

        with patch.object(cli, "fetch_peer_catalog", side_effect=OSError("offline")):
            report = cli.build_product_join(args)

        encoded = json.dumps(report, sort_keys=True)
        self.assertFalse(report["ok"], report)
        self.assertNotIn("miner-secret", encoded)
        self.assertNotIn("p2p-secret-value", encoded)
        p2pd = report["next_commands"][0]
        self.assertEqual(p2pd["label"], "start P2P discovery daemon")
        self.assertEqual(p2pd["requires_env"], ["CROWDTENSOR_P2P_PEER_SECRET"])
        local_coordinator = report["next_commands"][1]
        self.assertEqual(local_coordinator["label"], "start local Coordinator")
        self.assertEqual(
            local_coordinator["requires_env"],
            ["CROWDTENSOR_MINER_TOKEN", "CROWDTENSOR_P2P_PEER_SECRET"],
        )
        start_this_miner = report["next_commands"][2]
        self.assertEqual(start_this_miner["label"], "start this Miner")
        self.assertEqual(
            start_this_miner["requires_env"],
            ["CROWDTENSOR_MINER_TOKEN", "CROWDTENSOR_P2P_PEER_SECRET"],
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_join(report)
        rendered = stdout.getvalue()
        self.assertIn(
            "CROWDTENSOR_P2P_PEER_SECRET=${CROWDTENSOR_P2P_PEER_SECRET:?set CROWDTENSOR_P2P_PEER_SECRET} "
            "crowdtensor p2pd --port 8799 --run",
            rendered,
        )
        self.assertIn(
            "CROWDTENSOR_MINER_TOKEN=${CROWDTENSOR_MINER_TOKEN:?set CROWDTENSOR_MINER_TOKEN} "
            "CROWDTENSOR_P2P_PEER_SECRET=${CROWDTENSOR_P2P_PEER_SECRET:?set CROWDTENSOR_P2P_PEER_SECRET} "
            "crowdtensor serve --profile cpu-real-llm",
            rendered,
        )
        self.assertIn("# requires CROWDTENSOR_MINER_TOKEN, CROWDTENSOR_P2P_PEER_SECRET", rendered)
        self.assertNotIn("miner-secret", rendered)
        self.assertNotIn("p2p-secret-value", rendered)

    def test_product_join_missing_route_without_discovery_returns_actionable_report(self) -> None:
        args = cli.parse_args([
            "join",
            "--miner-id",
            "stage0-miner",
            "--stage",
            "stage0",
            "--json",
        ])

        report = cli.build_product_join(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("coordinator_route_missing", report["diagnosis_codes"])
        self.assertIn("Start the Coordinator", report["operator_action"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            "crowdtensor serve --profile cpu-real-llm --bind-host 127.0.0.1 --public-host 127.0.0.1 --port 8787 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage0-miner --stage stage0 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage1-miner --stage stage1 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}",
            next_lines,
        )
        self.assertIn(
            "crowdtensor join --p2p --peer-bootstrap http://127.0.0.1:8788 --miner-id stage0-miner --stage stage0 --run",
            next_lines,
        )

    def test_product_join_human_output_includes_action_and_redacts_token(self) -> None:
        args = cli.parse_args([
            "join",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--miner-id",
            "stage0-miner",
            "--stage",
            "stage0",
            "--miner-token",
            "miner-secret",
        ])

        report = cli.build_product_join(args)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_join(report)
        rendered = stdout.getvalue()

        self.assertIn("CrowdTensor join", rendered)
        self.assertIn("  command: ", rendered)
        self.assertIn("--miner-token '<redacted>'", rendered)
        self.assertIn("  action: Rerun with --run", rendered)
        self.assertIn("  next[3] check generation route: crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}", rendered)
        self.assertIn("# requires CROWDTENSOR_OBSERVER_TOKEN", rendered)
        self.assertIn("# requires CROWDTENSOR_MINER_TOKEN", rendered)
        self.assertNotIn("miner-secret", rendered)

    def test_discovery_refresh_rebuilds_signed_record_timestamps(self) -> None:
        peer = cli.build_p2p_peer(
            swarm_id="swarm",
            peer_id="stage0-miner",
            role="miner",
            backend="cpu",
            stage_role="stage0",
            ttl_seconds=60,
        )
        records: list[dict] = []

        def fake_announce(_: str, record: dict, **__: object) -> dict:
            records.append(record)
            return {"ok": True, "record": record}

        with patch.object(cli, "post_provider_record", side_effect=fake_announce), patch("crowdtensor.p2p_lite.time.time", side_effect=[1000.0, 1025.0]), patch("crowdtensor.real_p2p.time.time", side_effect=[1000.0, 1025.0]):
            first = cli.announce_discovery_peer("http://127.0.0.1:8888", peer, timeout=1, backend="real", peer_secret="secret")
            refresh = cli.DiscoveryRefreshThread(
                bootstrap="http://127.0.0.1:8888",
                peer=peer,
                timeout=1,
                backend="real",
                peer_secret="secret",
                interval_seconds=1,
            )
            refresh._run_once()

        self.assertTrue(first["ok"])
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0]["provider"]["last_seen"], records[1]["provider"]["last_seen"])
        self.assertNotEqual(
            records[0]["provider"]["peer_signature"]["signed_at"],
            records[1]["provider"]["peer_signature"]["signed_at"],
        )

    def test_product_generate_p2p_dry_run_requires_stage_peers(self) -> None:
        args = cli.parse_args([
            "generate",
            "--p2p",
            "--prompt-text",
            "CrowdTensor prompt",
            "--max-new-tokens",
            "2",
            "--dry-run",
            "--json",
        ])
        catalog = {
            "peers": [
                {"role": "coordinator", "peer_id": "coord", "urls": {"coordinator": "http://127.0.0.1:8787"}},
                {
                    "role": "miner",
                    "peer_id": "stage0",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                },
                {
                    "role": "miner",
                    "peer_id": "stage1",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]},
                },
            ]
        }

        with patch.object(cli, "fetch_peer_catalog", return_value=catalog):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["route"]["route_source"], "p2p-discovery")
        self.assertIn("p2p_generate_route_ready", report["diagnosis_codes"])
        self.assertTrue(report["ready_to_submit"]["ok"])
        self.assertFalse(report["ready_to_submit"]["fully_verified"])
        self.assertEqual(report["ready_to_submit"]["readiness_label"], "partial")
        self.assertEqual(
            report["ready_to_submit"]["readiness_summary"],
            "Request can be submitted, but Coordinator readiness is not fully verified.",
        )
        self.assertEqual(report["ready_to_submit"]["next_step"], "submit_with_caution")
        self.assertTrue(report["stage_preflight"]["ok"])
        self.assertEqual(report["stage_preflight"]["source"], "p2p-route")
        self.assertIn("coordinator_ready_preflight_skipped", report["diagnosis_codes"])
        self.assertIn("stage_preflight_ready", report["diagnosis_codes"])

    def test_product_generate_p2p_discovery_unreachable_returns_actionable_report(self) -> None:
        args = cli.parse_args([
            "generate",
            "CrowdTensor prompt",
            "--p2p",
            "--peer-bootstrap",
            "http://127.0.0.1:8799",
            "--max-new-tokens",
            "2",
            "--dry-run",
            "--json",
        ])

        with patch.object(cli, "fetch_peer_catalog", side_effect=OSError("offline")):
            report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("p2p_discovery_unreachable", report["diagnosis_codes"])
        self.assertIn("coordinator_route_missing", report["diagnosis_codes"])
        self.assertEqual(report["issue_summary"]["primary_code"], "p2p_discovery_unreachable")
        self.assertEqual(report["review_summary"]["primary_code"], "p2p_discovery_unreachable")
        self.assertEqual(report["p2p"]["discovery"]["error"], "OSError")
        self.assertIn("P2P discovery daemon", report["operator_action"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn("crowdtensor p2pd --port 8799 --run", next_lines)
        self.assertIn(
            "crowdtensor serve --profile cpu-real-llm --bind-host 127.0.0.1 --public-host 127.0.0.1 --port 8787 --p2p --peer-bootstrap http://127.0.0.1:8799 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor generate --max-new-tokens 2 --p2p --peer-bootstrap http://127.0.0.1:8799 --prompt-text '<prompt>' --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}",
            next_lines,
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn("  p2p: enabled=True backend=lite bootstrap=http://127.0.0.1:8799 peers=0 discovery_ok=False discovery_error=OSError", rendered)
        self.assertNotIn("CrowdTensor prompt", json.dumps(report, sort_keys=True))

    def test_product_generate_p2p_preserves_swarm_id_in_next_commands(self) -> None:
        args = cli.parse_args([
            "generate",
            "CrowdTensor prompt",
            "--p2p",
            "--swarm-id",
            "public-swarm-v2",
            "--peer-bootstrap",
            "http://127.0.0.1:8799",
            "--max-new-tokens",
            "2",
            "--dry-run",
            "--json",
        ])

        with patch.object(cli, "fetch_peer_catalog", side_effect=OSError("offline")):
            report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["p2p"]["swarm_id"], "public-swarm-v2")
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn("crowdtensor p2pd --port 8799 --swarm-id public-swarm-v2 --run", next_lines)
        self.assertIn(
            "crowdtensor generate --max-new-tokens 2 --p2p --swarm-id public-swarm-v2 --peer-bootstrap http://127.0.0.1:8799 --prompt-text '<prompt>' --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}",
            next_lines,
        )

    def test_product_generate_real_p2p_discovery_unreachable_suggests_real_daemon(self) -> None:
        args = cli.parse_args([
            "generate",
            "CrowdTensor prompt",
            "--p2p",
            "--p2p-backend",
            "real",
            "--peer-bootstrap",
            "http://127.0.0.1:8899",
            "--max-new-tokens",
            "2",
            "--dry-run",
            "--json",
        ])

        with patch.object(cli, "fetch_provider_catalog", side_effect=OSError("offline")):
            report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["p2p"]["backend"], "real")
        self.assertIn("p2p_discovery_unreachable", report["diagnosis_codes"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn("crowdtensor p2p-daemon --port 8899 --run", next_lines)
        self.assertFalse(any(line.startswith("crowdtensor p2pd ") for line in next_lines))
        self.assertNotIn("CrowdTensor prompt", json.dumps(report, sort_keys=True))

    def test_product_generate_p2p_dry_run_filters_coordinator_by_model_id(self) -> None:
        args = cli.parse_args([
            "generate",
            "--p2p",
            "--prompt-text",
            "CrowdTensor prompt",
            "--hf-model-id",
            "distilgpt2",
            "--max-new-tokens",
            "2",
            "--dry-run",
            "--json",
        ])
        catalog = {
            "peers": [
                {
                    "role": "coordinator",
                    "peer_id": "coord-tiny",
                    "urls": {"coordinator": "http://tiny.example:8787"},
                    "capabilities": {"backend": "cpu", "hf_model_id": "sshleifer/tiny-gpt2"},
                },
                {
                    "role": "coordinator",
                    "peer_id": "coord-distil",
                    "urls": {"coordinator": "http://distil.example:8787"},
                    "capabilities": {"backend": "cpu", "hf_model_id": "distilgpt2"},
                },
                {
                    "role": "miner",
                    "peer_id": "stage0-distil",
                    "capabilities": {
                        "backend": "cpu",
                        "hf_model_id": "distilgpt2",
                        "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"],
                    },
                },
                {
                    "role": "miner",
                    "peer_id": "stage1-distil",
                    "capabilities": {
                        "backend": "cpu",
                        "hf_model_id": "distilgpt2",
                        "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"],
                    },
                },
            ]
        }

        with patch.object(cli, "fetch_peer_catalog", return_value=catalog):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["route"]["coordinator_url"], "http://distil.example:8787")
        self.assertEqual(report["route"]["coordinator_filter"]["mismatched_peers"], ["coord-tiny"])
        self.assertIn("session_route_coordinator_filter_ready", report["route"]["diagnosis_codes"])
        self.assertEqual(report["coordinator_ready"]["reason"], "not_checked_for_discovered_remote_coordinator")
        self.assertTrue(report["ready_to_submit"]["ok"])

    def test_product_generate_p2p_dry_run_reports_missing_stage_preflight_action(self) -> None:
        args = cli.parse_args([
            "generate",
            "--p2p",
            "--prompt-text",
            "CrowdTensor prompt",
            "--max-new-tokens",
            "2",
            "--dry-run",
            "--json",
        ])
        catalog = {
            "peers": [
                {"role": "coordinator", "peer_id": "coord", "urls": {"coordinator": "http://127.0.0.1:8787"}},
                {
                    "role": "miner",
                    "peer_id": "stage0",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                },
            ]
        }

        with patch.object(cli, "fetch_peer_catalog", return_value=catalog):
            report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["ready_to_submit"]["ok"])
        self.assertFalse(report["stage_preflight"]["ok"])
        self.assertEqual(report["stage_preflight"]["missing_capabilities"], ["real_llm_sharded_stage1"])
        self.assertIn("stage_preflight_failed", report["diagnosis_codes"])
        self.assertIn("stage0 and stage1 Miners", report["operator_action"])

    def test_product_generate_real_p2p_dry_run_uses_route_lookup(self) -> None:
        args = cli.parse_args([
            "generate",
            "--p2p",
            "--p2p-backend",
            "real",
            "--peer-bootstrap",
            "http://127.0.0.1:8888",
            "--prompt-text",
            "CrowdTensor prompt",
            "--max-new-tokens",
            "2",
            "--dry-run",
            "--json",
        ])
        peers = [
            {"role": "coordinator", "peer_id": "coord", "urls": {"coordinator": "http://127.0.0.1:8787"}},
            {
                "role": "miner",
                "peer_id": "stage0",
                "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
            },
            {
                "role": "miner",
                "peer_id": "stage1",
                "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]},
            },
        ]
        catalog = {"schema": "real_p2p_provider_catalog_v1", "peers": peers}
        route_payload = {
            "schema": "real_p2p_route_lookup_v1",
            "ok": True,
            "route": {
                "route_source": "real-p2p-discovery",
                "coordinator_url": "http://127.0.0.1:8787",
                "coordinator_url_present": True,
                "required_capabilities": ["real_llm_sharded_stage0", "real_llm_sharded_stage1"],
                "missing_capabilities": [],
                "matched_peers": ["stage0", "stage1"],
                "usable_now": True,
                "diagnosis_codes": ["real_p2p_route_lookup_ready"],
            },
        }

        with patch.object(cli, "fetch_provider_catalog", return_value=catalog), patch.object(
            cli,
            "post_route_lookup",
            return_value=route_payload,
        ):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["route"]["route_source"], "real-p2p-discovery")
        self.assertEqual(report["p2p"]["backend"], "real")
        self.assertIn("real_p2p_generate_route_ready", report["diagnosis_codes"])

    def test_product_generate_real_p2p_route_lookup_uses_compatible_coordinator(self) -> None:
        args = cli.parse_args([
            "generate",
            "--p2p",
            "--p2p-backend",
            "real",
            "--peer-bootstrap",
            "http://127.0.0.1:8888",
            "--prompt-text",
            "CrowdTensor prompt",
            "--hf-model-id",
            "distilgpt2",
            "--max-new-tokens",
            "2",
            "--dry-run",
            "--json",
        ])
        peers = [
            {
                "role": "coordinator",
                "peer_id": "coord-tiny",
                "urls": {"coordinator": "http://tiny.example:8787"},
                "capabilities": {"backend": "cpu", "hf_model_id": "sshleifer/tiny-gpt2"},
            },
            {
                "role": "coordinator",
                "peer_id": "coord-distil",
                "urls": {"coordinator": "http://distil.example:8787"},
                "capabilities": {"backend": "cpu", "hf_model_id": "distilgpt2"},
            },
            {
                "role": "miner",
                "peer_id": "stage0-distil",
                "capabilities": {
                    "backend": "cpu",
                    "hf_model_id": "distilgpt2",
                    "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"],
                },
            },
            {
                "role": "miner",
                "peer_id": "stage1-distil",
                "capabilities": {
                    "backend": "cpu",
                    "hf_model_id": "distilgpt2",
                    "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"],
                },
            },
        ]
        catalog = {"schema": "real_p2p_provider_catalog_v1", "peers": peers}
        captured: dict[str, str] = {}

        def fake_route_lookup(
            bootstrap: str,
            session_request: dict,
            *,
            coordinator_url: str = "",
            timeout: float = 5.0,
        ) -> dict:
            del bootstrap, session_request, timeout
            captured["coordinator_url"] = coordinator_url
            return {
                "schema": "real_p2p_route_lookup_v1",
                "ok": True,
                "route": {
                    "route_source": "real-p2p-discovery",
                    "coordinator_url": coordinator_url,
                    "coordinator_url_present": True,
                    "required_capabilities": ["real_llm_sharded_stage0", "real_llm_sharded_stage1"],
                    "missing_capabilities": [],
                    "matched_capabilities": {
                        "real_llm_sharded_stage0": "stage0-distil",
                        "real_llm_sharded_stage1": "stage1-distil",
                    },
                    "usable_now": True,
                    "diagnosis_codes": ["real_p2p_route_lookup_ready"],
                },
            }

        with patch.object(cli, "fetch_provider_catalog", return_value=catalog), patch.object(
            cli,
            "post_route_lookup",
            side_effect=fake_route_lookup,
        ):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(captured["coordinator_url"], "http://distil.example:8787")
        self.assertEqual(report["route"]["coordinator_url"], "http://distil.example:8787")

    def test_product_generate_real_p2p_uses_route_lookup_coordinator_for_session_create(self) -> None:
        args = cli.parse_args([
            "generate",
            "--p2p",
            "--p2p-backend",
            "real",
            "--peer-bootstrap",
            "http://127.0.0.1:8888",
            "--prompt-text",
            "CrowdTensor prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--json",
        ])
        peers = [
            {
                "role": "coordinator",
                "peer_id": "coord-catalog",
                "urls": {"coordinator": "http://catalog.example:8787"},
                "capabilities": {"backend": "cpu", "hf_model_id": "sshleifer/tiny-gpt2"},
            },
            {
                "role": "miner",
                "peer_id": "stage0",
                "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
            },
            {
                "role": "miner",
                "peer_id": "stage1",
                "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]},
            },
        ]
        catalog = {"schema": "real_p2p_provider_catalog_v1", "peers": peers}
        route_payload = {
            "schema": "real_p2p_route_lookup_v1",
            "ok": True,
            "route": {
                "route_source": "real-p2p-discovery",
                "coordinator_url": "http://route.example:8787",
                "coordinator_url_present": True,
                "required_capabilities": ["real_llm_sharded_stage0", "real_llm_sharded_stage1"],
                "missing_capabilities": [],
                "matched_capabilities": {
                    "real_llm_sharded_stage0": "stage0",
                    "real_llm_sharded_stage1": "stage1",
                },
                "usable_now": True,
                "diagnosis_codes": ["real_p2p_route_lookup_ready"],
            },
        }
        base_urls: list[str] = []

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del payload, admin_token, timeout
            base_urls.append(base_url)
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "session-route-url",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                }
            self.assertIn("session_id=session-route-url", path)
            return {
                "results": [
                    {
                        "validation": {
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:route",
                            "decoded_tokens_match": True,
                        }
                    }
                ]
            }

        with patch.object(cli, "fetch_provider_catalog", return_value=catalog), patch.object(
            cli,
            "post_route_lookup",
            return_value=route_payload,
        ), patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertTrue(base_urls)
        self.assertTrue(all(url == "http://route.example:8787" for url in base_urls), base_urls)
        self.assertEqual(report["route"]["coordinator_url"], "http://route.example:8787")

    def test_product_generate_p2p_non_dry_run_blocks_when_route_unusable(self) -> None:
        args = cli.parse_args([
            "generate",
            "--p2p",
            "--peer-bootstrap",
            "http://127.0.0.1:8788",
            "--prompt-text",
            "CrowdTensor prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--json",
        ])
        catalog = {
            "peers": [
                {"role": "coordinator", "peer_id": "coord", "urls": {"coordinator": "http://127.0.0.1:8787"}},
                {
                    "role": "miner",
                    "peer_id": "stage0",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                },
            ]
        }

        with patch.object(cli, "fetch_peer_catalog", return_value=catalog), patch.object(
            cli,
            "request_json_url",
            side_effect=AssertionError("session creation should be blocked when p2p route is unusable"),
        ):
            report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["diagnosis_codes"], ["generate_route_unavailable"])
        self.assertIn("real_llm_sharded_stage1", report["route"]["missing_capabilities"])
        self.assertIn("stage0/stage1", report["operator_action"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            "crowdtensor generate --max-new-tokens 2 --p2p --peer-bootstrap http://127.0.0.1:8788 --prompt-text '<prompt>' --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}",
            next_lines,
        )
        self.assertIn(
            "crowdtensor join --p2p --peer-bootstrap http://127.0.0.1:8788 --miner-id stage1-miner --stage stage1 --run",
            next_lines,
        )
        self.assertNotIn("CrowdTensor prompt", json.dumps(report["next_commands"], sort_keys=True))

    def test_product_generate_p2p_non_dry_run_discovery_unreachable_is_actionable(self) -> None:
        args = cli.parse_args([
            "generate",
            "CrowdTensor prompt",
            "--p2p",
            "--peer-bootstrap",
            "http://127.0.0.1:8799",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--json",
        ])

        with patch.object(cli, "fetch_peer_catalog", side_effect=OSError("offline")), patch.object(
            cli,
            "request_json_url",
            side_effect=AssertionError("session creation should be blocked when discovery is offline"),
        ):
            report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("p2p_discovery_unreachable", report["diagnosis_codes"])
        self.assertIn("coordinator_route_missing", report["diagnosis_codes"])
        self.assertEqual(report["issue_summary"]["primary_code"], "p2p_discovery_unreachable")
        self.assertEqual(report["review_summary"]["primary_code"], "p2p_discovery_unreachable")
        self.assertEqual(report["p2p"]["discovery"]["error"], "OSError")
        self.assertIn("P2P discovery daemon", report["operator_action"])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        rendered = stdout.getvalue()
        self.assertIn("  p2p: enabled=True backend=lite bootstrap=http://127.0.0.1:8799 peers=0 discovery_ok=False discovery_error=OSError", rendered)
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn("crowdtensor p2pd --port 8799 --run", next_lines)
        self.assertIn(
            "crowdtensor generate --max-new-tokens 2 --p2p --peer-bootstrap http://127.0.0.1:8799 --prompt-text '<prompt>' --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}",
            next_lines,
        )
        self.assertNotIn("CrowdTensor prompt", json.dumps(report, sort_keys=True))

    def test_product_generate_missing_route_returns_actionable_report(self) -> None:
        args = cli.parse_args([
            "generate",
            "--prompt-text",
            "CrowdTensor prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--stream",
            "--include-output",
            "--json",
        ])

        report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("coordinator_route_missing", report["diagnosis_codes"])
        self.assertEqual(report["issue_summary"]["primary_code"], "coordinator_route_missing")
        self.assertEqual(report["review_summary"]["primary_code"], "coordinator_route_missing")
        self.assertTrue(report["stream"]["enabled"])
        self.assertTrue(report["stream"]["requested"])
        self.assertTrue(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertTrue(report["output_request"]["public_artifact_safe"])
        self.assertIn("Start a Coordinator", report["operator_action"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            "crowdtensor serve --profile cpu-real-llm --bind-host 127.0.0.1 --public-host 127.0.0.1 --port 8787 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage0-miner --stage stage0 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage1-miner --stage stage1 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor generate --max-new-tokens 2 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN} --stream --include-output",
            next_lines,
        )
        self.assertIn(
            "crowdtensor generate --max-new-tokens 2 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --stream --include-output",
            next_lines,
        )
        self.assertNotIn("CrowdTensor prompt", json.dumps(report["next_commands"], sort_keys=True))

    def test_product_generate_session_create_failure_preserves_requested_options(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-texts",
            "CrowdTensor prompt,Second private prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--stream",
            "--include-output",
            "--json",
        ])

        error_body = b"bad request echoed CrowdTensor prompt, Second private prompt, and admin-secret"
        with patch.object(
            cli,
            "request_json_url",
            side_effect=cli.HTTPError(
                "http://127.0.0.1:8787/admin/inference-sessions",
                400,
                "bad request",
                {},
                io.BytesIO(error_body),
            ),
        ):
            report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("session_create_failed", report["diagnosis_codes"])
        self.assertEqual(report["error"], "HTTPError")
        self.assertIn("<redacted>", report["detail"])
        self.assertNotIn("CrowdTensor prompt", report["detail"])
        self.assertNotIn("Second private prompt", report["detail"])
        self.assertNotIn("admin-secret", report["detail"])
        self.assertTrue(report["stream"]["enabled"])
        self.assertTrue(report["stream"]["requested"])
        self.assertTrue(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertTrue(report["output_request"]["public_artifact_safe"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            "crowdtensor generate --max-new-tokens 2 --coordinator-url http://127.0.0.1:8787 --prompt-texts '<prompt-1>,<prompt-2>' --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN} --stream --include-output",
            next_lines,
        )
        self.assertIn(
            "crowdtensor generate --max-new-tokens 2 --coordinator-url http://127.0.0.1:8787 --prompt-texts '<prompt-1>,<prompt-2>' --stream --include-output",
            next_lines,
        )
        self.assertNotIn("admin-secret", json.dumps(report, sort_keys=True))
        self.assertNotIn("CrowdTensor prompt", json.dumps(report, sort_keys=True))
        self.assertNotIn("Second private prompt", json.dumps(report, sort_keys=True))

    def test_product_generate_requires_admin_token_with_action(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--output-dir",
            str(output_dir),
            "--prompt-text",
            "CrowdTensor prompt",
            "--max-new-tokens",
            "2",
            "--json",
        ])

        report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("admin_token_required", report["diagnosis_codes"])
        self.assertEqual(report["operator_action"], "Pass --admin-token or set CROWDTENSOR_ADMIN_TOKEN.")
        self.assertEqual(report["user_status"]["state"], "blocked")
        self.assertEqual(report["user_status"]["headline"], "Pass --admin-token or set CROWDTENSOR_ADMIN_TOKEN.")
        self.assertEqual(report["user_status"]["next_step"], "fix_blockers")
        self.assertTrue(report["user_status"]["recommended_label"].startswith("submit generation"))
        self.assertEqual(report["saved_summary"]["path"], str(output_dir / "generate_summary.json"))
        self.assertTrue(report["artifacts"]["generate_summary"]["present"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            f"crowdtensor generate --max-new-tokens 2 --output-dir {output_dir} --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --dry-run --observer-token ${{CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}}",
            next_lines,
        )
        self.assertTrue(any("CROWDTENSOR_ADMIN_TOKEN" in item.get("requires_env", []) for item in report["next_commands"]))
        persisted = json.loads((output_dir / "generate_summary.json").read_text(encoding="utf-8"))
        self.assertIn("admin_token_required", persisted["diagnosis_codes"])
        self.assertEqual(persisted["user_status"]["state"], "blocked")
        self.assertNotIn("CrowdTensor prompt", json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "generate_summary.md").read_text(encoding="utf-8")
        self.assertIn("- OK: `False`", markdown)
        self.assertIn("## What To Do Next", markdown)
        self.assertIn("- State: `blocked`", markdown)
        self.assertIn("- Next step: `fix_blockers`", markdown)
        self.assertIn("- Recommended: `submit generation` reason=`set_admin_token`", markdown)
        self.assertIn("- Requires env: `CROWDTENSOR_ADMIN_TOKEN`", markdown)
        self.assertIn("- Action: Pass --admin-token or set CROWDTENSOR_ADMIN_TOKEN.", markdown)
        self.assertIn("## Details", markdown)
        self.assertIn(
            "- Status: `blocked: Pass --admin-token or set CROWDTENSOR_ADMIN_TOKEN. next=fix_blockers recommendation=submit generation public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Review next: `label=submit generation reason=set_admin_token command=CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor generate --max-new-tokens 2",
            markdown,
        )
        self.assertIn(
            "- Recommended next: `submit generation` reason=`set_admin_token` command=`CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor generate --max-new-tokens 2",
            markdown,
        )
        self.assertIn(
            "2. `submit generation`: `CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor generate --max-new-tokens 2",
            markdown,
        )
        self.assertIn("requires=`CROWDTENSOR_ADMIN_TOKEN`", markdown)
        self.assertIn("Pass --admin-token or set CROWDTENSOR_ADMIN_TOKEN.", markdown)
        self.assertNotIn("CrowdTensor prompt", markdown)

    def test_product_generate_preserves_safe_generation_counts(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--output-dir",
            str(output_dir),
            "--prompt-text",
            "CrowdTensor prompt",
            "--admin-token",
            "admin-secret",
            "--hf-model-id",
            "distilgpt2",
            "--max-new-tokens",
            "2",
            "--json",
        ])
        calls: list[tuple[str, str]] = []
        posted_payloads: list[dict] = []

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, admin_token, timeout
            calls.append((method, path))
            if method == "POST":
                posted_payloads.append(payload or {})
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-test",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                    "model_id": "distilgpt2",
                }
            return {
                "results": [
                    {
                        "validation": {
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:generated",
                            "generated_text": "local generated text must stay local",
                            "generated_token_ids": [1, 2],
                            "decoded_tokens_match": True,
                        }
                    }
                ]
            }

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_product_generate(args)

        encoded = json.dumps(report, sort_keys=True)
        self.assertTrue(report["ok"], report)
        self.assertEqual(posted_payloads[0]["hf_model_id"], "distilgpt2")
        self.assertEqual(report["session"]["hf_model_id"], "distilgpt2")
        self.assertEqual(report["generation"]["generated_token_count"], 2)
        self.assertEqual(report["generation"]["max_new_tokens"], 2)
        self.assertEqual(report["result"]["status"], "complete")
        self.assertEqual(report["result"]["generated_token_count"], 2)
        self.assertEqual(report["result"]["max_new_tokens"], 2)
        self.assertEqual(report["result"]["output_count"], 1)
        self.assertEqual(report["result"]["display"], "hash-only-json")
        self.assertEqual(report["result"]["generated_text_hash"], "sha256:generated")
        self.assertTrue(report["result"]["public_artifact_safe"])
        self.assertEqual(report["user_status"]["state"], "completed")
        self.assertEqual(report["user_status"]["headline"], "Generation completed.")
        self.assertEqual(report["user_status"]["next_step"], "rerun_or_review_artifacts")
        self.assertEqual(report["user_status"]["recommended_label"], "rerun generation")
        self.assertTrue(report["wait_progress"]["session_created"])
        self.assertTrue(report["wait_progress"]["ledger_endpoint_ready"])
        self.assertEqual(report["wait_progress"]["accepted_rows_seen"], 1)
        self.assertEqual(report["wait_progress"]["max_observed_token_count"], 2)
        self.assertTrue(report["wait_progress"]["completion_observed"])
        self.assertEqual(report["trace"]["session_id"], "real-llm-session-test")
        self.assertEqual(report["trace"]["workload_type"], "real_llm_sharded_infer")
        self.assertEqual(report["trace"]["request_count"], 1)
        self.assertEqual(report["trace"]["accepted_rows_seen"], 1)
        self.assertEqual(report["trace"]["stream_event_count"], 0)
        self.assertEqual(report["trace"]["source"], "public_swarm_product_cli_v1")
        self.assertFalse(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertTrue(report["output_request"]["public_artifact_safe"])
        self.assertFalse(report["trace"]["raw_prompt_public"])
        self.assertFalse(report["trace"]["raw_generated_text_public"])
        self.assertFalse(report["trace"]["generated_token_ids_public"])
        self.assertTrue(report["trace"]["public_artifact_safe"])
        self.assertEqual(report["prompt_scope"]["source"], "prompt-text")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 1)
        self.assertTrue(report["prompt_scope"]["inline_prompt_text"])
        self.assertTrue(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertTrue(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertFalse(report["prompt_scope"]["terminal_local_paths"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_public_safe"])
        self.assertTrue(report["prompt_scope"]["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(report["prompt_scope"]["public_artifact_safe"])
        self.assertFalse(report["safety"]["raw_prompt_public"])
        self.assertFalse(report["safety"]["raw_generated_text_public"])
        self.assertFalse(report["safety"]["generated_token_ids_public"])
        self.assertTrue(report["safety"]["read_only_workload"])
        self.assertTrue(report["safety"]["coordinator_backed"])
        self.assertTrue(report["safety"]["not_production"])
        self.assertTrue(report["safety"]["not_large_model_serving"])
        self.assertTrue(report["safety"]["not_arbitrary_public_prompt_serving"])
        self.assertEqual(len(report["trace"]["request_trace"]), 1)
        self.assertEqual(report["trace"]["request_trace"][0]["source"], "session-request")
        self.assertTrue(report["trace"]["request_trace"][0]["prompt_hash"])
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(report["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(report["shareable_summary"]["raw_generated_text_public"])
        self.assertFalse(report["shareable_summary"]["generated_token_ids_public"])
        self.assertFalse(report["shareable_summary"]["local_output_display_only"])
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "json-suppressed")
        self.assertFalse(report["shareable_summary"]["local_answer_terminal_only"])
        self.assertEqual(report["local_output"]["output_count"], 1)
        self.assertFalse(report["local_output"]["available"])
        self.assertTrue(report["local_output"]["public_artifact_safe"])
        self.assertEqual(report["answer_scope"]["scope_state"], "json-suppressed")
        self.assertEqual(
            report["local_output_note"],
            "Generated output is present, but raw text is suppressed in JSON/public output; rerun without --json for local display.",
        )
        self.assertEqual(report["issue_summary"]["state"], "completed")
        self.assertEqual(report["issue_summary"]["primary_code"], "public_swarm_generate_ready")
        self.assertEqual(report["issue_summary"]["next_step"], "rerun_or_review_artifacts")
        self.assertIn("accepted_rows=1", report["issue_summary"]["progress"])
        self.assertEqual(report["saved_summary"]["path"], str(output_dir / "generate_summary.json"))
        self.assertTrue(report["artifacts"]["generate_summary_markdown"]["present"])
        self.assertEqual(report["artifact_summary"]["inspect_first"], str(output_dir / "generate_summary.md"))
        self.assertEqual(report["artifact_summary"]["artifact_count"], 2)
        self.assertEqual(report["artifact_summary"]["present_artifact_count"], 2)
        self.assertTrue(report["artifact_summary"]["raw_generated_text_redacted"])
        self.assertTrue(report["artifact_summary"]["public_artifact_safe"])
        self.assertEqual(report["review_summary"]["state"], "completed")
        self.assertEqual(report["review_summary"]["next_step"], "rerun_or_review_artifacts")
        self.assertEqual(report["review_summary"]["inspect_first"], str(output_dir / "generate_summary.md"))
        self.assertEqual(report["review_summary"]["recommended_label"], "rerun generation")
        self.assertEqual(report["review_summary"]["primary_code"], "public_swarm_generate_ready")
        self.assertEqual(report["review_summary"]["attention"], "")
        self.assertEqual(report["review_summary"]["attention_detail"], "")
        self.assertEqual(report["recommended_next_command"]["label"], "rerun generation")
        self.assertNotIn("source_label", report["recommended_next_command"])
        self.assertEqual(report["recommended_next_command"]["reason_detail"], "Rerun the generation request.")
        self.assertIn("<prompt>", report["review_summary"]["next_command"])
        self.assertEqual(report["review_summary"]["requires_env"], ["CROWDTENSOR_ADMIN_TOKEN"])
        self.assertTrue(report["review_summary"]["has_recommended_command"])
        self.assertTrue(report["review_summary"]["public_artifact_safe"])
        self.assertNotIn("admin-secret", encoded)
        self.assertIn(("GET", "/admin/results?status=accepted&workload_type=real_llm_sharded_infer&limit=50&session_id=real-llm-session-test"), calls)
        persisted = json.loads((output_dir / "generate_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["user_status"]["state"], "completed")
        self.assertEqual(persisted["trace"]["session_id"], "real-llm-session-test")
        self.assertEqual(persisted["trace"]["accepted_rows_seen"], 1)
        self.assertEqual(persisted["trace"]["request_count"], 1)
        self.assertEqual(persisted["result"]["status"], "complete")
        self.assertEqual(persisted["result"]["generated_token_count"], 2)
        self.assertEqual(persisted["result"]["max_new_tokens"], 2)
        self.assertEqual(persisted["result"]["output_count"], 1)
        self.assertEqual(persisted["result"]["display"], "hash-only-json")
        self.assertTrue(persisted["result"]["public_artifact_safe"])
        self.assertFalse(persisted["output_request"]["include_output"])
        self.assertFalse(persisted["output_request"]["raw_prompt_public"])
        self.assertFalse(persisted["output_request"]["raw_generated_text_public"])
        self.assertFalse(persisted["output_request"]["generated_token_ids_public"])
        self.assertTrue(persisted["output_request"]["public_artifact_safe"])
        self.assertTrue(persisted["trace"]["request_trace"][0]["prompt_hash"])
        self.assertEqual(persisted["prompt_scope"]["source"], "prompt-text")
        self.assertTrue(persisted["prompt_scope"]["inline_prompt_text"])
        self.assertTrue(persisted["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertFalse(persisted["prompt_scope"]["terminal_local_paths"])
        self.assertTrue(persisted["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertFalse(persisted["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(persisted["prompt_scope"]["public_artifact_safe"])
        self.assertFalse(persisted["safety"]["raw_prompt_public"])
        self.assertFalse(persisted["safety"]["raw_generated_text_public"])
        self.assertFalse(persisted["safety"]["generated_token_ids_public"])
        self.assertTrue(persisted["safety"]["read_only_workload"])
        self.assertTrue(persisted["safety"]["coordinator_backed"])
        self.assertTrue(persisted["safety"]["not_production"])
        self.assertTrue(persisted["safety"]["not_large_model_serving"])
        self.assertTrue(persisted["safety"]["not_arbitrary_public_prompt_serving"])
        self.assertTrue(persisted["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertEqual(persisted["shareable_summary"]["answer_scope_state"], "json-suppressed")
        self.assertFalse(persisted["shareable_summary"]["local_answer_terminal_only"])
        self.assertEqual(persisted["local_output"]["output_count"], 1)
        self.assertFalse(persisted["local_output"]["available"])
        self.assertTrue(persisted["local_output"]["public_artifact_safe"])
        self.assertEqual(persisted["answer_scope"]["scope_state"], "json-suppressed")
        self.assertEqual(
            persisted["local_output_note"],
            "Generated output is present, but raw text is suppressed in JSON/public output; rerun without --json for local display.",
        )
        self.assertEqual(persisted["issue_summary"]["state"], "completed")
        self.assertEqual(persisted["artifact_summary"]["inspect_first"], str(output_dir / "generate_summary.md"))
        self.assertTrue(persisted["artifact_summary"]["public_artifact_safe"])
        self.assertEqual(persisted["review_summary"]["state"], "completed")
        self.assertEqual(persisted["review_summary"]["inspect_first"], str(output_dir / "generate_summary.md"))
        self.assertEqual(persisted["review_summary"]["attention_detail"], "")
        self.assertEqual(persisted["recommended_next_command"]["reason_detail"], "Rerun the generation request.")
        self.assertIn("<prompt>", persisted["review_summary"]["next_command"])
        self.assertNotIn("CrowdTensor prompt", persisted["review_summary"]["next_command"])
        self.assertTrue(persisted["review_summary"]["public_artifact_safe"])
        self.assertNotIn("local generated text must stay local", json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "generate_summary.md").read_text(encoding="utf-8")
        self.assertIn(
            "- Status: `completed: Generation completed. next=rerun_or_review_artifacts recommendation=rerun generation public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            f"- Review: `state=completed next=rerun_or_review_artifacts inspect={output_dir / 'generate_summary.md'} recommended=rerun generation primary=public_swarm_generate_ready attention=none public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Review next: `label=rerun generation reason=rerun_generation command=CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor generate --max-new-tokens 2",
            markdown,
        )
        self.assertIn(
            "- Issue: `state=completed primary=public_swarm_generate_ready next=rerun_or_review_artifacts progress=",
            markdown,
        )
        self.assertIn(
            "- Recommended next: `rerun generation` reason=`rerun_generation` command=`CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor generate --max-new-tokens 2",
            markdown,
        )
        self.assertIn(
            "2. `rerun generation`: `CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor generate --max-new-tokens 2",
            markdown,
        )
        self.assertIn("- Reason: Rerun the generation request.", markdown)
        self.assertIn("requires=`CROWDTENSOR_ADMIN_TOKEN`", markdown)
        self.assertIn("- Generation: `2/2` hash=`sha256:generated`", markdown)
        self.assertIn("- Result: `status=complete tokens=2/2 outputs=1 display=hash-only-json hash=sha256:generated public_artifact_safe=True`", markdown)
        self.assertIn(
            "- Trace: `session=real-llm-session-test requests=1 ledger_rows=1 stream_events=0 source=public_swarm_product_cli_v1 public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=json-suppressed local_answer_terminal_only=False`",
            markdown,
        )
        self.assertIn(
            "- Prompt scope: `source=prompt-text count=1 inline_prompt_text=True terminal_next_commands_local_private=True terminal_local_paths=False saved_artifacts_prompt_placeholders=True prompt_file_path_public=False raw_prompt_public=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Local output: `available=False display_only=False public_artifact_safe=True saved_redacted=True` count=`1` source=``",
            markdown,
        )
        self.assertIn(
            "- Answer scope: `state=json-suppressed terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Local output note: Generated output is present, but raw text is suppressed in JSON/public output; rerun without --json for local display.",
            markdown,
        )
        self.assertIn(
            f"- Artifacts: `inspect={output_dir / 'generate_summary.md'} json={output_dir / 'generate_summary.json'} markdown={output_dir / 'generate_summary.md'} present=2/2 public_artifact_safe=True`",
            markdown,
        )
        self.assertIn("## Artifacts", markdown)
        self.assertIn(
            "- `generate_summary`: path=`generate_summary.json` present=`True` kind=`crowdtensor_generate_summary`",
            markdown,
        )
        self.assertIn("Raw generated text and generated token ids are redacted", markdown)
        self.assertNotIn("local generated text must stay local", markdown)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn(
            "  status: completed: Generation completed. next=rerun_or_review_artifacts recommendation=rerun generation public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            f"  review: state=completed next=rerun_or_review_artifacts inspect={output_dir / 'generate_summary.md'} recommended=rerun generation primary=public_swarm_generate_ready attention=none public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  review_next: label=rerun generation reason=rerun_generation command=CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor generate --max-new-tokens 2",
            rendered,
        )
        self.assertIn(
            "  issue: state=completed primary=public_swarm_generate_ready next=rerun_or_review_artifacts progress=polls=",
            rendered,
        )
        self.assertIn(
            "  result: status=complete tokens=2/2 outputs=1 display=hash-only-json hash=sha256:generated public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  local_output: available=False display_only=False public_artifact_safe=True saved_redacted=True count=1 source=none",
            rendered,
        )
        self.assertNotIn("  answer:", rendered)
        self.assertIn(
            "  trace: session=real-llm-session-test requests=1 ledger_rows=1 stream_events=0 source=public_swarm_product_cli_v1 public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=json-suppressed local_answer_terminal_only=False",
            rendered,
        )
        self.assertIn(
            f"  artifacts: inspect={output_dir / 'generate_summary.md'} json={output_dir / 'generate_summary.json'} markdown={output_dir / 'generate_summary.md'} present=2/2 public_artifact_safe=True",
            rendered,
        )
        self.assertIn("next[1] check generation route", rendered)
        self.assertIn("next[2] rerun generation", rendered)
        self.assertIn(f"  output_dir: {output_dir}", rendered)
        self.assertIn("  recommended_reason: Rerun the generation request.", rendered)
        self.assertIn(f"markdown={output_dir / 'generate_summary.md'}", rendered)

    def test_product_generate_human_output_is_not_persisted_to_summary(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--output-dir",
            str(output_dir),
            "--prompt-text",
            "CrowdTensor prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
        ])

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, path, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-human",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                }
            return {
                "results": [
                    {
                        "validation": {
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:generated",
                            "generated_text": "local generated text must stay local",
                            "generated_token_ids": [1, 2],
                            "decoded_tokens_match": True,
                        }
                    }
                ]
            }

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["local_output"]["generated_text"], "local generated text must stay local")
        self.assertFalse(report["local_output"]["public_artifact_safe"])
        self.assertEqual(report["result"]["status"], "complete")
        self.assertEqual(report["result"]["output_count"], 1)
        self.assertEqual(report["result"]["display"], "local-private")
        self.assertFalse(report["result"]["public_artifact_safe"])
        self.assertEqual(report["output_display"]["terminal_display"], "local-private")
        self.assertTrue(report["output_display"]["terminal_text_available"])
        self.assertEqual(report["output_display"]["saved_artifact_display"], "hash-only")
        self.assertEqual(report["output_display"]["json_stdout_display"], "hash-only-json")
        self.assertFalse(report["output_display"]["include_output_requested"])
        self.assertTrue(report["answer_scope"]["terminal_only"])
        self.assertTrue(report["answer_scope"]["visible_in_terminal"])
        self.assertEqual(report["answer_scope"]["scope_state"], "terminal-visible")
        self.assertEqual(report["answer_scope"]["saved_json_display"], "hash-only")
        self.assertEqual(report["answer_scope"]["saved_markdown_display"], "hash-only")
        self.assertTrue(report["answer_scope"]["public_artifact_safe"])
        self.assertEqual(report["answer_scope"]["summary"], cli.LOCAL_ANSWER_SCOPE_TEXT)
        persisted = json.loads((output_dir / "generate_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["local_output"]["generated_text"], "")
        self.assertEqual(persisted["local_output"]["outputs"][0]["generated_text"], "")
        self.assertTrue(persisted["local_output"]["public_artifact_safe"])
        self.assertEqual(persisted["result"]["status"], "complete")
        self.assertEqual(persisted["result"]["output_count"], 1)
        self.assertEqual(persisted["result"]["display"], "hash-only")
        self.assertTrue(persisted["result"]["public_artifact_safe"])
        self.assertEqual(persisted["output_display"]["terminal_display"], "saved-terminal-redacted")
        self.assertFalse(persisted["output_display"]["terminal_text_available"])
        self.assertEqual(persisted["output_display"]["saved_artifact_display"], "hash-only")
        self.assertEqual(persisted["output_display"]["json_stdout_display"], "hash-only-json")
        self.assertFalse(persisted["output_display"]["include_output_requested"])
        self.assertFalse(persisted["answer_scope"]["terminal_only"])
        self.assertFalse(persisted["answer_scope"]["visible_in_terminal"])
        self.assertEqual(persisted["answer_scope"]["scope_state"], "saved-terminal-redacted")
        self.assertEqual(persisted["answer_scope"]["saved_json_display"], "hash-only")
        self.assertEqual(persisted["answer_scope"]["saved_markdown_display"], "hash-only")
        self.assertTrue(persisted["answer_scope"]["public_artifact_safe"])
        self.assertEqual(persisted["answer_scope"]["summary"], cli.SAVED_TERMINAL_ANSWER_SCOPE_TEXT)
        self.assertNotIn("local generated text must stay local", json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "generate_summary.md").read_text(encoding="utf-8")
        self.assertIn("- Result: `status=complete tokens=2/2 outputs=1 display=hash-only", markdown)
        self.assertIn(
            "- Output display: `terminal=saved-terminal-redacted terminal_text=False saved=hash-only json_stdout=hash-only-json include_output=False raw_public=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Local output: `available=False display_only=False public_artifact_safe=True saved_redacted=True` count=`1` source=`coordinator-validation`",
            markdown,
        )
        self.assertIn(
            "- Answer scope: `state=saved-terminal-redacted terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(f"- Answer scope note: {cli.SAVED_TERMINAL_ANSWER_SCOPE_TEXT}", markdown)
        self.assertNotIn("rerun without --json for local display", markdown)
        self.assertIn(
            "- Local output note: Raw generated text is shown only in local human output; JSON and public artifacts expose hashes only.",
            markdown,
        )
        self.assertNotIn("local generated text must stay local", markdown)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn("  answer: local generated text must stay local", rendered)
        self.assertIn(
            "  answer_scope: state=terminal-visible terminal_only=True visible_in_terminal=True saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            rendered,
        )
        self.assertIn(f"  answer_scope_note: {cli.LOCAL_ANSWER_SCOPE_TEXT}", rendered)
        self.assertIn("  result: status=complete tokens=2/2 outputs=1 display=local-private", rendered)
        self.assertIn(
            "  output_display: terminal=local-private terminal_text=True saved=hash-only json_stdout=hash-only-json include_output=False raw_public=False public_artifact_safe=True",
            rendered,
        )
        self.assertIn(f"  output_display_note: {cli.LOCAL_OUTPUT_DISPLAY_SCOPE_TEXT}", rendered)
        self.assertIn(
            "  local_output: available=True display_only=True public_artifact_safe=False count=1 source=coordinator-validation",
            rendered,
        )
        self.assertLess(rendered.index("  answer: local generated text must stay local"), rendered.index("  local_output: "))
        self.assertLess(rendered.index("  answer_scope: "), rendered.index("  local_output: "))
        self.assertLess(rendered.index("  answer: local generated text must stay local"), rendered.index("  trace: "))
        self.assertIn(f"  output_dir: {output_dir}", rendered)
        self.assertIn(f"markdown={output_dir / 'generate_summary.md'}", rendered)

    def test_product_generate_marks_truncated_local_output(self) -> None:
        output_dir = Path(self._tmp_dir())
        tail = "GENERATE_SECRET_TAIL"
        generated_text = ("g" * cli.LOCAL_OUTPUT_DISPLAY_MAX_CHARS) + tail
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--output-dir",
            str(output_dir),
            "--prompt-text",
            "CrowdTensor prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
        ])

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, path, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-truncated",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                }
            return {
                "results": [
                    {
                        "validation": {
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": cli.stable_hash_text(generated_text),
                            "generated_text": generated_text,
                            "decoded_tokens_match": True,
                        }
                    }
                ]
            }

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(len(report["local_output"]["generated_text"]), cli.LOCAL_OUTPUT_DISPLAY_MAX_CHARS)
        self.assertTrue(report["local_output"]["truncated"])
        self.assertEqual(report["local_output"]["max_chars"], cli.LOCAL_OUTPUT_DISPLAY_MAX_CHARS)
        self.assertEqual(report["local_output"]["omitted_char_count"], len(tail))
        self.assertTrue(report["local_output"]["outputs"][0]["truncated"])
        self.assertNotIn(tail, json.dumps(report, sort_keys=True))
        self.assertIn(
            f"Terminal answer text is truncated to {cli.LOCAL_OUTPUT_DISPLAY_MAX_CHARS} chars per output. Omitted chars: {len(tail)}.",
            report["local_output_note"],
        )
        persisted = json.loads((output_dir / "generate_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["local_output"]["generated_text"], "")
        self.assertTrue(persisted["local_output"]["truncated"])
        self.assertEqual(persisted["local_output"]["omitted_char_count"], len(tail))
        self.assertTrue(persisted["local_output"]["public_artifact_safe"])
        self.assertNotIn(tail, json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "generate_summary.md").read_text(encoding="utf-8")
        self.assertIn(
            f"- Local output: `available=False display_only=False public_artifact_safe=True saved_redacted=True truncated=True max_chars={cli.LOCAL_OUTPUT_DISPLAY_MAX_CHARS} omitted_chars={len(tail)}` count=`1` source=`coordinator-validation`",
            markdown,
        )
        self.assertIn("Terminal answer text is truncated", markdown)
        self.assertNotIn(tail, markdown)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn(
            f"local_output: available=True display_only=True public_artifact_safe=False truncated=True max_chars={cli.LOCAL_OUTPUT_DISPLAY_MAX_CHARS} omitted_chars={len(tail)} count=1 source=coordinator-validation",
            rendered,
        )
        self.assertIn("Terminal answer text is truncated", rendered)
        self.assertNotIn(tail, rendered)

    def test_product_generate_timeout_reports_safe_wait_progress(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-text",
            "CrowdTensor prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "4",
            "--timeout-seconds",
            "1",
            "--poll-interval",
            "0.01",
            "--http-timeout",
            "7",
            "--admin-results-limit",
            "9",
            "--stream",
            "--json",
        ])
        monotonic_values = iter([0.0, 0.2, 1.2])
        stream_failed = False

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            nonlocal stream_failed
            del base_url, payload, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-timeout",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 4,
                    "backend": "hf_transformers_cpu",
                }
            if path.startswith("/admin/session-stream"):
                if not stream_failed:
                    stream_failed = True
                    raise cli.HTTPError(
                        path,
                        500,
                        "stream failed with CrowdTensor prompt and admin-secret",
                        {},
                        io.BytesIO(b"stream echoed CrowdTensor prompt and admin-secret"),
                    )
                return {
                    "schema": "admin_session_stream_v1",
                    "events": [
                        {
                            "schema": "session_stream_event_v1",
                            "request_id": "req-1",
                            "prompt_hash": "sha256:prompt",
                            "generated_token_count": 1,
                            "max_new_tokens": 4,
                            "generated_text_hash": "sha256:step1",
                            "generated_text": "must not leak",
                            "generated_token_ids": [1],
                        }
                    ],
                }
            return {
                "results": [
                    {
                        "validation": {
                            "generated_token_count": 1,
                            "max_new_tokens": 4,
                            "generated_text_hash": "sha256:partial",
                            "generated_text": "must not leak",
                            "generated_token_ids": [1],
                            "decoded_tokens_match": True,
                        }
                    }
                ]
            }

        with patch.object(cli, "request_json_url", side_effect=fake_request), patch.object(
            cli.time,
            "monotonic",
            side_effect=lambda: next(monotonic_values),
        ), patch.object(cli.time, "sleep", return_value=None):
            report = cli.build_product_generate(args)

        encoded = json.dumps(report, sort_keys=True)
        self.assertFalse(report["ok"], report)
        self.assertIn("generation_timeout", report["diagnosis_codes"])
        self.assertTrue(report["wait_progress"]["session_created"])
        self.assertTrue(report["wait_progress"]["ledger_endpoint_ready"])
        self.assertFalse(report["wait_progress"]["stream_endpoint_ready"])
        self.assertEqual(report["wait_progress"]["poll_count"], 1)
        self.assertEqual(report["wait_progress"]["accepted_rows_seen"], 1)
        self.assertEqual(report["wait_progress"]["stream_event_count"], 1)
        self.assertEqual(report["wait_progress"]["max_observed_token_count"], 1)
        self.assertEqual(report["wait_progress"]["target_token_count"], 4)
        self.assertFalse(report["wait_progress"]["completion_observed"])
        self.assertEqual(report["wait_progress"]["last_error_type"], "HTTPError")
        self.assertIn("<redacted>", report["wait_progress"]["last_error_detail"])
        self.assertNotIn("CrowdTensor prompt", report["wait_progress"]["last_error_detail"])
        self.assertNotIn("admin-secret", report["wait_progress"]["last_error_detail"])
        self.assertIn("Generation reached 1/4 tokens", report["operator_action"])
        self.assertEqual(report["issue_summary"]["state"], "blocked")
        self.assertEqual(report["issue_summary"]["primary_code"], "generation_timeout")
        self.assertEqual(report["issue_summary"]["next_step"], "fix_blockers")
        self.assertIn("tokens=1/4", report["issue_summary"]["progress"])
        self.assertTrue(report["issue_summary"]["safe_detail_present"])
        self.assertNotIn("CrowdTensor prompt", json.dumps(report["issue_summary"], sort_keys=True))
        self.assertNotIn("admin-secret", json.dumps(report["issue_summary"], sort_keys=True))
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            "crowdtensor generate --max-new-tokens 4 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --stream --timeout-seconds 120 --poll-interval 0.01 --http-timeout 7.0 --admin-results-limit 9",
            next_lines,
        )
        self.assertIn(
            "crowdtensor generate --max-new-tokens 4 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN} --stream --timeout-seconds 1.0 --poll-interval 0.01 --http-timeout 7.0 --admin-results-limit 9",
            next_lines,
        )
        self.assertIn(
            "crowdtensor generate --max-new-tokens 4 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --stream --timeout-seconds 1.0 --poll-interval 0.01 --http-timeout 7.0 --admin-results-limit 9",
            next_lines,
        )
        retry = next(item for item in report["next_commands"] if item["label"] == "retry generation with longer timeout")
        self.assertEqual(retry["requires_env"], ["CROWDTENSOR_ADMIN_TOKEN"])
        self.assertEqual(retry["command"].count("--timeout-seconds"), 1)
        self.assertEqual(retry["command"][retry["command"].index("--timeout-seconds") + 1], "120")
        self.assertEqual(retry["command"].count("--poll-interval"), 1)
        self.assertEqual(retry["command"][retry["command"].index("--poll-interval") + 1], "0.01")
        self.assertEqual(retry["command"].count("--http-timeout"), 1)
        self.assertEqual(retry["command"][retry["command"].index("--http-timeout") + 1], "7.0")
        self.assertEqual(retry["command"].count("--admin-results-limit"), 1)
        self.assertEqual(retry["command"][retry["command"].index("--admin-results-limit") + 1], "9")
        self.assertNotIn("--timeout-seconds 1 --timeout-seconds 120", retry["command_line"])
        self.assertEqual(report["runtime_options"]["poll_interval"], 0.01)
        self.assertEqual(report["runtime_options"]["http_timeout"], 7.0)
        self.assertEqual(report["runtime_options"]["admin_results_limit"], 9)
        self.assertTrue(report["runtime_options"]["public_artifact_safe"])
        self.assertEqual(
            report["recommended_next_command"]["reason_detail"],
            "Retry the same request with a longer timeout after incomplete or partial progress.",
        )
        self.assertNotIn("must not leak", encoded)
        self.assertNotIn("CrowdTensor prompt", encoded)
        self.assertNotIn("admin-secret", encoded)
        self.assertNotIn('"generated_token_ids": [1]', encoded)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn(
            "  issue: state=blocked primary=generation_timeout next=fix_blockers progress=polls=1 accepted_rows=1 tokens=1/4 ledger=True stream=False last_error=HTTPError safe_detail=True",
            rendered,
        )
        self.assertIn("  stream_events: 1 source=admin-results-ledger-fallback complete=False", rendered)
        self.assertIn("  wait: polls=1 accepted_rows=1 tokens=1/4 ledger=True stream=False last_error=HTTPError", rendered)
        self.assertIn(
            "  runtime_options: timeout_seconds=1.0 poll_interval=0.01 http_timeout=7.0 admin_results_limit=9 public_artifact_safe=True",
            rendered,
        )
        self.assertNotIn("stream echoed CrowdTensor prompt", rendered)
        self.assertNotIn("admin-secret", rendered)
        self.assertIn("  action: Generation reached 1/4 tokens before timeout", rendered)
        self.assertIn("  next[", rendered)
        self.assertIn("retry generation with longer timeout", rendered)
        persisted = json.loads((Path(report["output_dir"]) / "generate_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["runtime_options"]["timeout_seconds"], 1.0)
        self.assertEqual(persisted["runtime_options"]["poll_interval"], 0.01)
        self.assertEqual(persisted["runtime_options"]["http_timeout"], 7.0)
        self.assertEqual(persisted["runtime_options"]["admin_results_limit"], 9)
        markdown = (Path(report["output_dir"]) / "generate_summary.md").read_text(encoding="utf-8")
        self.assertIn(
            "- Runtime options: `timeout_seconds=1.0 poll_interval=0.01 http_timeout=7.0 admin_results_limit=9 public_artifact_safe=True`",
            markdown,
        )

    def test_product_generate_batch_timeout_prints_request_progress(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-texts",
            "first private prompt,second private prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--timeout-seconds",
            "1",
            "--poll-interval",
            "0.01",
            "--json",
        ])
        monotonic_values = iter([0.0, 0.2, 1.2])

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-batch-timeout",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                    "request_count": 2,
                }
            self.assertIn("session_id=real-llm-session-batch-timeout", path)
            return {
                "results": [
                    {
                        "validation": {
                            "request_count": 2,
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:partial",
                            "decoded_tokens_match": True,
                            "inference_results": [
                                {
                                    "request_id": "req-1",
                                    "prompt_hash": "sha256:p1",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g1",
                                    "generated_text": "raw one must not leak",
                                    "generated_token_ids": [1, 2],
                                    "decoded_tokens_match": True,
                                }
                            ],
                        }
                    }
                ]
            }

        with patch.object(cli, "request_json_url", side_effect=fake_request), patch.object(
            cli.time,
            "monotonic",
            side_effect=lambda: next(monotonic_values),
        ), patch.object(cli.time, "sleep", return_value=None):
            report = cli.build_product_generate(args)

        encoded = json.dumps(report, sort_keys=True)
        self.assertFalse(report["ok"], report)
        self.assertIn("generation_timeout", report["diagnosis_codes"])
        self.assertEqual(report["wait_progress"]["expected_request_count"], 2)
        self.assertEqual(report["wait_progress"]["observed_request_count"], 1)
        self.assertFalse(report["wait_progress"]["batch_generation_ready"])
        self.assertIn("Only 1/2 batch results appeared", report["operator_action"])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn("  batch: requests=2 observed=1 ready=False", rendered)
        self.assertIn("  wait: polls=1 accepted_rows=1 tokens=2/2 requests=1/2 batch_ready=False ledger=True stream=False", rendered)
        self.assertNotIn("first private prompt", encoded)
        self.assertNotIn("second private prompt", encoded)
        self.assertNotIn("raw one must not leak", encoded)
        self.assertNotIn('"generated_token_ids":', encoded)
        self.assertNotIn("admin-secret", encoded)

    def test_product_generate_batch_timeout_without_rows_keeps_zero_request_progress(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-texts",
            "first private prompt,second private prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--timeout-seconds",
            "1",
            "--poll-interval",
            "0.01",
            "--json",
        ])
        monotonic_values = iter([0.0, 0.2, 1.2])

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, path, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-empty-batch-timeout",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                    "request_count": 2,
                }
            return {"results": []}

        with patch.object(cli, "request_json_url", side_effect=fake_request), patch.object(
            cli.time,
            "monotonic",
            side_effect=lambda: next(monotonic_values),
        ), patch.object(cli.time, "sleep", return_value=None):
            report = cli.build_product_generate(args)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["wait_progress"]["expected_request_count"], 2)
        self.assertEqual(report["wait_progress"]["observed_request_count"], 0)
        self.assertEqual(report["batch"]["observed_request_count"], 0)
        self.assertEqual(
            report["recommended_next_command"]["reason_detail"],
            "Retry the same request with a longer timeout after incomplete or partial progress.",
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn("  batch: requests=2 observed=0 ready=False", rendered)
        self.assertIn("  wait: polls=1 accepted_rows=0 tokens=0/2 requests=0/2 batch_ready=False ledger=True stream=False", rendered)

    def test_product_generate_uses_longer_timeout_for_session_create(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-text",
            "CrowdTensor prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--timeout-seconds",
            "120",
            "--http-timeout",
            "5",
            "--json",
        ])
        timeouts: list[tuple[str, float]] = []

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, payload, admin_token
            timeouts.append((path, timeout))
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-timeout",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                }
            return {
                "results": [
                    {
                        "validation": {
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:generated",
                            "decoded_tokens_match": True,
                        }
                    }
                ]
            }

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(timeouts[0], ("/admin/inference-sessions", 30.0))
        self.assertTrue(timeouts[1][0].startswith("/admin/results"))
        self.assertEqual(timeouts[1][1], 5.0)

    def test_product_generate_batch_uses_private_prompt_texts_and_safe_public_summary(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-texts",
            "first private prompt,second private prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--json",
        ])
        posted_payloads: list[dict] = []

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, admin_token, timeout
            if method == "POST":
                posted_payloads.append(payload or {})
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-batch",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                    "request_count": 2,
                }
            self.assertIn("session_id=real-llm-session-batch", path)
            return {
                "results": [
                    {
                        "validation": {
                            "request_count": 2,
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:batch",
                            "decoded_tokens_match": True,
                            "inference_results": [
                                {
                                    "request_id": "req-1",
                                    "prompt_hash": "sha256:p1",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g1",
                                    "generated_text": " raw one",
                                    "generated_token_ids": [1, 2],
                                    "decoded_tokens_match": True,
                                },
                                {
                                    "request_id": "req-2",
                                    "prompt_hash": "sha256:p2",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g2",
                                    "generated_text": " raw two",
                                    "generated_token_ids": [3, 4],
                                    "decoded_tokens_match": True,
                                },
                            ],
                        }
                    }
                ]
            }

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_product_generate(args)

        encoded = json.dumps(report, sort_keys=True)
        self.assertTrue(report["ok"], report)
        self.assertIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])
        self.assertTrue(report["batch"]["enabled"])
        self.assertEqual(report["batch"]["request_count"], 2)
        self.assertTrue(report["batch"]["batch_generation_ready"])
        self.assertEqual(posted_payloads[0]["request_count"], 2)
        self.assertEqual(posted_payloads[0]["prompt"], "first private prompt")
        self.assertEqual(posted_payloads[0]["prompt_texts"], ["first private prompt", "second private prompt"])
        self.assertEqual([row["generated_text_hash"] for row in report["generation"]["results"]], ["sha256:g1", "sha256:g2"])
        self.assertNotIn("first private prompt", encoded)
        self.assertNotIn("second private prompt", encoded)
        self.assertNotIn("raw one", encoded)
        self.assertNotIn("raw two", encoded)
        self.assertNotIn('"generated_token_ids":', encoded)
        self.assertNotIn("admin-secret", encoded)

    def test_product_generate_batch_waits_for_each_prompt_token_target(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-texts",
            "first private prompt,second private prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--poll-interval",
            "0.01",
            "--json",
        ])
        ledger_payloads = [
            {
                "results": [
                    {
                        "validation": {
                            "request_count": 2,
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:partial",
                            "decoded_tokens_match": True,
                            "inference_results": [
                                {
                                    "request_id": "req-1",
                                    "prompt_hash": "sha256:p1",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g1",
                                    "decoded_tokens_match": True,
                                },
                                {
                                    "request_id": "req-2",
                                    "prompt_hash": "sha256:p2",
                                    "generated_token_count": 1,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g2-partial",
                                    "decoded_tokens_match": True,
                                },
                            ],
                        }
                    }
                ]
            },
            {
                "results": [
                    {
                        "validation": {
                            "request_count": 2,
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:batch",
                            "decoded_tokens_match": True,
                            "inference_results": [
                                {
                                    "request_id": "req-1",
                                    "prompt_hash": "sha256:p1",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g1",
                                    "decoded_tokens_match": True,
                                },
                                {
                                    "request_id": "req-2",
                                    "prompt_hash": "sha256:p2",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g2",
                                    "decoded_tokens_match": True,
                                },
                            ],
                        }
                    }
                ]
            },
        ]

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-batch",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                    "request_count": 2,
                }
            self.assertIn("session_id=real-llm-session-batch", path)
            return ledger_payloads.pop(0)

        with patch.object(cli, "request_json_url", side_effect=fake_request), patch.object(cli.time, "sleep", return_value=None):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(ledger_payloads, [])
        self.assertTrue(report["batch"]["batch_generation_ready"])
        self.assertEqual(
            [row["generated_token_count"] for row in report["generation"]["results"]],
            [2, 2],
        )
        self.assertIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])

    def test_product_generate_batch_waits_for_missing_prompt_result(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-texts",
            "first private prompt,second private prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--poll-interval",
            "0.01",
            "--json",
        ])
        ledger_payloads = [
            {
                "results": [
                    {
                        "validation": {
                            "request_count": 2,
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:partial",
                            "decoded_tokens_match": True,
                            "inference_results": [
                                {
                                    "request_id": "req-1",
                                    "prompt_hash": "sha256:p1",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g1",
                                    "decoded_tokens_match": True,
                                }
                            ],
                        }
                    }
                ]
            },
            {
                "results": [
                    {
                        "validation": {
                            "request_count": 2,
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:batch",
                            "decoded_tokens_match": True,
                            "inference_results": [
                                {
                                    "request_id": "req-1",
                                    "prompt_hash": "sha256:p1",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g1",
                                    "decoded_tokens_match": True,
                                },
                                {
                                    "request_id": "req-2",
                                    "prompt_hash": "sha256:p2",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g2",
                                    "decoded_tokens_match": True,
                                },
                            ],
                        }
                    }
                ]
            },
        ]

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-batch",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                    "request_count": 2,
                }
            self.assertIn("session_id=real-llm-session-batch", path)
            return ledger_payloads.pop(0)

        with patch.object(cli, "request_json_url", side_effect=fake_request), patch.object(cli.time, "sleep", return_value=None):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(ledger_payloads, [])
        self.assertTrue(report["batch"]["batch_generation_ready"])
        self.assertEqual(report["generation"]["request_count"], 2)
        self.assertEqual(report["generation"]["observed_request_count"], 2)
        self.assertIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])

    def test_product_generate_batch_waits_for_per_request_results_not_aggregate_only(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-texts",
            "first private prompt,second private prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--poll-interval",
            "0.01",
            "--json",
        ])
        ledger_payloads = [
            {
                "results": [
                    {
                        "validation": {
                            "request_count": 2,
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:aggregate-only",
                            "decoded_tokens_match": True,
                        }
                    }
                ]
            },
            {
                "results": [
                    {
                        "validation": {
                            "request_count": 2,
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:batch",
                            "decoded_tokens_match": True,
                            "inference_results": [
                                {
                                    "request_id": "req-1",
                                    "prompt_hash": "sha256:p1",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g1",
                                    "decoded_tokens_match": True,
                                },
                                {
                                    "request_id": "req-2",
                                    "prompt_hash": "sha256:p2",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g2",
                                    "decoded_tokens_match": True,
                                },
                            ],
                        }
                    }
                ]
            },
        ]

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-batch",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                    "request_count": 2,
                }
            self.assertIn("session_id=real-llm-session-batch", path)
            return ledger_payloads.pop(0)

        with patch.object(cli, "request_json_url", side_effect=fake_request), patch.object(cli.time, "sleep", return_value=None):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(ledger_payloads, [])
        self.assertTrue(report["batch"]["batch_generation_ready"])
        self.assertEqual(report["generation"]["observed_request_count"], 2)
        self.assertEqual([row["generated_text_hash"] for row in report["generation"]["results"]], ["sha256:g1", "sha256:g2"])

    def test_product_generate_batch_waits_for_unique_request_identity(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-texts",
            "first private prompt,second private prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--poll-interval",
            "0.01",
            "--json",
        ])
        ledger_payloads = [
            {
                "results": [
                    {
                        "validation": {
                            "request_count": 2,
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:duplicate",
                            "decoded_tokens_match": True,
                            "inference_results": [
                                {
                                    "request_id": "req-1",
                                    "prompt_hash": "sha256:p1",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g1",
                                    "decoded_tokens_match": True,
                                },
                                {
                                    "request_id": "req-1",
                                    "prompt_hash": "sha256:p1",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g1-duplicate",
                                    "decoded_tokens_match": True,
                                },
                            ],
                        }
                    }
                ]
            },
            {
                "results": [
                    {
                        "validation": {
                            "request_count": 2,
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:batch",
                            "decoded_tokens_match": True,
                            "inference_results": [
                                {
                                    "request_id": "req-1",
                                    "prompt_hash": "sha256:p1",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g1",
                                    "decoded_tokens_match": True,
                                },
                                {
                                    "request_id": "req-2",
                                    "prompt_hash": "sha256:p2",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g2",
                                    "decoded_tokens_match": True,
                                },
                            ],
                        }
                    }
                ]
            },
        ]

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-batch",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                    "request_count": 2,
                }
            self.assertIn("session_id=real-llm-session-batch", path)
            return ledger_payloads.pop(0)

        with patch.object(cli, "request_json_url", side_effect=fake_request), patch.object(cli.time, "sleep", return_value=None):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(ledger_payloads, [])
        self.assertTrue(report["generation"]["batch_identity_ready"])
        self.assertTrue(report["batch"]["batch_generation_ready"])
        self.assertEqual([row["request_id"] for row in report["generation"]["results"]], ["req-1", "req-2"])

    def test_product_generate_batch_rejects_more_than_four_prompts(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "generate",
                "--coordinator-url",
                "http://127.0.0.1:8787",
                "--prompt-texts",
                "a,b,c,d,e",
                "--json",
            ])

    def test_product_generate_stream_reports_safe_progress_events(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-text",
            "CrowdTensor prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "3",
            "--stream",
            "--json",
        ])
        def event(count: int, miner_id: str) -> dict:
            return {
                "schema": "session_stream_event_v1",
                "task_id": f"stage1-step{count - 1}",
                "session_id": "real-llm-session-test",
                "miner_id": miner_id,
                "stage_id": 1,
                "generated_token_count": count,
                "max_new_tokens": 3,
                "generation_step": count - 1,
                "generated_text_hash": f"sha256:step{count - 1}",
                "decoded_tokens_match": True,
                "observed_at": float(count),
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            }

        ledgers = [
            {
                "results": [
                    {
                        "task_id": "stage1-step0",
                        "session_id": "real-llm-session-test",
                        "miner_id": "stage1-a",
                        "validation": {
                            "session_id": "real-llm-session-test",
                            "stage_id": 1,
                            "generation_step": 0,
                            "generated_token_count": 1,
                            "max_new_tokens": 3,
                            "generated_text_hash": "sha256:step0",
                            "generated_text": " raw step zero",
                            "generated_token_ids": [101],
                            "decoded_tokens_match": True,
                        },
                    }
                ]
            },
            {
                "results": [
                    {
                        "task_id": "stage1-step0",
                        "session_id": "real-llm-session-test",
                        "miner_id": "stage1-a",
                        "validation": {
                            "session_id": "real-llm-session-test",
                            "stage_id": 1,
                            "generation_step": 0,
                            "generated_token_count": 1,
                            "max_new_tokens": 3,
                            "generated_text_hash": "sha256:step0",
                            "generated_text": " raw step zero",
                            "generated_token_ids": [101],
                            "decoded_tokens_match": True,
                        },
                    },
                    {
                        "task_id": "stage1-step1",
                        "session_id": "real-llm-session-test",
                        "miner_id": "stage1-b",
                        "validation": {
                            "session_id": "real-llm-session-test",
                            "stage_id": 1,
                            "generation_step": 1,
                            "generated_token_count": 2,
                            "max_new_tokens": 3,
                            "generated_text_hash": "sha256:step1",
                            "generated_text": " raw step one",
                            "generated_token_ids": [101, 102],
                            "decoded_tokens_match": True,
                        },
                    },
                ]
            },
            {
                "results": [
                    {
                        "task_id": "stage1-step2",
                        "session_id": "real-llm-session-test",
                        "miner_id": "stage1-c",
                        "validation": {
                            "session_id": "real-llm-session-test",
                            "stage_id": 1,
                            "generation_step": 2,
                            "generated_token_count": 3,
                            "max_new_tokens": 3,
                            "generated_text_hash": "sha256:step2",
                            "generated_text": " raw final text",
                            "generated_token_ids": [101, 102, 103],
                            "decoded_tokens_match": True,
                        },
                    }
                ]
            },
        ]
        stream_payloads = [
            {"schema": "admin_session_stream_v1", "events": [event(1, "stage1-a")]},
            {"schema": "admin_session_stream_v1", "events": [event(1, "stage1-a"), event(2, "stage1-b")]},
            {
                "schema": "admin_session_stream_v1",
                "events": [event(1, "stage1-a"), event(2, "stage1-b"), event(3, "stage1-c")],
            },
        ]

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-test",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 3,
                    "backend": "hf_transformers_cpu",
                }
            if path.startswith("/admin/session-stream"):
                return stream_payloads.pop(0)
            return ledgers.pop(0)

        with patch.object(cli, "request_json_url", side_effect=fake_request), patch.object(cli.time, "sleep", return_value=None):
            report = cli.build_product_generate(args)

        encoded = json.dumps(report, sort_keys=True)
        self.assertTrue(report["ok"], report)
        self.assertIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_endpoint_ready", report["diagnosis_codes"])
        self.assertEqual(report["stream"]["event_count"], 3)
        self.assertEqual(report["stream"]["source"], "admin-session-stream")
        self.assertTrue(report["stream"]["endpoint_ready"])
        self.assertTrue(report["stream"]["stream_generation_ready"])
        self.assertTrue(report["stream"]["progress"]["stream_progress_complete"])
        self.assertTrue(report["stream"]["progress"]["all_token_events_ready"])
        self.assertTrue(report["stream"]["progress"]["monotonic_progress"])
        self.assertEqual(report["stream"]["progress"]["observed_token_counts"], [1, 2, 3])
        self.assertEqual(report["stream"]["progress"]["max_observed_token_count"], 3)
        self.assertEqual(
            [event["generated_token_count"] for event in report["stream"]["events"]],
            [1, 2, 3],
        )
        self.assertEqual(report["stream"]["events"][-1]["generated_text_hash"], "sha256:step2")
        self.assertTrue(report["stream"]["events"][-1]["generated_token_ids_public"] is False)
        self.assertNotIn("raw step", encoded)
        self.assertNotIn("raw final text", encoded)
        self.assertNotIn('"generated_token_ids":', encoded)
        self.assertNotIn("admin-secret", encoded)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn("  stream_progress: request=unknown tokens=3/3 counts=[1, 2, 3] complete=True", rendered)
        self.assertNotIn("raw final text", rendered)
        self.assertNotIn("admin-secret", rendered)

    def test_product_generate_batch_stream_requires_each_prompt_progress(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-texts",
            "first private prompt,second private prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--stream",
            "--json",
        ])

        def event(request_id: str, prompt_hash: str, count: int) -> dict:
            return {
                "schema": "session_stream_event_v1",
                "task_id": f"{request_id}-stage1-step{count - 1}",
                "session_id": "real-llm-session-batch-stream",
                "miner_id": f"stage1-{request_id}",
                "stage_id": 1,
                "request_id": request_id,
                "prompt_hash": prompt_hash,
                "generated_token_count": count,
                "max_new_tokens": 2,
                "generation_step": count - 1,
                "generated_text_hash": f"sha256:{request_id}-{count}",
                "decoded_tokens_match": True,
                "observed_at": float(count),
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            }

        final_row = {
            "validation": {
                "request_count": 2,
                "generated_token_count": 2,
                "max_new_tokens": 2,
                "generated_text_hash": "sha256:batch",
                "decoded_tokens_match": True,
                "inference_results": [
                    {
                        "request_id": "req-1",
                        "prompt_hash": "sha256:p1",
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:req-1-2",
                        "generated_text": " raw one",
                        "generated_token_ids": [1, 2],
                        "decoded_tokens_match": True,
                    },
                    {
                        "request_id": "req-2",
                        "prompt_hash": "sha256:p2",
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:req-2-2",
                        "generated_text": " raw two",
                        "generated_token_ids": [3, 4],
                        "decoded_tokens_match": True,
                    },
                ],
            }
        }
        stream_payload = {
            "schema": "admin_session_stream_v1",
            "events": [
                event("req-1", "sha256:p1", 1),
                event("req-1", "sha256:p1", 2),
                event("req-2", "sha256:p2", 1),
                event("req-2", "sha256:p2", 2),
            ],
        }

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-batch-stream",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                    "request_count": 2,
                }
            if path.startswith("/admin/session-stream"):
                return stream_payload
            return {"results": [final_row]}

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_product_generate(args)

        encoded = json.dumps(report, sort_keys=True)
        self.assertTrue(report["ok"], report)
        self.assertIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
        self.assertEqual(report["stream"]["event_count"], 4)
        self.assertTrue(report["stream"]["stream_generation_ready"])
        self.assertTrue(report["stream"]["progress"]["per_request_progress_complete"])
        self.assertEqual(
            [
                (entry["request_id"], entry["observed_token_counts"])
                for entry in report["stream"]["progress"]["per_request_progress"]
            ],
            [("req-1", [1, 2]), ("req-2", [1, 2])],
        )
        self.assertEqual(
            [(event["request_id"], event["generated_token_count"]) for event in report["stream"]["events"]],
            [("req-1", 1), ("req-1", 2), ("req-2", 1), ("req-2", 2)],
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn(
            "  stream_events: 4 source=admin-session-stream complete=True requests=2/2",
            rendered,
        )
        self.assertIn("  stream[1]: request=req-1 tokens=2/2 counts=[1, 2] complete=True missing=False", rendered)
        self.assertIn("  stream[2]: request=req-2 tokens=2/2 counts=[1, 2] complete=True missing=False", rendered)
        self.assertNotIn("first private prompt", encoded)
        self.assertNotIn("second private prompt", encoded)
        self.assertNotIn("raw one", encoded)
        self.assertNotIn("raw two", encoded)
        self.assertNotIn('"generated_token_ids":', encoded)
        self.assertNotIn("admin-secret", encoded)

    def test_product_generate_batch_stream_rejects_incomplete_prompt_progress(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--output-dir",
            str(output_dir),
            "--prompt-texts",
            "first private prompt,second private prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--stream",
            "--json",
        ])

        def event(request_id: str, prompt_hash: str, count: int) -> dict:
            return {
                "schema": "session_stream_event_v1",
                "session_id": "real-llm-session-batch-stream",
                "request_id": request_id,
                "prompt_hash": prompt_hash,
                "generated_token_count": count,
                "max_new_tokens": 2,
                "generation_step": count - 1,
                "generated_text_hash": f"sha256:{request_id}-{count}",
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            }

        final_row = {
            "validation": {
                "request_count": 2,
                "generated_token_count": 2,
                "max_new_tokens": 2,
                "generated_text_hash": "sha256:batch",
                "decoded_tokens_match": True,
                "inference_results": [
                    {
                        "request_id": "req-1",
                        "prompt_hash": "sha256:p1",
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:req-1-2",
                        "decoded_tokens_match": True,
                    },
                    {
                        "request_id": "req-2",
                        "prompt_hash": "sha256:p2",
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:req-2-2",
                        "decoded_tokens_match": True,
                    },
                ],
            }
        }
        stream_payload = {
            "schema": "admin_session_stream_v1",
            "events": [
                event("req-1", "sha256:p1", 1),
                event("req-1", "sha256:p1", 2),
                event("req-2", "sha256:p2", 1),
            ],
        }

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-batch-stream",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                    "request_count": 2,
                }
            if path.startswith("/admin/session-stream"):
                return stream_payload
            return {"results": [final_row]}

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_product_generate(args)

        encoded = json.dumps(report, sort_keys=True)
        self.assertTrue(report["ok"], report)
        self.assertNotIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
        self.assertFalse(report["stream"]["stream_generation_ready"])
        self.assertFalse(report["stream"]["progress"]["per_request_progress_complete"])
        self.assertEqual(report["stream"]["issue_summary"], "request[2]=req-2:1/2")
        self.assertEqual(
            report["operator_action"],
            "Generation completed, but stream progress is incomplete (request[2]=req-2:1/2); retry with --stream if you need live token evidence.",
        )
        self.assertEqual(report["stream"]["event_count"], 3)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn(
            "  stream_events: 3 source=admin-session-stream complete=False requests=2/2",
            rendered,
        )
        self.assertIn("  stream[1]: request=req-1 tokens=2/2 counts=[1, 2] complete=True missing=False", rendered)
        self.assertIn("  stream[2]: request=req-2 tokens=1/2 counts=[1] complete=False missing=False", rendered)
        self.assertIn("  stream_issue: request[2]=req-2:1/2", rendered)
        self.assertIn("  action: Generation completed, but stream progress is incomplete (request[2]=req-2:1/2); retry with --stream if you need live token evidence.", rendered)
        self.assertEqual(rendered.count("  action: "), 1)
        self.assertIn(f"  inspect_first: {output_dir / 'generate_summary.md'}", rendered)
        self.assertLess(rendered.index("  action: "), rendered.index("  diagnosis: "))
        self.assertLess(rendered.index("  inspect_first: "), rendered.index("  diagnosis: "))
        self.assertNotIn("first private prompt", encoded)
        self.assertNotIn("second private prompt", encoded)
        self.assertNotIn("first private prompt", rendered)
        self.assertNotIn("second private prompt", rendered)
        persisted = json.loads((output_dir / "generate_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["stream"]["issue_summary"], "request[2]=req-2:1/2")
        self.assertTrue(persisted["artifacts"]["generate_summary_markdown"]["present"])
        self.assertNotIn("first private prompt", json.dumps(persisted, sort_keys=True))
        self.assertNotIn("second private prompt", json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "generate_summary.md").read_text(encoding="utf-8")
        self.assertIn("- Stream issue: `request[2]=req-2:1/2`", markdown)
        self.assertIn("Generation completed, but stream progress is incomplete", markdown)
        self.assertNotIn("first private prompt", markdown)
        self.assertNotIn("second private prompt", markdown)

    def test_product_generate_batch_stream_prints_missing_prompt_progress(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-texts",
            "first private prompt,second private prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--stream",
            "--json",
        ])

        def event(request_id: str, prompt_hash: str, count: int) -> dict:
            return {
                "schema": "session_stream_event_v1",
                "session_id": "real-llm-session-batch-stream",
                "request_id": request_id,
                "prompt_hash": prompt_hash,
                "generated_token_count": count,
                "max_new_tokens": 2,
                "generation_step": count - 1,
                "generated_text_hash": f"sha256:{request_id}-{count}",
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            }

        final_row = {
            "validation": {
                "request_count": 2,
                "generated_token_count": 2,
                "max_new_tokens": 2,
                "generated_text_hash": "sha256:batch",
                "decoded_tokens_match": True,
                "inference_results": [
                    {
                        "request_id": "req-1",
                        "prompt_hash": "sha256:p1",
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:req-1-2",
                        "decoded_tokens_match": True,
                    },
                    {
                        "request_id": "req-2",
                        "prompt_hash": "sha256:p2",
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:req-2-2",
                        "decoded_tokens_match": True,
                    },
                ],
            }
        }
        stream_payload = {
            "schema": "admin_session_stream_v1",
            "events": [
                event("req-1", "sha256:p1", 1),
                event("req-1", "sha256:p1", 2),
            ],
        }

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-batch-stream",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                    "request_count": 2,
                }
            if path.startswith("/admin/session-stream"):
                return stream_payload
            return {"results": [final_row]}

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertFalse(report["stream"]["stream_generation_ready"])
        self.assertFalse(report["stream"]["progress"]["per_request_progress_complete"])
        self.assertEqual(
            report["stream"]["issue_summary"],
            "missing_requests=1/2 request[2]=missing",
        )
        self.assertEqual(
            report["operator_action"],
            "Generation completed, but stream progress is incomplete (missing_requests=1/2 request[2]=missing); retry with --stream if you need live token evidence.",
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn(
            "  stream_events: 2 source=admin-session-stream complete=False requests=1/2",
            rendered,
        )
        self.assertIn("  stream[1]: request=req-1 tokens=2/2 counts=[1, 2] complete=True missing=False", rendered)
        self.assertIn("  stream[2]: request=missing tokens=0/2 counts=[] complete=False missing=True", rendered)
        self.assertIn("  stream_issue: missing_requests=1/2 request[2]=missing", rendered)
        self.assertIn("  action: Generation completed, but stream progress is incomplete (missing_requests=1/2 request[2]=missing); retry with --stream if you need live token evidence.", rendered)
        self.assertNotIn("first private prompt", rendered)
        self.assertNotIn("second private prompt", rendered)

    def test_product_generate_live_stream_prints_safe_request_labels(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-texts",
            "first private prompt,second private prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "1",
            "--stream",
        ])

        def event(request_id: str, prompt_hash: str) -> dict:
            return {
                "schema": "session_stream_event_v1",
                "session_id": "real-llm-session-live-stream",
                "request_id": request_id,
                "prompt_hash": prompt_hash,
                "generated_token_count": 1,
                "max_new_tokens": 1,
                "generation_step": 0,
                "generated_text_hash": f"sha256:{request_id}",
                "generated_text": "must not leak",
                "generated_token_ids": [101],
            }

        final_row = {
            "validation": {
                "request_count": 2,
                "generated_token_count": 1,
                "max_new_tokens": 1,
                "generated_text_hash": "sha256:batch",
                "decoded_tokens_match": True,
                "inference_results": [
                    {
                        "request_id": "req-1",
                        "prompt_hash": "sha256:p1",
                        "generated_token_count": 1,
                        "max_new_tokens": 1,
                        "generated_text_hash": "sha256:req-1",
                        "decoded_tokens_match": True,
                    },
                    {
                        "request_id": "req-2",
                        "prompt_hash": "sha256:p2",
                        "generated_token_count": 1,
                        "max_new_tokens": 1,
                        "generated_text_hash": "sha256:req-2",
                        "decoded_tokens_match": True,
                    },
                ],
            }
        }

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-live-stream",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 1,
                    "backend": "hf_transformers_cpu",
                    "request_count": 2,
                }
            if path.startswith("/admin/session-stream"):
                return {
                    "schema": "admin_session_stream_v1",
                    "events": [event("req-1", "sha256:p1"), event("req-2", "sha256:p2")],
                }
            return {"results": [final_row]}

        stdout = io.StringIO()
        with patch.object(cli, "request_json_url", side_effect=fake_request), contextlib.redirect_stdout(stdout):
            report = cli.build_product_generate(args)

        rendered = stdout.getvalue()
        self.assertTrue(report["ok"], report)
        self.assertIn("stream request=req-1 1/1 hash=sha256:req-1", rendered)
        self.assertIn("stream request=req-2 1/1 hash=sha256:req-2", rendered)
        self.assertNotIn("first private prompt", rendered)
        self.assertNotIn("second private prompt", rendered)
        self.assertNotIn("must not leak", rendered)
        self.assertNotIn("generated_token_ids", rendered)

    def test_product_generate_batch_stream_ledger_fallback_expands_batch_rows(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-texts",
            "first private prompt,second private prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--stream",
        ])

        def batch_row(count: int) -> dict:
            return {
                "task_id": f"stage1-step{count - 1}",
                "session_id": "real-llm-session-batch-stream",
                "miner_id": "stage1-batch",
                "terminal_at": float(count),
                "validation": {
                    "session_id": "real-llm-session-batch-stream",
                    "stage_id": 1,
                    "generation_step": count - 1,
                    "generated_token_count": count,
                    "max_new_tokens": 2,
                    "generated_text_hash": f"sha256:batch-{count}",
                    "decoded_tokens_match": True,
                    "inference_results": [
                        {
                            "request_id": "req-1",
                            "prompt_hash": "sha256:p1",
                            "generation_step": count - 1,
                            "generated_token_count": count,
                            "max_new_tokens": 2,
                            "generated_text_hash": f"sha256:req-1-{count}",
                            "generated_text": f" raw one {count}",
                            "generated_token_ids": list(range(count)),
                            "decoded_tokens_match": True,
                        },
                        {
                            "request_id": "req-2",
                            "prompt_hash": "sha256:p2",
                            "generation_step": count - 1,
                            "generated_token_count": count,
                            "max_new_tokens": 2,
                            "generated_text_hash": f"sha256:req-2-{count}",
                            "generated_text": f" raw two {count}",
                            "generated_token_ids": list(range(count)),
                            "decoded_tokens_match": True,
                        },
                    ],
                },
            }

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-batch-stream",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                    "request_count": 2,
                }
            if path.startswith("/admin/session-stream"):
                raise cli.HTTPError(path, 404, "not found", {}, None)
            return {"results": [batch_row(2), batch_row(1)]}

        stdout = io.StringIO()
        with patch.object(cli, "request_json_url", side_effect=fake_request), contextlib.redirect_stdout(stdout):
            report = cli.build_product_generate(args)

        rendered = stdout.getvalue()
        stream_encoded = json.dumps(report["stream"], sort_keys=True)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["stream"]["source"], "admin-results-ledger-fallback")
        self.assertFalse(report["stream"]["endpoint_ready"])
        self.assertTrue(report["stream"]["stream_generation_ready"])
        self.assertEqual(
            [(event["request_id"], event["generated_token_count"]) for event in report["stream"]["events"]],
            [("req-1", 1), ("req-2", 1), ("req-1", 2), ("req-2", 2)],
        )
        self.assertTrue(report["stream"]["progress"]["per_request_progress_complete"])
        self.assertIn("stream request=req-1 1/2 hash=sha256:req-1-1", rendered)
        self.assertIn("stream request=req-2 1/2 hash=sha256:req-2-1", rendered)
        self.assertIn("stream request=req-1 2/2 hash=sha256:req-1-2", rendered)
        self.assertIn("stream request=req-2 2/2 hash=sha256:req-2-2", rendered)
        self.assertNotIn(" raw one", rendered)
        self.assertNotIn(" raw two", rendered)
        self.assertNotIn("raw one", stream_encoded)
        self.assertNotIn("raw two", stream_encoded)
        self.assertNotIn('"generated_token_ids":', stream_encoded)

    def test_product_generate_stream_orders_descending_ledger_progress(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-text",
            "CrowdTensor prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "3",
            "--stream",
            "--json",
        ])

        def row(count: int) -> dict:
            return {
                "event_index": count,
                "task_id": f"stage1-step{count - 1}",
                "session_id": "real-llm-session-test",
                "miner_id": f"stage1-{count}",
                "validation": {
                    "session_id": "real-llm-session-test",
                    "stage_id": 1,
                    "generation_step": count - 1,
                    "generated_token_count": count,
                    "max_new_tokens": 3,
                    "generated_text_hash": f"sha256:step{count - 1}",
                    "generated_text": f" raw step {count}",
                    "generated_token_ids": list(range(count)),
                    "decoded_tokens_match": True,
                },
            }

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-test",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 3,
                    "backend": "hf_transformers_cpu",
                }
            if path.startswith("/admin/session-stream"):
                raise cli.HTTPError(path, 404, "not found", {}, None)
            return {"results": [row(3), row(2), row(1)]}

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["stream"]["source"], "admin-results-ledger-fallback")
        self.assertFalse(report["stream"]["endpoint_ready"])
        self.assertEqual(
            [event["generated_token_count"] for event in report["stream"]["events"]],
            [1, 2, 3],
        )
        self.assertEqual(report["stream"]["event_count"], 3)
        self.assertTrue(report["stream"]["progress"]["monotonic_progress"])

    def test_product_generate_stream_requires_monotonic_progress_for_stream_ready(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-text",
            "CrowdTensor prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "3",
            "--stream",
            "--json",
        ])

        final_row = {
            "task_id": "stage1-step2",
            "session_id": "real-llm-session-test",
            "miner_id": "stage1-c",
            "validation": {
                "session_id": "real-llm-session-test",
                "stage_id": 1,
                "generation_step": 2,
                "generated_token_count": 3,
                "max_new_tokens": 3,
                "generated_text_hash": "sha256:step2",
                "generated_text": " raw final text",
                "generated_token_ids": [101, 102, 103],
                "decoded_tokens_match": True,
            },
        }

        def event(count: int) -> dict:
            return {
                "schema": "session_stream_event_v1",
                "session_id": "real-llm-session-test",
                "generated_token_count": count,
                "max_new_tokens": 3,
                "generation_step": count - 1,
                "generated_text_hash": f"sha256:step{count - 1}",
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            }

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-test",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 3,
                    "backend": "hf_transformers_cpu",
                }
            if path.startswith("/admin/session-stream"):
                return {"schema": "admin_session_stream_v1", "events": [event(2), event(1), event(3)]}
            return {"results": [final_row]}

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertNotIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
        self.assertTrue(report["stream"]["progress"]["stream_progress_complete"])
        self.assertFalse(report["stream"]["progress"]["monotonic_progress"])
        self.assertEqual(report["stream"]["progress"]["observed_token_counts"], [2, 1, 3])

    def test_product_generate_stream_requires_complete_progress_for_stream_ready(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-text",
            "CrowdTensor prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "3",
            "--stream",
            "--json",
        ])

        final_row = {
            "task_id": "stage1-step2",
            "session_id": "real-llm-session-test",
            "miner_id": "stage1-c",
            "validation": {
                "session_id": "real-llm-session-test",
                "stage_id": 1,
                "generation_step": 2,
                "generated_token_count": 3,
                "max_new_tokens": 3,
                "generated_text_hash": "sha256:step2",
                "generated_text": " raw final text",
                "generated_token_ids": [101, 102, 103],
                "decoded_tokens_match": True,
            },
        }

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-test",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 3,
                    "backend": "hf_transformers_cpu",
                }
            if path.startswith("/admin/session-stream"):
                return {
                    "schema": "admin_session_stream_v1",
                    "events": [
                        {
                            "schema": "session_stream_event_v1",
                            "session_id": "real-llm-session-test",
                            "generated_token_count": 1,
                            "max_new_tokens": 3,
                            "generation_step": 0,
                            "generated_text_hash": "sha256:step0",
                            "raw_generated_text_public": False,
                            "generated_token_ids_public": False,
                        }
                    ],
                }
            return {"results": [final_row]}

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_product_generate(args)

        self.assertTrue(report["ok"], report)
        self.assertNotIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
        self.assertEqual(report["stream"]["event_count"], 1)
        self.assertFalse(report["stream"]["progress"]["stream_progress_complete"])

    def test_product_generate_include_output_only_in_human_mode(self) -> None:
        base_argv = [
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-text",
            "CrowdTensor prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
            "--include-output",
        ]

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, path, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-test",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                }
            return {
                "results": [
                    {
                        "validation": {
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:generated",
                            "generated_text": " readable beta text",
                            "decoded_tokens_match": True,
                        }
                    }
                ]
            }

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            human_report = cli.build_product_generate(cli.parse_args(base_argv))
        with patch.object(cli, "request_json_url", side_effect=fake_request):
            json_report = cli.build_product_generate(cli.parse_args([*base_argv, "--json"]))

        self.assertTrue(human_report["ok"], human_report)
        self.assertTrue(human_report["output_request"]["include_output"])
        self.assertEqual(human_report["local_output"]["generated_text"], " readable beta text")
        self.assertEqual(human_report["result"]["display"], "local-private")
        self.assertEqual(human_report["output_display"]["terminal_display"], "local-private")
        self.assertTrue(human_report["output_display"]["terminal_text_available"])
        self.assertEqual(human_report["output_display"]["saved_artifact_display"], "hash-only")
        self.assertEqual(human_report["output_display"]["json_stdout_display"], "hash-only-json")
        self.assertTrue(human_report["output_display"]["include_output_requested"])
        self.assertFalse(human_report["output_display"]["raw_generated_text_public"])
        self.assertTrue(human_report["output_display"]["public_artifact_safe"])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(human_report)
        rendered = stdout.getvalue()
        self.assertIn(
            "  output_display: terminal=local-private terminal_text=True saved=hash-only json_stdout=hash-only-json include_output=True raw_public=False public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  output_request: include_output=True raw_generated_text_public=False public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  local_output: available=True display_only=True public_artifact_safe=False count=1 source=coordinator-validation",
            rendered,
        )
        self.assertIn("  answer:  readable beta text", rendered)
        self.assertIn(
            "  answer_scope: state=terminal-visible terminal_only=True visible_in_terminal=True saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            rendered,
        )
        self.assertLess(rendered.index("  answer:  readable beta text"), rendered.index("  local_output: "))
        self.assertLess(rendered.index("  answer_scope: "), rendered.index("  local_output: "))
        self.assertLess(rendered.index("  answer:  readable beta text"), rendered.index("  trace: "))
        self.assertIn(
            "crowdtensor generate --max-new-tokens 2 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --include-output",
            [item["command_line"] for item in human_report["next_commands"]],
        )
        self.assertTrue(json_report["ok"], json_report)
        self.assertTrue(json_report["output_request"]["include_output"])
        self.assertEqual(json_report["result"]["display"], "hash-only-json")
        self.assertTrue(json_report["result"]["public_artifact_safe"])
        self.assertEqual(json_report["output_display"]["terminal_display"], "hash-only-json")
        self.assertFalse(json_report["output_display"]["terminal_text_available"])
        self.assertEqual(json_report["output_display"]["saved_artifact_display"], "hash-only")
        self.assertEqual(json_report["output_display"]["json_stdout_display"], "hash-only-json")
        self.assertTrue(json_report["output_display"]["include_output_requested"])
        self.assertFalse(json_report["output_display"]["raw_generated_text_public"])
        self.assertFalse(json_report["answer_scope"]["visible_in_terminal"])
        self.assertFalse(json_report["answer_scope"]["terminal_only"])
        self.assertEqual(json_report["answer_scope"]["scope_state"], "json-suppressed")
        self.assertEqual(json_report["answer_scope"]["summary"], cli.SAVED_ANSWER_SCOPE_TEXT)
        self.assertEqual(json_report["local_output"]["output_count"], 1)
        self.assertFalse(json_report["local_output"]["available"])
        self.assertTrue(json_report["local_output"]["public_artifact_safe"])
        self.assertEqual(
            json_report["local_output_note"],
            "Raw generated text is suppressed in JSON/public output; rerun without --json for local display.",
        )
        self.assertIn(
            "crowdtensor generate --max-new-tokens 2 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --include-output",
            [item["command_line"] for item in json_report["next_commands"]],
        )
        self.assertNotIn("readable beta text", json.dumps(json_report, sort_keys=True))

    def test_product_generate_human_mode_shows_output_by_default(self) -> None:
        base_argv = [
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-text",
            "CrowdTensor prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
        ]

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, path, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-test",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                }
            return {
                "results": [
                    {
                        "validation": {
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:generated",
                            "generated_text": " default human output",
                            "decoded_tokens_match": True,
                        }
                    }
                ]
            }

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            human_report = cli.build_product_generate(cli.parse_args(base_argv))
        with patch.object(cli, "request_json_url", side_effect=fake_request):
            json_report = cli.build_product_generate(cli.parse_args([*base_argv, "--json"]))

        self.assertTrue(human_report["ok"], human_report)
        self.assertFalse(human_report["output_request"]["include_output"])
        self.assertEqual(human_report["local_output"]["generated_text"], " default human output")
        self.assertEqual(human_report["result"]["display"], "local-private")
        self.assertEqual(human_report["output_display"]["terminal_display"], "local-private")
        self.assertTrue(human_report["output_display"]["terminal_text_available"])
        self.assertFalse(human_report["output_display"]["include_output_requested"])
        self.assertNotIn("--include-output", json.dumps(human_report["next_commands"], sort_keys=True))
        self.assertTrue(json_report["ok"], json_report)
        self.assertFalse(json_report["output_request"]["include_output"])
        self.assertEqual(json_report["result"]["display"], "hash-only-json")
        self.assertTrue(json_report["result"]["public_artifact_safe"])
        self.assertEqual(json_report["output_display"]["terminal_display"], "hash-only-json")
        self.assertFalse(json_report["output_display"]["terminal_text_available"])
        self.assertFalse(json_report["output_display"]["include_output_requested"])
        self.assertEqual(json_report["local_output"]["output_count"], 1)
        self.assertFalse(json_report["local_output"]["available"])
        self.assertTrue(json_report["local_output"]["public_artifact_safe"])
        self.assertEqual(json_report["answer_scope"]["scope_state"], "json-suppressed")
        self.assertNotIn("default human output", json.dumps(json_report, sort_keys=True))

    def test_product_generate_human_batch_outputs_are_display_only(self) -> None:
        base_argv = [
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-texts",
            "first private prompt,second private prompt",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
        ]

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del base_url, path, payload, admin_token, timeout
            if method == "POST":
                return {
                    "schema": "real_llm_sharded_session_v1",
                    "session_id": "real-llm-session-batch",
                    "workload_type": "real_llm_sharded_infer",
                    "max_new_tokens": 2,
                    "backend": "hf_transformers_cpu",
                    "request_count": 2,
                }
            return {
                "results": [
                    {
                        "validation": {
                            "request_count": 2,
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:batch",
                            "decoded_tokens_match": True,
                            "inference_results": [
                                {
                                    "request_id": "req-1",
                                    "prompt_hash": "sha256:p1",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g1",
                                    "generated_text": " raw one",
                                    "generated_token_ids": [1, 2],
                                    "decoded_tokens_match": True,
                                },
                                {
                                    "request_id": "req-2",
                                    "prompt_hash": "sha256:p2",
                                    "generated_token_count": 2,
                                    "max_new_tokens": 2,
                                    "generated_text_hash": "sha256:g2",
                                    "generated_text": " raw two",
                                    "generated_token_ids": [3, 4],
                                    "decoded_tokens_match": True,
                                },
                            ],
                        }
                    }
                ]
            }

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            human_report = cli.build_product_generate(cli.parse_args(base_argv))
        with patch.object(cli, "request_json_url", side_effect=fake_request):
            json_report = cli.build_product_generate(cli.parse_args([*base_argv, "--json"]))

        self.assertTrue(human_report["ok"], human_report)
        self.assertEqual(human_report["local_output"]["generated_text"], " raw one")
        self.assertEqual([row["generated_text"] for row in human_report["local_output"]["outputs"]], [" raw one", " raw two"])
        self.assertEqual(human_report["result"]["display"], "local-private")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(human_report)
        rendered = stdout.getvalue()
        self.assertIn("  batch: requests=2 observed=2 ready=True", rendered)
        self.assertNotIn("  answer:  raw one", rendered)
        self.assertIn("  answer[1]:  raw one", rendered)
        self.assertIn("  answer[2]:  raw two", rendered)
        self.assertTrue(json_report["ok"], json_report)
        self.assertEqual(json_report["result"]["display"], "hash-only-json")
        self.assertEqual(json_report["result"]["output_count"], 2)
        self.assertTrue(json_report["result"]["public_artifact_safe"])
        self.assertEqual(json_report["local_output"]["output_count"], 2)
        self.assertFalse(json_report["local_output"]["available"])
        self.assertTrue(json_report["local_output"]["public_artifact_safe"])
        self.assertEqual(json_report["answer_scope"]["scope_state"], "json-suppressed")
        encoded = json.dumps(json_report, sort_keys=True)
        self.assertNotIn("raw one", encoded)
        self.assertNotIn("raw two", encoded)
        self.assertNotIn('"generated_token_ids":', encoded)

    def test_public_real_llm_swarm_beta_cli_wraps_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "package",
            "--output-dir",
            str(output_dir),
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_real_llm_swarm_beta_pack.py", command[1])
            self.assertIn("--usable-report", command)
            self.assertIn("16tok-kv-cache", command[command.index("--usable-report") + 1])
            self.assertIn("--public-swarm-v2-report", command)
            self.assertIn("public-swarm-inference-v2", command[command.index("--public-swarm-v2-report") + 1])
            self.assertIn("--external-report", command)
            self.assertIn("16tok-gpu-summary", command[command.index("--external-report") + 1])
            self.assertIn("--p2p-report", command)
            self.assertIn("16tok-batch-stream", command[command.index("--p2p-report") + 1])
            self.assertIn("--p2p-runtime-smoke-report", command)
            self.assertIn("kaggle-runtime-smoke", command[command.index("--p2p-runtime-smoke-report") + 1])
            self.assertIn("--p2p-external-report", command)
            self.assertIn("fresh-real-p2p-kaggle-16tok", command[command.index("--p2p-external-report") + 1])
            self.assertIn("--p2p-requeue-report", command)
            self.assertIn("petals-p2p-candidate-live-stage0", command[command.index("--p2p-requeue-report") + 1])
            self.assertIn("--p2p-batch-stream-report", command)
            self.assertIn("public-swarm-v2-batch-stream-16tok", command[command.index("--p2p-batch-stream-report") + 1])
            self.assertIn("--p2p-libp2p-port", command)
            self.assertIn("--public-swarm-v2-real-p2p-port", command)
            self.assertEqual(command[command.index("--public-swarm-v2-real-p2p-port") + 1], "9890")
            self.assertIn("--public-swarm-v2-real-p2p-coordinator-port", command)
            self.assertEqual(command[command.index("--public-swarm-v2-real-p2p-coordinator-port") + 1], "9891")
            self.assertIn("--public-swarm-v2-real-p2p-libp2p-port", command)
            self.assertEqual(command[command.index("--public-swarm-v2-real-p2p-libp2p-port") + 1], "0")
            self.assertIn("--public-swarm-v2-real-p2p-discovery-backend", command)
            self.assertEqual(command[command.index("--public-swarm-v2-real-p2p-discovery-backend") + 1], "http-provider-store")
            return completed({
                "schema": "public_real_llm_swarm_beta_v1",
                "ok": True,
                "mode": "package",
                "diagnosis_codes": ["public_real_llm_swarm_beta_package_ready"],
            })

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "public_real_llm_swarm_beta_cli_v1")
        self.assertTrue(any("package" in command for command in calls))

    def test_public_real_llm_swarm_beta_cli_forwards_usable_report(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--usable-report",
            "/tmp/usable-kv.json",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_real_llm_swarm_beta_pack.py", command[1])
            self.assertIn("--usable-report", command)
            self.assertEqual(command[command.index("--usable-report") + 1], "/tmp/usable-kv.json")
            return completed({
                "schema": "public_real_llm_swarm_beta_v1",
                "ok": True,
                "mode": "evidence-import",
                "diagnosis_codes": ["public_real_llm_swarm_beta_ready"],
            })

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "public_real_llm_swarm_beta_cli_v1")

    def test_public_real_llm_swarm_beta_cli_forwards_p2p_candidate_sources(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "release",
            "--output-dir",
            str(output_dir),
            "--p2p-runtime-smoke-report",
            "/tmp/runtime.json",
            "--p2p-external-report",
            "/tmp/external-p2p.json",
            "--p2p-requeue-report",
            "/tmp/requeue.json",
            "--p2p-batch-stream-report",
            "/tmp/batch-stream.json",
            "--p2p-libp2p-port",
            "10999",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_real_llm_swarm_beta_pack.py", command[1])
            self.assertEqual(command[command.index("--p2p-runtime-smoke-report") + 1], "/tmp/runtime.json")
            self.assertEqual(command[command.index("--p2p-external-report") + 1], "/tmp/external-p2p.json")
            self.assertEqual(command[command.index("--p2p-requeue-report") + 1], "/tmp/requeue.json")
            self.assertEqual(command[command.index("--p2p-batch-stream-report") + 1], "/tmp/batch-stream.json")
            self.assertEqual(command[command.index("--p2p-libp2p-port") + 1], "10999")
            return completed({
                "schema": "public_real_llm_swarm_beta_v1",
                "ok": True,
                "mode": "release",
                "diagnosis_codes": ["public_real_llm_swarm_beta_ready"],
            })

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "public_real_llm_swarm_beta_cli_v1")

    def test_public_real_llm_swarm_beta_cli_forwards_public_swarm_v2_report(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--public-swarm-v2-report",
            "/tmp/public-swarm-v2.json",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_real_llm_swarm_beta_pack.py", command[1])
            self.assertIn("--public-swarm-v2-report", command)
            self.assertEqual(command[command.index("--public-swarm-v2-report") + 1], "/tmp/public-swarm-v2.json")
            return completed({
                "schema": "public_real_llm_swarm_beta_v1",
                "ok": True,
                "mode": "evidence-import",
                "diagnosis_codes": ["public_real_llm_swarm_beta_ready"],
            })

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "public_real_llm_swarm_beta_cli_v1")

    def test_public_real_llm_swarm_beta_cli_forwards_bounded_prompt_batch(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "local-smoke",
            "--output-dir",
            str(output_dir),
            "--prompt-texts",
            "first prompt,second prompt",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_real_llm_swarm_beta_pack.py", command[1])
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
            self.assertNotIn("--prompt-text", command)
            return completed({
                "schema": "public_real_llm_swarm_beta_v1",
                "ok": True,
                "mode": "local-smoke",
                "diagnosis_codes": ["public_real_llm_swarm_beta_local_smoke_ready", "public_swarm_generate_batch_ready"],
            })

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "public_real_llm_swarm_beta_cli_v1")

    def test_public_real_llm_swarm_beta_cli_forwards_prompt_texts_file(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_file = output_dir / "prompts.txt"
        prompts = ["first prompt, with comma", "second prompt"]
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "local-smoke",
            "--output-dir",
            str(output_dir),
            "--prompt-texts-file",
            str(prompt_file),
            "--json",
        ])
        self.assertEqual(args.prompt_texts_list, prompts)

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_real_llm_swarm_beta_pack.py", command[1])
            self.assertIn("--prompt-texts-file", command)
            self.assertEqual(command[command.index("--prompt-texts-file") + 1], str(prompt_file))
            self.assertNotIn("--prompt-texts", command)
            self.assertNotIn("--prompt-text", command)
            command_text = " ".join(command)
            for prompt in prompts:
                self.assertNotIn(prompt, command_text)
            return completed({
                "schema": "public_real_llm_swarm_beta_v1",
                "ok": True,
                "mode": "local-smoke",
                "diagnosis_codes": ["public_real_llm_swarm_beta_local_smoke_ready", "public_swarm_generate_batch_ready"],
            })

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "public_real_llm_swarm_beta_cli_v1")

    def test_public_real_llm_swarm_beta_cli_forwards_stream_generation(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "local-smoke",
            "--output-dir",
            str(output_dir),
            "--stream-generation",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_real_llm_swarm_beta_pack.py", command[1])
            self.assertIn("--stream-generation", command)
            return completed({
                "schema": "public_real_llm_swarm_beta_v1",
                "ok": True,
                "mode": "local-smoke",
                "diagnosis_codes": ["public_real_llm_swarm_beta_local_smoke_ready", "public_swarm_generate_stream_ready"],
            })

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "public_real_llm_swarm_beta_cli_v1")

    def test_public_real_llm_swarm_beta_cli_forwards_local_model_variant(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "local-model-variant",
            "--output-dir",
            str(output_dir),
            "--hf-model-id",
            "distilgpt2",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_real_llm_swarm_beta_pack.py", command[1])
            self.assertEqual(command[2], "local-model-variant")
            self.assertEqual(command[command.index("--hf-model-id") + 1], "distilgpt2")
            return completed({
                "schema": "public_real_llm_swarm_beta_v1",
                "ok": True,
                "mode": "local-model-variant",
                "diagnosis_codes": ["public_real_llm_swarm_beta_local_model_variant_ready"],
            })

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "public_real_llm_swarm_beta_cli_v1")

    def test_public_real_llm_swarm_beta_cli_wraps_check(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "check",
            "--output-dir",
            str(output_dir),
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_real_llm_swarm_beta_check.py", command[1])
            self.assertIn("--mode", command)
            self.assertEqual(command[command.index("--mode") + 1], "release")
            self.assertIn("--output-dir", command)
            self.assertEqual(command[command.index("--output-dir") + 1], str(output_dir.resolve()))
            self.assertIn("--max-new-tokens", command)
            self.assertEqual(command[command.index("--max-new-tokens") + 1], "16")
            self.assertIn("--hf-model-id", command)
            self.assertEqual(command[command.index("--hf-model-id") + 1], "sshleifer/tiny-gpt2")
            self.assertIn("--json", command)
            return completed({
                "schema": "public_real_llm_swarm_beta_check_v1",
                "ok": True,
                "mode": "release",
                "max_new_tokens": 16,
                "output_dir": str(output_dir),
                "errors": [],
                "diagnosis_codes": ["public_real_llm_swarm_beta_check_ready"],
                "artifact_summary": {
                    "inspect_first": str(output_dir / "public_real_llm_swarm_beta.md"),
                    "machine_readable": str(output_dir / "public_real_llm_swarm_beta.json"),
                    "support_bundle": str(output_dir / "support_bundle.json"),
                    "check_json": str(output_dir / "public_real_llm_swarm_beta_check.json"),
                    "public_artifact_safe": True,
                },
                "review_summary": {
                    "state": "ready",
                    "ready": True,
                    "next_step": "review_checked_artifacts",
                    "inspect_first": str(output_dir / "public_real_llm_swarm_beta.md"),
                    "check_json": str(output_dir / "public_real_llm_swarm_beta_check.json"),
                    "error_count": 0,
                    "public_artifact_safe": True,
                },
                "operator_action": "Open inspect_first for the checked Markdown, support_bundle for diagnostics, and check_json for the validation record.",
                "output_request": {
                    "include_output": False,
                    "raw_prompt_public": False,
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                    "public_artifact_safe": True,
                },
                "prompt_scope": {
                    "source": "prompt-text",
                    "prompt_count": 1,
                    "inline_prompt_text": True,
                    "terminal_next_commands_local_private": True,
                    "terminal_local_paths": False,
                    "saved_artifacts_prompt_placeholders": True,
                    "prompt_file_path_public": False,
                    "raw_prompt_public": False,
                    "public_artifact_safe": True,
                    "summary": "Validation check keeps prompt text out of public artifacts.",
                },
                "answer_scope": {
                    "scope_state": "no-local-answer",
                    "visible_in_terminal": False,
                    "saved_json_display": "validation-only",
                    "public_artifact_safe": True,
                    "summary": "Validation-only check output; no local answer transcript.",
                },
                "shareable_summary": {
                    "saved_artifacts_public_safe": True,
                    "raw_prompt_public": False,
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                    "answer_scope_state": "no-local-answer",
                },
                "artifacts": {
                    "public_real_llm_swarm_beta_json": str(output_dir / "public_real_llm_swarm_beta.json"),
                },
            })

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)

        self.assertEqual(len(calls), 1)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "public_real_llm_swarm_beta_check_v1")
        self.assertEqual(report["cli_schema"], "public_real_llm_swarm_beta_cli_v1")
        self.assertEqual(report["cli_mode"], "check")
        self.assertEqual(report["mode"], "release")
        self.assertEqual(report["review_summary"]["next_step"], "review_checked_artifacts")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_public_real_llm_swarm_beta_check(report)
        rendered = stdout.getvalue()
        self.assertIn("CrowdTensor Public Real-LLM Swarm Beta Check", rendered)
        self.assertIn("  cli_mode: check", rendered)
        self.assertIn("  check_source: unknown", rendered)
        self.assertIn("  review: state=ready next=review_checked_artifacts", rendered)
        self.assertIn("  artifacts: inspect=", rendered)
        self.assertIn("  prompt_scope: source=prompt-text count=1 inline_prompt_text=True", rendered)
        self.assertIn("  prompt_scope_note: Validation check keeps prompt text out of public artifacts.", rendered)
        self.assertIn("  output_request: include_output=False", rendered)
        self.assertIn("raw_prompt_public=False", rendered)
        self.assertIn("raw_generated_text_public=False", rendered)
        self.assertIn("generated_token_ids_public=False", rendered)
        self.assertIn("public_artifact_safe=True", rendered)
        self.assertIn("  answer_scope: state=no-local-answer saved_json=validation-only public_artifact_safe=True", rendered)
        self.assertIn("  answer_scope_note: Validation-only check output; no local answer transcript.", rendered)
        self.assertIn("  operator_action:", rendered)
        self.assertIn("  artifact public_real_llm_swarm_beta_json:", rendered)
        self.assertNotIn("  model: None", rendered)

    def test_public_real_llm_swarm_beta_check_uses_local_model_variant_for_custom_model(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "check",
            "--output-dir",
            str(output_dir),
            "--hf-model-id",
            "distilgpt2",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_real_llm_swarm_beta_check.py", command[1])
            self.assertEqual(command[command.index("--mode") + 1], "local-model-variant")
            self.assertEqual(command[command.index("--hf-model-id") + 1], "distilgpt2")
            return completed({
                "schema": "public_real_llm_swarm_beta_check_v1",
                "ok": True,
                "mode": "local-model-variant",
                "max_new_tokens": 16,
                "diagnosis_codes": ["public_real_llm_swarm_beta_check_ready"],
            })

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_mode"], "check")
        self.assertEqual(report["mode"], "local-model-variant")

    def test_public_real_llm_swarm_beta_check_forwards_beta_report(self) -> None:
        output_dir = Path(self._tmp_dir())
        beta_report = output_dir / "public_real_llm_swarm_beta.json"
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "check",
            "--beta-report",
            str(beta_report),
            "--output-dir",
            str(output_dir / "check"),
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_real_llm_swarm_beta_check.py", command[1])
            self.assertIn("--beta-report", command)
            self.assertEqual(command[command.index("--beta-report") + 1], str(beta_report))
            return completed({
                "schema": "public_real_llm_swarm_beta_check_v1",
                "ok": True,
                "mode": "release",
                "check_source": "beta-report",
                "checked_beta_report": str(beta_report.resolve()),
                "diagnosis_codes": ["public_real_llm_swarm_beta_check_ready"],
            })

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_mode"], "check")
        self.assertEqual(report["check_source"], "beta-report")
        self.assertEqual(report["checked_beta_report"], str(beta_report.resolve()))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_public_real_llm_swarm_beta_check(report)
        rendered = stdout.getvalue()
        self.assertIn("  check_source: beta-report", rendered)
        self.assertIn(f"  checked_beta_report: {beta_report.resolve()}", rendered)

    def test_public_real_llm_swarm_beta_check_failure_includes_review_guidance(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "check",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_real_llm_swarm_beta_check.py", command[1])
            return subprocess.CompletedProcess(command, 1, stdout="check failed before json\n", stderr="validation failed\n")

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["schema"], "public_real_llm_swarm_beta_check_v1")
        self.assertEqual(report["cli_schema"], "public_real_llm_swarm_beta_cli_v1")
        self.assertEqual(report["cli_mode"], "check")
        self.assertEqual(report["review_summary"]["state"], "blocked")
        self.assertEqual(report["review_summary"]["next_step"], "review_diagnostics")
        self.assertEqual(report["review_summary"]["error_count"], 1)
        self.assertEqual(report["review_summary"]["recommended_check_command"], report["recommended_check_command"])
        self.assertIn("public real LLM swarm beta check command returned no JSON report", report["errors"])
        self.assertIn("public-real-llm-swarm-beta check", report["operator_action"])
        self.assertIn(report["recommended_check_command"]["command_line"], report["operator_action"])
        self.assertEqual(
            shlex.split(report["recommended_check_command"]["command_line"]),
            [
                "crowdtensor",
                "public-real-llm-swarm-beta",
                "check",
                "--output-dir",
                str(output_dir.resolve()),
                "--max-new-tokens",
                "16",
                "--json",
            ],
        )
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertFalse(report["answer_scope"]["visible_in_terminal"])
        self.assertFalse(report["shareable_summary"]["raw_generated_text_public"])
        self.assertIn("validation failed", report["step"]["stderr_tail"])
        self.assertNotIn("validation failed", json.dumps(report["review_summary"], sort_keys=True))
        self.assertNotIn("validation failed", json.dumps(report["operator_action"], sort_keys=True))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_public_real_llm_swarm_beta_check(report)
        rendered = stdout.getvalue()
        self.assertIn("CrowdTensor Public Real-LLM Swarm Beta Check", rendered)
        self.assertIn("  review: state=blocked next=review_diagnostics", rendered)
        self.assertIn("  recommended_check: crowdtensor public-real-llm-swarm-beta check", rendered)
        self.assertIn("  errors:", rendered)
        self.assertIn("    - public real LLM swarm beta check command returned no JSON report", rendered)
        self.assertNotIn("  model: None", rendered)

    def test_public_real_llm_swarm_beta_check_failure_preserves_beta_report_context(self) -> None:
        output_dir = Path(self._tmp_dir())
        beta_report = output_dir / "beta" / "public_real_llm_swarm_beta.json"
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "check",
            "--beta-report",
            str(beta_report),
            "--output-dir",
            str(output_dir / "check"),
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("--beta-report", command)
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="check crashed\n")

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["check_source"], "beta-report")
        self.assertEqual(report["checked_beta_report"], str(beta_report.resolve()))
        self.assertEqual(report["artifact_summary"]["machine_readable"], str(beta_report.resolve()))
        self.assertEqual(report["review_summary"]["recommended_check_command"], report["recommended_check_command"])
        self.assertEqual(report["recommended_check_command"]["check_source"], "beta-report")
        self.assertIn("--beta-report", report["operator_action"])
        self.assertIn(str(beta_report.resolve()), report["operator_action"])
        self.assertEqual(
            shlex.split(report["recommended_check_command"]["command_line"]),
            [
                "crowdtensor",
                "public-real-llm-swarm-beta",
                "check",
                "--beta-report",
                str(beta_report.resolve()),
                "--output-dir",
                str((output_dir / "check").resolve()),
                "--max-new-tokens",
                "16",
                "--json",
            ],
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_public_real_llm_swarm_beta_check(report)
        rendered = stdout.getvalue()
        self.assertIn("  check_source: beta-report", rendered)
        self.assertIn("  recommended_check: crowdtensor public-real-llm-swarm-beta check", rendered)
        self.assertIn(f"  checked_beta_report: {beta_report.resolve()}", rendered)
        self.assertIn("public_real_llm_swarm_beta.json", rendered)

    def test_public_real_llm_swarm_beta_check_failure_quotes_context_and_model(self) -> None:
        output_dir = Path(self._tmp_dir()) / "check output"
        beta_report = Path(self._tmp_dir()) / "beta source" / "public_real_llm_swarm_beta.json"
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "check",
            "--hf-model-id",
            "distilgpt2",
            "--beta-report",
            str(beta_report),
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "24",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("--hf-model-id", command)
            return subprocess.CompletedProcess(command, 1, stdout="not json\n", stderr="check crashed\n")

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)
        command_line = report["recommended_check_command"]["command_line"]

        self.assertFalse(report["ok"], report)
        self.assertIn("'", command_line)
        self.assertIn(command_line, report["operator_action"])
        self.assertEqual(
            shlex.split(command_line),
            [
                "crowdtensor",
                "public-real-llm-swarm-beta",
                "check",
                "--hf-model-id",
                "distilgpt2",
                "--beta-report",
                str(beta_report.resolve()),
                "--output-dir",
                str(output_dir.resolve()),
                "--max-new-tokens",
                "24",
                "--json",
            ],
        )

    def test_public_real_llm_swarm_beta_cli_failure_includes_review_guidance(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "release",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_real_llm_swarm_beta_pack.py", command[1])
            return subprocess.CompletedProcess(command, 1, stdout="pack failed before json\n", stderr="runtime failed\n")

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["cli_schema"], "public_real_llm_swarm_beta_cli_v1")
        self.assertEqual(report["review_summary"]["state"], "blocked")
        self.assertEqual(report["review_summary"]["next_step"], "review_diagnostics")
        self.assertEqual(report["review_summary"]["inspect_first"], "public_real_llm_swarm_beta.md")
        self.assertEqual(report["review_summary"]["not_completed_count"], 1)
        self.assertEqual(report["review_summary"]["recommended_next_command"], report["recommended_next_command"])
        self.assertEqual(report["artifact_summary"]["support_bundle"], "support_bundle.json")
        self.assertTrue(report["artifact_summary"]["public_artifact_safe"])
        self.assertIn("public real LLM swarm beta pack command returned no JSON report", report["not_completed"])
        self.assertIn("Inspect the CLI step payload", report["operator_action"][0])
        self.assertIn(report["recommended_next_command"]["command_line"], report["operator_action"][0])
        self.assertEqual(report["recommended_next_command"]["reason"], "rerun_public_real_llm_beta_pack")
        self.assertFalse(report["recommended_next_command"]["prompt_public"])
        self.assertEqual(
            shlex.split(report["recommended_next_command"]["command_line"])[:6],
            [
                "crowdtensor",
                "public-real-llm-swarm-beta",
                "release",
                "--output-dir",
                str(output_dir.resolve()),
                "--hf-model-id",
            ],
        )
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["answer_scope"]["visible_in_terminal"])
        self.assertFalse(report["shareable_summary"]["raw_generated_text_public"])
        self.assertIn("runtime failed", report["step"]["stderr_tail"])
        self.assertNotIn("runtime failed", json.dumps(report["review_summary"], sort_keys=True))
        self.assertNotIn("runtime failed", json.dumps(report["operator_action"], sort_keys=True))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_public_real_llm_swarm_beta(report)
        rendered = stdout.getvalue()
        self.assertIn(
            "  review: state=blocked next=review_diagnostics inspect=public_real_llm_swarm_beta.md support=support_bundle.json recommended=rerun public real LLM beta not_completed=1 public_artifact_safe=True",
            rendered,
        )
        self.assertIn("  recommended_next: crowdtensor public-real-llm-swarm-beta release", rendered)
        self.assertIn("  inspect_first: public_real_llm_swarm_beta.md", rendered)
        self.assertIn("  operator_action:", rendered)
        self.assertIn("    - Inspect the CLI step payload, then rerun: crowdtensor public-real-llm-swarm-beta release", rendered)
        self.assertIn("  not_completed:", rendered)

    def test_public_real_llm_swarm_beta_cli_failure_quotes_rerun_context_without_prompt(self) -> None:
        output_dir = Path(self._tmp_dir()) / "beta output"
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "local-model-variant",
            "--output-dir",
            str(output_dir),
            "--hf-model-id",
            "distilgpt2",
            "--prompt-text",
            "private prompt text",
            "--stream-generation",
            "--max-new-tokens",
            "24",
            "--http-timeout",
            "12.5",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_real_llm_swarm_beta_pack.py", command[1])
            return subprocess.CompletedProcess(command, 1, stdout="not json\n", stderr="pack crashed\n")

        report = cli.build_public_real_llm_swarm_beta(args, runner=fake_runner)
        command_line = report["recommended_next_command"]["command_line"]

        self.assertFalse(report["ok"], report)
        self.assertIn("'", command_line)
        self.assertNotIn("private prompt text", json.dumps(report, sort_keys=True))
        self.assertEqual(
            shlex.split(command_line)[:11],
            [
                "crowdtensor",
                "public-real-llm-swarm-beta",
                "local-model-variant",
                "--output-dir",
                str(output_dir.resolve()),
                "--hf-model-id",
                "distilgpt2",
                "--max-new-tokens",
                "24",
                "--http-timeout",
                "12.5",
            ],
        )
        self.assertIn("--stream-generation", shlex.split(command_line))
        self.assertIn(command_line, report["operator_action"][0])

    def test_public_real_llm_swarm_beta_cli_rejects_unbounded_prompt_batch(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "public-real-llm-swarm-beta",
                "local-smoke",
                "--prompt-texts",
                "one,two,three,four,five",
            ])

    def test_public_real_llm_swarm_beta_cli_rejects_inline_and_file_prompt_batch(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_file = output_dir / "prompts.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text("first prompt\n", encoding="utf-8")

        with self.assertRaises(SystemExit) as raised:
            cli.parse_args([
                "public-real-llm-swarm-beta",
                "local-smoke",
                "--prompt-texts",
                "first prompt,second prompt",
                "--prompt-texts-file",
                str(prompt_file),
            ])

        self.assertEqual(
            str(raised.exception),
            "public-real-llm-swarm-beta accepts either --prompt-texts or --prompt-texts-file, not both",
        )

    def test_public_real_llm_swarm_beta_check_skips_prompt_batch_validation(self) -> None:
        args = cli.parse_args([
            "public-real-llm-swarm-beta",
            "check",
            "--prompt-texts",
            "one,two,three,four,five",
        ])

        self.assertEqual(args.public_real_llm_swarm_beta_mode, "check")

    def test_public_real_llm_swarm_beta_help_shows_modes_artifacts_and_boundaries(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            cli.parse_args(["public-real-llm-swarm-beta", "--help"])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        self.assertIn("Build or verify the top-level Public Real-LLM Swarm Inference Beta evidence.", rendered)
        self.assertIn("Default release mode runs the product serve/join/generate path", rendered)
        self.assertIn("operator actions, and not_completed blockers", rendered)
        self.assertIn("Modes:", rendered)
        self.assertIn("release             run the full final 16-token aggregate gate", rendered)
        self.assertIn("local-smoke         run only a local product-path smoke", rendered)
        self.assertIn("local-model-variant prove the local model variant", rendered)
        self.assertIn("package             generate the runbook/package", rendered)
        self.assertIn("evidence-import     aggregate retained reports", rendered)
        self.assertIn("check               run the final validation check", rendered)
        self.assertIn("Review path: open public_real_llm_swarm_beta.md first", rendered)
        self.assertIn("support_bundle.json", rendered)
        self.assertIn("Safe shareable files", rendered)
        self.assertIn("Do not share private env", rendered)
        self.assertIn("generated token ids", rendered)
        self.assertIn("If ok is false, start with the Not Completed section", rendered)
        self.assertIn("printed not_completed lines", rendered)
        self.assertIn("crowdtensor public-real-llm-swarm-beta release --max-new-tokens 16", rendered)
        self.assertIn("crowdtensor public-real-llm-swarm-beta check --beta-report", rendered)
        self.assertIn("not production", rendered)
        self.assertIn("not Coordinator-free P2P", rendered)
        self.assertIn("not large-model serving", rendered)

    def test_public_real_llm_swarm_beta_prints_output_scope(self) -> None:
        report = {
            "schema": "public_real_llm_swarm_beta_v1",
            "cli_schema": "public_real_llm_swarm_beta_cli_v1",
            "ok": True,
            "mode": "release",
            "output_dir": "dist/beta",
            "beta": {
                "ready": True,
                "hf_model_id": "sshleifer/tiny-gpt2",
                "max_new_tokens": 16,
                "cpu_default_ready": True,
                "external_two_stage_ready": True,
                "external_stage_requeue_ready": True,
                "p2p_ready_product_beta": True,
                "p2p_batch_ready": True,
                "p2p_stream_ready": True,
                "public_swarm_v2_ready": True,
                "public_swarm_v2_batch_ready": True,
                "public_swarm_v2_stream_ready": True,
                "public_swarm_v2_real_p2p_local_ready": True,
                "public_swarm_v2_real_p2p_local_requeue_ready": True,
                "kv_cache_ready": True,
                "cuda_optional_fail_closed_ready": True,
                "batch": {"batch_generation_ready": True},
                "stream": {"stream_generation_ready": True},
            },
            "readiness": {
                "product_path": {"max_new_tokens": 16},
                "external_kaggle": {
                    "generated_token_count": 16,
                    "required_generated_token_count": 16,
                },
                "p2p_candidate": {
                    "generated_token_count": 16,
                    "required_generated_token_count": 16,
                },
                "public_swarm_v2": {
                    "generated_token_count": 16,
                    "required_generated_token_count": 16,
                    "accepted_rows": 32,
                    "required_stage_rows": 32,
                },
                "usable_p2p_kv_cache": {
                    "stage0": {"hit_count": 15},
                    "stage1": {"hit_count": 15},
                },
            },
            "output_request": {
                "include_output": False,
                "raw_generated_text_public": False,
                "public_artifact_safe": True,
            },
            "prompt_scope": {
                "source": "prompt-file",
                "prompt_count": 1,
                "inline_prompt_text": False,
                "terminal_next_commands_local_private": False,
                "terminal_local_paths": False,
                "saved_artifacts_prompt_placeholders": True,
                "prompt_file_path_public": False,
                "raw_prompt_public": False,
                "public_artifact_safe": True,
                "summary": "Prompt file paths and raw prompt text are excluded from public artifacts.",
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
                "summary": cli.SAVED_NO_ANSWER_SCOPE_TEXT,
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "artifact_summary": {
                "inspect_first": "public_real_llm_swarm_beta.md",
                "machine_readable": "public_real_llm_swarm_beta.json",
                "support_bundle": "support_bundle.json",
                "runbook": "PUBLIC_REAL_LLM_SWARM_BETA.md",
                "shareable_paths": [
                    "public_real_llm_swarm_beta.json",
                    "public_real_llm_swarm_beta.md",
                    "support_bundle.json",
                ],
                "public_artifact_safe": True,
            },
            "review_summary": {
                "state": "ready",
                "ready": True,
                "next_step": "run_beta_report_check",
                "inspect_first": "public_real_llm_swarm_beta.md",
                "support_bundle": "support_bundle.json",
                "recommended_check_command": {
                    "label": "validate beta report",
                    "command_line": "crowdtensor public-real-llm-swarm-beta check --beta-report dist/beta/public_real_llm_swarm_beta.json --output-dir dist/beta-check --max-new-tokens 16 --json",
                    "check_source": "beta-report",
                },
                "not_completed_count": 0,
                "public_artifact_safe": True,
            },
            "recommended_check_command": {
                "label": "validate beta report",
                "command_line": "crowdtensor public-real-llm-swarm-beta check --beta-report dist/beta/public_real_llm_swarm_beta.json --output-dir dist/beta-check --max-new-tokens 16 --json",
                "check_source": "beta-report",
            },
            "operator_action": [
                "Use `crowdtensor serve`, `crowdtensor join --stage stage0`, `crowdtensor join --stage stage1`, and `crowdtensor generate` as the primary user path.",
                "Share this top-level JSON/Markdown artifact; raw prompts, generated text, token ids, activations, and credentials are excluded.",
            ],
            "diagnosis_codes": ["public_real_llm_swarm_beta_ready"],
            "artifacts": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_public_real_llm_swarm_beta(report)
        output = stdout.getvalue()

        self.assertIn("  model: sshleifer/tiny-gpt2 tokens=16", output)
        self.assertIn("  external tokens: 16/16", output)
        self.assertIn("  p2p tokens: 16/16", output)
        self.assertIn("  public_swarm_v2 tokens: 16/16 accepted_rows=32/32", output)
        self.assertIn(
            "  review: state=ready next=run_beta_report_check inspect=public_real_llm_swarm_beta.md support=support_bundle.json recommended=validate beta report not_completed=0 public_artifact_safe=True",
            output,
        )
        self.assertIn("  recommended_check: crowdtensor public-real-llm-swarm-beta check --beta-report dist/beta/public_real_llm_swarm_beta.json", output)
        self.assertIn("  inspect_first: public_real_llm_swarm_beta.md", output)
        self.assertIn(
            "  artifacts: inspect=public_real_llm_swarm_beta.md json=public_real_llm_swarm_beta.json support=support_bundle.json runbook=PUBLIC_REAL_LLM_SWARM_BETA.md shareable=public_real_llm_swarm_beta.json,public_real_llm_swarm_beta.md,support_bundle.json public_artifact_safe=True",
            output,
        )
        self.assertIn("  public_swarm_v2 real_p2p_local: route=True requeue=True", output)
        self.assertIn("  batch ready: product=True p2p=True v2=True", output)
        self.assertIn("  stream ready: product=True p2p=True v2=True", output)
        self.assertIn("  kv_cache_ready: True", output)
        self.assertIn("  kv_cache hits: stage0=15 stage1=15", output)
        self.assertIn(
            "  output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  prompt_scope: source=prompt-file count=1 inline_prompt_text=False",
            output,
        )
        self.assertIn(
            "  prompt_scope_note: Prompt file paths and raw prompt text are excluded from public artifacts.",
            output,
        )
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            output,
        )
        self.assertIn("  operator_action:", output)
        self.assertIn("    - Use `crowdtensor serve`, `crowdtensor join --stage stage0`, `crowdtensor join --stage stage1`, and `crowdtensor generate` as the primary user path.", output)

    def test_public_real_llm_swarm_beta_prints_blockers(self) -> None:
        report = {
            "schema": "public_real_llm_swarm_beta_v1",
            "cli_schema": "public_real_llm_swarm_beta_cli_v1",
            "ok": False,
            "mode": "release",
            "output_dir": "dist/beta",
            "beta": {
                "ready": False,
                "hf_model_id": "sshleifer/tiny-gpt2",
                "max_new_tokens": 16,
                "cpu_default_ready": True,
                "external_two_stage_ready": False,
                "external_stage_requeue_ready": False,
                "p2p_ready_product_beta": True,
                "public_swarm_v2_ready": False,
                "kv_cache_ready": False,
            },
            "readiness": {
                "public_swarm_v2": {
                    "generated_token_count": 8,
                    "required_generated_token_count": 16,
                    "accepted_rows": 16,
                    "required_stage_rows": 32,
                },
                "usable_p2p_kv_cache": {
                    "stage0": {"hit_count": 7},
                    "stage1": {"hit_count": 7},
                },
            },
            "diagnosis_codes": ["public_real_llm_swarm_beta_blocked"],
            "not_completed": [
                "external Kaggle two-stage real LLM proof",
                "Public Swarm v2 generated token target",
                "persistent dual-stage KV-cache reuse",
            ],
            "review_summary": {
                "state": "blocked",
                "ready": False,
                "next_step": "review_not_completed",
                "inspect_first": "public_real_llm_swarm_beta.md",
                "support_bundle": "support_bundle.json",
                "not_completed_count": 3,
                "public_artifact_safe": True,
            },
            "operator_action": [
                "Use `crowdtensor serve`, `crowdtensor join --stage stage0`, `crowdtensor join --stage stage1`, and `crowdtensor generate` as the primary user path.",
                "Review the Not Completed list, then rerun the Beta gate after fixing missing evidence.",
            ],
            "artifacts": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_public_real_llm_swarm_beta(report)
        output = stdout.getvalue()

        self.assertIn("  ready: False", output)
        self.assertIn(
            "  review: state=blocked next=review_not_completed inspect=public_real_llm_swarm_beta.md support=support_bundle.json recommended=none not_completed=3 public_artifact_safe=True",
            output,
        )
        self.assertIn("  public_swarm_v2 tokens: 8/16 accepted_rows=16/32", output)
        self.assertIn("  kv_cache hits: stage0=7 stage1=7", output)
        self.assertIn("  operator_action:", output)
        self.assertIn("    - Review the Not Completed list, then rerun the Beta gate after fixing missing evidence.", output)
        self.assertIn("  not_completed:", output)
        self.assertIn("    - external Kaggle two-stage real LLM proof", output)
        self.assertIn("    - Public Swarm v2 generated token target", output)
        self.assertIn("    - persistent dual-stage KV-cache reuse", output)

    def test_usable_swarm_cli_wraps_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "usable-swarm",
            "local",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("usable_swarm_inference_pack.py", command[1])
            self.assertEqual(command[2], "local")
            self.assertEqual(command[command.index("--max-new-tokens") + 1], "8")
            return completed({
                "schema": "usable_swarm_inference_v1",
                "ok": True,
                "mode": "local",
                "diagnosis_codes": ["usable_swarm_inference_ready"],
            })

        report = cli.build_usable_swarm_inference(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "usable_swarm_inference_cli_v1")
        self.assertTrue(calls)

    def test_usable_swarm_help_explains_modes_and_output_scope(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            cli.main(["usable-swarm", "--help"])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        normalized = " ".join(rendered.split())
        self.assertIn("local runs the p2pd -> serve --p2p -> join stage0/stage1 -> generate --p2p path", normalized)
        self.assertIn("package writes the runbook without running services", normalized)
        self.assertIn("evidence-import validates an existing P2P report", normalized)
        self.assertIn("Artifacts are shareable readiness evidence, not answer transcripts", normalized)

    def test_public_swarm_v2_help_explains_modes_and_output_scope(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            cli.main(["public-swarm-v2", "--help"])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        normalized = " ".join(rendered.split())
        self.assertIn("local runs a fresh Usable Swarm v1 p2pd -> serve --p2p -> join stage0/stage1 -> generate --p2p proof", normalized)
        self.assertIn("local-model-variant validates the requested Hugging Face model locally", normalized)
        self.assertIn("package writes the runbook and shareable package without running services", normalized)
        self.assertIn("evidence-import validates retained usable, real-P2P, preview, and optional GPU evidence", normalized)
        self.assertIn("Artifacts are shareable readiness evidence, not answer transcripts", normalized)

    def test_preview_help_explains_modes_and_output_scope(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            cli.main(["preview", "--help"])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        normalized = " ".join(rendered.split())
        self.assertIn("local runs a localhost Product Beta serve/join/generate proof", normalized)
        self.assertIn("package writes join material and runbook artifacts without proving live readiness", normalized)
        self.assertIn("external-existing verifies an already running controlled Coordinator plus stage Miners", normalized)
        self.assertIn("evidence-import aggregates retained Product Beta and optional GPU generation evidence", normalized)
        self.assertIn("Artifacts are shareable preview evidence, not answer transcripts", normalized)

    def test_usable_swarm_prints_output_scope(self) -> None:
        report = {
            "schema": "usable_swarm_inference_v1",
            "cli_schema": "usable_swarm_inference_cli_v1",
            "ok": True,
            "mode": "local",
            "output_dir": "dist/usable",
            "usable_swarm": {"ready": True},
            "readiness": {
                "p2p_product_path": {
                    "route_ready": True,
                    "real_generate_ready": True,
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "distinct_stage_miners": True,
                    "stage_rescue_ready": True,
                    "real_stage_rescue_ready": True,
                }
            },
            "output_request": {
                "include_output": False,
                "raw_generated_text_public": False,
                "public_artifact_safe": True,
            },
            "prompt_scope": {
                "source": "prompt-file",
                "prompt_count": 1,
                "inline_prompt_text": False,
                "terminal_next_commands_local_private": False,
                "terminal_local_paths": False,
                "saved_artifacts_prompt_placeholders": True,
                "prompt_file_path_public": False,
                "raw_prompt_public": False,
                "public_artifact_safe": True,
                "summary": "Usable Swarm evidence excludes prompt file paths and raw prompt text.",
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
                "summary": cli.SAVED_NO_ANSWER_SCOPE_TEXT,
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "user_status": {
                "state": "completed",
                "headline": "Usable Swarm inference evidence is ready.",
                "next_step": "review_artifacts",
                "recommended_label": "inspect shareable summary",
                "recommended_reason": "review_artifacts",
                "public_artifact_safe": True,
            },
            "review_summary": {
                "state": "completed",
                "next_step": "review_artifacts",
                "inspect_first": "dist/usable/usable_swarm_inference.md",
                "support_bundle": "dist/usable/support_bundle.json",
                "runbook": "dist/usable/USABLE_SWARM_INFERENCE.md",
                "recommended_label": "inspect shareable summary",
                "recommended_reason": "review_artifacts",
                "next_command": "sed -n 1,220p dist/usable/usable_swarm_inference.md",
                "primary_code": "usable_swarm_inference_ready",
                "attention": "none",
                "public_artifact_safe": True,
            },
            "recommended_next_command": {
                "label": "inspect shareable summary",
                "reason": "review_artifacts",
                "command_line": "sed -n 1,220p dist/usable/usable_swarm_inference.md",
                "public_artifact_safe": True,
            },
            "next_commands": [
                {
                    "label": "inspect shareable summary",
                    "command_line": "sed -n 1,220p dist/usable/usable_swarm_inference.md",
                    "public_artifact_safe": True,
                },
                {
                    "label": "open runbook",
                    "command_line": "sed -n 1,260p dist/usable/USABLE_SWARM_INFERENCE.md",
                    "public_artifact_safe": True,
                },
            ],
            "diagnosis_codes": ["usable_swarm_inference_ready"],
            "artifacts": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_usable_swarm_inference(report)

        rendered = stdout.getvalue()
        self.assertIn("  review: state=completed next=review_artifacts", rendered)
        self.assertIn("  review_next: label=inspect shareable summary reason=review_artifacts", rendered)
        self.assertIn("  inspect_first: dist/usable/usable_swarm_inference.md", rendered)
        self.assertIn("  status: completed: Usable Swarm inference evidence is ready.", rendered)
        self.assertIn("  recommended_next: inspect shareable summary reason=review_artifacts", rendered)
        self.assertIn("  next[1] inspect shareable summary: sed -n 1,220p dist/usable/usable_swarm_inference.md", rendered)
        self.assertIn("  next[2] open runbook: sed -n 1,260p dist/usable/USABLE_SWARM_INFERENCE.md", rendered)
        self.assertIn(
            "  output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  prompt_scope: source=prompt-file count=1 inline_prompt_text=False",
            rendered,
        )
        self.assertIn(
            "  prompt_scope_note: Usable Swarm evidence excludes prompt file paths and raw prompt text.",
            rendered,
        )
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            rendered,
        )

    def test_usable_swarm_cli_forwards_bounded_prompt_batch(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "usable-swarm",
            "local",
            "--output-dir",
            str(output_dir),
            "--prompt-texts",
            "first prompt,second prompt",
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("usable_swarm_inference_pack.py", command[1])
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
            self.assertNotIn("--prompt-text", command)
            return completed({
                "schema": "usable_swarm_inference_v1",
                "ok": True,
                "mode": "local",
                "diagnosis_codes": ["usable_swarm_inference_ready", "public_swarm_generate_batch_ready"],
            })

        report = cli.build_usable_swarm_inference(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "usable_swarm_inference_cli_v1")
        self.assertTrue(calls)

    def test_usable_swarm_cli_forwards_prompt_texts_file(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_file = output_dir / "prompts.txt"
        prompts = ["first prompt, with comma", "second prompt"]
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        args = cli.parse_args([
            "usable-swarm",
            "local",
            "--output-dir",
            str(output_dir),
            "--prompt-texts-file",
            str(prompt_file),
            "--json",
        ])
        self.assertEqual(args.prompt_texts_list, prompts)
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("usable_swarm_inference_pack.py", command[1])
            self.assertIn("--prompt-texts-file", command)
            self.assertEqual(command[command.index("--prompt-texts-file") + 1], str(prompt_file))
            self.assertNotIn("--prompt-texts", command)
            self.assertNotIn("--prompt-text", command)
            command_text = " ".join(command)
            self.assertNotIn(prompts[0], command_text)
            self.assertNotIn(prompts[1], command_text)
            return completed({
                "schema": "usable_swarm_inference_v1",
                "ok": True,
                "mode": "local",
                "prompt_scope": {
                    "source": "prompt-texts-file",
                    "prompt_count": 2,
                    "raw_prompt_public": False,
                },
                "diagnosis_codes": ["usable_swarm_inference_ready", "public_swarm_generate_batch_ready"],
            })

        report = cli.build_usable_swarm_inference(args, runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "usable_swarm_inference_cli_v1")
        self.assertEqual(report["prompt_scope"]["source"], "prompt-texts-file")
        self.assertNotIn(prompts[0], encoded)
        self.assertNotIn(prompts[1], encoded)
        self.assertTrue(calls)

    def test_usable_swarm_cli_forwards_stream_generation(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "usable-swarm",
            "local",
            "--output-dir",
            str(output_dir),
            "--stream-generation",
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("usable_swarm_inference_pack.py", command[1])
            self.assertIn("--stream-generation", command)
            return completed({
                "schema": "usable_swarm_inference_v1",
                "ok": True,
                "mode": "local",
                "diagnosis_codes": ["usable_swarm_inference_ready", "public_swarm_generate_stream_ready"],
            })

        report = cli.build_usable_swarm_inference(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "usable_swarm_inference_cli_v1")
        self.assertTrue(calls)

    def test_usable_swarm_cli_rejects_unbounded_prompt_batch(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "usable-swarm",
                "local",
                "--prompt-texts",
                "one,two,three,four,five",
            ])

    def test_usable_swarm_cli_rejects_inline_and_file_prompt_batch(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_file = output_dir / "prompts.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text("first prompt\n", encoding="utf-8")

        with self.assertRaises(SystemExit) as raised:
            cli.parse_args([
                "usable-swarm",
                "local",
                "--prompt-texts",
                "first prompt,second prompt",
                "--prompt-texts-file",
                str(prompt_file),
            ])

        self.assertEqual(
            str(raised.exception),
            "usable-swarm accepts either --prompt-texts or --prompt-texts-file, not both",
        )

    def test_public_swarm_v2_cli_wraps_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-swarm-v2",
            "local",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "16",
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_v2_pack.py", command[1])
            self.assertEqual(command[2], "local")
            self.assertEqual(command[command.index("--max-new-tokens") + 1], "16")
            self.assertIn("--usable-report", command)
            self.assertIn("--real-p2p-report", command)
            self.assertIn("--gpu-report", command)
            self.assertIn("--fresh-external-attempt-report", command)
            self.assertEqual(command[command.index("--fresh-external-attempt-report") + 1], "")
            self.assertIn("--real-p2p-port", command)
            self.assertIn("--real-p2p-coordinator-port", command)
            self.assertIn("--real-p2p-libp2p-port", command)
            self.assertIn("--real-p2p-discovery-backend", command)
            self.assertEqual(command[command.index("--real-p2p-discovery-backend") + 1], "http-provider-store")
            self.assertNotIn("--fresh-external-report", command)
            return completed({
                "schema": "public_swarm_inference_v2",
                "ok": True,
                "mode": "local",
                "diagnosis_codes": ["public_swarm_inference_v2_ready"],
            })

        report = cli.build_public_swarm_inference_v2(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "public_swarm_inference_v2_cli_v1")
        self.assertTrue(calls)

    def test_infer_local_defaults_to_product_loopback(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "CrowdTensor user prompt"
        args = cli.parse_args([
            "infer",
            prompt,
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("product_swarm_mvp_check.py", command[1])
            self.assertNotIn("--prompt-text", command)
            self.assertNotIn(prompt, command)
            self.assertIn("--prompt-file", command)
            prompt_path = Path(command[command.index("--prompt-file") + 1])
            self.assertEqual(prompt_path.parent, output_dir.resolve() / ".private")
            self.assertEqual(prompt_path.read_text(encoding="utf-8"), prompt)
            self.assertEqual(command[command.index("--max-new-tokens") + 1], "8")
            self.assertIn("--port", command)
            selected_port = int(command[command.index("--port") + 1])
            self.assertGreater(selected_port, 0)
            self.assertIn("--require-hf-runtime", command)
            return completed({
                "schema": "product_swarm_mvp_check_v1",
                "ok": True,
                "mode": "local-loopback",
                "hf_model_id": "sshleifer/tiny-gpt2",
                "generation": {
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "generated_text_hash": "sha256:generated",
                    "decoded_tokens_match": True,
                },
                "stage_assignment": {"distinct_stage_miners": True},
                "ledger": {"accepted_rows": 16},
                "diagnosis_codes": ["product_swarm_mvp_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "crowdtensor_infer_cli_v1")
        self.assertEqual(report["mode"], "local")
        self.assertEqual(report["generation"]["generated_token_count"], 8)
        self.assertEqual(report["result"]["status"], "complete")
        self.assertEqual(report["result"]["generated_token_count"], 8)
        self.assertEqual(report["result"]["max_new_tokens"], 8)
        self.assertEqual(report["result"]["display"], "hash-only-json")
        self.assertTrue(report["result"]["public_artifact_safe"])
        self.assertEqual(report["local_output"]["output_count"], 1)
        self.assertEqual(
            report["local_output_note"],
            "Generated output is present, but raw text is suppressed in JSON/public output; rerun without --json for local display.",
        )
        self.assertEqual(report["answer_scope"]["scope_state"], "json-suppressed")
        self.assertEqual(report["answer_scope"]["summary"], cli.SAVED_ANSWER_SCOPE_TEXT)
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "json-suppressed")
        self.assertEqual(report["recommended_next_command"]["label"], "optional broader local evidence")
        self.assertEqual(report["recommended_next_command"]["reason"], "collect_broader_evidence")
        self.assertEqual(
            report["recommended_next_command"]["reason_detail"],
            "Optionally run the broader local evidence path for stronger proof.",
        )
        self.assertEqual(
            report["operator_action"],
            "Inference completed; optionally rerun with --full-evidence for the broader Public Swarm v2 proof.",
        )
        self.assertEqual(report["runtime_options"]["coordinator_port"], int(calls[0][calls[0].index("--port") + 1]))
        self.assertTrue(report["runtime_options"]["coordinator_port_auto"])
        self.assertFalse(report["runtime_options"]["coordinator_port_explicit"])
        self.assertEqual(report["route"]["route_source"], "local-product-loopback")
        self.assertIn("crowdtensor_infer_ready", report["diagnosis_codes"])
        self.assertFalse((output_dir / ".private").exists())
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode local --output-dir {output_dir} --max-new-tokens 8",
            next_lines,
        )
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode local --output-dir {output_dir} --max-new-tokens 8 --full-evidence",
            next_lines,
        )
        self.assertEqual([item["label"] for item in report["next_commands"]], [
            "rerun local inference",
            "optional broader local evidence",
        ])
        self.assertNotIn(prompt, json.dumps(report, sort_keys=True))
        self.assertNotIn(".private", json.dumps(report, sort_keys=True))
        self.assertTrue(report["artifacts"]["product_swarm_mvp_report"]["present"] is False)
        self.assertTrue((output_dir / "infer_summary.json").is_file())
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["result"]["status"], "complete")
        self.assertEqual(persisted["result"]["display"], "hash-only-json")
        self.assertTrue(persisted["result"]["public_artifact_safe"])
        self.assertEqual(persisted["batch"]["observed_request_count"], 1)
        self.assertEqual(persisted["local_output"]["output_count"], 1)
        self.assertEqual(
            persisted["local_output_note"],
            "Generated output is present, but raw text is suppressed in JSON/public output; rerun without --json for local display.",
        )
        self.assertEqual(persisted["answer_scope"]["scope_state"], "json-suppressed")
        self.assertEqual(persisted["answer_scope"]["summary"], cli.SAVED_ANSWER_SCOPE_TEXT)
        self.assertEqual(persisted["shareable_summary"]["answer_scope_state"], "json-suppressed")
        self.assertEqual(persisted["recommended_next_command"]["label"], "optional broader local evidence")
        self.assertEqual(
            persisted["recommended_next_command"]["reason_detail"],
            "Optionally run the broader local evidence path for stronger proof.",
        )
        self.assertEqual(
            persisted["operator_action"],
            "Inference completed; optionally rerun with --full-evidence for the broader Public Swarm v2 proof.",
        )
        self.assertEqual(persisted["runtime_options"]["coordinator_port"], int(calls[0][calls[0].index("--port") + 1]))
        self.assertTrue(persisted["runtime_options"]["coordinator_port_auto"])
        self.assertFalse(persisted["runtime_options"]["coordinator_port_explicit"])
        self.assertNotIn(prompt, json.dumps(persisted, sort_keys=True))
        self.assertNotIn(".private", json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn(
            "- Local output: `available=False display_only=False public_artifact_safe=True saved_redacted=True` count=`1` source=``",
            markdown,
        )
        self.assertIn("- Batch: enabled=`False` requests=`1/1` ready=`False`", markdown)
        self.assertIn(
            "- Local output note: Generated output is present, but raw text is suppressed in JSON/public output; rerun without --json for local display.",
            markdown,
        )
        self.assertIn(
            "- Action: Inference completed; optionally rerun with --full-evidence for the broader Public Swarm v2 proof.",
            markdown,
        )
        self.assertIn("- Answer scope: `state=json-suppressed ", markdown)
        self.assertIn(f"- Answer scope note: {cli.SAVED_ANSWER_SCOPE_TEXT}", markdown)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        rendered = stdout.getvalue()
        self.assertIn(
            "  action: Inference completed; optionally rerun with --full-evidence for the broader Public Swarm v2 proof.",
            rendered,
        )
        self.assertIn("coordinator_port_auto=True", rendered)
        self.assertIn("recommended_next: optional broader local evidence reason=collect_broader_evidence", rendered)
        self.assertIn("next[1] rerun local inference", rendered)
        self.assertIn("next[2] optional broader local evidence", rendered)
        self.assertIn(
            "  local_output: available=False display_only=False public_artifact_safe=True saved_redacted=True count=1 source=none",
            rendered,
        )
        self.assertNotIn("  answer:", rendered)
        self.assertTrue(calls)

    def test_infer_local_preserves_explicit_coordinator_port(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--output-dir",
            str(output_dir),
            "--coordinator-port",
            "9321",
            "--max-new-tokens",
            "2",
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("--port", command)
            self.assertEqual(command[command.index("--port") + 1], "9321")
            return completed({
                "schema": "product_swarm_mvp_check_v1",
                "ok": True,
                "mode": "local-loopback",
                "hf_model_id": "sshleifer/tiny-gpt2",
                "generation": {
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "generated_text_hash": "sha256:generated",
                    "decoded_tokens_match": True,
                },
                "stage_assignment": {"distinct_stage_miners": True},
                "ledger": {"accepted_rows": 4},
                "diagnosis_codes": ["product_swarm_mvp_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["runtime_options"]["coordinator_port"], 9321)
        self.assertFalse(report["runtime_options"]["coordinator_port_auto"])
        self.assertTrue(report["runtime_options"]["coordinator_port_explicit"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode local --output-dir {output_dir} --coordinator-port 9321 --max-new-tokens 2",
            next_lines,
        )
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["runtime_options"]["coordinator_port"], 9321)
        self.assertFalse(persisted["runtime_options"]["coordinator_port_auto"])
        self.assertTrue(persisted["runtime_options"]["coordinator_port_explicit"])
        self.assertTrue(calls)

    def test_infer_main_prints_safe_start_hint_before_human_output(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "CrowdTensor user prompt"

        def fake_build_infer(args: object) -> dict[str, object]:
            self.assertEqual(getattr(args, "prompt_text"), prompt)
            return {
                "schema": "crowdtensor_infer_cli_v1",
                "ok": True,
                "mode": "local",
                "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
                "user_status": {
                    "state": "completed",
                    "next_step": "rerun_or_review_artifacts",
                    "headline": "Inference completed.",
                    "public_artifact_safe": True,
                },
                "generation": {"generated_token_count": 2, "max_new_tokens": 2, "generated_text_hash": "sha256:generated"},
                "result": {
                    "status": "complete",
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "output_count": 1,
                    "display": "hash-only",
                    "generated_text_hash": "sha256:generated",
                    "public_artifact_safe": True,
                },
                "answer_scope": {
                    "scope_state": "no-local-answer",
                    "terminal_only": False,
                    "visible_in_terminal": False,
                    "saved_json_display": "hash-only",
                    "saved_markdown_display": "hash-only",
                    "public_artifact_safe": True,
                },
                "runtime_options": {
                    "timeout_seconds": 420.0,
                    "poll_interval": 1.0,
                    "http_timeout": 30.0,
                    "admin_results_limit": 50,
                    "public_artifact_safe": True,
                },
                "route": {"route_source": "local-product-loopback", "route_ready": True},
                "shareable_summary": {"saved_artifacts_public_safe": True},
                "output_dir": str(output_dir),
                "diagnosis_codes": ["crowdtensor_infer_ready"],
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli, "build_infer", side_effect=fake_build_infer):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                cli.main([
                    "infer",
                    prompt,
                    "--output-dir",
                    str(output_dir),
                    "--max-new-tokens",
                    "2",
                ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        progress = stderr.getvalue()
        self.assertIn("CrowdTensor infer", rendered)
        self.assertIn("starting local two-stage tiny-model proof", progress)
        self.assertIn("review, review_next", progress)
        self.assertIn("inspect_first", progress)
        self.assertIn("status/action", progress)
        self.assertIn("later lines include answer_scope", progress)
        self.assertIn("answer_scope_note", progress)
        self.assertIn("output_display_note", progress)
        self.assertIn("runtime_options", progress)
        self.assertIn("redacted JSON/Markdown artifacts", progress)
        self.assertNotIn(prompt, progress)
        self.assertLess(rendered.index("  status: "), rendered.index("  answer_scope: "))
        self.assertLess(rendered.index("  answer_scope: "), rendered.index("  runtime_options: "))

    def test_infer_existing_dry_run_start_hint_prefers_route_check_over_credentials(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_build_infer(args: object) -> dict[str, object]:
            self.assertEqual(getattr(args, "infer_mode"), "existing")
            self.assertTrue(getattr(args, "dry_run"))
            self.assertEqual(getattr(args, "admin_token"), "")
            return {
                "schema": "crowdtensor_infer_cli_v1",
                "ok": False,
                "mode": "existing",
                "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
                "user_status": {
                    "state": "blocked",
                    "next_step": "fix_blockers",
                    "headline": "Route is not ready.",
                    "public_artifact_safe": True,
                },
                "generation": {"generated_token_count": 0, "max_new_tokens": 2},
                "answer_scope": {
                    "scope_state": "no-local-answer",
                    "terminal_only": False,
                    "visible_in_terminal": False,
                    "saved_json_display": "hash-only",
                    "saved_markdown_display": "hash-only",
                    "public_artifact_safe": True,
                },
                "runtime_options": {
                    "timeout_seconds": 420.0,
                    "poll_interval": 1.0,
                    "http_timeout": 30.0,
                    "admin_results_limit": 50,
                    "public_artifact_safe": True,
                },
                "route": {"route_source": "coordinator-url", "route_ready": False},
                "output_dir": str(output_dir),
                "diagnosis_codes": ["coordinator_route_missing"],
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli, "build_infer", side_effect=fake_build_infer):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                cli.main([
                    "infer",
                    "CrowdTensor prompt",
                    "--mode",
                    "existing",
                    "--dry-run",
                    "--output-dir",
                    str(output_dir),
                ])

        self.assertEqual(raised.exception.code, 1)
        progress = stderr.getvalue()
        self.assertIn("checking the existing route before submitting work", progress)
        self.assertNotIn("checking credentials and request requirements", progress)
        self.assertIn("later lines include answer_scope", progress)
        self.assertIn("CrowdTensor infer", stdout.getvalue())

    def test_infer_json_suppresses_start_hint(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_build_infer(args: object) -> dict[str, object]:
            self.assertTrue(getattr(args, "json"))
            return {"schema": "crowdtensor_infer_cli_v1", "ok": True, "mode": "local"}

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli, "build_infer", side_effect=fake_build_infer):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                cli.main([
                    "infer",
                    "CrowdTensor user prompt",
                    "--output-dir",
                    str(output_dir),
                    "--max-new-tokens",
                    "2",
                    "--json",
                ])

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertIn('"schema": "crowdtensor_infer_cli_v1"', stdout.getvalue())

    def test_infer_existing_without_admin_token_prints_credential_start_hint(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "CrowdTensor user prompt"

        def fake_build_infer(args: object) -> dict[str, object]:
            self.assertEqual(getattr(args, "infer_mode"), "existing")
            self.assertEqual(getattr(args, "admin_token"), "")
            return {
                "schema": "crowdtensor_infer_cli_v1",
                "ok": False,
                "mode": "existing",
                "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
                "generation": {"generated_token_count": 0, "max_new_tokens": 2},
                "route": {"route_source": "coordinator-url", "route_ready": True},
                "operator_action": "Pass --admin-token or set CROWDTENSOR_ADMIN_TOKEN.",
                "output_dir": str(output_dir),
                "diagnosis_codes": ["admin_token_required"],
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli, "build_infer", side_effect=fake_build_infer):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                cli.main([
                    "infer",
                    prompt,
                    "--mode",
                    "existing",
                    "--coordinator-url",
                    "http://127.0.0.1:8787",
                    "--output-dir",
                    str(output_dir),
                    "--max-new-tokens",
                    "2",
                ])

        self.assertEqual(raised.exception.code, 1)
        progress = stderr.getvalue()
        rendered = stdout.getvalue()
        self.assertIn("checking credentials and request requirements", progress)
        self.assertNotIn("submitting to the existing swarm", progress)
        self.assertNotIn(prompt, progress)
        self.assertIn("  action: Pass --admin-token or set CROWDTENSOR_ADMIN_TOKEN.", rendered)
        self.assertNotIn("generation: None/None", rendered)
        self.assertNotIn("hash=None", rendered)

    def test_infer_markdown_without_generation_data_says_not_run(self) -> None:
        markdown = cli.render_infer_summary_markdown({
            "schema": "crowdtensor_infer_cli_v1",
            "ok": False,
            "mode": "existing",
            "diagnosis_codes": ["admin_token_required"],
            "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
            "generation": {},
            "result": {},
            "route": {"route_source": "coordinator-url", "route_ready": False},
        })

        self.assertIn("- Generation: `not-run`", markdown)
        self.assertNotIn("None/None", markdown)
        self.assertNotIn("hash=`None`", markdown)

    def test_infer_local_batch_uses_private_prompt_texts_file_for_product_loopback(self) -> None:
        output_dir = Path(self._tmp_dir())
        private_prompts = ["first prompt", "second prompt"]
        args = cli.parse_args([
            "infer",
            "--prompt-texts",
            ",".join(private_prompts),
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("product_swarm_mvp_check.py", command[1])
            self.assertNotIn("--prompt-text", command)
            self.assertNotIn("--prompt-texts", command)
            for prompt in private_prompts:
                self.assertNotIn(prompt, command)
            self.assertIn("--prompt-texts-file", command)
            prompt_path = Path(command[command.index("--prompt-texts-file") + 1])
            self.assertEqual(prompt_path.parent, output_dir.resolve() / ".private")
            self.assertEqual(prompt_path.read_text(encoding="utf-8").splitlines(), private_prompts)
            return completed({
                "schema": "product_swarm_mvp_check_v1",
                "ok": True,
                "mode": "local-loopback",
                "hf_model_id": "sshleifer/tiny-gpt2",
                "generation": {
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "generated_text_hash": "sha256:generated",
                    "decoded_tokens_match": True,
                    "request_count": 2,
                    "batch_generation_ready": True,
                },
                "batch": {
                    "enabled": True,
                    "request_count": 2,
                    "observed_request_count": 2,
                    "batch_generation_ready": True,
                },
                "stage_assignment": {"distinct_stage_miners": True},
                "ledger": {"accepted_rows": 16},
                "diagnosis_codes": ["product_swarm_mvp_ready", "product_swarm_mvp_batch_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["batch"]["enabled"])
        self.assertEqual(report["prompt"]["prompt_count"], 2)
        self.assertFalse((output_dir / ".private").exists())
        self.assertTrue(calls)

    def test_infer_local_batch_file_forwards_prompt_texts_file_to_product_loopback(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompts = ["first prompt, with comma", "second prompt"]
        prompt_file = output_dir / "prompts.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        args = cli.parse_args([
            "infer",
            "--prompt-texts-file",
            str(prompt_file),
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("product_swarm_mvp_check.py", command[1])
            self.assertNotIn("--prompt-text", command)
            self.assertNotIn("--prompt-texts", command)
            self.assertIn("--prompt-texts-file", command)
            self.assertNotIn(str(prompt_file), command)
            prompt_path = Path(command[command.index("--prompt-texts-file") + 1])
            self.assertEqual(prompt_path.parent, output_dir.resolve() / ".private")
            self.assertEqual(prompt_path.read_text(encoding="utf-8").splitlines(), prompts)
            return completed({
                "schema": "product_swarm_mvp_check_v1",
                "ok": True,
                "mode": "local-loopback",
                "hf_model_id": "sshleifer/tiny-gpt2",
                "generation": {
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "generated_text_hash": "sha256:generated",
                    "decoded_tokens_match": True,
                    "request_count": 2,
                    "batch_generation_ready": True,
                },
                "batch": {
                    "enabled": True,
                    "request_count": 2,
                    "observed_request_count": 2,
                    "batch_generation_ready": True,
                },
                "stage_assignment": {"distinct_stage_miners": True},
                "ledger": {"accepted_rows": 16},
                "diagnosis_codes": ["product_swarm_mvp_ready", "product_swarm_mvp_batch_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["prompt"]["prompt_count"], 2)
        encoded = json.dumps(report, sort_keys=True)
        for prompt in prompts:
            self.assertNotIn(prompt, encoded)
        self.assertNotIn(".private", encoded)
        self.assertFalse((output_dir / ".private").exists())
        self.assertTrue(calls)

    def test_infer_local_prompt_file_forwards_prompt_file_to_product_loopback(self) -> None:
        output_dir = Path(self._tmp_dir())
        private_prompt = "single private prompt from file"
        prompt_file = output_dir / "prompt.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(private_prompt + "\n", encoding="utf-8")
        args = cli.parse_args([
            "infer",
            "--prompt-file",
            str(prompt_file),
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("product_swarm_mvp_check.py", command[1])
            self.assertNotIn("--prompt-text", command)
            self.assertNotIn(private_prompt, command)
            self.assertIn("--prompt-file", command)
            self.assertNotIn(str(prompt_file), command)
            prompt_path = Path(command[command.index("--prompt-file") + 1])
            self.assertEqual(prompt_path.parent, output_dir.resolve() / ".private")
            self.assertEqual(prompt_path.read_text(encoding="utf-8"), private_prompt)
            return completed({
                "schema": "product_swarm_mvp_check_v1",
                "ok": True,
                "mode": "local-loopback",
                "hf_model_id": "sshleifer/tiny-gpt2",
                "generation": {
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "generated_text_hash": "sha256:generated",
                    "decoded_tokens_match": True,
                    "request_count": 1,
                },
                "batch": {
                    "enabled": False,
                    "request_count": 1,
                    "observed_request_count": 1,
                    "batch_generation_ready": False,
                },
                "stage_assignment": {"distinct_stage_miners": True},
                "ledger": {"accepted_rows": 16},
                "diagnosis_codes": ["product_swarm_mvp_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["prompt"]["prompt_count"], 1)
        self.assertEqual(report["prompt_scope"]["source"], "prompt-file")
        self.assertFalse(report["prompt_scope"]["inline_prompt_text"])
        self.assertTrue(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertTrue(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertTrue(report["prompt_scope"]["terminal_local_paths"])
        self.assertTrue(report["prompt_scope"]["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertFalse(report["prompt_scope"]["prompt_file_path_public"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn(private_prompt, encoded)
        self.assertNotIn(".private", encoded)
        self.assertFalse((output_dir / ".private").exists())
        self.assertTrue(calls)

    def test_infer_local_prompt_stdin_uses_private_prompt_file_for_product_loopback(self) -> None:
        output_dir = Path(self._tmp_dir())
        private_prompt = "single private stdin prompt"
        with patch.object(cli.sys, "stdin", io.StringIO(private_prompt + "\n")):
            args = cli.parse_args([
                "infer",
                "--prompt-stdin",
                "--output-dir",
                str(output_dir),
                "--max-new-tokens",
                "8",
                "--json",
            ])
        prompt_paths: list[Path] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("product_swarm_mvp_check.py", command[1])
            self.assertNotIn("--prompt-text", command)
            self.assertNotIn(private_prompt, command)
            self.assertIn("--prompt-file", command)
            prompt_path = Path(command[command.index("--prompt-file") + 1])
            prompt_paths.append(prompt_path)
            self.assertEqual(prompt_path.parent, output_dir.resolve() / ".private")
            self.assertEqual(prompt_path.read_text(encoding="utf-8"), private_prompt)
            return completed({
                "schema": "product_swarm_mvp_check_v1",
                "ok": True,
                "mode": "local-loopback",
                "hf_model_id": "sshleifer/tiny-gpt2",
                "generation": {
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "generated_text_hash": "sha256:generated",
                    "decoded_tokens_match": True,
                    "request_count": 1,
                },
                "batch": {
                    "enabled": False,
                    "request_count": 1,
                    "observed_request_count": 1,
                    "batch_generation_ready": False,
                },
                "stage_assignment": {"distinct_stage_miners": True},
                "ledger": {"accepted_rows": 16},
                "diagnosis_codes": ["product_swarm_mvp_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["prompt_scope"]["source"], "prompt-stdin")
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn(private_prompt, encoded)
        self.assertTrue(prompt_paths)
        self.assertFalse(prompt_paths[0].exists())
        self.assertFalse((output_dir / ".private").exists())

    def test_infer_local_failure_redacts_prompt_from_step_output(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "CrowdTensor private prompt"
        args = cli.parse_args([
            "infer",
            prompt,
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertNotIn("--prompt-text", command)
            self.assertNotIn(prompt, command)
            self.assertIn("--prompt-file", command)
            prompt_path = Path(command[command.index("--prompt-file") + 1])
            self.assertEqual(prompt_path.parent, output_dir.resolve() / ".private")
            self.assertEqual(prompt_path.read_text(encoding="utf-8"), prompt)
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout=f"no json because {prompt} failed\n",
                stderr=f"runtime echoed {prompt}\n",
            )

        report = cli.build_infer(args, runner=fake_runner)

        self.assertFalse(report["ok"], report)
        self.assertIn("<redacted>", report["step"]["stderr_tail"])
        self.assertIn("<redacted>", report["step"]["stdout_tail"])
        self.assertNotIn(prompt, json.dumps(report, sort_keys=True))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        rendered = stdout.getvalue()
        self.assertIn("  step: name=crowdtensor_infer_local_product_loopback ok=False returncode=1 error=command emitted no JSON object", rendered)
        self.assertNotIn(prompt, rendered)
        self.assertNotIn("runtime echoed", rendered)
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertNotIn(prompt, json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("- Step: `name=crowdtensor_infer_local_product_loopback ok=False returncode=1 error=command emitted no JSON object`", markdown)
        self.assertNotIn(prompt, markdown)
        self.assertNotIn("runtime echoed", markdown)
        self.assertFalse((output_dir / ".private").exists())

    def test_infer_local_source_serve_start_failure_is_actionable(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "CrowdTensor private prompt"
        args = cli.parse_args([
            "infer",
            prompt,
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertNotIn("--prompt-text", command)
            self.assertNotIn(prompt, command)
            self.assertIn("--prompt-file", command)
            prompt_path = Path(command[command.index("--prompt-file") + 1])
            self.assertEqual(prompt_path.parent, output_dir.resolve() / ".private")
            self.assertEqual(prompt_path.read_text(encoding="utf-8"), prompt)
            return completed({
                "schema": "product_swarm_mvp_check_v1",
                "ok": False,
                "mode": "local-loopback",
                "diagnosis_codes": ["serve_start_failed", "public_report_safety_failed"],
            })

        report = cli.build_infer(args, runner=fake_runner)

        self.assertFalse(report["ok"], report)
        self.assertIn("serve_start_failed", report["diagnosis_codes"])
        self.assertEqual(report["issue_summary"]["primary_code"], "serve_start_failed")
        self.assertEqual(report["review_summary"]["primary_code"], "serve_start_failed")
        self.assertIn("loopback Coordinator", report["operator_action"])
        self.assertIn("--coordinator-port", report["operator_action"])
        self.assertEqual(report["user_status"]["recommended_label"], "retry local inference on a fresh port")
        self.assertEqual(report["recommended_next_command"]["label"], "retry local inference on a fresh port")
        self.assertEqual(report["recommended_next_command"]["reason"], "follow_operator_action")
        next_lines = [item["command_line"] for item in report["next_commands"]]
        retry_lines = [line for line in next_lines if "--coordinator-port" in line]
        self.assertEqual(len(retry_lines), 1, next_lines)
        self.assertIn(f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode local --output-dir {output_dir}", retry_lines[0])
        self.assertIn("--max-new-tokens 8", retry_lines[0])
        self.assertNotIn(prompt, json.dumps(report, sort_keys=True))
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["issue_summary"]["primary_code"], "serve_start_failed")
        self.assertIn("loopback Coordinator", persisted["operator_action"])
        self.assertEqual(persisted["recommended_next_command"]["label"], "retry local inference on a fresh port")
        self.assertIn("--coordinator-port", persisted["recommended_next_command"]["command_line"])
        self.assertNotIn(prompt, json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("primary=serve_start_failed", markdown)
        self.assertIn("loopback Coordinator", markdown)
        self.assertIn("--coordinator-port", markdown)
        self.assertNotIn(prompt, markdown)
        self.assertFalse((output_dir / ".private").exists())

    def test_infer_full_evidence_uses_public_swarm_v2_local_gate(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--full-evidence",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "16",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertEqual(command[2], "local")
            child_dir = output_dir / "public-swarm-v2"
            child_dir.mkdir(parents=True, exist_ok=True)
            (child_dir / "public_swarm_inference_v2.json").write_text("{}\n", encoding="utf-8")
            (child_dir / "public_swarm_inference_v2.md").write_text("# v2\n", encoding="utf-8")
            return completed({
                "schema": "public_swarm_inference_v2",
                "ok": True,
                "mode": "local",
                "output_dir": str(child_dir),
                "user_status": {
                    "state": "ready",
                    "headline": "Public Swarm v2 inference evidence is ready.",
                    "next_step": "review_artifacts",
                    "recommended_label": "review v2 evidence",
                    "public_artifact_safe": True,
                },
                "review_summary": {
                    "state": "ready",
                    "headline": "Public Swarm v2 inference evidence is ready.",
                    "next_step": "review_artifacts",
                    "inspect_first": str(child_dir / "public_swarm_inference_v2.md"),
                    "recommended_label": "review v2 evidence",
                    "recommended_reason": "v2_ready",
                    "next_command": "less public_swarm_inference_v2.md",
                    "requires_env": [],
                    "primary_code": "public_swarm_inference_v2_ready",
                    "attention": "",
                    "public_artifact_safe": True,
                },
                "recommended_next_command": {
                    "label": "review v2 evidence",
                    "reason": "v2_ready",
                    "command_line": "less public_swarm_inference_v2.md",
                    "requires_env": [],
                    "public_artifact_safe": True,
                },
                "readiness": {
                    "local_p2p_generate": {
                        "route_ready": True,
                        "distinct_stage_miners": True,
                        "generation": {
                            "generated_token_count": 16,
                            "max_new_tokens": 16,
                            "generated_text_hash": "sha256:generated",
                        },
                    }
                },
                "diagnosis_codes": ["public_swarm_inference_v2_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        child_markdown = str(output_dir / "public-swarm-v2" / "public_swarm_inference_v2.md")
        self.assertEqual(report["source_report"]["summary_markdown_path"], child_markdown)
        self.assertEqual(report["source_report"]["summary_markdown_relative_path"], "public-swarm-v2/public_swarm_inference_v2.md")
        self.assertEqual(report["source_report"]["label"], "evidence summary")
        self.assertEqual(report["source_report"]["review_summary"]["state"], "ready")
        self.assertEqual(report["artifact_summary"]["inspect_first"], child_markdown)
        self.assertEqual(report["review_summary"]["inspect_first"], child_markdown)
        self.assertEqual(report["review_summary"]["recommended_label"], "review v2 evidence")
        self.assertEqual(report["review_summary"]["primary_code"], "public_swarm_inference_v2_ready")
        self.assertEqual(report["recommended_next_command"]["label"], "review v2 evidence")
        self.assertEqual(report["recommended_next_command"]["reason"], "v2_ready")
        self.assertEqual(report["recommended_next_command"]["command_line"], f"less {child_markdown}")
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["review_summary"]["inspect_first"], child_markdown)
        self.assertEqual(persisted["artifact_summary"]["inspect_first"], child_markdown)
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn(f"- Inspect first: `{child_markdown}`", markdown)
        self.assertIn(f"- Source evidence summary: json=`{output_dir / 'public-swarm-v2' / 'public_swarm_inference_v2.json'}` markdown=`{child_markdown}`", markdown)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        rendered = stdout.getvalue()
        self.assertIn(f"  inspect_first: {child_markdown}", rendered)
        self.assertIn(f"  recommended_next: review v2 evidence reason=v2_ready less {child_markdown}", rendered)

    def test_infer_full_evidence_blocked_keeps_full_evidence_next_step(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "CrowdTensor user prompt"
        args = cli.parse_args([
            "infer",
            prompt,
            "--full-evidence",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "16",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_inference_v2_pack.py", command[1])
            child_dir = output_dir / "public-swarm-v2"
            child_dir.mkdir(parents=True, exist_ok=True)
            (child_dir / "public_swarm_inference_v2.json").write_text("{}\n", encoding="utf-8")
            (child_dir / "public_swarm_inference_v2.md").write_text("# v2 blocked\n", encoding="utf-8")
            return completed({
                "schema": "public_swarm_inference_v2",
                "ok": False,
                "mode": "local",
                "output_dir": str(child_dir),
                "user_status": {
                    "state": "blocked",
                    "headline": "Public Swarm v2 evidence is blocked: local p2p generate route.",
                    "next_step": "fix_blockers",
                    "recommended_label": "rerun local v2 gate",
                    "public_artifact_safe": True,
                },
                "review_summary": {
                    "state": "blocked",
                    "headline": "Public Swarm v2 inference evidence needs attention.",
                    "next_step": "fix_blockers",
                    "inspect_first": str(child_dir / "public_swarm_inference_v2.md"),
                    "recommended_label": "rerun local v2 gate",
                    "recommended_reason": "fix_local_v2_evidence",
                    "next_command": "crowdtensor public-swarm-v2 local --max-new-tokens 16 --json",
                    "requires_env": ["CROWDTENSOR_ADMIN_TOKEN", "CROWDTENSOR_MINER_TOKEN"],
                    "primary_code": "public_swarm_inference_v2_blocked",
                    "attention": "local p2p generate route",
                    "attention_detail": "local p2p generate route",
                    "not_completed_count": 1,
                    "public_artifact_safe": True,
                },
                "recommended_next_command": {
                    "label": "rerun local v2 gate",
                    "reason": "fix_local_v2_evidence",
                    "command_line": "crowdtensor public-swarm-v2 local --max-new-tokens 16 --json",
                    "requires_env": ["CROWDTENSOR_ADMIN_TOKEN", "CROWDTENSOR_MINER_TOKEN"],
                    "public_artifact_safe": True,
                },
                "readiness": {
                    "local_p2p_generate": {
                        "route_ready": False,
                        "generation": {"generated_token_count": 0, "max_new_tokens": 16},
                    },
                },
                "diagnosis_codes": ["public_swarm_inference_v2_blocked"],
                "missing_requirements": ["local p2p generate route"],
            })

        report = cli.build_infer(args, runner=fake_runner)

        self.assertFalse(report["ok"], report)
        child_markdown = str(output_dir / "public-swarm-v2" / "public_swarm_inference_v2.md")
        self.assertIn("public_swarm_inference_v2_blocked", report["diagnosis_codes"])
        self.assertEqual(report["issue_summary"]["primary_code"], "public_swarm_inference_v2_blocked")
        self.assertEqual(report["review_summary"]["primary_code"], "public_swarm_inference_v2_blocked")
        self.assertEqual(report["review_summary"]["inspect_first"], child_markdown)
        self.assertEqual(report["review_summary"]["recommended_label"], "rerun local v2 gate")
        self.assertEqual(report["source_report"]["review_summary"]["not_completed_count"], 1)
        self.assertIn("Full local Public Swarm v2 evidence is blocked", report["operator_action"])
        self.assertEqual(report["recommended_next_command"]["label"], "rerun local v2 gate")
        self.assertIn("public-swarm-v2 local", report["recommended_next_command"]["command_line"])
        self.assertNotIn(prompt, json.dumps(report, sort_keys=True))
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["issue_summary"]["primary_code"], "public_swarm_inference_v2_blocked")
        self.assertEqual(persisted["review_summary"]["inspect_first"], child_markdown)
        self.assertEqual(persisted["recommended_next_command"]["label"], "rerun local v2 gate")
        self.assertIn("public-swarm-v2 local", persisted["recommended_next_command"]["command_line"])
        self.assertNotIn(prompt, json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("primary=public_swarm_inference_v2_blocked", markdown)
        self.assertIn(child_markdown, markdown)
        self.assertNotIn(prompt, markdown)

    def test_infer_full_evidence_batch_uses_private_prompt_texts_file(self) -> None:
        output_dir = Path(self._tmp_dir())
        private_prompts = ["first prompt", "second prompt"]
        args = cli.parse_args([
            "infer",
            "--prompt-texts",
            ",".join(private_prompts),
            "--full-evidence",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "16",
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_v2_pack.py", command[1])
            self.assertNotIn("--prompt-text", command)
            self.assertNotIn("--prompt-texts", command)
            for prompt in private_prompts:
                self.assertNotIn(prompt, command)
            self.assertIn("--prompt-texts-file", command)
            prompt_path = Path(command[command.index("--prompt-texts-file") + 1])
            self.assertEqual(prompt_path.parent, output_dir.resolve() / ".private")
            self.assertEqual(prompt_path.read_text(encoding="utf-8").splitlines(), private_prompts)
            return completed({
                "schema": "public_swarm_inference_v2",
                "ok": True,
                "mode": "local",
                "readiness": {
                    "local_p2p_generate": {
                        "route_ready": True,
                        "distinct_stage_miners": True,
                        "generation": {
                            "generated_token_count": 16,
                            "max_new_tokens": 16,
                            "generated_text_hash": "sha256:generated",
                            "request_count": 2,
                            "batch_generation_ready": True,
                        },
                        "batch": {
                            "enabled": True,
                            "request_count": 2,
                            "observed_request_count": 2,
                            "batch_generation_ready": True,
                        },
                    }
                },
                "diagnosis_codes": ["public_swarm_inference_v2_ready", "public_swarm_v2_batch_generation_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["batch"]["enabled"])
        self.assertEqual(report["prompt"]["prompt_count"], 2)
        self.assertFalse((output_dir / ".private").exists())
        self.assertTrue(calls)

    def test_infer_full_evidence_batch_file_forwards_prompt_texts_file(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_file = output_dir / "prompts.txt"
        prompts = ["first prompt, with comma", "second prompt"]
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        args = cli.parse_args([
            "infer",
            "--prompt-texts-file",
            str(prompt_file),
            "--full-evidence",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "16",
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_v2_pack.py", command[1])
            self.assertNotIn("--prompt-text", command)
            self.assertNotIn("--prompt-texts", command)
            self.assertIn("--prompt-texts-file", command)
            self.assertNotIn(str(prompt_file), command)
            prompt_path = Path(command[command.index("--prompt-texts-file") + 1])
            self.assertEqual(prompt_path.parent, output_dir.resolve() / ".private")
            self.assertEqual(prompt_path.read_text(encoding="utf-8").splitlines(), prompts)
            return completed({
                "schema": "public_swarm_inference_v2",
                "ok": True,
                "mode": "local",
                "readiness": {
                    "local_p2p_generate": {
                        "route_ready": True,
                        "distinct_stage_miners": True,
                        "generation": {
                            "generated_token_count": 16,
                            "max_new_tokens": 16,
                            "generated_text_hash": "sha256:generated",
                            "request_count": 2,
                            "batch_generation_ready": True,
                        },
                        "batch": {
                            "enabled": True,
                            "request_count": 2,
                            "observed_request_count": 2,
                            "batch_generation_ready": True,
                        },
                    }
                },
                "diagnosis_codes": ["public_swarm_inference_v2_ready", "public_swarm_v2_batch_generation_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["prompt"]["prompt_count"], 2)
        for prompt in prompts:
            self.assertNotIn(prompt, encoded)
        self.assertNotIn(".private", encoded)
        self.assertFalse((output_dir / ".private").exists())
        self.assertTrue(calls)

    def test_infer_full_evidence_prompt_file_forwards_prompt_file(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_file = output_dir / "prompt.txt"
        private_prompt = "single private full evidence prompt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(private_prompt + "\n", encoding="utf-8")
        args = cli.parse_args([
            "infer",
            "--prompt-file",
            str(prompt_file),
            "--full-evidence",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "16",
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_v2_pack.py", command[1])
            self.assertNotIn("--prompt-text", command)
            self.assertNotIn(private_prompt, command)
            self.assertIn("--prompt-file", command)
            self.assertNotIn(str(prompt_file), command)
            prompt_path = Path(command[command.index("--prompt-file") + 1])
            self.assertEqual(prompt_path.parent, output_dir.resolve() / ".private")
            self.assertEqual(prompt_path.read_text(encoding="utf-8"), private_prompt)
            return completed({
                "schema": "public_swarm_inference_v2",
                "ok": True,
                "mode": "local",
                "readiness": {
                    "local_p2p_generate": {
                        "route_ready": True,
                        "distinct_stage_miners": True,
                        "generation": {
                            "generated_token_count": 16,
                            "max_new_tokens": 16,
                            "generated_text_hash": "sha256:generated",
                            "request_count": 1,
                        },
                        "batch": {
                            "enabled": False,
                            "request_count": 1,
                            "observed_request_count": 1,
                            "batch_generation_ready": False,
                        },
                    }
                },
                "diagnosis_codes": ["public_swarm_inference_v2_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["prompt"]["prompt_count"], 1)
        self.assertEqual(report["prompt_scope"]["source"], "prompt-file")
        self.assertFalse(report["prompt_scope"]["inline_prompt_text"])
        self.assertTrue(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertTrue(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertTrue(report["prompt_scope"]["terminal_local_paths"])
        self.assertNotIn(private_prompt, encoded)
        self.assertNotIn(".private", encoded)
        self.assertFalse((output_dir / ".private").exists())
        self.assertTrue(calls)

    def test_infer_full_evidence_prompt_stdin_uses_private_prompt_file(self) -> None:
        output_dir = Path(self._tmp_dir())
        private_prompt = "single private full evidence stdin prompt"
        with patch.object(cli.sys, "stdin", io.StringIO(private_prompt + "\n")):
            args = cli.parse_args([
                "infer",
                "--prompt-stdin",
                "--full-evidence",
                "--output-dir",
                str(output_dir),
                "--max-new-tokens",
                "16",
                "--json",
            ])
        prompt_paths: list[Path] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_inference_v2_pack.py", command[1])
            self.assertNotIn("--prompt-text", command)
            self.assertNotIn(private_prompt, command)
            self.assertIn("--prompt-file", command)
            prompt_path = Path(command[command.index("--prompt-file") + 1])
            prompt_paths.append(prompt_path)
            self.assertEqual(prompt_path.parent, output_dir.resolve() / ".private")
            self.assertEqual(prompt_path.read_text(encoding="utf-8"), private_prompt)
            return completed({
                "schema": "public_swarm_inference_v2",
                "ok": True,
                "mode": "local",
                "readiness": {
                    "local_p2p_generate": {
                        "route_ready": True,
                        "distinct_stage_miners": True,
                        "generation": {
                            "generated_token_count": 16,
                            "max_new_tokens": 16,
                            "generated_text_hash": "sha256:generated",
                            "request_count": 1,
                        },
                        "batch": {
                            "enabled": False,
                            "request_count": 1,
                            "observed_request_count": 1,
                            "batch_generation_ready": False,
                        },
                    }
                },
                "diagnosis_codes": ["public_swarm_inference_v2_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["prompt_scope"]["source"], "prompt-stdin")
        self.assertNotIn(private_prompt, encoded)
        self.assertTrue(prompt_paths)
        self.assertFalse(prompt_paths[0].exists())
        self.assertFalse((output_dir / ".private").exists())

    def test_infer_local_preserves_safe_stream_progress(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "3",
            "--stream",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("--stream-generation", command)
            return completed({
                "schema": "product_swarm_mvp_check_v1",
                "ok": True,
                "mode": "local-loopback",
                "hf_model_id": "sshleifer/tiny-gpt2",
                "generation": {
                    "generated_token_count": 3,
                    "max_new_tokens": 3,
                    "generated_text_hash": "sha256:generated",
                    "decoded_tokens_match": True,
                },
                "stream": {
                    "enabled": True,
                    "requested": True,
                    "event_count": 3,
                    "source": "admin-session-stream",
                    "stream_generation_ready": True,
                    "progress": {
                        "stream_progress_complete": True,
                        "all_token_events_ready": True,
                        "monotonic_progress": True,
                        "observed_token_counts": [1, 2, 3],
                        "max_observed_token_count": 3,
                        "target_token_count": 3,
                        "expected_request_count": 1,
                    },
                    "events": [
                        {
                            "schema": "session_stream_event_v1",
                            "request_id": "req-1",
                            "prompt_hash": "sha256:p1",
                            "generated_token_count": 1,
                            "max_new_tokens": 3,
                            "generation_step": 0,
                            "generated_text_hash": "sha256:step0",
                            "generated_text": "must not leak",
                            "generated_token_ids": [1],
                        },
                        {
                            "schema": "session_stream_event_v1",
                            "request_id": "req-1",
                            "prompt_hash": "sha256:p1",
                            "generated_token_count": 3,
                            "max_new_tokens": 3,
                            "generation_step": 2,
                            "generated_text_hash": "sha256:step2",
                            "generated_text": "must not leak final",
                            "generated_token_ids": [1, 2, 3],
                        },
                    ],
                },
                "stage_assignment": {"distinct_stage_miners": True},
                "ledger": {"accepted_rows": 6},
                "diagnosis_codes": ["product_swarm_mvp_ready", "public_swarm_generate_stream_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["stream"]["ready"])
        self.assertEqual(report["stream"]["progress"]["observed_token_counts"], [1, 2, 3])
        self.assertEqual(report["stream"]["events"][0]["generated_text_hash"], "sha256:step0")
        self.assertEqual(report["trace"]["request_trace"][0]["generated_token_count"], 3)
        self.assertEqual(report["trace"]["request_trace"][0]["generated_text_hash"], "sha256:step2")
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("must not leak", encoded)
        self.assertNotIn("must not leak final", encoded)
        self.assertNotIn('"generated_token_ids": [1]', encoded)
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        persisted_encoded = json.dumps(persisted, sort_keys=True)
        self.assertEqual(persisted["stream"]["progress"]["observed_token_counts"], [1, 2, 3])
        self.assertEqual(persisted["trace"]["request_trace"][0]["generated_token_count"], 3)
        self.assertEqual(persisted["trace"]["request_trace"][0]["generated_text_hash"], "sha256:step2")
        self.assertNotIn("must not leak", persisted_encoded)
        self.assertNotIn("must not leak final", persisted_encoded)
        self.assertNotIn('"generated_token_ids": [1]', persisted_encoded)

    def test_infer_local_can_display_private_generated_text_without_persisting_it(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "CrowdTensor user prompt"
        generated_text = " local generated answer"
        args = cli.parse_args([
            "infer",
            prompt,
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("--keep-private-state", command)
            state_dir = output_dir / "product-swarm-mvp" / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "type": "task_completed",
                "validation": {
                    "generated_text": generated_text,
                    "generated_text_hash": cli.stable_hash_text(generated_text),
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "prompt_hash": cli.stable_hash_text(prompt),
                    "decoded_tokens_match": True,
                    "stage_id": 1,
                },
            }
            (state_dir / "tasks.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            return completed({
                "schema": "product_swarm_mvp_check_v1",
                "ok": True,
                "mode": "local-loopback",
                "generation": {
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "generated_text_hash": cli.stable_hash_text(generated_text),
                    "decoded_tokens_match": True,
                },
                "stage_assignment": {"distinct_stage_miners": True},
                "ledger": {"accepted_rows": 16},
                "diagnosis_codes": ["product_swarm_mvp_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["local_output"]["generated_text"], generated_text)
        self.assertEqual(report["local_output"]["source"], "local-private-task-state")
        self.assertEqual(report["local_output"]["outputs"][0]["generated_text"], generated_text)
        self.assertFalse(report["local_output"]["public_artifact_safe"])
        self.assertEqual(report["result"]["status"], "complete")
        self.assertEqual(report["result"]["output_count"], 1)
        self.assertEqual(report["result"]["display"], "local-private")
        self.assertFalse(report["result"]["public_artifact_safe"])
        self.assertEqual(report["output_display"]["terminal_display"], "local-private")
        self.assertTrue(report["output_display"]["terminal_text_available"])
        self.assertEqual(report["output_display"]["saved_artifact_display"], "hash-only")
        self.assertEqual(report["output_display"]["json_stdout_display"], "hash-only-json")
        self.assertFalse(report["output_display"]["include_output_requested"])
        self.assertFalse(report["output_display"]["raw_generated_text_public"])
        self.assertTrue(report["output_display"]["public_artifact_safe"])
        self.assertTrue(report["answer_scope"]["terminal_only"])
        self.assertTrue(report["answer_scope"]["visible_in_terminal"])
        self.assertEqual(report["answer_scope"]["scope_state"], "terminal-visible")
        self.assertEqual(report["answer_scope"]["saved_json_display"], "hash-only")
        self.assertEqual(report["answer_scope"]["saved_markdown_display"], "hash-only")
        self.assertTrue(report["answer_scope"]["public_artifact_safe"])
        self.assertEqual(report["answer_scope"]["summary"], cli.LOCAL_ANSWER_SCOPE_TEXT)
        self.assertEqual(
            report["local_output_note"],
            "Shown only in local human output; JSON and saved artifacts keep raw generated text redacted.",
        )
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["local_output"]["generated_text"], "")
        self.assertFalse((output_dir / "product-swarm-mvp" / "state").exists())
        self.assertEqual(persisted["private_runtime_state"]["state_dir"], "product-swarm-mvp/state")
        self.assertTrue(persisted["private_runtime_state"]["removed"])
        self.assertFalse(persisted["private_runtime_state"]["present_after_cleanup"])
        self.assertFalse(persisted["safety"]["private_runtime_state_kept"])
        self.assertTrue(persisted["safety"]["raw_runtime_state_removed"])
        self.assertIn("private_runtime_state_cleaned", persisted["diagnosis_codes"])
        self.assertEqual(persisted["local_output"]["outputs"][0]["generated_text"], "")
        self.assertFalse(persisted["local_output"]["available"])
        self.assertFalse(persisted["local_output"]["display_only"])
        self.assertTrue(persisted["local_output"]["public_artifact_safe"])
        self.assertEqual(persisted["result"]["status"], "complete")
        self.assertEqual(persisted["result"]["output_count"], 1)
        self.assertEqual(persisted["result"]["display"], "hash-only")
        self.assertTrue(persisted["result"]["public_artifact_safe"])
        self.assertEqual(persisted["output_display"]["terminal_display"], "saved-terminal-redacted")
        self.assertFalse(persisted["output_display"]["terminal_text_available"])
        self.assertEqual(persisted["output_display"]["saved_artifact_display"], "hash-only")
        self.assertEqual(persisted["output_display"]["json_stdout_display"], "hash-only-json")
        self.assertFalse(persisted["output_display"]["raw_generated_text_public"])
        self.assertTrue(persisted["output_display"]["public_artifact_safe"])
        self.assertFalse(persisted["answer_scope"]["terminal_only"])
        self.assertFalse(persisted["answer_scope"]["visible_in_terminal"])
        self.assertEqual(persisted["answer_scope"]["scope_state"], "saved-terminal-redacted")
        self.assertEqual(persisted["answer_scope"]["saved_json_display"], "hash-only")
        self.assertEqual(persisted["answer_scope"]["saved_markdown_display"], "hash-only")
        self.assertTrue(persisted["answer_scope"]["public_artifact_safe"])
        self.assertEqual(persisted["answer_scope"]["summary"], cli.SAVED_TERMINAL_ANSWER_SCOPE_TEXT)
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("- Model: `sshleifer/tiny-gpt2` backend=`cpu`", markdown)
        self.assertIn("- Prompt: `count=1 hash=", markdown)
        self.assertIn("raw_public=False`", markdown)
        self.assertIn("- Result: `status=complete tokens=8/8 outputs=1 display=hash-only hash=", markdown)
        self.assertIn("public_artifact_safe=True`", markdown)
        self.assertIn(
            "- Output display: `terminal=saved-terminal-redacted terminal_text=False saved=hash-only json_stdout=hash-only-json include_output=False raw_public=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Local output: `available=False display_only=False public_artifact_safe=True saved_redacted=True` count=`1` source=`local-private-task-state`",
            markdown,
        )
        self.assertIn(
            "- Answer scope: `state=saved-terminal-redacted terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(f"- Answer scope note: {cli.SAVED_TERMINAL_ANSWER_SCOPE_TEXT}", markdown)
        self.assertNotIn("rerun without --json for local display", markdown)
        self.assertIn(
            "- Local output note: Shown only in local human output; JSON and saved artifacts keep raw generated text redacted.",
            markdown,
        )
        self.assertIn("## Artifacts", markdown)
        self.assertIn("- `infer_summary`: path=`infer_summary.json` present=`True` kind=`crowdtensor_infer_summary`", markdown)
        self.assertIn(
            "- `product_swarm_mvp_report`: path=`product-swarm-mvp/product_swarm_mvp_check.json` present=`False` kind=`product_swarm_mvp_check`",
            markdown,
        )
        self.assertNotIn(generated_text, markdown)
        self.assertNotIn(prompt, markdown)

    def test_infer_local_marks_truncated_private_generated_text(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "CrowdTensor user prompt"
        tail = "INFER_SECRET_TAIL"
        generated_text = ("i" * cli.LOCAL_OUTPUT_DISPLAY_MAX_CHARS) + tail
        args = cli.parse_args([
            "infer",
            prompt,
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("--keep-private-state", command)
            state_dir = output_dir / "product-swarm-mvp" / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "type": "task_completed",
                "validation": {
                    "generated_text": generated_text,
                    "generated_text_hash": cli.stable_hash_text(generated_text),
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "prompt_hash": cli.stable_hash_text(prompt),
                    "decoded_tokens_match": True,
                    "stage_id": 1,
                },
            }
            (state_dir / "tasks.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            return completed({
                "schema": "product_swarm_mvp_check_v1",
                "ok": True,
                "mode": "local-loopback",
                "generation": {
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "generated_text_hash": cli.stable_hash_text(generated_text),
                    "decoded_tokens_match": True,
                },
                "stage_assignment": {"distinct_stage_miners": True},
                "ledger": {"accepted_rows": 16},
                "diagnosis_codes": ["product_swarm_mvp_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(len(report["local_output"]["generated_text"]), cli.LOCAL_OUTPUT_DISPLAY_MAX_CHARS)
        self.assertTrue(report["local_output"]["truncated"])
        self.assertEqual(report["local_output"]["max_chars"], cli.LOCAL_OUTPUT_DISPLAY_MAX_CHARS)
        self.assertEqual(report["local_output"]["omitted_char_count"], len(tail))
        self.assertTrue(report["local_output"]["outputs"][0]["truncated"])
        self.assertNotIn(tail, json.dumps(report, sort_keys=True))
        self.assertIn("Terminal answer text is truncated", report["local_output_note"])
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertFalse((output_dir / "product-swarm-mvp" / "state").exists())
        self.assertEqual(persisted["private_runtime_state"]["state_dir"], "product-swarm-mvp/state")
        self.assertTrue(persisted["private_runtime_state"]["removed"])
        self.assertFalse(persisted["private_runtime_state"]["present_after_cleanup"])
        self.assertTrue(persisted["safety"]["raw_runtime_state_removed"])
        self.assertEqual(persisted["local_output"]["generated_text"], "")
        self.assertEqual(persisted["local_output"]["outputs"][0]["generated_text"], "")
        self.assertTrue(persisted["local_output"]["truncated"])
        self.assertEqual(persisted["local_output"]["omitted_char_count"], len(tail))
        self.assertTrue(persisted["local_output"]["public_artifact_safe"])
        self.assertNotIn(tail, json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn(
            f"- Local output: `available=False display_only=False public_artifact_safe=True saved_redacted=True truncated=True max_chars={cli.LOCAL_OUTPUT_DISPLAY_MAX_CHARS} omitted_chars={len(tail)}` count=`1` source=`local-private-task-state`",
            markdown,
        )
        self.assertIn("Terminal answer text is truncated", markdown)
        self.assertNotIn(tail, markdown)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        rendered = stdout.getvalue()
        self.assertIn(
            f"local_output: available=True display_only=True public_artifact_safe=False truncated=True max_chars={cli.LOCAL_OUTPUT_DISPLAY_MAX_CHARS} omitted_chars={len(tail)} count=1 source=local-private-task-state",
            rendered,
        )
        self.assertIn("Terminal answer text is truncated", rendered)
        self.assertNotIn(tail, rendered)

    def test_infer_local_shareable_terminal_persisted_note_hides_answer(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "CrowdTensor shareable local prompt"
        generated_text = " local answer hidden from shareable terminal"
        args = cli.parse_args([
            "infer",
            prompt,
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
            "--shareable-terminal",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("--keep-private-state", command)
            state_dir = output_dir / "product-swarm-mvp" / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "type": "task_completed",
                "validation": {
                    "generated_text": generated_text,
                    "generated_text_hash": cli.stable_hash_text(generated_text),
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "prompt_hash": cli.stable_hash_text(prompt),
                    "decoded_tokens_match": True,
                    "stage_id": 1,
                },
            }
            (state_dir / "tasks.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            return completed({
                "schema": "product_swarm_mvp_check_v1",
                "ok": True,
                "mode": "local-loopback",
                "generation": {
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "generated_text_hash": cli.stable_hash_text(generated_text),
                    "decoded_tokens_match": True,
                },
                "stage_assignment": {"distinct_stage_miners": True},
                "ledger": {"accepted_rows": 16},
                "diagnosis_codes": ["product_swarm_mvp_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)

        self.assertEqual(report["shareable_terminal"]["answer_text_redacted"], True)
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["shareable_terminal"]["answer_text_redacted"], True)
        self.assertEqual(persisted["local_output"]["note"], cli.SHAREABLE_TERMINAL_ANSWER_SCOPE_TEXT)
        self.assertEqual(persisted["local_output_note"], cli.SHAREABLE_TERMINAL_ANSWER_SCOPE_TEXT)
        self.assertEqual(persisted["output_display"]["terminal_display"], "shareable-terminal-redacted")
        self.assertFalse(persisted["output_display"]["terminal_text_available"])
        self.assertEqual(
            persisted["output_display"]["summary"],
            cli.SHAREABLE_TERMINAL_OUTPUT_DISPLAY_SCOPE_TEXT,
        )
        self.assertNotIn("may show local generated text", persisted["output_display"]["summary"])
        self.assertEqual(persisted["answer_scope"]["scope_state"], "shareable-terminal-redacted")
        self.assertEqual(persisted["answer_scope"]["summary"], cli.SHAREABLE_TERMINAL_ANSWER_SCOPE_TEXT)
        self.assertEqual(persisted["shareable_summary"]["answer_scope_state"], "shareable-terminal-redacted")
        self.assertFalse(persisted["shareable_summary"]["local_answer_terminal_only"])
        self.assertTrue(persisted["local_output"]["shareable_terminal_redacted"])
        self.assertNotIn(generated_text, json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn(f"- Local output note: {cli.SHAREABLE_TERMINAL_ANSWER_SCOPE_TEXT}", markdown)
        self.assertIn(f"- Answer scope note: {cli.SHAREABLE_TERMINAL_ANSWER_SCOPE_TEXT}", markdown)
        self.assertIn(f"- Output display note: {cli.SHAREABLE_TERMINAL_OUTPUT_DISPLAY_SCOPE_TEXT}", markdown)
        self.assertNotIn("- Output display note: Non-JSON human output may show local generated text", markdown)
        self.assertIn(
            "- Output display: `terminal=shareable-terminal-redacted terminal_text=False saved=hash-only json_stdout=hash-only-json",
            markdown,
        )
        self.assertIn(
            "- Shareable terminal: `enabled=True prompt_sources_redacted=True answer_text_redacted=True public_artifact_safe=True`",
            markdown,
        )
        self.assertNotIn("Shown only in local human output", markdown)
        self.assertNotIn(generated_text, markdown)

    def test_infer_existing_uses_generate_and_does_not_persist_raw_text(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--admin-token",
            "admin-secret",
            "--include-output",
            "--output-dir",
            str(output_dir),
        ])
        generate_payload = {
            "schema": "public_swarm_product_cli_v1",
            "ok": True,
            "mode": "generate",
            "session": {
                "session_id": "real-llm-session-infer",
                "workload_type": "real_llm_sharded_infer",
                "hf_model_id": "sshleifer/tiny-gpt2",
            },
            "generation": {
                "generated_token_count": 16,
                "max_new_tokens": 16,
                "generated_text_hash": "sha256:generated",
                "decoded_tokens_match": True,
                "results": [
                    {
                        "request_id": "req-1",
                        "prompt_hash": "sha256:p1",
                        "generated_token_count": 16,
                        "max_new_tokens": 16,
                        "generated_text_hash": "sha256:generated",
                        "generated_text": "must not leak",
                        "generated_token_ids": [1, 2],
                    }
                ],
            },
            "wait_progress": {
                "poll_count": 2,
                "accepted_rows_seen": 1,
                "max_observed_token_count": 16,
                "target_token_count": 16,
                "ledger_endpoint_ready": True,
                "stream_endpoint_ready": False,
                "public_artifact_safe": True,
            },
            "route": {"route_source": "coordinator-url", "coordinator_url_present": True},
            "local_output": {"generated_text": "local text only"},
            "saved_summary": {
                "path": str(output_dir / "generate" / "generate_summary.json"),
                "markdown_path": str(output_dir / "generate" / "generate_summary.md"),
                "raw_generated_text_redacted": True,
                "public_artifact_safe": True,
            },
            "diagnosis_codes": ["public_swarm_generate_ready"],
        }

        with patch.object(cli, "build_product_generate", return_value=generate_payload):
            report = cli.build_infer(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "existing")
        self.assertEqual(report["wait_progress"]["poll_count"], 2)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        self.assertIn("  wait: polls=2 accepted_rows=1 tokens=16/16 ledger=True stream=False", stdout.getvalue())
        self.assertIn("  prompt: count=1 hash=", stdout.getvalue())
        self.assertIn("raw_public=False", stdout.getvalue())
        self.assertIn(
            "  result: status=complete tokens=16/16 outputs=1 display=local-private hash=sha256:generated terminal_private=True saved_public_artifact_safe=True",
            stdout.getvalue(),
        )
        self.assertIn(
            "  output_display: terminal=local-private terminal_text=True saved=hash-only json_stdout=hash-only-json include_output=True raw_public=False public_artifact_safe=True",
            stdout.getvalue(),
        )
        self.assertIn(
            "  status: completed: Inference completed. next=rerun_or_review_artifacts recommendation=rerun inference public_artifact_safe=True",
            stdout.getvalue(),
        )
        self.assertIn(
            f"  review: state=completed next=rerun_or_review_artifacts inspect={output_dir / 'infer_summary.md'} recommended=rerun inference primary=crowdtensor_infer_ready attention=none public_artifact_safe=True",
            stdout.getvalue(),
        )
        self.assertLess(stdout.getvalue().index("  review_next: "), stdout.getvalue().index("  inspect_first: "))
        self.assertLess(stdout.getvalue().index("  inspect_first: "), stdout.getvalue().index("  status: "))
        self.assertLess(stdout.getvalue().index("  review: "), stdout.getvalue().index("  status: "))
        self.assertLess(stdout.getvalue().index("  status: "), stdout.getvalue().index("  ok: "))
        self.assertIn(
            "  review_next: label=rerun inference reason=rerun_inference command=CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor infer '<prompt>' --mode existing",
            stdout.getvalue(),
        )
        self.assertIn(
            "  issue: state=completed primary=crowdtensor_infer_ready next=rerun_or_review_artifacts progress=polls=2 accepted_rows=1 tokens=16/16 ledger=True stream=False safe_detail=False",
            stdout.getvalue(),
        )
        self.assertIn(
            "  trace: session=real-llm-session-infer requests=1 ledger_rows=1 stream_events=0 source=public_swarm_product_cli_v1 public_artifact_safe=True",
            stdout.getvalue(),
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=True answer_scope_state=terminal-visible local_answer_terminal_only=True",
            stdout.getvalue(),
        )
        self.assertIn(
            "  output_request: include_output=True raw_generated_text_public=False public_artifact_safe=True",
            stdout.getvalue(),
        )
        self.assertIn(
            "  local_output: available=True display_only=True public_artifact_safe=False count=1 source=none",
            stdout.getvalue(),
        )
        self.assertLess(stdout.getvalue().index("  answer: local text only"), stdout.getvalue().index("  local_output: "))
        self.assertLess(stdout.getvalue().index("  answer: local text only"), stdout.getvalue().index("  trace: "))
        self.assertIn(
            f"  saved_summary: {output_dir / 'infer_summary.json'} markdown={output_dir / 'infer_summary.md'} raw_generated_text_redacted=True public_artifact_safe=True",
            stdout.getvalue(),
        )
        self.assertIn(
            f"  source_summary: {output_dir / 'generate' / 'generate_summary.json'} markdown={output_dir / 'generate' / 'generate_summary.md'} public_artifact_safe=True",
            stdout.getvalue(),
        )
        self.assertIn(
            f"  artifacts: inspect={output_dir / 'infer_summary.md'} json={output_dir / 'infer_summary.json'} markdown={output_dir / 'infer_summary.md'} present=2/4 public_artifact_safe=True",
            stdout.getvalue(),
        )
        self.assertIn("Raw generated text is shown only in local human output", stdout.getvalue())
        self.assertIn("next[1] check existing swarm", stdout.getvalue())
        self.assertIn("next[2] rerun inference", stdout.getvalue())
        self.assertIn(
            "recommended_next: rerun inference reason=rerun_inference CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor infer '<prompt>' --mode existing",
            stdout.getvalue(),
        )
        self.assertIn("  recommended_reason: Rerun the inference request.", stdout.getvalue())
        self.assertIn("CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor infer", stdout.getvalue())
        self.assertIn("# requires CROWDTENSOR_ADMIN_TOKEN", stdout.getvalue())
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --include-output --max-new-tokens 8 --dry-run --coordinator-url http://127.0.0.1:8787 --observer-token ${{CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}}",
            next_lines,
        )
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --include-output --max-new-tokens 8 --coordinator-url http://127.0.0.1:8787",
            next_lines,
        )
        self.assertNotIn("CrowdTensor user prompt", json.dumps(report["next_commands"], sort_keys=True))
        self.assertTrue(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertTrue(report["output_request"]["public_artifact_safe"])
        self.assertEqual(report["output_display"]["terminal_display"], "local-private")
        self.assertTrue(report["output_display"]["terminal_text_available"])
        self.assertTrue(report["output_display"]["include_output_requested"])
        self.assertFalse(report["output_display"]["raw_generated_text_public"])
        self.assertTrue(report["output_display"]["public_artifact_safe"])
        self.assertEqual(report["saved_summary"]["path"], str(output_dir / "infer_summary.json"))
        self.assertEqual(report["saved_summary"]["markdown_path"], str(output_dir / "infer_summary.md"))
        self.assertTrue(report["saved_summary"]["raw_generated_text_redacted"])
        self.assertTrue(report["saved_summary"]["public_artifact_safe"])
        self.assertTrue(report["artifacts"]["infer_summary_markdown"]["present"])
        self.assertEqual(report["source_report"]["summary_path"], str(output_dir / "generate" / "generate_summary.json"))
        self.assertEqual(report["source_report"]["summary_markdown_path"], str(output_dir / "generate" / "generate_summary.md"))
        self.assertTrue(report["source_report"]["public_artifact_safe"])
        self.assertEqual(report["artifact_summary"]["inspect_first"], str(output_dir / "infer_summary.md"))
        self.assertEqual(report["artifact_summary"]["summary_json"], str(output_dir / "infer_summary.json"))
        self.assertEqual(report["artifact_summary"]["summary_markdown"], str(output_dir / "infer_summary.md"))
        self.assertEqual(report["artifact_summary"]["source_summary_json"], str(output_dir / "generate" / "generate_summary.json"))
        self.assertEqual(report["artifact_summary"]["source_summary_markdown"], str(output_dir / "generate" / "generate_summary.md"))
        self.assertEqual(report["artifact_summary"]["artifact_count"], 4)
        self.assertEqual(report["artifact_summary"]["present_artifact_count"], 2)
        self.assertTrue(report["artifact_summary"]["public_artifact_safe"])
        self.assertEqual(report["review_summary"]["state"], "completed")
        self.assertEqual(report["review_summary"]["next_step"], "rerun_or_review_artifacts")
        self.assertEqual(report["review_summary"]["inspect_first"], str(output_dir / "infer_summary.md"))
        self.assertEqual(report["review_summary"]["recommended_label"], "rerun inference")
        self.assertEqual(report["review_summary"]["primary_code"], "crowdtensor_infer_ready")
        self.assertEqual(report["review_summary"]["attention"], "")
        self.assertEqual(report["review_summary"]["attention_detail"], "")
        self.assertEqual(report["recommended_next_command"]["label"], "rerun inference")
        self.assertNotIn("source_label", report["recommended_next_command"])
        self.assertEqual(report["recommended_next_command"]["reason_detail"], "Rerun the inference request.")
        self.assertIn(cli.INFER_PROMPT_PLACEHOLDER, report["review_summary"]["next_command"])
        self.assertEqual(report["review_summary"]["requires_env"], ["CROWDTENSOR_ADMIN_TOKEN"])
        self.assertTrue(report["review_summary"]["has_recommended_command"])
        self.assertTrue(report["review_summary"]["public_artifact_safe"])
        self.assertEqual(report["artifacts"]["source_generate_summary"]["path"], "generate/generate_summary.json")
        self.assertTrue(report["artifacts"]["source_generate_summary"]["present"] is False)
        self.assertEqual(report["artifacts"]["source_generate_summary_markdown"]["path"], "generate/generate_summary.md")
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(report["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(report["shareable_summary"]["raw_generated_text_public"])
        self.assertFalse(report["shareable_summary"]["generated_token_ids_public"])
        self.assertTrue(report["shareable_summary"]["local_output_display_only"])
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "terminal-visible")
        self.assertTrue(report["shareable_summary"]["local_answer_terminal_only"])
        self.assertEqual(report["local_output"]["generated_text"], "local text only")
        self.assertFalse(report["local_output"]["public_artifact_safe"])
        self.assertEqual(
            report["local_output_note"],
            "Raw generated text is shown only in local human output; JSON and saved artifacts expose hashes only.",
        )
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["wait_progress"]["max_observed_token_count"], 16)
        self.assertEqual(persisted["saved_summary"]["path"], str(output_dir / "infer_summary.json"))
        self.assertEqual(persisted["saved_summary"]["markdown_path"], str(output_dir / "infer_summary.md"))
        self.assertTrue(persisted["saved_summary"]["raw_generated_text_redacted"])
        self.assertTrue(persisted["saved_summary"]["public_artifact_safe"])
        self.assertTrue(persisted["artifacts"]["infer_summary_markdown"]["present"])
        self.assertEqual(persisted["source_report"]["summary_path"], str(output_dir / "generate" / "generate_summary.json"))
        self.assertEqual(persisted["source_report"]["summary_markdown_path"], str(output_dir / "generate" / "generate_summary.md"))
        self.assertEqual(persisted["artifact_summary"]["inspect_first"], str(output_dir / "infer_summary.md"))
        self.assertEqual(persisted["artifact_summary"]["source_summary_markdown"], str(output_dir / "generate" / "generate_summary.md"))
        self.assertTrue(persisted["artifact_summary"]["public_artifact_safe"])
        self.assertEqual(persisted["review_summary"]["state"], "completed")
        self.assertEqual(persisted["review_summary"]["inspect_first"], str(output_dir / "infer_summary.md"))
        self.assertEqual(persisted["review_summary"]["attention_detail"], "")
        self.assertEqual(persisted["recommended_next_command"]["reason_detail"], "Rerun the inference request.")
        self.assertIn(cli.INFER_PROMPT_PLACEHOLDER, persisted["review_summary"]["next_command"])
        self.assertNotIn("CrowdTensor user prompt", persisted["review_summary"]["next_command"])
        self.assertTrue(persisted["review_summary"]["public_artifact_safe"])
        self.assertEqual(persisted["artifacts"]["source_generate_summary"]["path"], "generate/generate_summary.json")
        self.assertEqual(persisted["artifacts"]["source_generate_summary_markdown"]["path"], "generate/generate_summary.md")
        self.assertTrue(persisted["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(persisted["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(persisted["shareable_summary"]["raw_generated_text_public"])
        self.assertFalse(persisted["shareable_summary"]["generated_token_ids_public"])
        self.assertTrue(persisted["shareable_summary"]["local_output_display_only"])
        self.assertEqual(persisted["shareable_summary"]["answer_scope_state"], "saved-terminal-redacted")
        self.assertFalse(persisted["shareable_summary"]["local_answer_terminal_only"])
        self.assertTrue(persisted["output_request"]["include_output"])
        self.assertFalse(persisted["output_request"]["raw_prompt_public"])
        self.assertFalse(persisted["output_request"]["raw_generated_text_public"])
        self.assertFalse(persisted["output_request"]["generated_token_ids_public"])
        self.assertTrue(persisted["output_request"]["public_artifact_safe"])
        self.assertEqual(persisted["local_output"]["generated_text"], "")
        self.assertFalse(persisted["local_output"]["display_only"])
        self.assertTrue(persisted["local_output"]["public_artifact_safe"])
        self.assertEqual(persisted["result"]["status"], "complete")
        self.assertEqual(persisted["result"]["output_count"], 1)
        self.assertEqual(persisted["result"]["display"], "hash-only")
        self.assertTrue(persisted["result"]["public_artifact_safe"])
        self.assertEqual(persisted["output_display"]["terminal_display"], "saved-terminal-redacted")
        self.assertFalse(persisted["output_display"]["terminal_text_available"])
        self.assertTrue(persisted["output_display"]["include_output_requested"])
        self.assertFalse(persisted["output_display"]["raw_generated_text_public"])
        self.assertTrue(persisted["output_display"]["public_artifact_safe"])
        self.assertEqual(persisted["user_status"]["state"], "completed")
        self.assertEqual(persisted["user_status"]["headline"], "Inference completed.")
        self.assertEqual(persisted["user_status"]["next_step"], "rerun_or_review_artifacts")
        self.assertEqual(persisted["user_status"]["recommended_label"], "rerun inference")
        self.assertTrue(persisted["user_status"]["public_artifact_safe"])
        self.assertEqual(persisted["issue_summary"]["state"], "completed")
        self.assertEqual(persisted["issue_summary"]["primary_code"], "crowdtensor_infer_ready")
        self.assertEqual(persisted["issue_summary"]["next_step"], "rerun_or_review_artifacts")
        self.assertFalse(persisted["issue_summary"]["safe_detail_present"])
        self.assertEqual(persisted["trace"]["session_id"], "real-llm-session-infer")
        self.assertEqual(persisted["trace"]["workload_type"], "real_llm_sharded_infer")
        self.assertEqual(persisted["trace"]["accepted_rows_seen"], 1)
        self.assertEqual(persisted["trace"]["request_trace"][0]["request_id"], "req-1")
        self.assertEqual(persisted["trace"]["request_trace"][0]["prompt_hash"], "sha256:p1")
        self.assertFalse(persisted["trace"]["raw_prompt_public"])
        self.assertFalse(persisted["trace"]["raw_generated_text_public"])
        self.assertTrue(persisted["trace"]["public_artifact_safe"])
        self.assertEqual(persisted["recommended_next_command"]["label"], "rerun inference")
        self.assertNotIn("source_label", persisted["recommended_next_command"])
        self.assertEqual(persisted["recommended_next_command"]["reason"], "rerun_inference")
        self.assertEqual(persisted["recommended_next_command"]["requires_env"], ["CROWDTENSOR_ADMIN_TOKEN"])
        self.assertIn(cli.INFER_PROMPT_PLACEHOLDER, persisted["recommended_next_command"]["command_line"])
        self.assertNotIn("CrowdTensor user prompt", json.dumps(persisted["recommended_next_command"], sort_keys=True))
        self.assertNotIn("must not leak", json.dumps(persisted, sort_keys=True))
        self.assertNotIn('"generated_token_ids": [1, 2]', json.dumps(persisted, sort_keys=True))
        self.assertEqual(
            persisted["local_output_note"],
            "Raw generated text is shown only in local human output; JSON and saved artifacts expose hashes only.",
        )
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("# CrowdTensor Infer Summary", markdown)
        self.assertIn("- OK: `True`", markdown)
        self.assertIn("- Mode: `existing`", markdown)
        self.assertLess(markdown.index("- Review: "), markdown.index("- OK: "))
        self.assertLess(markdown.index("- Review: "), markdown.index("- Status: "))
        self.assertIn(f"- Inspect first: `{output_dir / 'infer_summary.md'}`", markdown)
        self.assertLess(markdown.index("- Review next: "), markdown.index("- Inspect first: "))
        self.assertLess(markdown.index("- Inspect first: "), markdown.index("- Status: "))
        self.assertLess(markdown.index("- Status: "), markdown.index("- Issue: "))
        self.assertLess(markdown.index("- Issue: "), markdown.index("- OK: "))
        self.assertIn("## What To Do Next", markdown)
        self.assertIn("- State: `completed`", markdown)
        self.assertIn("- Next step: `rerun_or_review_artifacts`", markdown)
        self.assertIn("- Recommended: `rerun inference` reason=`rerun_inference`", markdown)
        self.assertIn("- Reason: Rerun the inference request.", markdown)
        self.assertIn("- Copy command: `CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor infer '<prompt>' --mode existing", markdown)
        self.assertIn(
            "- Prompt input: saved Markdown keeps `<prompt>` placeholders; terminal `review_next` / `recommended_next` render safe local prompt sources for copy/paste when available, and saved commands should prefer `--prompt-file`, `--prompt-stdin`, or `--prompt-texts-file`.",
            markdown,
        )
        self.assertIn(
            "- Terminal prompt scope: human terminal `review_next`, `recommended_next`, and `next[...]` may render inline local prompts for copy/paste. Treat terminal logs as local-private; saved JSON/Markdown keep placeholders.",
            markdown,
        )
        self.assertIn("- Requires env: `CROWDTENSOR_ADMIN_TOKEN`", markdown)
        self.assertIn("- Safety: saved Markdown keeps prompt placeholders and redacted generated output.", markdown)
        self.assertIn(f"- Safety: saved Markdown keeps prompt placeholders and redacted generated output. {cli.SAVED_TERMINAL_ANSWER_SCOPE_TEXT}", markdown)
        self.assertIn("## Details", markdown)
        self.assertIn(
            "- Status: `completed: Inference completed. next=rerun_or_review_artifacts recommendation=rerun inference public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            f"- Review: `state=completed next=rerun_or_review_artifacts inspect={output_dir / 'infer_summary.md'} recommended=rerun inference primary=crowdtensor_infer_ready attention=none public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Review next: `label=rerun inference reason=rerun_inference command=CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor infer '<prompt>' --mode existing",
            markdown,
        )
        self.assertIn(
            "- Issue: `state=completed primary=crowdtensor_infer_ready next=rerun_or_review_artifacts progress=",
            markdown,
        )
        self.assertIn("- Model: `sshleifer/tiny-gpt2` backend=`cpu`", markdown)
        self.assertIn("- Prompt: `count=1 hash=", markdown)
        self.assertIn("raw_public=False`", markdown)
        self.assertIn("- Generation: `16/16` hash=`sha256:generated`", markdown)
        self.assertIn(
            "- Result: `status=complete tokens=16/16 outputs=1 display=hash-only hash=sha256:generated public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Output display: `terminal=saved-terminal-redacted terminal_text=False saved=hash-only json_stdout=hash-only-json include_output=True raw_public=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Trace: `session=real-llm-session-infer requests=1 ledger_rows=1 stream_events=0 source=public_swarm_product_cli_v1 public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=True answer_scope_state=saved-terminal-redacted local_answer_terminal_only=False`",
            markdown,
        )
        self.assertIn(
            "- Recommended next: `rerun inference` reason=`rerun_inference` command=`CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor infer '<prompt>' --mode existing",
            markdown,
        )
        self.assertIn("requires=`CROWDTENSOR_ADMIN_TOKEN`", markdown)
        self.assertIn("- Wait: `polls=2 accepted_rows=1 tokens=16/16 ledger=True stream=False`", markdown)
        self.assertIn(
            "- Local output: `available=False display_only=False public_artifact_safe=True saved_redacted=True` count=`1` source=``",
            markdown,
        )
        self.assertIn(
            "- Local output note: Raw generated text is shown only in local human output; JSON and saved artifacts expose hashes only.",
            markdown,
        )
        self.assertIn(
            f"- Source generate summary: json=`{output_dir / 'generate' / 'generate_summary.json'}` markdown=`{output_dir / 'generate' / 'generate_summary.md'}`",
            markdown,
        )
        self.assertIn(
            f"- Artifacts: `inspect={output_dir / 'infer_summary.md'} json={output_dir / 'infer_summary.json'} markdown={output_dir / 'infer_summary.md'} present=2/4 public_artifact_safe=True`",
            markdown,
        )
        self.assertIn("## Artifacts", markdown)
        self.assertIn("- `infer_summary_markdown`: path=`infer_summary.md` present=`True` kind=`crowdtensor_infer_summary_markdown`", markdown)
        self.assertIn(
            "- `source_generate_summary`: path=`generate/generate_summary.json` present=`False` kind=`crowdtensor_generate_summary`",
            markdown,
        )
        self.assertIn("Raw generated text and generated token ids are redacted", markdown)
        self.assertIn("`rerun inference`", markdown)
        self.assertNotIn("`submit inference`", markdown)
        self.assertIn("Prompt placeholder `<prompt>` is redacted. To rerun safely", markdown)
        self.assertIn("replace the placeholder with `--prompt-file prompt.txt`", markdown)
        self.assertIn("`printf %s '<prompt>' | ... --prompt-stdin`", markdown)
        self.assertIn("do not paste private prompt text into saved commands", markdown)
        self.assertIn("Set required environment variables before running commands: `CROWDTENSOR_ADMIN_TOKEN, CROWDTENSOR_OBSERVER_TOKEN`.", markdown)
        self.assertNotIn("local text only", markdown)
        self.assertNotIn("CrowdTensor user prompt", markdown)

    def test_infer_main_prints_copyable_local_prompt_without_persisting_it(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "CrowdTensor user prompt"

        def fake_build_infer(args: object) -> dict[str, object]:
            return {
                "schema": "crowdtensor_infer_cli_v1",
                "ok": True,
                "mode": "existing",
                "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
                "generation": {"generated_token_count": 8, "max_new_tokens": 8, "generated_text_hash": "sha256:generated"},
                "route": {"route_source": "coordinator-url", "route_ready": True, "distinct_stage_miners": True},
                "stream": {},
                "local_output": {},
                "output_dir": str(output_dir),
                "next_commands": [
                    cli.command_entry(
                        "check existing swarm",
                        [
                            "crowdtensor",
                            "infer",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--mode",
                            "existing",
                            "--output-dir",
                            str(output_dir),
                            "--dry-run",
                        ],
                    )
                ],
                "recommended_next_command": {
                    **cli.command_entry(
                        "check existing swarm",
                        [
                            "crowdtensor",
                            "infer",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--mode",
                            "existing",
                            "--output-dir",
                            str(output_dir),
                            "--dry-run",
                        ],
                    ),
                    "reason": "verify_stage_miners",
                    "source_index": 1,
                },
                "review_summary": {
                    "state": "preflight-ready",
                    "next_step": "submit",
                    "recommended_label": "check existing swarm",
                    "recommended_reason": "verify_stage_miners",
                    "next_command": f"crowdtensor infer '<prompt>' --mode existing --output-dir {output_dir} --dry-run",
                    "public_artifact_safe": True,
                },
                "diagnosis_codes": ["crowdtensor_infer_ready"],
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli, "build_infer", side_effect=fake_build_infer):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                cli.main([
                    "infer",
                    prompt,
                    "--mode",
                    "existing",
                    "--coordinator-url",
                    "http://127.0.0.1:8787",
                    "--admin-token",
                    "admin-secret",
                    "--output-dir",
                    str(output_dir),
                ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        self.assertIn(
            "prompt_scope: terminal_next_commands=local-private inline_prompt_text=True terminal_local_paths=False saved_artifacts=prompt-placeholders prefer_prompt_file_or_stdin_for_shareable_logs=True",
            rendered,
        )
        self.assertIn("source=prompt-text", rendered)
        self.assertLess(rendered.index("  prompt_scope: "), rendered.index("  review_next: "))
        self.assertIn(f"review_next: label=check existing swarm reason=verify_stage_miners command=crowdtensor infer '{prompt}' --mode existing", rendered)
        self.assertIn(f"recommended_next: check existing swarm reason=verify_stage_miners crowdtensor infer '{prompt}' --mode existing", rendered)
        self.assertIn(f"next[1] check existing swarm: crowdtensor infer '{prompt}' --mode existing", rendered)
        self.assertNotIn(cli.INFER_PROMPT_PLACEHOLDER, rendered)

    def test_infer_main_prints_prompt_file_without_expanding_prompt(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "Infer file private text"
        prompt_file = output_dir / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        def fake_build_infer(args: object) -> dict[str, object]:
            self.assertEqual(getattr(args, "prompt_text"), prompt)
            self.assertEqual(getattr(args, "prompt_file"), str(prompt_file))
            return {
                "schema": "crowdtensor_infer_cli_v1",
                "ok": True,
                "mode": "existing",
                "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
                "generation": {"generated_token_count": 8, "max_new_tokens": 8, "generated_text_hash": "sha256:generated"},
                "route": {"route_source": "coordinator-url", "route_ready": True, "distinct_stage_miners": True},
                "stream": {},
                "local_output": {},
                "output_dir": str(output_dir),
                "next_commands": [
                    cli.command_entry(
                        "check existing swarm",
                        [
                            "crowdtensor",
                            "infer",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--mode",
                            "existing",
                            "--output-dir",
                            str(output_dir),
                            "--dry-run",
                        ],
                    )
                ],
                "recommended_next_command": {
                    **cli.command_entry(
                        "check existing swarm",
                        [
                            "crowdtensor",
                            "infer",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--mode",
                            "existing",
                            "--output-dir",
                            str(output_dir),
                            "--dry-run",
                        ],
                    ),
                    "reason": "verify_stage_miners",
                    "source_index": 1,
                },
                "review_summary": {
                    "state": "preflight-ready",
                    "next_step": "submit",
                    "recommended_label": "check existing swarm",
                    "recommended_reason": "verify_stage_miners",
                    "next_command": f"crowdtensor infer '<prompt>' --mode existing --output-dir {output_dir} --dry-run",
                    "public_artifact_safe": True,
                },
                "diagnosis_codes": ["crowdtensor_infer_ready"],
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli, "build_infer", side_effect=fake_build_infer):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                cli.main([
                    "infer",
                    "--prompt-file",
                    str(prompt_file),
                    "--mode",
                    "existing",
                    "--coordinator-url",
                    "http://127.0.0.1:8787",
                    "--admin-token",
                    "admin-secret",
                    "--output-dir",
                    str(output_dir),
                ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        progress = stderr.getvalue()
        self.assertIn(f"--prompt-file {prompt_file}", rendered)
        self.assertIn(
            "prompt_scope: terminal_next_commands=local-private inline_prompt_text=False terminal_local_paths=True saved_artifacts=prompt-placeholders prefer_prompt_file_or_stdin_for_shareable_logs=True source=prompt-file prompt_file_path_public=False raw_prompt_public=False",
            rendered,
        )
        self.assertIn("terminal_local_paths=True", rendered)
        self.assertNotIn("infer '<prompt>'", rendered)
        self.assertNotIn(prompt, rendered)
        self.assertNotIn(prompt, progress)

    def test_infer_shareable_terminal_hides_inline_prompt_and_answer(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "CrowdTensor private infer prompt"
        answer = "local infer answer"

        def fake_build_infer(args: object) -> dict[str, object]:
            self.assertTrue(getattr(args, "shareable_terminal"))
            return {
                "schema": "crowdtensor_infer_cli_v1",
                "ok": True,
                "mode": "existing",
                "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
                "prompt": {"prompt_count": 1, "prompt_hash": "sha256:prompt", "raw_prompt_public": False},
                "generation": {"generated_token_count": 2, "max_new_tokens": 2, "generated_text_hash": "sha256:generated"},
                "result": {
                    "status": "complete",
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "output_count": 1,
                    "display": "local-private",
                    "generated_text_hash": "sha256:generated",
                    "public_artifact_safe": False,
                },
                "output_display": {
                    "terminal_display": "local-private",
                    "terminal_text_available": True,
                    "saved_artifact_display": "hash-only",
                    "json_stdout_display": "hash-only-json",
                    "raw_generated_text_public": False,
                    "public_artifact_safe": True,
                },
                "answer_scope": {
                    "scope_state": "terminal-visible",
                    "terminal_only": True,
                    "visible_in_terminal": True,
                    "saved_json_display": "hash-only",
                    "saved_markdown_display": "hash-only",
                    "public_artifact_safe": True,
                },
                "local_output": {
                    "available": True,
                    "generated_text": answer,
                    "output_count": 1,
                    "source": "local-private-task-state",
                    "display_only": True,
                    "public_artifact_safe": False,
                },
                "trace": {"request_count": 1, "public_artifact_safe": True},
                "shareable_summary": {
                    "saved_artifacts_public_safe": True,
                    "raw_prompt_public": False,
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                    "local_output_display_only": True,
                    "answer_scope_state": "terminal-visible",
                    "local_answer_terminal_only": True,
                },
                "route": {"route_source": "coordinator-url", "route_ready": True, "distinct_stage_miners": True},
                "stream": {},
                "output_dir": str(output_dir),
                "next_commands": [
                    cli.command_entry(
                        "rerun inference",
                        ["crowdtensor", "infer", cli.INFER_PROMPT_PLACEHOLDER, "--mode", "existing"],
                    )
                ],
                "recommended_next_command": {
                    **cli.command_entry(
                        "rerun inference",
                        ["crowdtensor", "infer", cli.INFER_PROMPT_PLACEHOLDER, "--mode", "existing"],
                    ),
                    "reason": "rerun_inference",
                },
                "review_summary": {
                    "state": "completed",
                    "next_step": "rerun_or_review_artifacts",
                    "recommended_label": "rerun inference",
                    "recommended_reason": "rerun_inference",
                    "next_command": "crowdtensor infer '<prompt>' --mode existing",
                    "public_artifact_safe": True,
                },
                "diagnosis_codes": ["crowdtensor_infer_ready"],
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli, "build_infer", side_effect=fake_build_infer):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                cli.main([
                    "infer",
                    prompt,
                    "--mode",
                    "existing",
                    "--coordinator-url",
                    "http://127.0.0.1:8787",
                    "--admin-token",
                    "admin-secret",
                    "--output-dir",
                    str(output_dir),
                    "--shareable-terminal",
                ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        progress = stderr.getvalue()
        self.assertNotIn(prompt, rendered)
        self.assertNotIn(answer, rendered)
        self.assertNotIn("admin-secret", rendered)
        self.assertNotIn("  answer:", rendered)
        self.assertNotIn("prompt_scope:", rendered)
        self.assertIn("command=crowdtensor infer '<prompt>' --mode existing", rendered)
        self.assertIn("next[1] rerun inference: crowdtensor infer '<prompt>' --mode existing", rendered)
        self.assertIn(
            "  shareable_terminal: enabled=True prompt_sources_redacted=True answer_text_redacted=True public_artifact_safe=True",
            rendered,
        )
        self.assertIn("answer_scope: state=shareable-terminal-redacted", rendered)
        self.assertIn(f"answer_scope_note: {cli.SHAREABLE_TERMINAL_ANSWER_SCOPE_TEXT}", rendered)
        self.assertIn("output_display: terminal=shareable-terminal-redacted", rendered)
        self.assertIn(f"output_display_note: {cli.SHAREABLE_TERMINAL_OUTPUT_DISPLAY_SCOPE_TEXT}", rendered)
        self.assertIn("result: status=complete tokens=2/2 outputs=1 display=hash-only hash=sha256:generated public_artifact_safe=True", rendered)
        self.assertIn("local_output: available=False display_only=False public_artifact_safe=True saved_redacted=True", rendered)
        self.assertNotIn(prompt, progress)
        self.assertNotIn(answer, progress)
        self.assertNotIn("admin-secret", progress)

    def test_infer_shareable_terminal_keeps_safe_stdin_next_command(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "Infer stdin private text"

        def fake_build_infer(args: object) -> dict[str, object]:
            self.assertTrue(getattr(args, "shareable_terminal"))
            self.assertTrue(getattr(args, "prompt_stdin"))
            self.assertEqual(getattr(args, "prompt_text"), prompt)
            return {
                "schema": "crowdtensor_infer_cli_v1",
                "ok": True,
                "mode": "existing",
                "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
                "generation": {"generated_token_count": 0, "max_new_tokens": 2, "generated_text_hash": ""},
                "result": {"status": "preflight-ready", "output_count": 0, "display": "hash-only", "public_artifact_safe": True},
                "route": {"route_source": "coordinator-url", "route_ready": True, "distinct_stage_miners": True},
                "stream": {},
                "local_output": {},
                "output_dir": str(output_dir),
                "next_commands": [
                    cli.command_entry(
                        "check existing swarm",
                        ["crowdtensor", "infer", cli.INFER_PROMPT_PLACEHOLDER, "--mode", "existing", "--dry-run"],
                    )
                ],
                "recommended_next_command": {
                    **cli.command_entry(
                        "check existing swarm",
                        ["crowdtensor", "infer", cli.INFER_PROMPT_PLACEHOLDER, "--mode", "existing", "--dry-run"],
                    ),
                    "reason": "verify_stage_miners",
                },
                "review_summary": {
                    "state": "preflight-ready",
                    "next_step": "submit",
                    "recommended_label": "check existing swarm",
                    "recommended_reason": "verify_stage_miners",
                    "next_command": "crowdtensor infer '<prompt>' --mode existing --dry-run",
                    "public_artifact_safe": True,
                },
                "diagnosis_codes": ["crowdtensor_infer_ready"],
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli.sys, "stdin", io.StringIO(prompt + "\n")):
            with patch.object(cli, "build_infer", side_effect=fake_build_infer):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    cli.main([
                        "infer",
                        "--prompt-stdin",
                        "--mode",
                        "existing",
                        "--coordinator-url",
                        "http://127.0.0.1:8787",
                        "--admin-token",
                        "admin-secret",
                        "--output-dir",
                        str(output_dir),
                        "--dry-run",
                        "--shareable-terminal",
                    ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        progress = stderr.getvalue()
        self.assertIn("printf %s '<prompt>' | crowdtensor infer --prompt-stdin --mode existing --dry-run", rendered)
        self.assertNotIn("infer '<prompt>'", rendered)
        self.assertNotIn(prompt, rendered)
        self.assertNotIn(prompt, progress)
        self.assertNotIn("admin-secret", rendered)
        self.assertIn(
            "  shareable_terminal: enabled=True prompt_sources_redacted=True answer_text_redacted=False public_artifact_safe=True",
            rendered,
        )

    def test_infer_shareable_terminal_next_commands_preserve_shareable_mode(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "zeta-alpha-infer-needle-927"
        args = cli.parse_args([
            "infer",
            prompt,
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--dry-run",
            "--skip-live-preflight",
            "--shareable-terminal",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        with patch.object(
            cli,
            "request_json_url",
            side_effect=AssertionError("skip-live-preflight should not touch live Coordinator endpoints"),
        ):
            report = cli.build_infer(args)

        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertTrue(next_lines, report)
        self.assertTrue(all("--shareable-terminal" in line for line in next_lines), next_lines)
        self.assertIn("--shareable-terminal", report["recommended_next_command"]["command_line"])
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        persisted_lines = [item["command_line"] for item in persisted["next_commands"]]
        self.assertTrue(all("--shareable-terminal" in line for line in persisted_lines), persisted_lines)
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("--shareable-terminal", markdown)
        self.assertNotIn(prompt, json.dumps(report, sort_keys=True))
        self.assertNotIn(prompt, markdown)

    def test_infer_prompt_file_does_not_pollute_startup_commands(self) -> None:
        prompt_file = Path(self._tmp_dir()) / "prompt.txt"
        prompt_file.write_text("Infer file private text", encoding="utf-8")
        report = {
            "local_prompt_file": str(prompt_file),
            "next_commands": [
                cli.command_entry("start Coordinator", ["crowdtensor", "serve", "--run"]),
                cli.command_entry("check existing swarm", ["crowdtensor", "infer", cli.INFER_PROMPT_PLACEHOLDER, "--mode", "existing"]),
            ],
        }

        self.assertEqual(
            cli.local_infer_command_line(report["next_commands"][0], report),
            "crowdtensor serve --run",
        )
        self.assertEqual(
            cli.local_infer_command_line(report["next_commands"][1], report),
            f"crowdtensor infer --prompt-file {prompt_file} --mode existing",
        )

    def test_infer_main_prints_prompt_texts_file_without_expanding_prompts(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompts = ["Infer file prompt, with comma", "Second infer file prompt"]
        prompt_file = output_dir / "prompts.txt"
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")

        def fake_build_infer(args: object) -> dict[str, object]:
            self.assertEqual(getattr(args, "prompt_texts_file"), str(prompt_file))
            self.assertEqual(getattr(args, "prompt_texts_list"), prompts)
            return {
                "schema": "crowdtensor_infer_cli_v1",
                "ok": True,
                "mode": "existing",
                "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
                "generation": {"generated_token_count": 8, "max_new_tokens": 8, "generated_text_hash": "sha256:generated"},
                "route": {"route_source": "coordinator-url", "route_ready": True, "distinct_stage_miners": True},
                "stream": {},
                "local_output": {},
                "output_dir": str(output_dir),
                "next_commands": [
                    cli.command_entry(
                        "check existing swarm",
                        [
                            "crowdtensor",
                            "infer",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--mode",
                            "existing",
                            "--output-dir",
                            str(output_dir),
                            "--prompt-texts",
                            cli.INFER_BATCH_PROMPTS_PLACEHOLDER,
                            "--dry-run",
                        ],
                    )
                ],
                "recommended_next_command": {
                    **cli.command_entry(
                        "check existing swarm",
                        [
                            "crowdtensor",
                            "infer",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--mode",
                            "existing",
                            "--output-dir",
                            str(output_dir),
                            "--prompt-texts",
                            cli.INFER_BATCH_PROMPTS_PLACEHOLDER,
                            "--dry-run",
                        ],
                    ),
                    "reason": "verify_stage_miners",
                    "source_index": 1,
                },
                "review_summary": {
                    "state": "preflight-ready",
                    "next_step": "submit",
                    "recommended_label": "check existing swarm",
                    "recommended_reason": "verify_stage_miners",
                    "next_command": f"crowdtensor infer '<prompt>' --mode existing --output-dir {output_dir} --prompt-texts '<prompt-1>,<prompt-2>' --dry-run",
                    "public_artifact_safe": True,
                },
                "diagnosis_codes": ["crowdtensor_infer_ready"],
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli, "build_infer", side_effect=fake_build_infer):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                cli.main([
                    "infer",
                    "--prompt-texts-file",
                    str(prompt_file),
                    "--mode",
                    "existing",
                    "--coordinator-url",
                    "http://127.0.0.1:8787",
                    "--admin-token",
                    "admin-secret",
                    "--output-dir",
                    str(output_dir),
                ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        progress = stderr.getvalue()
        self.assertIn(f"--prompt-texts-file {prompt_file}", rendered)
        self.assertIn(
            "prompt_scope: terminal_next_commands=local-private inline_prompt_text=False terminal_local_paths=True saved_artifacts=prompt-placeholders prefer_prompt_file_or_stdin_for_shareable_logs=True source=prompt-texts-file prompt_file_path_public=False raw_prompt_public=False",
            rendered,
        )
        self.assertIn("terminal_local_paths=True", rendered)
        self.assertNotIn("infer '<prompt>'", rendered)
        self.assertNotIn("--prompt-texts '<prompt-1>,<prompt-2>'", rendered)
        for prompt in prompts:
            self.assertNotIn(prompt, rendered)
            self.assertNotIn(prompt, progress)

    def test_infer_main_prints_prompt_stdin_without_expanding_prompt(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "Infer stdin private text"

        def fake_build_infer(args: object) -> dict[str, object]:
            self.assertEqual(getattr(args, "prompt_text"), prompt)
            self.assertTrue(getattr(args, "prompt_stdin"))
            return {
                "schema": "crowdtensor_infer_cli_v1",
                "ok": True,
                "mode": "existing",
                "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
                "generation": {"generated_token_count": 8, "max_new_tokens": 8, "generated_text_hash": "sha256:generated"},
                "route": {"route_source": "coordinator-url", "route_ready": True, "distinct_stage_miners": True},
                "stream": {},
                "local_output": {},
                "output_dir": str(output_dir),
                "next_commands": [
                    cli.command_entry(
                        "check existing swarm",
                        [
                            "crowdtensor",
                            "infer",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--mode",
                            "existing",
                            "--output-dir",
                            str(output_dir),
                            "--dry-run",
                        ],
                    )
                ],
                "recommended_next_command": {
                    **cli.command_entry(
                        "check existing swarm",
                        [
                            "crowdtensor",
                            "infer",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--mode",
                            "existing",
                            "--output-dir",
                            str(output_dir),
                            "--dry-run",
                        ],
                    ),
                    "reason": "verify_stage_miners",
                    "source_index": 1,
                },
                "review_summary": {
                    "state": "preflight-ready",
                    "next_step": "submit",
                    "recommended_label": "check existing swarm",
                    "recommended_reason": "verify_stage_miners",
                    "next_command": f"crowdtensor infer '<prompt>' --mode existing --output-dir {output_dir} --dry-run",
                    "public_artifact_safe": True,
                },
                "diagnosis_codes": ["crowdtensor_infer_ready"],
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli.sys, "stdin", io.StringIO(prompt + "\n")):
            with patch.object(cli, "build_infer", side_effect=fake_build_infer):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    cli.main([
                        "infer",
                        "--prompt-stdin",
                        "--mode",
                        "existing",
                        "--coordinator-url",
                        "http://127.0.0.1:8787",
                        "--admin-token",
                        "admin-secret",
                        "--output-dir",
                        str(output_dir),
                    ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        progress = stderr.getvalue()
        self.assertIn("printf %s '<prompt>' | crowdtensor infer", rendered)
        self.assertIn("--prompt-stdin", rendered)
        self.assertIn(
            "prompt_scope: terminal_next_commands=shareable inline_prompt_text=False terminal_local_paths=False saved_artifacts=prompt-placeholders prefer_prompt_file_or_stdin_for_shareable_logs=True source=prompt-stdin prompt_file_path_public=False raw_prompt_public=False",
            rendered,
        )
        self.assertNotIn("terminal_next_commands=local-private", rendered)
        self.assertNotIn("infer '<prompt>'", rendered)
        self.assertNotIn(prompt, rendered)
        self.assertNotIn(prompt, progress)

    def test_infer_prompt_stdin_does_not_pollute_startup_or_env_commands(self) -> None:
        report = {
            "local_prompt_stdin": True,
            "next_commands": [
                cli.command_entry("start Coordinator", ["crowdtensor", "serve", "--run"]),
                cli.command_entry("start stage0 Miner", ["crowdtensor", "join", "--stage", "stage0", "--run"]),
                cli.command_entry(
                    "submit inference",
                    ["crowdtensor", "infer", cli.INFER_PROMPT_PLACEHOLDER, "--mode", "existing"],
                    requires_env=["CROWDTENSOR_ADMIN_TOKEN"],
                ),
            ],
        }

        self.assertEqual(
            cli.local_infer_command_line(report["next_commands"][0], report),
            "crowdtensor serve --run",
        )
        self.assertEqual(
            cli.local_infer_command_line(report["next_commands"][1], report),
            "crowdtensor join --stage stage0 --run",
        )
        rendered_submit = cli.human_next_command_line(
            report["next_commands"][2],
            cli.local_infer_command_line(report["next_commands"][2], report),
        )
        self.assertEqual(
            rendered_submit,
            "printf %s '<prompt>' | CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor infer --prompt-stdin --mode existing",
        )

    def test_infer_main_prints_copyable_local_batch_prompts_without_persisting_them(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt = "first prompt"
        prompt_texts = "first prompt,second prompt"

        def fake_build_infer(args: object) -> dict[str, object]:
            del args
            return {
                "schema": "crowdtensor_infer_cli_v1",
                "ok": True,
                "mode": "existing",
                "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
                "generation": {"generated_token_count": 8, "max_new_tokens": 8, "generated_text_hash": "sha256:generated"},
                "route": {"route_source": "coordinator-url", "route_ready": True, "distinct_stage_miners": True},
                "stream": {},
                "local_output": {},
                "output_dir": str(output_dir),
                "next_commands": [
                    cli.command_entry(
                        "check existing swarm",
                        [
                            "crowdtensor",
                            "infer",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--mode",
                            "existing",
                            "--output-dir",
                            str(output_dir),
                            "--prompt-text",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--prompt-texts",
                            cli.INFER_BATCH_PROMPTS_PLACEHOLDER,
                            "--dry-run",
                        ],
                    )
                ],
                "recommended_next_command": {
                    **cli.command_entry(
                        "check existing swarm",
                        [
                            "crowdtensor",
                            "infer",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--mode",
                            "existing",
                            "--output-dir",
                            str(output_dir),
                            "--prompt-text",
                            cli.INFER_PROMPT_PLACEHOLDER,
                            "--prompt-texts",
                            cli.INFER_BATCH_PROMPTS_PLACEHOLDER,
                            "--dry-run",
                        ],
                    ),
                    "reason": "verify_stage_miners",
                    "source_index": 1,
                },
                "review_summary": {
                    "state": "preflight-ready",
                    "next_step": "submit",
                    "recommended_label": "check existing swarm",
                    "recommended_reason": "verify_stage_miners",
                    "next_command": f"crowdtensor infer '<prompt>' --mode existing --output-dir {output_dir} --prompt-text '<prompt>' --prompt-texts '<prompt-1>,<prompt-2>' --dry-run",
                    "public_artifact_safe": True,
                },
                "diagnosis_codes": ["crowdtensor_infer_ready"],
            }

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(cli, "build_infer", side_effect=fake_build_infer):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                cli.main([
                    "infer",
                    "--prompt-texts",
                    prompt_texts,
                    "--mode",
                    "existing",
                    "--coordinator-url",
                    "http://127.0.0.1:8787",
                    "--admin-token",
                    "admin-secret",
                    "--output-dir",
                    str(output_dir),
                ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        self.assertIn("prompt_scope: terminal_next_commands=local-private inline_prompt_text=True", rendered)
        self.assertIn(
            f"review_next: label=check existing swarm reason=verify_stage_miners command=crowdtensor infer --mode existing --output-dir {output_dir} --prompt-texts '{prompt_texts}' --dry-run",
            rendered,
        )
        self.assertIn(
            f"recommended_next: check existing swarm reason=verify_stage_miners crowdtensor infer --mode existing --output-dir {output_dir} --prompt-texts '{prompt_texts}' --dry-run",
            rendered,
        )
        self.assertIn(
            f"next[1] check existing swarm: crowdtensor infer --mode existing --output-dir {output_dir} --prompt-texts '{prompt_texts}' --dry-run",
            rendered,
        )
        self.assertNotIn(f"crowdtensor infer '{prompt}' --mode existing", rendered)
        self.assertNotIn("--prompt-text '<prompt>'", rendered)
        self.assertNotIn("--prompt-text 'first prompt'", rendered)
        self.assertNotIn(cli.INFER_BATCH_PROMPTS_PLACEHOLDER, rendered)

    def test_infer_review_next_fallback_cleans_batch_prompt_conflict(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_texts = "first prompt,second prompt"
        report = {
            "review_summary": {
                "recommended_label": "check existing swarm",
                "recommended_reason": "verify_stage_miners",
                "next_command": f"crowdtensor infer '<prompt>' --mode existing --output-dir {output_dir} --prompt-text '<prompt>' --prompt-texts '<prompt-1>,<prompt-2>' --dry-run",
                "public_artifact_safe": True,
            },
            "local_prompt_texts": prompt_texts,
        }

        summary = cli.display_review_summary(report, cli.local_infer_command_line)
        rendered = cli.review_next_command_text(summary)

        self.assertEqual(
            summary["next_command"],
            f"crowdtensor infer --mode existing --output-dir {output_dir} --prompt-texts '{prompt_texts}' --dry-run",
        )
        self.assertNotIn("infer '<prompt>'", rendered)
        self.assertNotIn("--prompt-text '<prompt>'", rendered)

    def test_infer_existing_batch_outputs_are_display_only(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "--prompt-texts",
            "first prompt,second prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
        ])
        generate_payload = {
            "schema": "public_swarm_product_cli_v1",
            "ok": True,
            "mode": "generate",
            "session": {"hf_model_id": "sshleifer/tiny-gpt2"},
            "generation": {
                "generated_token_count": 2,
                "max_new_tokens": 2,
                "generated_text_hash": "sha256:batch",
                "decoded_tokens_match": True,
                "request_count": 2,
                "batch_generation_ready": True,
                "results": [
                    {
                        "request_id": "req-1",
                        "prompt_hash": "sha256:p1",
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:g1",
                    },
                    {
                        "request_id": "req-2",
                        "prompt_hash": "sha256:p2",
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:g2",
                    },
                ],
            },
            "batch": {"enabled": True, "request_count": 2, "batch_generation_ready": True},
            "wait_progress": {
                "poll_count": 2,
                "accepted_rows_seen": 1,
                "max_observed_token_count": 2,
                "target_token_count": 2,
                "expected_request_count": 2,
                "observed_request_count": 2,
                "batch_generation_ready": True,
                "ledger_endpoint_ready": True,
                "stream_endpoint_ready": False,
                "public_artifact_safe": True,
            },
            "route": {"route_source": "coordinator-url", "coordinator_url_present": True},
            "local_output": {
                "generated_text": " first output",
                "outputs": [
                    {"request_id": "req-1", "prompt_hash": "sha256:p1", "generated_token_count": 2, "generated_text": " first output"},
                    {"request_id": "req-2", "prompt_hash": "sha256:p2", "generated_token_count": 2, "generated_text": " second output"},
                ],
            },
            "diagnosis_codes": ["public_swarm_generate_ready", "public_swarm_generate_batch_ready"],
        }

        with patch.object(cli, "build_product_generate", return_value=generate_payload):
            report = cli.build_infer(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["local_output"]["output_count"], 2)
        self.assertEqual([row["generated_text"] for row in report["local_output"]["outputs"]], [" first output", " second output"])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        rendered = stdout.getvalue()
        self.assertIn("  batch: requests=2 observed=2 ready=True", rendered)
        self.assertIn("  wait: polls=2 accepted_rows=1 tokens=2/2 requests=2/2 batch_ready=True ledger=True stream=False", rendered)
        self.assertIn(
            "  trace_request[1]: request=req-1 tokens=2/2 hash=sha256:g1 source=generation-results",
            rendered,
        )
        self.assertIn(
            "  trace_request[2]: request=req-2 tokens=2/2 hash=sha256:g2 source=generation-results",
            rendered,
        )
        self.assertNotIn("  answer:  first output", rendered)
        self.assertIn("  answer[1]:  first output", rendered)
        self.assertIn("  answer[2]:  second output", rendered)
        self.assertIn(
            "  answer_scope: state=terminal-visible terminal_only=True visible_in_terminal=True saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            rendered,
        )
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["local_output"]["output_count"], 2)
        self.assertEqual([row["generated_text"] for row in persisted["local_output"]["outputs"]], ["", ""])
        self.assertEqual(persisted["trace"]["request_count"], 2)
        self.assertEqual(
            [
                (row["request_id"], row["prompt_hash"], row["generated_text_hash"], row["source"])
                for row in persisted["trace"]["request_trace"]
            ],
            [
                ("req-1", "sha256:p1", "sha256:g1", "generation-results"),
                ("req-2", "sha256:p2", "sha256:g2", "generation-results"),
            ],
        )
        self.assertTrue(persisted["trace"]["public_artifact_safe"])
        self.assertFalse(persisted["local_output"]["available"])
        self.assertFalse(persisted["local_output"]["display_only"])
        self.assertNotIn("first output", json.dumps(persisted, sort_keys=True))
        self.assertNotIn("second output", json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn(
            "- Trace request[1]: request=req-1 tokens=2/2 hash=sha256:g1 source=generation-results",
            markdown,
        )
        self.assertIn(
            "- Trace request[2]: request=req-2 tokens=2/2 hash=sha256:g2 source=generation-results",
            markdown,
        )

    def test_infer_existing_does_not_infer_batch_observed_without_batch_ready(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "--prompt-texts",
            "first prompt,second prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
            "--json",
        ])
        generate_payload = {
            "schema": "public_swarm_product_cli_v1",
            "ok": True,
            "mode": "generate",
            "generation": {
                "generated_token_count": 2,
                "max_new_tokens": 2,
                "generated_text_hash": "sha256:batch",
                "decoded_tokens_match": True,
            },
            "batch": {
                "enabled": True,
                "request_count": 2,
                "observed_request_count": 0,
                "batch_generation_ready": False,
            },
            "route": {"route_source": "coordinator-url", "coordinator_url_present": True},
            "diagnosis_codes": ["public_swarm_generate_ready"],
        }

        with patch.object(cli, "build_product_generate", return_value=generate_payload):
            report = cli.build_infer(args)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["result"]["status"], "blocked")
        self.assertEqual(report["result"]["output_count"], 0)
        self.assertIn("generation_timeout", report["diagnosis_codes"])
        self.assertIn("crowdtensor_infer_blocked", report["diagnosis_codes"])
        self.assertEqual(report["batch"]["request_count"], 2)
        self.assertEqual(report["batch"]["observed_request_count"], 0)
        self.assertFalse(report["batch"]["ready"])
        self.assertIn("Only 0/2 batch results appeared", report["operator_action"])
        self.assertEqual(report["recommended_next_command"]["label"], "retry inference with longer timeout")
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertFalse(persisted["ok"], persisted)
        self.assertEqual(persisted["result"]["status"], "blocked")
        self.assertEqual(persisted["batch"]["observed_request_count"], 0)
        self.assertEqual(persisted["wait_progress"]["expected_request_count"], 2)
        self.assertEqual(persisted["wait_progress"]["observed_request_count"], 0)
        self.assertIn("Only 0/2 batch results appeared", persisted["operator_action"])
        self.assertEqual(persisted["recommended_next_command"]["label"], "retry inference with longer timeout")
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("- OK: `False`", markdown)
        self.assertIn("- Result: `status=blocked", markdown)
        self.assertIn("- Batch: enabled=`True` requests=`0/2` ready=`False`", markdown)
        self.assertIn("Only 0/2 batch results appeared", markdown)
        self.assertEqual(markdown.count("- Action: Only 0/2 batch results appeared"), 1)

    def test_infer_markdown_hides_missing_stream_progress_when_stream_disabled(self) -> None:
        markdown = cli.render_infer_summary_markdown({
            "ok": True,
            "mode": "local",
            "diagnosis_codes": ["crowdtensor_infer_ready"],
            "generation": {"generated_token_count": 2, "max_new_tokens": 2},
            "prompt": {"prompt_count": 1, "prompt_hash": "sha256:p", "raw_prompt_public": False},
            "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
            "result": {"status": "complete", "generated_token_count": 2, "max_new_tokens": 2, "output_count": 1},
            "stream": {
                "enabled": False,
                "requested": False,
                "event_count": 0,
                "source": "disabled",
                "progress": {
                    "expected_request_count": 1,
                    "target_token_count": 2,
                    "per_request_progress": [],
                    "stream_progress_complete": False,
                },
            },
            "trace": {
                "session_id": "session",
                "request_count": 1,
                "accepted_rows_seen": 4,
                "stream_event_count": 0,
                "source": "product_swarm_mvp_check_v1",
                "public_artifact_safe": True,
                "request_trace": [
                    {
                        "request_id": "req-1",
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:g",
                        "source": "generation-results",
                    }
                ],
            },
        })

        self.assertIn("- Stream: enabled=`False`", markdown)
        self.assertIn("- Trace request[1]: request=req-1 tokens=2/2 hash=sha256:g source=generation-results", markdown)
        self.assertNotIn("- Stream progress: request=missing", markdown)
        self.assertNotIn("- Stream request[1]: request=missing", markdown)

    def test_infer_existing_batch_stream_progress_is_human_readable_and_safe(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "--prompt-texts",
            "first prompt,second prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--admin-token",
            "admin-secret",
            "--stream",
            "--output-dir",
            str(output_dir),
        ])
        generate_payload = {
            "schema": "public_swarm_product_cli_v1",
            "ok": True,
            "mode": "generate",
            "generation": {
                "generated_token_count": 2,
                "max_new_tokens": 2,
                "generated_text_hash": "sha256:batch",
                "decoded_tokens_match": True,
                "request_count": 2,
                "batch_generation_ready": True,
            },
            "batch": {"enabled": True, "request_count": 2, "batch_generation_ready": True},
            "route": {"route_source": "coordinator-url", "coordinator_url_present": True},
            "stream": {
                "enabled": True,
                "event_count": 3,
                "source": "admin-session-stream",
                "stream_generation_ready": False,
                "progress": {
                    "stream_progress_complete": False,
                    "all_token_events_ready": True,
                    "monotonic_progress": False,
                    "observed_token_counts": [1, 2, 1],
                    "max_observed_token_count": 2,
                    "target_token_count": 2,
                    "expected_request_count": 2,
                    "per_request_progress_complete": False,
                    "per_request_monotonic_progress": True,
                    "per_request_progress": [
                        {
                            "request_id": "req-1",
                            "prompt_hash": "sha256:p1",
                            "event_count": 2,
                            "observed_token_counts": [1, 2],
                            "max_observed_token_count": 2,
                            "target_token_count": 2,
                            "monotonic_progress": True,
                            "stream_progress_complete": True,
                        },
                        {
                            "request_id": "req-2",
                            "prompt_hash": "sha256:p2",
                            "event_count": 1,
                            "observed_token_counts": [1],
                            "max_observed_token_count": 1,
                            "target_token_count": 2,
                            "monotonic_progress": True,
                            "stream_progress_complete": False,
                        },
                    ],
                },
                "issue_summary": "request[2]=req-2:1/2",
                "events": [
                    {
                        "schema": "session_stream_event_v1",
                        "request_id": "req-1",
                        "prompt_hash": "sha256:p1",
                        "generated_token_count": 1,
                        "max_new_tokens": 2,
                        "generation_step": 0,
                        "generated_text_hash": "sha256:r1-step0",
                        "generated_text": "must not leak",
                        "generated_token_ids": [1],
                    }
                ],
            },
            "diagnosis_codes": ["public_swarm_generate_ready"],
        }

        with patch.object(cli, "build_product_generate", return_value=generate_payload):
            report = cli.build_infer(args)

        self.assertTrue(report["ok"], report)
        self.assertFalse(report["stream"]["ready"])
        self.assertEqual(report["stream"]["issue_summary"], "request[2]=req-2:1/2")
        self.assertEqual(report["review_summary"]["attention"], "request[2]=req-2:1/2")
        self.assertIn("stream progress is incomplete", report["review_summary"]["attention_detail"])
        self.assertEqual(report["review_summary"]["state"], "completed")
        self.assertEqual(
            report["operator_action"],
            "Inference completed, but stream progress is incomplete (request[2]=req-2:1/2); rerun with --stream if you need live token evidence.",
        )
        self.assertEqual(report["stream"]["progress"]["expected_request_count"], 2)
        self.assertEqual(
            cli.stream_progress_issue_summary(report["stream"]["progress"]),
            "request[2]=req-2:1/2",
        )
        self.assertEqual(
            [item["observed_token_counts"] for item in report["stream"]["progress"]["per_request_progress"]],
            [[1, 2], [1]],
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        rendered = stdout.getvalue()
        self.assertIn("  review: state=completed", rendered)
        self.assertIn("  review_next: label=rerun inference reason=rerun_inference", rendered)
        self.assertIn("attention=request[2]=req-2:1/2", rendered)
        self.assertIn(
            "  attention: request[2]=req-2:1/2 - stream progress is incomplete; rerun with --stream if you need live token evidence.",
            rendered,
        )
        self.assertIn("  stream[1]: request=req-1 tokens=2/2 counts=[1, 2] complete=True missing=False", rendered)
        self.assertIn("  stream[2]: request=req-2 tokens=1/2 counts=[1] complete=False missing=False", rendered)
        self.assertIn("  stream_issue: request[2]=req-2:1/2", rendered)
        self.assertIn("  action: Inference completed, but stream progress is incomplete (request[2]=req-2:1/2); rerun with --stream if you need live token evidence.", rendered)
        self.assertEqual(rendered.count("  action: "), 1)
        self.assertIn(f"  inspect_first: {output_dir / 'infer_summary.md'}", rendered)
        self.assertLess(rendered.index("  review_next: "), rendered.index("  inspect_first: "))
        self.assertLess(rendered.index("  inspect_first: "), rendered.index("  attention: "))
        self.assertLess(rendered.index("  attention: "), rendered.index("  action: "))
        self.assertLess(rendered.index("  action: "), rendered.index("  ok: "))
        self.assertLess(rendered.index("  ok: "), rendered.index("  model: "))
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("must not leak", encoded)
        self.assertNotIn('"generated_token_ids": [1]', encoded)
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["stream"]["issue_summary"], "request[2]=req-2:1/2")
        self.assertEqual(persisted["review_summary"]["attention"], "request[2]=req-2:1/2")
        self.assertIn("stream progress is incomplete", persisted["review_summary"]["attention_detail"])
        self.assertEqual(persisted["trace"]["stream_event_count"], 3)
        self.assertEqual(
            [(row["request_id"], row["prompt_hash"]) for row in persisted["trace"]["request_trace"]],
            [("req-1", "sha256:p1"), ("req-2", "sha256:p2")],
        )
        self.assertEqual(
            [
                (row["generated_token_count"], row["max_new_tokens"], row.get("generated_text_hash"))
                for row in persisted["trace"]["request_trace"]
            ],
            [(2, 2, None), (1, 2, None)],
        )
        self.assertEqual(
            persisted["operator_action"],
            "Inference completed, but stream progress is incomplete (request[2]=req-2:1/2); rerun with --stream if you need live token evidence.",
        )
        self.assertNotIn("must not leak", json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("attention=request[2]=req-2:1/2", markdown)
        self.assertIn(
            "- Attention: `request[2]=req-2:1/2 - stream progress is incomplete; rerun with --stream if you need live token evidence.`",
            markdown,
        )
        self.assertIn(
            "- Stream request[1]: request=req-1 tokens=2/2 counts=[1, 2] complete=True missing=False",
            markdown,
        )
        self.assertIn(
            "- Stream request[2]: request=req-2 tokens=1/2 counts=[1] complete=False missing=False",
            markdown,
        )
        self.assertIn("- Review next: `label=rerun inference reason=rerun_inference", markdown)
        self.assertIn("- Stream issue: `request[2]=req-2:1/2`", markdown)
        self.assertIn("Inference completed, but stream progress is incomplete", markdown)
        self.assertIn(
            "- Prompt input: saved Markdown keeps `<prompt-1>,<prompt-2>` placeholders; terminal `review_next` / `recommended_next` render safe local prompt sources for copy/paste when available, and saved commands should prefer `--prompt-file`, `--prompt-stdin`, or `--prompt-texts-file`.",
            markdown,
        )
        self.assertIn("Batch placeholder `<prompt-1>,<prompt-2>` is redacted. To rerun safely", markdown)
        self.assertIn("one prompt per non-empty line in `prompts.txt`", markdown)
        self.assertIn("replace the placeholder with `--prompt-texts-file prompts.txt`", markdown)
        self.assertNotIn("must not leak", markdown)
        self.assertNotIn("first prompt", markdown)
        self.assertNotIn("second prompt", markdown)

    def test_infer_existing_recomputes_missing_stream_issue_summary(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "--prompt-texts",
            "first prompt,second prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--admin-token",
            "admin-secret",
            "--stream",
            "--output-dir",
            str(output_dir),
        ])
        generate_payload = {
            "schema": "public_swarm_product_cli_v1",
            "ok": True,
            "mode": "generate",
            "generation": {
                "generated_token_count": 2,
                "max_new_tokens": 2,
                "generated_text_hash": "sha256:batch",
                "request_count": 2,
                "batch_generation_ready": True,
            },
            "batch": {"enabled": True, "request_count": 2, "batch_generation_ready": True},
            "route": {"route_source": "coordinator-url", "coordinator_url_present": True},
            "stream": {
                "enabled": True,
                "event_count": 2,
                "source": "admin-session-stream",
                "stream_generation_ready": False,
                "progress": {
                    "expected_request_count": 2,
                    "target_token_count": 2,
                    "per_request_progress_complete": False,
                    "per_request_progress": [
                        {
                            "request_id": "req-1",
                            "prompt_hash": "sha256:p1",
                            "event_count": 2,
                            "observed_token_counts": [1, 2],
                            "max_observed_token_count": 2,
                            "target_token_count": 2,
                            "stream_progress_complete": True,
                        },
                    ],
                },
            },
            "diagnosis_codes": ["public_swarm_generate_ready"],
        }

        with patch.object(cli, "build_product_generate", return_value=generate_payload):
            report = cli.build_infer(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["stream"]["issue_summary"], "missing_requests=1/2 request[2]=missing")
        self.assertEqual(
            report["operator_action"],
            "Inference completed, but stream progress is incomplete (missing_requests=1/2 request[2]=missing); rerun with --stream if you need live token evidence.",
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        rendered = stdout.getvalue()
        self.assertIn("  stream_issue: missing_requests=1/2 request[2]=missing", rendered)
        self.assertIn("  action: Inference completed, but stream progress is incomplete (missing_requests=1/2 request[2]=missing); rerun with --stream if you need live token evidence.", rendered)
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["stream"]["issue_summary"], "missing_requests=1/2 request[2]=missing")
        self.assertEqual(
            persisted["operator_action"],
            "Inference completed, but stream progress is incomplete (missing_requests=1/2 request[2]=missing); rerun with --stream if you need live token evidence.",
        )

    def test_infer_json_suppresses_raw_text_in_returned_payload(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--admin-token",
            "admin-secret",
            "--include-output",
            "--output-dir",
            str(output_dir),
            "--json",
        ])
        generate_payload = {
            "schema": "public_swarm_product_cli_v1",
            "ok": True,
            "mode": "generate",
            "generation": {
                "generated_token_count": 16,
                "max_new_tokens": 16,
                "generated_text_hash": "sha256:generated",
            },
            "route": {"route_source": "coordinator-url", "coordinator_url_present": True},
            "local_output": {"generated_text": "must not be returned in json"},
            "diagnosis_codes": ["public_swarm_generate_ready"],
        }

        with patch.object(cli, "build_product_generate", return_value=generate_payload):
            report = cli.build_infer(args)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertTrue(report["output_request"]["public_artifact_safe"])
        self.assertEqual(report["local_output"]["generated_text"], "")
        self.assertTrue(report["local_output"]["public_artifact_safe"])
        self.assertFalse(report["safety"]["raw_prompt_public"])
        self.assertFalse(report["safety"]["raw_generated_text_public"])
        self.assertFalse(report["safety"]["generated_token_ids_public"])
        self.assertTrue(report["safety"]["read_only_workload"])
        self.assertTrue(report["safety"]["coordinator_backed"])
        self.assertTrue(report["safety"]["not_production"])
        self.assertTrue(report["safety"]["not_large_model_serving"])
        self.assertTrue(report["safety"]["not_arbitrary_public_prompt_serving"])
        self.assertEqual(report["prompt_scope"]["source"], "prompt-text")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 1)
        self.assertTrue(report["prompt_scope"]["inline_prompt_text"])
        self.assertTrue(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertTrue(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertFalse(report["prompt_scope"]["terminal_local_paths"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_public_safe"])
        self.assertTrue(report["prompt_scope"]["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(report["prompt_scope"]["public_artifact_safe"])
        self.assertFalse(report["answer_scope"]["visible_in_terminal"])
        self.assertFalse(report["answer_scope"]["terminal_only"])
        self.assertEqual(report["answer_scope"]["scope_state"], "json-suppressed")
        self.assertEqual(report["answer_scope"]["summary"], cli.SAVED_ANSWER_SCOPE_TEXT)
        self.assertEqual(
            report["local_output_note"],
            "Raw generated text is suppressed in JSON/public output; rerun without --json for local display.",
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        self.assertNotIn("  answer_scope: ", stdout.getvalue())
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertTrue(persisted["output_request"]["include_output"])
        self.assertFalse(persisted["output_request"]["raw_prompt_public"])
        self.assertFalse(persisted["output_request"]["raw_generated_text_public"])
        self.assertFalse(persisted["output_request"]["generated_token_ids_public"])
        self.assertTrue(persisted["output_request"]["public_artifact_safe"])
        self.assertTrue(persisted["local_output"]["public_artifact_safe"])
        self.assertEqual(persisted["local_output"]["output_count"], 1)
        self.assertFalse(persisted["safety"]["raw_prompt_public"])
        self.assertFalse(persisted["safety"]["raw_generated_text_public"])
        self.assertFalse(persisted["safety"]["generated_token_ids_public"])
        self.assertTrue(persisted["safety"]["read_only_workload"])
        self.assertTrue(persisted["safety"]["coordinator_backed"])
        self.assertTrue(persisted["safety"]["not_production"])
        self.assertTrue(persisted["safety"]["not_large_model_serving"])
        self.assertTrue(persisted["safety"]["not_arbitrary_public_prompt_serving"])
        self.assertEqual(persisted["prompt_scope"]["source"], "prompt-text")
        self.assertTrue(persisted["prompt_scope"]["inline_prompt_text"])
        self.assertTrue(persisted["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertFalse(persisted["prompt_scope"]["terminal_local_paths"])
        self.assertTrue(persisted["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertFalse(persisted["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(persisted["prompt_scope"]["public_artifact_safe"])
        self.assertFalse(persisted["answer_scope"]["visible_in_terminal"])
        self.assertFalse(persisted["answer_scope"]["terminal_only"])
        self.assertEqual(persisted["answer_scope"]["scope_state"], "json-suppressed")
        self.assertEqual(persisted["answer_scope"]["summary"], cli.SAVED_ANSWER_SCOPE_TEXT)
        self.assertEqual(
            persisted["local_output_note"],
            "Raw generated text is suppressed in JSON/public output; rerun without --json for local display.",
        )
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn(
            "- Local output: `available=False display_only=False public_artifact_safe=True saved_redacted=True` count=`1` source=``",
            markdown,
        )
        self.assertIn(
            "- Prompt scope: `source=prompt-text count=1 inline_prompt_text=True terminal_next_commands_local_private=True terminal_local_paths=False saved_artifacts_prompt_placeholders=True prompt_file_path_public=False raw_prompt_public=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Local output note: Raw generated text is suppressed in JSON/public output; rerun without --json for local display.",
            markdown,
        )
        self.assertIn(f"- Answer scope note: {cli.SAVED_ANSWER_SCOPE_TEXT}", markdown)
        self.assertIn(f"- Safety: saved Markdown keeps prompt placeholders and redacted generated output. {cli.SAVED_ANSWER_SCOPE_TEXT}", markdown)
        self.assertNotIn("inspect terminal output for any local answer", markdown)
        self.assertNotIn("must not be returned in json", markdown)

    def test_print_infer_batch_outputs_are_not_duplicated(self) -> None:
        report = {
            "ok": True,
            "mode": "local",
            "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
            "generation": {"generated_token_count": 2, "max_new_tokens": 2, "generated_text_hash": "sha256:batch"},
            "route": {"route_source": "local-product-loopback", "route_ready": True, "distinct_stage_miners": True},
            "stream": {},
            "local_output": {
                "generated_text": " first answer",
                "outputs": [
                    {"generated_text": " first answer"},
                    {"generated_text": " second answer"},
                ],
                "note": "local only",
            },
            "output_dir": "/tmp/infer",
            "diagnosis_codes": ["crowdtensor_infer_ready"],
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)

        rendered = stdout.getvalue()
        self.assertNotIn("  answer:  first answer", rendered)
        self.assertIn("  answer[1]:  first answer", rendered)
        self.assertIn("  answer[2]:  second answer", rendered)
        self.assertIn(
            "  answer_scope: state=terminal-visible terminal_only=True visible_in_terminal=True saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            rendered,
        )

    def test_print_infer_prompt_scope_note_marks_inline_prompt_terminal_private(self) -> None:
        report = {
            "ok": True,
            "mode": "local",
            "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
            "review_summary": {
                "state": "completed",
                "next_step": "rerun_or_review_artifacts",
                "recommended_label": "rerun local inference",
                "recommended_reason": "rerun_inference",
                "next_command": "crowdtensor infer '<prompt>' --mode local",
                "public_artifact_safe": True,
            },
            "recommended_next_command": cli.command_entry(
                "rerun local inference",
                ["crowdtensor", "infer", cli.INFER_PROMPT_PLACEHOLDER, "--mode", "local"],
            ),
            "local_prompt_text": "private prompt",
            "prompt_scope": cli._prompt_scope(source="prompt-text", prompt_count=1),
            "diagnosis_codes": ["crowdtensor_infer_ready"],
            "stream": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)

        rendered = stdout.getvalue()
        self.assertIn("  prompt_scope: terminal_next_commands=local-private", rendered)
        self.assertIn(
            "  prompt_scope_note: Treat this terminal log as local-private because rerun commands may include inline prompt text. "
            "Saved JSON/Markdown keep prompt placeholders and omit raw prompt text.",
            rendered,
        )
        self.assertIn("  review_next: label=rerun local inference", rendered)
        self.assertLess(rendered.index("  prompt_scope_note: "), rendered.index("  review_next: "))

    def test_print_infer_shareable_terminal_suppresses_prompt_scope_note(self) -> None:
        report = {
            "ok": True,
            "mode": "local",
            "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
            "review_summary": {
                "state": "completed",
                "next_step": "rerun_or_review_artifacts",
                "recommended_label": "rerun local inference",
                "recommended_reason": "rerun_inference",
                "next_command": "crowdtensor infer '<prompt>' --mode local",
                "public_artifact_safe": True,
            },
            "local_prompt_text": "private prompt",
            "prompt_scope": cli._prompt_scope(source="prompt-text", prompt_count=1),
            "shareable_terminal": {"enabled": True},
            "diagnosis_codes": ["crowdtensor_infer_ready"],
            "stream": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)

        rendered = stdout.getvalue()
        self.assertNotIn("  prompt_scope: ", rendered)
        self.assertNotIn("  prompt_scope_note: ", rendered)
        self.assertIn("  shareable_terminal: enabled=True", rendered)

    def test_infer_trace_keeps_generation_hash_source_when_local_output_lacks_hash(self) -> None:
        trace = cli._infer_request_trace_from_payload(
            {},
            generation={
                "results": [
                    {
                        "request_id": "req-1",
                        "prompt_hash": "sha256:p1",
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:g1",
                    }
                ]
            },
            stream_events=[],
            stream_progress={},
            local_output={
                "outputs": [
                    {
                        "request_id": "req-1",
                        "prompt_hash": "sha256:p1",
                        "generated_token_count": 2,
                        "generated_text": "local only",
                    }
                ]
            },
        )

        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["source"], "generation-results")
        self.assertEqual(trace[0]["generated_text_hash"], "sha256:g1")

    def test_print_generate_no_local_answer_shows_answer_scope(self) -> None:
        report = {
            "ok": False,
            "diagnosis_codes": ["generation_timeout"],
            "result": {"status": "blocked", "output_count": 0, "display": "hash-only"},
            "generation": {"generated_token_count": 0, "max_new_tokens": 4, "generated_text_hash": ""},
            "output_display": {
                "terminal_display": "hash-only",
                "terminal_text_available": False,
                "saved_artifact_display": "hash-only",
                "json_stdout_display": "hash-only-json",
                "include_output_requested": False,
                "raw_generated_text_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
                "summary": cli.SAVED_NO_ANSWER_SCOPE_TEXT,
            },
            "local_output": {
                "available": False,
                "output_count": 0,
                "source": "",
                "public_artifact_safe": True,
            },
            "stream": {},
            "route": {"route_source": "coordinator-url", "coordinator_url_present": True},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)

        rendered = stdout.getvalue()
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            rendered,
        )
        self.assertIn(f"  answer_scope_note: {cli.SAVED_NO_ANSWER_SCOPE_TEXT}", rendered)
        self.assertNotIn("  answer: ", rendered)
        self.assertLess(rendered.index("  output_display: "), rendered.index("  answer_scope: "))

    def test_infer_recommended_next_prioritizes_startup_blocker_over_timeout(self) -> None:
        next_commands = [
            cli.command_entry("start Coordinator", ["crowdtensor", "serve", "--run"]),
            cli.command_entry(
                "check existing swarm",
                ["crowdtensor", "infer", cli.INFER_PROMPT_PLACEHOLDER, "--mode", "existing", "--dry-run"],
                requires_env=["CROWDTENSOR_OBSERVER_TOKEN"],
            ),
            cli.command_entry(
                "retry inference with longer timeout",
                ["crowdtensor", "infer", cli.INFER_PROMPT_PLACEHOLDER, "--mode", "existing", "--timeout-seconds", "240"],
                requires_env=["CROWDTENSOR_ADMIN_TOKEN"],
            ),
        ]

        recommended = cli._infer_recommended_next_command(
            next_commands,
            ok=False,
            mode="existing",
            dry_run=False,
            ready_to_submit={},
            diagnosis_codes={"coordinator_route_missing", "generation_timeout"},
            full_evidence=False,
        )

        self.assertEqual(recommended["label"], "start Coordinator")
        self.assertEqual(recommended["reason"], "start_coordinator")

    def test_generate_recommended_next_prioritizes_startup_blocker_over_timeout(self) -> None:
        next_commands = [
            cli.command_entry("start Coordinator", ["crowdtensor", "serve", "--run"]),
            cli.command_entry(
                "check generation route",
                ["crowdtensor", "generate", "--prompt-text", cli.INFER_PROMPT_PLACEHOLDER, "--dry-run"],
                requires_env=["CROWDTENSOR_OBSERVER_TOKEN"],
            ),
            cli.command_entry(
                "retry generation with longer timeout",
                ["crowdtensor", "generate", "--prompt-text", cli.INFER_PROMPT_PLACEHOLDER, "--timeout-seconds", "240"],
                requires_env=["CROWDTENSOR_ADMIN_TOKEN"],
            ),
        ]

        recommended = cli._generate_recommended_next_command(
            next_commands,
            ok=False,
            dry_run=False,
            ready_to_submit={},
            diagnosis_codes={"coordinator_route_missing", "generation_timeout"},
        )

        self.assertEqual(recommended["label"], "start Coordinator")
        self.assertEqual(recommended["reason"], "start_coordinator")

    def test_infer_dry_run_recommends_preflight_before_timeout_retry(self) -> None:
        next_commands = [
            cli.command_entry(
                "check existing swarm",
                ["crowdtensor", "infer", cli.INFER_PROMPT_PLACEHOLDER, "--mode", "existing", "--dry-run"],
                requires_env=["CROWDTENSOR_OBSERVER_TOKEN"],
            ),
            cli.command_entry(
                "retry inference with longer timeout",
                ["crowdtensor", "infer", cli.INFER_PROMPT_PLACEHOLDER, "--mode", "existing", "--timeout-seconds", "240"],
                requires_env=["CROWDTENSOR_ADMIN_TOKEN"],
            ),
        ]

        recommended = cli._infer_recommended_next_command(
            next_commands,
            ok=True,
            mode="existing",
            dry_run=True,
            ready_to_submit={"next_step": "run_live_preflight"},
            diagnosis_codes={"generation_timeout", "crowdtensor_infer_preflight_partial"},
            full_evidence=False,
        )

        self.assertEqual(recommended["label"], "check existing swarm")
        self.assertEqual(recommended["reason"], "confirm_live_preflight")

    def test_generate_dry_run_recommends_preflight_before_timeout_retry(self) -> None:
        next_commands = [
            cli.command_entry(
                "check generation route",
                ["crowdtensor", "generate", "--prompt-text", cli.INFER_PROMPT_PLACEHOLDER, "--dry-run"],
                requires_env=["CROWDTENSOR_OBSERVER_TOKEN"],
            ),
            cli.command_entry(
                "retry generation with longer timeout",
                ["crowdtensor", "generate", "--prompt-text", cli.INFER_PROMPT_PLACEHOLDER, "--timeout-seconds", "240"],
                requires_env=["CROWDTENSOR_ADMIN_TOKEN"],
            ),
        ]

        recommended = cli._generate_recommended_next_command(
            next_commands,
            ok=True,
            dry_run=True,
            ready_to_submit={"next_step": "run_live_preflight"},
            diagnosis_codes={"generation_timeout", "generate_request_shape_ready"},
        )

        self.assertEqual(recommended["label"], "check generation route")
        self.assertEqual(recommended["reason"], "confirm_live_preflight")

    def test_print_infer_no_local_answer_shows_answer_scope(self) -> None:
        report = {
            "ok": False,
            "mode": "existing",
            "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
            "diagnosis_codes": ["coordinator_route_missing"],
            "generation": {"generated_token_count": 0, "max_new_tokens": 8, "generated_text_hash": ""},
            "output_display": {
                "terminal_display": "hash-only",
                "terminal_text_available": False,
                "saved_artifact_display": "hash-only",
                "json_stdout_display": "hash-only-json",
                "include_output_requested": False,
                "raw_generated_text_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
                "summary": cli.SAVED_NO_ANSWER_SCOPE_TEXT,
            },
            "local_output": {
                "available": False,
                "output_count": 0,
                "source": "",
                "public_artifact_safe": True,
            },
            "route": {"route_source": "coordinator-url", "route_ready": False},
            "stream": {},
            "output_dir": "/tmp/infer",
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)

        rendered = stdout.getvalue()
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            rendered,
        )
        self.assertIn(f"  answer_scope_note: {cli.SAVED_NO_ANSWER_SCOPE_TEXT}", rendered)
        self.assertNotIn("  answer: ", rendered)
        self.assertLess(rendered.index("  output_display: "), rendered.index("  answer_scope: "))

    def test_print_infer_multiline_answer_is_indented(self) -> None:
        report = {
            "ok": True,
            "mode": "local",
            "model": {"hf_model_id": "sshleifer/tiny-gpt2", "backend": "cpu"},
            "generation": {"generated_token_count": 2, "max_new_tokens": 2, "generated_text_hash": "sha256:answer"},
            "route": {"route_source": "local-product-loopback", "route_ready": True, "distinct_stage_miners": True},
            "stream": {},
            "local_output": {
                "generated_text": "first line\nsecond line",
                "output_count": 1,
                "source": "local-private-task-state",
                "note": "local only",
            },
            "diagnosis_codes": ["crowdtensor_infer_ready"],
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)

        rendered = stdout.getvalue()
        self.assertIn("  answer: first line\n          second line\n", rendered)
        self.assertIn(
            "  answer_scope: state=terminal-visible terminal_only=True visible_in_terminal=True saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  local_output: available=True display_only=False public_artifact_safe=False count=1 source=local-private-task-state",
            rendered,
        )
        self.assertNotIn("\nsecond line\n", rendered)
        self.assertLess(rendered.index("  answer: first line"), rendered.index("  local_output: "))
        self.assertLess(rendered.index("  answer_scope: "), rendered.index("  local_output: "))

    def test_print_generate_multiline_batch_answers_are_indented(self) -> None:
        report = {
            "ok": True,
            "diagnosis_codes": ["public_swarm_generate_ready"],
            "result": {
                "status": "complete",
                "generated_token_count": 2,
                "max_new_tokens": 2,
                "output_count": 2,
                "display": "local-private",
                "generated_text_hash": "sha256:answer",
                "public_artifact_safe": False,
            },
            "local_output": {
                "outputs": [
                    {"generated_text": "first one\nsecond one"},
                    {"generated_text": "first two\nsecond two"},
                ],
                "display_only": True,
                "public_artifact_safe": False,
            },
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)

        rendered = stdout.getvalue()
        self.assertIn("  answer[1]: first one\n             second one\n", rendered)
        self.assertIn("  answer[2]: first two\n             second two\n", rendered)
        self.assertIn(
            "  answer_scope: state=terminal-visible terminal_only=True visible_in_terminal=True saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            rendered,
        )
        self.assertNotIn("\nsecond one\n", rendered)
        self.assertNotIn("\nsecond two\n", rendered)

    def test_infer_failure_includes_operator_action(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "2",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            return completed({
                "schema": "product_swarm_mvp_check_v1",
                "ok": False,
                "mode": "local-loopback",
                "generation": {"generated_token_count": 0, "max_new_tokens": 2},
                "diagnosis_codes": ["hf_dependencies_missing"],
            }, returncode=1)

        report = cli.build_infer(args, runner=fake_runner)

        self.assertFalse(report["ok"], report)
        self.assertIn("pip install -e '.[hf]'", report["operator_action"])
        self.assertEqual(report["user_status"]["state"], "blocked")
        self.assertIn("pip install -e '.[hf]'", report["user_status"]["headline"])
        self.assertEqual(report["user_status"]["next_step"], "fix_blockers")
        self.assertEqual(report["user_status"]["recommended_label"], "install Hugging Face runtime")
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn("python -m pip install -e '.[hf]'", next_lines)
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode local --output-dir {output_dir} --max-new-tokens 2",
            next_lines,
        )
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertIn("pip install -e '.[hf]'", persisted["operator_action"])
        self.assertEqual(persisted["user_status"]["state"], "blocked")
        self.assertEqual(persisted["user_status"]["recommended_label"], "install Hugging Face runtime")
        self.assertIn("python -m pip install -e '.[hf]'", [item["command_line"] for item in persisted["next_commands"]])

    def test_infer_existing_route_failure_includes_startup_next_commands(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
            "--json",
        ])
        payload = {
            "schema": "public_swarm_product_cli_v1",
            "ok": False,
            "mode": "generate",
            "diagnosis_codes": ["coordinator_route_missing"],
            "route": {"route_source": "coordinator-url", "coordinator_url_present": False},
            "generation": {"generated_token_count": 0, "max_new_tokens": 8},
        }

        report = cli._infer_summary_from_payload(args, payload, mode="existing", output_dir=output_dir)

        self.assertFalse(report["ok"], report)
        self.assertIn("Start a Coordinator", report["operator_action"])
        self.assertEqual(report["issue_summary"]["primary_code"], "coordinator_route_missing")
        self.assertEqual(report["review_summary"]["primary_code"], "coordinator_route_missing")
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn("crowdtensor serve --profile cpu-real-llm --bind-host 127.0.0.1 --public-host 127.0.0.1 --port 8787 --run", next_lines)
        self.assertIn("crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage0-miner --stage stage0 --run", next_lines)
        self.assertIn("crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage1-miner --stage stage1 --run", next_lines)
        self.assertNotIn("CrowdTensor user prompt", json.dumps(report, sort_keys=True))

    def test_infer_existing_stage_not_checked_includes_startup_next_commands(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--output-dir",
            str(output_dir),
            "--json",
        ])
        payload = {
            "schema": "public_swarm_product_cli_v1",
            "ok": False,
            "mode": "generate",
            "diagnosis_codes": ["stage_preflight_not_checked"],
            "route": {
                "route_source": "coordinator-url",
                "coordinator_url": "http://127.0.0.1:8787",
                "coordinator_url_present": True,
            },
            "generation": {"generated_token_count": 0, "max_new_tokens": 8},
        }

        report = cli._infer_summary_from_payload(args, payload, mode="existing", output_dir=output_dir)

        self.assertFalse(report["ok"], report)
        self.assertIn("Coordinator readiness first", report["operator_action"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn("crowdtensor serve --profile cpu-real-llm --bind-host 127.0.0.1 --public-host 127.0.0.1 --port 8787 --run", next_lines)
        self.assertIn("crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage0-miner --stage stage0 --run", next_lines)
        self.assertIn("crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage1-miner --stage stage1 --run", next_lines)

    def test_infer_timeout_action_uses_wait_progress(self) -> None:
        cases = [
            (
                {
                    "session_created": True,
                    "ledger_endpoint_ready": True,
                    "accepted_rows_seen": 0,
                    "max_observed_token_count": 0,
                    "target_token_count": 4,
                    "expected_request_count": 1,
                    "observed_request_count": 0,
                },
                "No accepted result rows appeared",
            ),
            (
                {
                    "session_created": True,
                    "ledger_endpoint_ready": True,
                    "accepted_rows_seen": 1,
                    "max_observed_token_count": 2,
                    "target_token_count": 4,
                    "expected_request_count": 1,
                    "observed_request_count": 1,
                    "last_error_type": "HTTPError",
                },
                "Generation reached 2/4 tokens",
            ),
            (
                {
                    "session_created": True,
                    "ledger_endpoint_ready": True,
                    "accepted_rows_seen": 0,
                    "max_observed_token_count": 0,
                    "target_token_count": 4,
                    "expected_request_count": 1,
                    "observed_request_count": 0,
                    "last_error_type": "HTTPError",
                },
                "Coordinator polling reported HTTPError",
            ),
            (
                {
                    "session_created": True,
                    "ledger_endpoint_ready": False,
                    "accepted_rows_seen": 0,
                    "max_observed_token_count": 0,
                    "target_token_count": 4,
                    "expected_request_count": 1,
                    "observed_request_count": 0,
                },
                "/admin/results was not reachable",
            ),
        ]
        for case_index, (wait_progress, expected) in enumerate(cases):
            with self.subTest(expected=expected):
                output_dir = Path(self._tmp_dir())
                argv = [
                    "infer",
                    "CrowdTensor user prompt",
                    "--mode",
                    "existing",
                    "--coordinator-url",
                    "http://127.0.0.1:8787",
                    "--admin-token",
                    "admin-secret",
                    "--output-dir",
                    str(output_dir),
                    "--max-new-tokens",
                    "4",
                    "--json",
                ]
                if case_index == 0:
                    argv.extend([
                        "--timeout-seconds",
                        "90",
                        "--poll-interval",
                        "0.5",
                        "--http-timeout",
                        "8",
                        "--admin-results-limit",
                        "7",
                    ])
                    wait_progress = {**wait_progress, "timeout_seconds": 90}
                args = cli.parse_args(argv)
                payload = {
                    "schema": "public_swarm_product_cli_v1",
                    "ok": False,
                    "mode": "generate",
                    "diagnosis_codes": ["generation_timeout"],
                    "route": {"route_source": "coordinator-url", "coordinator_url_present": True},
                    "generation": {"generated_token_count": 0, "max_new_tokens": 4},
                    "wait_progress": wait_progress,
                }

                report = cli._infer_summary_from_payload(args, payload, mode="existing", output_dir=output_dir)

                self.assertFalse(report["ok"], report)
                self.assertIn(expected, report["operator_action"])
                self.assertIn("crowdtensor_infer_blocked", report["diagnosis_codes"])
                if case_index == 0:
                    self.assertEqual(report["runtime_options"]["timeout_seconds"], 90.0)
                    self.assertEqual(report["runtime_options"]["poll_interval"], 0.5)
                    self.assertEqual(report["runtime_options"]["http_timeout"], 8.0)
                    self.assertEqual(report["runtime_options"]["admin_results_limit"], 7)
                    self.assertTrue(report["runtime_options"]["public_artifact_safe"])
                next_lines = [item["command_line"] for item in report["next_commands"]]
                expected_retry_timeout = "180" if case_index == 0 else "240"
                expected_extra = (
                    " --poll-interval 0.5 --http-timeout 8.0 --admin-results-limit 7"
                    if case_index == 0
                    else ""
                )
                self.assertIn(
                    f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --max-new-tokens 4 --coordinator-url http://127.0.0.1:8787 --timeout-seconds {expected_retry_timeout}{expected_extra}",
                    next_lines,
                )
                retry = next(item for item in report["next_commands"] if item["label"] == "retry inference with longer timeout")
                self.assertEqual(retry["requires_env"], ["CROWDTENSOR_ADMIN_TOKEN"])
                self.assertEqual(retry["command"].count("--timeout-seconds"), 1)
                self.assertEqual(retry["command"][retry["command"].index("--timeout-seconds") + 1], expected_retry_timeout)
                if case_index == 0:
                    self.assertEqual(retry["command"].count("--poll-interval"), 1)
                    self.assertEqual(retry["command"][retry["command"].index("--poll-interval") + 1], "0.5")
                    self.assertEqual(retry["command"].count("--http-timeout"), 1)
                    self.assertEqual(retry["command"][retry["command"].index("--http-timeout") + 1], "8.0")
                    self.assertEqual(retry["command"].count("--admin-results-limit"), 1)
                    self.assertEqual(retry["command"][retry["command"].index("--admin-results-limit") + 1], "7")
                else:
                    self.assertNotIn("--poll-interval", retry["command"])
                    self.assertNotIn("--http-timeout", retry["command"])
                    self.assertNotIn("--admin-results-limit", retry["command"])
                self.assertNotIn("--timeout-seconds 90 --timeout-seconds 180", retry["command_line"])
                self.assertEqual(
                    report["recommended_next_command"]["reason_detail"],
                    "Retry the same request with a longer timeout after incomplete or partial progress.",
                )
                self.assertNotIn("CrowdTensor user prompt", json.dumps(report["next_commands"], sort_keys=True))
                self.assertNotIn("admin-secret", json.dumps(report, sort_keys=True))
                persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
                self.assertIn(expected, persisted["operator_action"])
                if case_index == 0:
                    self.assertEqual(persisted["runtime_options"]["timeout_seconds"], 90.0)
                    self.assertEqual(persisted["runtime_options"]["poll_interval"], 0.5)
                    self.assertEqual(persisted["runtime_options"]["http_timeout"], 8.0)
                    self.assertEqual(persisted["runtime_options"]["admin_results_limit"], 7)
                self.assertIn(
                    f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --max-new-tokens 4 --coordinator-url http://127.0.0.1:8787 --timeout-seconds {expected_retry_timeout}{expected_extra}",
                    [item["command_line"] for item in persisted["next_commands"]],
                )
                if case_index == 0:
                    markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
                    self.assertIn(
                        "- Runtime options: `timeout_seconds=90.0 poll_interval=0.5 http_timeout=8.0 admin_results_limit=7 public_artifact_safe=True`",
                        markdown,
                    )
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        cli.print_infer(report)
                    rendered = stdout.getvalue()
                    self.assertIn(
                        "  runtime_options: timeout_seconds=90.0 poll_interval=0.5 http_timeout=8.0 admin_results_limit=7 public_artifact_safe=True",
                        rendered,
                    )

    def test_infer_existing_requires_route(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        report = cli.build_infer(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("coordinator_route_missing", report["diagnosis_codes"])
        self.assertIn("crowdtensor_infer_blocked", report["diagnosis_codes"])
        self.assertEqual(report["issue_summary"]["primary_code"], "coordinator_route_missing")
        self.assertEqual(report["review_summary"]["primary_code"], "coordinator_route_missing")
        self.assertEqual(
            report["operator_action"],
            "Start a Coordinator and two stage Miners, or pass --coordinator-url/--peer-bootstrap for an existing swarm.",
        )
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertEqual(report["recommended_next_command"]["label"], "start Coordinator")
        self.assertEqual(report["recommended_next_command"]["reason"], "start_coordinator")
        self.assertIn(
            "crowdtensor serve --profile cpu-real-llm --bind-host 127.0.0.1 --public-host 127.0.0.1 --port 8787 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage0-miner --stage stage0 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage1-miner --stage stage1 --run",
            next_lines,
        )
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --max-new-tokens 8 --dry-run --coordinator-url http://127.0.0.1:8787 --observer-token ${{CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}}",
            next_lines,
        )
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --max-new-tokens 8 --coordinator-url http://127.0.0.1:8787",
            next_lines,
        )
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertIn("coordinator_route_missing", persisted["diagnosis_codes"])
        self.assertEqual(persisted["recommended_next_command"]["label"], "start Coordinator")

    def test_infer_existing_missing_admin_token_returns_actionable_report(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        report = cli.build_infer(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("admin_token_required", report["diagnosis_codes"])
        self.assertIn("crowdtensor_infer_blocked", report["diagnosis_codes"])
        self.assertEqual(report["operator_action"], "Pass --admin-token or set CROWDTENSOR_ADMIN_TOKEN.")
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --max-new-tokens 8 --dry-run --coordinator-url http://127.0.0.1:8787 --observer-token ${{CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}}",
            next_lines,
        )
        self.assertTrue(any("CROWDTENSOR_OBSERVER_TOKEN" in item.get("requires_env", []) for item in report["next_commands"]))
        self.assertTrue(any("CROWDTENSOR_ADMIN_TOKEN" in item.get("requires_env", []) for item in report["next_commands"]))
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertIn("admin_token_required", persisted["diagnosis_codes"])

    def test_infer_existing_p2p_discovery_unreachable_is_actionable(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--p2p",
            "--peer-bootstrap",
            "http://127.0.0.1:8799",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        with patch.object(cli, "fetch_peer_catalog", side_effect=OSError("offline")), patch.object(
            cli,
            "request_json_url",
            side_effect=AssertionError("session creation should be blocked when discovery is offline"),
        ):
            report = cli.build_infer(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("p2p_discovery_unreachable", report["diagnosis_codes"])
        self.assertIn("coordinator_route_missing", report["diagnosis_codes"])
        self.assertIn("crowdtensor_infer_blocked", report["diagnosis_codes"])
        self.assertEqual(report["p2p"]["discovery"]["error"], "OSError")
        self.assertIn("P2P discovery daemon", report["operator_action"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn("crowdtensor p2pd --port 8799 --run", next_lines)
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --max-new-tokens 8 --dry-run --peer-bootstrap http://127.0.0.1:8799 --p2p --observer-token ${{CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}}",
            next_lines,
        )
        self.assertNotIn("CrowdTensor user prompt", json.dumps(report, sort_keys=True))
        self.assertNotIn("admin-secret", json.dumps(report, sort_keys=True))
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertIn("p2p_discovery_unreachable", persisted["diagnosis_codes"])
        self.assertEqual(persisted["p2p"]["discovery"]["error"], "OSError")
        self.assertNotIn("CrowdTensor user prompt", json.dumps(persisted, sort_keys=True))
        self.assertNotIn("admin-secret", json.dumps(persisted, sort_keys=True))

    def test_infer_existing_real_p2p_discovery_unreachable_suggests_real_daemon(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--p2p",
            "--p2p-backend",
            "real",
            "--peer-bootstrap",
            "http://127.0.0.1:8899",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        with patch.object(cli, "fetch_provider_catalog", side_effect=OSError("offline")), patch.object(
            cli,
            "request_json_url",
            side_effect=AssertionError("session creation should be blocked when discovery is offline"),
        ):
            report = cli.build_infer(args)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["p2p"]["backend"], "real")
        self.assertIn("p2p_discovery_unreachable", report["diagnosis_codes"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn("crowdtensor p2p-daemon --port 8899 --run", next_lines)
        self.assertFalse(any(line.startswith("crowdtensor p2pd ") for line in next_lines))
        self.assertNotIn("CrowdTensor user prompt", json.dumps(report, sort_keys=True))
        self.assertNotIn("admin-secret", json.dumps(report, sort_keys=True))

    def test_infer_existing_dry_run_preflights_without_admin_token(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--dry-run",
            "--stream",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del payload, admin_token, timeout
            self.assertEqual(method, "GET")
            self.assertEqual(base_url, "http://127.0.0.1:8787")
            self.assertEqual(path, "/ready")
            return {"schema": "ready_v1", "service": "crowdtensord-coordinator", "protocol": "runtime_contract_v1"}

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_infer(args)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["dry_run"])
        self.assertTrue(report["stream"]["enabled"])
        self.assertFalse(report["output_request"]["include_output"])
        self.assertEqual(report["route"]["route_source"], "coordinator-url")
        self.assertTrue(report["coordinator_ready"]["ok"])
        self.assertEqual(report["coordinator_ready"]["protocol"], "runtime_contract_v1")
        self.assertFalse(report["stage_preflight"]["checked"])
        self.assertEqual(report["stage_preflight"]["reason"], "observer_token_missing")
        self.assertEqual(report["stage_preflight"]["missing_summary"], "not_checked")
        self.assertEqual(report["ready_to_submit"], {
            "ok": True,
            "fully_verified": False,
            "readiness_label": "partial",
            "readiness_summary": "Request can be submitted, but stage Miner readiness is not fully verified.",
            "next_step": "run_stage_preflight",
            "route_ready": True,
            "coordinator_ready": True,
            "coordinator_preflight_required": True,
            "stage_preflight_ok": None,
            "stage_preflight_required": False,
            "stage_verification": "skipped",
            "warning_codes": ["stage_preflight_skipped"],
            "source": "dry-run-preflight",
            "public_artifact_safe": True,
        })
        self.assertIn("coordinator_ready_preflight_ready", report["diagnosis_codes"])
        self.assertIn("stage_preflight_skipped", report["diagnosis_codes"])
        self.assertNotIn("coordinator_ready_preflight_skipped", report["diagnosis_codes"])
        self.assertNotIn("generate_dry_run_ready", report["diagnosis_codes"])
        self.assertNotIn("generate_dry_run_partial", report["diagnosis_codes"])
        self.assertNotIn("generate_request_shape_ready", report["diagnosis_codes"])
        self.assertIn("crowdtensor_infer_preflight_partial", report["diagnosis_codes"])
        self.assertIn("user_friendly_infer_preflight_partial", report["diagnosis_codes"])
        self.assertNotIn("crowdtensor_infer_preflight_ready", report["diagnosis_codes"])
        self.assertNotIn("crowdtensor_infer_ready", report["diagnosis_codes"])
        self.assertEqual(
            report["operator_action"],
            "Inference can be submitted, but stage0/stage1 were not fully verified; run the printed stage-preflight next command with --observer-token before the submit next command.",
        )
        self.assertEqual(report["user_status"]["state"], "preflight-partial")
        self.assertEqual(
            report["user_status"]["headline"],
            "Request can be submitted, but stage Miner readiness is not fully verified.",
        )
        self.assertEqual(report["user_status"]["next_step"], "run_stage_preflight")
        self.assertEqual(report["user_status"]["recommended_label"], "check existing swarm")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        self.assertIn(
            "  status: preflight-partial: Request can be submitted, but stage Miner readiness is not fully verified. next=run_stage_preflight recommendation=check existing swarm public_artifact_safe=True",
            stdout.getvalue(),
        )
        self.assertIn(
            "  stage_preflight: checked=False ok=None matched_miners=None missing=not_checked reason=observer_token_missing source=not-checked",
            stdout.getvalue(),
        )
        next_commands = report["next_commands"]
        self.assertTrue(any("CROWDTENSOR_OBSERVER_TOKEN" in item.get("requires_env", []) for item in next_commands))
        self.assertEqual(report["recommended_next_command"]["label"], "check existing swarm")
        self.assertEqual(report["recommended_next_command"]["reason"], "verify_stage_miners")
        self.assertEqual(report["recommended_next_command"]["requires_env"], ["CROWDTENSOR_OBSERVER_TOKEN"])
        next_lines = [item["command_line"] for item in next_commands]
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --stream --max-new-tokens 8 --dry-run --coordinator-url http://127.0.0.1:8787 --observer-token ${{CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}}",
            next_lines,
        )
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --stream --max-new-tokens 8 --coordinator-url http://127.0.0.1:8787",
            next_lines,
        )
        labels = [item["label"] for item in next_commands]
        self.assertIn("submit inference after stage preflight", labels)
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertTrue(persisted["dry_run"])
        self.assertTrue(persisted["stream"]["enabled"])
        self.assertTrue(persisted["coordinator_ready"]["ok"])
        self.assertFalse(persisted["stage_preflight"]["checked"])
        self.assertEqual(persisted["stage_preflight"]["missing_summary"], "not_checked")
        self.assertTrue(persisted["ready_to_submit"]["ok"])
        self.assertFalse(persisted["ready_to_submit"]["fully_verified"])
        self.assertEqual(persisted["ready_to_submit"]["readiness_label"], "partial")
        self.assertEqual(persisted["ready_to_submit"]["stage_verification"], "skipped")
        self.assertIn("crowdtensor_infer_preflight_partial", persisted["diagnosis_codes"])
        self.assertNotIn("crowdtensor_infer_preflight_ready", persisted["diagnosis_codes"])
        self.assertFalse(persisted["local_output"]["available"])
        self.assertEqual(persisted["recommended_next_command"]["label"], "check existing swarm")
        self.assertEqual(persisted["recommended_next_command"]["reason"], "verify_stage_miners")
        self.assertEqual(persisted["user_status"]["state"], "preflight-partial")
        self.assertEqual(persisted["user_status"]["next_step"], "run_stage_preflight")
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn(
            "- Status: `preflight-partial: Request can be submitted, but stage Miner readiness is not fully verified. next=run_stage_preflight recommendation=check existing swarm public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Ready to submit: label=`partial` next_step=`run_stage_preflight` fully_verified=`False`",
            markdown,
        )
        self.assertIn(
            "- Coordinator: `ready service=crowdtensord-coordinator protocol=runtime_contract_v1`",
            markdown,
        )
        self.assertIn(
            "- Stage preflight: checked=`False` ok=`None` missing=`not_checked`",
            markdown,
        )
        self.assertIn(
            "- Recommended next: `check existing swarm` reason=`verify_stage_miners` command=`crowdtensor infer '<prompt>' --mode existing",
            markdown,
        )
        self.assertNotIn("CrowdTensor user prompt", markdown)

    def test_infer_plain_dry_run_defaults_to_existing_preflight(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--dry-run",
            "--skip-live-preflight",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        self.assertEqual(args.infer_mode, "existing")
        self.assertFalse(args.infer_mode_explicit)
        with patch.object(
            cli,
            "request_json_url",
            side_effect=AssertionError("plain dry-run preflight should honor --skip-live-preflight"),
        ):
            report = cli.build_infer(args)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["mode"], "existing")
        self.assertEqual(report["ready_to_submit"]["readiness_label"], "skipped")
        self.assertEqual(report["user_status"]["state"], "preflight-partial")
        self.assertEqual(report["recommended_next_command"]["label"], "check existing swarm")
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --max-new-tokens 8 --dry-run --coordinator-url http://127.0.0.1:8787 --observer-token ${{CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}}",
            next_lines,
        )
        self.assertNotIn("CrowdTensor user prompt", json.dumps(report, sort_keys=True))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        rendered = stdout.getvalue()
        self.assertIn("  mode: existing", rendered)
        self.assertIn("  status: preflight-partial:", rendered)
        self.assertIn("  review_next: label=check existing swarm", rendered)
        self.assertNotIn("CrowdTensor user prompt", rendered)
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["mode"], "existing")
        self.assertTrue(persisted["dry_run"])
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("- Mode: `existing`", markdown)
        self.assertIn("- Dry run: `True`", markdown)
        self.assertIn("- Recommended: `check existing swarm` reason=`confirm_live_preflight`", markdown)
        self.assertNotIn("CrowdTensor user prompt", markdown)

    def test_infer_shareable_terminal_persists_artifact_scope_for_prompt_file(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_dir = Path(self._tmp_dir())
        prompt_file = prompt_dir / "private_infer_prompt.txt"
        prompt_file.write_text("CrowdTensor private infer file prompt", encoding="utf-8")
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            cli.main([
                "infer",
                "--mode",
                "existing",
                "--prompt-file",
                str(prompt_file),
                "--shareable-terminal",
                "--coordinator-url",
                "http://127.0.0.1:8787",
                "--dry-run",
                "--skip-live-preflight",
                "--output-dir",
                str(output_dir),
            ])

        self.assertEqual(raised.exception.code, 0)
        rendered = stdout.getvalue()
        progress = stderr.getvalue()
        self.assertNotIn(str(prompt_file), rendered)
        self.assertNotIn(str(prompt_file), progress)
        self.assertNotIn("prompt_scope:", rendered)
        self.assertIn("--prompt-file prompt.txt", rendered)
        self.assertIn(
            "  shareable_terminal: enabled=True prompt_sources_redacted=True answer_text_redacted=False public_artifact_safe=True",
            rendered,
        )
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        source_persisted = json.loads((output_dir / "generate" / "generate_summary.json").read_text(encoding="utf-8"))
        expected_shareable_terminal = {
            "enabled": True,
            "prompt_sources_redacted": True,
            "answer_text_redacted": False,
            "public_artifact_safe": True,
        }
        self.assertEqual(persisted["shareable_terminal"], expected_shareable_terminal)
        self.assertEqual(source_persisted["shareable_terminal"], expected_shareable_terminal)
        self.assertEqual(persisted["prompt_scope"]["source"], "prompt-file")
        self.assertTrue(persisted["prompt_scope"]["terminal_local_paths"])
        self.assertNotIn(str(prompt_file), json.dumps(persisted, sort_keys=True))
        self.assertNotIn(str(prompt_file), json.dumps(source_persisted, sort_keys=True))
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        source_markdown = (output_dir / "generate" / "generate_summary.md").read_text(encoding="utf-8")
        shareable_line = "- Shareable terminal: `enabled=True prompt_sources_redacted=True answer_text_redacted=False public_artifact_safe=True`"
        shareable_prompt_scope = (
            "- Terminal prompt scope: `--shareable-terminal` hid inline prompts, local prompt file paths, "
            "and local answer text from terminal logs; saved JSON/Markdown keep placeholders."
        )
        local_private_prompt_scope = "Treat terminal logs as local-private"
        self.assertIn(shareable_line, markdown)
        self.assertIn(shareable_line, source_markdown)
        self.assertIn(shareable_prompt_scope, markdown)
        self.assertIn(shareable_prompt_scope, source_markdown)
        self.assertNotIn(local_private_prompt_scope, markdown)
        self.assertNotIn(local_private_prompt_scope, source_markdown)
        self.assertNotIn(str(prompt_file), markdown)
        self.assertNotIn(str(prompt_file), source_markdown)

    def test_infer_existing_dry_run_can_skip_live_preflight_for_ci(self) -> None:
        output_dir = Path(self._tmp_dir())
        argv = [
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--dry-run",
            "--skip-live-preflight",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
        args = cli.parse_args(argv)

        with patch.object(
            cli,
            "request_json_url",
            side_effect=AssertionError("skip-live-preflight should not touch live Coordinator endpoints"),
        ):
            report = cli.build_infer(args)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["dry_run"])
        self.assertIsNone(report["ready_to_submit"]["ok"])
        self.assertFalse(report["ready_to_submit"]["fully_verified"])
        self.assertEqual(report["ready_to_submit"]["readiness_label"], "skipped")
        self.assertEqual(report["ready_to_submit"]["next_step"], "run_live_preflight")
        self.assertEqual(report["coordinator_ready"]["reason"], "live_preflight_skipped")
        self.assertEqual(report["stage_preflight"]["reason"], "live_preflight_skipped")
        self.assertIn("coordinator_ready_preflight_skipped", report["diagnosis_codes"])
        self.assertIn("stage_preflight_skipped", report["diagnosis_codes"])
        self.assertIn("crowdtensor_infer_preflight_partial", report["diagnosis_codes"])
        self.assertNotIn("crowdtensor_infer_preflight_ready", report["diagnosis_codes"])
        self.assertEqual(
            report["operator_action"],
            "Inference request shape is valid, but live readiness was skipped; rerun --dry-run without --skip-live-preflight before submitting.",
        )
        self.assertEqual(report["recommended_next_command"]["label"], "check existing swarm")
        self.assertEqual(report["recommended_next_command"]["reason"], "confirm_live_preflight")
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --max-new-tokens 8 --dry-run --coordinator-url http://127.0.0.1:8787 --observer-token ${{CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}}",
            next_lines,
        )
        check_command = next(item for item in report["next_commands"] if item["label"] == "check existing swarm")
        self.assertNotIn("--skip-live-preflight", check_command["command_line"])
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --max-new-tokens 8 --coordinator-url http://127.0.0.1:8787",
            next_lines,
        )
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["ready_to_submit"]["readiness_label"], "skipped")
        self.assertEqual(persisted["ready_to_submit"]["next_step"], "run_live_preflight")
        self.assertNotIn("CrowdTensor user prompt", json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("- State: `preflight-partial`", markdown)
        self.assertIn("- Next step: `run_live_preflight`", markdown)
        self.assertIn("- Recommended: `check existing swarm` reason=`confirm_live_preflight`", markdown)
        self.assertIn("status=`not_checked label=skipped fully_verified=False route=True coordinator=not_checked stage=skipped", markdown)
        self.assertIn("live readiness was skipped", markdown)
        self.assertNotIn("CrowdTensor user prompt", markdown)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        rendered = stdout.getvalue()
        self.assertIn("  coordinator_ready: not_checked service=none protocol=none reason=live_preflight_skipped", rendered)
        self.assertIn(
            "  ready_to_submit: not_checked label=skipped fully_verified=False route=True coordinator=not_checked stage=skipped stage_verification=skipped next_step=run_live_preflight warnings=coordinator_preflight_skipped,stage_preflight_skipped",
            rendered,
        )
        self.assertNotIn("ready_to_submit: None", rendered)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            cli.print_infer_start_hint(cli.parse_args([item for item in argv if item != "--json"]))
        progress = stderr.getvalue()
        self.assertIn("checking request shape only", progress)
        self.assertIn("live Coordinator and stage readiness are skipped", progress)
        self.assertNotIn("checking the existing route before submitting work", progress)

    def test_infer_batch_file_dry_run_preflight_recommended_before_timeout_retry(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_file = Path(self._tmp_dir()) / "private_prompts.txt"
        prompt_file.write_text("first private prompt\nsecond private prompt\n", encoding="utf-8")
        args = cli.parse_args([
            "infer",
            "--mode",
            "existing",
            "--prompt-texts-file",
            str(prompt_file),
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--dry-run",
            "--skip-live-preflight",
            "--shareable-terminal",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        with patch.object(
            cli,
            "request_json_url",
            side_effect=AssertionError("skip-live-preflight should not touch live Coordinator endpoints"),
        ):
            report = cli.build_infer(args)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["dry_run"])
        self.assertNotIn("generation_timeout", report["diagnosis_codes"])
        self.assertEqual(report["ready_to_submit"]["next_step"], "run_live_preflight")
        self.assertEqual(report["recommended_next_command"]["label"], "check existing swarm")
        self.assertEqual(report["recommended_next_command"]["reason"], "confirm_live_preflight")
        self.assertFalse(any(item["label"] == "retry inference with longer timeout" for item in report["next_commands"]))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        rendered = stdout.getvalue()
        self.assertIn("recommended_next: check existing swarm reason=confirm_live_preflight", rendered)
        self.assertNotIn("recommended_next: retry inference with longer timeout", rendered)
        self.assertIn("--prompt-texts-file prompts.txt", rendered)
        self.assertNotIn(str(prompt_file), rendered)
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["recommended_next_command"]["label"], "check existing swarm")
        self.assertEqual(persisted["recommended_next_command"]["reason"], "confirm_live_preflight")
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("- Recommended: `check existing swarm` reason=`confirm_live_preflight`", markdown)
        self.assertIn("- Prompt input: saved Markdown keeps the `prompts.txt` placeholder", markdown)
        self.assertNotIn(str(prompt_file), markdown)

    def test_infer_existing_p2p_preserves_swarm_id_in_next_commands(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--p2p",
            "--swarm-id",
            "public-swarm-v2",
            "--peer-bootstrap",
            "http://127.0.0.1:8799",
            "--admin-token",
            "admin-secret",
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        with patch.object(cli, "fetch_peer_catalog", side_effect=OSError("offline")), patch.object(
            cli,
            "request_json_url",
            side_effect=AssertionError("session creation should be blocked when discovery is offline"),
        ):
            report = cli.build_infer(args)

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["p2p"]["swarm_id"], "public-swarm-v2")
        self.assertFalse(report["ready_to_submit"]["ok"])
        self.assertFalse(report["ready_to_submit"]["fully_verified"])
        self.assertEqual(report["ready_to_submit"]["readiness_label"], "blocked")
        self.assertFalse(report["ready_to_submit"]["route_ready"])
        self.assertIsNone(report["ready_to_submit"]["coordinator_ready"])
        self.assertFalse(report["ready_to_submit"]["stage_preflight_required"])
        self.assertEqual(report["ready_to_submit"]["stage_verification"], "not_checked")
        self.assertEqual(
            report["ready_to_submit"]["warning_codes"],
            ["route_not_ready", "stage_preflight_not_checked"],
        )
        self.assertIn("stage_preflight_not_checked", report["diagnosis_codes"])
        self.assertNotIn("stage_preflight_skipped", report["diagnosis_codes"])
        self.assertNotIn("stage_preflight_failed", report["diagnosis_codes"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn("crowdtensor p2pd --port 8799 --swarm-id public-swarm-v2 --run", next_lines)
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --max-new-tokens 8 --dry-run --peer-bootstrap http://127.0.0.1:8799 --p2p --swarm-id public-swarm-v2 --observer-token ${{CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}}",
            next_lines,
        )
        self.assertNotIn("CrowdTensor user prompt", json.dumps(report, sort_keys=True))
        self.assertNotIn("admin-secret", json.dumps(report, sort_keys=True))

    def test_infer_existing_dry_run_with_observer_token_checks_stage_state(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--observer-token",
            "observer-secret",
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--json",
        ])
        calls: list[tuple[str, str, str, str, str]] = []

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            observer_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del payload, timeout
            calls.append((method, base_url, path, admin_token, observer_token))
            if path == "/ready":
                return {
                    "schema": "ready_v1",
                    "service": "crowdtensord-coordinator",
                    "protocol": "runtime_contract_v1",
                    "auth": {"observer_required": True},
                }
            if path == "/state":
                self.assertEqual(observer_token, "observer-secret")
                return {
                    "miner_profiles": {
                        "stage0-miner": {
                            "runtime": "python-cli",
                            "backend": "cpu",
                            "last_capabilities": {
                                "runtime": "python-cli",
                                "backend": "cpu",
                                "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"],
                            },
                        },
                        "stage1-miner": {
                            "runtime": "python-cli",
                            "backend": "cpu",
                            "last_capabilities": {
                                "runtime": "python-cli",
                                "backend": "cpu",
                                "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"],
                            },
                        },
                    }
                }
            self.fail(f"unexpected request path {path}")

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_infer(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual([call[2] for call in calls], ["/ready", "/state"])
        self.assertEqual(calls[0][3], "")
        self.assertEqual(calls[0][4], "")
        self.assertEqual(calls[1][4], "observer-secret")
        self.assertTrue(report["stage_preflight"]["checked"])
        self.assertTrue(report["stage_preflight"]["ok"])
        self.assertEqual(report["stage_preflight"]["matched_capabilities"]["real_llm_sharded_stage0"], "stage0-miner")
        self.assertEqual(report["stage_preflight"]["matched_capabilities"]["real_llm_sharded_stage1"], "stage1-miner")
        self.assertTrue(report["ready_to_submit"]["ok"])
        self.assertTrue(report["ready_to_submit"]["fully_verified"])
        self.assertEqual(report["ready_to_submit"]["readiness_label"], "verified")
        self.assertEqual(
            report["ready_to_submit"]["readiness_summary"],
            "Route, Coordinator, and distinct stage Miners are verified.",
        )
        self.assertEqual(report["ready_to_submit"]["next_step"], "submit")
        self.assertTrue(report["ready_to_submit"]["route_ready"])
        self.assertTrue(report["ready_to_submit"]["coordinator_ready"])
        self.assertTrue(report["ready_to_submit"]["stage_preflight_ok"])
        self.assertTrue(report["ready_to_submit"]["stage_preflight_required"])
        self.assertEqual(report["ready_to_submit"]["stage_verification"], "ready")
        self.assertEqual(report["ready_to_submit"]["warning_codes"], [])
        self.assertIn("stage_preflight_ready", report["diagnosis_codes"])
        self.assertEqual(report["operator_action"], "Dry-run is verified; run the printed submit inference next command.")
        self.assertEqual(report["recommended_next_command"]["label"], "submit inference")
        self.assertEqual(report["recommended_next_command"]["reason"], "submit_verified_inference")
        self.assertEqual(report["recommended_next_command"]["requires_env"], ["CROWDTENSOR_ADMIN_TOKEN"])
        self.assertEqual(report["user_status"]["state"], "preflight-ready")
        self.assertEqual(report["user_status"]["headline"], "Preflight passed; submit inference next.")
        self.assertEqual(report["user_status"]["next_step"], "submit")
        self.assertEqual(report["user_status"]["recommended_label"], "submit inference")
        self.assertNotIn("observer-secret", json.dumps(report, sort_keys=True))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        rendered = stdout.getvalue()
        self.assertIn(
            "  status: preflight-ready: Preflight passed; submit inference next. next=submit recommendation=submit inference public_artifact_safe=True",
            rendered,
        )
        self.assertLess(rendered.index("  review_next: "), rendered.index("  inspect_first: "))
        self.assertLess(rendered.index("  inspect_first: "), rendered.index("  status: "))
        self.assertLess(rendered.index("  review: "), rendered.index("  status: "))
        self.assertLess(rendered.index("  status: "), rendered.index("  ok: "))
        self.assertIn("  ready_to_submit: ready label=verified fully_verified=True route=True coordinator=ready stage=ready stage_verification=ready next_step=submit warnings=none", rendered)
        self.assertIn("  readiness: Route, Coordinator, and distinct stage Miners are verified.", rendered)
        self.assertIn("recommended_next: submit inference reason=submit_verified_inference", rendered)
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertTrue(persisted["stage_preflight"]["ok"])
        self.assertTrue(persisted["ready_to_submit"]["ok"])
        self.assertTrue(persisted["ready_to_submit"]["fully_verified"])
        self.assertEqual(persisted["ready_to_submit"]["readiness_label"], "verified")
        self.assertEqual(persisted["recommended_next_command"]["label"], "submit inference")
        self.assertEqual(persisted["recommended_next_command"]["reason"], "submit_verified_inference")
        self.assertEqual(persisted["user_status"]["state"], "preflight-ready")
        self.assertEqual(persisted["user_status"]["next_step"], "submit")
        self.assertNotIn("observer-secret", json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertLess(markdown.index("- Review: "), markdown.index("- OK: "))
        self.assertLess(markdown.index("- Review: "), markdown.index("- Status: "))
        self.assertIn(f"- Inspect first: `{output_dir / 'infer_summary.md'}`", markdown)
        self.assertLess(markdown.index("- Review next: "), markdown.index("- Inspect first: "))
        self.assertLess(markdown.index("- Inspect first: "), markdown.index("- Status: "))
        self.assertLess(markdown.index("- Status: "), markdown.index("- Issue: "))
        self.assertLess(markdown.index("- Issue: "), markdown.index("- OK: "))
        self.assertIn(
            "- Status: `preflight-ready: Preflight passed; submit inference next. next=submit recommendation=submit inference public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Ready to submit: label=`verified` next_step=`submit` fully_verified=`True` status=`ready label=verified fully_verified=True route=True coordinator=ready stage=ready stage_verification=ready next_step=submit warnings=none`",
            markdown,
        )
        self.assertIn(
            "- Coordinator: `ready service=crowdtensord-coordinator protocol=runtime_contract_v1`",
            markdown,
        )
        self.assertIn(
            "- Stage preflight: checked=`True` ok=`True` missing=`none`",
            markdown,
        )
        self.assertIn(
            "- Recommended next: `submit inference` reason=`submit_verified_inference` command=`CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor infer '<prompt>' --mode existing",
            markdown,
        )
        self.assertNotIn("observer-secret", markdown)
        self.assertNotIn("CrowdTensor user prompt", markdown)

    def test_infer_existing_batch_next_commands_use_only_batch_placeholder(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-texts",
            "first private prompt,second private prompt",
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        with patch.object(cli, "request_json_url", return_value={
            "schema": "ready_v1",
            "service": "crowdtensord-coordinator",
            "protocol": "runtime_contract_v1",
        }):
            report = cli.build_infer(args)

        self.assertTrue(report["ok"], report)
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            f"crowdtensor infer --mode existing --output-dir {output_dir} --prompt-texts '<prompt-1>,<prompt-2>' --max-new-tokens 8 --dry-run --coordinator-url http://127.0.0.1:8787 --observer-token ${{CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}}",
            next_lines,
        )
        self.assertIn(
            f"crowdtensor infer --mode existing --output-dir {output_dir} --prompt-texts '<prompt-1>,<prompt-2>' --max-new-tokens 8 --coordinator-url http://127.0.0.1:8787",
            next_lines,
        )
        for line in next_lines:
            self.assertNotIn("infer '<prompt>' --mode existing", line)
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("first private prompt", encoded)
        self.assertNotIn("second private prompt", encoded)

    def test_infer_rejects_ambiguous_prompt_sources(self) -> None:
        prompt_file = Path(self._tmp_dir()) / "prompt.txt"
        prompt_file.write_text("file prompt", encoding="utf-8")
        prompts_file = Path(self._tmp_dir()) / "prompts.txt"
        prompts_file.write_text("first prompt\nsecond prompt\n", encoding="utf-8")
        cases = [
            ["infer", "positional prompt", "--prompt-text", "flag prompt"],
            ["infer", "positional prompt", "--prompt-texts", "first prompt,second prompt"],
            ["infer", "--prompt", "flag prompt", "--prompt-texts", "first prompt,second prompt"],
            ["infer", "--prompt-file", str(prompt_file), "--prompt-text", "flag prompt"],
            ["infer", "positional prompt", "--prompt-file", str(prompt_file)],
            ["infer", "--prompt-stdin", "--prompt-text", "flag prompt"],
            ["infer", "--prompt-stdin", "--prompt-file", str(prompt_file)],
            ["infer", "--prompt-stdin", "--prompt-texts", "first prompt,second prompt"],
            ["infer", "--prompt-texts-file", str(prompts_file), "--prompt-texts", "first prompt,second prompt"],
            ["infer", "--prompt-texts-file", str(prompts_file), "--prompt-file", str(prompt_file)],
        ]
        for argv in cases:
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as raised:
                cli.parse_args(argv)
            self.assertEqual(
                str(raised.exception),
                "infer accepts one prompt source: positional prompt, --prompt-text/--prompt, --prompt-file, --prompt-stdin, --prompt-texts, or --prompt-texts-file",
            )

    def test_infer_rejects_empty_prompt_stdin(self) -> None:
        with patch.object(cli.sys, "stdin", io.StringIO("")):
            with self.assertRaises(SystemExit) as raised:
                cli.parse_args(["infer", "--prompt-stdin"])

        self.assertEqual(str(raised.exception), "prompt_stdin is empty")

    def test_infer_rejects_empty_prompt_texts_file(self) -> None:
        prompt_file = Path(self._tmp_dir()) / "empty-infer-prompts.txt"
        prompt_file.write_text("\n\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as raised:
            cli.parse_args(["infer", "--prompt-texts-file", str(prompt_file)])

        self.assertEqual(str(raised.exception), "prompt_texts_file is empty")

    def test_infer_rejects_long_prompt_texts_file_line_with_line_number(self) -> None:
        prompt_file = Path(self._tmp_dir()) / "long-infer-prompts.txt"
        private_prompt = "x" * 257
        prompt_file.write_text(f"first prompt\n\n{private_prompt}\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as raised:
            cli.parse_args(["infer", "--prompt-texts-file", str(prompt_file)])

        error = str(raised.exception)
        self.assertEqual(error, "prompt_texts_file line 3 must be at most 256 characters")
        self.assertNotIn(private_prompt, error)

    def test_infer_existing_dry_run_with_observer_token_blocks_missing_stage_state(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--observer-token",
            "observer-secret",
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            observer_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del method, base_url, payload, admin_token, observer_token, timeout
            if path == "/ready":
                return {"schema": "ready_v1", "service": "crowdtensord-coordinator", "protocol": "runtime_contract_v1"}
            if path == "/state":
                return {
                    "miner_profiles": {
                        "stage0-miner": {
                            "last_capabilities": {
                                "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"],
                            },
                        },
                    }
                }
            self.fail(f"unexpected request path {path}")

        with patch.object(cli, "request_json_url", side_effect=fake_request):
            report = cli.build_infer(args)

        self.assertFalse(report["ok"], report)
        self.assertTrue(report["stage_preflight"]["checked"])
        self.assertFalse(report["stage_preflight"]["ok"])
        self.assertEqual(report["stage_preflight"]["missing_capabilities"], ["real_llm_sharded_stage1"])
        self.assertFalse(report["ready_to_submit"]["ok"])
        self.assertFalse(report["ready_to_submit"]["fully_verified"])
        self.assertEqual(report["ready_to_submit"]["readiness_label"], "blocked")
        self.assertEqual(
            report["ready_to_submit"]["readiness_summary"],
            "Request is not ready to submit; follow operator_action and rerun preflight.",
        )
        self.assertEqual(report["ready_to_submit"]["next_step"], "fix_blockers")
        self.assertEqual(report["ready_to_submit"]["stage_verification"], "failed")
        self.assertIn("stage_preflight_failed", report["ready_to_submit"]["warning_codes"])
        self.assertIn("stage_preflight_failed", report["diagnosis_codes"])
        self.assertIn("crowdtensor_infer_blocked", report["diagnosis_codes"])
        self.assertIn("stage0 and stage1", report["operator_action"])

    def test_infer_existing_dry_run_uses_p2p_route_preflight(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--p2p",
            "--peer-bootstrap",
            "http://127.0.0.1:8788",
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--json",
        ])
        catalog = {
            "peers": [
                {"role": "coordinator", "peer_id": "coord", "urls": {"coordinator": "http://127.0.0.1:8787"}},
                {
                    "role": "miner",
                    "peer_id": "stage0",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                },
                {
                    "role": "miner",
                    "peer_id": "stage1",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]},
                },
            ]
        }
        request_paths: list[str] = []

        def fake_request(
            method: str,
            base_url: str,
            path: str,
            payload: dict | None = None,
            *,
            admin_token: str = "",
            observer_token: str = "",
            timeout: float = 10.0,
        ) -> dict:
            del method, base_url, payload, admin_token, observer_token, timeout
            request_paths.append(path)
            self.assertEqual(path, "/ready")
            return {"schema": "ready_v1", "service": "crowdtensord-coordinator", "protocol": "runtime_contract_v1"}

        with patch.object(cli, "fetch_peer_catalog", return_value=catalog), patch.object(
            cli,
            "request_json_url",
            side_effect=fake_request,
        ):
            report = cli.build_infer(args)

        self.assertTrue(report["ok"], report)
        self.assertEqual(request_paths, ["/ready"])
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["route"]["route_source"], "p2p-discovery")
        self.assertTrue(report["route"]["route_ready"])
        self.assertTrue(report["coordinator_ready"]["ok"])
        self.assertEqual(report["stage_preflight"]["source"], "p2p-route")
        self.assertTrue(report["stage_preflight"]["ok"])
        self.assertEqual(report["stage_preflight"]["matched_capabilities"]["real_llm_sharded_stage0"], "stage0")
        self.assertEqual(report["stage_preflight"]["matched_capabilities"]["real_llm_sharded_stage1"], "stage1")
        self.assertIn("p2p_generate_route_ready", report["diagnosis_codes"])
        self.assertIn("stage_preflight_ready", report["diagnosis_codes"])
        self.assertIn("crowdtensor_infer_preflight_ready", report["diagnosis_codes"])
        self.assertIn("user_friendly_infer_preflight_ready", report["diagnosis_codes"])
        self.assertNotIn("crowdtensor_infer_preflight_partial", report["diagnosis_codes"])

    def test_infer_existing_dry_run_blocks_when_coordinator_ready_fails(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8792",
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        with patch.object(cli, "request_json_url", side_effect=OSError("offline")):
            report = cli.build_infer(args)

        self.assertFalse(report["ok"], report)
        self.assertTrue(report["dry_run"])
        self.assertTrue(report["route"]["route_ready"])
        self.assertFalse(report["coordinator_ready"]["ok"])
        self.assertEqual(report["coordinator_ready"]["error"], "OSError")
        self.assertIn("coordinator_ready_failed", report["diagnosis_codes"])
        self.assertIn("stage_preflight_not_checked", report["diagnosis_codes"])
        self.assertIn("crowdtensor_infer_blocked", report["diagnosis_codes"])
        self.assertNotIn("crowdtensor_infer_preflight_ready", report["diagnosis_codes"])
        self.assertNotIn("crowdtensor_infer_preflight_partial", report["diagnosis_codes"])
        self.assertNotIn("coordinator_ready_preflight_skipped", report["diagnosis_codes"])
        self.assertNotIn("stage_preflight_skipped", report["diagnosis_codes"])
        self.assertNotIn("generate_dry_run_ready", report["diagnosis_codes"])
        self.assertNotIn("generate_request_shape_ready", report["diagnosis_codes"])
        self.assertIn("Coordinator route exists", report["operator_action"])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_infer(report)
        self.assertIn(
            "  coordinator_ready: not_ready service=none protocol=none error=OSError",
            stdout.getvalue(),
        )
        self.assertIn(
            "  route: source=coordinator-url candidate=True distinct_stage_miners=not_checked",
            stdout.getvalue(),
        )
        self.assertIn(
            "  stage_preflight: checked=False ok=None matched_miners=None missing=not_checked reason=coordinator_not_ready source=not-checked",
            stdout.getvalue(),
        )
        self.assertIn("next_step=fix_blockers", stdout.getvalue())
        self.assertIn("warnings=coordinator_not_ready,stage_preflight_not_checked", stdout.getvalue())
        self.assertIn("next[5] submit inference after checks pass", stdout.getvalue())
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            "crowdtensor serve --profile cpu-real-llm --bind-host 127.0.0.1 --public-host 127.0.0.1 --port 8792 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor join --coordinator-url http://127.0.0.1:8792 --miner-id stage0-miner --stage stage0 --run",
            next_lines,
        )

    def test_infer_existing_low_loopback_port_suggests_safe_local_startup_port(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:9",
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        with patch.object(cli, "request_json_url", side_effect=OSError("offline")):
            report = cli.build_infer(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("coordinator_ready_failed", report["diagnosis_codes"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            "crowdtensor serve --profile cpu-real-llm --bind-host 127.0.0.1 --public-host 127.0.0.1 --port 8787 --run",
            next_lines,
        )
        self.assertIn(
            "crowdtensor join --coordinator-url http://127.0.0.1:8787 --miner-id stage0-miner --stage stage0 --run",
            next_lines,
        )
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --max-new-tokens 8 --dry-run --coordinator-url http://127.0.0.1:8787 --observer-token ${{CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}}",
            next_lines,
        )
        self.assertFalse(any("--port 9 --run" in line for line in next_lines))
        self.assertFalse(any("--coordinator-url http://127.0.0.1:9" in line for line in next_lines))

    def test_infer_existing_remote_ready_failure_keeps_remote_route_commands(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "https://coordinator.example:9443",
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--json",
        ])

        with patch.object(cli, "request_json_url", side_effect=OSError("offline")):
            report = cli.build_infer(args)

        self.assertFalse(report["ok"], report)
        self.assertIn("coordinator_ready_failed", report["diagnosis_codes"])
        self.assertIn("remote /ready is not reachable", report["operator_action"])
        self.assertIn("remote Coordinator service", report["operator_action"])
        self.assertEqual(report["recommended_next_command"]["label"], "check existing swarm")
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertFalse(any(line.startswith("crowdtensor serve ") for line in next_lines))
        self.assertFalse(any(line.startswith("crowdtensor join ") for line in next_lines))
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --max-new-tokens 8 --dry-run --coordinator-url https://coordinator.example:9443 --observer-token ${{CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN}}",
            next_lines,
        )
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --max-new-tokens 8 --coordinator-url https://coordinator.example:9443",
            next_lines,
        )

    def test_infer_dry_run_is_existing_mode_only(self) -> None:
        args = cli.parse_args(["infer", "prompt", "--dry-run"])
        self.assertEqual(args.infer_mode, "existing")
        self.assertFalse(args.infer_mode_explicit)
        with self.assertRaisesRegex(SystemExit, "omit --mode or use --mode existing"):
            cli.parse_args(["infer", "prompt", "--mode", "local", "--dry-run"])

    def test_infer_token_limits_match_mode(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args(["infer", "prompt", "--max-new-tokens", "16"])
        args = cli.parse_args(["infer", "prompt", "--max-new-tokens", "2"])
        self.assertEqual(args.max_new_tokens, 2)
        full_args = cli.parse_args(["infer", "prompt", "--full-evidence", "--max-new-tokens", "16"])
        self.assertEqual(full_args.max_new_tokens, 16)
        args = cli.parse_args([
            "infer",
            "prompt",
            "--mode",
            "existing",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
        ])
        self.assertEqual(args.max_new_tokens, 2)

    def test_public_swarm_v2_cli_forwards_real_p2p_local_options(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-swarm-v2",
            "local",
            "--output-dir",
            str(output_dir),
            "--real-p2p-port",
            "29990",
            "--real-p2p-coordinator-port",
            "29991",
            "--real-p2p-libp2p-port",
            "29992",
            "--real-p2p-discovery-backend",
            "libp2p-kad",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertEqual(command[command.index("--real-p2p-port") + 1], "29990")
            self.assertEqual(command[command.index("--real-p2p-coordinator-port") + 1], "29991")
            self.assertEqual(command[command.index("--real-p2p-libp2p-port") + 1], "29992")
            self.assertEqual(command[command.index("--real-p2p-discovery-backend") + 1], "libp2p-kad")
            return completed({
                "schema": "public_swarm_inference_v2",
                "ok": True,
                "mode": "local",
                "diagnosis_codes": ["public_swarm_inference_v2_ready"],
            })

        report = cli.build_public_swarm_inference_v2(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)

    def test_public_swarm_v2_cli_forwards_local_model_variant(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-swarm-v2",
            "local-model-variant",
            "--output-dir",
            str(output_dir),
            "--hf-model-id",
            "distilgpt2",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_inference_v2_pack.py", command[1])
            self.assertEqual(command[2], "local-model-variant")
            self.assertEqual(command[command.index("--hf-model-id") + 1], "distilgpt2")
            return completed({
                "schema": "public_swarm_inference_v2",
                "ok": True,
                "mode": "local-model-variant",
                "diagnosis_codes": ["public_swarm_inference_v2_local_model_variant_ready"],
            })

        report = cli.build_public_swarm_inference_v2(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "public_swarm_inference_v2_cli_v1")

    def test_public_swarm_v2_cli_forwards_bounded_prompt_batch(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-swarm-v2",
            "local",
            "--output-dir",
            str(output_dir),
            "--prompt-texts",
            "first prompt,second prompt",
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_v2_pack.py", command[1])
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
            self.assertNotIn("--prompt-text", command)
            return completed({
                "schema": "public_swarm_inference_v2",
                "ok": True,
                "mode": "local",
                "diagnosis_codes": ["public_swarm_inference_v2_ready", "public_swarm_generate_batch_ready"],
            })

        report = cli.build_public_swarm_inference_v2(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "public_swarm_inference_v2_cli_v1")
        self.assertTrue(calls)

    def test_public_swarm_v2_cli_forwards_prompt_texts_file(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_file = output_dir / "prompts.txt"
        prompts = ["first prompt, with comma", "second prompt"]
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        args = cli.parse_args([
            "public-swarm-v2",
            "local",
            "--output-dir",
            str(output_dir),
            "--prompt-texts-file",
            str(prompt_file),
            "--json",
        ])
        self.assertEqual(args.prompt_texts_list, prompts)
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_v2_pack.py", command[1])
            self.assertIn("--prompt-texts-file", command)
            self.assertEqual(command[command.index("--prompt-texts-file") + 1], str(prompt_file))
            self.assertNotIn("--prompt-texts", command)
            self.assertNotIn("--prompt-text", command)
            command_text = " ".join(command)
            self.assertNotIn(prompts[0], command_text)
            self.assertNotIn(prompts[1], command_text)
            return completed({
                "schema": "public_swarm_inference_v2",
                "ok": True,
                "mode": "local",
                "prompt_scope": {
                    "source": "prompt-texts-file",
                    "prompt_count": 2,
                    "raw_prompt_public": False,
                },
                "diagnosis_codes": ["public_swarm_inference_v2_ready", "public_swarm_generate_batch_ready"],
            })

        report = cli.build_public_swarm_inference_v2(args, runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "public_swarm_inference_v2_cli_v1")
        self.assertEqual(report["prompt_scope"]["source"], "prompt-texts-file")
        self.assertNotIn(prompts[0], encoded)
        self.assertNotIn(prompts[1], encoded)
        self.assertTrue(calls)

    def test_public_swarm_v2_cli_forwards_stream_generation(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-swarm-v2",
            "local",
            "--output-dir",
            str(output_dir),
            "--stream-generation",
            "--json",
        ])
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_v2_pack.py", command[1])
            self.assertIn("--stream-generation", command)
            return completed({
                "schema": "public_swarm_inference_v2",
                "ok": True,
                "mode": "local",
                "diagnosis_codes": ["public_swarm_inference_v2_ready", "public_swarm_generate_stream_ready"],
            })

        report = cli.build_public_swarm_inference_v2(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cli_schema"], "public_swarm_inference_v2_cli_v1")
        self.assertTrue(calls)

    def test_public_swarm_v2_cli_rejects_unbounded_prompt_batch(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "public-swarm-v2",
                "local",
                "--prompt-texts",
                "one,two,three,four,five",
            ])

    def test_public_swarm_v2_cli_rejects_inline_and_file_prompt_batch(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_file = output_dir / "prompts.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text("first prompt\n", encoding="utf-8")

        with self.assertRaises(SystemExit) as raised:
            cli.parse_args([
                "public-swarm-v2",
                "local",
                "--prompt-texts",
                "first prompt,second prompt",
                "--prompt-texts-file",
                str(prompt_file),
            ])

        self.assertEqual(
            str(raised.exception),
            "public-swarm-v2 accepts either --prompt-texts or --prompt-texts-file, not both",
        )

    def test_public_swarm_v2_cli_forwards_fresh_external_flag(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-swarm-v2",
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--fresh-external-attempt-report",
            "fresh-attempt.json",
            "--fresh-external-report",
            "--json",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("--fresh-external-report", command)
            self.assertEqual(command[command.index("--fresh-external-attempt-report") + 1], "fresh-attempt.json")
            return completed({
                "schema": "public_swarm_inference_v2",
                "ok": True,
                "mode": "evidence-import",
                "diagnosis_codes": ["public_swarm_inference_v2_ready"],
            })

        report = cli.build_public_swarm_inference_v2(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)

    def test_public_swarm_v2_human_summary_shows_stage_rows_and_stream_state(self) -> None:
        report = {
            "schema": "public_swarm_inference_v2",
            "ok": True,
            "mode": "evidence-import",
            "output_dir": "dist/public-swarm-inference-v2",
            "public_swarm_v2": {"ready": True},
            "user_status": {
                "state": "ready",
                "headline": "Public Swarm v2 inference evidence is ready.",
                "next_step": "review_artifacts",
                "recommended_label": "review v2 evidence",
                "public_artifact_safe": True,
            },
            "review_summary": {
                "state": "ready",
                "next_step": "review_artifacts",
                "inspect_first": "dist/public-swarm-inference-v2/public_swarm_inference_v2.md",
                "recommended_label": "review v2 evidence",
                "recommended_reason": "v2_ready",
                "next_command": "less public_swarm_inference_v2.md",
                "requires_env": [],
                "primary_code": "public_swarm_inference_v2_ready",
                "attention": "",
                "public_artifact_safe": True,
            },
            "recommended_next_command": {
                "label": "review v2 evidence",
                "reason": "v2_ready",
                "command_line": "less public_swarm_inference_v2.md",
                "requires_env": [],
                "public_artifact_safe": True,
            },
            "readiness": {
                "local_p2p_generate": {
                    "generated_token_count": 16,
                    "max_new_tokens": 16,
                    "accepted_rows": 32,
                    "accepted_rows_ready": True,
                    "kv_cache_ready": True,
                    "batch_ready": True,
                    "stream_ready": True,
                    "model": {"compatible": True},
                },
                "external_validation": {
                    "ready": True,
                    "generated_token_count": 16,
                    "max_new_tokens": 16,
                    "accepted_rows": 32,
                    "accepted_rows_ready": True,
                    "model": {"compatible": True},
                },
                "p2p_route_hardening": {
                    "preferred_route": "real-p2p",
                    "ready": True,
                    "model": {"compatible": True},
                },
                "cuda_optional": {"fail_closed_ready": True},
                "performance": {
                    "stage_latency_ready": True,
                    "throughput_summary_ready": True,
                    "memory_or_vram_summary_ready": True,
                },
            },
            "output_request": {
                "include_output": False,
                "raw_generated_text_public": False,
                "public_artifact_safe": True,
            },
            "prompt_scope": {
                "source": "prompt-file",
                "prompt_count": 1,
                "inline_prompt_text": False,
                "terminal_next_commands_local_private": False,
                "terminal_local_paths": False,
                "saved_artifacts_prompt_placeholders": True,
                "prompt_file_path_public": False,
                "raw_prompt_public": False,
                "public_artifact_safe": True,
                "summary": "Public Swarm v2 excludes prompt file paths and raw prompt text.",
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
                "summary": "This Public Swarm v2 report is shareable aggregate evidence.",
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "next_commands": [
                {
                    "label": "inspect shareable summary",
                    "command_line": "sed -n 1,220p public_swarm_inference_v2.md",
                    "public_artifact_safe": True,
                },
                {
                    "label": "run local v2 proof",
                    "command_line": "crowdtensor public-swarm-v2 local --max-new-tokens 16 --prompt-text '<prompt>'",
                    "public_artifact_safe": True,
                },
            ],
            "artifact_summary": {
                "inspect_first": "dist/public-swarm-inference-v2/public_swarm_inference_v2.md",
                "support_bundle": "dist/public-swarm-inference-v2/support_bundle.json",
                "present_artifact_count": 5,
                "artifact_count": 5,
                "public_artifact_safe": True,
            },
            "diagnosis_codes": ["public_swarm_inference_v2_ready"],
            "artifacts": {},
        }
        buf = io.StringIO()

        with contextlib.redirect_stdout(buf):
            cli.print_public_swarm_inference_v2(report)
        output = buf.getvalue()

        self.assertIn("local accepted rows: 32 ready=True", output)
        self.assertIn(
            "  status: ready: Public Swarm v2 inference evidence is ready. next=review_artifacts recommendation=review v2 evidence public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  review: state=ready next=review_artifacts inspect=dist/public-swarm-inference-v2/public_swarm_inference_v2.md recommended=review v2 evidence primary=public_swarm_inference_v2_ready attention=none public_artifact_safe=True",
            output,
        )
        self.assertIn("  review_next: label=review v2 evidence reason=v2_ready command=less public_swarm_inference_v2.md", output)
        self.assertIn("  recommended_next: review v2 evidence reason=v2_ready less public_swarm_inference_v2.md", output)
        self.assertIn("kv cache ready: True", output)
        self.assertIn("batch ready: True", output)
        self.assertIn("stream ready: True", output)
        self.assertIn("external ready: True tokens=16/16 accepted_rows=32 rows_ready=True", output)
        self.assertIn("model match: local=True external=True p2p=True", output)
        self.assertIn(
            "  prompt_scope: source=prompt-file count=1 inline_prompt_text=False",
            output,
        )
        self.assertIn(
            "  prompt_scope_note: Public Swarm v2 excludes prompt file paths and raw prompt text.",
            output,
        )
        self.assertIn(
            "  output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope_note: This Public Swarm v2 report is shareable aggregate evidence.",
            output,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            output,
        )
        self.assertIn("  next[1] inspect shareable summary: sed -n 1,220p public_swarm_inference_v2.md", output)
        self.assertIn("  next[2] run local v2 proof: crowdtensor public-swarm-v2 local --max-new-tokens 16 --prompt-text '<prompt>'", output)
        self.assertIn("  inspect_first: dist/public-swarm-inference-v2/public_swarm_inference_v2.md", output)
        self.assertIn(
            "  artifacts: present=5/5 support=dist/public-swarm-inference-v2/support_bundle.json public_artifact_safe=True",
            output,
        )

    def test_peer_check_wraps_discovery_check(self) -> None:
        args = cli.parse_args(["peer", "check", "--json"])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("p2p_lite_discovery_check.py", command[1])
            return completed({"schema": "p2p_lite_discovery_check_v1", "ok": True})

        report = cli.build_peer_cli(args, runner=fake_runner)

        self.assertTrue(report["ok"])
        self.assertEqual(report["schema"], "p2p_lite_discovery_check_v1")

    def test_home_infer_wraps_evidence_pack_and_writes_safe_summary(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("home_compute_evidence_pack.py", command[1])
            (output_dir / "home_compute_evidence.json").write_text("{}", encoding="utf-8")
            (output_dir / "home_compute_evidence.md").write_text("# Evidence\n", encoding="utf-8")
            return completed({
                "ok": True,
                "schema": "home_compute_evidence_v1",
                "diagnosis_codes": ["home_compute_ready"],
                "route_decision": {
                    "name": "local_cpu_model_bundle_infer",
                    "target": "cpu_baseline",
                    "workload": "model_bundle_infer",
                    "confidence": "ready",
                    "usable_now": True,
                },
                "inference_summary": {
                    "present": True,
                    "ok": True,
                    "workload_type": "model_bundle_infer",
                    "scenario_schema": "model_bundle_inference_scenario_v1",
                    "scenario_id": "route-baseline",
                    "scenario_description": "Fixed CPU read-only route prompts from the built-in bundle corpus.",
                    "scenario_request_count": 8,
                    "request_count": 4,
                    "request_trace_count": 4,
                    "requests_per_second": 123.4,
                    "read_only": True,
                    "redaction_ok": True,
                },
            })

        args = cli.parse_args([
            "home-infer",
            "--output-dir",
            str(output_dir),
            "--port",
            "9010",
            "--request-count",
            "4",
        ])

        summary = cli.build_home_inference(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "home_inference_cli_v1")
        self.assertEqual(summary["evidence_schema"], "home_compute_evidence_v1")
        self.assertEqual(summary["route"]["name"], "local_cpu_model_bundle_infer")
        self.assertEqual(summary["diagnosis_codes"], ["home_compute_ready"])
        self.assertEqual(summary["scenario"]["scenario_id"], "route-baseline")
        self.assertEqual(summary["scenario"]["scenario_schema"], "model_bundle_inference_scenario_v1")
        self.assertEqual(summary["inference"]["request_trace_count"], 4)
        self.assertTrue(summary["artifacts"]["home_compute_evidence_json"]["present"])
        self.assertTrue((output_dir / "home_inference_cli_summary.json").is_file())
        self.assertTrue(any("--json-out" in command for command in calls))
        self.assertTrue(any("--scenario-id" in command and "route-baseline" in command for command in calls))

    def test_home_infer_forwards_runtime_report(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("--runtime-report", command)
            self.assertIn("/tmp/runtime.json", command)
            return completed({"ok": True, "schema": "home_compute_evidence_v1"})

        args = cli.parse_args([
            "home-infer",
            "--output-dir",
            str(output_dir),
            "--runtime-report",
            "/tmp/runtime.json",
        ])

        summary = cli.build_home_inference(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)

    def test_home_infer_failure_preserves_diagnosis_and_redacts_sensitive_payloads(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            return completed({
                "ok": False,
                "schema": "home_compute_evidence_v1",
                "diagnosis_codes": ["trace_missing"],
                "inference_results": [{"raw": "payload"}],
                "lease_token": "secret-lease",
            }, returncode=1)

        args = cli.parse_args(["home-infer", "--output-dir", str(output_dir)])

        summary = cli.build_home_inference(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertFalse(summary["ok"])
        self.assertIn("trace_missing", summary["diagnosis_codes"])
        self.assertNotIn("secret-lease", serialized)
        self.assertNotIn("lease_token", serialized)
        self.assertNotIn("inference_results", serialized)

    def test_main_home_infer_json_outputs_summary(self) -> None:
        summary = {"schema": "home_inference_cli_v1", "ok": True}
        with patch.object(cli, "build_home_inference", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["home-infer", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "home_inference_cli_v1")

    def test_llm_infer_wraps_external_llm_evidence_and_redacts_runtime_secrets(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("external_llm_evidence_pack.py", command[1])
            (output_dir / "external_llm_evidence.json").write_text("{}", encoding="utf-8")
            (output_dir / "external_llm_evidence.md").write_text("# LLM Evidence\n", encoding="utf-8")
            return completed({
                "ok": True,
                "schema": "external_llm_evidence_v1",
                "diagnosis_codes": ["external_llm_evidence_ready"],
                "adapter": {
                    "kind": "http_openai_chat",
                    "model_id": "local-model",
                    "operator_owned_runtime": True,
                },
                "summary": {
                    "request_count": 3,
                    "completion_count": 3,
                    "output_chars": 128,
                    "requests_per_second": 12.5,
                },
            })

        args = cli.parse_args([
            "llm-infer",
            "--output-dir",
            str(output_dir),
            "--port",
            "9019",
            "--request-count",
            "3",
            "--llm-runtime-url",
            "http://127.0.0.1:11434/v1/chat/completions",
            "--llm-runtime-api-key",
            "secret-api-key",
            "--llm-runtime-model-id",
            "local-model",
        ])

        summary = cli.build_llm_inference(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "llm_inference_cli_v1")
        self.assertEqual(summary["evidence_schema"], "external_llm_evidence_v1")
        self.assertEqual(summary["adapter"]["kind"], "http_openai_chat")
        self.assertEqual(summary["inference"]["completion_count"], 3)
        self.assertEqual(summary["diagnosis_codes"], ["external_llm_evidence_ready"])
        self.assertTrue(summary["artifacts"]["external_llm_evidence_json"]["present"])
        self.assertTrue((output_dir / "llm_inference_cli_summary.json").is_file())
        self.assertTrue(any("--llm-runtime-url" in command for command in calls))
        self.assertNotIn("secret-api-key", serialized)
        self.assertNotIn("http://127.0.0.1:11434", serialized)

    def test_llm_infer_rejects_conflicting_runtime_modes(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "llm-infer",
                "--llm-runtime-cmd",
                "/bin/echo",
                "--llm-runtime-url",
                "http://127.0.0.1:11434/v1/chat/completions",
            ])

    def test_main_llm_infer_json_outputs_summary(self) -> None:
        summary = {"schema": "llm_inference_cli_v1", "ok": True}
        with patch.object(cli, "build_llm_inference", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["llm-infer", "--mock", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "llm_inference_cli_v1")

    def test_cpu_infer_wraps_beta_pack_and_redacts_runtime_secrets(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("cpu_inference_beta_pack.py", command[1])
            self.assertIn("--mode", command)
            self.assertIn("remote-existing", command)
            self.assertIn("--workload", command)
            self.assertIn("external-llm", command)
            return completed({
                "schema": "cpu_inference_beta_v1",
                "ok": True,
                "mode": "remote-existing",
                "diagnosis_codes": ["cpu_inference_beta_ready"],
                "steps": [{"name": "remote_existing_external_llm_verify", "ok": True}],
                "step": {"stderr_tail": "observer-secret admin-secret runtime-secret http://127.0.0.1:11434"},
            })

        args = cli.parse_args([
            "cpu-infer",
            "--mode",
            "remote-existing",
            "--workload",
            "external-llm",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--llm-runtime-url",
            "http://127.0.0.1:11434",
            "--llm-runtime-api-key",
            "runtime-secret",
            "--output-dir",
            str(output_dir),
        ])

        summary = cli.build_cpu_inference_beta(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "cpu_inference_beta_v1")
        self.assertEqual(summary["cli_schema"], "cpu_inference_beta_cli_v1")
        self.assertIn("cpu_inference_beta_ready", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertNotIn("runtime-secret", serialized)
        self.assertNotIn("http://127.0.0.1:11434", serialized)
        self.assertTrue(calls)

    def test_cpu_infer_remote_existing_requires_auth(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "cpu-infer",
                "--mode",
                "remote-existing",
                "--coordinator-url",
                "https://coord.example",
            ])

    def test_cpu_infer_beta_rc_wraps_rc_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("cpu_inference_beta_rc_pack.py", command[1])
            self.assertIn("--kaggle-real-runtime-report", command)
            self.assertIn("/tmp/kaggle-real.json", command)
            return completed({
                "schema": "cpu_inference_beta_rc_v1",
                "ok": True,
                "mode": "beta-rc",
                "diagnosis_codes": ["cpu_inference_beta_rc_ready"],
            })

        args = cli.parse_args([
            "cpu-infer",
            "--mode",
            "beta-rc",
            "--output-dir",
            str(output_dir),
            "--kaggle-real-runtime-report",
            "/tmp/kaggle-real.json",
        ])

        summary = cli.build_cpu_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "cpu_inference_beta_rc_v1")
        self.assertEqual(summary["cli_schema"], "cpu_inference_beta_rc_cli_v1")
        self.assertIn("cpu_inference_beta_rc_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_main_cpu_infer_json_outputs_summary(self) -> None:
        summary = {"schema": "cpu_inference_beta_v1", "ok": True}
        with patch.object(cli, "build_cpu_inference_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["cpu-infer", "--mode", "local", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "cpu_inference_beta_v1")

    def test_shard_infer_wraps_evidence_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("sharded_inference_evidence_pack.py", command[1])
            self.assertIn("--failure-mode", command)
            self.assertIn("kill-stage-after-claim", command)
            return completed({
                "schema": "sharded_inference_evidence_v1",
                "ok": True,
                "diagnosis_codes": [
                    "sharded_inference_ready",
                    "stage_0_accepted",
                    "stage_1_accepted",
                    "baseline_match",
                    "activation_transport_ready",
                    "stage_requeue_ready",
                ],
                "session": {"session_id": "shard-session-test", "stage_count": 2},
                "stage_summary": {"stage_1": {"baseline_match": True}},
                "safety": {"read_only": True, "redaction_ok": True},
            })

        args = cli.parse_args([
            "shard-infer",
            "--output-dir",
            str(output_dir),
            "--failure-mode",
            "kill-stage-after-claim",
        ])

        summary = cli.build_sharded_inference(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "sharded_inference_cli_v1")
        self.assertIn("sharded_inference_ready", summary["diagnosis_codes"])
        self.assertTrue(summary["artifacts"]["sharded_inference_cli_summary"]["present"])
        self.assertTrue(calls)

    def test_main_shard_infer_json_outputs_summary(self) -> None:
        summary = {"schema": "sharded_inference_cli_v1", "ok": True}
        with patch.object(cli, "build_sharded_inference", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["shard-infer", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "sharded_inference_cli_v1")

    def test_micro_llm_shard_infer_wraps_evidence_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("micro_llm_sharded_inference_evidence_pack.py", command[1])
            self.assertIn("--decode-steps", command)
            self.assertIn("--stage-mode", command)
            self.assertIn("--micro-llm-artifact", command)
            self.assertIn("4", command)
            return completed({
                "schema": "micro_llm_sharded_evidence_v1",
                "ok": True,
                "diagnosis_codes": [
                    "micro_llm_sharded_ready",
                    "stage_0_accepted",
                    "stage_1_accepted",
                    "baseline_match",
                    "decoded_tokens_match",
                    "activation_transport_ready",
                ],
                "session": {"session_id": "micro-llm-session-test", "stage_count": 2, "decode_steps": 4},
                "stage_summary": {"stage_1": {"baseline_match": True, "decoded_tokens_match": True}},
                "safety": {"read_only": True, "redaction_ok": True},
            })

        args = cli.parse_args([
            "micro-llm-shard-infer",
            "--output-dir",
            str(output_dir),
            "--decode-steps",
            "4",
            "--stage-mode",
            "split",
            "--require-distinct-stage-miners",
            "--micro-llm-artifact",
            str(output_dir / "artifact"),
        ])

        summary = cli.build_micro_llm_sharded_inference(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "micro_llm_sharded_cli_v1")
        self.assertIn("micro_llm_sharded_ready", summary["diagnosis_codes"])
        self.assertTrue(summary["artifacts"]["micro_llm_sharded_cli_summary"]["present"])
        self.assertTrue(calls)

    def test_micro_llm_artifact_cli_builds_artifact(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("micro_llm_artifact_pack.py", command[1])
            return completed({
                "schema": "micro_llm_artifact_v1",
                "ok": True,
                "artifact_id": "crowdtensor-micro-llm-alpha",
                "artifact_hash": "sha256:artifact",
                "artifact_version": 1,
                "manifest_path": str(output_dir / "manifest.json"),
            })

        args = cli.parse_args(["micro-llm-artifact", "--output-dir", str(output_dir)])
        summary = cli.build_micro_llm_artifact(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "micro_llm_artifact_cli_v1")
        self.assertEqual(summary["artifact_hash"], "sha256:artifact")
        self.assertTrue(summary["artifacts"]["micro_llm_artifact_cli_summary"]["present"])
        self.assertTrue(calls)

    def test_real_llm_shard_infer_wraps_evidence_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("real_llm_sharded_inference_evidence_pack.py", command[1])
            self.assertIn("--hf-model-id", command)
            self.assertIn("--stage-mode", command)
            return completed({
                "schema": "real_llm_sharded_evidence_v1",
                "ok": True,
                "diagnosis_codes": [
                    "real_llm_sharded_ready",
                    "stage_0_accepted",
                    "stage_1_accepted",
                    "baseline_match",
                    "decoded_tokens_match",
                    "activation_transport_ready",
                    "real_llm_artifact_ready",
                ],
                "session": {"session_id": "real-llm-session-test", "stage_count": 2, "model_id": "sshleifer/tiny-gpt2"},
                "artifact": {"model_id": "sshleifer/tiny-gpt2", "artifact_hash": "sha256:real"},
                "stage_summary": {"stage_1": {"baseline_match": True, "decoded_tokens_match": True}},
                "safety": {"read_only": True, "redaction_ok": True},
            })

        args = cli.parse_args([
            "real-llm-shard-infer",
            "--output-dir",
            str(output_dir),
            "--stage-mode",
            "split",
            "--require-distinct-stage-miners",
            "--hf-model-id",
            "sshleifer/tiny-gpt2",
        ])

        summary = cli.build_real_llm_sharded_inference(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "real_llm_sharded_cli_v1")
        self.assertIn("real_llm_sharded_ready", summary["diagnosis_codes"])
        self.assertTrue(summary["artifacts"]["real_llm_sharded_cli_summary"]["present"])
        self.assertTrue(calls)

    def test_main_micro_llm_shard_infer_json_outputs_summary(self) -> None:
        summary = {"schema": "micro_llm_sharded_cli_v1", "ok": True}
        with patch.object(cli, "build_micro_llm_sharded_inference", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["micro-llm-shard-infer", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "micro_llm_sharded_cli_v1")

    def test_shard_infer_beta_wraps_beta_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_sharded_inference_beta_pack.py", command[1])
            self.assertIn("--mode", command)
            self.assertIn("remote-loopback", command)
            self.assertIn("--failure-mode", command)
            return completed({
                "schema": "remote_sharded_inference_beta_v1",
                "ok": True,
                "mode": "remote-loopback",
                "diagnosis_codes": [
                    "remote_sharded_inference_ready",
                    "remote_sharded_loopback_ready",
                    "sharded_inference_ready",
                    "stage_0_accepted",
                    "stage_1_accepted",
                    "baseline_match",
                    "activation_transport_ready",
                ],
                "artifacts": {},
            })

        args = cli.parse_args([
            "shard-infer-beta",
            "--output-dir",
            str(output_dir),
            "--mode",
            "remote-loopback",
        ])
        summary = cli.build_remote_sharded_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_sharded_inference_beta_v1")
        self.assertEqual(summary["cli_schema"], "remote_sharded_inference_beta_cli_v1")
        self.assertIn("remote_sharded_inference_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_main_shard_infer_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "remote_sharded_inference_beta_v1", "ok": True}
        with patch.object(cli, "build_remote_sharded_inference_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["shard-infer-beta", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "remote_sharded_inference_beta_v1")

    def test_micro_llm_shard_infer_beta_wraps_beta_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_micro_llm_sharded_beta_pack.py", command[1])
            self.assertIn("--decode-steps", command)
            self.assertIn("--mode", command)
            self.assertIn("--stage-mode", command)
            return completed({
                "schema": "remote_micro_llm_sharded_beta_v1",
                "ok": True,
                "mode": "remote-loopback",
                "diagnosis_codes": [
                    "remote_micro_llm_sharded_ready",
                    "remote_micro_llm_sharded_loopback_ready",
                    "micro_llm_sharded_ready",
                    "stage_0_accepted",
                    "stage_1_accepted",
                    "baseline_match",
                    "decoded_tokens_match",
                    "activation_transport_ready",
                ],
                "artifacts": {},
            })

        args = cli.parse_args([
            "micro-llm-shard-infer-beta",
            "--output-dir",
            str(output_dir),
            "--mode",
            "remote-loopback",
            "--decode-steps",
            "4",
            "--stage-mode",
            "split",
            "--require-distinct-stage-miners",
        ])
        summary = cli.build_remote_micro_llm_sharded_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_micro_llm_sharded_beta_v1")
        self.assertEqual(summary["cli_schema"], "remote_micro_llm_sharded_beta_cli_v1")
        self.assertIn("remote_micro_llm_sharded_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_main_micro_llm_shard_infer_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "remote_micro_llm_sharded_beta_v1", "ok": True}
        with patch.object(cli, "build_remote_micro_llm_sharded_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["micro-llm-shard-infer-beta", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "remote_micro_llm_sharded_beta_v1")

    def test_real_llm_shard_infer_beta_wraps_beta_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_real_llm_sharded_beta_pack.py", command[1])
            self.assertIn("--hf-model-id", command)
            self.assertIn("--mode", command)
            self.assertIn("--stage-mode", command)
            return completed({
                "schema": "remote_real_llm_sharded_beta_v1",
                "ok": True,
                "mode": "remote-loopback",
                "diagnosis_codes": [
                    "remote_real_llm_sharded_ready",
                    "remote_real_llm_sharded_loopback_ready",
                    "real_llm_sharded_ready",
                    "stage_0_accepted",
                    "stage_1_accepted",
                    "baseline_match",
                    "decoded_tokens_match",
                    "activation_transport_ready",
                    "real_llm_artifact_ready",
                ],
                "artifacts": {},
            })

        args = cli.parse_args([
            "real-llm-shard-infer-beta",
            "--output-dir",
            str(output_dir),
            "--mode",
            "remote-loopback",
            "--stage-mode",
            "split",
            "--require-distinct-stage-miners",
            "--hf-model-id",
            "sshleifer/tiny-gpt2",
        ])
        summary = cli.build_remote_real_llm_sharded_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_real_llm_sharded_beta_v1")
        self.assertEqual(summary["cli_schema"], "remote_real_llm_sharded_beta_cli_v1")
        self.assertIn("remote_real_llm_sharded_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_main_real_llm_shard_infer_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "remote_real_llm_sharded_beta_v1", "ok": True}
        with patch.object(cli, "build_remote_real_llm_sharded_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["real-llm-shard-infer-beta", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "remote_real_llm_sharded_beta_v1")

    def test_swarm_infer_beta_wraps_pack_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("swarm_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "verify")
            self.assertIn("--real-internet-beta-report", command)
            return completed({
                "schema": "swarm_inference_beta_v1",
                "ok": True,
                "mode": "verify",
                "diagnosis_codes": ["swarm_inference_beta_ready", "operator-secret", "admin-secret"],
                "step": {"stderr_tail": "operator-secret admin-secret"},
            })

        args = cli.parse_args([
            "swarm-infer-beta",
            "verify",
            "--output-dir",
            str(output_dir),
            "--observer-token",
            "operator-secret",
            "--admin-token",
            "admin-secret",
            "--real-internet-beta-report",
            "/tmp/real_llm_internet_beta.json",
        ])
        summary = cli.build_swarm_inference_beta(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "swarm_inference_beta_v1")
        self.assertEqual(summary["cli_schema"], "swarm_inference_beta_cli_v1")
        self.assertNotIn("operator-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(calls)

    def test_swarm_infer_beta_live_wraps_public_kaggle_auto_path(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("swarm_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "live")
            self.assertIn("--public-host", command)
            self.assertEqual(command[command.index("--public-host") + 1], "24.199.118.54")
            self.assertIn("--base-port", command)
            self.assertIn("--kaggle-owner", command)
            self.assertIn("--inline-kernel-payload", command)
            self.assertNotIn("--keep-live-private-artifacts", command)
            return completed({
                "schema": "swarm_inference_beta_v1",
                "ok": True,
                "mode": "live",
                "diagnosis_codes": ["swarm_inference_beta_live_ready"],
            })

        args = cli.parse_args([
            "swarm-infer-beta",
            "live",
            "--output-dir",
            str(output_dir),
            "--public-host",
            "24.199.118.54",
            "--port",
            "9210",
            "--base-port",
            "9211",
            "--kaggle-owner",
            "xuyuhaosuyi",
        ])
        summary = cli.build_swarm_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "swarm_inference_beta_v1")
        self.assertEqual(summary["cli_schema"], "swarm_inference_beta_cli_v1")
        self.assertTrue(calls)

        args_keep = cli.parse_args([
            "swarm-infer-beta",
            "live",
            "--output-dir",
            str(output_dir),
            "--keep-live-private-artifacts",
        ])

        def fake_runner_keep(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("--keep-live-private-artifacts", command)
            return completed({"schema": "swarm_inference_beta_v1", "ok": True, "mode": "live"})

        cli.build_swarm_inference_beta(args_keep, runner=fake_runner_keep)

    def test_main_swarm_infer_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "swarm_inference_beta_v1", "ok": True}
        with patch.object(cli, "build_swarm_inference_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["swarm-infer-beta", "clean", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "swarm_inference_beta_v1")

    def test_print_swarm_infer_beta_outputs_scope_summary(self) -> None:
        report = {
            "schema": "swarm_inference_beta_v1",
            "cli_schema": "swarm_inference_beta_cli_v1",
            "ok": True,
            "mode": "live",
            "output_request": {
                "include_output": False,
                "raw_generated_text_public": False,
                "public_artifact_safe": True,
            },
            "prompt_scope": {
                "source": "prompt-text",
                "prompt_count": 1,
                "inline_prompt_text": True,
                "terminal_next_commands_local_private": True,
                "terminal_local_paths": False,
                "saved_artifacts_prompt_placeholders": True,
                "prompt_file_path_public": False,
                "raw_prompt_public": False,
                "public_artifact_safe": True,
                "summary": "Raw prompt text is excluded from public artifacts.",
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "visible_in_terminal": False,
                "terminal_only": False,
                "public_artifact_safe": True,
                "summary": "This Swarm Inference Beta report is shareable operator evidence.",
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "output_dir": str(Path(self._tmp_dir())),
            "diagnosis_codes": ["swarm_inference_beta_live_ready"],
            "artifacts": {},
        }
        with patch("builtins.print") as mocked_print:
            cli.print_swarm_inference_beta(report)

        rendered = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list)
        self.assertIn("prompt_scope: source=prompt-text count=1 inline_prompt_text=True", rendered)
        self.assertIn("prompt_scope_note: Raw prompt text is excluded from public artifacts.", rendered)
        self.assertIn("output_request: include_output=False", rendered)
        self.assertIn("raw_generated_text_public=False", rendered)
        self.assertIn("answer_scope: state=no-local-answer", rendered)
        self.assertIn("answer_scope_note: This Swarm Inference Beta report is shareable operator evidence.", rendered)
        self.assertIn("generated_token_ids_public=False", rendered)
        self.assertIn("shareable: saved_artifacts=True", rendered)

    def test_swarm_session_wraps_public_alpha_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_alpha_pack.py", command[1])
            self.assertIn("--mode", command)
            self.assertEqual(command[command.index("--mode") + 1], "live-kaggle")
            self.assertIn("--failure-mode", command)
            self.assertEqual(command[command.index("--failure-mode") + 1], "kill-stage0-after-claim")
            self.assertIn("--kaggle-owner", command)
            self.assertIn("--kaggle-push-timeout-seconds", command)
            self.assertEqual(command.count("--kaggle-push-timeout-seconds"), 1)
            self.assertNotIn("--keep-child-artifacts", command)
            return completed({
                "schema": "public_swarm_inference_alpha_v1",
                "ok": True,
                "mode": "live-kaggle",
                "diagnosis_codes": ["public_swarm_inference_alpha_ready"],
            })

        args = cli.parse_args([
            "swarm-session",
            "--mode",
            "live-kaggle",
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "xuyuhaosuyi",
        ])
        summary = cli.build_public_swarm_inference_alpha(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_inference_alpha_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_inference_alpha_cli_v1")
        self.assertTrue(calls)

    def test_main_swarm_session_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_inference_alpha_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_inference_alpha", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["swarm-session", "--mode", "local-generated", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_inference_alpha_v1")

    def test_print_swarm_session_outputs_scope_summary(self) -> None:
        report = {
            "schema": "public_swarm_inference_alpha_v1",
            "cli_schema": "public_swarm_inference_alpha_cli_v1",
            "ok": True,
            "mode": "local-generated",
            "session": {
                "model_id": "sshleifer/tiny-gpt2",
                "live_external_runtime_verified": False,
                "local_stage_requeue_verified": True,
            },
            "output_request": {
                "include_output": False,
                "raw_generation_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "visible_in_terminal": False,
                "terminal_only": False,
                "public_artifact_safe": True,
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generation_public": False,
                "generation_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "output_dir": str(Path(self._tmp_dir())),
            "diagnosis_codes": ["public_swarm_inference_alpha_ready"],
            "artifacts": {},
        }
        with patch("builtins.print") as mocked_print:
            cli.print_public_swarm_inference_alpha(report)

        rendered = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list)
        self.assertIn("output_request: include_output=False", rendered)
        self.assertIn("raw_generated_text_public=False", rendered)
        self.assertIn("answer_scope: state=no-local-answer", rendered)
        self.assertIn("generated_token_ids_public=False", rendered)
        self.assertIn("shareable: saved_artifacts=True", rendered)

    def test_public_swarm_alpha_rc_wraps_rc_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_alpha_rc_pack.py", command[1])
            self.assertIn("--mode", command)
            self.assertEqual(command[command.index("--mode") + 1], "evidence-import")
            self.assertIn("--stage0-report", command)
            self.assertIn("--stage1-report", command)
            self.assertIn("--summary-report", command)
            return completed({
                "schema": "public_swarm_inference_alpha_rc_v1",
                "ok": True,
                "mode": "evidence-import",
                "diagnosis_codes": ["public_swarm_inference_alpha_rc_ready"],
            })

        args = cli.parse_args([
            "public-swarm-alpha-rc",
            "--output-dir",
            str(output_dir),
        ])
        summary = cli.build_public_swarm_inference_alpha_rc(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_inference_alpha_rc_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_inference_alpha_rc_cli_v1")
        self.assertTrue(calls)

    def test_main_public_swarm_alpha_rc_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_inference_alpha_rc_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_inference_alpha_rc", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["public-swarm-alpha-rc", "--mode", "local-smoke", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_inference_alpha_rc_v1")

    def test_print_public_swarm_alpha_rc_outputs_scope_summary(self) -> None:
        report = {
            "schema": "public_swarm_inference_alpha_rc_v1",
            "cli_schema": "public_swarm_inference_alpha_rc_cli_v1",
            "ok": True,
            "mode": "evidence-import",
            "release_candidate": {"ready": True},
            "output_request": {
                "include_output": False,
                "raw_generation_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "visible_in_terminal": False,
                "terminal_only": False,
                "public_artifact_safe": True,
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generation_public": False,
                "generation_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "output_dir": str(Path(self._tmp_dir())),
            "diagnosis_codes": ["public_swarm_inference_alpha_rc_ready"],
            "artifacts": {},
        }
        with patch("builtins.print") as mocked_print:
            cli.print_public_swarm_inference_alpha_rc(report)

        rendered = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list)
        self.assertIn("output_request: include_output=False", rendered)
        self.assertIn("raw_generated_text_public=False", rendered)
        self.assertIn("answer_scope: state=no-local-answer", rendered)
        self.assertIn("generated_token_ids_public=False", rendered)
        self.assertIn("shareable: saved_artifacts=True", rendered)

    def test_public_swarm_beta_wraps_beta_pack_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "prepare")
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            return completed({
                "schema": "public_swarm_inference_beta_v1",
                "ok": True,
                "mode": "prepare",
                "diagnosis_codes": ["public_swarm_inference_beta_ready", "operator-secret", "admin-secret"],
            })

        args = cli.parse_args([
            "public-swarm-beta",
            "prepare",
            "--output-dir",
            str(output_dir),
            "--observer-token",
            "operator-secret",
            "--admin-token",
            "admin-secret",
        ])
        summary = cli.build_public_swarm_inference_beta(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_inference_beta_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_inference_beta_cli_v1")
        self.assertNotIn("operator-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(calls)

    def test_public_swarm_beta_local_loopback_forwards_split_runtime_flags(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "local-loopback")
            self.assertIn("--base-port", command)
            self.assertIn("--hf-model-id", command)
            return completed({
                "schema": "public_swarm_inference_beta_v1",
                "ok": True,
                "mode": "local-loopback",
                "diagnosis_codes": ["public_swarm_inference_beta_ready", "local_loopback_ready"],
            })

        args = cli.parse_args([
            "public-swarm-beta",
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--base-port",
            "9290",
            "--hf-model-id",
            "sshleifer/tiny-gpt2",
        ])
        summary = cli.build_public_swarm_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "public_swarm_inference_beta_cli_v1")

    def test_public_swarm_beta_evidence_import_forwards_retained_reports(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "evidence-import")
            self.assertIn("--alpha-rc-report", command)
            self.assertIn("--stage0-report", command)
            self.assertIn("--stage1-report", command)
            self.assertIn("--summary-report", command)
            return completed({
                "schema": "public_swarm_inference_beta_v1",
                "ok": True,
                "mode": "evidence-import",
                "diagnosis_codes": ["public_swarm_inference_beta_ready", "external_live_evidence_imported"],
            })

        args = cli.parse_args([
            "public-swarm-beta",
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--alpha-rc-report",
            "/tmp/alpha_rc.json",
            "--stage0-report",
            "/tmp/stage0.json",
            "--stage1-report",
            "/tmp/stage1.json",
            "--summary-report",
            "/tmp/summary.json",
        ])
        summary = cli.build_public_swarm_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertIn("external_live_evidence_imported", summary["diagnosis_codes"])

    def test_public_swarm_beta_product_beta_forwards_product_flags(self) -> None:
        output_dir = Path(self._tmp_dir())
        gpu_report = output_dir / "gpu.json"
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "product-beta")
            self.assertIn("--gpu-report", command)
            self.assertEqual(command[command.index("--gpu-report") + 1], str(gpu_report))
            self.assertIn("--max-new-tokens", command)
            self.assertIn("--cpu-request-count", command)
            self.assertIn("--external-llm-request-count", command)
            return completed({
                "schema": "public_swarm_inference_beta_v1",
                "ok": True,
                "mode": "product-beta",
                "diagnosis_codes": [
                    "public_swarm_inference_beta_ready",
                    "public_swarm_product_beta_ready",
                    "session_protocol_ready",
                    "p2p_lite_discovery_ready",
                    "cpu_fallback_ready",
                ],
            })

        args = cli.parse_args([
            "public-swarm-beta",
            "product-beta",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(gpu_report),
            "--max-new-tokens",
            "4",
            "--cpu-request-count",
            "1",
            "--external-llm-request-count",
            "1",
        ])
        summary = cli.build_public_swarm_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["mode"], "product-beta")
        self.assertEqual(summary["cli_schema"], "public_swarm_inference_beta_cli_v1")
        self.assertIn("public_swarm_product_beta_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_main_public_swarm_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_inference_beta_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_inference_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["public-swarm-beta", "local-loopback", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_inference_beta_v1")

    def test_print_public_swarm_beta_outputs_scope_summary(self) -> None:
        report = {
            "schema": "public_swarm_inference_beta_v1",
            "cli_schema": "public_swarm_inference_beta_cli_v1",
            "ok": True,
            "mode": "product-beta",
            "beta": {"ready": True},
            "output_dir": "/tmp/public-swarm-beta",
            "output_request": {
                "include_output": False,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "public_artifact_safe": True,
            },
            "prompt_scope": {
                "source": "prompt-text",
                "prompt_count": 1,
                "inline_prompt_text": True,
                "terminal_next_commands_local_private": True,
                "terminal_local_paths": False,
                "saved_artifacts_prompt_placeholders": True,
                "prompt_file_path_public": False,
                "raw_prompt_public": False,
                "public_artifact_safe": True,
                "summary": "Public Swarm Beta excludes raw prompt text from public artifacts.",
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
                "summary": "This Public Swarm Inference Beta report is shareable evidence.",
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "diagnosis_codes": ["public_swarm_inference_beta_ready"],
            "artifacts": {},
        }

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            cli.print_public_swarm_inference_beta(report)

        output = stream.getvalue()
        self.assertIn("output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True", output)
        self.assertIn("prompt_scope: source=prompt-text count=1 inline_prompt_text=True", output)
        self.assertIn("prompt_scope_note: Public Swarm Beta excludes raw prompt text from public artifacts.", output)
        self.assertIn("answer_scope: state=no-local-answer", output)
        self.assertIn("answer_scope_note: This Public Swarm Inference Beta report is shareable evidence.", output)
        self.assertIn("saved_json=hash-only", output)
        self.assertIn("shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False", output)
        self.assertIn("generated_token_ids_public=False", output)

    def test_public_swarm_beta_rc_wraps_rc_pack_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_beta_rc_pack.py", command[1])
            self.assertEqual(command[2], "external-existing")
            self.assertIn("--coordinator-url", command)
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            self.assertIn("--max-new-tokens", command)
            return completed({
                "schema": "public_swarm_inference_beta_rc_v1",
                "ok": True,
                "mode": "external-existing",
                "diagnosis_codes": [
                    "public_swarm_inference_beta_rc_ready",
                    "observer-secret",
                    "admin-secret",
                ],
            })

        args = cli.parse_args([
            "public-swarm-beta-rc",
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            "http://127.0.0.1:9999",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--max-new-tokens",
            "2",
        ])
        summary = cli.build_public_swarm_inference_beta_rc(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_inference_beta_rc_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_inference_beta_rc_cli_v1")
        self.assertEqual(summary["mode"], "external-existing")
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_public_swarm_beta_rc_cli_forwards_bounded_prompt_batch(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []
        args = cli.parse_args([
            "public-swarm-beta-rc",
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--prompt-texts",
            "first prompt,second prompt",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_beta_rc_pack.py", command[1])
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
            self.assertNotIn("--prompt-text", command)
            return completed({
                "schema": "public_swarm_inference_beta_rc_v1",
                "ok": True,
                "mode": "local-loopback",
                "diagnosis_codes": ["public_swarm_inference_beta_rc_ready", "public_swarm_generate_batch_ready"],
            })

        summary = cli.build_public_swarm_inference_beta_rc(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "public_swarm_inference_beta_rc_cli_v1")
        self.assertTrue(calls)

    def test_public_swarm_beta_rc_cli_forwards_prompt_texts_file(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_file = output_dir / "prompts.txt"
        prompts = ["first prompt, with comma", "second prompt"]
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        calls: list[list[str]] = []
        args = cli.parse_args([
            "public-swarm-beta-rc",
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--prompt-texts-file",
            str(prompt_file),
        ])
        self.assertEqual(args.prompt_texts_list, prompts)

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_inference_beta_rc_pack.py", command[1])
            self.assertIn("--prompt-texts-file", command)
            self.assertEqual(command[command.index("--prompt-texts-file") + 1], str(prompt_file))
            self.assertNotIn("--prompt-texts", command)
            self.assertNotIn("--prompt-text", command)
            command_text = " ".join(command)
            for prompt in prompts:
                self.assertNotIn(prompt, command_text)
            return completed({
                "schema": "public_swarm_inference_beta_rc_v1",
                "ok": True,
                "mode": "local-loopback",
                "diagnosis_codes": ["public_swarm_inference_beta_rc_ready", "public_swarm_generate_batch_ready"],
            })

        summary = cli.build_public_swarm_inference_beta_rc(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "public_swarm_inference_beta_rc_cli_v1")
        self.assertTrue(calls)

    def test_public_swarm_beta_rc_cli_rejects_inline_and_file_prompt_batch(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_file = output_dir / "prompts.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text("first prompt\n", encoding="utf-8")

        with self.assertRaises(SystemExit) as raised:
            cli.parse_args([
                "public-swarm-beta-rc",
                "local-loopback",
                "--prompt-texts",
                "first prompt,second prompt",
                "--prompt-texts-file",
                str(prompt_file),
            ])

        self.assertEqual(
            str(raised.exception),
            "public-swarm-beta-rc accepts either --prompt-texts or --prompt-texts-file, not both",
        )

    def test_public_swarm_beta_rc_cli_forwards_stream_generation(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-swarm-beta-rc",
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--stream-generation",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_inference_beta_rc_pack.py", command[1])
            self.assertIn("--stream-generation", command)
            return completed({
                "schema": "public_swarm_inference_beta_rc_v1",
                "ok": True,
                "mode": "local-loopback",
                "diagnosis_codes": ["public_swarm_inference_beta_rc_ready", "public_swarm_generate_stream_ready"],
            })

        summary = cli.build_public_swarm_inference_beta_rc(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "public_swarm_inference_beta_rc_cli_v1")

    def test_main_public_swarm_beta_rc_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_inference_beta_rc_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_inference_beta_rc", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["public-swarm-beta-rc", "local-loopback", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_inference_beta_rc_v1")

    def test_print_public_swarm_beta_rc_outputs_scope_summary(self) -> None:
        report = {
            "schema": "public_swarm_inference_beta_rc_v1",
            "cli_schema": "public_swarm_inference_beta_rc_cli_v1",
            "ok": True,
            "mode": "local-loopback",
            "rc": {"ready": True},
            "output_dir": "/tmp/public-swarm-beta-rc",
            "output_request": {
                "include_output": False,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
                "summary": "This Public Swarm Inference Beta RC report is shareable evidence.",
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "diagnosis_codes": ["public_swarm_inference_beta_rc_ready"],
            "artifacts": {},
        }

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            cli.print_public_swarm_inference_beta_rc(report)

        output = stream.getvalue()
        self.assertIn("output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True", output)
        self.assertIn("answer_scope: state=no-local-answer", output)
        self.assertIn("answer_scope_note: This Public Swarm Inference Beta RC report is shareable evidence.", output)
        self.assertIn("saved_json=hash-only", output)
        self.assertIn("shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False", output)
        self.assertIn("generated_token_ids_public=False", output)

    def test_public_swarm_product_beta_wraps_product_pack_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_product_beta_pack.py", command[1])
            self.assertEqual(command[2], "external-existing")
            self.assertIn("--coordinator-url", command)
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            return completed({
                "schema": "public_swarm_product_beta_v1",
                "ok": True,
                "mode": "external-existing",
                "diagnosis_codes": [
                    "public_swarm_product_beta_ready",
                    "observer-secret",
                    "admin-secret",
                ],
            })

        args = cli.parse_args([
            "public-swarm-product-beta",
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            "http://127.0.0.1:9999",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
        ])
        summary = cli.build_public_swarm_product_beta(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_product_beta_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_product_beta_cli_v1")
        self.assertEqual(summary["mode"], "external-existing")
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(calls)

    def test_public_swarm_product_beta_cli_forwards_bounded_prompt_batch(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-swarm-product-beta",
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--prompt-texts",
            "first prompt,second prompt",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_product_beta_pack.py", command[1])
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
            self.assertNotIn("--prompt-text", command)
            return completed({
                "schema": "public_swarm_product_beta_v1",
                "ok": True,
                "mode": "local-loopback",
                "diagnosis_codes": ["public_swarm_product_beta_ready", "public_swarm_generate_batch_ready"],
            })

        summary = cli.build_public_swarm_product_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "public_swarm_product_beta_cli_v1")

    def test_public_swarm_product_beta_cli_forwards_prompt_texts_file(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_file = output_dir / "prompts.txt"
        prompts = ["first prompt, with comma", "second prompt"]
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        args = cli.parse_args([
            "public-swarm-product-beta",
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--prompt-texts-file",
            str(prompt_file),
        ])
        self.assertEqual(args.prompt_texts_list, prompts)

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_product_beta_pack.py", command[1])
            self.assertIn("--prompt-texts-file", command)
            self.assertEqual(command[command.index("--prompt-texts-file") + 1], str(prompt_file))
            self.assertNotIn("--prompt-texts", command)
            self.assertNotIn("--prompt-text", command)
            command_text = " ".join(command)
            for prompt in prompts:
                self.assertNotIn(prompt, command_text)
            return completed({
                "schema": "public_swarm_product_beta_v1",
                "ok": True,
                "mode": "local-loopback",
                "diagnosis_codes": ["public_swarm_product_beta_ready", "public_swarm_generate_batch_ready"],
            })

        summary = cli.build_public_swarm_product_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "public_swarm_product_beta_cli_v1")

    def test_public_swarm_product_beta_cli_rejects_inline_and_file_prompt_batch(self) -> None:
        output_dir = Path(self._tmp_dir())
        prompt_file = output_dir / "prompts.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text("first prompt\n", encoding="utf-8")

        with self.assertRaises(SystemExit) as raised:
            cli.parse_args([
                "public-swarm-product-beta",
                "local-loopback",
                "--prompt-texts",
                "first prompt,second prompt",
                "--prompt-texts-file",
                str(prompt_file),
            ])

        self.assertEqual(
            str(raised.exception),
            "public-swarm-product-beta accepts either --prompt-texts or --prompt-texts-file, not both",
        )

    def test_public_swarm_product_beta_cli_forwards_stream_generation(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "public-swarm-product-beta",
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--stream-generation",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_product_beta_pack.py", command[1])
            self.assertIn("--stream-generation", command)
            return completed({
                "schema": "public_swarm_product_beta_v1",
                "ok": True,
                "mode": "local-loopback",
                "diagnosis_codes": ["public_swarm_product_beta_ready", "public_swarm_generate_stream_ready"],
            })

        summary = cli.build_public_swarm_product_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "public_swarm_product_beta_cli_v1")

    def test_public_swarm_product_beta_prints_output_scope(self) -> None:
        report = {
            "schema": "public_swarm_product_beta_v1",
            "cli_schema": "public_swarm_product_beta_cli_v1",
            "ok": True,
            "mode": "local-loopback",
            "output_dir": "dist/product-beta",
            "product_beta": {"ready": True},
            "output_request": {
                "include_output": False,
                "raw_generated_text_public": False,
                "public_artifact_safe": True,
            },
            "prompt_scope": {
                "source": "prompt-text",
                "prompt_count": 1,
                "inline_prompt_text": True,
                "terminal_next_commands_local_private": True,
                "terminal_local_paths": False,
                "saved_artifacts_prompt_placeholders": True,
                "prompt_file_path_public": False,
                "raw_prompt_public": False,
                "public_artifact_safe": True,
                "summary": "Public Swarm Product Beta excludes raw prompt text from public artifacts.",
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
                "summary": "This Public Swarm Product Beta report is shareable evidence.",
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "diagnosis_codes": ["public_swarm_product_beta_ready"],
            "artifacts": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_public_swarm_product_beta(report)
        output = stdout.getvalue()

        self.assertIn(
            "  output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  prompt_scope: source=prompt-text count=1 inline_prompt_text=True",
            output,
        )
        self.assertIn(
            "  prompt_scope_note: Public Swarm Product Beta excludes raw prompt text from public artifacts.",
            output,
        )
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope_note: This Public Swarm Product Beta report is shareable evidence.",
            output,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            output,
        )

    def test_public_swarm_product_rc_prints_output_scope(self) -> None:
        report = {
            "schema": "public_swarm_product_rc_v1",
            "cli_schema": "public_swarm_product_cli_v1",
            "ok": True,
            "output_dir": "dist/product-rc",
            "product_surface_ready": True,
            "output_request": {
                "include_output": False,
                "raw_generated_text_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
                "summary": "This Public Swarm Product RC report is shareable evidence.",
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "diagnosis_codes": ["public_swarm_product_rc_ready"],
            "artifacts": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_public_swarm_product_rc(report)
        output = stdout.getvalue()

        self.assertIn(
            "  output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope_note: This Public Swarm Product RC report is shareable evidence.",
            output,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            output,
        )

    def test_main_public_swarm_product_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_product_beta_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_product_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["public-swarm-product-beta", "local-loopback", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_product_beta_v1")

    def test_preview_wraps_developer_preview_pack_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_developer_preview_pack.py", command[1])
            self.assertEqual(command[2], "external-existing")
            self.assertIn("--coordinator-url", command)
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            self.assertIn("--product-beta-report", command)
            return completed({
                "schema": "public_swarm_developer_preview_v1",
                "ok": True,
                "mode": "external-existing",
                "diagnosis_codes": [
                    "developer_preview_ready",
                    "observer-secret",
                    "admin-secret",
                ],
            })

        args = cli.parse_args([
            "preview",
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            "http://127.0.0.1:9999",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
        ])
        summary = cli.build_public_swarm_developer_preview(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_developer_preview_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_developer_preview_cli_v1")
        self.assertEqual(summary["mode"], "external-existing")
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(calls)

    def test_main_preview_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_developer_preview_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_developer_preview", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["preview", "local", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_developer_preview_v1")

    def test_public_swarm_developer_preview_prints_output_scope(self) -> None:
        report = {
            "schema": "public_swarm_developer_preview_v1",
            "cli_schema": "public_swarm_developer_preview_cli_v1",
            "ok": True,
            "mode": "local",
            "output_dir": "dist/preview",
            "developer_preview": {"ready": True},
            "user_status": {
                "state": "ready",
                "headline": "Public Swarm Developer Preview evidence is ready.",
                "next_step": "review_artifacts",
                "recommended_label": "inspect Developer Preview evidence",
                "recommended_reason": "review_artifacts",
                "not_completed_count": 0,
                "public_artifact_safe": True,
            },
            "review_summary": {
                "state": "ready",
                "next_step": "review_artifacts",
                "inspect_first": "dist/preview/public_swarm_developer_preview.md",
                "recommended_label": "inspect Developer Preview evidence",
                "recommended_reason": "review_artifacts",
                "next_command": "sed -n 1,220p dist/preview/public_swarm_developer_preview.md",
                "primary_code": "public_swarm_developer_preview_ready",
                "attention": "none",
                "not_completed_count": 0,
                "public_artifact_safe": True,
            },
            "recommended_next_command": {
                "label": "inspect Developer Preview evidence",
                "reason": "review_artifacts",
                "command_line": "sed -n 1,220p dist/preview/public_swarm_developer_preview.md",
                "public_artifact_safe": True,
            },
            "output_request": {
                "include_output": False,
                "raw_generated_text_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
                "summary": "This Public Swarm Developer Preview report is shareable evidence.",
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "next_commands": [
                {
                    "label": "inspect shareable summary",
                    "command_line": "sed -n 1,220p dist/preview/public_swarm_developer_preview.md",
                    "public_artifact_safe": True,
                },
                {
                    "label": "run local Developer Preview proof",
                    "command_line": "crowdtensor preview local --output-dir dist/preview --max-new-tokens 2 --prompt-text '<prompt>'",
                    "public_artifact_safe": True,
                },
            ],
            "artifact_summary": {
                "inspect_first": "dist/preview/public_swarm_developer_preview.md",
                "support_bundle": "dist/preview/support_bundle.json",
                "present_artifact_count": 4,
                "artifact_count": 4,
                "public_artifact_safe": True,
            },
            "diagnosis_codes": ["public_swarm_developer_preview_ready"],
            "artifacts": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_public_swarm_developer_preview(report)
        output = stdout.getvalue()

        self.assertIn(
            "  output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope_note: This Public Swarm Developer Preview report is shareable evidence.",
            output,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            output,
        )
        self.assertIn(
            "  status: ready: Public Swarm Developer Preview evidence is ready. next=review_artifacts recommendation=inspect Developer Preview evidence public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  review: state=ready next=review_artifacts inspect=dist/preview/public_swarm_developer_preview.md recommended=inspect Developer Preview evidence primary=public_swarm_developer_preview_ready attention=none public_artifact_safe=True",
            output,
        )
        self.assertIn("  recommended_next: inspect Developer Preview evidence reason=review_artifacts sed -n 1,220p dist/preview/public_swarm_developer_preview.md", output)
        self.assertIn("  next[1] inspect shareable summary: sed -n 1,220p dist/preview/public_swarm_developer_preview.md", output)
        self.assertIn("  next[2] run local Developer Preview proof: crowdtensor preview local --output-dir dist/preview --max-new-tokens 2 --prompt-text '<prompt>'", output)
        self.assertIn("  artifacts: present=4/4 support=dist/preview/support_bundle.json public_artifact_safe=True", output)

    def test_live_preview_wraps_rc_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_live_preview_rc_pack.py", command[1])
            self.assertEqual(command[2], "live-kaggle")
            self.assertIn("--failure-mode", command)
            self.assertIn("--kaggle-owner", command)
            self.assertIn("--developer-preview-report", command)
            self.assertIn("--alpha-rc-report", command)
            return completed({
                "schema": "public_swarm_live_preview_rc_v1",
                "ok": True,
                "mode": "live-kaggle",
                "diagnosis_codes": ["public_swarm_live_preview_rc_ready"],
            })

        args = cli.parse_args([
            "live-preview",
            "live-kaggle",
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "owner",
            "--failure-mode",
            "kill-stage0-after-claim",
        ])
        summary = cli.build_public_swarm_live_preview_rc(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_live_preview_rc_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_live_preview_rc_cli_v1")
        self.assertEqual(summary["mode"], "live-kaggle")
        self.assertTrue(calls)

    def test_main_live_preview_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_live_preview_rc_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_live_preview_rc", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["live-preview", "local-smoke", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_live_preview_rc_v1")

    def test_print_live_preview_outputs_scope_summary(self) -> None:
        report = {
            "schema": "public_swarm_live_preview_rc_v1",
            "cli_schema": "public_swarm_live_preview_rc_cli_v1",
            "ok": True,
            "mode": "local-smoke",
            "live_preview": {
                "ready": True,
                "external_runtime_verified": False,
                "fresh_live_kaggle_run": False,
            },
            "output_request": {
                "include_output": False,
                "raw_generated_text_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "visible_in_terminal": False,
                "terminal_only": False,
                "public_artifact_safe": True,
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "output_dir": str(Path(self._tmp_dir())),
            "diagnosis_codes": ["public_swarm_live_preview_rc_ready"],
            "artifacts": {},
        }
        with patch("builtins.print") as mocked_print:
            cli.print_public_swarm_live_preview_rc(report)

        rendered = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list)
        self.assertIn("output_request: include_output=False", rendered)
        self.assertIn("answer_scope: state=no-local-answer", rendered)
        self.assertIn("shareable: saved_artifacts=True", rendered)

    def test_operator_preview_wraps_pack_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_operator_preview_pack.py", command[1])
            self.assertEqual(command[2], "live-kaggle")
            self.assertIn("--live-stage0-report", command)
            self.assertIn("--live-stage1-report", command)
            self.assertIn("--release-readiness-report", command)
            self.assertIn("--kaggle-owner", command)
            return completed({
                "schema": "public_swarm_operator_preview_v1",
                "ok": True,
                "mode": "live-kaggle",
                "diagnosis_codes": [
                    "public_swarm_operator_preview_ready",
                    "observer-secret",
                ],
            })

        args = cli.parse_args([
            "operator-preview",
            "live-kaggle",
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "owner",
            "--failure-mode",
            "kill-stage0-after-claim",
        ])
        summary = cli.build_public_swarm_operator_preview(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_operator_preview_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_operator_preview_cli_v1")
        self.assertEqual(summary["mode"], "live-kaggle")
        self.assertNotIn("observer-secret", serialized)
        self.assertTrue(calls)

    def test_main_operator_preview_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_operator_preview_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_operator_preview", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["operator-preview", "local-smoke", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_operator_preview_v1")

    def test_public_swarm_operator_preview_prints_output_scope(self) -> None:
        report = {
            "schema": "public_swarm_operator_preview_v1",
            "cli_schema": "public_swarm_operator_preview_cli_v1",
            "ok": True,
            "mode": "local-smoke",
            "output_dir": "dist/operator-preview",
            "operator_preview": {
                "ready": True,
                "serve_join_generate_ready": True,
                "cpu_fallback_ready": True,
                "live_preview_ready": True,
                "external_runtime_verified": False,
                "external_runtime_blocked": False,
            },
            "output_request": {
                "include_output": False,
                "raw_generated_text_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
                "summary": "This Public Swarm Operator Preview report is shareable evidence.",
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "diagnosis_codes": ["public_swarm_operator_preview_ready"],
            "artifacts": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_public_swarm_operator_preview(report)
        output = stdout.getvalue()

        self.assertIn(
            "  output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope_note: This Public Swarm Operator Preview report is shareable evidence.",
            output,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            output,
        )

    def test_swarm_trial_wraps_pack_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_trial_pack.py", command[1])
            self.assertEqual(command[2], "live-kaggle")
            self.assertIn("--product-beta-report", command)
            self.assertIn("--operator-preview-report", command)
            self.assertIn("--live-stage0-report", command)
            self.assertIn("--live-stage1-report", command)
            self.assertIn("--release-readiness-report", command)
            self.assertIn("--kaggle-owner", command)
            return completed({
                "schema": "public_swarm_trial_v1",
                "ok": True,
                "mode": "live-kaggle",
                "diagnosis_codes": [
                    "public_swarm_trial_ready",
                    "observer-secret",
                ],
            })

        args = cli.parse_args([
            "swarm-trial",
            "live-kaggle",
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "owner",
            "--failure-mode",
            "kill-stage0-after-claim",
        ])
        summary = cli.build_public_swarm_trial(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_trial_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_trial_cli_v1")
        self.assertEqual(summary["mode"], "live-kaggle")
        self.assertNotIn("observer-secret", serialized)
        self.assertTrue(calls)

    def test_main_swarm_trial_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_trial_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_trial", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["swarm-trial", "evidence-import", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_trial_v1")

    def test_public_swarm_trial_prints_output_scope(self) -> None:
        report = {
            "schema": "public_swarm_trial_v1",
            "cli_schema": "public_swarm_trial_cli_v1",
            "ok": True,
            "mode": "local-loopback",
            "output_dir": "dist/swarm-trial",
            "trial": {
                "ready": True,
                "serve_join_generate_trial_ready": True,
                "degraded_cpu_fallback_ready": False,
                "gpu_generation_ready": True,
                "external_runtime_verified": False,
            },
            "output_request": {
                "include_output": False,
                "raw_generated_text_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "diagnosis_codes": ["public_swarm_trial_ready"],
            "artifacts": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_public_swarm_trial(report)
        output = stdout.getvalue()

        self.assertIn(
            "  output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            output,
        )

    def test_preview_v04_wraps_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("public_swarm_preview_v04_pack.py", command[1])
            self.assertEqual(command[2], "package")
            self.assertIn("--live-stage0-report", command)
            self.assertIn("--live-stage1-report", command)
            self.assertIn("--product-mvp-report", command)
            self.assertIn("--optional-model-id", command)
            return completed({
                "schema": "public_swarm_preview_v04_v1",
                "ok": True,
                "mode": "package",
                "diagnosis_codes": [
                    "public_swarm_preview_v04_ready",
                    "observer-secret",
                ],
            })

        args = cli.parse_args([
            "preview-v04",
            "package",
            "--output-dir",
            str(output_dir),
            "--optional-model-id",
            "distilgpt2",
        ])
        summary = cli.build_public_swarm_preview_v04(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_preview_v04_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_preview_v04_cli_v1")
        self.assertEqual(summary["mode"], "package")
        self.assertNotIn("observer-secret", serialized)
        self.assertTrue(calls)

    def test_main_preview_v04_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_preview_v04_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_preview_v04", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["preview-v04", "evidence-import", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_preview_v04_v1")

    def test_public_swarm_preview_v04_prints_output_scope(self) -> None:
        report = {
            "schema": "public_swarm_preview_v04_v1",
            "cli_schema": "public_swarm_preview_v04_cli_v1",
            "ok": True,
            "mode": "evidence-import",
            "output_dir": "dist/preview-v04",
            "preview": {
                "ready": True,
                "external_two_stage_generation_ready": True,
                "external_stage_requeue_ready": True,
                "stage_latency_ready": True,
                "throughput_summary_ready": True,
                "memory_or_vram_summary_ready": True,
                "optional_model_ready": False,
            },
            "output_request": {
                "include_output": False,
                "raw_generated_text_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "diagnosis_codes": ["public_swarm_preview_v04_ready"],
            "artifacts": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_public_swarm_preview_v04(report)
        output = stdout.getvalue()

        self.assertIn(
            "  output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            output,
        )

    def test_p2p_swarm_v06_wraps_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("p2p_swarm_inference_v06_pack.py", command[1])
            self.assertEqual(command[2], "local-smoke")
            self.assertIn("--preview-v04-report", command)
            self.assertIn("--p2p-port", command)
            return completed({
                "schema": "p2p_swarm_inference_v06_v1",
                "ok": True,
                "diagnosis_codes": ["p2p_swarm_inference_v06_ready"],
            })

        args = cli.parse_args([
            "p2p-swarm-v06",
            "local-smoke",
            "--output-dir",
            str(output_dir),
            "--hf-cache-dir",
            "/tmp/hf-cache",
        ])
        summary = cli.build_p2p_swarm_inference_v06(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "p2p_swarm_inference_v06_v1")
        self.assertEqual(summary["cli_schema"], "p2p_swarm_inference_v06_cli_v1")
        self.assertTrue(calls)
        self.assertEqual(calls[0][calls[0].index("--hf-cache-dir") + 1], "/tmp/hf-cache")

    def test_p2p_swarm_v06_prints_output_scope(self) -> None:
        report = {
            "schema": "p2p_swarm_inference_v06_v1",
            "cli_schema": "p2p_swarm_inference_v06_cli_v1",
            "ok": True,
            "mode": "evidence-import",
            "output_dir": "dist/p2p-v06",
            "p2p": {
                "ready": True,
                "hf_model_id": "sshleifer/tiny-gpt2",
                "observed_hf_model_id": "sshleifer/tiny-gpt2",
                "model_id_match": True,
                "generate_route": {"usable_now": True},
                "real_generate_ready": True,
                "stage_rescue_ready": True,
                "real_stage_rescue_ready": True,
            },
            "inference": {
                "workload_type": "real_llm_sharded_infer",
                "max_new_tokens": 16,
            },
            "output_request": {
                "include_output": False,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "public_artifact_safe": True,
            },
            "prompt_scope": {
                "source": "imported-or-built-in-validation-prompts",
                "prompt_count": 0,
                "inline_prompt_text": False,
                "terminal_next_commands_local_private": False,
                "terminal_local_paths": False,
                "saved_artifacts_prompt_placeholders": True,
                "prompt_file_path_public": False,
                "raw_prompt_public": False,
                "public_artifact_safe": True,
                "summary": "P2P v0.6 validation evidence excludes raw prompt text.",
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
                "summary": "This P2P v0.6 report is shareable evidence, not a local answer transcript.",
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "diagnosis_codes": ["p2p_swarm_inference_v06_ready"],
            "artifacts": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_p2p_swarm_inference_v06(report)
        output = stdout.getvalue()

        self.assertIn(
            "  output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  prompt_scope: source=imported-or-built-in-validation-prompts count=0 inline_prompt_text=False",
            output,
        )
        self.assertIn(
            "  prompt_scope_note: P2P v0.6 validation evidence excludes raw prompt text.",
            output,
        )
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope_note: This P2P v0.6 report is shareable evidence, not a local answer transcript.",
            output,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            output,
        )

    def test_public_p2p_v1_rc_prints_output_scope(self) -> None:
        report = {
            "schema": "public_p2p_swarm_inference_v1_rc_v1",
            "cli_schema": "public_p2p_swarm_inference_v1_rc_cli_v1",
            "ok": True,
            "mode": "evidence-import",
            "output_dir": "dist/public-p2p-v1-rc",
            "rc": {
                "signed_local_ready": True,
                "external_runtime_ready": True,
                "generation_ready": True,
                "stage_rescue_ready": True,
                "model_metadata_ready": True,
            },
            "p2p": {
                "hf_model_id": "sshleifer/tiny-gpt2",
                "signed_announcement_required": True,
                "signed_peer_count": 3,
                "healthy_peer_count": 3,
            },
            "inference": {
                "workload_type": "real_llm_sharded_infer",
                "max_new_tokens": 16,
            },
            "output_request": {
                "include_output": False,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "diagnosis_codes": ["public_p2p_swarm_inference_v1_rc_ready"],
            "artifacts": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_public_p2p_swarm_inference_v1_rc(report)
        output = stdout.getvalue()

        self.assertIn(
            "  output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            output,
        )

    def test_p2p_swarm_v06_forwards_bounded_prompt_batch(self) -> None:
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
            return completed({
                "schema": "p2p_swarm_inference_v06_v1",
                "ok": True,
                "diagnosis_codes": ["p2p_swarm_inference_v06_ready", "p2p_real_generate_batch_ready"],
            })

        args = cli.parse_args([
            "p2p-swarm-v06",
            "local-smoke",
            "--prompt-texts",
            "first prompt,second prompt",
        ])
        summary = cli.build_p2p_swarm_inference_v06(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertTrue(calls)

    def test_p2p_swarm_v06_forwards_stream_generation(self) -> None:
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("--stream-generation", command)
            return completed({
                "schema": "p2p_swarm_inference_v06_v1",
                "ok": True,
                "diagnosis_codes": ["p2p_swarm_inference_v06_ready", "p2p_real_generate_stream_ready"],
            })

        args = cli.parse_args([
            "p2p-swarm-v06",
            "local-smoke",
            "--stream-generation",
        ])
        summary = cli.build_p2p_swarm_inference_v06(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertTrue(calls)

    def test_p2p_swarm_v06_rejects_unbounded_prompt_batch(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "p2p-swarm-v06",
                "local-smoke",
                "--prompt-texts",
                "one,two,three,four,five",
            ])

    def test_p2p_swarm_v06_wraps_external_existing_options(self) -> None:
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertEqual(command[2], "external-existing")
            self.assertEqual(command[command.index("--peer-bootstrap") + 1], "http://p2p.example")
            self.assertIn("--verify-generate", command)
            self.assertEqual(command[command.index("--admin-token") + 1], "admin-secret")
            self.assertEqual(command[command.index("--hf-model-id") + 1], "distilgpt2")
            self.assertIn("--prompt-texts", command)
            self.assertIn("--stream-generation", command)
            return completed({
                "schema": "p2p_swarm_inference_v06_v1",
                "ok": True,
                "diagnosis_codes": ["p2p_swarm_inference_v06_ready"],
            })

        args = cli.parse_args([
            "p2p-swarm-v06",
            "external-existing",
            "--peer-bootstrap",
            "http://p2p.example",
            "--admin-token",
            "admin-secret",
            "--verify-generate",
            "--hf-model-id",
            "distilgpt2",
            "--prompt-texts",
            "first prompt,second prompt",
            "--stream-generation",
        ])
        summary = cli.build_p2p_swarm_inference_v06(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertTrue(calls)

    def test_p2p_swarm_v06_wraps_kaggle_auto_options(self) -> None:
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertEqual(command[2], "kaggle-auto")
            self.assertEqual(command[command.index("--kaggle-owner") + 1], "owner")
            self.assertEqual(command[command.index("--kernel-slug-prefix") + 1], "ct-p2p-v06-test")
            self.assertIn("--kaggle-push-timeout-seconds", command)
            self.assertIn("--kaggle-delete-timeout-seconds", command)
            self.assertIn("--kaggle-stage-timeout-seconds", command)
            self.assertEqual(command[command.index("--kaggle-stage-timeout-seconds") + 1], "321.0")
            return completed({
                "schema": "p2p_swarm_inference_v06_v1",
                "ok": True,
                "diagnosis_codes": ["p2p_swarm_inference_v06_ready", "p2p_swarm_inference_v06_kaggle_auto_ready"],
            })

        args = cli.parse_args([
            "p2p-swarm-v06",
            "kaggle-auto",
            "--kaggle-owner",
            "owner",
            "--kernel-slug-prefix",
            "ct-p2p-v06-test",
            "--kaggle-stage-timeout-seconds",
            "321",
        ])
        summary = cli.build_p2p_swarm_inference_v06(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertTrue(calls)

    def test_main_p2p_swarm_v06_json_outputs_summary(self) -> None:
        summary = {"schema": "p2p_swarm_inference_v06_v1", "ok": True}
        with patch.object(cli, "build_p2p_swarm_inference_v06", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["p2p-swarm-v06", "evidence-import", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "p2p_swarm_inference_v06_v1")

    def test_real_p2p_rc_wraps_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("real_p2p_swarm_inference_core_rc_pack.py", command[1])
            self.assertEqual(command[2], "local-smoke")
            self.assertEqual(command[command.index("--output-dir") + 1], str(output_dir.resolve()))
            self.assertIn("--discovery-backend", command)
            self.assertIn("--peer-secret", command)
            self.assertIn("--json", command)
            return completed({
                "schema": "real_p2p_swarm_inference_core_rc_v1",
                "ok": True,
                "diagnosis_codes": ["real_p2p_swarm_inference_core_rc_ready"],
            })

        args = cli.parse_args([
            "real-p2p-rc",
            "local-smoke",
            "--output-dir",
            str(output_dir),
            "--peer-secret",
            "test-secret",
            "--hf-cache-dir",
            "/tmp/hf-cache",
        ])
        summary = cli.build_real_p2p_swarm_inference_core_rc(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "real_p2p_swarm_inference_core_rc_v1")
        self.assertEqual(summary["cli_schema"], "real_p2p_swarm_inference_core_rc_cli_v1")
        self.assertTrue(calls)
        self.assertEqual(calls[0][calls[0].index("--hf-cache-dir") + 1], "/tmp/hf-cache")

    def test_real_p2p_rc_prints_output_scope(self) -> None:
        report = {
            "schema": "real_p2p_swarm_inference_core_rc_v1",
            "cli_schema": "real_p2p_swarm_inference_core_rc_cli_v1",
            "ok": True,
            "mode": "evidence-import",
            "output_dir": "dist/real-p2p-rc",
            "hf_model_id": "sshleifer/tiny-gpt2",
            "p2p": {
                "discovery_backend": "libp2p-kad",
                "provider_count": 3,
                "route": {"usable_now": True},
            },
            "generation": {
                "generated_token_count": 8,
                "max_new_tokens": 8,
            },
            "stage_assignment": {
                "completed_rows": 16,
                "distinct_stage_miners": True,
            },
            "external": {
                "external_runtime_verified": True,
                "external_generate_verified": True,
            },
            "live_requeue_summary": {
                "enabled": True,
                "target_stage": "stage0",
                "rescue_miner_used": True,
                "accepted_result_after_requeue": True,
            },
            "output_request": {
                "include_output": False,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "diagnosis_codes": ["real_p2p_swarm_inference_core_rc_ready"],
            "artifacts": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_real_p2p_swarm_inference_core_rc(report)
        output = stdout.getvalue()

        self.assertIn(
            "  output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            output,
        )

    def test_real_p2p_rc_external_existing_forwards_options(self) -> None:
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertEqual(command[2], "external-existing")
            self.assertEqual(command[command.index("--peer-bootstrap") + 1], "http://p2p.example")
            self.assertEqual(command[command.index("--admin-token") + 1], "admin-secret")
            self.assertEqual(command[command.index("--hf-model-id") + 1], "distilgpt2")
            self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
            self.assertIn("--verify-generate", command)
            self.assertIn("--stream-generation", command)
            return completed({
                "schema": "real_p2p_swarm_inference_core_rc_v1",
                "ok": True,
                "diagnosis_codes": ["external_real_p2p_stage_discovery_ready", "external_real_p2p_generate_ready"],
            })

        args = cli.parse_args([
            "real-p2p-rc",
            "external-existing",
            "--peer-bootstrap",
            "http://p2p.example",
            "--admin-token",
            "admin-secret",
            "--verify-generate",
            "--hf-model-id",
            "distilgpt2",
            "--prompt-texts",
            "first prompt,second prompt",
            "--stream-generation",
        ])
        summary = cli.build_real_p2p_swarm_inference_core_rc(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertTrue(calls)

    def test_real_p2p_rc_batch_stream_requires_external_verify_generate(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args(["real-p2p-rc", "local-smoke", "--prompt-texts", "a,b"])
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "real-p2p-rc",
                "external-existing",
                "--peer-bootstrap",
                "http://p2p.example",
                "--stream-generation",
            ])

    def test_real_p2p_rc_kaggle_auto_forwards_options(self) -> None:
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertEqual(command[2], "kaggle-auto")
            self.assertEqual(command[command.index("--kaggle-owner") + 1], "owner")
            self.assertEqual(command[command.index("--kernel-slug-prefix") + 1], "ct-real-p2p-test")
            self.assertEqual(command[command.index("--libp2p-port") + 1], "10860")
            self.assertIn("--kaggle-push-timeout-seconds", command)
            self.assertIn("--kaggle-delete-timeout-seconds", command)
            self.assertIn("--kaggle-stage-timeout-seconds", command)
            self.assertEqual(command[command.index("--failure-mode") + 1], "kill-stage0-after-claim")
            self.assertEqual(command[command.index("--max-request-attempts") + 1], "123")
            return completed({
                "schema": "real_p2p_swarm_inference_core_rc_v1",
                "ok": True,
                "diagnosis_codes": ["real_p2p_swarm_inference_core_rc_ready", "external_real_p2p_generate_ready"],
            })

        args = cli.parse_args([
            "real-p2p-rc",
            "kaggle-auto",
            "--kaggle-owner",
            "owner",
            "--kernel-slug-prefix",
            "ct-real-p2p-test",
            "--kaggle-stage-timeout-seconds",
            "321",
            "--libp2p-port",
            "10860",
            "--failure-mode",
            "kill-stage0-after-claim",
            "--max-request-attempts",
            "123",
        ])
        summary = cli.build_real_p2p_swarm_inference_core_rc(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertTrue(calls)

    def test_real_p2p_rc_kaggle_runtime_smoke_forwards_options(self) -> None:
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertEqual(command[2], "kaggle-runtime-smoke")
            self.assertEqual(command[command.index("--discovery-backend") + 1], "libp2p-kad")
            self.assertEqual(command[command.index("--kaggle-status-poll-seconds") + 1], "7.0")
            self.assertEqual(command[command.index("--kaggle-owner") + 1], "owner")
            return completed({
                "schema": "real_p2p_swarm_inference_core_rc_v1",
                "ok": True,
                "diagnosis_codes": ["real_p2p_kaggle_runtime_smoke_ready"],
            })

        args = cli.parse_args([
            "real-p2p-rc",
            "kaggle-runtime-smoke",
            "--discovery-backend",
            "libp2p-kad",
            "--kaggle-owner",
            "owner",
            "--kaggle-status-poll-seconds",
            "7",
        ])
        summary = cli.build_real_p2p_swarm_inference_core_rc(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertTrue(calls)

    def test_real_p2p_rc_evidence_import_requires_report(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args(["real-p2p-rc", "evidence-import"])

    def test_main_real_p2p_rc_json_outputs_summary(self) -> None:
        summary = {"schema": "real_p2p_swarm_inference_core_rc_v1", "ok": True}
        with patch.object(cli, "build_real_p2p_swarm_inference_core_rc", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["real-p2p-rc", "package", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "real_p2p_swarm_inference_core_rc_v1")

    def test_petals_candidate_wraps_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("petals_class_p2p_candidate_pack.py", command[1])
            self.assertEqual(command[2], "evidence-import")
            self.assertEqual(command[command.index("--output-dir") + 1], str(output_dir.resolve()))
            self.assertEqual(command[command.index("--requeue-report") + 1], "requeue.json")
            self.assertIn("--json", command)
            return completed({
                "schema": "petals_class_p2p_candidate_v1",
                "ok": True,
                "diagnosis_codes": ["petals_class_p2p_candidate_ready"],
            })

        args = cli.parse_args([
            "petals-candidate",
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--local-report",
            "local.json",
            "--runtime-smoke-report",
            "runtime.json",
            "--external-report",
            "external.json",
            "--requeue-report",
            "requeue.json",
            "--max-new-tokens",
            "8",
        ])
        summary = cli.build_petals_class_p2p_candidate(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "petals_class_p2p_candidate_v1")
        self.assertEqual(summary["cli_schema"], "petals_class_p2p_candidate_cli_v1")
        self.assertTrue(calls)

    def test_petals_candidate_prints_output_scope(self) -> None:
        report = {
            "schema": "petals_class_p2p_candidate_v1",
            "cli_schema": "petals_class_p2p_candidate_cli_v1",
            "ok": True,
            "mode": "evidence-import",
            "output_dir": "dist/petals-candidate",
            "candidate": {
                "external_generated_token_count": 16,
                "max_new_tokens": 16,
                "local_libp2p_ready": True,
                "kaggle_runtime_smoke_ready": True,
                "external_libp2p_generate_ready": True,
                "p2p_live_requeue_ready": True,
                "victim_result_not_accepted": True,
                "batch_ready": True,
                "stream_ready": True,
                "batch": {"expected_request_count": 2},
                "stream": {"event_count": 16},
                "live_requeue_summary": {
                    "rescue_miner_used": True,
                },
            },
            "output_request": {
                "include_output": False,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "diagnosis_codes": ["petals_class_p2p_candidate_ready"],
            "artifacts": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_petals_class_p2p_candidate(report)
        output = stdout.getvalue()

        self.assertIn(
            "  output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            output,
        )

    def test_main_petals_candidate_json_outputs_summary(self) -> None:
        summary = {"schema": "petals_class_p2p_candidate_v1", "ok": True}
        with patch.object(cli, "build_petals_class_p2p_candidate", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["petals-candidate", "package", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "petals_class_p2p_candidate_v1")

    def test_public_swarm_gpu_beta_wraps_gpu_pack(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_gpu_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "local-loopback")
            self.assertIn("--base-port", command)
            self.assertIn("--hf-model-id", command)
            return completed({
                "schema": "public_swarm_gpu_inference_beta_v1",
                "ok": True,
                "mode": "local-loopback",
                "diagnosis_codes": ["public_swarm_gpu_beta_ready", "hf_transformers_cuda_ready"],
            })

        args = cli.parse_args([
            "public-swarm-gpu-beta",
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--base-port",
            "9390",
        ])
        summary = cli.build_public_swarm_gpu_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "public_swarm_gpu_inference_beta_v1")
        self.assertEqual(summary["cli_schema"], "public_swarm_gpu_inference_beta_cli_v1")

    def test_public_swarm_gpu_beta_evidence_import_forwards_gpu_report(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_gpu_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "evidence-import")
            self.assertIn("--gpu-report", command)
            self.assertEqual(command[command.index("--gpu-report") + 1], "/tmp/gpu.json")
            return completed({
                "schema": "public_swarm_gpu_inference_beta_v1",
                "ok": True,
                "mode": "evidence-import",
                "diagnosis_codes": ["public_swarm_gpu_beta_evidence_import_ready"],
            })

        args = cli.parse_args([
            "public-swarm-gpu-beta",
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            "/tmp/gpu.json",
        ])
        summary = cli.build_public_swarm_gpu_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "public_swarm_gpu_inference_beta_cli_v1")

    def test_public_swarm_gpu_beta_kaggle_auto_forwards_kaggle_options(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("public_swarm_gpu_inference_beta_pack.py", command[1])
            self.assertEqual(command[2], "kaggle-auto")
            self.assertIn("--public-host", command)
            self.assertEqual(command[command.index("--public-host") + 1], "24.199.118.54")
            self.assertIn("--kaggle-owner", command)
            self.assertEqual(command[command.index("--kaggle-owner") + 1], "xuyuhaosuyi")
            self.assertIn("--kernel-slug-prefix", command)
            self.assertIn("--inline-kernel-payload", command)
            return completed({
                "schema": "public_swarm_gpu_inference_beta_v1",
                "ok": True,
                "mode": "kaggle-auto",
                "diagnosis_codes": ["public_swarm_gpu_beta_kaggle_auto_ready"],
            })

        args = cli.parse_args([
            "public-swarm-gpu-beta",
            "kaggle-auto",
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "xuyuhaosuyi",
            "--kernel-slug-prefix",
            "crowdtensor-public-swarm-gpu-beta-test",
        ])
        summary = cli.build_public_swarm_gpu_inference_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "public_swarm_gpu_inference_beta_cli_v1")

    def test_gpu_generate_wraps_generation_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("gpu_sharded_generation_beta_pack.py", command[1])
            self.assertIn("local-loopback", command)
            self.assertIn("--max-new-tokens", command)
            self.assertEqual(command[command.index("--max-new-tokens") + 1], "8")
            return completed({
                "schema": "gpu_sharded_generation_beta_v1",
                "ok": True,
                "diagnosis_codes": ["gpu_sharded_generation_ready"],
            })

        args = cli.parse_args([
            "gpu-generate",
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "8",
        ])

        summary = cli.build_gpu_sharded_generation_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["cli_schema"], "gpu_sharded_generation_beta_cli_v1")
        self.assertTrue(calls)

    def test_gpu_generate_evidence_import_forwards_report(self) -> None:
        output_dir = Path(self._tmp_dir())
        source = output_dir / "gpu.json"
        source.write_text("{}", encoding="utf-8")

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("evidence-import", command)
            self.assertIn("--gpu-report", command)
            self.assertEqual(command[command.index("--gpu-report") + 1], str(source))
            return completed({
                "schema": "gpu_sharded_generation_beta_v1",
                "ok": True,
                "diagnosis_codes": ["gpu_sharded_generation_ready"],
            })

        args = cli.parse_args([
            "gpu-generate",
            "evidence-import",
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(source),
            "--max-new-tokens",
            "4",
        ])

        summary = cli.build_gpu_sharded_generation_beta(args, runner=fake_runner)

        self.assertEqual(summary["cli_schema"], "gpu_sharded_generation_beta_cli_v1")

    def test_main_public_swarm_gpu_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "public_swarm_gpu_inference_beta_v1", "ok": True}
        with patch.object(cli, "build_public_swarm_gpu_inference_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["public-swarm-gpu-beta", "local-smoke", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "public_swarm_gpu_inference_beta_v1")

    def test_public_swarm_gpu_beta_prints_output_scope(self) -> None:
        report = {
            "schema": "public_swarm_gpu_inference_beta_v1",
            "cli_schema": "public_swarm_gpu_inference_beta_cli_v1",
            "ok": True,
            "mode": "local-smoke",
            "output_dir": "dist/public-swarm-gpu-beta",
            "beta": {"ready": True, "backend": "hf_transformers_cuda"},
            "output_request": {
                "include_output": False,
                "raw_generated_text_public": False,
                "public_artifact_safe": True,
            },
            "prompt_scope": {
                "source": "imported-or-built-in-validation-prompts",
                "prompt_count": 1,
                "inline_prompt_text": False,
                "terminal_next_commands_local_private": False,
                "terminal_local_paths": False,
                "saved_artifacts_prompt_placeholders": True,
                "prompt_file_path_public": False,
                "raw_prompt_public": False,
                "public_artifact_safe": True,
                "summary": "GPU generation evidence excludes raw prompt text.",
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "diagnosis_codes": ["public_swarm_gpu_beta_smoke_ready"],
            "artifacts": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_public_swarm_gpu_inference_beta(report)
        output = stdout.getvalue()

        self.assertIn(
            "  output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  prompt_scope: source=imported-or-built-in-validation-prompts count=1 inline_prompt_text=False",
            output,
        )
        self.assertIn(
            "  prompt_scope_note: GPU generation evidence excludes raw prompt text.",
            output,
        )
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            output,
        )

    def test_main_gpu_generate_json_outputs_summary(self) -> None:
        summary = {"schema": "gpu_sharded_generation_beta_v1", "ok": True}
        with patch.object(cli, "build_gpu_sharded_generation_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["gpu-generate", "local-loopback", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "gpu_sharded_generation_beta_v1")

    def test_gpu_generate_prints_output_scope(self) -> None:
        report = {
            "schema": "gpu_sharded_generation_beta_v1",
            "cli_schema": "gpu_sharded_generation_beta_cli_v1",
            "ok": True,
            "mode": "evidence-import",
            "output_dir": "dist/gpu-generate",
            "generation": {"generated_token_count": 4, "max_new_tokens": 4},
            "gpu": {"backend": "hf_transformers_cuda", "model_id": "sshleifer/tiny-gpt2"},
            "output_request": {
                "include_output": False,
                "raw_generated_text_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
                "summary": "This GPU sharded generation report is shareable evidence, not a local answer transcript.",
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "diagnosis_codes": ["gpu_sharded_generation_ready"],
            "artifacts": {},
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            cli.print_gpu_sharded_generation_beta(report)
        output = stdout.getvalue()

        self.assertIn(
            "  output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope: state=no-local-answer terminal_only=False visible_in_terminal=False saved_json=hash-only saved_markdown=hash-only public_artifact_safe=True",
            output,
        )
        self.assertIn(
            "  answer_scope_note: This GPU sharded generation report is shareable evidence, not a local answer transcript.",
            output,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False answer_scope_state=no-local-answer local_answer_terminal_only=False",
            output,
        )


    def test_micro_llm_live_rc_wraps_rc_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("micro_llm_live_rc_pack.py", command[1])
            self.assertIn("--mode", command)
            self.assertEqual(command[command.index("--mode") + 1], "local-generated")
            self.assertIn("--decode-steps", command)
            self.assertEqual(command[command.index("--decode-steps") + 1], "3")
            self.assertIn("--max-request-attempts", command)
            return completed({
                "schema": "micro_llm_live_rc_v1",
                "ok": True,
                "mode": "local-generated",
                "diagnosis_codes": [
                    "micro_llm_live_rc_ready",
                    "local_generated_stage_upload_standins_ready",
                    "kaggle_micro_llm_sharded_ready",
                ],
                "artifacts": {},
            })

        args = cli.parse_args([
            "micro-llm-live-rc",
            "--output-dir",
            str(output_dir),
            "--port",
            "9180",
            "--request-count",
            "2",
            "--decode-steps",
            "3",
        ])
        summary = cli.build_micro_llm_live_rc(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "micro_llm_live_rc_v1")
        self.assertEqual(summary["cli_schema"], "micro_llm_live_rc_cli_v1")
        self.assertIn("micro_llm_live_rc_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_micro_llm_live_rc_external_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("micro_llm_live_rc_pack.py", command[1])
            self.assertEqual(command[command.index("--mode") + 1], "external-existing")
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            self.assertIn("--coordinator-url", command)
            return completed({
                "schema": "micro_llm_live_rc_v1",
                "ok": True,
                "mode": "external-existing",
                "diagnosis_codes": ["micro_llm_live_rc_ready", "external_runtime_verified"],
                "step": {"stderr_tail": "observer-secret admin-secret"},
            })

        args = cli.parse_args([
            "micro-llm-live-rc",
            "--mode",
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            "http://24.199.118.54:9180",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
        ])
        summary = cli.build_micro_llm_live_rc(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertIn("external_runtime_verified", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_main_micro_llm_live_rc_json_outputs_summary(self) -> None:
        summary = {"schema": "micro_llm_live_rc_v1", "ok": True}
        with patch.object(cli, "build_micro_llm_live_rc", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["micro-llm-live-rc", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "micro_llm_live_rc_v1")

    def test_real_llm_live_rc_wraps_rc_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("real_llm_live_rc_pack.py", command[1])
            self.assertIn("--mode", command)
            self.assertEqual(command[command.index("--mode") + 1], "local-generated")
            self.assertIn("--hf-model-id", command)
            self.assertIn("--max-request-attempts", command)
            return completed({
                "schema": "real_llm_live_rc_v1",
                "ok": True,
                "mode": "local-generated",
                "diagnosis_codes": [
                    "real_llm_live_rc_ready",
                    "local_generated_real_llm_stage_upload_standins_ready",
                    "remote_real_llm_sharded_ready",
                ],
                "artifacts": {},
            })

        args = cli.parse_args([
            "real-llm-live-rc",
            "--output-dir",
            str(output_dir),
            "--port",
            "9184",
            "--request-count",
            "1",
            "--hf-model-id",
            "sshleifer/tiny-gpt2",
        ])
        summary = cli.build_real_llm_live_rc(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "real_llm_live_rc_v1")
        self.assertEqual(summary["cli_schema"], "real_llm_live_rc_cli_v1")
        self.assertIn("real_llm_live_rc_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_real_llm_live_rc_external_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("real_llm_live_rc_pack.py", command[1])
            self.assertEqual(command[command.index("--mode") + 1], "external-existing")
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            self.assertIn("--coordinator-url", command)
            return completed({
                "schema": "real_llm_live_rc_v1",
                "ok": True,
                "mode": "external-existing",
                "diagnosis_codes": ["real_llm_live_rc_ready", "external_runtime_verified"],
                "step": {"stderr_tail": "observer-secret admin-secret"},
            })

        args = cli.parse_args([
            "real-llm-live-rc",
            "--mode",
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            "http://24.199.118.54:9184",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
        ])
        summary = cli.build_real_llm_live_rc(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertIn("external_runtime_verified", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_main_real_llm_live_rc_json_outputs_summary(self) -> None:
        summary = {"schema": "real_llm_live_rc_v1", "ok": True}
        with patch.object(cli, "build_real_llm_live_rc", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["real-llm-live-rc", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "real_llm_live_rc_v1")

    def test_real_llm_internet_alpha_wraps_alpha_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("real_llm_internet_alpha_pack.py", command[1])
            self.assertEqual(command[command.index("--mode") + 1], "local-generated")
            self.assertIn("--base-port", command)
            self.assertIn("--hf-model-id", command)
            return completed({
                "schema": "real_llm_internet_alpha_v1",
                "ok": True,
                "mode": "local-generated",
                "diagnosis_codes": [
                    "real_llm_internet_alpha_ready",
                    "real_llm_stage_requeue_ready",
                    "real_llm_live_rc_ready",
                ],
                "artifacts": {},
            })

        args = cli.parse_args([
            "real-llm-internet-alpha",
            "--output-dir",
            str(output_dir),
            "--port",
            "9186",
            "--base-port",
            "9188",
            "--request-count",
            "1",
        ])
        summary = cli.build_real_llm_internet_alpha(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "real_llm_internet_alpha_v1")
        self.assertEqual(summary["cli_schema"], "real_llm_internet_alpha_cli_v1")
        self.assertIn("real_llm_internet_alpha_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_real_llm_internet_alpha_external_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("real_llm_internet_alpha_pack.py", command[1])
            self.assertEqual(command[command.index("--mode") + 1], "external-existing")
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            self.assertIn("--coordinator-url", command)
            return completed({
                "schema": "real_llm_internet_alpha_v1",
                "ok": True,
                "mode": "external-existing",
                "diagnosis_codes": ["real_llm_internet_alpha_ready", "external_runtime_verified"],
                "step": {"stderr_tail": "observer-secret admin-secret"},
            })

        args = cli.parse_args([
            "real-llm-internet-alpha",
            "--mode",
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            "http://24.199.118.54:9186",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
        ])
        summary = cli.build_real_llm_internet_alpha(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertIn("external_runtime_verified", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_main_real_llm_internet_alpha_json_outputs_summary(self) -> None:
        summary = {"schema": "real_llm_internet_alpha_v1", "ok": True}
        with patch.object(cli, "build_real_llm_internet_alpha", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["real-llm-internet-alpha", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "real_llm_internet_alpha_v1")

    def test_real_llm_internet_beta_wraps_beta_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("real_llm_internet_beta_pack.py", command[1])
            self.assertEqual(command[command.index("--mode") + 1], "kaggle-auto")
            self.assertIn("--kaggle-owner", command)
            self.assertIn("--kaggle-push-timeout-seconds", command)
            self.assertIn("--kaggle-status-timeout-seconds", command)
            return completed({
                "schema": "real_llm_internet_beta_v1",
                "ok": True,
                "mode": "kaggle-auto",
                "diagnosis_codes": [
                    "real_llm_internet_beta_ready",
                    "real_llm_internet_alpha_ready",
                    "external_runtime_verified",
                    "kaggle_kernels_deleted",
                ],
                "artifacts": {},
            })

        args = cli.parse_args([
            "real-llm-internet-beta",
            "--output-dir",
            str(output_dir),
            "--port",
            "9190",
            "--base-port",
            "9191",
            "--request-count",
            "2",
            "--kaggle-owner",
            "xuyuhaosuyi",
        ])
        summary = cli.build_real_llm_internet_beta(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "real_llm_internet_beta_v1")
        self.assertEqual(summary["cli_schema"], "real_llm_internet_beta_cli_v1")
        self.assertIn("real_llm_internet_beta_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_main_real_llm_internet_beta_json_outputs_summary(self) -> None:
        summary = {"schema": "real_llm_internet_beta_v1", "ok": True}
        with patch.object(cli, "build_real_llm_internet_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["real-llm-internet-beta", "--kaggle-owner", "xuyuhaosuyi", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "real_llm_internet_beta_v1")

    def test_print_real_llm_internet_beta_outputs_scope_summary(self) -> None:
        report = {
            "schema": "real_llm_internet_beta_v1",
            "cli_schema": "real_llm_internet_beta_cli_v1",
            "ok": True,
            "mode": "kaggle-auto",
            "coordinator_url": "http://127.0.0.1:9190",
            "output_dir": "/tmp/real-llm-internet-beta",
            "output_request": {
                "include_output": False,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "public_artifact_safe": True,
            },
            "answer_scope": {
                "scope_state": "no-local-answer",
                "terminal_only": False,
                "visible_in_terminal": False,
                "saved_json_display": "hash-only",
                "saved_markdown_display": "hash-only",
                "public_artifact_safe": True,
            },
            "shareable_summary": {
                "saved_artifacts_public_safe": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "local_output_display_only": False,
                "answer_scope_state": "no-local-answer",
                "local_answer_terminal_only": False,
            },
            "runtime_classification": {"kaggle_auto": True, "external_runtime_verified": True},
            "kaggle_lifecycle": {"kernels_deleted": True},
            "diagnosis_codes": ["real_llm_internet_beta_ready"],
            "artifacts": {},
        }

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            cli.print_real_llm_internet_beta(report)

        output = stream.getvalue()
        self.assertIn("output_request: include_output=False raw_generated_text_public=False public_artifact_safe=True", output)
        self.assertIn("answer_scope: state=no-local-answer", output)
        self.assertIn("saved_json=hash-only", output)
        self.assertIn("shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False", output)
        self.assertIn("generated_token_ids_public=False", output)

    def test_release_ready_wraps_readiness_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("release_readiness_pack.py", command[1])
            return completed({
                "ok": True,
                "schema": "release_readiness_v1",
                "release_status": {
                    "ready": True,
                    "status": "ready",
                    "diagnosis_codes": ["release_ready"],
                },
            })

        args = cli.parse_args([
            "release-ready",
            "--output-dir",
            str(output_dir),
            "--base-port",
            "9024",
            "--request-count",
            "4",
            "--allow-dirty",
        ])

        report = cli.build_release_ready(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "release_readiness_v1")
        self.assertIn("release_ready", report["release_status"]["diagnosis_codes"])
        self.assertTrue(any("--allow-dirty" in command for command in calls))

    def test_main_release_ready_json_outputs_report(self) -> None:
        report = {"schema": "release_readiness_v1", "ok": True}
        with patch.object(cli, "build_release_ready", return_value=report), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["release-ready", "--allow-dirty", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "release_readiness_v1")

    def test_remote_runbook_wraps_pack_and_writes_safe_summary(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_demo_runbook_pack.py", command[1])
            (output_dir / "remote_demo_runbook.json").write_text("{}", encoding="utf-8")
            (output_dir / "remote_demo_runbook.md").write_text("# Runbook\n", encoding="utf-8")
            (output_dir / "operator.private.env").write_text("CROWDTENSOR_ADMIN_TOKEN=secret\n", encoding="utf-8")
            (output_dir / "miner.private.env").write_text("CROWDTENSOR_MINER_TOKEN=secret\n", encoding="utf-8")
            return completed({
                "ok": True,
                "schema": "remote_demo_runbook_v1",
                "demo": {
                    "workload_type": "model_bundle_infer",
                    "scenario_schema": "model_bundle_inference_scenario_v1",
                    "scenario_id": "route-baseline",
                    "scenario_description": "Fixed CPU read-only route prompts from the built-in bundle corpus.",
                    "scenario_request_count": 8,
                },
            })

        args = cli.parse_args([
            "remote-runbook",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--output-dir",
            str(output_dir),
        ])

        summary = cli.build_remote_runbook(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_runbook_cli_v1")
        self.assertEqual(summary["runbook_schema"], "remote_demo_runbook_v1")
        self.assertEqual(summary["workload_type"], "model_bundle_infer")
        self.assertEqual(summary["scenario"]["scenario_id"], "route-baseline")
        self.assertTrue(summary["artifacts"]["operator_private_env"]["present"])
        self.assertTrue((output_dir / "remote_runbook_cli_summary.json").is_file())
        self.assertTrue(any("--coordinator-url" in command for command in calls))
        self.assertTrue(any("--scenario-id" in command and "route-baseline" in command for command in calls))

    def test_remote_runbook_replace_forwards_to_pack(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("--replace", command)
            return completed({"ok": True, "schema": "remote_demo_runbook_v1"})

        args = cli.parse_args([
            "remote-runbook",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--output-dir",
            str(output_dir),
            "--replace",
        ])

        summary = cli.build_remote_runbook(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)

    def test_remote_acceptance_defaults_to_create_session_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("--create-session", command)
            (output_dir / "remote_demo_acceptance.json").write_text("{}", encoding="utf-8")
            (output_dir / "remote_demo_acceptance.md").write_text("# Acceptance\n", encoding="utf-8")
            return completed({
                "ok": True,
                "schema": "remote_demo_acceptance_v1",
                "scenario": {"scenario_id": "route-baseline"},
                "diagnosis_codes": ["acceptance_ready"],
            })

        args = cli.parse_args([
            "remote-acceptance",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
        ])

        summary = cli.build_remote_acceptance(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_acceptance_cli_v1")
        self.assertTrue(summary["create_session"])
        self.assertEqual(summary["scenario"]["scenario_id"], "route-baseline")
        self.assertEqual(summary["diagnosis_codes"], ["acceptance_ready"])
        self.assertTrue(any("--scenario-id" in command and "route-baseline" in command for command in calls))
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue((output_dir / "remote_acceptance_cli_summary.json").is_file())

    def test_remote_acceptance_no_create_session_uses_wait_only_mode(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertNotIn("--create-session", command)
            return completed({"ok": True, "schema": "remote_demo_acceptance_v1"})

        args = cli.parse_args([
            "remote-acceptance",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--no-create-session",
            "--output-dir",
            str(output_dir),
        ])

        summary = cli.build_remote_acceptance(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertFalse(summary["create_session"])

    def test_remote_acceptance_failure_tail_redacts_token_values(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout=json.dumps({"ok": False, "diagnosis_codes": ["observer_auth_failed"]}) + "\n",
                stderr="token observer-secret rejected; admin-secret was not accepted",
            )

        args = cli.parse_args([
            "remote-acceptance",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
        ])

        summary = cli.build_remote_acceptance(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertFalse(summary["ok"])
        self.assertIn("observer_auth_failed", summary["diagnosis_codes"])
        self.assertIn("<redacted>", serialized)
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_remote_demo_prepare_wraps_home_compute_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "prepare")
            return completed({
                "schema": "remote_home_compute_demo_v1",
                "ok": True,
                "mode": "prepare",
                "diagnosis_codes": ["remote_home_compute_prepare_ready"],
            })

        args = cli.parse_args([
            "remote-demo",
            "prepare",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--output-dir",
            str(output_dir),
            "--replace",
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_home_compute_demo_v1")
        self.assertEqual(summary["mode"], "prepare")
        self.assertTrue(any("--replace" in command for command in calls))

    def test_remote_demo_prepare_forwards_kaggle_target(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("--target", command)
            self.assertIn("kaggle", command)
            return completed({
                "schema": "remote_home_compute_demo_v1",
                "ok": True,
                "mode": "prepare",
                "target_environment": {"name": "kaggle", "kaggle_remote_miner_beta": True},
                "diagnosis_codes": ["kaggle_remote_miner_prepare_ready"],
            })

        args = cli.parse_args([
            "remote-demo",
            "prepare",
            "--target",
            "kaggle",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "kaggle-cpu-1",
            "--output-dir",
            str(output_dir),
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["target_environment"]["name"], "kaggle")
        self.assertIn("kaggle_remote_miner_prepare_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_remote_demo_verify_defaults_create_session_and_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "verify")
            self.assertIn("--create-session", command)
            return completed({
                "schema": "remote_home_compute_demo_v1",
                "ok": True,
                "mode": "verify",
                "diagnosis_codes": ["remote_home_compute_ready"],
                "step": {
                    "stderr_tail": "observer-secret should be redacted",
                },
            })

        args = cli.parse_args([
            "remote-demo",
            "verify",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_home_compute_demo_v1")
        self.assertIn("remote_home_compute_ready", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(any("--remote-timeout-seconds" in command for command in calls))

    def test_remote_demo_external_llm_forwards_workload_and_runtime_flags(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "verify")
            self.assertIn("--workload", command)
            self.assertIn("external-llm", command)
            self.assertIn("--mock", command)
            self.assertIn("--llm-runtime-url", command)
            self.assertIn("--llm-runtime-api-key", command)
            return completed({
                "schema": "remote_home_compute_demo_v1",
                "ok": True,
                "mode": "verify",
                "diagnosis_codes": ["remote_external_llm_ready", "remote_home_compute_ready"],
                "demo": {"workload_kind": "external-llm", "workload_type": "external_llm_infer"},
            })

        args = cli.parse_args([
            "remote-demo",
            "verify",
            "--workload",
            "external-llm",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
            "--mock",
            "--llm-runtime-url",
            "http://127.0.0.1:11434/v1/chat/completions",
            "--llm-runtime-api-key",
            "runtime-secret",
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["demo"]["workload_type"], "external_llm_infer")
        self.assertIn("remote_external_llm_ready", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertNotIn("runtime-secret", serialized)
        self.assertNotIn("http://127.0.0.1:11434", serialized)

    def test_remote_demo_micro_llm_forwards_workload_and_decode_steps(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "verify")
            self.assertIn("--workload", command)
            self.assertIn("micro-llm-sharded", command)
            self.assertIn("--decode-steps", command)
            self.assertEqual(command[command.index("--decode-steps") + 1], "3")
            self.assertIn("--micro-llm-artifact", command)
            self.assertEqual(command[command.index("--micro-llm-artifact") + 1], "dist/micro-llm-artifact")
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "arn,ten")
            return completed({
                "schema": "remote_home_compute_demo_v1",
                "ok": True,
                "mode": "verify",
                "diagnosis_codes": ["remote_micro_llm_sharded_ready", "remote_home_compute_ready"],
                "demo": {"workload_kind": "micro-llm-sharded", "workload_type": "micro_llm_sharded_infer"},
            })

        args = cli.parse_args([
            "remote-demo",
            "verify",
            "--workload",
            "micro-llm-sharded",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
            "--decode-steps",
            "3",
            "--stage-mode",
            "split",
            "--require-distinct-stage-miners",
            "--micro-llm-artifact",
            "dist/micro-llm-artifact",
            "--prompt-texts",
            "arn,ten",
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["demo"]["workload_type"], "micro_llm_sharded_infer")
        self.assertIn("remote_micro_llm_sharded_ready", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(calls)

    def test_remote_demo_real_llm_forwards_hf_and_split_flags(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "verify")
            self.assertIn("--workload", command)
            self.assertIn("real-llm-sharded", command)
            self.assertIn("--stage-mode", command)
            self.assertEqual(command[command.index("--stage-mode") + 1], "split")
            self.assertIn("--require-distinct-stage-miners", command)
            self.assertIn("--hf-model-id", command)
            self.assertEqual(command[command.index("--hf-model-id") + 1], "sshleifer/tiny-gpt2")
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "real prompt")
            return completed({
                "schema": "remote_home_compute_demo_v1",
                "ok": True,
                "mode": "verify",
                "diagnosis_codes": ["remote_real_llm_sharded_ready", "remote_home_compute_ready"],
                "demo": {"workload_kind": "real-llm-sharded", "workload_type": "real_llm_sharded_infer"},
            })

        args = cli.parse_args([
            "remote-demo",
            "verify",
            "--workload",
            "real-llm-sharded",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
            "--hf-model-id",
            "sshleifer/tiny-gpt2",
            "--prompt-texts",
            "real prompt",
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["demo"]["workload_type"], "real_llm_sharded_infer")
        self.assertIn("remote_real_llm_sharded_ready", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(calls)

    def test_remote_demo_doctor_forwards_tokens_and_require_result(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "doctor")
            self.assertIn("--require-result", command)
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            return completed({
                "schema": "remote_home_compute_doctor_v1",
                "ok": True,
                "diagnosis_codes": ["remote_home_compute_doctor_ready"],
            })

        args = cli.parse_args([
            "remote-demo",
            "doctor",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
            "--require-result",
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_home_compute_doctor_v1")
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertTrue(calls)

    def test_remote_demo_collect_forwards_task_and_external_runtime_flags(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "collect")
            self.assertIn("--task-id", command)
            self.assertIn("task-1", command)
            self.assertIn("--mock", command)
            self.assertIn("--llm-runtime-url", command)
            return completed({
                "schema": "remote_home_compute_collect_v1",
                "ok": True,
                "diagnosis_codes": ["remote_home_compute_collect_ready"],
            })

        args = cli.parse_args([
            "remote-demo",
            "collect",
            "--workload",
            "external-llm",
            "--coordinator-url",
            "https://coord.example",
            "--miner-id",
            "remote-linux-1",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
            "--task-id",
            "task-1",
            "--mock",
            "--llm-runtime-url",
            "http://127.0.0.1:11434/v1/chat/completions",
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_home_compute_collect_v1")
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)
        self.assertNotIn("http://127.0.0.1:11434", serialized)

    def test_remote_demo_clean_uses_cleanup_mode_without_workload_args(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("remote_home_compute_demo_pack.py", command[1])
            self.assertEqual(command[2], "clean")
            self.assertIn("--apply", command)
            self.assertIn("--include-private", command)
            self.assertNotIn("--workload", command)
            return completed({
                "schema": "remote_home_compute_cleanup_v1",
                "ok": True,
                "diagnosis_codes": ["remote_home_compute_cleanup_ready"],
            })

        args = cli.parse_args([
            "remote-demo",
            "clean",
            "--output-dir",
            str(output_dir),
            "--apply",
            "--include-private",
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "remote_home_compute_cleanup_v1")
        self.assertTrue(calls)

    def test_remote_demo_kaggle_real_prepare_wraps_acceptance_pack(self) -> None:
        output_dir = Path(self._tmp_dir())
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            self.assertIn("kaggle_real_runtime_acceptance_pack.py", command[1])
            self.assertEqual(command[2], "prepare")
            self.assertIn("--public-host", command)
            self.assertIn("24.199.118.54", command)
            self.assertIn("--port", command)
            self.assertIn("9180", command)
            self.assertIn("--workload", command)
            self.assertEqual(command[command.index("--workload") + 1], "micro-llm-sharded")
            self.assertIn("--decode-steps", command)
            self.assertEqual(command[command.index("--decode-steps") + 1], "3")
            self.assertIn("--stage-mode", command)
            self.assertEqual(command[command.index("--stage-mode") + 1], "split")
            self.assertIn("--micro-llm-artifact", command)
            self.assertEqual(command[command.index("--micro-llm-artifact") + 1], "dist/micro-llm-artifact")
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "arn,ten")
            self.assertIn("--require-distinct-stage-miners", command)
            self.assertIn("--replace", command)
            return completed({
                "schema": "kaggle_real_runtime_acceptance_v1",
                "ok": True,
                "mode": "prepare",
                "diagnosis_codes": ["kaggle_artifacts_ready"],
            })

        args = cli.parse_args([
            "remote-demo",
            "kaggle-real",
            "--action",
            "prepare",
            "--public-host",
            "24.199.118.54",
            "--port",
            "9180",
            "--miner-id",
            "kaggle-cpu-1",
            "--workload",
            "micro-llm-sharded",
            "--decode-steps",
            "3",
            "--stage-mode",
            "split",
            "--micro-llm-artifact",
            "dist/micro-llm-artifact",
            "--prompt-texts",
            "arn,ten",
            "--output-dir",
            str(output_dir),
            "--replace",
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)

        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["schema"], "kaggle_real_runtime_acceptance_v1")
        self.assertIn("kaggle_artifacts_ready", summary["diagnosis_codes"])
        self.assertTrue(calls)

    def test_remote_demo_kaggle_real_verify_redacts_tokens(self) -> None:
        output_dir = Path(self._tmp_dir())

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("kaggle_real_runtime_acceptance_pack.py", command[1])
            self.assertEqual(command[2], "verify")
            self.assertIn("--observer-token", command)
            self.assertIn("--admin-token", command)
            self.assertIn("--remote-timeout-seconds", command)
            self.assertIn("--workload", command)
            self.assertEqual(command[command.index("--workload") + 1], "micro-llm-sharded")
            self.assertIn("--stage-mode", command)
            self.assertIn("--require-distinct-stage-miners", command)
            self.assertIn("--micro-llm-artifact", command)
            self.assertEqual(command[command.index("--micro-llm-artifact") + 1], "dist/micro-llm-artifact")
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "arn,ten")
            return completed({
                "schema": "kaggle_real_runtime_acceptance_v1",
                "ok": True,
                "mode": "verify",
                "diagnosis_codes": ["kaggle_real_runtime_ready"],
                "step": {"stderr_tail": "observer-secret admin-secret"},
            })

        args = cli.parse_args([
            "remote-demo",
            "kaggle-real",
            "--action",
            "verify",
            "--public-host",
            "24.199.118.54",
            "--port",
            "9180",
            "--miner-id",
            "kaggle-cpu-1",
            "--workload",
            "micro-llm-sharded",
            "--stage-mode",
            "split",
            "--micro-llm-artifact",
            "dist/micro-llm-artifact",
            "--prompt-texts",
            "arn,ten",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
        ])

        summary = cli.build_remote_demo(args, runner=fake_runner)
        serialized = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ok"], summary)
        self.assertIn("kaggle_real_runtime_ready", summary["diagnosis_codes"])
        self.assertNotIn("observer-secret", serialized)
        self.assertNotIn("admin-secret", serialized)

    def test_main_remote_runbook_json_outputs_summary(self) -> None:
        summary = {"schema": "remote_runbook_cli_v1", "ok": True}
        with patch.object(cli, "build_remote_runbook", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["remote-runbook", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "remote_runbook_cli_v1")

    def test_main_remote_acceptance_json_outputs_summary(self) -> None:
        summary = {"schema": "remote_acceptance_cli_v1", "ok": True}
        with patch.object(cli, "build_remote_acceptance", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main([
                    "remote-acceptance",
                    "--coordinator-url",
                    "https://coord.example",
                    "--miner-id",
                    "remote-linux-1",
                    "--observer-token",
                    "observer-secret",
                    "--admin-token",
                    "admin-secret",
                    "--json",
                ])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "remote_acceptance_cli_v1")

    def test_main_remote_demo_json_outputs_summary(self) -> None:
        summary = {"schema": "remote_home_compute_demo_v1", "ok": True}
        with patch.object(cli, "build_remote_demo", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main([
                    "remote-demo",
                    "verify",
                    "--coordinator-url",
                    "https://coord.example",
                    "--miner-id",
                    "remote-linux-1",
                    "--observer-token",
                    "observer-secret",
                    "--admin-token",
                    "admin-secret",
                    "--json",
                ])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "remote_home_compute_demo_v1")


if __name__ == "__main__":
    unittest.main()
