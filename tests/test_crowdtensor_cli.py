from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
from pathlib import Path
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
            cli.stream_progress_lines(batch),
            [
                "  stream[1]: request=req-1 tokens=2/2 counts=[1, 2] complete=True missing=False",
                "  stream[2]: request=req-2 tokens=1/2 counts=[1] complete=False missing=False",
                "  stream[3]: request=missing tokens=0/2 counts=[] complete=False missing=True",
            ],
        )
        self.assertEqual(
            cli.stream_progress_issue_summary(batch),
            "missing_requests=1/3 request[2]=req-2:1/2 request[3]=missing",
        )
        self.assertEqual(
            cli.stream_request_label({"prompt_hash": "sha256:abcdef1234567890zz"}),
            "sha256:abcdef1234567890z",
        )
        self.assertEqual(cli.stream_request_label({"request_key": "<redacted>"}), "unknown")

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
        self.assertIn("Start with the status/user_status line", rendered)
        self.assertIn("preflight-ready", rendered)
        self.assertIn("preflight-partial", rendered)
        self.assertIn("Reports include action, recommended_next, and next[...] lines", rendered)
        self.assertIn("ready_to_submit labels mean", rendered)
        self.assertIn("ready_to_submit.next_step is the script-friendly", rendered)
        self.assertIn("stage_preflight_unknown means rerun the stage preflight", rendered)
        self.assertIn("stage_preflight_not_checked means fix route/Coordinator, then rerun with observer token", rendered)
        self.assertIn("partial can submit but still needs", rendered)
        self.assertIn("printed", rendered)
        self.assertIn("follow-up preflight", rendered)
        self.assertIn("Use one prompt source at a time: positional prompt, --prompt-text/--prompt", rendered)
        self.assertIn("ambiguous mixes are rejected", rendered)
        self.assertIn("output_request.include_output", rendered)
        self.assertIn("output_request.raw_generated_text_public false", rendered)
        self.assertIn("The trace line summarizes session, request count, ledger rows, stream events", rendered)
        self.assertIn("The result line summarizes completion state", rendered)
        self.assertIn("display safety: local-private for terminal-only", rendered)
        self.assertIn("hash-only for redacted summaries", rendered)
        self.assertIn("hash-only-json for JSON stdout", rendered)
        self.assertIn("the terminal prints it as answer: or answer[n]:", rendered)
        self.assertIn("local_output safety metadata with safe count/source fields", rendered)
        self.assertIn("local-private-task-state or coordinator-validation", rendered)
        self.assertIn("The output_display line makes the display policy explicit", rendered)
        self.assertIn("--include-output records that local display was requested", rendered)
        self.assertIn("it does not make raw generated text public", rendered)
        self.assertIn("safe per-request ids or prompt hashes", rendered)
        self.assertIn("never exposes raw prompt text", rendered)
        self.assertIn("The issue line summarizes state, primary diagnosis, next step, safe progress", rendered)
        self.assertIn("redacted detail is available", rendered)
        self.assertIn("The artifacts line points to the first Markdown summary to inspect", rendered)
        self.assertIn("redacted JSON/Markdown artifact paths", rendered)
        self.assertIn("Start with the review/review_summary line", rendered)
        self.assertIn("recommended command label", rendered)
        self.assertIn("attention warnings such as incomplete stream evidence", rendered)
        self.assertIn("review_next line", rendered)
        self.assertIn("safe recommended command", rendered)
        self.assertIn("terminal output renders your local prompt for copying", rendered)
        self.assertIn("saved artifacts", rendered)
        self.assertIn("keep prompt placeholders", rendered)
        self.assertIn("existing mode only: check route/session readiness", rendered)
        self.assertIn("mutually exclusive with positional", rendered)
        self.assertIn("prompt and --prompt-texts", rendered)
        self.assertIn("mutually exclusive with single-prompt sources", rendered)
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
        self.assertIn("Start with the status/user_status line", rendered)
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
        self.assertIn("ambiguous mixes are rejected", rendered)
        self.assertIn("output_request.include_output", rendered)
        self.assertIn("output_request.raw_generated_text_public false", rendered)
        self.assertIn("The trace line summarizes session, request count, ledger rows, stream events", rendered)
        self.assertIn("The result line summarizes completion state", rendered)
        self.assertIn("display safety: local-private for terminal-only", rendered)
        self.assertIn("hash-only for redacted summaries", rendered)
        self.assertIn("hash-only-json for JSON stdout", rendered)
        self.assertIn("the terminal prints it as answer: or answer[n]:", rendered)
        self.assertIn("local_output safety metadata with safe count/source fields", rendered)
        self.assertIn("local-private-task-state or coordinator-validation", rendered)
        self.assertIn("The output_display line makes the display policy explicit", rendered)
        self.assertIn("--include-output records that local display was requested", rendered)
        self.assertIn("it does not make raw generated text public", rendered)
        self.assertIn("safe per-request ids or prompt hashes", rendered)
        self.assertIn("never exposes raw prompt text", rendered)
        self.assertIn("The issue line summarizes state, primary diagnosis, next step, safe progress", rendered)
        self.assertIn("The artifacts line points to the first Markdown summary to inspect", rendered)
        self.assertIn("redacted JSON/Markdown artifact paths", rendered)
        self.assertIn("Start with the review/review_summary line", rendered)
        self.assertIn("recommended command label", rendered)
        self.assertIn("attention warnings such as incomplete stream evidence", rendered)
        self.assertIn("review_next line", rendered)
        self.assertIn("safe recommended command", rendered)
        self.assertIn("terminal output renders your local prompt for copying", rendered)
        self.assertIn("saved artifacts", rendered)
        self.assertIn("keep prompt placeholders", rendered)
        self.assertIn("redacted detail is available", rendered)
        self.assertIn("check route/session readiness without submitting a", rendered)
        self.assertIn("generation task", rendered)
        self.assertIn("mutually exclusive with positional", rendered)
        self.assertIn("prompt and --prompt-texts", rendered)
        self.assertIn("mutually exclusive with single-prompt sources", rendered)
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
            "False service=None protocol=None error=OSError",
        )
        self.assertEqual(
            cli.coordinator_ready_text({"ok": None, "reason": "live_preflight_skipped"}),
            "None service=None protocol=None reason=live_preflight_skipped",
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
            self.assertIn("mixed prompt sources", rendered)
            self.assertIn("output_request.include_output", rendered)
            self.assertIn("output_request.raw_generated_text_public", rendered)
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
            self.assertIn("`recommended_next` plus `next[...]`", rendered)
            self.assertIn("`trace`", rendered)
            self.assertIn("`result`", rendered)
            self.assertIn("completion state", rendered)
            self.assertIn("token count, output count", rendered)
            self.assertIn("display safety", rendered)
            self.assertIn("`local-private` for terminal-only generated text", rendered)
            self.assertIn("`hash-only` for redacted", rendered)
            self.assertIn("`hash-only-json` for JSON stdout", rendered)
            self.assertIn("`answer:`", rendered)
            self.assertIn("`answer[n]:`", rendered)
            self.assertIn("`local_output` safety metadata", rendered)
            self.assertIn("safe output `count` and `source` fields", rendered)
            self.assertIn("`local-private-task-state`", rendered)
            self.assertIn("`coordinator-validation`", rendered)
            self.assertIn("`issue`", rendered)
            self.assertIn("`issue_summary`", rendered)
            self.assertIn("`artifact_summary`", rendered)
            self.assertIn("`review_summary`", rendered)
            self.assertIn("current state, next step, first artifact", rendered)
            self.assertIn("`attention` value for warnings", rendered)
            self.assertIn("`review_next` line", rendered)
            self.assertIn("safe recommended command", rendered)
            self.assertIn("human terminal output renders it", rendered)
            self.assertIn("with your current local prompt for copying", rendered)
            self.assertIn("JSON/Markdown artifacts keep", rendered)
            self.assertIn("prompt placeholders", rendered)
            self.assertIn("first Markdown summary to inspect", rendered)
            self.assertIn("accepted ledger rows", rendered)
            self.assertIn("primary diagnosis code", rendered)
            self.assertIn("safe progress text", rendered)
            self.assertIn("per-request ids or prompt hashes", rendered)
            self.assertIn("failure", rendered)
            self.assertIn("redacted", rendered)
            self.assertIn("prompt text", rendered)

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
        self.assertEqual(len(report["trace"]["request_trace"]), 1)
        self.assertEqual(report["trace"]["request_trace"][0]["source"], "session-request")
        self.assertTrue(report["trace"]["request_trace"][0]["prompt_hash"])
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(report["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(report["shareable_summary"]["raw_generated_text_public"])
        self.assertFalse(report["shareable_summary"]["generated_token_ids_public"])
        self.assertFalse(report["shareable_summary"]["local_output_display_only"])
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
        self.assertIn("<prompt>", persisted["review_summary"]["next_command"])
        self.assertNotIn("CrowdTensor prompt", persisted["review_summary"]["next_command"])
        self.assertTrue(persisted["review_summary"]["public_artifact_safe"])
        self.assertFalse(persisted["output_request"]["raw_generated_text_public"])
        self.assertEqual(persisted["user_status"]["state"], "preflight-partial")
        self.assertEqual(persisted["user_status"]["next_step"], "run_live_preflight")
        self.assertEqual(persisted["trace"]["request_count"], 1)
        self.assertTrue(persisted["trace"]["request_trace"][0]["prompt_hash"])
        self.assertTrue(persisted["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(persisted["shareable_summary"]["raw_prompt_public"])
        self.assertNotIn("CrowdTensor prompt", json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "generate_summary.md").read_text(encoding="utf-8")
        self.assertIn("# CrowdTensor Generate Summary", markdown)
        self.assertIn("- OK: `True`", markdown)
        self.assertIn("- Dry run: `True`", markdown)
        self.assertIn("## What To Do Next", markdown)
        self.assertIn("- State: `preflight-partial`", markdown)
        self.assertIn("- Next step: `run_live_preflight`", markdown)
        self.assertIn("- Recommended: `check generation route` reason=`confirm_live_preflight`", markdown)
        self.assertIn("- Copy command: `crowdtensor generate --max-new-tokens 16", markdown)
        self.assertIn("- Requires env: `CROWDTENSOR_OBSERVER_TOKEN`", markdown)
        self.assertIn("- Safety: saved Markdown keeps prompt placeholders and redacted generated output.", markdown)
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
            "- Shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False`",
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
        self.assertIn("Replace `<prompt>` with your local prompt before running saved commands.", markdown)
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
            "  trace: session=none requests=1 ledger_rows=0 stream_events=0 source=public_swarm_product_cli_v1 public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False",
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
        self.assertNotIn("stream_events: None", rendered)

    def test_generate_main_prints_copyable_local_prompt_without_persisting_it(self) -> None:
        prompt = "CrowdTensor prompt"

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
        self.assertIn("status, action, recommended_next", progress)
        self.assertNotIn(prompt, progress)
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
        self.assertIn(
            f"review_next: label=check generation route reason=verify_stage_miners command=crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-texts '{prompts}' --dry-run",
            rendered,
        )
        self.assertIn(f"recommended_next: check generation route reason=verify_stage_miners crowdtensor generate --max-new-tokens 16 --coordinator-url http://127.0.0.1:8787 --prompt-texts '{prompts}' --dry-run", rendered)
        self.assertIn(f"--prompt-texts '{prompts}' --dry-run", rendered)
        self.assertNotIn("--prompt-text '<prompt>'", rendered)
        self.assertNotIn(cli.INFER_PROMPT_PLACEHOLDER, rendered)

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

    def test_generate_rejects_ambiguous_prompt_sources(self) -> None:
        cases = [
            ["generate", "positional prompt", "--prompt-text", "flag prompt"],
            ["generate", "positional prompt", "--prompt-texts", "first prompt,second prompt"],
            ["generate", "--prompt-text", "flag prompt", "--prompt-texts", "first prompt,second prompt"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as raised:
                cli.parse_args(argv)
            self.assertEqual(
                str(raised.exception),
                "generate accepts one prompt source: positional prompt, --prompt-text, or --prompt-texts",
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
        self.assertIn("  coordinator_ready: True service=crowdtensord protocol=runtime_contract_v1", rendered)
        self.assertIn("  stage_preflight: checked=True ok=True matched_miners=2 missing=none", rendered)
        self.assertIn("  ready_to_submit: True label=verified fully_verified=True route=True coordinator=True stage=ready stage_verification=ready next_step=submit warnings=none", rendered)
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
            "  ready_to_submit: True label=partial fully_verified=False route=True coordinator=True stage=skipped stage_verification=skipped next_step=run_stage_preflight warnings=stage_preflight_skipped",
            rendered,
        )
        self.assertIn("next[2] submit generation after stage preflight:", rendered)

    def test_product_generate_dry_run_can_skip_live_preflight_for_ci(self) -> None:
        args = cli.parse_args([
            "generate",
            "--coordinator-url",
            "http://127.0.0.1:8787",
            "--prompt-text",
            "CrowdTensor prompt",
            "--dry-run",
            "--skip-live-preflight",
            "--json",
        ])

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
        self.assertIn(
            "warnings=coordinator_preflight_skipped,stage_preflight_skipped",
            stdout.getvalue(),
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
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        self.assertIn(
            "  status: blocked: Coordinator route exists but /ready failed; start or restart the Coordinator and retry generate --dry-run. next=fix_blockers recommendation=start Coordinator public_artifact_safe=True",
            stdout.getvalue(),
        )
        self.assertIn(
            "  coordinator_ready: False service=None protocol=None error=OSError",
            stdout.getvalue(),
        )
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
        self.assertEqual(report["user_status"]["recommended_label"], "submit generation")
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
        self.assertFalse(report["trace"]["raw_prompt_public"])
        self.assertFalse(report["trace"]["raw_generated_text_public"])
        self.assertFalse(report["trace"]["generated_token_ids_public"])
        self.assertTrue(report["trace"]["public_artifact_safe"])
        self.assertEqual(len(report["trace"]["request_trace"]), 1)
        self.assertEqual(report["trace"]["request_trace"][0]["source"], "session-request")
        self.assertTrue(report["trace"]["request_trace"][0]["prompt_hash"])
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(report["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(report["shareable_summary"]["raw_generated_text_public"])
        self.assertFalse(report["shareable_summary"]["generated_token_ids_public"])
        self.assertFalse(report["shareable_summary"]["local_output_display_only"])
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
        self.assertEqual(report["review_summary"]["recommended_label"], "submit generation")
        self.assertEqual(report["review_summary"]["primary_code"], "public_swarm_generate_ready")
        self.assertEqual(report["review_summary"]["attention"], "")
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
        self.assertTrue(persisted["trace"]["request_trace"][0]["prompt_hash"])
        self.assertTrue(persisted["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertEqual(persisted["issue_summary"]["state"], "completed")
        self.assertEqual(persisted["artifact_summary"]["inspect_first"], str(output_dir / "generate_summary.md"))
        self.assertTrue(persisted["artifact_summary"]["public_artifact_safe"])
        self.assertEqual(persisted["review_summary"]["state"], "completed")
        self.assertEqual(persisted["review_summary"]["inspect_first"], str(output_dir / "generate_summary.md"))
        self.assertIn("<prompt>", persisted["review_summary"]["next_command"])
        self.assertNotIn("CrowdTensor prompt", persisted["review_summary"]["next_command"])
        self.assertTrue(persisted["review_summary"]["public_artifact_safe"])
        self.assertNotIn("local generated text must stay local", json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "generate_summary.md").read_text(encoding="utf-8")
        self.assertIn(
            "- Status: `completed: Generation completed. next=rerun_or_review_artifacts recommendation=submit generation public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            f"- Review: `state=completed next=rerun_or_review_artifacts inspect={output_dir / 'generate_summary.md'} recommended=submit generation primary=public_swarm_generate_ready attention=none public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Review next: `label=submit generation reason=rerun_generation command=CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor generate --max-new-tokens 2",
            markdown,
        )
        self.assertIn(
            "- Issue: `state=completed primary=public_swarm_generate_ready next=rerun_or_review_artifacts progress=`",
            markdown,
        )
        self.assertIn(
            "- Recommended next: `submit generation` reason=`rerun_generation` command=`CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor generate --max-new-tokens 2",
            markdown,
        )
        self.assertIn("requires=`CROWDTENSOR_ADMIN_TOKEN`", markdown)
        self.assertIn("- Generation: `2/2` hash=`sha256:generated`", markdown)
        self.assertIn("- Result: `status=complete tokens=2/2 outputs=1 display=hash-only-json hash=sha256:generated public_artifact_safe=True`", markdown)
        self.assertIn(
            "- Trace: `session=real-llm-session-test requests=1 ledger_rows=1 stream_events=0 source=public_swarm_product_cli_v1 public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False`",
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
            "  status: completed: Generation completed. next=rerun_or_review_artifacts recommendation=submit generation public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            f"  review: state=completed next=rerun_or_review_artifacts inspect={output_dir / 'generate_summary.md'} recommended=submit generation primary=public_swarm_generate_ready attention=none public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  review_next: label=submit generation reason=rerun_generation command=CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor generate --max-new-tokens 2",
            rendered,
        )
        self.assertIn(
            "  issue: state=completed primary=public_swarm_generate_ready next=rerun_or_review_artifacts progress=`polls=",
            rendered,
        )
        self.assertIn(
            "  result: status=complete tokens=2/2 outputs=1 display=hash-only-json hash=sha256:generated public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  trace: session=real-llm-session-test requests=1 ledger_rows=1 stream_events=0 source=public_swarm_product_cli_v1 public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=False",
            rendered,
        )
        self.assertIn(
            f"  artifacts: inspect={output_dir / 'generate_summary.md'} json={output_dir / 'generate_summary.json'} markdown={output_dir / 'generate_summary.md'} present=2/2 public_artifact_safe=True",
            rendered,
        )
        self.assertIn(f"  output_dir: {output_dir}", rendered)
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
        persisted = json.loads((output_dir / "generate_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["local_output"]["generated_text"], "")
        self.assertEqual(persisted["local_output"]["outputs"][0]["generated_text"], "")
        self.assertTrue(persisted["local_output"]["public_artifact_safe"])
        self.assertEqual(persisted["result"]["status"], "complete")
        self.assertEqual(persisted["result"]["output_count"], 1)
        self.assertEqual(persisted["result"]["display"], "hash-only")
        self.assertTrue(persisted["result"]["public_artifact_safe"])
        self.assertEqual(persisted["output_display"]["terminal_display"], "local-private")
        self.assertFalse(persisted["output_display"]["terminal_text_available"])
        self.assertEqual(persisted["output_display"]["saved_artifact_display"], "hash-only")
        self.assertEqual(persisted["output_display"]["json_stdout_display"], "hash-only-json")
        self.assertFalse(persisted["output_display"]["include_output_requested"])
        self.assertNotIn("local generated text must stay local", json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "generate_summary.md").read_text(encoding="utf-8")
        self.assertIn("- Result: `status=complete tokens=2/2 outputs=1 display=hash-only", markdown)
        self.assertIn(
            "- Output display: `terminal=local-private terminal_text=False saved=hash-only json_stdout=hash-only-json include_output=False raw_public=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Local output: `available=False display_only=False public_artifact_safe=True` count=`1` source=`coordinator-validation`",
            markdown,
        )
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
        self.assertIn("  result: status=complete tokens=2/2 outputs=1 display=local-private", rendered)
        self.assertIn(
            "  output_display: terminal=local-private terminal_text=True saved=hash-only json_stdout=hash-only-json include_output=False raw_public=False public_artifact_safe=True",
            rendered,
        )
        self.assertIn(
            "  local_output: available=True display_only=True public_artifact_safe=False count=1 source=coordinator-validation",
            rendered,
        )
        self.assertLess(rendered.index("  answer: local generated text must stay local"), rendered.index("  local_output: "))
        self.assertLess(rendered.index("  answer: local generated text must stay local"), rendered.index("  trace: "))
        self.assertIn(f"  output_dir: {output_dir}", rendered)
        self.assertIn(f"markdown={output_dir / 'generate_summary.md'}", rendered)

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
            "crowdtensor generate --max-new-tokens 4 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --stream --timeout-seconds 120",
            next_lines,
        )
        self.assertIn(
            "crowdtensor generate --max-new-tokens 4 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --dry-run --observer-token ${CROWDTENSOR_OBSERVER_TOKEN:?set CROWDTENSOR_OBSERVER_TOKEN} --stream",
            next_lines,
        )
        self.assertIn(
            "crowdtensor generate --max-new-tokens 4 --coordinator-url http://127.0.0.1:8787 --prompt-text '<prompt>' --stream",
            next_lines,
        )
        retry = next(item for item in report["next_commands"] if item["label"] == "retry generation with longer timeout")
        self.assertEqual(retry["requires_env"], ["CROWDTENSOR_ADMIN_TOKEN"])
        self.assertEqual(retry["command"].count("--timeout-seconds"), 1)
        self.assertEqual(retry["command"][retry["command"].index("--timeout-seconds") + 1], "120")
        self.assertNotIn("--timeout-seconds 1 --timeout-seconds 120", retry["command_line"])
        self.assertNotIn("must not leak", encoded)
        self.assertNotIn("CrowdTensor prompt", encoded)
        self.assertNotIn("admin-secret", encoded)
        self.assertNotIn('"generated_token_ids": [1]', encoded)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.print_product_generate(report)
        rendered = stdout.getvalue()
        self.assertIn(
            "  issue: state=blocked primary=generation_timeout next=fix_blockers progress=`polls=1 accepted_rows=1 tokens=1/4 ledger=True stream=False last_error=HTTPError` safe_detail=True",
            rendered,
        )
        self.assertIn("  stream_events: 1 source=admin-results-ledger-fallback complete=False", rendered)
        self.assertIn("  wait: polls=1 accepted_rows=1 tokens=1/4 ledger=True stream=False last_error=HTTPError", rendered)
        self.assertNotIn("stream echoed CrowdTensor prompt", rendered)
        self.assertNotIn("admin-secret", rendered)
        self.assertIn("  action: Generation reached 1/4 tokens before timeout", rendered)
        self.assertIn("  next[", rendered)
        self.assertIn("retry generation with longer timeout", rendered)

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
        self.assertLess(rendered.index("  answer:  readable beta text"), rendered.index("  local_output: "))
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
        self.assertEqual(
            json_report["local_output_note"],
            "Raw generated text is suppressed in JSON/public output; rerun without --json for local display.",
        )
        self.assertNotIn("local_output", json_report)
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
        self.assertNotIn("local_output", json_report)
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
        self.assertNotIn("local_output", json_report)
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

    def test_public_real_llm_swarm_beta_cli_rejects_unbounded_prompt_batch(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args([
                "public-real-llm-swarm-beta",
                "local-smoke",
                "--prompt-texts",
                "one,two,three,four,five",
            ])

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
        args = cli.parse_args([
            "infer",
            "CrowdTensor user prompt",
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
            self.assertEqual(command[command.index("--prompt-text") + 1], "CrowdTensor user prompt")
            self.assertEqual(command[command.index("--max-new-tokens") + 1], "8")
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
        self.assertEqual(report["recommended_next_command"]["label"], "run broader local evidence")
        self.assertEqual(report["recommended_next_command"]["reason"], "collect_broader_evidence")
        self.assertEqual(report["route"]["route_source"], "local-product-loopback")
        self.assertIn("crowdtensor_infer_ready", report["diagnosis_codes"])
        next_lines = [item["command_line"] for item in report["next_commands"]]
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode local --output-dir {output_dir} --max-new-tokens 8",
            next_lines,
        )
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode local --output-dir {output_dir} --max-new-tokens 8 --full-evidence",
            next_lines,
        )
        self.assertNotIn("CrowdTensor user prompt", json.dumps(report, sort_keys=True))
        self.assertTrue(report["artifacts"]["product_swarm_mvp_report"]["present"] is False)
        self.assertTrue((output_dir / "infer_summary.json").is_file())
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["result"]["status"], "complete")
        self.assertEqual(persisted["result"]["display"], "hash-only-json")
        self.assertTrue(persisted["result"]["public_artifact_safe"])
        self.assertEqual(persisted["recommended_next_command"]["label"], "run broader local evidence")
        self.assertNotIn("CrowdTensor user prompt", json.dumps(persisted, sort_keys=True))
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
        self.assertIn("status, action, recommended_next", progress)
        self.assertNotIn(prompt, progress)

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
        self.assertIn("checking credentials and request requirements", progress)
        self.assertNotIn("submitting to the existing swarm", progress)
        self.assertNotIn(prompt, progress)

    def test_infer_local_batch_forwards_only_prompt_texts_to_product_loopback(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "--prompt-texts",
            "first prompt,second prompt",
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
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
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
        self.assertTrue(calls)

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
            self.assertEqual(command[command.index("--prompt-text") + 1], prompt)
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
            self.assertEqual(command[command.index("--prompt-text") + 1], prompt)
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
        self.assertIn(
            f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode local --output-dir {output_dir} --coordinator-port 9799 --max-new-tokens 8",
            next_lines,
        )
        self.assertNotIn(prompt, json.dumps(report, sort_keys=True))
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["issue_summary"]["primary_code"], "serve_start_failed")
        self.assertIn("loopback Coordinator", persisted["operator_action"])
        self.assertEqual(persisted["recommended_next_command"]["label"], "retry local inference on a fresh port")
        self.assertIn("--coordinator-port 9799", persisted["recommended_next_command"]["command_line"])
        self.assertNotIn(prompt, json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("primary=serve_start_failed", markdown)
        self.assertIn("loopback Coordinator", markdown)
        self.assertIn("--coordinator-port 9799", markdown)
        self.assertNotIn(prompt, markdown)

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
                        },
                    }
                },
                "diagnosis_codes": ["public_swarm_inference_v2_ready"],
            })

        report = cli.build_infer(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)

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
            return completed({
                "schema": "public_swarm_inference_v2",
                "ok": False,
                "mode": "local",
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
        self.assertIn("public_swarm_inference_v2_blocked", report["diagnosis_codes"])
        self.assertEqual(report["issue_summary"]["primary_code"], "public_swarm_inference_v2_blocked")
        self.assertEqual(report["review_summary"]["primary_code"], "public_swarm_inference_v2_blocked")
        self.assertIn("Full local Public Swarm v2 evidence is blocked", report["operator_action"])
        self.assertEqual(report["recommended_next_command"]["label"], "rerun full local evidence")
        self.assertIn("--full-evidence", report["recommended_next_command"]["command_line"])
        self.assertNotIn(prompt, json.dumps(report, sort_keys=True))
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["issue_summary"]["primary_code"], "public_swarm_inference_v2_blocked")
        self.assertEqual(persisted["recommended_next_command"]["label"], "rerun full local evidence")
        self.assertIn("--full-evidence", persisted["recommended_next_command"]["command_line"])
        self.assertNotIn(prompt, json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("primary=public_swarm_inference_v2_blocked", markdown)
        self.assertIn("--full-evidence", markdown)
        self.assertNotIn(prompt, markdown)

    def test_infer_full_evidence_batch_forwards_only_prompt_texts(self) -> None:
        output_dir = Path(self._tmp_dir())
        args = cli.parse_args([
            "infer",
            "--prompt-texts",
            "first prompt,second prompt",
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
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
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
        self.assertTrue(calls)

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
                            "generated_token_count": 1,
                            "max_new_tokens": 3,
                            "generation_step": 0,
                            "generated_text_hash": "sha256:step0",
                            "generated_text": "must not leak",
                            "generated_token_ids": [1],
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
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("must not leak", encoded)
        self.assertNotIn('"generated_token_ids": [1]', encoded)
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        persisted_encoded = json.dumps(persisted, sort_keys=True)
        self.assertEqual(persisted["stream"]["progress"]["observed_token_counts"], [1, 2, 3])
        self.assertNotIn("must not leak", persisted_encoded)
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
        self.assertEqual(
            report["local_output_note"],
            "Shown only in local human output; JSON and saved artifacts keep raw generated text redacted.",
        )
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["local_output"]["generated_text"], "")
        self.assertEqual(persisted["local_output"]["outputs"][0]["generated_text"], "")
        self.assertFalse(persisted["local_output"]["available"])
        self.assertFalse(persisted["local_output"]["display_only"])
        self.assertTrue(persisted["local_output"]["public_artifact_safe"])
        self.assertEqual(persisted["result"]["status"], "complete")
        self.assertEqual(persisted["result"]["output_count"], 1)
        self.assertEqual(persisted["result"]["display"], "hash-only")
        self.assertTrue(persisted["result"]["public_artifact_safe"])
        self.assertEqual(persisted["output_display"]["terminal_display"], "local-private")
        self.assertFalse(persisted["output_display"]["terminal_text_available"])
        self.assertEqual(persisted["output_display"]["saved_artifact_display"], "hash-only")
        self.assertEqual(persisted["output_display"]["json_stdout_display"], "hash-only-json")
        self.assertFalse(persisted["output_display"]["raw_generated_text_public"])
        self.assertTrue(persisted["output_display"]["public_artifact_safe"])
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("- Model: `sshleifer/tiny-gpt2` backend=`cpu`", markdown)
        self.assertIn("- Prompt: `count=1 hash=", markdown)
        self.assertIn("raw_public=False`", markdown)
        self.assertIn("- Result: `status=complete tokens=8/8 outputs=1 display=hash-only hash=", markdown)
        self.assertIn("public_artifact_safe=True`", markdown)
        self.assertIn(
            "- Output display: `terminal=local-private terminal_text=False saved=hash-only json_stdout=hash-only-json include_output=False raw_public=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Local output: `available=False display_only=False public_artifact_safe=True` count=`1` source=`local-private-task-state`",
            markdown,
        )
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
            "  result: status=complete tokens=16/16 outputs=1 display=local-private hash=sha256:generated public_artifact_safe=False",
            stdout.getvalue(),
        )
        self.assertIn(
            "  output_display: terminal=local-private terminal_text=True saved=hash-only json_stdout=hash-only-json include_output=True raw_public=False public_artifact_safe=True",
            stdout.getvalue(),
        )
        self.assertIn(
            "  status: completed: Inference completed. next=rerun_or_review_artifacts recommendation=submit inference public_artifact_safe=True",
            stdout.getvalue(),
        )
        self.assertIn(
            f"  review: state=completed next=rerun_or_review_artifacts inspect={output_dir / 'infer_summary.md'} recommended=submit inference primary=crowdtensor_infer_ready attention=none public_artifact_safe=True",
            stdout.getvalue(),
        )
        self.assertIn(
            "  review_next: label=submit inference reason=rerun_inference command=CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor infer '<prompt>' --mode existing",
            stdout.getvalue(),
        )
        self.assertIn(
            "  issue: state=completed primary=crowdtensor_infer_ready next=rerun_or_review_artifacts progress=`polls=2 accepted_rows=1 tokens=16/16 ledger=True stream=False` safe_detail=False",
            stdout.getvalue(),
        )
        self.assertIn(
            "  trace: session=real-llm-session-infer requests=1 ledger_rows=1 stream_events=0 source=public_swarm_product_cli_v1 public_artifact_safe=True",
            stdout.getvalue(),
        )
        self.assertIn(
            "  shareable: saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=True",
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
        self.assertIn("next[2] submit inference", stdout.getvalue())
        self.assertIn(
            "recommended_next: submit inference reason=rerun_inference CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor infer '<prompt>' --mode existing",
            stdout.getvalue(),
        )
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
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
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
        self.assertEqual(report["review_summary"]["recommended_label"], "submit inference")
        self.assertEqual(report["review_summary"]["primary_code"], "crowdtensor_infer_ready")
        self.assertEqual(report["review_summary"]["attention"], "")
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
        self.assertTrue(persisted["output_request"]["include_output"])
        self.assertFalse(persisted["output_request"]["raw_generated_text_public"])
        self.assertEqual(persisted["local_output"]["generated_text"], "")
        self.assertFalse(persisted["local_output"]["display_only"])
        self.assertTrue(persisted["local_output"]["public_artifact_safe"])
        self.assertEqual(persisted["result"]["status"], "complete")
        self.assertEqual(persisted["result"]["output_count"], 1)
        self.assertEqual(persisted["result"]["display"], "hash-only")
        self.assertTrue(persisted["result"]["public_artifact_safe"])
        self.assertEqual(persisted["output_display"]["terminal_display"], "local-private")
        self.assertFalse(persisted["output_display"]["terminal_text_available"])
        self.assertTrue(persisted["output_display"]["include_output_requested"])
        self.assertFalse(persisted["output_display"]["raw_generated_text_public"])
        self.assertTrue(persisted["output_display"]["public_artifact_safe"])
        self.assertEqual(persisted["user_status"]["state"], "completed")
        self.assertEqual(persisted["user_status"]["headline"], "Inference completed.")
        self.assertEqual(persisted["user_status"]["next_step"], "rerun_or_review_artifacts")
        self.assertEqual(persisted["user_status"]["recommended_label"], "submit inference")
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
        self.assertEqual(persisted["recommended_next_command"]["label"], "submit inference")
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
        self.assertIn("## What To Do Next", markdown)
        self.assertIn("- State: `completed`", markdown)
        self.assertIn("- Next step: `rerun_or_review_artifacts`", markdown)
        self.assertIn("- Recommended: `submit inference` reason=`rerun_inference`", markdown)
        self.assertIn("- Copy command: `CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor infer '<prompt>' --mode existing", markdown)
        self.assertIn("- Requires env: `CROWDTENSOR_ADMIN_TOKEN`", markdown)
        self.assertIn("- Safety: saved Markdown keeps prompt placeholders and redacted generated output.", markdown)
        self.assertIn("## Details", markdown)
        self.assertIn(
            "- Status: `completed: Inference completed. next=rerun_or_review_artifacts recommendation=submit inference public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            f"- Review: `state=completed next=rerun_or_review_artifacts inspect={output_dir / 'infer_summary.md'} recommended=submit inference primary=crowdtensor_infer_ready attention=none public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Review next: `label=submit inference reason=rerun_inference command=CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor infer '<prompt>' --mode existing",
            markdown,
        )
        self.assertIn(
            "- Issue: `state=completed primary=crowdtensor_infer_ready next=rerun_or_review_artifacts progress=`",
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
            "- Output display: `terminal=local-private terminal_text=False saved=hash-only json_stdout=hash-only-json include_output=True raw_public=False public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Trace: `session=real-llm-session-infer requests=1 ledger_rows=1 stream_events=0 source=public_swarm_product_cli_v1 public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False local_output_display_only=True`",
            markdown,
        )
        self.assertIn(
            "- Recommended next: `submit inference` reason=`rerun_inference` command=`CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} crowdtensor infer '<prompt>' --mode existing",
            markdown,
        )
        self.assertIn("requires=`CROWDTENSOR_ADMIN_TOKEN`", markdown)
        self.assertIn("- Wait: `polls=2 accepted_rows=1 tokens=16/16 ledger=True stream=False`", markdown)
        self.assertIn(
            "- Local output: `available=False display_only=False public_artifact_safe=True` count=`1` source=``",
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
        self.assertIn("`submit inference`", markdown)
        self.assertIn("Replace `<prompt>` with your local prompt before running saved commands.", markdown)
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
        self.assertIn(f"review_next: label=check existing swarm reason=verify_stage_miners command=crowdtensor infer '{prompt}' --mode existing", rendered)
        self.assertIn(f"recommended_next: check existing swarm reason=verify_stage_miners crowdtensor infer '{prompt}' --mode existing", rendered)
        self.assertIn(f"next[1] check existing swarm: crowdtensor infer '{prompt}' --mode existing", rendered)
        self.assertNotIn(cli.INFER_PROMPT_PLACEHOLDER, rendered)

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
        self.assertNotIn("  answer:  first output", rendered)
        self.assertIn("  answer[1]:  first output", rendered)
        self.assertIn("  answer[2]:  second output", rendered)
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["local_output"]["output_count"], 2)
        self.assertEqual([row["generated_text"] for row in persisted["local_output"]["outputs"]], ["", ""])
        self.assertEqual(persisted["trace"]["request_count"], 2)
        self.assertEqual(
            [(row["request_id"], row["prompt_hash"]) for row in persisted["trace"]["request_trace"]],
            [("req-1", "sha256:p1"), ("req-2", "sha256:p2")],
        )
        self.assertTrue(persisted["trace"]["public_artifact_safe"])
        self.assertFalse(persisted["local_output"]["available"])
        self.assertFalse(persisted["local_output"]["display_only"])
        self.assertNotIn("first output", json.dumps(persisted, sort_keys=True))
        self.assertNotIn("second output", json.dumps(persisted, sort_keys=True))

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
        self.assertIn("  review_next: label=submit inference reason=rerun_inference", rendered)
        self.assertIn("attention=request[2]=req-2:1/2", rendered)
        self.assertIn("  stream[1]: request=req-1 tokens=2/2 counts=[1, 2] complete=True missing=False", rendered)
        self.assertIn("  stream[2]: request=req-2 tokens=1/2 counts=[1] complete=False missing=False", rendered)
        self.assertIn("  stream_issue: request[2]=req-2:1/2", rendered)
        self.assertIn("  action: Inference completed, but stream progress is incomplete (request[2]=req-2:1/2); rerun with --stream if you need live token evidence.", rendered)
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("must not leak", encoded)
        self.assertNotIn('"generated_token_ids": [1]', encoded)
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["stream"]["issue_summary"], "request[2]=req-2:1/2")
        self.assertEqual(persisted["review_summary"]["attention"], "request[2]=req-2:1/2")
        self.assertEqual(persisted["trace"]["stream_event_count"], 3)
        self.assertEqual(
            [(row["request_id"], row["prompt_hash"]) for row in persisted["trace"]["request_trace"]],
            [("req-1", "sha256:p1"), ("req-2", "sha256:p2")],
        )
        self.assertEqual(
            persisted["operator_action"],
            "Inference completed, but stream progress is incomplete (request[2]=req-2:1/2); rerun with --stream if you need live token evidence.",
        )
        self.assertNotIn("must not leak", json.dumps(persisted, sort_keys=True))
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn("attention=request[2]=req-2:1/2", markdown)
        self.assertIn("- Review next: `label=submit inference reason=rerun_inference", markdown)
        self.assertIn("- Stream issue: `request[2]=req-2:1/2`", markdown)
        self.assertIn("Inference completed, but stream progress is incomplete", markdown)
        self.assertIn("Replace `<prompt-1>,<prompt-2>` with your comma-separated local prompts before running saved commands.", markdown)
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
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertEqual(report["local_output"]["generated_text"], "")
        self.assertTrue(report["local_output"]["public_artifact_safe"])
        self.assertEqual(
            report["local_output_note"],
            "Raw generated text is suppressed in JSON/public output; rerun without --json for local display.",
        )
        persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
        self.assertTrue(persisted["output_request"]["include_output"])
        self.assertFalse(persisted["output_request"]["raw_generated_text_public"])
        self.assertTrue(persisted["local_output"]["public_artifact_safe"])
        self.assertEqual(
            persisted["local_output_note"],
            "Raw generated text is suppressed in JSON/public output; rerun without --json for local display.",
        )
        markdown = (output_dir / "infer_summary.md").read_text(encoding="utf-8")
        self.assertIn(
            "- Local output: `available=False display_only=False public_artifact_safe=True` count=`0` source=``",
            markdown,
        )
        self.assertIn(
            "- Local output note: Raw generated text is suppressed in JSON/public output; rerun without --json for local display.",
            markdown,
        )
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
            "  local_output: available=True display_only=False public_artifact_safe=False count=1 source=local-private-task-state",
            rendered,
        )
        self.assertNotIn("\nsecond line\n", rendered)
        self.assertLess(rendered.index("  answer: first line"), rendered.index("  local_output: "))

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
                    argv.extend(["--timeout-seconds", "90"])
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
                next_lines = [item["command_line"] for item in report["next_commands"]]
                expected_retry_timeout = "180" if case_index == 0 else "240"
                self.assertIn(
                    f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --max-new-tokens 4 --coordinator-url http://127.0.0.1:8787 --timeout-seconds {expected_retry_timeout}",
                    next_lines,
                )
                retry = next(item for item in report["next_commands"] if item["label"] == "retry inference with longer timeout")
                self.assertEqual(retry["requires_env"], ["CROWDTENSOR_ADMIN_TOKEN"])
                self.assertEqual(retry["command"].count("--timeout-seconds"), 1)
                self.assertEqual(retry["command"][retry["command"].index("--timeout-seconds") + 1], expected_retry_timeout)
                self.assertNotIn("--timeout-seconds 90 --timeout-seconds 180", retry["command_line"])
                self.assertNotIn("CrowdTensor user prompt", json.dumps(report["next_commands"], sort_keys=True))
                self.assertNotIn("admin-secret", json.dumps(report, sort_keys=True))
                persisted = json.loads((output_dir / "infer_summary.json").read_text(encoding="utf-8"))
                self.assertIn(expected, persisted["operator_action"])
                self.assertIn(
                    f"crowdtensor infer '{cli.INFER_PROMPT_PLACEHOLDER}' --mode existing --output-dir {output_dir} --max-new-tokens 4 --coordinator-url http://127.0.0.1:8787 --timeout-seconds {expected_retry_timeout}",
                    [item["command_line"] for item in persisted["next_commands"]],
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
            "source": "infer-existing-preflight",
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
            "- Coordinator: `True service=crowdtensord-coordinator protocol=runtime_contract_v1`",
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
        self.assertIn("  ready_to_submit: True label=verified fully_verified=True route=True coordinator=True stage=ready stage_verification=ready next_step=submit warnings=none", rendered)
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
        self.assertIn(
            "- Status: `preflight-ready: Preflight passed; submit inference next. next=submit recommendation=submit inference public_artifact_safe=True`",
            markdown,
        )
        self.assertIn(
            "- Ready to submit: label=`verified` next_step=`submit` fully_verified=`True`",
            markdown,
        )
        self.assertIn(
            "- Coordinator: `True service=crowdtensord-coordinator protocol=runtime_contract_v1`",
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
        cases = [
            ["infer", "positional prompt", "--prompt-text", "flag prompt"],
            ["infer", "positional prompt", "--prompt-texts", "first prompt,second prompt"],
            ["infer", "--prompt", "flag prompt", "--prompt-texts", "first prompt,second prompt"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as raised:
                cli.parse_args(argv)
            self.assertEqual(
                str(raised.exception),
                "infer accepts one prompt source: positional prompt, --prompt-text/--prompt, or --prompt-texts",
            )

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
            "  coordinator_ready: False service=None protocol=None error=OSError",
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
        with self.assertRaises(SystemExit):
            cli.parse_args(["infer", "prompt", "--dry-run"])

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
            "diagnosis_codes": ["public_swarm_inference_v2_ready"],
            "artifacts": {},
        }
        buf = io.StringIO()

        with contextlib.redirect_stdout(buf):
            cli.print_public_swarm_inference_v2(report)
        output = buf.getvalue()

        self.assertIn("local accepted rows: 32 ready=True", output)
        self.assertIn("kv cache ready: True", output)
        self.assertIn("batch ready: True", output)
        self.assertIn("stream ready: True", output)
        self.assertIn("external ready: True tokens=16/16 accepted_rows=32 rows_ready=True", output)
        self.assertIn("model match: local=True external=True p2p=True", output)

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

    def test_main_gpu_generate_json_outputs_summary(self) -> None:
        summary = {"schema": "gpu_sharded_generation_beta_v1", "ok": True}
        with patch.object(cli, "build_gpu_sharded_generation_beta", return_value=summary), patch("builtins.print") as mocked_print:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["gpu-generate", "local-loopback", "--json"])

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["schema"], "gpu_sharded_generation_beta_v1")


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
