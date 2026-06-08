#!/usr/bin/env python3
"""CI-safe checks for user-friendly infer/generate front door artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor import cli  # noqa: E402


SCHEMA = "user_friendly_inference_frontdoor_check_v1"
CHECK_READY = "user_friendly_inference_frontdoor_check_ready"
CHECK_FAILED = "user_friendly_inference_frontdoor_check_failed"
PROMPT_TEXT = "CrowdTensor frontdoor private prompt"
INFER_TEXT = "frontdoor infer answer must remain local only"
GENERATE_TEXT = "frontdoor generate answer must remain local only"
ADMIN_TOKEN = "frontdoor-admin-secret"
SECRET_FRAGMENTS = [
    PROMPT_TEXT,
    INFER_TEXT,
    GENERATE_TEXT,
    ADMIN_TOKEN,
    "lease_token",
    "idempotency_key",
    "Bearer ",
    '"generated_text": "frontdoor',
    '"prompt_text":',
    '"generated_token_ids": [',
    "input_ids",
    "hidden_state",
    "logits",
    "activation_results",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _generation_summary(text: str, *, max_new_tokens: int, request_count: int = 1) -> dict[str, Any]:
    return {
        "generated_token_count": max_new_tokens,
        "max_new_tokens": max_new_tokens,
        "generated_text_hash": cli.stable_hash_text(text),
        "decoded_tokens_match": True,
        "request_count": request_count,
        "observed_request_count": request_count,
        "expected_request_count": request_count,
        "batch_generation_ready": request_count > 1,
        "multi_token_generation_ready": max_new_tokens > 1,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }


def _local_output(text: str, *, source: str, max_new_tokens: int, request_id: str) -> dict[str, Any]:
    return {
        "available": True,
        "source": source,
        "generated_text": text,
        "outputs": [
            {
                "request_id": request_id,
                "prompt_hash": cli.stable_hash_text(PROMPT_TEXT),
                "generated_token_count": max_new_tokens,
                "generated_text": text,
                "truncated": False,
                "max_chars": cli.LOCAL_OUTPUT_DISPLAY_MAX_CHARS,
                "omitted_char_count": 0,
            }
        ],
        "output_count": 1,
        "display_only": True,
        "public_artifact_safe": False,
        "truncated": False,
        "max_chars": cli.LOCAL_OUTPUT_DISPLAY_MAX_CHARS,
        "omitted_char_count": 0,
        "note": "Raw generated text is shown only in local human output; JSON and saved artifacts expose hashes only.",
    }


def fake_infer_payload(*, max_new_tokens: int) -> dict[str, Any]:
    return {
        "schema": "product_swarm_mvp_check_v1",
        "ok": True,
        "mode": "local-loopback",
        "generation": _generation_summary(INFER_TEXT, max_new_tokens=max_new_tokens),
        "route": {
            "route_ready": True,
            "route_source": "local-product-loopback",
            "coordinator_url_present": True,
            "distinct_stage_miners": True,
            "stage_assignment_valid": True,
            "public_artifact_safe": True,
        },
        "stage_assignment": {
            "distinct_stage_miners": True,
            "stage0_miner_id": "frontdoor-stage0",
            "stage1_miner_id": "frontdoor-stage1",
            "completed_rows": 2,
            "public_artifact_safe": True,
        },
        "ledger": {"accepted_rows": 2, "public_artifact_safe": True},
        "wait_progress": {
            "session_created": True,
            "accepted_rows_seen": 2,
            "observed_request_count": 1,
            "expected_request_count": 1,
            "max_observed_token_count": max_new_tokens,
            "target_token_count": max_new_tokens,
            "batch_generation_ready": True,
            "completion_observed": True,
            "ledger_endpoint_ready": True,
            "stream_endpoint_ready": False,
            "public_artifact_safe": True,
        },
        "local_output": _local_output(
            INFER_TEXT,
            source="local-private-task-state",
            max_new_tokens=max_new_tokens,
            request_id="infer-frontdoor-1",
        ),
        "safety": {
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "raw_runtime_state_removed": True,
            "private_runtime_state_kept": False,
            "public_artifact_safe": True,
        },
        "diagnosis_codes": [
            "product_swarm_mvp_ready",
            "public_swarm_generate_ready",
            "distinct_stage_miners",
            "stage_assignment_valid",
        ],
    }


def build_fake_infer_report(output_dir: Path, *, max_new_tokens: int) -> dict[str, Any]:
    args = cli.parse_args([
        "infer",
        PROMPT_TEXT,
        "--output-dir",
        str(output_dir),
        "--max-new-tokens",
        str(max_new_tokens),
    ])
    args.coordinator_port_auto = True
    args.coordinator_port_explicit = False
    return cli._infer_summary_from_payload(
        args,
        fake_infer_payload(max_new_tokens=max_new_tokens),
        mode="local",
        output_dir=output_dir,
    )


def fake_generate_report(output_dir: Path, *, max_new_tokens: int) -> dict[str, Any]:
    return {
        "schema": cli.PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
        "generated_at": cli.utc_now(),
        "ok": True,
        "mode": "generate",
        "json_mode": False,
        "dry_run": False,
        "output_dir": str(output_dir),
        "output_dir_explicit": True,
        "route": {
            "route_source": "coordinator-url",
            "coordinator_url_present": True,
            "coordinator_url": "http://127.0.0.1:8787",
            "route_ready": True,
            "backend": "cpu",
            "hf_model_id": "sshleifer/tiny-gpt2",
            "public_artifact_safe": True,
        },
        "session_request": {
            "scenario_id": "frontdoor-check",
            "hf_model_id": "sshleifer/tiny-gpt2",
            "backend": "cpu",
            "max_new_tokens": max_new_tokens,
            "request_count": 1,
            "prompt_count": 1,
            "prompt_hashes": [cli.stable_hash_text(PROMPT_TEXT)],
            "prompt_char_counts": [len(PROMPT_TEXT)],
            "raw_prompt_public": False,
            "public_artifact_safe": True,
        },
        "prompt": {
            "prompt_hash": cli.stable_hash_payload([cli.stable_hash_text(PROMPT_TEXT)]),
            "prompt_count": 1,
            "raw_prompt_public": False,
        },
        "prompt_scope": cli._prompt_scope(source="prompt-text", prompt_count=1),
        "session": {
            "schema": "real_llm_sharded_session_v1",
            "session_id": "frontdoor-generate-session",
            "workload_type": "real_llm_sharded_infer_v1",
            "max_new_tokens": max_new_tokens,
            "backend": "hf_transformers_cpu",
            "hf_model_id": "sshleifer/tiny-gpt2",
        },
        "generation": _generation_summary(GENERATE_TEXT, max_new_tokens=max_new_tokens),
        "batch": {
            "enabled": False,
            "request_count": 1,
            "expected_request_count": 1,
            "observed_request_count": 1,
            "ready": True,
        },
        "wait_progress": {
            "session_created": True,
            "poll_count": 1,
            "accepted_rows_seen": 2,
            "last_accepted_rows": 2,
            "max_observed_token_count": max_new_tokens,
            "target_token_count": max_new_tokens,
            "observed_request_count": 1,
            "expected_request_count": 1,
            "batch_generation_ready": True,
            "completion_observed": True,
            "ledger_endpoint_ready": True,
            "stream_endpoint_ready": False,
            "public_artifact_safe": True,
        },
        "stream": {
            "enabled": False,
            "requested": False,
            "ready": False,
            "event_count": 0,
            "events": [],
            "progress": {},
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        },
        "local_output": _local_output(
            GENERATE_TEXT,
            source="coordinator-validation",
            max_new_tokens=max_new_tokens,
            request_id="generate-frontdoor-1",
        ),
        "local_output_note": "Raw generated text is shown only in local human output; JSON and saved artifacts expose hashes only.",
        "output_request": {
            "include_output": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "public_artifact_safe": True,
        },
        "runtime_options": {
            "timeout_seconds": 120.0,
            "poll_interval": 1.0,
            "http_timeout": 30.0,
            "admin_results_limit": 50,
            "public_artifact_safe": True,
        },
        "diagnosis_codes": [
            "public_swarm_generate_ready",
            "distinct_stage_miners",
            "stage_assignment_valid",
        ],
        "limitations": [
            "Coordinator-backed, read-only, tiny/small-model scoped; not production Hivemind/Petals parity or large-model serving.",
        ],
    }


def build_fake_generate_report(output_dir: Path, *, max_new_tokens: int) -> dict[str, Any]:
    return cli._finalize_product_generate_report(
        fake_generate_report(output_dir, max_new_tokens=max_new_tokens),
        admin_token=ADMIN_TOKEN,
        output_dir=output_dir,
    )


def _required_markdown_fragments(kind: str) -> list[str]:
    return [
        "- Verdict:",
        "- Verdict note:",
        "answer=saved-terminal-redacted",
        "answer_visible=False",
        "- Answer scope:",
        "- Output display:",
        "- GPU status:",
        "fresh_kaggle_gpu=False",
        "saved JSON/Markdown contain no generated text",
        "Raw prompts are not public.",
    ]


def _validate_frontdoor_artifact(
    *,
    kind: str,
    report: dict[str, Any],
    summary_path: Path,
    markdown_path: Path,
    raw_answer: str,
    errors: list[str],
) -> dict[str, Any]:
    persisted = load_json(summary_path)
    markdown = markdown_path.read_text(encoding="utf-8")
    verdict = persisted.get("inference_verdict") if isinstance(persisted.get("inference_verdict"), dict) else {}
    terminal_verdict = report.get("inference_verdict") if isinstance(report.get("inference_verdict"), dict) else {}
    answer_scope = persisted.get("answer_scope") if isinstance(persisted.get("answer_scope"), dict) else {}
    local_output = persisted.get("local_output") if isinstance(persisted.get("local_output"), dict) else {}
    output_display = persisted.get("output_display") if isinstance(persisted.get("output_display"), dict) else {}
    runtime = persisted.get("runtime_provenance") if isinstance(persisted.get("runtime_provenance"), dict) else {}
    evidence = persisted.get("evidence_scope") if isinstance(persisted.get("evidence_scope"), dict) else {}
    gpu = persisted.get("gpu_status") if isinstance(persisted.get("gpu_status"), dict) else {}
    combined = json.dumps(persisted, sort_keys=True) + "\n" + markdown

    if terminal_verdict.get("answer_visible_in_terminal") is not True:
        errors.append(f"{kind}_terminal_verdict_answer_visibility_mismatch")
    if report.get("answer_scope", {}).get("scope_state") != "terminal-visible":
        errors.append(f"{kind}_terminal_answer_scope_mismatch")
    if verdict.get("schema") != "crowdtensor_inference_verdict_v1":
        errors.append(f"{kind}_verdict_schema_mismatch")
    if verdict.get("kind") != kind:
        errors.append(f"{kind}_verdict_kind_mismatch")
    if verdict.get("state") != "completed" or verdict.get("completed") is not True:
        errors.append(f"{kind}_verdict_not_completed")
    if verdict.get("result_status") != "complete":
        errors.append(f"{kind}_verdict_result_status_mismatch")
    if verdict.get("answer_scope_state") != "saved-terminal-redacted":
        errors.append(f"{kind}_saved_verdict_answer_scope_mismatch")
    if verdict.get("answer_visible_in_terminal") is not False:
        errors.append(f"{kind}_saved_verdict_answer_visibility_mismatch")
    if verdict.get("saved_artifacts_public_safe") is not True or verdict.get("public_artifact_safe") is not True:
        errors.append(f"{kind}_verdict_public_artifact_safe_mismatch")
    if verdict.get("fresh_kaggle_gpu_verified") is not False:
        errors.append(f"{kind}_fresh_kaggle_gpu_claim_mismatch")
    if answer_scope.get("scope_state") != "saved-terminal-redacted":
        errors.append(f"{kind}_saved_answer_scope_mismatch")
    if answer_scope.get("visible_in_terminal") is not False:
        errors.append(f"{kind}_saved_answer_visible_mismatch")
    if answer_scope.get("public_artifact_safe") is not True:
        errors.append(f"{kind}_answer_scope_public_safe_mismatch")
    if local_output.get("available") is not False:
        errors.append(f"{kind}_local_output_available_mismatch")
    if local_output.get("generated_text") not in {"", None}:
        errors.append(f"{kind}_local_output_text_not_redacted")
    if local_output.get("public_artifact_safe") is not True:
        errors.append(f"{kind}_local_output_public_safe_mismatch")
    if output_display.get("terminal_text_available") is not False:
        errors.append(f"{kind}_saved_output_display_terminal_text_mismatch")
    if output_display.get("terminal_display") != "saved-terminal-redacted":
        errors.append(f"{kind}_saved_output_display_state_mismatch")
    if output_display.get("saved_artifact_display") != "hash-only":
        errors.append(f"{kind}_saved_output_display_artifact_mismatch")
    if runtime.get("fresh_kaggle_gpu_verified") is not False:
        errors.append(f"{kind}_runtime_fresh_kaggle_gpu_mismatch")
    if evidence.get("fresh_kaggle_gpu_verified") is not False:
        errors.append(f"{kind}_evidence_fresh_kaggle_gpu_mismatch")
    if gpu.get("fresh_kaggle_gpu_verified") is not False:
        errors.append(f"{kind}_gpu_fresh_kaggle_gpu_mismatch")
    for fragment in _required_markdown_fragments(kind):
        if fragment not in markdown:
            errors.append(f"{kind}_markdown_missing_{fragment[:24].strip().replace(' ', '_').replace('`', '')}")
    for fragment in SECRET_FRAGMENTS:
        if fragment and fragment in combined:
            errors.append(f"{kind}_artifact_leaked_{fragment[:32]}")
    if raw_answer in combined:
        errors.append(f"{kind}_raw_answer_leaked")
    return {
        "summary_path": str(summary_path),
        "markdown_path": str(markdown_path),
        "terminal_verdict": terminal_verdict,
        "saved_verdict": verdict,
        "saved_answer_scope": answer_scope.get("scope_state") or "",
        "saved_output_display": output_display.get("terminal_display") or "",
        "evidence_level": verdict.get("evidence_level") or "",
        "executed_where": verdict.get("executed_where") or "",
        "gpu_state": verdict.get("gpu_state") or "",
        "public_artifact_safe": bool(verdict.get("public_artifact_safe")),
    }


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    temp_ctx = None
    if args.output_dir:
        output_root = Path(args.output_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
    elif args.keep_temp:
        output_root = Path(tempfile.mkdtemp(prefix="crowdtensor_frontdoor_check_"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="crowdtensor_frontdoor_check_")
        output_root = Path(temp_ctx.name)
    try:
        infer_dir = output_root / "infer"
        generate_dir = output_root / "generate"
        errors: list[str] = []
        infer_report = build_fake_infer_report(infer_dir, max_new_tokens=args.max_new_tokens)
        generate_report = build_fake_generate_report(generate_dir, max_new_tokens=args.max_new_tokens)
        infer_check = _validate_frontdoor_artifact(
            kind="Inference",
            report=infer_report,
            summary_path=infer_dir / "infer_summary.json",
            markdown_path=infer_dir / "infer_summary.md",
            raw_answer=INFER_TEXT,
            errors=errors,
        )
        generate_check = _validate_frontdoor_artifact(
            kind="Generation",
            report=generate_report,
            summary_path=generate_dir / "generate_summary.json",
            markdown_path=generate_dir / "generate_summary.md",
            raw_answer=GENERATE_TEXT,
            errors=errors,
        )
        result = {
            "schema": SCHEMA,
            "ok": not errors,
            "output_dir": str(output_root),
            "checked_infer_verdict": infer_check.get("saved_verdict"),
            "checked_generate_verdict": generate_check.get("saved_verdict"),
            "infer": infer_check,
            "generate": generate_check,
            "errors": errors,
            "diagnosis_codes": [CHECK_READY] if not errors else [CHECK_FAILED],
            "safety": {
                "ci_safe": True,
                "uses_fake_completed_frontdoor_reports": True,
                "started_coordinator": False,
                "submitted_live_task": False,
                "fresh_kaggle_gpu_attempted": False,
                "fresh_kaggle_gpu_verified": False,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "public_artifact_safe": not errors,
            },
        }
        write_json(output_root / "user_friendly_inference_frontdoor_check.json", result)
        return result
    finally:
        if temp_ctx is not None and not args.keep_temp:
            temp_ctx.cleanup()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the user-friendly infer/generate front door report check.")
    parser.add_argument("--output-dir", default="", help="keep generated check artifacts in this directory")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--keep-temp", action="store_true", help="keep the temporary directory when --output-dir is omitted")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = run_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"User-friendly inference frontdoor check ok={result.get('ok')} errors={','.join(result.get('errors') or [])}")
        infer_verdict = result.get("checked_infer_verdict") if isinstance(result.get("checked_infer_verdict"), dict) else {}
        generate_verdict = (
            result.get("checked_generate_verdict")
            if isinstance(result.get("checked_generate_verdict"), dict)
            else {}
        )
        if infer_verdict:
            print(f"infer_verdict: {cli.inference_verdict_text(infer_verdict)}")
        if generate_verdict:
            print(f"generate_verdict: {cli.inference_verdict_text(generate_verdict)}")
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
