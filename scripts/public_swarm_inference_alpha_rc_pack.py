#!/usr/bin/env python3
"""Build the Public Swarm Inference Alpha release-candidate evidence pack.

This layer does not create live Kaggle resources.  It imports retained public
Alpha reports, verifies that both stage-specific live requeue proofs are
present and shareable, and emits a single release-candidate artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import support_bundle  # noqa: E402


SCHEMA = "public_swarm_inference_alpha_rc_v1"
ALPHA_SCHEMA = "public_swarm_inference_alpha_v1"
SUMMARY_SCHEMA = "public_swarm_inference_alpha_live_requeue_summary_v1"
CHECK_SCHEMA = "public_swarm_inference_alpha_check_v1"
MODE_EVIDENCE_IMPORT = "evidence-import"
MODE_LOCAL_SMOKE = "local-smoke"
DEFAULT_OUTPUT_DIR = "dist/public-swarm-inference-alpha-rc"
DEFAULT_STAGE0_REPORT = (
    "dist/public-swarm-inference-alpha-live-stage0-requeue-20260527165830/"
    "public_swarm_inference_alpha.json"
)
DEFAULT_STAGE1_REPORT = (
    "dist/public-swarm-inference-alpha-live-stage1-requeue-20260527170600/"
    "public_swarm_inference_alpha.json"
)
DEFAULT_SUMMARY_REPORT = "dist/public-swarm-inference-alpha-live-requeue-summary.json"

Runner = Callable[..., subprocess.CompletedProcess[str]]

SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "Bearer ",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    "real_llm_sharded_result",
    "generated_text",
    "generated_token_ids",
)

PRIVATE_ARTIFACT_NAMES = {
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
    "kernel.py",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


def redact_text(text: str) -> str:
    redacted = str(text)
    for fragment in SECRET_FRAGMENTS:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def redact_values(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_values(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_values(item) for key, item in value.items()}
    return value


def diagnosis_codes(*payloads: dict[str, Any]) -> list[str]:
    codes: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for code in payload.get("diagnosis_codes") or []:
            if isinstance(code, str):
                codes.add(code)
        summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
        for summary in summaries.values():
            if isinstance(summary, dict):
                for code in summary.get("diagnosis_codes") or []:
                    if isinstance(code, str):
                        codes.add(code)
    return sorted(codes)


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generation_public": False,
        "generation_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Public Swarm Alpha RC artifacts summarize retained stage requeue "
            "readiness, cleanup state, hashes/counts from imported evidence, and "
            "safety diagnostics only. They do not include answer text."
        ),
    }


def answer_scope_summary() -> dict[str, Any]:
    return {
        "scope_state": "no-local-answer",
        "terminal_only": False,
        "visible_in_terminal": False,
        "saved_json_display": "hash-only",
        "saved_markdown_display": "hash-only",
        "json_stdout_display": "hash-only-json",
        "raw_prompt_public": False,
        "raw_generation_public": False,
        "generation_ids_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This Alpha RC report is shareable retained-evidence audit output, not "
            "a local answer transcript; raw prompts, answer text, token ids, "
            "activations, leases, credentials, private env files, and raw runtime "
            "state are excluded."
        ),
    }


def shareable_summary() -> dict[str, Any]:
    return {
        "saved_artifacts_public_safe": True,
        "raw_prompt_public": False,
        "raw_generation_public": False,
        "generation_ids_public": False,
        "local_output_display_only": False,
        "answer_scope_state": "no-local-answer",
        "local_answer_terminal_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Share public_swarm_inference_alpha_rc.json/md; they contain retained "
            "readiness evidence, cleanup state, hashes, counts, and diagnostics, "
            "not raw prompts or answers."
        ),
    }


def find_private_artifacts(root: Path) -> list[str]:
    if not root.exists():
        return []
    matches: list[str] = []
    for child in root.rglob("*"):
        if child.name in PRIVATE_ARTIFACT_NAMES or child.suffix in {".tar", ".gz", ".tgz"}:
            try:
                matches.append(child.relative_to(root).as_posix())
            except ValueError:
                matches.append(str(child))
    return sorted(matches)


def live_requeue_summary(payload: dict[str, Any]) -> dict[str, Any]:
    session = payload.get("session") if isinstance(payload.get("session"), dict) else {}
    live_summary = session.get("live_summary") if isinstance(session.get("live_summary"), dict) else {}
    summary = live_summary.get("live_requeue_summary")
    if isinstance(summary, dict):
        return summary
    payload_summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
    live_payload = payload_summaries.get("live_kaggle") if isinstance(payload_summaries.get("live_kaggle"), dict) else {}
    summary = live_payload.get("live_requeue_summary")
    if isinstance(summary, dict):
        return summary
    beta = live_payload.get("beta_summary") if isinstance(live_payload.get("beta_summary"), dict) else {}
    summary = beta.get("live_requeue_summary")
    return summary if isinstance(summary, dict) else {}


def validate_stage_report(path: Path, payload: dict[str, Any], *, stage: str) -> dict[str, Any]:
    expected_failure = f"kill-{stage}-after-claim"
    expected_code = f"live_{stage}_requeue_ready"
    codes = set(diagnosis_codes(payload))
    session = payload.get("session") if isinstance(payload.get("session"), dict) else {}
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    cleanup = payload.get("artifact_cleanup") if isinstance(payload.get("artifact_cleanup"), dict) else {}
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    private_artifacts = find_private_artifacts(path.parent) if path.parent.exists() else []
    summary = live_requeue_summary(payload)
    required_codes = {
        "public_swarm_inference_alpha_ready",
        "public_swarm_session_ready",
        "public_swarm_live_requeue_ready",
        "public_swarm_live_kaggle_ready",
        "external_stage_requeue_ready",
        "external_runtime_verified",
        "kaggle_kernels_deleted",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "token_rotation_required",
        expected_code,
    }
    missing_codes = sorted(required_codes - codes)
    failed_checks: list[str] = []
    if not path.is_file():
        failed_checks.append("report_missing")
    if payload.get("schema") != ALPHA_SCHEMA:
        failed_checks.append("schema_mismatch")
    if payload.get("ok") is not True:
        failed_checks.append("report_not_ok")
    if payload.get("mode") != "live-kaggle":
        failed_checks.append("mode_not_live_kaggle")
    if payload.get("failure_mode") != expected_failure:
        failed_checks.append("failure_mode_mismatch")
    if missing_codes:
        failed_checks.append("missing_readiness_codes")
    if session.get("live_external_runtime_verified") is not True:
        failed_checks.append("external_runtime_not_verified")
    if session.get("live_stage_requeue_verified") is not True:
        failed_checks.append("live_requeue_not_verified")
    if session.get("live_kaggle_kernels_deleted") is not True:
        failed_checks.append("kaggle_kernels_not_deleted")
    if safety.get("cpu_only") is not True:
        failed_checks.append("cpu_only_missing")
    if safety.get("read_only_workload") != "real_llm_sharded_infer":
        failed_checks.append("read_only_workload_missing")
    for key in ["not_production", "not_p2p", "not_large_model_serving", "not_public_prompt_serving"]:
        if safety.get(key) is not True:
            failed_checks.append(f"{key}_missing")
    if cleanup.get("child_artifacts_pruned") is not True:
        failed_checks.append("child_artifacts_not_pruned")
    for artifact_name in ["local_requeue_json", "live_swarm_beta_json", "live_support_bundle_json"]:
        artifact = artifacts.get(artifact_name) if isinstance(artifacts.get(artifact_name), dict) else {}
        if artifact.get("present") not in {False, None}:
            failed_checks.append(f"{artifact_name}_still_present")
    if private_artifacts:
        failed_checks.append("private_artifacts_present")
    if summary.get("target_stage") != stage:
        failed_checks.append("target_stage_mismatch")
    for key in ["claim_observed", "victim_kernel_deleted", "rescued_result"]:
        if summary.get(key) is not True:
            failed_checks.append(f"requeue_{key}_missing")
    if not summary.get("lease_expired"):
        failed_checks.append("requeue_lease_timeout_missing")
    if summary.get("victim_result_accepted") is not False:
        failed_checks.append("victim_result_not_rejected")
    return {
        "stage": stage,
        "path": str(path),
        "present": path.is_file(),
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "failure_mode": payload.get("failure_mode"),
        "ready": not failed_checks,
        "missing_codes": missing_codes,
        "failed_checks": sorted(set(failed_checks)),
        "diagnosis_codes": sorted(codes),
        "live_requeue_summary": {
            "target_stage": summary.get("target_stage"),
            "claim_observed": summary.get("claim_observed"),
            "victim_kernel_deleted": summary.get("victim_kernel_deleted"),
            "lease_expired": bool(summary.get("lease_expired")),
            "rescued_result": summary.get("rescued_result"),
            "victim_result_accepted": summary.get("victim_result_accepted"),
        },
        "artifact_cleanup": {
            "child_artifacts_pruned": cleanup.get("child_artifacts_pruned"),
            "retained_public_artifacts": cleanup.get("retained_public_artifacts"),
        },
        "private_artifacts": private_artifacts,
    }


def validate_summary_report(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    failed: list[str] = []
    if not path.is_file():
        failed.append("summary_missing")
    if payload.get("schema") != SUMMARY_SCHEMA:
        failed.append("summary_schema_mismatch")
    if payload.get("ok") is not True:
        failed.append("summary_not_ok")
    proofs = payload.get("proofs") if isinstance(payload.get("proofs"), list) else []
    stages = {proof.get("target_stage") for proof in proofs if isinstance(proof, dict)}
    for stage in ["stage0", "stage1"]:
        if stage not in stages:
            failed.append(f"{stage}_summary_missing")
    for proof in proofs:
        if not isinstance(proof, dict):
            continue
        if proof.get("ok") is not True:
            failed.append(f"{proof.get('target_stage')}_summary_not_ok")
        for key in ["claim_observed", "victim_kernel_deleted", "rescued_result"]:
            if proof.get(key) is not True:
                failed.append(f"{proof.get('target_stage')}_{key}_missing")
        if proof.get("victim_result_accepted") is not False:
            failed.append(f"{proof.get('target_stage')}_victim_not_rejected")
    return {
        "path": str(path),
        "present": path.is_file(),
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": not failed,
        "failed_checks": sorted(set(failed)),
        "proof_count": len(proofs),
        "target_stages": sorted(stage for stage in stages if isinstance(stage, str)),
    }


def build_evidence_import_report(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    stage0_path = Path(args.stage0_report).resolve()
    stage1_path = Path(args.stage1_report).resolve()
    summary_path = Path(args.summary_report).resolve()
    stage0_payload = load_json_file(stage0_path)
    stage1_payload = load_json_file(stage1_path)
    summary_payload = load_json_file(summary_path)
    stage0 = validate_stage_report(stage0_path, stage0_payload, stage="stage0")
    stage1 = validate_stage_report(stage1_path, stage1_payload, stage="stage1")
    summary = validate_summary_report(summary_path, summary_payload)
    imported_codes = set(diagnosis_codes(stage0_payload, stage1_payload))
    codes = set(imported_codes)
    if stage0.get("ready"):
        codes.add("stage0_live_requeue_evidence_ready")
    if stage1.get("ready"):
        codes.add("stage1_live_requeue_evidence_ready")
    if stage0.get("ready") and stage1.get("ready"):
        codes.add("public_swarm_live_requeue_evidence_ready")
    if summary.get("ready"):
        codes.add("public_swarm_live_requeue_summary_ready")
    private_clear = not stage0.get("private_artifacts") and not stage1.get("private_artifacts")
    if private_clear:
        codes.add("public_swarm_alpha_private_artifacts_absent")
    ok = bool(stage0.get("ready") and stage1.get("ready") and summary.get("ready") and private_clear)
    if ok:
        codes.update({
            "public_swarm_alpha_rc_evidence_imported",
            "public_swarm_alpha_rc_safety_ready",
            "public_swarm_inference_alpha_rc_ready",
        })
    else:
        codes.add("public_swarm_inference_alpha_rc_blocked")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": MODE_EVIDENCE_IMPORT,
        "output_dir": str(output_dir),
        "release_candidate": {
            "ready": ok,
            "model_id": "sshleifer/tiny-gpt2",
            "workload_type": "real_llm_sharded_infer",
            "stage_count": 2,
            "evidence_imported": True,
            "stage0_live_requeue_ready": bool(stage0.get("ready")),
            "stage1_live_requeue_ready": bool(stage1.get("ready")),
            "summary_ready": bool(summary.get("ready")),
        },
        "imported_reports": {
            "stage0": stage0,
            "stage1": stage1,
            "summary": summary,
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "public_swarm_inference_alpha_rc_json": artifact_entry(
                output_dir / "public_swarm_inference_alpha_rc.json",
                output_dir,
                kind="public_swarm_inference_alpha_rc",
                schema=SCHEMA,
                ok=ok,
            ),
            "public_swarm_inference_alpha_rc_markdown": artifact_entry(
                output_dir / "public_swarm_inference_alpha_rc.md",
                output_dir,
                kind="public_swarm_inference_alpha_rc_markdown",
            ),
            "stage0_public_swarm_report": artifact_entry(
                stage0_path,
                output_dir,
                kind="public_swarm_inference_alpha_stage0",
                schema=ALPHA_SCHEMA,
                ok=stage0_payload.get("ok") if stage0_payload else None,
            ),
            "stage1_public_swarm_report": artifact_entry(
                stage1_path,
                output_dir,
                kind="public_swarm_inference_alpha_stage1",
                schema=ALPHA_SCHEMA,
                ok=stage1_payload.get("ok") if stage1_payload else None,
            ),
            "live_requeue_summary": artifact_entry(
                summary_path,
                output_dir,
                kind="public_swarm_inference_alpha_live_requeue_summary",
                schema=SUMMARY_SCHEMA,
                ok=summary_payload.get("ok") if summary_payload else None,
            ),
        },
        "safety": {
            "cpu_only": True,
            "read_only_workload": "real_llm_sharded_infer",
            "retained_public_artifacts_only": private_clear,
            "activation_payloads_redacted": True,
            "token_values_redacted": True,
            "child_debug_artifacts_pruned": private_clear,
            "not_production": True,
            "not_p2p": True,
            "not_gpu_tpu_pooling": True,
            "not_large_model_serving": True,
            "not_public_prompt_serving": True,
        },
        "operator_action": [
            "Use evidence-import as a release-candidate audit of retained public reports; it does not create a fresh Kaggle run.",
            "Rotate tokens after every temporary public HTTP/Kaggle run.",
        ],
        "limitations": [
            "Public Swarm Inference Alpha RC imports retained evidence; it is not a fresh live proof by itself.",
            "CPU-only read-only tiny GPT split evidence; not production Swarm Inference, P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, payments, or staking.",
        ],
    }


def run_json_step(name: str, command: list[str], *, runner: Runner, timeout_seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        completed = runner(
            command,
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "ok": False,
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "timeout",
        }, {}
    step: dict[str, Any] = {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    payload: dict[str, Any] = {}
    try:
        payload = json_from_stdout(completed.stdout)
    except ValueError as exc:
        step["ok"] = False
        step["error"] = str(exc)
    else:
        step["payload_schema"] = payload.get("schema")
        step["payload_ok"] = payload.get("ok")
        step["ok"] = bool(step["ok"] and payload.get("ok"))
    if not step["ok"]:
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1200:])
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1200:])
    return step, payload


def build_local_smoke_report(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    child_dir = output_dir / "local-smoke"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_inference_alpha_check.py"),
        "--output-dir",
        str(child_dir),
        "--mode",
        "live-kaggle",
        "--request-count",
        str(args.request_count),
        "--json",
    ]
    step, payload = run_json_step("public_swarm_inference_alpha_contract_check", command, runner=runner, timeout_seconds=args.timeout_seconds)
    codes = set(diagnosis_codes(payload))
    ready = bool(step.get("ok") and payload.get("schema") == CHECK_SCHEMA and "public_swarm_inference_alpha_check_ready" in codes)
    if ready:
        codes.update({"public_swarm_alpha_rc_local_smoke_ready", "public_swarm_alpha_rc_contract_ready"})
    else:
        codes.add("public_swarm_alpha_rc_local_smoke_blocked")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": MODE_LOCAL_SMOKE,
        "output_dir": str(output_dir),
        "release_candidate": {
            "ready": False,
            "evidence_imported": False,
            "local_smoke_ready": ready,
        },
        "steps": [step],
        "payload_summaries": {
            "public_swarm_inference_alpha_check": {
                "schema": payload.get("schema"),
                "ok": payload.get("ok"),
                "diagnosis_codes": diagnosis_codes(payload),
            },
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "public_swarm_inference_alpha_rc_json": artifact_entry(
                output_dir / "public_swarm_inference_alpha_rc.json",
                output_dir,
                kind="public_swarm_inference_alpha_rc",
                schema=SCHEMA,
                ok=ready,
            ),
            "public_swarm_inference_alpha_rc_markdown": artifact_entry(
                output_dir / "public_swarm_inference_alpha_rc.md",
                output_dir,
                kind="public_swarm_inference_alpha_rc_markdown",
            ),
        },
        "safety": {
            "cpu_only": True,
            "read_only_workload": "real_llm_sharded_infer",
            "not_production": True,
            "not_p2p": True,
            "not_large_model_serving": True,
        },
        "limitations": [
            "Local smoke validates the RC contract without importing retained live evidence.",
            "It does not create Kaggle resources and does not prove a fresh external runtime.",
        ],
    }


def persist_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report = support_bundle.sanitize(redact_values(report))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "Public Swarm Inference Alpha RC report contained secret-like fragments"
    json_path = output_dir / "public_swarm_inference_alpha_rc.json"
    md_path = output_dir / "public_swarm_inference_alpha_rc.md"
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    if "artifacts" in report:
        report["artifacts"]["public_swarm_inference_alpha_rc_json"]["present"] = True
        report["artifacts"]["public_swarm_inference_alpha_rc_markdown"]["present"] = True
        write_json(json_path, report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    rc = report.get("release_candidate") if isinstance(report.get("release_candidate"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    lines = [
        "# CrowdTensor Public Swarm Inference Alpha RC",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- ready: `{rc.get('ready')}`",
        f"- evidence_imported: `{rc.get('evidence_imported')}`",
        f"- model_id: `{rc.get('model_id')}`",
        f"- stage0_live_requeue_ready: `{rc.get('stage0_live_requeue_ready')}`",
        f"- stage1_live_requeue_ready: `{rc.get('stage1_live_requeue_ready')}`",
        "",
        "## Output Scope",
        "",
        f"- include output: `{output_request.get('include_output')}`",
        f"- answer scope: `{answer_scope.get('scope_state')}`",
        f"- saved JSON display: `{answer_scope.get('saved_json_display')}`",
        f"- saved Markdown display: `{answer_scope.get('saved_markdown_display')}`",
        f"- shareable: `saved_artifacts={shareable.get('saved_artifacts_public_safe')} raw_prompt_public={shareable.get('raw_prompt_public')} raw_generation_public={shareable.get('raw_generation_public')} generation_ids_public={shareable.get('generation_ids_public')} answer_scope_state={shareable.get('answer_scope_state')} local_answer_terminal_only={shareable.get('local_answer_terminal_only')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Artifacts",
        "",
    ]
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    lines.extend(["", "## Boundaries", ""])
    for limitation in report.get("limitations") or []:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == MODE_LOCAL_SMOKE:
        report = build_local_smoke_report(args, output_dir=output_dir, runner=runner)
    else:
        report = build_evidence_import_report(args, output_dir=output_dir)
    return persist_report(report, output_dir=output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Public Swarm Inference Alpha RC evidence.")
    parser.add_argument("--mode", choices=[MODE_EVIDENCE_IMPORT, MODE_LOCAL_SMOKE], default=MODE_EVIDENCE_IMPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage0-report", default=DEFAULT_STAGE0_REPORT)
    parser.add_argument("--stage1-report", default=DEFAULT_STAGE1_REPORT)
    parser.add_argument("--summary-report", default=DEFAULT_SUMMARY_REPORT)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.request_count < 1 or args.request_count > 4:
        raise SystemExit("--request-count must be between 1 and 4")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    return args


def print_human(report: dict[str, Any]) -> None:
    rc = report.get("release_candidate") if isinstance(report.get("release_candidate"), dict) else {}
    print("CrowdTensor Public Swarm Inference Alpha RC")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {rc.get('ready')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human(report)
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
