#!/usr/bin/env python3
"""Build the Public Swarm Product RC evidence artifact."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import gpu_sharded_generation_beta_pack as gpu_pack  # noqa: E402
from crowdtensor import cli  # noqa: E402
from crowdtensor.session_protocol import public_leak_paths  # noqa: E402


SCHEMA = "public_swarm_product_rc_v1"
Runner = Callable[..., subprocess.CompletedProcess[str]]
DEFAULT_GPU_REPORT = (
    "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
    "gpu_sharded_generation_beta_kaggle_auto.json"
)
SECRET_FRAGMENTS = {
    "CROWDTENSOR_ADMIN_TOKEN",
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "lease_token",
    "idempotency_key",
    "Bearer ",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
}
SAFE_PUBLIC_LEAK_PATH_PARTS = (
    ".output_request.",
    ".prompt_scope.",
    ".answer_scope.",
    ".shareable_summary.",
)


def shell_command(parts: list[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def command_entry(
    label: str,
    command: list[Any],
    *,
    reason: str = "",
    requires_private_credentials: bool = False,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "label": label,
        "command": [str(part) for part in command],
        "command_line": shell_command(command),
        "public_artifact_safe": True,
    }
    if reason:
        entry["reason"] = reason
    if requires_private_credentials:
        entry["requires_private_credentials"] = True
        entry["credential_note"] = (
            "Use local private env/runbook values when running this command; "
            "credential values are intentionally excluded from public artifacts."
        )
    return entry


def artifact_command(output_dir: Path, filename: str, *, lines: str = "1,220p") -> list[str]:
    return ["sed", "-n", lines, str(output_dir / filename)]


def artifact_summary(output_dir: Path) -> dict[str, Any]:
    paths = {
        "inspect_first": output_dir / "public_swarm_product_rc.md",
        "summary_json": output_dir / "public_swarm_product_rc.json",
        "summary_markdown": output_dir / "public_swarm_product_rc.md",
        "support_bundle": output_dir / "support_bundle.json",
    }
    present = sum(1 for path in paths.values() if path.is_file())
    return {
        **{name: str(path) for name, path in paths.items()},
        "artifact_count": len(paths),
        "present_artifact_count": present,
        "shareable_paths": [
            str(paths["summary_json"]),
            str(paths["summary_markdown"]),
            str(paths["support_bundle"]),
        ],
        "public_artifact_safe": True,
    }


def product_rc_command(args: argparse.Namespace, output_dir: Path) -> list[Any]:
    command: list[Any] = [
        "crowdtensor",
        "public-swarm-product-rc",
        "--output-dir",
        str(output_dir),
        "--gpu-report",
        str(args.gpu_report),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    command.extend(["--prompt-text", "PROMPT_TEXT"])
    return command


def not_completed_items(report: dict[str, Any]) -> list[str]:
    product_surface = report.get("product_surface") if isinstance(report.get("product_surface"), dict) else {}
    session_protocol = report.get("session_protocol") if isinstance(report.get("session_protocol"), dict) else {}
    p2p_lite = report.get("p2p_lite") if isinstance(report.get("p2p_lite"), dict) else {}
    gpu_import = report.get("gpu_generation_import") if isinstance(report.get("gpu_generation_import"), dict) else {}
    items = [
        ("Coordinator product surface ready", report.get("product_surface_ready")),
        ("serve command ready", (product_surface.get("serve") or {}).get("ok") if isinstance(product_surface.get("serve"), dict) else None),
        ("stage0 join command ready", (product_surface.get("join_stage0") or {}).get("ok") if isinstance(product_surface.get("join_stage0"), dict) else None),
        ("stage1 join command ready", (product_surface.get("join_stage1") or {}).get("ok") if isinstance(product_surface.get("join_stage1"), dict) else None),
        ("generate command ready", (product_surface.get("generate") or {}).get("ok") if isinstance(product_surface.get("generate"), dict) else None),
        ("session protocol ready", session_protocol.get("ok")),
        ("P2P-lite discovery ready", p2p_lite.get("ok")),
        ("GPU generation evidence import ready", gpu_import.get("ok")),
    ]
    return [label for label, ready in items if ready is not True]


def recommended_next_command(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    not_completed: list[str],
) -> dict[str, Any]:
    if report.get("ok"):
        return command_entry(
            "inspect Product RC evidence",
            artifact_command(output_dir, "public_swarm_product_rc.md"),
            reason="review_artifacts",
        )
    if "GPU generation evidence import ready" in not_completed:
        return command_entry(
            "import retained GPU generation evidence",
            product_rc_command(args, output_dir),
            reason="provide_gpu_generation_evidence",
        )
    return command_entry(
        "rerun Product RC proof",
        product_rc_command(args, output_dir),
        reason="fix_product_rc_blockers" if not_completed else "rerun_product_rc",
    )


def next_commands(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    recommended: dict[str, Any],
) -> list[dict[str, Any]]:
    commands = [
        command_entry(
            "inspect shareable summary",
            artifact_command(output_dir, "public_swarm_product_rc.md"),
            reason="review_artifacts",
        ),
        command_entry(
            "inspect support bundle",
            artifact_command(output_dir, "support_bundle.json", lines="1,220p"),
            reason="inspect_diagnostics",
        ),
    ]
    if report.get("ok"):
        commands.append(command_entry(
            "refresh Product RC proof",
            product_rc_command(args, output_dir),
            reason="refresh_product_rc",
        ))
    else:
        commands.append(dict(recommended))
    return commands


def user_status(
    *,
    ready: bool,
    recommended: dict[str, Any],
    not_completed: list[str],
) -> dict[str, Any]:
    return {
        "state": "ready" if ready else "blocked",
        "headline": (
            "Public Swarm Product RC evidence is ready."
            if ready
            else "Public Swarm Product RC evidence needs attention."
        ),
        "next_step": "review_artifacts" if ready else "fix_blockers",
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "not_completed_count": len(not_completed),
        "public_artifact_safe": True,
    }


def review_summary(
    report: dict[str, Any],
    *,
    output_dir: Path,
    recommended: dict[str, Any],
    not_completed: list[str],
) -> dict[str, Any]:
    codes = [str(code) for code in (report.get("diagnosis_codes") or [])]
    ready = bool(report.get("ok"))
    return {
        "schema": "public_swarm_product_rc_review_summary_v1",
        "state": "ready" if ready else "blocked",
        "headline": (
            "Public Swarm Product RC evidence is ready."
            if ready
            else "Public Swarm Product RC evidence needs attention."
        ),
        "next_step": "review_artifacts" if ready else "fix_blockers",
        "inspect_first": str(output_dir / "public_swarm_product_rc.md"),
        "support_bundle": str(output_dir / "support_bundle.json"),
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "next_command": recommended.get("command_line") or "",
        "primary_code": "public_swarm_product_rc_ready" if ready else (codes[0] if codes else "public_swarm_product_rc_blocked"),
        "attention": "none" if ready else (not_completed[0] if not_completed else "public_swarm_product_rc_blocked"),
        "attention_detail": "; ".join(not_completed[:5]),
        "not_completed_count": len(not_completed),
        "public_artifact_safe": True,
    }


def attach_user_guidance(report: dict[str, Any], args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    missing = not_completed_items(report)
    recommended = recommended_next_command(report, args, output_dir=output_dir, not_completed=missing)
    report["not_completed"] = missing
    report["recommended_next_command"] = recommended
    report["next_commands"] = next_commands(report, args, output_dir=output_dir, recommended=recommended)
    report["user_status"] = user_status(
        ready=bool(report.get("ok")),
        recommended=recommended,
        not_completed=missing,
    )
    report["review_summary"] = review_summary(
        report,
        output_dir=output_dir,
        recommended=recommended,
        not_completed=missing,
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    return report


def support_bundle_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "public_swarm_product_rc_support_bundle_v1",
        "ok": report.get("ok"),
        "output_dir": report.get("output_dir"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "review_summary": report.get("review_summary"),
        "user_status": report.get("user_status"),
        "recommended_next_command": report.get("recommended_next_command"),
        "next_commands": report.get("next_commands"),
        "artifact_summary": report.get("artifact_summary"),
        "not_completed": report.get("not_completed"),
        "product_surface_ready": report.get("product_surface_ready"),
        "session_protocol": report.get("session_protocol"),
        "p2p_lite": report.get("p2p_lite"),
        "gpu_generation_import": report.get("gpu_generation_import"),
        "safety": report.get("safety"),
        "limitations": report.get("limitations"),
        "public_artifact_safe": True,
    }


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Public Swarm Product RC artifacts summarize Coordinator product "
            "surface readiness, session protocol status, P2P-lite discovery, "
            "GPU generation import hashes/counts, and diagnostics only. They "
            "do not include answer text."
        ),
    }


def prompt_scope_summary() -> dict[str, Any]:
    return {
        "source": "prompt-text",
        "prompt_count": 1,
        "inline_prompt_text": True,
        "terminal_next_commands_local_private": True,
        "terminal_logs_local_private": True,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": True,
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This Product RC artifact records prompt source/count and placeholder "
            "safety only; raw prompt text is excluded from public JSON and Markdown."
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
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This Product RC report is shareable operator evidence, not a "
            "local answer transcript; raw prompts, generated text, token ids, "
            "activations, leases, credentials, and private runtime state are "
            "excluded."
        ),
    }


def shareable_summary() -> dict[str, Any]:
    return {
        "saved_artifacts_public_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "answer_scope_state": "no-local-answer",
        "local_answer_terminal_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Share public_swarm_product_rc*.json/md artifacts; they contain "
            "readiness evidence, route status, hashes, counts, and diagnostics, "
            "not raw prompts or answers."
        ),
    }


def run_cli_builds(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    serve_args = cli.parse_args([
        "serve",
        "--profile",
        "gpu-generation",
        "--bind-host",
        "127.0.0.1",
        "--port",
        "8787",
        "--json",
    ])
    join_stage0_args = cli.parse_args([
        "join",
        "--coordinator-url",
        "http://127.0.0.1:8787",
        "--miner-id",
        "product-stage0",
        "--stage",
        "stage0",
        "--backend",
        "cuda",
        "--json",
    ])
    join_stage1_args = cli.parse_args([
        "join",
        "--coordinator-url",
        "http://127.0.0.1:8787",
        "--miner-id",
        "product-stage1",
        "--stage",
        "stage1",
        "--backend",
        "cuda",
        "--json",
    ])
    generate_args = cli.parse_args([
        "generate",
        "--coordinator-url",
        "http://127.0.0.1:8787",
        "--prompt-text",
        args.prompt_text,
        "--backend",
        "cuda",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dry-run",
        "--skip-live-preflight",
        "--json",
    ])
    peer_args = cli.parse_args(["peer", "check", "--json"])
    return {
        "serve": cli.build_product_serve(serve_args),
        "join_stage0": cli.build_product_join(join_stage0_args),
        "join_stage1": cli.build_product_join(join_stage1_args),
        "generate": cli.build_product_generate(generate_args),
        "peer_check": cli.build_peer_cli(peer_args),
    }


def import_gpu_generation(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    source = Path(args.gpu_report).resolve()
    if not source.is_file():
        return {
            "schema": "gpu_sharded_generation_beta_v1",
            "ok": False,
            "source_path": str(source),
            "diagnosis_codes": ["gpu_generation_evidence_missing"],
        }
    return gpu_pack.build_report(gpu_pack.parse_args([
        "evidence-import",
        "--output-dir",
        str(output_dir / "gpu-generation-import"),
        "--gpu-report",
        str(source),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]))


def validate_public_artifact(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    encoded = json.dumps(report, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    for path in public_leak_paths(report):
        # The artifact stores schema names and safety field names containing
        # prompt/generated words, but it must not contain raw payload keys.
        if (
            path.endswith(".session_request.prompt_hash")
            or ".safety." in path
            or any(part in path for part in SAFE_PUBLIC_LEAK_PATH_PARTS)
        ):
            continue
        errors.append(f"public_leak:{path}")
    return sorted(set(errors))


def render_markdown(report: dict[str, Any]) -> str:
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    user = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    artifact_report = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    next_items = report.get("next_commands") if isinstance(report.get("next_commands"), list) else []
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    lines = [
        "# CrowdTensor Public Swarm Product RC",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- output_dir: `{report.get('output_dir')}`",
        "",
        "## Review",
        "",
        f"- state: `{review.get('state')}`",
        f"- status: `{user.get('headline')}`",
        f"- next step: `{review.get('next_step')}`",
        f"- inspect first: `{review.get('inspect_first')}`",
        f"- recommended: `{recommended.get('label')}` reason=`{recommended.get('reason')}`",
        f"- recommended command: `{recommended.get('command_line')}`",
        f"- not completed count: `{review.get('not_completed_count')}`",
        "",
        "## What To Do Next",
        "",
    ]
    if next_items:
        lines.extend(
            (
                f"- {item.get('label')}: `{item.get('command_line')}`"
                + (" (requires private credentials; see runbook)" if item.get("requires_private_credentials") else "")
            )
            for item in next_items
            if isinstance(item, dict)
        )
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Output Scope",
        "",
        f"- include output: `{output_request.get('include_output')}`",
        f"- output request note: {output_request.get('summary') or 'Public artifacts summarize inference evidence only and do not include answer text.'}",
        f"- prompt scope: `source={prompt_scope.get('source')} count={prompt_scope.get('prompt_count')} inline_prompt_text={prompt_scope.get('inline_prompt_text')} terminal_next_commands_local_private={prompt_scope.get('terminal_next_commands_local_private')} saved_artifacts_prompt_placeholders={prompt_scope.get('saved_artifacts_prompt_placeholders')} prompt_file_path_public={prompt_scope.get('prompt_file_path_public')} raw_prompt_public={prompt_scope.get('raw_prompt_public')} public_artifact_safe={prompt_scope.get('public_artifact_safe')}`",
        f"- answer scope: `{answer_scope.get('scope_state')}`",
        f"- answer scope note: {answer_scope.get('summary') or 'Public artifacts contain no local answer transcript or raw generated text.'}",
        f"- saved JSON display: `{answer_scope.get('saved_json_display')}`",
        f"- saved Markdown display: `{answer_scope.get('saved_markdown_display')}`",
        f"- shareable: `saved_artifacts={shareable.get('saved_artifacts_public_safe')} raw_prompt_public={shareable.get('raw_prompt_public')} raw_generated_text_public={shareable.get('raw_generated_text_public')} generated_token_ids_public={shareable.get('generated_token_ids_public')} local_output_display_only={shareable.get('local_output_display_only')} answer_scope_state={shareable.get('answer_scope_state')} local_answer_terminal_only={shareable.get('local_answer_terminal_only')}`",
        "",
        "## Artifact Summary",
        "",
        f"- inspect first: `{artifact_report.get('inspect_first')}`",
        f"- summary JSON: `{artifact_report.get('summary_json')}`",
        f"- summary Markdown: `{artifact_report.get('summary_markdown')}`",
        f"- support bundle: `{artifact_report.get('support_bundle')}`",
        f"- present: `{artifact_report.get('present_artifact_count')}` / `{artifact_report.get('artifact_count')}`",
        f"- public artifact safe: `{artifact_report.get('public_artifact_safe')}`",
        "",
        "## Readiness",
        "",
        f"- product surface ready: `{report.get('product_surface_ready')}`",
        f"- session protocol ready: `{(report.get('session_protocol') or {}).get('ok')}`",
        f"- P2P-lite ready: `{(report.get('p2p_lite') or {}).get('ok')}`",
        f"- GPU generation import ready: `{(report.get('gpu_generation_import') or {}).get('ok')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Not Completed",
        "",
    ])
    not_completed = report.get("not_completed") or []
    lines.extend(f"- {item}" for item in not_completed) if not_completed else lines.append("- none")
    lines.append("")
    if report.get("artifacts"):
        lines.extend(["## Artifacts", ""])
        for name, artifact in sorted((report.get("artifacts") or {}).items()):
            if isinstance(artifact, dict):
                lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
            else:
                lines.append(f"- `{name}`: `{artifact}`")
        lines.append("")
    lines.extend(["## Boundaries", ""])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def persist_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault("prompt_scope", prompt_scope_summary())
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    if "artifact_summary" not in report:
        report["artifact_summary"] = artifact_summary(output_dir)
    report.setdefault("artifacts", {})
    json_path = output_dir / "public_swarm_product_rc.json"
    markdown_path = output_dir / "public_swarm_product_rc.md"
    support_path = output_dir / "support_bundle.json"
    report["artifacts"]["public_swarm_product_rc_json"] = artifact_entry(
        json_path,
        output_dir,
        kind="public_swarm_product_rc",
        schema=SCHEMA,
        ok=report.get("ok"),
    )
    report["artifacts"]["public_swarm_product_rc_markdown"] = artifact_entry(
        markdown_path,
        output_dir,
        kind="public_swarm_product_rc_markdown",
        ok=report.get("ok"),
    )
    report["artifacts"]["gpu_generation_import"] = artifact_entry(
        output_dir / "gpu-generation-import" / "gpu_sharded_generation_beta_evidence_import.json",
        output_dir,
        kind="gpu_sharded_generation_beta_evidence_import",
        schema="gpu_sharded_generation_beta_v1",
    )
    report["artifacts"]["support_bundle_json"] = artifact_entry(
        support_path,
        output_dir,
        kind="support_bundle",
        schema="public_swarm_product_rc_support_bundle_v1",
        ok=report.get("ok"),
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    if isinstance(report.get("review_summary"), dict):
        report["review_summary"]["inspect_first"] = report["artifact_summary"]["inspect_first"]
        report["review_summary"]["support_bundle"] = report["artifact_summary"]["support_bundle"]
    safety_errors = validate_public_artifact(report)
    if safety_errors:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report["diagnosis_codes"]) | {"public_artifact_safety_failed"})
        report["safety_errors"] = safety_errors
        for artifact in report["artifacts"].values():
            if isinstance(artifact, dict) and "ok" in artifact:
                artifact["ok"] = False
    write_json(json_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    write_json(support_path, support_bundle_payload(report))
    report["artifacts"]["public_swarm_product_rc_json"]["present"] = True
    report["artifacts"]["public_swarm_product_rc_markdown"]["present"] = True
    report["artifacts"]["support_bundle_json"]["present"] = True
    report["artifact_summary"] = artifact_summary(output_dir)
    if isinstance(report.get("review_summary"), dict):
        report["review_summary"]["inspect_first"] = report["artifact_summary"]["inspect_first"]
        report["review_summary"]["support_bundle"] = report["artifact_summary"]["support_bundle"]
    write_json(json_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    write_json(support_path, support_bundle_payload(report))
    return report


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="crowdtensor_public_swarm_product_rc_")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cli_reports = run_cli_builds(args, output_dir)
    session_step, session_payload = cli.run_json_step(
        "session_protocol_check",
        [sys.executable, str(ROOT / "scripts" / "session_protocol_check.py"), "--json"],
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    p2p_step, p2p_payload = cli.run_json_step(
        "p2p_lite_discovery_check",
        [sys.executable, str(ROOT / "scripts" / "p2p_lite_discovery_check.py"), "--json"],
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    gpu_report = import_gpu_generation(args, output_dir)
    product_surface_ready = all(
        cli_reports[name].get("ok")
        for name in ["serve", "join_stage0", "join_stage1", "generate"]
    )
    ok = bool(product_surface_ready and session_payload.get("ok") and p2p_payload.get("ok") and gpu_report.get("ok"))
    diagnosis_codes = {
        "public_swarm_product_rc_ready" if ok else "public_swarm_product_rc_blocked",
        "coordinator_product_surface_ready" if product_surface_ready else "coordinator_product_surface_blocked",
        "session_protocol_ready" if session_payload.get("ok") else "session_protocol_blocked",
        "p2p_lite_discovery_ready" if p2p_payload.get("ok") else "p2p_lite_discovery_blocked",
        "gpu_generation_evidence_import_ready" if gpu_report.get("ok") else "gpu_generation_evidence_import_blocked",
    }
    generation = gpu_report.get("generation") if isinstance(gpu_report.get("generation"), dict) else {}
    report = {
        "schema": SCHEMA,
        "ok": ok,
        "output_dir": str(output_dir),
        "product_surface_ready": product_surface_ready,
        "product_surface": {
            name: {
                "ok": payload.get("ok"),
                "mode": payload.get("mode"),
                "diagnosis_codes": payload.get("diagnosis_codes"),
                "command": payload.get("command"),
                "dry_run": payload.get("dry_run"),
            }
            for name, payload in cli_reports.items()
        },
        "session_protocol": {
            "ok": session_payload.get("ok"),
            "schema": session_payload.get("schema"),
            "route_usable": (session_payload.get("route") or {}).get("usable_now"),
            "step": session_step,
        },
        "p2p_lite": {
            "ok": p2p_payload.get("ok"),
            "schema": p2p_payload.get("schema"),
            "cpu_route_ok": (p2p_payload.get("cpu_route") or {}).get("ok"),
            "cuda_route_ok": (p2p_payload.get("cuda_route") or {}).get("ok"),
            "step": p2p_step,
        },
        "gpu_generation_import": {
            "ok": gpu_report.get("ok"),
            "schema": gpu_report.get("schema"),
            "mode": gpu_report.get("mode"),
            "generated_token_count": generation.get("generated_token_count"),
            "generated_text_hash": generation.get("generated_text_hash"),
            "raw_generated_text_public": False,
        },
        "diagnosis_codes": sorted(diagnosis_codes),
        "prompt_scope": prompt_scope_summary(),
        "safety": {
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "tokens_public": False,
            "not_production": True,
            "not_libp2p": True,
            "not_dht": True,
            "not_nat_traversal": True,
            "coordinator_remains_execution_authority": True,
        },
        "limitations": [
            "Product RC exposes Coordinator-backed serve/join/generate ergonomics; it is not Hivemind/Petals-level production serving.",
            "P2P-lite discovers routes over HTTP gossip only; it is not libp2p, DHT, NAT traversal, or decentralized security.",
            "GPU generation evidence is imported from retained tiny GPT Kaggle proof; no large-model serving or public arbitrary prompt API is claimed.",
        ],
        "artifacts": {},
    }
    report = attach_user_guidance(report, args, output_dir=output_dir)
    return persist_report(report, output_dir=output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Public Swarm Product RC artifact.")
    parser.add_argument("--output-dir", default="dist/public-swarm-product-rc")
    parser.add_argument("--gpu-report", default=DEFAULT_GPU_REPORT)
    parser.add_argument("--prompt-text", default="CrowdTensor product RC")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
        answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
        print(f"Public Swarm Product RC ready: {report.get('ok')}")
        if output_request:
            print(f"  output_request: include_output={bool(output_request.get('include_output'))} raw_generated_text_public={bool(output_request.get('raw_generated_text_public'))} public_artifact_safe={bool(output_request.get('public_artifact_safe'))}")
            print(f"  output_request_note: {output_request.get('summary') or 'Public artifacts summarize inference evidence only and do not include answer text.'}")
        if answer_scope:
            print(f"  answer_scope: {answer_scope.get('scope_state')}")
            print(f"  answer_scope_note: {answer_scope.get('summary') or 'Public artifacts contain no local answer transcript or raw generated text.'}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
