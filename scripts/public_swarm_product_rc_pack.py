#!/usr/bin/env python3
"""Build the Public Swarm Product RC evidence artifact."""

from __future__ import annotations

import argparse
import json
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
    ".answer_scope.",
    ".shareable_summary.",
)


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


def run_cli_builds(output_dir: Path) -> dict[str, Any]:
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
        "CrowdTensor product RC",
        "--backend",
        "cuda",
        "--max-new-tokens",
        "8",
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
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    lines = [
        "# CrowdTensor Public Swarm Product RC",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- output_dir: `{report.get('output_dir')}`",
        "",
        "## Output Scope",
        "",
        f"- include output: `{output_request.get('include_output')}`",
        f"- answer scope: `{answer_scope.get('scope_state')}`",
        f"- saved JSON display: `{answer_scope.get('saved_json_display')}`",
        f"- saved Markdown display: `{answer_scope.get('saved_markdown_display')}`",
        f"- shareable: `saved_artifacts={shareable.get('saved_artifacts_public_safe')} raw_prompt_public={shareable.get('raw_prompt_public')} raw_generated_text_public={shareable.get('raw_generated_text_public')} generated_token_ids_public={shareable.get('generated_token_ids_public')} local_output_display_only={shareable.get('local_output_display_only')} answer_scope_state={shareable.get('answer_scope_state')} local_answer_terminal_only={shareable.get('local_answer_terminal_only')}`",
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
    ]
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
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report.setdefault("artifacts", {})
    json_path = output_dir / "public_swarm_product_rc.json"
    markdown_path = output_dir / "public_swarm_product_rc.md"
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
    report["artifacts"]["public_swarm_product_rc_json"]["present"] = True
    report["artifacts"]["public_swarm_product_rc_markdown"]["present"] = True
    write_json(json_path, report)
    return report


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="crowdtensor_public_swarm_product_rc_")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cli_reports = run_cli_builds(output_dir)
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
    return persist_report(report, output_dir=output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Public Swarm Product RC artifact.")
    parser.add_argument("--output-dir", default="dist/public-swarm-product-rc")
    parser.add_argument("--gpu-report", default=DEFAULT_GPU_REPORT)
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
        print(f"Public Swarm Product RC ready: {report.get('ok')}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
