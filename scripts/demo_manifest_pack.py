#!/usr/bin/env python3
"""Build a single local-loopback CrowdTensor demo manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in [ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import remote_compute_evidence_pack  # noqa: E402
import runtime_matrix  # noqa: E402
import support_bundle  # noqa: E402


MANIFEST_SCHEMA = "demo_manifest_v1"
DEFAULT_MINER_ID = "demo-manifest-miner"
DEFAULT_OBSERVER_TOKEN = "demo-manifest-observer"
DEFAULT_ADMIN_TOKEN = "demo-manifest-admin"
DEFAULT_INVITE_TOKEN = "demo-manifest-token"

SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN",
    "lease_token",
    "idempotency_key",
    "inference_results",
    "Bearer ",
    DEFAULT_INVITE_TOKEN,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relative_artifact_path(path: Path, output_dir: Path) -> str:
    return path.resolve().relative_to(output_dir.resolve()).as_posix()


def artifact_entry(
    *,
    output_dir: Path,
    path: Path,
    kind: str,
    schema: str = "",
    ok: bool | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "kind": kind,
        "path": relative_artifact_path(path, output_dir),
        "present": path.is_file(),
    }
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(payload), encoding="utf-8")


def runtime_summary(matrix: dict[str, Any]) -> dict[str, Any]:
    profile = matrix.get("host_profile") if isinstance(matrix.get("host_profile"), dict) else {}
    return {
        "ok": bool(matrix.get("ok")),
        "host": {
            "python": profile.get("python"),
            "os": profile.get("os"),
            "machine": profile.get("machine"),
            "cpu_count": profile.get("cpu_count"),
        },
        "available_workloads": list((matrix.get("summary") or {}).get("available_workloads") or []),
        "blocked_workloads": list((matrix.get("summary") or {}).get("blocked_workloads") or []),
        "hardware_targets": [
            {
                "name": target.get("name"),
                "status": target.get("status"),
                "usable_now": target.get("usable_now"),
                "operator_action": target.get("operator_action"),
                "diagnosis_codes": list(target.get("diagnosis_codes") or []),
            }
            for target in matrix.get("hardware_targets") or []
            if isinstance(target, dict)
        ],
        "recommended_routes": [
            {
                "name": route.get("name"),
                "target": route.get("target"),
                "workload": route.get("workload"),
                "status": route.get("status"),
                "usable_now": route.get("usable_now"),
                "confidence": route.get("confidence"),
                "operator_action": route.get("operator_action"),
                "diagnosis_codes": list(route.get("diagnosis_codes") or []),
            }
            for route in matrix.get("recommended_routes") or []
            if isinstance(route, dict)
        ],
        "diagnosis_summary": matrix.get("diagnosis_summary") or {},
        "hardware_diagnosis_summary": matrix.get("hardware_diagnosis_summary") or {},
    }


def remote_evidence_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    route = evidence.get("route_decision") if isinstance(evidence.get("route_decision"), dict) else {}
    summary = evidence.get("inference_summary") if isinstance(evidence.get("inference_summary"), dict) else {}
    miner = evidence.get("miner") if isinstance(evidence.get("miner"), dict) else {}
    profile = miner.get("profile") if isinstance(miner.get("profile"), dict) else {}
    safety = evidence.get("safety") if isinstance(evidence.get("safety"), dict) else {}
    observability = support_bundle.safe_observability_summary(
        evidence.get("observability_summary") if isinstance(evidence.get("observability_summary"), dict) else {}
    )
    return {
        "ok": bool(evidence.get("ok")),
        "schema": evidence.get("schema"),
        "mode": evidence.get("mode"),
        "route": {
            "name": route.get("name"),
            "target": route.get("target"),
            "workload": route.get("workload"),
            "confidence": route.get("confidence"),
            "usable_now": route.get("usable_now"),
            "matched_capabilities": list(route.get("matched_capabilities") or []),
            "missing_capabilities": list(route.get("missing_capabilities") or []),
        },
        "miner": {
            "miner_id": miner.get("miner_id"),
            "runtime": profile.get("runtime"),
            "backend": profile.get("backend"),
            "accepted": profile.get("accepted"),
            "rejected": profile.get("rejected"),
        },
        "inference": {
            "ok": summary.get("ok"),
            "workload_type": summary.get("workload_type"),
            "request_count": summary.get("request_count"),
            "expected_request_count": summary.get("expected_request_count"),
            "request_trace_count": summary.get("request_trace_count"),
            "accuracy": summary.get("accuracy"),
            "elapsed_ms": summary.get("elapsed_ms"),
            "requests_per_second": summary.get("requests_per_second"),
        },
        "observability": observability,
        "safety": {
            "read_only": safety.get("read_only"),
            "redaction_ok": safety.get("redaction_ok"),
            "registry_hashed": safety.get("registry_hashed"),
            "raw_payloads_exposed": safety.get("raw_payloads_exposed"),
        },
    }


def support_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    reports = bundle.get("reports") if isinstance(bundle.get("reports"), dict) else {}
    remote = reports.get("remote") if isinstance(reports.get("remote"), dict) else {}
    return {
        "doctor_ok": (bundle.get("doctor") or {}).get("ok"),
        "release_gate_ok": (bundle.get("release_gate") or {}).get("ok"),
        "online_enabled": (bundle.get("online") or {}).get("enabled"),
        "remote_report": {
            "present": remote.get("present"),
            "ok": remote.get("ok"),
            "schema": "remote_compute_evidence_v1" if remote.get("present") else "",
            "diagnosis_codes": list(remote.get("diagnosis_codes") or []),
            "observability_summaries": list(remote.get("observability_summaries") or []),
        },
    }


def secret_leak_fragments(payload: dict[str, Any]) -> list[str]:
    encoded = json.dumps(payload, sort_keys=True)
    return [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]


def build_manifest(
    *,
    output_dir: Path,
    mode: str,
    request_count: int,
    runtime: dict[str, Any],
    remote_evidence: dict[str, Any],
    support: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    generated_at: str | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_at": generated_at or utc_now(),
        "mode": mode,
        "ok": bool(runtime.get("ok") and remote_evidence.get("ok") and (support.get("release_gate") or {}).get("ok")),
        "output_dir_name": output_dir.name,
        "artifacts": artifacts,
        "summaries": {
            "runtime_matrix": runtime_summary(runtime),
            "remote_compute_evidence": remote_evidence_summary(remote_evidence),
            "support_bundle": support_summary(support),
        },
        "recommended_next_commands": [
            "python3 scripts/demo_manifest_check.py --base-port 8914",
            "python3 scripts/remote_demo_runbook_pack.py --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id remote-linux-1 --output-dir dist/remote-demo",
            "python3 scripts/remote_demo_acceptance_pack.py --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id remote-linux-1 --observer-token \"$CROWDTENSOR_OBSERVER_TOKEN\" --admin-token \"$CROWDTENSOR_ADMIN_TOKEN\" --output-dir dist/remote-demo-acceptance",
        ],
        "limitations": [
            "Local-loopback CPU-only demo manifest; not production Swarm Inference",
            "Controlled registry-backed Python Miner path; not public-internet hardening",
            "No GPU pooling, WebGPU model shards, libp2p discovery, NAT traversal, or incentives are claimed",
        ],
    }
    leaks = secret_leak_fragments(manifest)
    if leaks:
        manifest["ok"] = False
        manifest["safety_error"] = f"manifest leaked secret-like fragments: {', '.join(leaks)}"
    return support_bundle.sanitize(manifest)


def render_markdown(payload: dict[str, Any]) -> str:
    runtime = (payload.get("summaries") or {}).get("runtime_matrix") or {}
    remote = (payload.get("summaries") or {}).get("remote_compute_evidence") or {}
    support = (payload.get("summaries") or {}).get("support_bundle") or {}
    route = remote.get("route") or {}
    inference = remote.get("inference") or {}
    lines = [
        "# CrowdTensor Demo Manifest",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"Schema: `{payload.get('schema', '')}`",
        f"Mode: `{payload.get('mode', '')}`",
        f"OK: `{payload.get('ok')}`",
        "",
        "## Runtime Matrix",
        "",
        f"- OK: `{runtime.get('ok')}`",
        f"- Host: `{(runtime.get('host') or {}).get('os')}` `{(runtime.get('host') or {}).get('machine')}`",
        f"- Available workloads: `{', '.join(runtime.get('available_workloads') or [])}`",
        "",
        "## Remote Compute Evidence",
        "",
        f"- OK: `{remote.get('ok')}`",
        f"- Route: `{route.get('name')}`",
        f"- Workload: `{route.get('workload')}`",
        f"- Requests: `{inference.get('request_count')}`",
        f"- Requests/sec: `{inference.get('requests_per_second')}`",
        f"- Observability: `{(remote.get('observability') or {}).get('schema')}`",
        "",
        "## Support Bundle",
        "",
        f"- Doctor: `{support.get('doctor_ok')}`",
        f"- Release gate: `{support.get('release_gate_ok')}`",
        f"- Remote report: `{(support.get('remote_report') or {}).get('ok')}`",
        "",
        "## Artifacts",
        "",
    ]
    for name, artifact in (payload.get("artifacts") or {}).items():
        state = "present" if artifact.get("present") else "missing"
        lines.append(f"- `{name}`: `{state}` `{artifact.get('path', '')}`")
    lines.extend(["", "## Limitations", ""])
    for limitation in payload.get("limitations") or []:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def build_pack(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_payload = runtime_matrix.build_matrix(root=ROOT)
    runtime_json = output_dir / "runtime_matrix.json"
    write_json(runtime_payload, runtime_json)

    remote_args = remote_compute_evidence_pack.parse_args([
        "--mode",
        "local-loopback",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--request-count",
        str(args.request_count),
        "--miner-id",
        args.miner_id,
        "--observer-token",
        args.observer_token,
        "--admin-token",
        args.admin_token,
        "--invite-token",
        args.invite_token,
    ] + (["--state-dir", args.state_dir] if args.state_dir else []))
    remote_payload = remote_compute_evidence_pack.run_local_loopback(remote_args)
    remote_json = output_dir / "remote_compute_evidence.json"
    remote_md = output_dir / "remote_compute_evidence.md"
    remote_compute_evidence_pack.write_json(remote_payload, str(remote_json))
    remote_compute_evidence_pack.write_markdown(remote_payload, str(remote_md))

    support_args = support_bundle.parse_args([
        "--root",
        str(ROOT),
        "--state-dir",
        str(output_dir / "support-state"),
        "--port",
        "0",
        "--remote-report",
        str(remote_json),
    ])
    support_payload = support_bundle.build_bundle(support_args)
    support_json = output_dir / "support_bundle.json"
    support_md = output_dir / "support_bundle.md"
    support_bundle.write_json(support_payload, str(support_json))
    support_bundle.write_markdown(support_payload, str(support_md))

    manifest_json = output_dir / "demo_manifest.json"
    manifest_md = output_dir / "demo_manifest.md"
    artifacts = {
        "runtime_matrix": artifact_entry(
            output_dir=output_dir,
            path=runtime_json,
            kind="runtime_matrix",
            ok=runtime_payload.get("ok"),
        ),
        "remote_compute_evidence_json": artifact_entry(
            output_dir=output_dir,
            path=remote_json,
            kind="remote_compute_evidence",
            schema=str(remote_payload.get("schema") or ""),
            ok=remote_payload.get("ok"),
        ),
        "remote_compute_evidence_markdown": artifact_entry(
            output_dir=output_dir,
            path=remote_md,
            kind="remote_compute_evidence_markdown",
        ),
        "support_bundle_json": artifact_entry(
            output_dir=output_dir,
            path=support_json,
            kind="support_bundle",
            ok=(support_payload.get("release_gate") or {}).get("ok"),
        ),
        "support_bundle_markdown": artifact_entry(
            output_dir=output_dir,
            path=support_md,
            kind="support_bundle_markdown",
        ),
        "demo_manifest_markdown": {
            "kind": "demo_manifest_markdown",
            "path": relative_artifact_path(manifest_md, output_dir),
            "present": False,
        },
    }
    manifest = build_manifest(
        output_dir=output_dir,
        mode="local-loopback",
        request_count=args.request_count,
        runtime=runtime_payload,
        remote_evidence=remote_payload,
        support=support_payload,
        artifacts=artifacts,
    )
    write_json(manifest, manifest_json)
    write_markdown(manifest, manifest_md)
    manifest["artifacts"]["demo_manifest_markdown"]["present"] = manifest_md.is_file()
    write_json(manifest, manifest_json)
    write_markdown(manifest, manifest_md)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CrowdTensor local-loopback demo manifest.")
    parser.add_argument("--output-dir", default="dist/demo-manifest")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8914)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--miner-id", default=DEFAULT_MINER_ID)
    parser.add_argument("--observer-token", default=DEFAULT_OBSERVER_TOKEN)
    parser.add_argument("--admin-token", default=DEFAULT_ADMIN_TOKEN)
    parser.add_argument("--invite-token", default=DEFAULT_INVITE_TOKEN)
    args = parser.parse_args(argv)
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    return args


def main() -> None:
    try:
        manifest = build_pack(parse_args())
        print(json.dumps(manifest, sort_keys=True))
        raise SystemExit(0 if manifest.get("ok") else 1)
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
