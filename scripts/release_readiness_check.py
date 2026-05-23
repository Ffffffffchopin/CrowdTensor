#!/usr/bin/env python3
"""Acceptance check for CrowdTensorD release readiness reporting."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SECRET_FRAGMENTS = (
    "demo-manifest-token",
    "CROWDTENSOR_MINER_TOKEN",
    "lease_token",
    "idempotency_key",
    "inference_results",
    "external_llm_results",
    "output_text",
    "Bearer ",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the CrowdTensor release readiness pack.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=8924)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--external-llm-request-count", type=int, default=3)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-external-llm-evidence", action="store_true")
    args = parser.parse_args()
    if args.base_port < 1:
        raise SystemExit("--base-port must be positive")
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    if args.external_llm_request_count < 1:
        raise SystemExit("--external-llm-request-count must be at least 1")
    return args


def load_payload(output_dir: Path, stdout: str) -> dict:
    path = output_dir / "release_readiness.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit("release_readiness_pack.py emitted no JSON")
    return json.loads(lines[-1])


def main() -> None:
    args = parse_args()
    temp_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_release_readiness_")
        output_dir = Path(temp_dir.name)

    try:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "release_readiness_pack.py"),
            "--output-dir",
            str(output_dir),
            "--host",
            args.host,
            "--base-port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--external-llm-request-count",
            str(args.external_llm_request_count),
        ]
        if args.allow_dirty:
            command.append("--allow-dirty")
        if args.skip_external_llm_evidence:
            command.append("--skip-external-llm-evidence")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        payload = load_payload(output_dir, completed.stdout)
        if completed.returncode != 0 and payload.get("ok") is True:
            raise SystemExit(
                "release_readiness_pack.py failed despite ok payload\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        if payload.get("schema") != "release_readiness_v1":
            raise SystemExit(f"unexpected release readiness schema: {payload.get('schema')}")
        status = payload.get("release_status") or {}
        if not isinstance(status.get("diagnosis_codes"), list):
            raise SystemExit(f"missing diagnosis codes: {status}")
        if args.allow_dirty and payload.get("ok") is not True:
            raise SystemExit(f"release readiness failed in allow-dirty mode: {json.dumps(payload, sort_keys=True)}")
        if not args.allow_dirty and payload.get("ok") is True and (payload.get("git") or {}).get("dirty"):
            raise SystemExit(f"dirty git state passed without --allow-dirty: {payload.get('git')}")

        checks = payload.get("checks") or {}
        for name in ["release_gate", "security_preflight"]:
            check = checks.get(name) or {}
            if check.get("ok") is not True:
                raise SystemExit(f"{name} summary is not ok: {check}")
            if int(check.get("total") or 0) < 1:
                raise SystemExit(f"{name} summary has no checks: {check}")

        demo = ((payload.get("reports") or {}).get("demo_manifest") or {}).get("summary") or {}
        if demo.get("ok") is not True or demo.get("schema") != "demo_manifest_v1":
            raise SystemExit(f"demo manifest summary is invalid: {demo}")
        if int(demo.get("artifact_count") or 0) < 4:
            raise SystemExit(f"demo manifest artifact count is too low: {demo}")

        artifacts = payload.get("artifacts") or {}
        for name in ["release_readiness_json", "release_readiness_markdown", "demo_manifest_json", "demo_manifest_markdown"]:
            artifact = artifacts.get(name) or {}
            relative = Path(str(artifact.get("path") or ""))
            if relative.is_absolute() or not relative.parts:
                raise SystemExit(f"artifact path is not relative for {name}: {artifact}")
            if artifact.get("present") is not True:
                raise SystemExit(f"artifact is not present for {name}: {artifact}")
            if not (output_dir / relative).is_file():
                raise SystemExit(f"artifact path does not exist for {name}: {artifact}")

        acceptance = ((payload.get("reports") or {}).get("acceptance") or {})
        for name in ["runtime", "browser", "remote"]:
            if name not in acceptance:
                raise SystemExit(f"missing acceptance report slot: {name}")
        warnings = set(status.get("warnings") or [])
        for expected in ["runtime_report_missing", "browser_report_missing", "remote_report_missing"]:
            if expected not in warnings:
                raise SystemExit(f"missing expected warning {expected}: {warnings}")

        encoded = json.dumps(payload, sort_keys=True)
        for fragment in SECRET_FRAGMENTS:
            if fragment in encoded:
                raise SystemExit(f"release readiness leaked secret-like material: {fragment}")

        print(json.dumps({
            "ok": True,
            "schema": payload["schema"],
            "status": status.get("status"),
            "diagnosis_codes": status.get("diagnosis_codes"),
            "artifact_count": len(artifacts),
            "warnings": sorted(warnings),
        }, sort_keys=True))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
