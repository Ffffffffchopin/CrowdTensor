#!/usr/bin/env python3
"""Acceptance check for the local-loopback CrowdTensor demo manifest."""

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
    "Bearer ",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the CrowdTensor demo manifest pack.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=8914)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--request-count", type=int, default=4)
    args = parser.parse_args()
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    return args


def load_manifest(output_dir: Path, stdout: str) -> dict:
    manifest_path = output_dir / "demo_manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit("demo_manifest_pack.py emitted no manifest JSON")
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
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_demo_manifest_")
        output_dir = Path(temp_dir.name)

    try:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "demo_manifest_pack.py"),
            "--output-dir",
            str(output_dir),
            "--host",
            args.host,
            "--port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        if completed.returncode != 0:
            raise SystemExit(
                "demo_manifest_pack.py failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        manifest = load_manifest(output_dir, completed.stdout)
        if manifest.get("schema") != "demo_manifest_v1":
            raise SystemExit(f"unexpected manifest schema: {manifest.get('schema')}")
        if manifest.get("mode") != "local-loopback":
            raise SystemExit(f"unexpected manifest mode: {manifest.get('mode')}")
        if manifest.get("ok") is not True:
            raise SystemExit(f"demo manifest failed: {json.dumps(manifest, sort_keys=True)}")

        artifacts = manifest.get("artifacts") or {}
        required = [
            "runtime_matrix",
            "remote_compute_evidence_json",
            "remote_compute_evidence_markdown",
            "support_bundle_json",
            "support_bundle_markdown",
            "demo_manifest_markdown",
        ]
        for name in required:
            artifact = artifacts.get(name) or {}
            relative = Path(str(artifact.get("path") or ""))
            if relative.is_absolute() or not relative.parts:
                raise SystemExit(f"artifact path is not relative for {name}: {artifact}")
            if artifact.get("present") is not True:
                raise SystemExit(f"artifact is not present for {name}: {artifact}")
            if not (output_dir / relative).is_file():
                raise SystemExit(f"artifact path does not exist for {name}: {artifact}")

        summaries = manifest.get("summaries") or {}
        runtime = summaries.get("runtime_matrix") or {}
        remote = summaries.get("remote_compute_evidence") or {}
        support = summaries.get("support_bundle") or {}
        route = remote.get("route") or {}
        inference = remote.get("inference") or {}
        observability = remote.get("observability") or {}
        remote_report = support.get("remote_report") or {}
        if runtime.get("ok") is not True:
            raise SystemExit(f"runtime matrix summary is not ok: {runtime}")
        if remote.get("ok") is not True or route.get("name") != "remote_python_model_bundle_infer":
            raise SystemExit(f"remote evidence summary is invalid: {remote}")
        if int(inference.get("request_count") or 0) != args.request_count:
            raise SystemExit(f"request count mismatch: {inference}")
        if float(inference.get("requests_per_second") or 0.0) <= 0.0:
            raise SystemExit(f"invalid throughput: {inference}")
        if observability.get("schema") != "remote_compute_observability_v1":
            raise SystemExit(f"missing safe observability summary: {observability}")
        if support.get("release_gate_ok") is not True or remote_report.get("ok") is not True:
            raise SystemExit(f"support bundle summary is invalid: {support}")
        if not remote_report.get("observability_summaries"):
            raise SystemExit(f"support bundle did not preserve remote observability: {support}")

        encoded = json.dumps(manifest, sort_keys=True)
        for fragment in SECRET_FRAGMENTS:
            if fragment in encoded:
                raise SystemExit(f"demo manifest leaked secret-like material: {fragment}")

        print(json.dumps({
            "ok": True,
            "schema": manifest["schema"],
            "mode": manifest["mode"],
            "route": route.get("name"),
            "request_count": inference.get("request_count"),
            "requests_per_second": inference.get("requests_per_second"),
            "observability_schema": observability.get("schema"),
            "artifact_count": len(artifacts),
        }, sort_keys=True))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
