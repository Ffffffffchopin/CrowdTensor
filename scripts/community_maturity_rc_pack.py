#!/usr/bin/env python3
"""Pack P0-P4 evidence into a portable Community Maturity RC directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.model_adapter import stable_hash
from crowdtensor.version import public_version
from scripts.community_maturity_rc_check import (
    REQUIRED_ARTIFACTS,
    SCHEMA,
    check_report,
    derive_requirements,
    derive_readiness,
    source_checks,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("community_maturity_source_invalid")
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _portable_source(name: str, source: Path, output: Path) -> Path:
    try:
        source.relative_to(output)
        return source
    except ValueError:
        pass
    evidence = output / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    if name == "release":
        destination = evidence / "release"
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(source.parent, destination)
        return destination / source.name
    destination = evidence / f"{name}.json"
    shutil.copy2(source, destination)
    return destination


def pack(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    supplied = {
        "local_gate": Path(args.local_gate).expanduser().resolve(),
        "release": Path(args.release).expanduser().resolve(),
        "docs": Path(args.docs).expanduser().resolve(),
        "minio": Path(args.minio).expanduser().resolve(),
        "wheel_smoke": Path(args.wheel_smoke).expanduser().resolve(),
        "kaggle_live": Path(args.kaggle_live).expanduser().resolve(),
        "smollm_live": Path(args.smollm_live).expanduser().resolve(),
        "cleanup_audit": Path(args.cleanup_audit).expanduser().resolve(),
        "gpu_diagnostic": Path(args.gpu_diagnostic).expanduser().resolve(),
    }
    if set(supplied) != REQUIRED_ARTIFACTS or any(not path.is_file() for path in supplied.values()):
        raise ValueError("community_maturity_required_source_missing")
    paths = {name: _portable_source(name, path, output) for name, path in supplied.items()}
    values = {name: _read(path) for name, path in paths.items()}
    checks = source_checks(values, paths)
    requirements = derive_requirements(values, checks)
    gates = derive_readiness(values, checks)
    blockers = [
        "community_maturity_" + field.removesuffix("_ready") + "_incomplete"
        for field in ("p0_ready", "p1_ready", "p2_ready", "p3_ready", "p4_ready", "cleanup_ready")
        if gates[field] is not True
    ]
    artifacts = {
        name: {
            "relative_path": str(path.relative_to(output)),
            "sha256": _hash(path),
            "source_schema": str(values[name].get("schema") or ""),
            "source_content_hash": str(values[name].get("content_hash") or ""),
        }
        for name, path in sorted(paths.items())
    }
    report = {
        "schema": SCHEMA,
        "release_name": "CrowdTensor Community Maturity RC",
        "versions": public_version(),
        "node_scope": "Kaggle logical multi-node",
        "physical_multi_machine_verified": False,
        "community_maturity_rc_ready": gates["community_maturity_rc_ready"],
        "gates": gates,
        "requirements": requirements,
        "source_checks": checks,
        "artifacts": artifacts,
        "blockers": sorted(blockers),
        "external_publish_performed": False,
        "credential_values_public": False,
        "credential_paths_public": False,
        "cookies_public": False,
        "private_urls_public": False,
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    privacy = scan_public_value(report)
    report["public_safety"] = privacy
    report["public_artifact_safe"] = privacy["ok"] is True
    if not privacy["ok"]:
        report["community_maturity_rc_ready"] = False
        report["blockers"] = sorted(set(report["blockers"]) | {"community_maturity_public_safety_failed"})
    report["content_hash"] = stable_hash(report)
    structural_errors = check_report(report, require_ready=False)
    if structural_errors:
        raise RuntimeError("community_maturity_pack_invalid:" + ",".join(structural_errors))
    (output / "community_maturity_rc.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    for name in sorted(REQUIRED_ARTIFACTS):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = pack(args)
    print(json.dumps(report, sort_keys=True) if args.json else f"ready={report['community_maturity_rc_ready']}")
    return 0 if report["community_maturity_rc_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
