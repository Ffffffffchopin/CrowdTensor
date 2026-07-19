#!/usr/bin/env python3
"""Pack the self-contained Volunteer Campaign Operator Beta RC."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_files
from crowdtensor.training_contract import sha256_file, sha256_json
from crowdtensor.volunteer_training_protocol import with_public_safety


SCHEMA = "crowdtensor_volunteer_campaign_single_host_operator_beta_rc_v1"
PROBE_SCHEMA = "crowdtensor_volunteer_campaign_single_host_operator_beta_probe_v1"
RELEASE_SCHEMA = "crowdtensor_volunteer_operator_beta_release_probe_v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Operator Beta evidence must be a JSON object")
    return value


def pack(
    *,
    probe_path: str | Path,
    release_path: str | Path,
    retained_real_rc_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path.cwd().resolve()
    probe_source = Path(probe_path).expanduser().resolve()
    release_source = Path(release_path).expanduser().resolve()
    real_source = Path(retained_real_rc_path).expanduser().resolve()
    probe = _read(probe_source)
    release = _read(release_source)
    if probe.get("schema") != PROBE_SCHEMA:
        raise ValueError("Operator Beta probe schema mismatch")
    if release.get("schema") != RELEASE_SCHEMA:
        raise ValueError("Operator Beta release schema mismatch")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    evidence = output / "evidence"
    evidence.mkdir(parents=True)
    artifacts: dict[str, str] = {}
    hashes: dict[str, str] = {}

    def copy(name: str, source: Path, destination: Path) -> None:
        if not source.is_file():
            raise FileNotFoundError(f"Operator Beta artifact missing: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(0o644)
        artifacts[name] = destination.relative_to(output).as_posix()
        hashes[name] = sha256_file(destination)

    copy("probe", probe_source, evidence / "operator_probe.json")
    for name, relative in (probe.get("artifacts") or {}).items():
        copy(
            "probe_" + str(name),
            probe_source.parent / str(relative),
            evidence / "probe" / Path(str(relative)).name,
        )
    copy("release", release_source, evidence / "release_probe.json")
    wheel_name = str((release.get("wheel") or {}).get("file_name") or "")
    copy("wheel", release_source.parent / wheel_name, output / "artifacts" / wheel_name)

    real_destination = evidence / "retained-real-peft"
    for source in sorted(item for item in real_source.parent.rglob("*") if item.is_file()):
        relative = source.relative_to(real_source.parent)
        copy(
            "real_peft_" + relative.as_posix().replace("/", "_").replace(".", "_"),
            source,
            real_destination / relative,
        )
    retained_key = next(
        name
        for name, relative in artifacts.items()
        if relative.endswith("/volunteer_training_internet_beta_engineering_rc.json")
    )

    docs = {
        "operator_runbook": root / "docs/volunteer-training-operator-beta.md",
        "internet_beta": root / "docs/volunteer-training-internet-beta.md",
        "readme": root / "README.md",
    }
    for name, source in docs.items():
        copy("doc_" + name, source, output / "docs" / source.name)

    ready = bool(
        probe.get("ok") is True
        and probe.get("volunteer_campaign_single_host_operator_beta_verified")
        is True
        and release.get("ok") is True
        and (probe.get("cleanup") or {}).get("cleanup_verified") is True
        and probe.get("public_artifact_scan_ok") is True
        and release.get("public_artifact_scan_ok") is True
    )
    report = with_public_safety(
        {
            "schema": SCHEMA,
            "volunteer_campaign_single_host_operator_beta_ready": ready,
            "goal_achieved": ready,
            "evidence_scope": probe.get("evidence_scope"),
            "security": probe.get("security"),
            "deployment": probe.get("deployment"),
            "lifecycle": probe.get("lifecycle"),
            "stress": probe.get("stress"),
            "faults": probe.get("faults"),
            "monitoring": probe.get("monitoring"),
            "retained_real_peft": probe.get("retained_real_peft"),
            "release": {
                "ok": release.get("ok") is True,
                "wheel": release.get("wheel"),
                "clean_install": release.get("clean_install"),
                "container": release.get("container"),
                "external_publish_performed": release.get(
                    "external_publish_performed"
                ),
                "public_artifact_scan_ok": release.get("public_artifact_scan_ok"),
            },
            "cleanup": probe.get("cleanup"),
            "limitations": probe.get("limitations"),
            "retained_real_peft_rc_artifact": artifacts[retained_key],
            "artifacts": artifacts,
            "artifact_hashes": hashes,
            "documentation": {
                name: artifacts["doc_" + name] for name in docs
            },
        }
    )
    public_paths = [
        output / relative
        for name, relative in artifacts.items()
        if (
            (name == "probe" or name.startswith("probe_") or name == "release")
            and name != "wheel"
        )
    ]
    scan = scan_public_files(public_paths)
    report["public_artifact_scan"] = scan
    report["public_artifact_scan_ok"] = scan.get("ok") is True
    report["content_hash"] = sha256_json(report)
    report_path = output / "volunteer_training_operator_beta_rc.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.chmod(0o644)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--retained-real-rc", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = pack(
        probe_path=args.probe,
        release_path=args.release,
        retained_real_rc_path=args.retained_real_rc,
        output_dir=args.output_dir,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "operator_beta_rc_ready="
            + str(report["volunteer_campaign_single_host_operator_beta_ready"])
        )
    return 0 if report["goal_achieved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
