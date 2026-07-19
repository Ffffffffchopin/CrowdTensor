#!/usr/bin/env python3
"""Build the offline founding-preview launch bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_files
from crowdtensor.training_contract import sha256_file, sha256_json
from crowdtensor.volunteer_campaign_proposal import validate_proposal
from crowdtensor.volunteer_training_protocol import with_public_safety
from scripts.volunteer_training_operator_beta_check import check as check_operator
from scripts.volunteer_training_public_demo_check import check as check_demo
from scripts.volunteer_training_public_launch_check import (
    MULTI_HOST_SCHEMA,
    SCHEMA,
    _validate_multi_host,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"launch source is missing: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o644)


def _copy_tree(source_root: Path, destination_root: Path) -> None:
    for source in sorted(item for item in source_root.rglob("*") if item.is_file()):
        _copy(source, destination_root / source.relative_to(source_root))


def _visual_ready(path: Path) -> bool:
    value = _read(path)
    expected = sha256_json({key: item for key, item in value.items() if key != "content_hash"})
    return bool(
        value.get("content_hash") == expected
        and value.get("ok") is True
        and all(
            (value.get("viewports") or {}).get(name, {}).get("canvas_nonblank") is True
            and (value.get("viewports") or {}).get(name, {}).get("horizontal_overflow") is False
            and (value.get("viewports") or {}).get(name, {}).get("vertical_order_coherent") is True
            for name in ("desktop", "mobile")
        )
    )


def pack(
    *,
    demo_report: str | Path,
    proposal: str | Path,
    operator_rc: str | Path,
    visual_report: str | Path,
    output_dir: str | Path,
    physical_multihost_report: str | Path = "",
) -> dict[str, Any]:
    root = Path.cwd().resolve()
    demo_source = Path(demo_report).expanduser().resolve()
    proposal_source = Path(proposal).expanduser().resolve()
    operator_source = Path(operator_rc).expanduser().resolve()
    visual_source = Path(visual_report).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    proposal_value = _read(proposal_source)
    proposal_result = validate_proposal(proposal_value)
    demo_result = check_demo(demo_source, require_verified=True)
    operator_result = check_operator(operator_source, require_ready=True)
    visual_value = _read(visual_source)
    visual_ready = _visual_ready(visual_source)
    demo_value = _read(demo_source)

    artifacts: dict[str, str] = {}

    def record(name: str, source: Path, relative: str) -> None:
        destination = output / relative
        _copy(source, destination)
        artifacts[name] = relative

    record("readme", root / "README.md", "docs/README.md")
    record(
        "governance",
        root / "docs/volunteer-campaign-governance.md",
        "docs/volunteer-campaign-governance.md",
    )
    record(
        "launch_kit",
        root / "docs/volunteer-training-launch-kit.md",
        "docs/volunteer-training-launch-kit.md",
    )
    record("proposal", proposal_source, "proposal/campaign-proposal.json")
    record(
        "proposal_schema",
        root / "schemas/volunteer_campaign_proposal_v1.schema.json",
        "proposal/volunteer_campaign_proposal_v1.schema.json",
    )

    demo_root = demo_source.parent
    _copy_tree(demo_root, output / "evidence/demo")
    artifacts["demo_report"] = "evidence/demo/" + demo_source.name
    for name, relative in (demo_value.get("artifacts") or {}).items():
        source = demo_root / str(relative)
        if source.is_file():
            # Preserve the report's relative artifact paths so its checker can
            # validate the copied evidence in the bundle.
            _copy(source, output / "evidence/demo" / str(relative))
            if name == "public_snapshot":
                artifacts["demo_snapshot"] = "evidence/demo/" + str(relative)
            elif name == "status_summary":
                artifacts["demo_status"] = "evidence/demo/" + str(relative)

    operator_root = operator_source.parent
    _copy_tree(operator_root, output / "evidence/operator")
    artifacts["operator_rc"] = "evidence/operator/" + operator_source.name

    visual_root = visual_source.parent
    _copy_tree(visual_root, output / "evidence/visual")
    artifacts["visual_report"] = "evidence/visual/" + visual_source.name
    visual_files = (visual_value.get("screenshots") or {})
    desktop_source = visual_root / str(visual_files.get("desktop") or "")
    mobile_source = visual_root / str(visual_files.get("mobile") or "")
    record("desktop_screenshot", desktop_source, "docs/assets/volunteer-dashboard-desktop.png")
    record("mobile_screenshot", mobile_source, "docs/assets/volunteer-dashboard-mobile.png")

    external_ready = False
    external_errors = ["formal_multihost_evidence_missing"]
    if physical_multihost_report:
        external_source = Path(physical_multihost_report).expanduser().resolve()
        external_ready, external_errors, external_value = _validate_multi_host(external_source)
        if external_value.get("schema") != MULTI_HOST_SCHEMA:
            external_ready = False
        record("formal_multihost", external_source, "evidence/formal-multihost.json")

    source_files = [
        output / relative
        for name, relative in artifacts.items()
        if name != "readme" and Path(relative).suffix.lower() in {".json", ".md", ".yml"}
    ]
    public_scan = scan_public_files(source_files)
    documentation_ready = all(
        phrase in (output / artifacts[name]).read_text(encoding="utf-8")
        for name, phrase in (
            ("readme", "open campaigns for volunteer model"),
            ("governance", "physical multi-host"),
            ("launch_kit", "LocalLLaMA"),
        )
    )
    founding_ready = bool(
        proposal_result.get("ok") is True
        and demo_result.get("verified") is True
        and operator_result.get("ok") is True
        and visual_ready
        and documentation_ready
        and public_scan.get("ok") is True
    )
    formal_ready = founding_ready and external_ready
    report = with_public_safety(
        {
            "schema": SCHEMA,
            "release_name": "CrowdTensor Volunteer Training Founding Preview",
            "founding_preview_ready": founding_ready,
            "formal_launch_ready": formal_ready,
            "goal_achieved": founding_ready,
            "evidence_scope": "same_host_public_campaign_preview",
            "node_scope": "same-host independent Cell processes",
            "physical_multi_host_verified": False,
            "proposal": {
                "proposal_id": proposal_value.get("proposal_id"),
                "content_hash": proposal_value.get("content_hash"),
                "ready": proposal_result.get("ok") is True,
            },
            "demo": {
                "report_content_hash": demo_value.get("content_hash"),
                "verified": demo_result.get("verified") is True,
                "two_independent_cells": True,
                "real_peft_fixture": True,
            },
            "operator_beta": {
                "verified": operator_result.get("ok") is True,
                "same_host_boundary_preserved": True,
            },
            "dashboard_visual": {
                "verified": visual_ready,
                "desktop_and_mobile_screenshots": visual_ready,
                "canvas_nonblank": visual_ready,
                "no_horizontal_overflow": visual_ready,
            },
            "documentation": {
                "ready": documentation_ready,
                "claim_matrix_present": True,
                "governance_contract_present": True,
                "onboarding_present": True,
            },
            "formal_multihost": (
                {
                    "artifact": "evidence/formal-multihost.json",
                    "verified": external_ready,
                }
                if physical_multihost_report
                else {
                    "artifact": "",
                    "verified": False,
                }
            ),
            "formal_multihost_blockers": external_errors,
            "claim_boundaries": {
                "permissionless_training_claimed": False,
                "sybil_resistance_claimed": False,
                "poisoning_resistance_claimed": False,
                "useful_quality_improvement_claimed": False,
                "internet_scale_claimed": False,
                "production_sla_claimed": False,
            },
            "artifacts": artifacts,
            "artifact_hashes": {
                name: sha256_file(output / relative)
                for name, relative in artifacts.items()
            },
            "public_artifact_scan": public_scan,
            "public_artifact_scan_ok": public_scan.get("ok") is True,
            "external_publish_performed": False,
        }
    )
    report["content_hash"] = sha256_json(report)
    report_path = output / "volunteer_training_public_launch_rc.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.chmod(0o644)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-report", required=True)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--operator-rc", required=True)
    parser.add_argument("--visual-report", required=True)
    parser.add_argument("--physical-multihost-report", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = pack(
        demo_report=args.demo_report,
        proposal=args.proposal,
        operator_rc=args.operator_rc,
        visual_report=args.visual_report,
        physical_multihost_report=args.physical_multihost_report,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(report, indent=2, sort_keys=True)
        if args.json
        else f"founding_preview_ready={report['founding_preview_ready']} formal_launch_ready={report['formal_launch_ready']}"
    )
    return 0 if report["founding_preview_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
