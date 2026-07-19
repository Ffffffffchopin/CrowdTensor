#!/usr/bin/env python3
"""Check Community governance/docs/templates and local Markdown links."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_files


SCHEMA = "crowdtensor_community_docs_check_v1"
REQUIRED = (
    "LICENSE",
    "README.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "ROADMAP.md",
    "CHANGELOG.md",
    "docs/community-quickstart.md",
    "docs/community-architecture.md",
    "docs/model-adapters.md",
    "docs/providers.md",
    "docs/threat-model.md",
    "docs/benchmarks.md",
    "docs/compatibility-matrix.md",
    "docs/governance.md",
    "docs/license-audit.md",
    "docs/community-release.md",
    "docs/volunteer-campaign-governance.md",
    "docs/volunteer-training-launch-kit.md",
    "docs/rfcs/0001-model-adapter-v1.md",
    "examples/community/README.md",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/feature_request.md",
    ".github/ISSUE_TEMPLATE/good_first_issue.md",
    ".github/ISSUE_TEMPLATE/rfc.md",
    ".github/pull_request_template.md",
    ".github/release.yml",
    ".github/workflows/community-rc.yml",
)
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def check(root: str | Path = ".") -> dict[str, Any]:
    base = Path(root).resolve()
    missing = [name for name in REQUIRED if not (base / name).is_file()]
    broken: list[dict[str, str]] = []
    markdown = [base / name for name in REQUIRED if name.endswith(".md") and (base / name).is_file()]
    for path in markdown:
        text = path.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            target = target.strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                broken.append(
                    {
                        "source_hash": "sha256:" + hashlib.sha256(str(path.relative_to(base)).encode()).hexdigest(),
                        "target_hash": "sha256:" + hashlib.sha256(target.encode()).hexdigest(),
                    }
                )
    readme = (base / "README.md").read_text(encoding="utf-8") if (base / "README.md").is_file() else ""
    contributing = (base / "CONTRIBUTING.md").read_text(encoding="utf-8") if (base / "CONTRIBUTING.md").is_file() else ""
    required_phrases = {
        "readme_community_entry": "crowdtensor community init" in readme,
        "readme_logical_node_boundary": "Kaggle logical multi-node" in readme,
        "readme_campaign_positioning": "open campaigns for volunteer model" in readme,
        "readme_preview_boundary": "formal_launch_ready" in readme,
        "contributing_rfc": "docs/rfcs/" in contributing,
        "contributing_conduct": "CODE_OF_CONDUCT.md" in contributing,
    }
    strict_markdown = [
        path
        for path in markdown
        if path.name == "CODE_OF_CONDUCT.md"
        or path.parent.name == "rfcs"
        or path.name.startswith("community-")
        or path.name
        in {
            "model-adapters.md",
            "providers.md",
            "threat-model.md",
            "benchmarks.md",
            "compatibility-matrix.md",
            "governance.md",
            "license-audit.md",
            "volunteer-campaign-governance.md",
            "volunteer-training-launch-kit.md",
        }
    ]
    privacy = scan_public_files(strict_markdown)
    report = {
        "schema": SCHEMA,
        "ok": not missing and not broken and all(required_phrases.values()) and privacy["ok"],
        "required_file_count": len(REQUIRED),
        "missing_files": missing,
        "broken_link_count": len(broken),
        "broken_links": broken,
        "required_phrases": required_phrases,
        "strict_public_markdown_count": len(strict_markdown),
        "documentation_example_file_count": len(markdown) - len(strict_markdown),
        "documentation_placeholders_are_not_runtime_credentials": True,
        "public_safety": privacy,
        "absolute_paths_public": False,
        "public_artifact_safe": privacy["ok"] is True,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = check(args.root)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True) if args.json else f"docs_ok={report['ok']} missing={len(report['missing_files'])}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
