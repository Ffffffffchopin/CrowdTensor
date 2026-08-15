#!/usr/bin/env python3
"""Check the compact training-first repository boundary."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "crowdtensor_repository_check_v1"
REQUIRED = (
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "ROADMAP.md",
    "CHANGELOG.md",
    "RELEASE_NOTES.md",
    "architecture/module-map.json",
    "architecture/archive-manifest.json",
    "docs/architecture.md",
    "docs/archive.md",
    "docs/rfcs/0002-training-first-architecture-v2.md",
    "crowdtensor/core/contracts.py",
    "crowdtensor/release.py",
    "crowdtensor/backends/registry.py",
    "crowdtensor/adapters/providers.py",
    "scripts/install_contributor.sh",
    ".github/workflows/release.yml",
)
FORBIDDEN = (
    "coordinator.py",
    "miner_cli.py",
    "package.json",
    "package-lock.json",
    "compose.yaml",
    "web",
    "site",
    "deploy",
    "examples",
    "requirements",
    "docs/releases",
    "docs/project-memory.md",
    "docs/operations.md",
    "docs/remote-miner.md",
    "docs/glm52-kaggle-alpha.md",
    "docs/kaggle-tpu-v5e8-runbook.md",
    "docs/training-foundation.md",
    "docs/api.md",
    "crowdtensor/community_cli.py",
)
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
ROOT_MARKDOWN = (
    "AGENTS.md",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "README.md",
    "ROADMAP.md",
    "SECURITY.md",
)


def _maintained_markdown(root: Path) -> list[Path]:
    paths = [root / item for item in ROOT_MARKDOWN]
    for directory in (root / "docs", root / "plugins", root / ".github"):
        if directory.is_dir():
            paths.extend(directory.rglob("*.md"))
    return sorted({path.resolve() for path in paths if path.is_file()})


def _local_links(root: Path) -> list[str]:
    broken: list[str] = []
    for path in _maintained_markdown(root):
        for raw in LINK.findall(path.read_text(encoding="utf-8")):
            target = raw.strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (path.parent / target).resolve().exists():
                broken.append(f"{path.relative_to(root)} -> {target}")
    return broken


def _run_import_boundary(root: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import crowdtensor.cli; "
                "blocked={'torch','jax','transformers','accelerate','deepspeed'}; "
                "assert not blocked.intersection(sys.modules), "
                "sorted(blocked.intersection(sys.modules))"
            ),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "ok": result.returncode == 0,
        "return_code": result.returncode,
        "details_public": result.stderr[-240:] if result.returncode else "",
    }


def check(root: str | Path = ".") -> dict[str, Any]:
    base = Path(root).resolve()
    missing = [item for item in REQUIRED if not (base / item).exists()]
    forbidden_present = [item for item in FORBIDDEN if (base / item).exists()]
    generated_build_artifacts = []
    if (base / "build").exists():
        generated_build_artifacts.append("build")
    generated_build_artifacts.extend(
        path.name for path in sorted(base.glob("*.egg-info")) if path.exists()
    )
    scripts = sorted(
        item.relative_to(base).as_posix()
        for item in (base / "scripts").iterdir()
        if item.is_file()
    ) if (base / "scripts").is_dir() else []
    script_boundary_ok = scripts == [
        "scripts/check_repository.py",
        "scripts/install_contributor.sh",
    ]
    pyproject = (base / "pyproject.toml").read_text(encoding="utf-8") if (base / "pyproject.toml").is_file() else ""
    entrypoint_ok = (
        '[project.scripts]\n' in pyproject
        and 'crowdtensor = "crowdtensor.cli:main"' in pyproject
        and 'crowdtensord = ' not in pyproject
        and 'crowdtensor-miner = ' not in pyproject
        and 'crowdtensor-community = ' not in pyproject
    )
    links = _local_links(base)
    import_boundary = _run_import_boundary(base)
    checks = {
        "required_files": not missing,
        "archived_paths_absent": not forbidden_present,
        "generated_build_artifacts_absent": not generated_build_artifacts,
        "script_boundary": script_boundary_ok,
        "single_public_entrypoint": entrypoint_ok,
        "local_markdown_links": not links,
        "lazy_optional_import_boundary": import_boundary["ok"],
    }
    return {
        "schema": SCHEMA,
        "ok": all(checks.values()),
        "checks": checks,
        "missing_files": missing,
        "forbidden_paths_present": forbidden_present,
        "generated_build_artifacts": generated_build_artifacts,
        "scripts": scripts,
        "broken_links": links,
        "import_boundary": import_boundary,
        "public_artifact_safe": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = check(args.root)
    print(json.dumps(report, sort_keys=True) if args.json else f"repository_ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
