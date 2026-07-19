"""Helpers for reusing the local google-colab-cli runtime package.

The Colab live probes are usually launched with the project Python, while the
official google-colab-cli package is installed as an isolated uv tool. Keep the
project dependency surface unchanged and make that tool environment discoverable
only for the bounded live probes that need it.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping


def _paths_from_tool_python(tool_python: Path) -> list[Path]:
    tool_root = tool_python.resolve().parents[1]
    return sorted((tool_root / "lib").glob("python*/site-packages"))


def _tool_python_from_script(colab_executable: Path) -> Path | None:
    try:
        first_line = colab_executable.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except Exception:
        return None
    if not first_line.startswith("#!"):
        return None
    candidate = Path(first_line[2:].strip().split()[0]).expanduser()
    return candidate if candidate.exists() else None


def site_package_candidates(
    *,
    env: Mapping[str, str] | None = None,
    colab_executable: str | None = None,
) -> list[Path]:
    env = env or os.environ
    candidates: list[Path] = []
    explicit_site = str(env.get("COLAB_CLI_SITE_PACKAGES") or "").strip()
    if explicit_site:
        candidates.append(Path(explicit_site).expanduser())
    explicit_python = str(env.get("COLAB_CLI_PYTHON") or "").strip()
    if explicit_python:
        candidates.extend(_paths_from_tool_python(Path(explicit_python).expanduser()))
    executable = colab_executable or shutil.which("colab") or str(Path.home() / ".local" / "bin" / "colab")
    if executable:
        tool_python = _tool_python_from_script(Path(executable).expanduser())
        if tool_python is not None:
            candidates.extend(_paths_from_tool_python(tool_python))
    candidates.extend(sorted((Path.home() / ".local" / "share" / "uv" / "tools" / "google-colab-cli" / "lib").glob("python*/site-packages")))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path)
        if resolved in seen or not path.is_dir():
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def ensure_importable(paths: Iterable[Path] | None = None) -> list[str]:
    added: list[str] = []
    for path in paths or site_package_candidates():
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
            added.append(value)
    return added


def load_colab_runtime_class() -> Any:
    try:
        from colab_cli.runtime import ColabRuntime

        return ColabRuntime
    except ModuleNotFoundError:
        ensure_importable()
        from colab_cli.runtime import ColabRuntime

        return ColabRuntime
