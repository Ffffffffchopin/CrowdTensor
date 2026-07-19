from __future__ import annotations

from pathlib import Path

from scripts import colab_cli_runtime


def test_site_package_candidates_reads_uv_tool_shebang(tmp_path: Path) -> None:
    tool_root = tmp_path / "google-colab-cli"
    tool_python = tool_root / "bin" / "python"
    site_packages = tool_root / "lib" / "python3.13" / "site-packages"
    tool_python.parent.mkdir(parents=True)
    tool_python.write_text("#!/bin/sh\n", encoding="utf-8")
    site_packages.mkdir(parents=True)
    colab = tmp_path / "colab"
    colab.write_text(f"#!{tool_python}\nfrom colab_cli.cli import main\n", encoding="utf-8")

    candidates = colab_cli_runtime.site_package_candidates(env={}, colab_executable=str(colab))

    assert site_packages in candidates


def test_ensure_importable_adds_candidate_once(tmp_path: Path, monkeypatch) -> None:
    candidate = tmp_path / "site-packages"
    candidate.mkdir()
    original_path = list(colab_cli_runtime.sys.path)
    monkeypatch.setattr(colab_cli_runtime.sys, "path", list(original_path))

    first = colab_cli_runtime.ensure_importable([candidate])
    second = colab_cli_runtime.ensure_importable([candidate])

    assert first == [str(candidate)]
    assert second == []
    assert colab_cli_runtime.sys.path.count(str(candidate)) == 1
