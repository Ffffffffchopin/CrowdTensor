from __future__ import annotations

import json
from pathlib import Path

import pytest

from crowdtensor.release import prepare_public_release_dir, verify_public_release_dir
from crowdtensor.version import __version__


def _sources(root: Path) -> dict[str, Path]:
    wheel = root / f"crowdtensord-{__version__}-py3-none-any.whl"
    sdist = root / f"crowdtensord-{__version__}.tar.gz"
    installer = root / "install-contributor.sh"
    notes = root / "RELEASE_NOTES.md"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    installer.write_text(
        f'#!/bin/sh\nVERSION="{__version__}"\n', encoding="utf-8"
    )
    notes.write_text("release notes\n", encoding="utf-8")
    return {"wheel": wheel, "sdist": sdist, "installer": installer, "notes": notes}


def test_prepare_and_verify_campaign_release(tmp_path) -> None:
    sources = _sources(tmp_path)
    release = tmp_path / "release"
    report = prepare_public_release_dir(
        release, **sources, commit="test-commit"
    )
    assert report["ok"] is True
    assert report["prepared"] is True
    assert report["version"] == __version__
    verified = verify_public_release_dir(release)
    assert verified["checksums_verified"] is True
    assert verified["wheel_sha256"].startswith("sha256:")
    assert verified["release_manifest_sha256"].startswith("sha256:")
    assert str(tmp_path) not in json.dumps(verified, sort_keys=True)


def test_release_verifier_rejects_mutated_wheel(tmp_path) -> None:
    sources = _sources(tmp_path)
    release = tmp_path / "release"
    prepare_public_release_dir(release, **sources, commit="test-commit")
    wheel = release / f"crowdtensord-{__version__}-py3-none-any.whl"
    wheel.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact_integrity_failed"):
        verify_public_release_dir(release)
