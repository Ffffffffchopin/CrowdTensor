"""Build and verify one immutable, Campaign-served contributor release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .version import __version__


PUBLIC_RELEASE_ARTIFACT_NAMES = frozenset(
    {
        f"crowdtensord-{__version__}-py3-none-any.whl",
        f"crowdtensord-{__version__}.tar.gz",
        "SHA256SUMS",
        "RELEASE_NOTES.md",
        "install-contributor.sh",
        "release.json",
    }
)
MANIFEST_ARTIFACT_NAMES = PUBLIC_RELEASE_ARTIFACT_NAMES - {
    "SHA256SUMS",
    "release.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_revision() -> str:
    configured = str(os.environ.get("GITHUB_SHA") or "").strip()
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unbound-source"
    return result.stdout.strip() if result.returncode == 0 else "unbound-source"


def resolve_public_release_dir(value: str | Path | None = None) -> Path | None:
    """Resolve one complete, version-matched Campaign release directory."""

    configured = str(
        value or os.environ.get("CROWDTENSOR_PUBLIC_RELEASE_DIR") or ""
    ).strip()
    if not configured:
        return None
    root = Path(configured).expanduser().resolve()
    if not root.is_dir() or any(
        not (root / name).is_file() for name in PUBLIC_RELEASE_ARTIFACT_NAMES
    ):
        raise ValueError("volunteer_release_directory_incomplete")
    try:
        manifest = json.loads((root / "release.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("volunteer_release_manifest_invalid") from exc
    records = manifest.get("artifacts") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "crowdtensor_release_v1"
        or manifest.get("version") != __version__
        or not str(manifest.get("commit") or "")
        or not isinstance(records, list)
        or any(not isinstance(item, dict) for item in records)
        or {str(item.get("name") or "") for item in records}
        != MANIFEST_ARTIFACT_NAMES
    ):
        raise ValueError("volunteer_release_manifest_invalid")
    for record in records:
        try:
            name = str(record["name"])
            path = root / name
            valid = (
                str(record.get("sha256") or "") == _sha256(path)
                and int(record.get("byte_count") or -1) == path.stat().st_size
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise ValueError("volunteer_release_manifest_invalid") from exc
        if not valid:
            raise ValueError("volunteer_release_artifact_integrity_failed")
    checksums: dict[str, str] = {}
    checksum_lines = 0
    try:
        for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, name = line.split(None, 1)
            checksum_lines += 1
            checksums[name.lstrip("* ")] = digest
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("volunteer_release_checksums_invalid") from exc
    expected_checksum_names = PUBLIC_RELEASE_ARTIFACT_NAMES - {"SHA256SUMS"}
    if set(checksums) != expected_checksum_names or checksum_lines != len(checksums):
        raise ValueError("volunteer_release_checksums_invalid")
    for name, expected in checksums.items():
        if _sha256(root / name) != expected:
            raise ValueError("volunteer_release_artifact_integrity_failed")
    return root


def verify_public_release_dir(value: str | Path) -> dict[str, Any]:
    root = resolve_public_release_dir(value)
    assert root is not None
    manifest = json.loads((root / "release.json").read_text(encoding="utf-8"))
    report = {
        "schema": "crowdtensor_release_verification_v1",
        "ok": True,
        "version": __version__,
        "commit": manifest["commit"],
        "artifact_count": len(PUBLIC_RELEASE_ARTIFACT_NAMES),
        "checksums_verified": True,
        "release_manifest_sha256": "sha256:" + _sha256(root / "release.json"),
        "checksums_sha256": "sha256:" + _sha256(root / "SHA256SUMS"),
        "wheel_sha256": "sha256:"
        + _sha256(root / f"crowdtensord-{__version__}-py3-none-any.whl"),
        "sdist_sha256": "sha256:"
        + _sha256(root / f"crowdtensord-{__version__}.tar.gz"),
        "public_artifact_safe": True,
        "private_paths_public": False,
        "credential_values_public": False,
    }
    report["content_hash"] = "sha256:" + hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return report


def prepare_public_release_dir(
    destination: str | Path,
    *,
    wheel: str | Path | None = None,
    sdist: str | Path | None = None,
    installer: str | Path = "scripts/install_contributor.sh",
    notes: str | Path = "RELEASE_NOTES.md",
    commit: str = "",
) -> dict[str, Any]:
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError("crowdtensor_release_destination_exists")
    sources = {
        f"crowdtensord-{__version__}-py3-none-any.whl": Path(
            wheel or f"dist/crowdtensord-{__version__}-py3-none-any.whl"
        ).expanduser(),
        f"crowdtensord-{__version__}.tar.gz": Path(
            sdist or f"dist/crowdtensord-{__version__}.tar.gz"
        ).expanduser(),
        "install-contributor.sh": Path(installer).expanduser(),
        "RELEASE_NOTES.md": Path(notes).expanduser(),
    }
    if any(not path.is_file() for path in sources.values()):
        raise FileNotFoundError("crowdtensor_release_source_artifact_missing")
    installer_text = sources["install-contributor.sh"].read_text(encoding="utf-8")
    if f'VERSION="{__version__}"' not in installer_text:
        raise ValueError("crowdtensor_release_installer_version_mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.{secrets.token_hex(6)}.tmp")
    try:
        staging.mkdir(mode=0o700)
        for name, source in sources.items():
            target = staging / name
            shutil.copyfile(source, target)
            target.chmod(0o755 if name.endswith(".sh") else 0o644)
        artifacts = [
            {
                "name": name,
                "byte_count": (staging / name).stat().st_size,
                "sha256": _sha256(staging / name),
            }
            for name in sorted(MANIFEST_ARTIFACT_NAMES)
        ]
        manifest = {
            "schema": "crowdtensor_release_v1",
            "version": __version__,
            "commit": str(commit or _source_revision()),
            "artifacts": artifacts,
            "public_artifact_safe": True,
        }
        (staging / "release.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "release.json").chmod(0o644)
        checksum_names = PUBLIC_RELEASE_ARTIFACT_NAMES - {"SHA256SUMS"}
        (staging / "SHA256SUMS").write_text(
            "".join(
                _sha256(staging / name) + "  " + name + "\n"
                for name in sorted(checksum_names)
            ),
            encoding="utf-8",
        )
        (staging / "SHA256SUMS").chmod(0o644)
        os.replace(staging, output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    report = verify_public_release_dir(output)
    return {**report, "prepared": True}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crowdtensor release")
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare")
    prepare.add_argument("destination")
    prepare.add_argument("--wheel", default="")
    prepare.add_argument("--sdist", default="")
    prepare.add_argument("--installer", default="scripts/install_contributor.sh")
    prepare.add_argument("--notes", default="RELEASE_NOTES.md")
    prepare.add_argument("--commit", default="")
    prepare.add_argument("--json", action="store_true")
    verify = actions.add_parser("verify")
    verify.add_argument("release_dir")
    verify.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            report = prepare_public_release_dir(
                args.destination,
                wheel=args.wheel or None,
                sdist=args.sdist or None,
                installer=args.installer,
                notes=args.notes,
                commit=args.commit,
            )
        else:
            report = verify_public_release_dir(args.release_dir)
    except (OSError, ValueError) as exc:
        report = {
            "schema": "crowdtensor_release_command_error_v1",
            "ok": False,
            "error": str(exc),
            "public_artifact_safe": True,
        }
    print(json.dumps(report, sort_keys=True) if args.json else report)
    raise SystemExit(0 if report.get("ok") is True else 1)


if __name__ == "__main__":
    main()
