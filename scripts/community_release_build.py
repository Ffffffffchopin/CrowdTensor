#!/usr/bin/env python3
"""Build an offline public-safe Community RC release bundle."""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_files, scan_public_value
from crowdtensor.model_adapter import adapter_registry_report, stable_hash
from crowdtensor.version import __version__, public_version


SCHEMA = "crowdtensor_community_release_manifest_v1"
REPORT_SCHEMA = "crowdtensor_community_release_build_v1"


def _hash(path_or_bytes: str | Path | bytes) -> str:
    if isinstance(path_or_bytes, bytes):
        value = path_or_bytes
    else:
        path = Path(path_or_bytes)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: float = 1800,
) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "ok": process.returncode == 0,
        "returncode": process.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "output_hash": _hash((process.stdout or "").encode()),
    }


def _artifact(path: Path, *, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "file_name": path.name,
        "sha256": _hash(path),
        "size_bytes": path.stat().st_size,
    }


def _wheel_metadata(wheel: Path) -> tuple[dict[str, Any], list[str]]:
    required = {
        "crowdtensor/community_cli.py",
        "crowdtensor/community_protocol.py",
        "crowdtensor/community_security.py",
        "crowdtensor/community_reliability.py",
        "crowdtensor/model_adapter.py",
        "crowdtensor/smollm_training.py",
        "crowdtensor/community_live_training.py",
    }
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = email.parser.Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        license_present = any(name.endswith(".dist-info/licenses/LICENSE") for name in names)
    report = {
        "package_name": str(metadata.get("Name") or ""),
        "package_version": str(metadata.get("Version") or ""),
        "requires_python": str(metadata.get("Requires-Python") or ""),
        "license_expression": str(metadata.get("License-Expression") or metadata.get("License") or ""),
        "required_modules_present": required.issubset(names),
        "missing_required_modules": sorted(required - names),
        "license_file_present": license_present,
        "entry_points_present": any(name.endswith(".dist-info/entry_points.txt") for name in names),
    }
    return report, list(metadata.get_all("Requires-Dist") or [])


def _sdist_metadata(sdist: Path) -> dict[str, Any]:
    with tarfile.open(sdist, "r:gz") as archive:
        names = set(archive.getnames())
    suffixes = ("/pyproject.toml", "/LICENSE", "/README.md", "/crowdtensor/community_cli.py")
    return {
        "required_sources_present": all(any(name.endswith(suffix) for name in names) for suffix in suffixes),
        "source_file_count": len(names),
        "absolute_members_present": any(name.startswith("/") or ".." in Path(name).parts for name in names),
    }


def _clean_install(wheel: Path, *, constraints: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ct-community-clean-install-") as temporary:
        root = Path(temporary)
        venv = root / "venv"
        created = _run([sys.executable, "-m", "venv", str(venv)], cwd=root, timeout=180)
        if not created["ok"]:
            return {"ok": False, "phase": "venv_create", "workspace_import_used": False}
        python = venv / "bin" / "python"
        pip = venv / "bin" / "pip"
        installed = _run(
            [str(pip), "install", "--disable-pip-version-check", "--constraint", str(constraints), str(wheel)],
            cwd=root,
            timeout=600,
        )
        if not installed["ok"]:
            return {"ok": False, "phase": "wheel_install", "workspace_import_used": False, "install": installed}
        workspace = root / "workflow"
        cli = venv / "bin" / "crowdtensor-community"
        commands = [
            [str(cli), "init", str(workspace), "--target-steps", "100", "--json"],
            [str(cli), "validate", str(workspace), "--json"],
            [str(cli), "plan", str(workspace), "--json"],
            [str(cli), "coordinator", "up", str(workspace), "--dry-run", "--json"],
            [str(cli), "miner", "join", str(workspace), "--dry-run", "--json"],
            [str(cli), "train", str(workspace), "--dry-run", "--json"],
        ]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        command_reports = [_run(command, cwd=root, env=env, timeout=120) for command in commands]
        location_process = subprocess.run(
            [str(python), "-c", "import crowdtensor,json,pathlib; print(json.dumps({'version':crowdtensor.__version__,'under_venv':str(pathlib.Path(crowdtensor.__file__).resolve()).startswith(str(pathlib.Path(__import__('sys').prefix).resolve()))}))"],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            location = json.loads(location_process.stdout)
        except json.JSONDecodeError:
            location = {}
        inventory_process = subprocess.run(
            [
                str(python),
                "-c",
                "import importlib.metadata as m,json; print(json.dumps(sorted([{'name':d.metadata.get('Name',''),'version':d.version,'license':d.metadata.get('License-Expression') or d.metadata.get('License') or 'UNKNOWN'} for d in m.distributions()], key=lambda x:(x['name'].lower(),x['version']))))",
            ],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            inventory = json.loads(inventory_process.stdout)
        except json.JSONDecodeError:
            inventory = []
        return {
            "ok": bool(
                installed["ok"]
                and all(item["ok"] for item in command_reports)
                and location_process.returncode == 0
                and location.get("version") == __version__
                and location.get("under_venv") is True
            ),
            "phase": "completed",
            "fresh_venv": True,
            "wheel_installed": installed["ok"],
            "command_count": len(command_reports),
            "all_golden_commands_passed": all(item["ok"] for item in command_reports),
            "installed_version": str(location.get("version") or ""),
            "installed_under_venv": location.get("under_venv") is True,
            "workspace_import_used": False,
            "dependency_inventory": inventory,
            "temporary_environment_removed_after_return": True,
        }


def _sbom(inventory: list[dict[str, Any]], requires_dist: list[str]) -> dict[str, Any]:
    components = []
    for item in inventory:
        name = str(item.get("name") or "")
        version = str(item.get("version") or "")
        if not name:
            continue
        license_value = str(item.get("license") or "UNKNOWN")
        components.append(
            {
                "type": "library",
                "name": name,
                "version": version,
                "purl": f"pkg:pypi/{name.lower().replace('_', '-')}@{version}",
                "licenses": [{"license": {"name": license_value}}],
            }
        )
    serial = uuid.uuid5(uuid.NAMESPACE_URL, f"crowdtensord:{__version__}:community-rc")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "crowdtensord",
                "version": __version__,
                "licenses": [{"license": {"id": "Apache-2.0"}}],
            }
        },
        "components": components,
        "properties": [
            {"name": "crowdtensor:declared-requirement", "value": item}
            for item in sorted(requires_dist)
        ],
    }


def _container_build(root: Path, *, enabled: bool) -> dict[str, Any]:
    compose = _run(["docker", "compose", "config", "--quiet"], cwd=root, timeout=120)
    if not enabled:
        return {
            "requested": False,
            "compose_config_valid": compose["ok"],
            "image_built": False,
            "image_removed": True,
            "published": False,
        }
    tag = "crowdtensor-community-rc:" + secrets.token_hex(6)
    built = _run(["docker", "build", "--tag", tag, "."], cwd=root, timeout=1800)
    identity = ""
    if built["ok"]:
        inspect = subprocess.run(
            ["docker", "image", "inspect", tag, "--format", "{{.Id}}"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        identity = inspect.stdout.strip() if inspect.returncode == 0 else ""
    removed = _run(["docker", "image", "rm", "--force", tag], cwd=root, timeout=180) if identity else {"ok": True}
    return {
        "requested": True,
        "compose_config_valid": compose["ok"],
        "image_built": built["ok"] and bool(identity),
        "image_id_hash": _hash(identity.encode()) if identity else "",
        "dockerfile_hash": _hash(root / "Dockerfile"),
        "compose_hash": _hash(root / "compose.yaml"),
        "image_removed": removed["ok"],
        "published": False,
    }


def _deterministic_bundle(bundle: Path, files: list[Path], *, base: Path) -> None:
    with tarfile.open(bundle, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(files, key=lambda item: str(item.relative_to(base))):
            data = path.read_bytes()
            info = tarfile.TarInfo(str(path.relative_to(base)))
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, __import__("io").BytesIO(data))


def _prepare_python_artifacts(
    artifacts_dir: Path,
    *,
    root: Path,
    build_python: str,
    reuse_python_artifacts_from: str | Path | None,
    expected_wheel_sha256: str,
) -> tuple[Path, Path, dict[str, Any]]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if reuse_python_artifacts_from is None:
        for old in artifacts_dir.glob("crowdtensord-*"):
            old.unlink()
        build = _run(
            [
                str(build_python),
                "-m",
                "build",
                "--wheel",
                "--sdist",
                "--outdir",
                str(artifacts_dir),
            ],
            cwd=root,
            timeout=1200,
        )
        wheels = sorted(artifacts_dir.glob("*.whl"))
        sdists = sorted(artifacts_dir.glob("*.tar.gz"))
        if not build["ok"] or len(wheels) != 1 or len(sdists) != 1:
            raise RuntimeError("community_release_python_build_failed")
        wheel, sdist = wheels[0], sdists[0]
        mode = "built_from_source"
        rebuilt = True
    else:
        if not expected_wheel_sha256.startswith("sha256:"):
            raise ValueError("community_release_reuse_expected_wheel_sha256_required")
        source = Path(reuse_python_artifacts_from).expanduser().resolve()
        wheels = sorted(source.glob("*.whl"))
        sdists = sorted(source.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise ValueError("community_release_reuse_python_artifacts_invalid")
        source_wheel, source_sdist = wheels[0], sdists[0]
        if _hash(source_wheel) != expected_wheel_sha256:
            raise ValueError("community_release_reuse_wheel_sha256_mismatch")
        if source != artifacts_dir.resolve():
            for old in artifacts_dir.glob("crowdtensord-*"):
                old.unlink()
            wheel = artifacts_dir / source_wheel.name
            sdist = artifacts_dir / source_sdist.name
            shutil.copy2(source_wheel, wheel)
            shutil.copy2(source_sdist, sdist)
        else:
            wheel, sdist = source_wheel, source_sdist
        if _hash(wheel) != _hash(source_wheel) or _hash(sdist) != _hash(source_sdist):
            raise RuntimeError("community_release_reuse_copy_identity_mismatch")
        mode = "existing_python_artifacts_reused"
        rebuilt = False
    if expected_wheel_sha256 and _hash(wheel) != expected_wheel_sha256:
        raise ValueError("community_release_expected_wheel_sha256_mismatch")
    provenance = {
        "mode": mode,
        "wheel_rebuilt": rebuilt,
        "sdist_rebuilt": rebuilt,
        "expected_wheel_sha256_enforced": bool(expected_wheel_sha256),
        "wheel_sha256": _hash(wheel),
        "sdist_sha256": _hash(sdist),
        "source_paths_public": False,
    }
    return wheel, sdist, provenance


def build_release(
    output_dir: str | Path,
    *,
    build_python: str,
    build_container: bool,
    reuse_python_artifacts_from: str | Path | None = None,
    expected_wheel_sha256: str = "",
) -> dict[str, Any]:
    root = Path.cwd().resolve()
    output = Path(output_dir).expanduser().resolve()
    artifacts_dir = output / "artifacts"
    wheel, sdist, python_artifact_provenance = _prepare_python_artifacts(
        artifacts_dir,
        root=root,
        build_python=build_python,
        reuse_python_artifacts_from=reuse_python_artifacts_from,
        expected_wheel_sha256=expected_wheel_sha256,
    )
    wheel_report, requirements = _wheel_metadata(wheel)
    sdist_report = _sdist_metadata(sdist)
    constraints = root / "requirements" / "constraints-community.txt"
    kaggle_runtime_lock = root / "requirements" / "community-kaggle-runtime.lock"
    clean_install = _clean_install(wheel, constraints=constraints)
    inventory = list(clean_install.pop("dependency_inventory", []))
    sbom = _sbom(inventory, requirements)
    sbom_path = output / "community_sbom.cdx.json"
    sbom_path.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dependency_path = output / "community_dependency_license_inventory.json"
    dependency_report = {
        "schema": "crowdtensor_community_dependency_license_inventory_v1",
        "project_license": "Apache-2.0",
        "dependencies": inventory,
        "unknown_license_count": sum(str(item.get("license") or "UNKNOWN").upper() == "UNKNOWN" for item in inventory),
        "unknown_licenses_require_human_review": True,
        "model_weights_redistributed": False,
        "public_artifact_safe": True,
    }
    dependency_path.write_text(json.dumps(dependency_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    container = _container_build(root, enabled=build_container)
    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    ).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True).stdout.strip())
    manifest = {
        "schema": SCHEMA,
        "release_name": "CrowdTensor Community Maturity RC",
        "release_level": "community-rc",
        "versions": public_version(),
        "source_revision_hash": _hash(source_revision.encode()) if source_revision else "",
        "source_worktree_dirty": dirty,
        "artifacts": {
            "wheel": _artifact(wheel, kind="python-wheel"),
            "sdist": _artifact(sdist, kind="python-sdist"),
            "sbom": _artifact(sbom_path, kind="cyclonedx-sbom"),
            "dependency_inventory": _artifact(dependency_path, kind="dependency-license-inventory"),
            "constraints": _artifact(constraints, kind="dependency-constraints"),
            "kaggle_runtime_lock": _artifact(
                kaggle_runtime_lock, kind="kaggle-runtime-exact-lock"
            ),
        },
        "wheel_contract": wheel_report,
        "sdist_contract": sdist_report,
        "python_artifact_provenance": python_artifact_provenance,
        "clean_install": clean_install,
        "container": container,
        "model_adapter_registry": adapter_registry_report(),
        "license_audit": {
            "project_spdx": "Apache-2.0",
            "license_file_present": (root / "LICENSE").is_file(),
            "smollm_model_license": "apache-2.0",
            "model_weights_redistributed": False,
            "dependency_unknowns_recorded": True,
        },
        "signing": {
            "required_for_rc": False,
            "supported_methods": ["gpg-detached", "sigstore"],
            "private_signing_keys_in_bundle": False,
        },
        "publishing": {
            "pypi_uploaded": False,
            "github_release_created": False,
            "container_registry_pushed": False,
        },
        "documentation": {
            "quickstart": "docs/community-quickstart.md",
            "architecture": "docs/community-architecture.md",
            "threat_model": "docs/threat-model.md",
            "providers": "docs/providers.md",
            "model_adapters": "docs/model-adapters.md",
            "benchmarks": "docs/benchmarks.md",
            "compatibility": "docs/compatibility-matrix.md",
            "governance": "docs/governance.md",
            "release_checklist": "docs/community-release.md",
            "license_audit": "docs/license-audit.md",
        },
        "public_artifact_safe": True,
    }
    manifest["content_hash"] = stable_hash(manifest)
    manifest_path = output / "community_release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    docs = [root / value for value in manifest["documentation"].values()]
    bundle = output / f"crowdtensord-{__version__}-community-rc.tar.gz"
    bundle_files = [
        wheel,
        sdist,
        sbom_path,
        dependency_path,
        manifest_path,
        constraints,
        kaggle_runtime_lock,
        root / "LICENSE",
        root / "README.md",
        *docs,
    ]
    _deterministic_bundle(bundle, bundle_files, base=root)
    bundle_hash = {
        "schema": "crowdtensor_community_release_bundle_hash_v1",
        "file_name": bundle.name,
        "sha256": _hash(bundle),
        "size_bytes": bundle.stat().st_size,
        "signature_present": False,
        "public_artifact_safe": True,
    }
    bundle_hash_path = output / "community_release_bundle_hash.json"
    bundle_hash_path.write_text(json.dumps(bundle_hash, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    public_files = [manifest_path, sbom_path, dependency_path, bundle_hash_path, *docs]
    privacy = scan_public_files(public_files)
    report = {
        "schema": REPORT_SCHEMA,
        "ok": bool(
            wheel_report["required_modules_present"]
            and wheel_report["license_file_present"]
            and wheel_report["entry_points_present"]
            and wheel_report["package_version"] == __version__
            and sdist_report["required_sources_present"]
            and not sdist_report["absolute_members_present"]
            and clean_install["ok"]
            and container["compose_config_valid"]
            and (not build_container or (container["image_built"] and container["image_removed"]))
            and privacy["ok"]
        ),
        "release_manifest": manifest,
        "bundle": bundle_hash,
        "public_safety": privacy,
        "temporary_build_environment_removed": True,
        "container_left_running": False,
        "external_publish_performed": False,
        "private_paths_public": False,
        "public_artifact_safe": privacy["ok"] is True,
    }
    report["content_hash"] = stable_hash(report)
    report_path = output / "community_release_build.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--build-python", default=sys.executable)
    parser.add_argument("--build-container", action="store_true")
    parser.add_argument("--reuse-python-artifacts-from")
    parser.add_argument("--expected-wheel-sha256", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_release(
        args.output_dir,
        build_python=args.build_python,
        build_container=args.build_container,
        reuse_python_artifacts_from=args.reuse_python_artifacts_from,
        expected_wheel_sha256=args.expected_wheel_sha256,
    )
    print(json.dumps(report, sort_keys=True) if args.json else f"release_build_ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
