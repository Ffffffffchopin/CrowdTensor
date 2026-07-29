#!/usr/bin/env python3
"""Pack the reproducible One-Click Contributor Beta release candidate."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_files, scan_public_value
from crowdtensor.model_adapter import stable_hash
from crowdtensor.version import __version__


SCHEMA = "crowdtensor_one_click_contributor_release_rc_v1"
MANIFEST_SCHEMA = "crowdtensor_one_click_contributor_release_manifest_v1"
E2E_SCHEMA = "crowdtensor_one_click_contributor_e2e_v1"
COMMUNITY_SCHEMA = "crowdtensor_community_release_build_v1"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("one_click_release_json_object_required")
    return value


def _artifact(path: Path, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "file_name": path.name,
        "sha256": _hash(path),
        "size_bytes": path.stat().st_size,
    }


def _community_artifact(report_path: Path, report: dict[str, Any], name: str) -> Path:
    manifest = report.get("release_manifest") or {}
    item = (manifest.get("artifacts") or {}).get(name) or {}
    file_name = str(item.get("file_name") or "")
    candidates = [report_path.parent / "artifacts" / file_name, report_path.parent / file_name]
    source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None or _hash(source) != item.get("sha256"):
        raise ValueError("one_click_release_community_artifact_invalid:" + name)
    return source


def _wheel_contract(wheel: Path) -> dict[str, Any]:
    required = {
        "crowdtensor/project_site/join.html",
        "crowdtensor/project_site/join.css",
        "crowdtensor/project_site/join.js",
        "crowdtensor/project_site/join_worker.js",
        "crowdtensor/volunteer_agent_status.py",
        "crowdtensor/volunteer_browser_probe.py",
        "crowdtensor/volunteer_training_api.py",
        "crowdtensor/volunteer_training_cli.py",
    }
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    return {
        "required_one_click_files_present": required.issubset(names),
        "missing_files": sorted(required - names),
        "join_page_packaged": "crowdtensor/project_site/join.html" in names,
        "webgpu_worker_packaged": "crowdtensor/project_site/join_worker.js" in names,
        "native_status_page_packaged": "crowdtensor/volunteer_agent_status.py" in names,
    }


def _clean_install(wheel: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ct-one-click-clean-") as temporary:
        root = Path(temporary)
        venv = root / "venv"
        created = subprocess.run(
            [sys.executable, "-m", "venv", str(venv)],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
            check=False,
        )
        if created.returncode != 0:
            return {"ok": False, "phase": "venv_create", "workspace_import_used": False}
        python = venv / "bin" / "python"
        pip = venv / "bin" / "pip"
        installed = subprocess.run(
            [str(pip), "install", "--disable-pip-version-check", str(wheel)],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
            check=False,
        )
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        commands = [
            [str(venv / "bin" / "crowdtensor"), "volunteer", "contract", "--json"],
            [str(venv / "bin" / "crowdtensor"), "volunteer", "join", "--help"],
            [
                str(python),
                "-c",
                (
                    "import importlib.resources as r,json,crowdtensor,pathlib,sys;"
                    "p=r.files('crowdtensor.project_site');"
                    "print(json.dumps({'version':crowdtensor.__version__,"
                    "'join':p.joinpath('join.html').is_file(),"
                    "'worker':p.joinpath('join_worker.js').is_file(),"
                    "'under_venv':str(pathlib.Path(crowdtensor.__file__).resolve()).startswith(str(pathlib.Path(sys.prefix).resolve()))}))"
                ),
            ],
        ]
        results = [
            subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
                check=False,
            )
            for command in commands
        ]
        try:
            asset_report = json.loads(results[-1].stdout)
        except (json.JSONDecodeError, IndexError):
            asset_report = {}
        return {
            "ok": bool(
                installed.returncode == 0
                and all(result.returncode == 0 for result in results)
                and asset_report.get("version") == __version__
                and asset_report.get("join") is True
                and asset_report.get("worker") is True
                and asset_report.get("under_venv") is True
            ),
            "fresh_venv": True,
            "wheel_installed": installed.returncode == 0,
            "golden_command_count": len(results),
            "all_golden_commands_passed": all(
                result.returncode == 0 for result in results
            ),
            "installed_version": str(asset_report.get("version") or ""),
            "packaged_site_assets_loaded": bool(
                asset_report.get("join") and asset_report.get("worker")
            ),
            "workspace_import_used": asset_report.get("under_venv") is not True,
            "temporary_environment_removed_after_return": True,
        }


def _bundle(path: Path, files: list[Path], base: Path) -> None:
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for source in sorted(files, key=lambda item: item.name):
            data = source.read_bytes()
            info = tarfile.TarInfo(source.relative_to(base).as_posix())
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o755 if source.name.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(data))


def build_release(
    *,
    community_report_path: str | Path,
    e2e_report_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    community_path = Path(community_report_path).expanduser().resolve()
    e2e_path = Path(e2e_report_path).expanduser().resolve()
    community = _load(community_path)
    e2e = _load(e2e_path)
    if community.get("schema") != COMMUNITY_SCHEMA or community.get("ok") is not True:
        raise ValueError("one_click_release_community_rc_not_ready")
    if e2e.get("schema") != E2E_SCHEMA or e2e.get("ok") is not True:
        raise ValueError("one_click_release_e2e_not_ready")

    wheel_source = _community_artifact(community_path, community, "wheel")
    sdist_source = _community_artifact(community_path, community, "sdist")
    wheel = output / wheel_source.name
    sdist = output / sdist_source.name
    shutil.copy2(wheel_source, wheel)
    shutil.copy2(sdist_source, sdist)
    installer = output / "install-contributor.sh"
    notes = output / "RELEASE_NOTES.md"
    evidence = output / "one_click_contributor_e2e.json"
    screenshot = output / "one_click_browser_e2e.png"
    shutil.copy2(root / "scripts" / "install_contributor.sh", installer)
    installer.chmod(0o755)
    shutil.copy2(root / "docs" / "releases" / "one-click-contributor-beta-rc7.md", notes)
    shutil.copy2(e2e_path, evidence)
    screenshot_source = e2e_path.parent / str((e2e.get("browser") or {}).get("screenshot_file") or "")
    if not screenshot_source.is_file() or _hash(screenshot_source) != (
        e2e.get("browser") or {}
    ).get("screenshot_sha256"):
        raise ValueError("one_click_release_e2e_screenshot_invalid")
    shutil.copy2(screenshot_source, screenshot)

    wheel_contract = _wheel_contract(wheel)
    clean_install = _clean_install(wheel)
    community_manifest = community.get("release_manifest") or {}
    container = community_manifest.get("container") or {}
    artifacts = {
        "wheel": _artifact(wheel, "python-wheel"),
        "sdist": _artifact(sdist, "python-sdist"),
        "installer": _artifact(installer, "posix-installer"),
        "release_notes": _artifact(notes, "release-notes"),
        "e2e_report": _artifact(evidence, "one-click-e2e"),
        "browser_screenshot": _artifact(screenshot, "browser-e2e-screenshot"),
    }
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "package_version": __version__,
        "release_name": "CrowdTensor One-Click Contributor Beta RC7",
        "artifacts": artifacts,
        "wheel_contract": wheel_contract,
        "clean_install": clean_install,
        "container": {
            "image_built": container.get("image_built") is True,
            "image_removed": container.get("image_removed") is True,
            "compose_config_valid": container.get("compose_config_valid") is True,
            "published": False,
        },
        "e2e": {
            "browser_task_accepted": (e2e.get("browser") or {}).get("ok") is True,
            "browser_runtime": (e2e.get("browser") or {}).get("runtime"),
            "browser_model_update": False,
            "native_lora_update_accepted": (e2e.get("agent") or {}).get("ok") is True,
            "real_peft_lora": (e2e.get("agent") or {}).get("real_peft_lora") is True,
            "one_time_codes_persisted": False,
            "cleanup_complete": (e2e.get("cleanup") or {}).get(
                "live_resources_left_running"
            )
            is False,
        },
        "install": {
            "browser_path": "/join",
            "wheel_path": f"/downloads/{wheel.name}",
            "installer_path": "/downloads/install-contributor.sh",
            "first_contribution_maximum_steps": 3,
            "clone_required": False,
            "manual_invite_file_required": False,
            "native_agent_device_types": ["cpu", "cuda"],
            "jax_tpu_one_click": False,
            "jax_tpu_advanced_workflow": True,
            "cpu_only_torch_on_cpu_hosts": True,
            "provider_torch_index_override": True,
            "pinned_contributor_runtime": True,
            "storage_extra_installed_by_default": False,
        },
        "boundaries": {
            "controlled_enrollment": True,
            "permissionless": False,
            "browser_lora_training": False,
            "browser_large_model_sharding": False,
            "kimi_k3": False,
            "production_sla": False,
        },
        "publishing": {
            "site_release_directory_ready": True,
            "github_release_created": False,
            "pypi_uploaded": False,
            "container_registry_pushed": False,
            "github_release_channel": "tag-triggered GitHub Actions with GITHUB_TOKEN",
        },
        "credential_values_public": False,
        "pairing_code_values_public": False,
        "public_artifact_safe": True,
    }
    manifest["content_hash"] = stable_hash(manifest)
    release_json = output / "release.json"
    release_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sums_targets = [wheel, sdist, installer, notes, evidence, screenshot, release_json]
    sums = output / "SHA256SUMS"
    sums.write_text(
        "".join(f"{_hash(path).split(':', 1)[1]}  {path.name}\n" for path in sums_targets),
        encoding="ascii",
    )
    bundle = output / f"crowdtensord-{__version__}-one-click-contributor-beta.tar.gz"
    bundle_files = [*sums_targets, sums]
    _bundle(bundle, bundle_files, output)
    privacy = scan_public_files([installer, notes, evidence, release_json, sums])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": bool(
            wheel_contract["required_one_click_files_present"]
            and clean_install["ok"]
            and container.get("image_built") is True
            and container.get("image_removed") is True
            and manifest["e2e"]["browser_task_accepted"]
            and manifest["e2e"]["native_lora_update_accepted"]
            and privacy["ok"]
        ),
        "release_manifest": manifest,
        "release_json": _artifact(release_json, "release-manifest"),
        "checksums": _artifact(sums, "sha256sums"),
        "bundle": _artifact(bundle, "release-bundle"),
        "public_safety": privacy,
        "external_publish_performed": False,
        "release_bundle_ready_for_tag_publish": True,
        "public_artifact_safe": privacy["ok"] is True,
    }
    report["content_hash"] = stable_hash(report)
    report_path = output / "one_click_contributor_release_rc.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if scan_public_value(report)["ok"] is not True:
        raise RuntimeError("one_click_release_report_public_safety_failed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--community-report", required=True)
    parser.add_argument("--e2e-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_release(
        community_report_path=args.community_report,
        e2e_report_path=args.e2e_report,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, sort_keys=True) if args.json else f"one_click_release_ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
