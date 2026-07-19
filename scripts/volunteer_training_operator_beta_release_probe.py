#!/usr/bin/env python3
"""Build and clean-install the Volunteer Operator Beta release surface."""

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
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_files
from crowdtensor.training_contract import sha256_file, sha256_json
from crowdtensor.volunteer_training_protocol import with_public_safety


SCHEMA = "crowdtensor_volunteer_operator_beta_release_probe_v1"
REQUIRED_MODULES = {
    "crowdtensor/miner_invite.py",
    "crowdtensor/volunteer_training_api.py",
    "crowdtensor/volunteer_training_campaign.py",
    "crowdtensor/volunteer_training_cell.py",
    "crowdtensor/volunteer_training_cli.py",
    "crowdtensor/volunteer_training_coordinator.py",
    "crowdtensor/volunteer_training_operator.py",
    "crowdtensor/volunteer_training_protocol.py",
    "crowdtensor/volunteer_training_storage.py",
}


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: float = 900.0,
) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout,
    )
    return {
        "ok": process.returncode == 0,
        "return_code": int(process.returncode),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "output_hash": _hash_text(process.stdout or ""),
    }


def _wheel_contract(wheel: Path) -> dict[str, Any]:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = email.parser.Parser().parsestr(
            archive.read(metadata_name).decode("utf-8")
        )
        requirements = list(metadata.get_all("Requires-Dist") or [])
    return with_public_safety(
        {
            "package_name": str(metadata.get("Name") or ""),
            "package_version": str(metadata.get("Version") or ""),
            "required_modules_present": REQUIRED_MODULES.issubset(names),
            "missing_required_module_count": len(REQUIRED_MODULES - names),
            "volunteer_cli_entry_point_present": any(
                name.endswith(".dist-info/entry_points.txt") for name in names
            ),
            "storage_extra_declared": any(
                'extra == "storage"' in item for item in requirements
            ),
            "hf_extra_declared": any('extra == "hf"' in item for item in requirements),
        }
    )


def run_release_probe(output_dir: Path) -> dict[str, Any]:
    root = Path.cwd().resolve()
    output = output_dir.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    image_tag = "crowdtensor-volunteer-operator-beta:" + secrets.token_hex(6)
    image_removed = False
    container_left_running = False
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="ct-volunteer-release-") as temporary:
        private = Path(temporary)
        wheel_dir = private / "wheel"
        wheel_dir.mkdir()
        build = _run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--disable-pip-version-check",
                "--no-deps",
                "--wheel-dir",
                str(wheel_dir),
                ".",
            ],
            cwd=root,
            timeout=600,
        )
        wheels = list(wheel_dir.glob("*.whl"))
        if not build["ok"] or len(wheels) != 1:
            raise RuntimeError("volunteer_operator_wheel_build_failed")
        wheel = wheels[0]
        wheel_contract = _wheel_contract(wheel)
        retained_wheel = output / wheel.name
        shutil.copyfile(wheel, retained_wheel)
        retained_wheel.chmod(0o644)

        venv = private / "venv"
        venv_create = _run(
            [sys.executable, "-m", "venv", str(venv)], cwd=private, timeout=180
        )
        pip = venv / "bin" / "pip"
        cli = venv / "bin" / "crowdtensor"
        install = _run(
            [
                str(pip),
                "install",
                "--disable-pip-version-check",
                str(wheel),
            ],
            cwd=private,
            timeout=600,
        )
        clean_env = dict(os.environ)
        clean_env.pop("PYTHONPATH", None)
        contract = _run(
            [str(cli), "volunteer", "contract", "--json"],
            cwd=private,
            env=clean_env,
            timeout=60,
        )
        help_probe = _run(
            [str(cli), "volunteer", "operator", "--help"],
            cwd=private,
            env=clean_env,
            timeout=60,
        )
        pip_check = _run([str(pip), "check"], cwd=private, env=clean_env, timeout=60)
        location_process = subprocess.run(
            [
                str(venv / "bin" / "python"),
                "-c",
                (
                    "import crowdtensor,json,pathlib,sys;"
                    "print(json.dumps({'under_venv':str(pathlib.Path(crowdtensor.__file__).resolve()).startswith(str(pathlib.Path(sys.prefix).resolve()))}))"
                ),
            ],
            cwd=private,
            env=clean_env,
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            location = json.loads(location_process.stdout)
        except json.JSONDecodeError:
            location = {}
        clean_install = with_public_safety(
            {
                "ok": bool(
                    venv_create["ok"]
                    and install["ok"]
                    and contract["ok"]
                    and help_probe["ok"]
                    and pip_check["ok"]
                    and location.get("under_venv") is True
                ),
                "fresh_isolated_venv": venv_create["ok"],
                "wheel_installed_with_declared_base_dependencies": install["ok"],
                "workspace_pythonpath_removed": "PYTHONPATH" not in clean_env,
                "installed_module_under_venv": location.get("under_venv") is True,
                "volunteer_contract_command_verified": contract["ok"],
                "one_command_operator_help_verified": help_probe["ok"],
                "dependency_check_verified": pip_check["ok"],
                "hf_and_storage_runtime_extras_executed": False,
                "temporary_venv_removed_after_probe": True,
            }
        )

        try:
            compose = _run(
                ["docker", "compose", "config", "--quiet"], cwd=root, timeout=120
            )
            image_build = _run(
                ["docker", "build", "--tag", image_tag, "."],
                cwd=root,
                timeout=1200,
            )
            inspect = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    image_tag,
                    "--format",
                    "{{.Id}}|{{.Config.User}}",
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            image_id, _, configured_user = inspect.stdout.strip().partition("|")
            container_contract = _run(
                [
                    "docker",
                    "run",
                    "--rm",
                    image_tag,
                    "crowdtensor",
                    "volunteer",
                    "contract",
                    "--json",
                ],
                cwd=root,
                timeout=120,
            )
            container_report = with_public_safety(
                {
                    "ok": bool(
                        compose["ok"]
                        and image_build["ok"]
                        and image_id
                        and configured_user == "crowdtensor"
                        and container_contract["ok"]
                    ),
                    "compose_configuration_valid": compose["ok"],
                    "project_image_built_from_current_source": image_build["ok"],
                    "image_id_hash": sha256_json({"image_id": image_id})
                    if image_id
                    else "",
                    "non_root_configured_user_verified": configured_user == "crowdtensor",
                    "volunteer_contract_inside_container_verified": container_contract["ok"],
                    "container_registry_publish_performed": False,
                }
            )
        finally:
            removed = _run(
                ["docker", "image", "rm", "--force", image_tag],
                cwd=root,
                timeout=180,
            )
            image_removed = removed["ok"]
            live = subprocess.run(
                ["docker", "ps", "--quiet", "--filter", f"ancestor={image_tag}"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            container_left_running = bool(live.stdout.strip())

    report = with_public_safety(
        {
            "schema": SCHEMA,
            "ok": bool(
                build["ok"]
                and wheel_contract["required_modules_present"]
                and wheel_contract["volunteer_cli_entry_point_present"]
                and wheel_contract["storage_extra_declared"]
                and wheel_contract["hf_extra_declared"]
                and clean_install["ok"]
                and container_report["ok"]
                and image_removed
                and not container_left_running
            ),
            "wheel": {
                "file_name": retained_wheel.name,
                "sha256": sha256_file(retained_wheel),
                "byte_count": retained_wheel.stat().st_size,
                "build_from_current_source": build["ok"],
                "contract": wheel_contract,
            },
            "clean_install": clean_install,
            "container": {
                **container_report,
                "image_removed": image_removed,
                "container_left_running": container_left_running,
            },
            "external_publish_performed": False,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
    )
    report["content_hash"] = sha256_json(report)
    report_path = output / "volunteer_training_operator_beta_release_probe.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    scan = scan_public_files([report_path])
    report["public_artifact_scan"] = scan
    report["public_artifact_scan_ok"] = scan.get("ok") is True
    report["content_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_release_probe(Path(args.output_dir))
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"operator_beta_release_ok={report['ok']}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
