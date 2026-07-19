#!/usr/bin/env python3
"""Verify core and Model Adapter wheels in an isolated no-dependency venv."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.model_adapter import stable_hash


SCHEMA = "crowdtensor_model_adapter_plugin_install_smoke_v1"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run_smoke(
    *,
    core_wheel: str | Path,
    adapter_wheel: str | Path,
    output_dir: str | Path,
    adapter_id: str = "mistral_lora_v1",
) -> dict[str, Any]:
    core = Path(core_wheel).expanduser().resolve()
    plugin = Path(adapter_wheel).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not core.is_file() or not plugin.is_file():
        raise ValueError("model_adapter_plugin_smoke_wheel_missing")
    report: dict[str, Any]
    try:
        with tempfile.TemporaryDirectory(prefix="ct-adapter-smoke-") as temporary:
            root = Path(temporary)
            venv.EnvBuilder(with_pip=True, clear=True).create(root)
            python = _python(root)
            clean_env = dict(os.environ)
            clean_env.pop("PYTHONPATH", None)
            clean_env["PYTHONNOUSERSITE"] = "1"
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-deps",
                    "--force-reinstall",
                    str(core),
                    str(plugin),
                ],
                env=clean_env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=180,
            )
            code = f'''import json
import pathlib
from crowdtensor.model_adapter import adapter_registry_report, check_model_adapter_conformance, get_model_adapter, get_model_adapter_registration
adapter = get_model_adapter({adapter_id!r})
registry = adapter_registry_report()
registration = get_model_adapter_registration({adapter_id!r})
conformance = check_model_adapter_conformance(adapter)
print(json.dumps({{
    "adapter_id": adapter.adapter_id,
    "family": adapter.family,
    "model_id": adapter.default_model_id,
    "model_revision": adapter.default_revision,
    "registry": registry,
    "registration": registration,
    "conformance": conformance,
    "workspace_import_used": False,
}}))
'''
            completed = subprocess.run(
                [str(python), "-I", "-c", code],
                env=clean_env,
                check=True,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
            )
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        registration = dict(payload.get("registration") or {})
        conformance = dict(payload.get("conformance") or {})
        registry = dict(payload.get("registry") or {})
        report = {
            "schema": SCHEMA,
            "ok": bool(
                payload.get("adapter_id") == adapter_id
                and registration.get("kind") == "entry_point_plugin"
                and conformance.get("ok") is True
                and int(registry.get("plugin_adapter_count") or 0) >= 1
            ),
            "adapter_id": str(payload.get("adapter_id") or ""),
            "family": str(payload.get("family") or ""),
            "model_id": str(payload.get("model_id") or ""),
            "model_revision": str(payload.get("model_revision") or ""),
            "entry_point_group": str(registry.get("entry_point_group") or ""),
            "plugin_adapter_count": int(registry.get("plugin_adapter_count") or 0),
            "supported_model_families": sorted(registry.get("supported_model_families") or []),
            "registration_kind": str(registration.get("kind") or ""),
            "distribution_name": str(registration.get("distribution_name") or ""),
            "distribution_version": str(registration.get("distribution_version") or ""),
            "conformance_verified": conformance.get("ok") is True,
            "canonical_config_verified": conformance.get("canonical_config_verified") is True,
            "partition_verified": conformance.get("partition_verified") is True,
            "core_wheel": {"file_name": core.name, "sha256": _hash(core)},
            "adapter_wheel": {"file_name": plugin.name, "sha256": _hash(plugin)},
            "isolated_venv": True,
            "dependencies_installed": False,
            "wheel_install_no_deps": True,
            "workspace_import_used": False,
            "installed_locations_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
            "blockers": [],
        }
    except BaseException as exc:
        report = {
            "schema": SCHEMA,
            "ok": False,
            "adapter_id": adapter_id,
            "core_wheel": {"file_name": core.name, "sha256": _hash(core)},
            "adapter_wheel": {"file_name": plugin.name, "sha256": _hash(plugin)},
            "isolated_venv": True,
            "dependencies_installed": False,
            "wheel_install_no_deps": True,
            "workspace_import_used": False,
            "installed_locations_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
            "blockers": ["model_adapter_plugin_smoke_failed:" + type(exc).__name__],
        }
    safety = scan_public_value(report)
    report["public_safety"] = safety
    report["public_artifact_safe"] = safety["ok"] is True
    report["ok"] = bool(report.get("ok") and safety["ok"])
    report["content_hash"] = stable_hash(report)
    (output / "model_adapter_plugin_smoke.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-wheel", required=True)
    parser.add_argument("--adapter-wheel", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-id", default="mistral_lora_v1")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_smoke(
        core_wheel=args.core_wheel,
        adapter_wheel=args.adapter_wheel,
        output_dir=args.output_dir,
        adapter_id=args.adapter_id,
    )
    print(json.dumps(report, sort_keys=True) if args.json else f"ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
