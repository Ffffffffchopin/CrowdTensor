#!/usr/bin/env python3
"""Run the public Model Adapter conformance contract for one installed adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.model_adapter import (
    MODEL_ADAPTER_ENTRY_POINT_GROUP,
    check_model_adapter_conformance,
    get_model_adapter,
    get_model_adapter_registration,
    stable_hash,
)


SCHEMA = "crowdtensor_model_adapter_conformance_check_v1"


def check(
    adapter_id: str,
    *,
    require_plugin: bool = False,
    expected_distribution: str = "",
) -> dict[str, Any]:
    adapter = get_model_adapter(adapter_id)
    conformance = check_model_adapter_conformance(adapter)
    registration = get_model_adapter_registration(adapter_id)
    errors = list(conformance.get("errors") or [])
    if require_plugin and registration.get("kind") != "entry_point_plugin":
        errors.append("model_adapter_conformance_entry_point_plugin_required")
    if expected_distribution and registration.get("distribution_name") != expected_distribution:
        errors.append("model_adapter_conformance_distribution_mismatch")
    if registration.get("kind") == "entry_point_plugin" and (
        registration.get("entry_point_group") != MODEL_ADAPTER_ENTRY_POINT_GROUP
        or registration.get("entry_point_name") != adapter.adapter_id
    ):
        errors.append("model_adapter_conformance_entry_point_registration_invalid")
    report = {
        "schema": SCHEMA,
        "ok": not errors,
        "adapter_id": adapter.adapter_id,
        "family": adapter.family,
        "model_id": adapter.default_model_id,
        "model_revision": adapter.default_revision,
        "model_license": adapter.default_model_license,
        "registration": registration,
        "conformance": conformance,
        "errors": sorted(set(errors)),
        "plugin_code_is_trusted_installed_code": True,
        "real_weight_live_required_separately": True,
        "credential_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    privacy = scan_public_value(report)
    report["public_safety"] = privacy
    report["public_artifact_safe"] = privacy["ok"] is True
    report["ok"] = bool(report["ok"] and privacy["ok"])
    report["content_hash"] = stable_hash(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--require-plugin", action="store_true")
    parser.add_argument("--expected-distribution", default="")
    parser.add_argument("--output-dir")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = check(
        args.adapter_id,
        require_plugin=args.require_plugin,
        expected_distribution=args.expected_distribution,
    )
    if args.output_dir:
        output = Path(args.output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "model_adapter_conformance.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, sort_keys=True) if args.json else f"ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
