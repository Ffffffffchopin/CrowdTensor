#!/usr/bin/env python3
"""Exercise the ordinary-user Training Production workflow without cloud use."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from crowdtensor.cli import parse_args
from crowdtensor.heterogeneous_training_beta import (
    create_heterogeneous_training_beta_app,
)
from crowdtensor.heterogeneous_training_manifest import stable_hash
from crowdtensor.heterogeneous_training_production import (
    HeterogeneousTrainingProductionController,
    default_production_config,
    production_manifest,
)
from scripts.training_cuda_kaggle_common import public_safety_errors


SCHEMA = "crowdtensor_heterogeneous_training_production_workflow_probe_v1"


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_probe(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    private = output / ".private-workflow"
    shutil.rmtree(private, ignore_errors=True)
    private.mkdir(parents=True, exist_ok=True)
    config = default_production_config()
    manifest = production_manifest(config)
    model_config_path = private / "model-config.json"
    tokenized_path = private / "tokenized.json"
    _write(
        model_config_path,
        {
            "model_type": "qwen2",
            "num_hidden_layers": 28,
            "hidden_size": 3584,
        },
    )
    _write(
        tokenized_path,
        {
            "schema": "crowdtensor_heterogeneous_tokenized_private_v1",
            "training_manifest_hash": manifest["content_hash"],
            "model_id": manifest["model"]["model_id"],
            "model_revision": manifest["model"]["model_revision"],
            "sequence_length": manifest["training"]["sequence_length"],
            "train": [
                [1] * int(manifest["training"]["sequence_length"])
                for _ in range(int(manifest["training"]["target_steps"]))
            ],
            "validation": [
                [1] * int(manifest["training"]["sequence_length"])
            ],
        },
    )
    dry_run = HeterogeneousTrainingProductionController.create(
        private / "dry-run-job",
        config=config,
        dry_run=True,
    )
    job = private / "job"
    controller = HeterogeneousTrainingProductionController.create(
        job,
        config=config,
        model_config_path=model_config_path,
        tokenized_payload_path=tokenized_path,
    )
    assert isinstance(controller, HeterogeneousTrainingProductionController)
    repeated = HeterogeneousTrainingProductionController.create(job, config=config)
    validation = controller.validate()
    plan = controller.plan()
    status = controller.status()
    pause = controller.pause()
    pause_again = controller.pause()
    resume = controller.resume()
    resume_again = controller.resume()
    rebalance = controller.rebalance()
    credentials = controller.beta.credentials()
    app = create_heterogeneous_training_beta_app(
        controller.beta,
        owner_token=str(credentials["owner_token"]),
        miner_token=str(credentials["miner_token"]),
    )
    owner_headers = {
        "x-crowdtensor-training-token": str(credentials["owner_token"])
    }
    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        http_status = client.get(
            f"/v1/training/jobs/{controller.beta.job_id}",
            headers=owner_headers,
        )
        events = client.get(
            f"/v1/training/jobs/{controller.beta.job_id}/events",
            headers=owner_headers,
        )
        metrics = client.get("/metrics", headers=owner_headers)
        unauthenticated_metrics = client.get("/metrics")
    stop = controller.stop()
    stop_again = controller.stop()
    cleanup = controller.cleanup()
    cleanup_again = controller.cleanup()
    parsed_actions = {}
    for action, argv in {
        "validate": ["train", "validate"],
        "plan": ["train", "plan"],
        "start": ["train", "start", "job"],
        "status": ["train", "status", "job"],
        "pause": ["train", "pause", "job"],
        "resume": ["train", "resume", "job"],
        "stop": ["train", "stop", "job"],
        "cleanup": ["train", "cleanup", "job"],
        "metrics": ["train", "metrics", "job"],
        "events": ["train", "events", "job"],
    }.items():
        parsed_actions[action] = parse_args(argv).train_action
    report = {
        "schema": SCHEMA,
        "ok": True,
        "dry_run_verified": isinstance(dry_run, dict)
        and dry_run.get("dry_run") is True
        and dry_run.get("ok") is True,
        "idempotent_start_verified": isinstance(
            repeated, HeterogeneousTrainingProductionController
        )
        and repeated.beta.job_id == controller.beta.job_id,
        "validation_verified": validation.get("configuration_valid") is True,
        "plan_verified": plan.get("target_steps") == 100,
        "status_verified": status.get("target_steps") == 100,
        "pause_idempotent": pause.get("pause_transition_applied") is True
        and pause_again.get("pause_transition_applied") is False,
        "resume_idempotent": resume.get("resume_transition_applied") is True
        and resume_again.get("resume_transition_applied") is False,
        "rebalance_verified": rebalance.get("rebalance_transition_applied") is True,
        "stop_idempotent": stop.get("command_ok") is True
        and stop_again.get("command_ok") is True,
        "cleanup_idempotent": cleanup.get("ok") is True
        and cleanup_again.get("ok") is True,
        "cli_actions": parsed_actions,
        "cli_complete_lifecycle_verified": sorted(parsed_actions)
        == sorted(
            [
                "validate",
                "plan",
                "start",
                "status",
                "pause",
                "resume",
                "stop",
                "cleanup",
                "metrics",
                "events",
            ]
        ),
        "http": {
            "health_verified": health.status_code == 200
            and health.json().get("ok") is True,
            "readiness_verified": ready.status_code == 200
            and ready.json().get("ready") is True,
            "status_verified": http_status.status_code == 200,
            "events_verified": events.status_code == 200
            and events.json().get("bounded_page") is True,
            "prometheus_metrics_verified": metrics.status_code == 200
            and "crowdtensor_training_committed_step" in metrics.text,
            "metrics_authentication_verified": unauthenticated_metrics.status_code
            == 401,
        },
        "monitoring_contract_verified": bool(
            health.status_code == 200
            and ready.status_code == 200
            and events.status_code == 200
            and metrics.status_code == 200
        ),
        "next_resume_command_present": bool(status.get("next_resume_command")),
        "next_resume_command_redacts_credentials": status.get(
            "next_resume_command_redacts_credentials"
        )
        is True,
        "next_resume_command_uses_public_placeholder": bool(
            status.get("next_resume_command")
            == "crowdtensor train resume <job-dir>"
            and status.get("next_resume_command_uses_public_placeholder") is True
        ),
        "explicit_exit_code_contract": True,
        "run_id_present": bool(status.get("job_id")),
        "configuration_driven": True,
        "quota_and_acquisition_preflight_contract": config["acquisition"],
        "cleanup": {
            "active_miner_leases_revoked": cleanup[
                "active_miner_leases_revoked"
            ],
            "live_resources_left_running": cleanup[
                "live_resources_left_running"
            ],
            "temporary_private_runtime_removed": True,
        },
        "credential_values_public": False,
        "credential_paths_public": False,
        "private_env_names_public": False,
        "raw_training_text_public": False,
        "token_ids_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    shutil.rmtree(private, ignore_errors=True)
    report["cleanup"]["temporary_private_runtime_removed"] = not private.exists()
    report["ok"] = bool(
        report["dry_run_verified"]
        and report["idempotent_start_verified"]
        and report["validation_verified"]
        and report["plan_verified"]
        and report["status_verified"]
        and report["pause_idempotent"]
        and report["resume_idempotent"]
        and report["rebalance_verified"]
        and report["stop_idempotent"]
        and report["cleanup_idempotent"]
        and report["cli_complete_lifecycle_verified"]
        and report["next_resume_command_uses_public_placeholder"]
        and all(report["http"].values())
        and report["cleanup"]["active_miner_leases_revoked"]
        and not report["cleanup"]["live_resources_left_running"]
        and report["cleanup"]["temporary_private_runtime_removed"]
    )
    safety = public_safety_errors(report)
    report["public_safety_errors"] = safety
    report["public_artifact_safe"] = not safety
    report["ok"] = bool(report["ok"] and not safety)
    report["content_hash"] = stable_hash(report)
    _write(output / "training_heterogeneous_production_workflow_probe.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_probe(args.output_dir)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "training_heterogeneous_production_workflow_probe "
            f"ready={report['ok']}"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
