#!/usr/bin/env python3
"""Exercise the authenticated Qwen Training Beta HTTP surface locally."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from crowdtensor.qwen15b_training import MODEL_ID, MODEL_REVISION  # noqa: E402
from crowdtensor.training_qwen15b_beta_service import (  # noqa: E402
    TrainingBetaController,
    TrainingBetaJobStore,
    create_training_beta_app,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    token = "private-service-token"
    headers = {"x-crowdtensor-training-token": token}
    with tempfile.TemporaryDirectory(prefix="ct-qwen15b-beta-service-smoke-") as root_text:
        root = Path(root_text)
        store_path = root / "private" / "jobs.sqlite3"
        store = TrainingBetaJobStore(store_path, max_queue_size=2)
        controller = TrainingBetaController(store, runner=lambda _request: {})
        client = TestClient(create_training_beta_app(controller, token=token))
        health_ok = client.get("/health").status_code == 200
        request = {
            "model": MODEL_ID,
            "topology": "kaggle-2x-t4x2",
            "steps": 8,
            "job_dir": str(root / "job-one"),
            "idempotency_key": "service-smoke-one",
            "kaggle_token_files": [str(root / "private-token-file")],
            "execute": False,
        }
        unauthorized_status = client.post("/v1/training/jobs", json=request).status_code
        submitted = client.post(
            "/v1/training/jobs", json=request, headers=headers
        ).json()
        repeated = client.post(
            "/v1/training/jobs", json=request, headers=headers
        ).json()
        job_id = str(submitted.get("job_id") or "")
        submit_idempotent = bool(job_id and repeated.get("job_id") == job_id)
        status_response = client.get(
            f"/v1/training/jobs/{job_id}", headers=headers
        )
        status_route_ok = status_response.status_code == 200

        claimed = store.claim_next(worker_id="service-smoke", lease_seconds=1)
        step4 = {
            **dict(claimed["public"]),
            "overall_state": "blocked",
            "current_phase": "recovery",
            "global_step": 4,
            "retry_count": 1,
            "blockers": ["service_smoke_injected_restart"],
        }
        store.update_status(job_id, step4, event_id="service-smoke-step4")
        restarted_store = TrainingBetaJobStore(store_path, max_queue_size=2)
        restarted_controller = TrainingBetaController(restarted_store, runner=lambda _request: {})
        restarted_client = TestClient(
            create_training_beta_app(restarted_controller, token=token)
        )
        recovered_status = restarted_client.get(
            f"/v1/training/jobs/{job_id}", headers=headers
        ).json()
        process_restart_recovery_verified = bool(
            recovered_status.get("global_step") == 4
            and recovered_status.get("current_phase") == "recovery"
        )
        resumed = restarted_client.post(
            f"/v1/training/jobs/{job_id}/resume",
            headers=headers,
            json={"execute": False},
        ).json()
        resume_route_ok = resumed.get("overall_state") == "recovery_required"

        adapter = root / "job-one" / "exported_adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"service-smoke-adapter")
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": MODEL_ID}), encoding="utf-8"
        )
        export_response = restarted_client.post(
            f"/v1/training/jobs/{job_id}/export",
            headers=headers,
            json={"output_dir": str(root / "export")},
        )
        export_route_ok = bool(
            export_response.status_code == 200
            and export_response.json().get("standard_peft_layout") is True
        )
        artifacts_route_ok = (
            restarted_client.get(
                f"/v1/training/jobs/{job_id}/artifacts", headers=headers
            ).status_code
            == 200
        )
        events = restarted_client.get(
            f"/v1/training/jobs/{job_id}/events", headers=headers
        ).json()
        events_route_ok = bool(events and all(item.get("public_artifact_safe") for item in events))

        second_request = {
            **request,
            "job_dir": str(root / "job-two"),
            "idempotency_key": "service-smoke-two",
        }
        second = restarted_client.post(
            "/v1/training/jobs", json=second_request, headers=headers
        ).json()
        cancelled = restarted_client.post(
            f"/v1/training/jobs/{second['job_id']}/cancel", headers=headers
        ).json()
        cancelled_again = restarted_client.post(
            f"/v1/training/jobs/{second['job_id']}/cancel", headers=headers
        ).json()
        cancel_idempotent = bool(
            cancelled.get("overall_state") == "cancelled"
            and cancelled_again.get("overall_state") == "cancelled"
            and cancelled_again.get("revision") == cancelled.get("revision")
            and len(
                [
                    item
                    for item in restarted_store.events(second["job_id"])
                    if item.get("event_id") == "cancel-requested"
                ]
            )
            == 1
        )
        third_request = {
            **request,
            "job_dir": str(root / "job-three"),
            "idempotency_key": "service-smoke-three",
        }
        third = restarted_client.post(
            "/v1/training/jobs", json=third_request, headers=headers
        ).json()
        restarted_store.claim_next(
            worker_id="service-smoke-running-cancel",
            preferred_job_id=third["job_id"],
        )
        running_cancelled = restarted_client.post(
            f"/v1/training/jobs/{third['job_id']}/cancel", headers=headers
        ).json()
        running_cancelled_again = restarted_client.post(
            f"/v1/training/jobs/{third['job_id']}/cancel", headers=headers
        ).json()
        cancel_marker = root / "job-three" / ".private-service" / "cancel.requested"
        running_cancel_marker_ready = bool(
            running_cancelled.get("overall_state") == "running"
            and running_cancelled.get("cancel_requested") is True
            and running_cancelled_again.get("revision") == running_cancelled.get("revision")
            and cancel_marker.is_file()
            and cancel_marker.stat().st_mode & 0o777 == 0o600
        )
        cleanup_response = restarted_client.post(
            f"/v1/training/jobs/{job_id}/cleanup", headers=headers
        )
        cleanup_repeated = restarted_client.post(
            f"/v1/training/jobs/{job_id}/cleanup", headers=headers
        )
        cleanup_route_ok = bool(
            cleanup_response.status_code == 200
            and cleanup_response.json().get("command_ok") is True
            and cleanup_repeated.status_code == 200
            and cleanup_repeated.json().get("command_ok") is True
            and cleanup_repeated.json().get("overall_state") == "cleaned"
            and cleanup_repeated.json().get("revision")
            == cleanup_response.json().get("revision")
        )
        encoded_responses = json.dumps(
            {
                "submitted": submitted,
                "status": recovered_status,
                "resumed": resumed,
                "cancelled": cancelled,
                "cleanup": cleanup_response.json(),
            },
            sort_keys=True,
        )
        private_inputs_redacted = bool(
            token not in encoded_responses
            and str(root) not in encoded_responses
            and "private-token-file" not in encoded_responses
        )
        report = {
            "schema": "crowdtensor_training_qwen15b_beta_service_smoke_v1",
            "ok": all(
                (
                    health_ok,
                    unauthorized_status == 401,
                    submit_idempotent,
                    status_route_ok,
                    process_restart_recovery_verified,
                    resume_route_ok,
                    export_route_ok,
                    artifacts_route_ok,
                    events_route_ok,
                    cancel_idempotent,
                    running_cancel_marker_ready,
                    cleanup_route_ok,
                    private_inputs_redacted,
                )
            ),
            "health_route_ready": health_ok,
            "authentication_required": unauthorized_status == 401,
            "submit_route_ready": bool(job_id),
            "submit_idempotent": submit_idempotent,
            "status_route_ready": status_route_ok,
            "resume_route_ready": resume_route_ok,
            "cancel_route_ready": cancel_idempotent,
            "running_cancel_marker_ready": running_cancel_marker_ready,
            "export_route_ready": export_route_ok,
            "cleanup_route_ready": cleanup_route_ok,
            "artifacts_route_ready": artifacts_route_ok,
            "events_route_ready": events_route_ok,
            "persistent_process_restart_recovery_verified": process_restart_recovery_verified,
            "recovered_global_step": int(recovered_status.get("global_step") or 0),
            "bounded_queue_ready": restarted_store.summary().get("max_queue_size") == 2,
            "one_live_gpu_job_enforced": restarted_store.summary().get("one_live_gpu_job") is True,
            "private_inputs_redacted": private_inputs_redacted,
            "live_gpu_run_performed": False,
            "credential_values_public": False,
            "credential_paths_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
    _write(output / "training_qwen15b_beta_service_smoke.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build(args.output_dir)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"training_qwen15b_beta_service_smoke ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
