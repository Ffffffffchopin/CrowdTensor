from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import kaggle_web_tpu_active_event_probe as probe


def test_parse_active_events_from_body_detects_queued_tpu_event() -> None:
    body = "\n".join(
        [
            "Session is starting...",
            "notebook8d4184babd",
            "Interactive Session with TPU v5e-8",
            "Queued",
            "an hour",
            "1 Active Event",
        ]
    )

    events = probe.parse_active_events_from_body(body)

    assert len(events) == 1
    assert events[0]["accelerator_public"] == "TPU v5e-8"
    assert events[0]["queued"] is True
    assert events[0]["running"] is False
    assert events[0]["event_title_public"] is False
    assert events[0]["event_title_hash"].startswith("sha256:")


def test_build_report_records_queued_active_event_as_not_ready(tmp_path: Path) -> None:
    args = argparse.Namespace(output_dir=str(tmp_path))
    body = "\n".join(
        [
            "notebook8d4184babd",
            "Interactive Session with TPU v5e-8",
            "Queued",
            "an hour",
            "1 Active Event",
        ]
    )

    report = probe.build_report(
        args,
        initial_observation={"body_text": "View Active Events", "frames": []},
        final_observation={"body_text": body, "frames": []},
        steps=[{"name": "open_active_events_dialog", "ok": True}],
        output_dir=tmp_path,
    )

    assert report["active_event_probe_ready"] is True
    assert report["active_event_count"] == 1
    assert report["tpu_v5e_active_event_visible"] is True
    assert report["active_event_queued"] is True
    assert report["active_event_runtime_ready"] is False
    assert "kaggle_web_tpu_active_event_queued" in report["blocker_codes"]
    assert "kaggle_web_tpu_jupyter_frame_not_visible" in report["blocker_codes"]
    assert probe.public_redaction_errors(report) == []


def test_build_report_accepts_running_event_with_jupyter_session(tmp_path: Path) -> None:
    args = argparse.Namespace(output_dir=str(tmp_path))
    body = "\n".join(
        [
            "notebook8d4184babd",
            "Interactive Session with TPU v5e-8",
            "Running",
            "a minute",
            "1 Active Event",
        ]
    )

    report = probe.build_report(
        args,
        initial_observation={"body_text": "View Active Events", "frames": []},
        final_observation={
            "body_text": body,
            "frames": [{"has_jupyterapp": True, "session_count": 1, "kernel_count": 1}],
        },
        steps=[
            {"name": "open_active_events_dialog", "ok": True},
            {"name": "open_running_active_event", "ok": True},
        ],
        output_dir=tmp_path,
    )

    assert report["active_event_running"] is True
    assert report["active_event_runtime_ready"] is True
    assert report["blocked_reason"] == ""
    assert report["active_event_opened"] is True
    assert probe.public_redaction_errors(report) == []


def test_parse_active_events_accepts_running_with_duration() -> None:
    body = "\n".join(
        [
            "notebook8d4184babd",
            "Interactive Session with TPU v5e-8",
            "Running: 14 minutes",
            "an hour",
            "1 Active Event",
        ]
    )

    events = probe.parse_active_events_from_body(body)

    assert len(events) == 1
    assert events[0]["running"] is True
    assert events[0]["queued"] is False
