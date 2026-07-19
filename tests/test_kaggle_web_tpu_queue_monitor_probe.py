from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import kaggle_web_tpu_queue_monitor_check as check
from scripts import kaggle_web_tpu_queue_monitor_probe as probe


def _args(tmp_path: Path, *, read_only: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=str(tmp_path),
        wait_seconds=60.0,
        poll_seconds=10.0,
        read_only=read_only,
        force_start_click=False,
        stop_after_session_started_polls=0,
    )


def test_parse_queue_prompt_extracts_position() -> None:
    parsed = probe.parse_queue_prompt(
        "TPUs are popular right now. You are #38 in the queue. You can wait, try connecting again later, or use another accelerator."
    )

    assert parsed["queue_prompt_visible"] is True
    assert parsed["queue_position"] == 38
    assert parsed["queue_position_visible"] is True
    assert parsed["queue_prompt_hash"].startswith("sha256:")


def test_parse_queue_prompt_handles_line_breaks() -> None:
    parsed = probe.parse_queue_prompt("TPUs are popular right now.\nYou are #\n6\nin the queue.\nYou can wait.")

    assert parsed["queue_position"] == 6


def test_queue_progress_detects_position_change() -> None:
    progress = probe.queue_progress([
        {"queue_position": 9},
        {"queue_position": 7},
        {"queue_position": 7},
    ])

    assert progress["queue_position_observed"] is True
    assert progress["queue_position_changed"] is True
    assert progress["queue_position_decreased"] is True
    assert progress["first_queue_position"] == 9
    assert progress["last_queue_position"] == 7


def test_build_report_records_static_queue_position(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        output_dir=tmp_path,
        steps=[{"name": "click_start_session", "ok": True}],
        observations=[
            {
                "label": "poll_1",
                "elapsed_seconds": 10.0,
                "body_text_hash": "sha256:a",
                "queue_prompt_visible": True,
                "queue_position": 19,
                "queue_position_visible": True,
                "queue_prompt_hash": "sha256:q",
                "active_event_queued": True,
                "session_starting_text_visible": True,
                "jupyter_frame_visible": False,
                "jupyter_session_or_kernel_visible": False,
                "web_tpu_runtime_ready": False,
            },
            {
                "label": "poll_2",
                "elapsed_seconds": 20.0,
                "body_text_hash": "sha256:b",
                "queue_prompt_visible": True,
                "queue_position": 19,
                "queue_position_visible": True,
                "queue_prompt_hash": "sha256:q",
                "active_event_queued": True,
                "session_starting_text_visible": True,
                "jupyter_frame_visible": False,
                "jupyter_session_or_kernel_visible": False,
                "web_tpu_runtime_ready": False,
            },
        ],
    )

    assert report["start_clicked"] is True
    assert report["queue_progress"]["queue_position_observed"] is True
    assert report["queue_progress"]["queue_position_changed"] is False
    assert "kaggle_web_tpu_queue_position_static" in report["blocker_codes"]
    assert check.validate(report) == []


def test_build_report_accepts_ready_runtime(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        output_dir=tmp_path,
        steps=[{"name": "open_running_active_event", "ok": True}],
        observations=[
            {
                "label": "poll_1",
                "elapsed_seconds": 10.0,
                "body_text_hash": "sha256:a",
                "queue_prompt_visible": False,
                "queue_position": None,
                "queue_position_visible": False,
                "active_event_running": True,
                "jupyter_frame_visible": True,
                "jupyter_session_or_kernel_visible": True,
                "web_tpu_runtime_ready": True,
            }
        ],
    )

    assert report["web_tpu_runtime_ready"] is True
    assert report["blocked_reason"] == ""
    assert check.validate(report, require_ready=True) == []


def test_build_report_records_session_started_early_stop(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.stop_after_session_started_polls = 2
    report = probe.build_report(
        args,
        output_dir=tmp_path,
        steps=[{"name": "stop_after_session_started_stable", "ok": True, "polls": 2}],
        observations=[
            {
                "label": "poll_2",
                "elapsed_seconds": 20.0,
                "body_text_hash": "sha256:a",
                "queue_prompt_visible": False,
                "queue_position": None,
                "queue_position_visible": False,
                "session_started_text_visible": True,
                "session_starting_text_visible": False,
                "active_event_queued": False,
                "jupyter_frame_visible": False,
                "jupyter_session_or_kernel_visible": False,
                "web_tpu_runtime_ready": False,
            }
        ],
    )

    assert report["stop_after_session_started_polls"] == 2
    assert report["session_started_early_stop_triggered"] is True
    assert "kaggle_web_tpu_session_started_text_without_runtime" in report["blocker_codes"]
    assert "kaggle_web_tpu_jupyter_frame_not_visible" in report["blocker_codes"]
    assert check.validate(report) == []


def test_build_report_treats_final_running_event_as_not_queued(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        output_dir=tmp_path,
        steps=[{"name": "click_start_session", "ok": True}],
        observations=[
            {
                "label": "poll_1",
                "elapsed_seconds": 10.0,
                "body_text_hash": "sha256:a",
                "queue_prompt_visible": True,
                "queue_position": 18,
                "queue_position_visible": True,
                "queue_prompt_hash": "sha256:q",
                "active_event_queued": True,
                "active_event_running": False,
                "session_starting_text_visible": True,
                "jupyter_frame_visible": False,
                "jupyter_session_or_kernel_visible": False,
                "web_tpu_runtime_ready": False,
            },
            {
                "label": "poll_2",
                "elapsed_seconds": 60.0,
                "body_text_hash": "sha256:b",
                "queue_prompt_visible": False,
                "queue_position": None,
                "queue_position_visible": False,
                "queue_prompt_hash": "",
                "active_event_queued": False,
                "active_event_running": True,
                "session_started_text_visible": True,
                "session_starting_text_visible": False,
                "jupyter_frame_visible": False,
                "jupyter_session_or_kernel_visible": False,
                "web_tpu_runtime_ready": False,
            },
        ],
    )

    assert report["active_event_running"] is True
    assert "kaggle_web_tpu_active_event_queued" not in report["blocker_codes"]
    assert "kaggle_web_tpu_queue_prompt_visible" not in report["blocker_codes"]
    assert "kaggle_web_tpu_session_still_starting" not in report["blocker_codes"]
    assert "kaggle_web_tpu_jupyter_frame_not_visible" in report["blocker_codes"]
    assert report["cleanup_status"]["live_resources_left_running"] is True
    assert check.validate(report) == []


def test_session_started_handoff_candidate_requires_runtime_signal() -> None:
    text_only = {
        "session_started_text_visible": True,
        "queue_prompt_visible": False,
        "session_starting_text_visible": False,
        "active_event_running": False,
        "jupyter_frame_visible": False,
        "jupyter_session_or_kernel_visible": False,
        "web_tpu_runtime_ready": False,
    }

    assert probe.session_started_without_queue(text_only) is True
    assert probe.session_started_handoff_candidate(text_only) is False

    with_active_event = dict(text_only)
    with_active_event["active_event_running"] = True
    assert probe.session_started_handoff_candidate(with_active_event) is True

    with_jupyter_frame = dict(text_only)
    with_jupyter_frame["jupyter_frame_visible"] = True
    assert probe.session_started_handoff_candidate(with_jupyter_frame) is True


def test_write_live_status_records_queue_position_without_private_state(tmp_path: Path) -> None:
    args = _args(tmp_path)
    probe.write_live_status(
        tmp_path,
        args,
        steps=[{"name": "click_start_session", "ok": True}],
        observations=[
            {
                "label": "poll_1",
                "elapsed_seconds": 30.0,
                "body_text_hash": "sha256:a",
                "queue_prompt_visible": True,
                "queue_position": 12,
                "queue_position_visible": True,
                "queue_prompt_hash": "sha256:q",
                "session_started_text_visible": False,
                "session_starting_text_visible": True,
                "active_event_queued": True,
                "active_event_running": False,
                "active_event_count": 1,
                "jupyter_frame_visible": False,
                "jupyter_session_or_kernel_visible": False,
                "web_tpu_runtime_ready": False,
            }
        ],
    )

    status = json.loads((tmp_path / "kaggle_web_tpu_queue_monitor_live_status.json").read_text(encoding="utf-8"))
    assert status["schema"] == probe.LIVE_STATUS_SCHEMA
    assert status["queue_position"] == 12
    assert status["active_event_queued"] is True
    assert status["session_started_handoff_candidate"] is False
    assert status["start_clicked"] is True
    assert status["safety"]["credentials_public"] is False


def test_checker_rejects_read_only_start_click(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path, read_only=True),
        output_dir=tmp_path,
        steps=[{"name": "click_start_session", "ok": True}],
        observations=[
            {
                "label": "poll_1",
                "elapsed_seconds": 10.0,
                "body_text_hash": "sha256:a",
                "queue_prompt_visible": False,
                "queue_position": None,
                "queue_position_visible": False,
                "jupyter_frame_visible": False,
                "jupyter_session_or_kernel_visible": False,
                "web_tpu_runtime_ready": False,
            }
        ],
    )

    assert "read_only_report_clicked_start" in check.validate(report)
