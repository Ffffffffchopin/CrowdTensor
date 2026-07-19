from __future__ import annotations

import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_web_tpu_start_wait_probe as probe


def _args(tmp_path: Path):
    return probe.parse_args(["--output-dir", str(tmp_path), "--wait-seconds", "30", "--poll-seconds", "5"])


def test_script_compiles() -> None:
    py_compile.compile(str(Path(probe.__file__)), doraise=True)


def test_build_report_marks_runtime_ready(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        output_dir=tmp_path,
        steps=[{"name": "click_start_session", "ok": True}],
        observations=[
            {
                "elapsed_seconds": 12.0,
                "web_tpu_ui_runtime_ready": True,
                "jupyter_frame_visible": True,
                "jupyter_session_or_kernel_visible": True,
                "jupyter_session_count": 1,
                "jupyter_kernel_count": 0,
                "session_started_text_visible": True,
                "queue_visible": False,
                "start_session_visible": False,
            }
        ],
    )

    assert report["ok"] is True
    assert report["web_tpu_ui_runtime_ready"] is True
    assert report["start_clicked"] is True
    assert report["blocker_codes"] == []
    assert probe.public_redaction_errors(report) == []


def test_build_report_records_not_ready_without_overclaim(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        output_dir=tmp_path,
        steps=[{"name": "click_start_session", "ok": True}],
        observations=[
            {
                "elapsed_seconds": 30.0,
                "web_tpu_ui_runtime_ready": False,
                "jupyter_frame_visible": True,
                "jupyter_session_or_kernel_visible": False,
                "jupyter_session_count": 0,
                "jupyter_kernel_count": 0,
                "session_started_text_visible": False,
                "session_starting_text_visible": True,
                "queue_visible": True,
                "start_session_visible": False,
            }
        ],
    )

    assert report["ok"] is True
    assert report["web_tpu_ui_runtime_ready"] is False
    assert "kaggle_web_tpu_queue_visible" in report["blocker_codes"]
    assert "kaggle_web_tpu_session_still_starting" in report["blocker_codes"]
    assert "kaggle_web_tpu_jupyter_session_not_visible" in report["blocker_codes"]


def test_build_report_requires_successful_start_click_when_not_ready(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        output_dir=tmp_path,
        steps=[{"name": "click_start_session", "ok": False}],
        observations=[
            {
                "elapsed_seconds": 5.0,
                "web_tpu_ui_runtime_ready": False,
                "jupyter_frame_visible": False,
                "jupyter_session_or_kernel_visible": False,
                "jupyter_session_count": 0,
                "jupyter_kernel_count": 0,
                "session_started_text_visible": False,
                "queue_visible": False,
                "start_session_visible": True,
            }
        ],
    )

    assert report["start_clicked"] is False
    assert "kaggle_web_tpu_start_session_not_clicked" in report["blocker_codes"]
    assert "kaggle_web_tpu_start_session_visible" in report["blocker_codes"]


def test_build_report_records_observe_failure_without_hanging(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        output_dir=tmp_path,
        steps=[{"name": "observe_wait_loop", "ok": False, "error_type": "TimeoutError"}],
        observations=[
            probe.timeout_observation("observe_wait_loop_failed", elapsed_seconds=35.0)
        ],
    )

    assert report["ok"] is True
    assert report["web_tpu_ui_runtime_ready"] is False
    assert report["final_observation"]["probe_error"] == "observe_wait_loop_failed"
    assert "kaggle_web_tpu_start_wait_observe_failed" in report["blocker_codes"]
