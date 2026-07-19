from __future__ import annotations

import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_web_tpu_ui_state_probe as probe


def _args(tmp_path: Path):
    return probe.parse_args(["--output-dir", str(tmp_path)])


def test_script_compiles() -> None:
    py_compile.compile(str(Path(probe.__file__)), doraise=True)


def test_build_report_marks_runtime_ready(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        output_dir=tmp_path,
        observation={
            "body_text": "Draft saved\nDraft Session\nCPU\nTPU\nSession started.\nTPU v5e-8",
            "start_session_visible": False,
            "frames": [
                {"has_jupyterapp": True, "session_count": 1, "kernel_count": 1},
            ],
            "controls": [
                {"selector": "button", "index": 0, "text": "Run", "aria": "", "title": "", "disabled": False, "visible": True}
            ],
        },
    )

    assert report["ok"] is True
    assert report["web_tpu_ui_runtime_ready"] is True
    assert report["jupyter_session_or_kernel_visible"] is True
    assert report["jupyter_session_count"] == 1
    assert report["jupyter_kernel_count"] == 1
    assert report["blocker_codes"] == []
    assert probe.public_redaction_errors(report) == []


def test_build_report_records_starting_state(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        output_dir=tmp_path,
        observation={
            "body_text": "Draft saved\nDraft Session\nStarting\nSession is starting...\nTPUs are popular right now. You are #9 in the queue.",
            "start_session_visible": True,
            "frames": [
                {"has_jupyterapp": True, "session_count": 0, "kernel_count": 0},
            ],
            "controls": [
                {"selector": "button", "index": 0, "text": "power_settings_new", "aria": "Start session", "title": "", "disabled": False, "visible": True}
            ],
        },
    )

    assert report["ok"] is True
    assert report["web_tpu_ui_runtime_ready"] is False
    assert "kaggle_web_tpu_session_still_starting" in report["blocker_codes"]
    assert "kaggle_web_tpu_queue_visible" in report["blocker_codes"]
    assert "kaggle_web_tpu_jupyter_session_not_visible" in report["blocker_codes"]


def test_report_redacts_private_url_and_token_material(tmp_path: Path) -> None:
    report = probe.build_report(
        _args(tmp_path),
        output_dir=tmp_path,
        observation={
            "body_text": "Session started. jupyter-proxy?token=secret",
            "start_session_visible": False,
            "frames": [
                {"has_jupyterapp": True, "session_count": 1, "kernel_count": 0},
            ],
            "controls": [
                {
                    "selector": "button",
                    "index": 0,
                    "text": "token=secret",
                    "aria": "jupyter-proxy",
                    "title": "Cookie:",
                    "disabled": False,
                    "visible": True,
                }
            ],
        },
    )

    assert report["public_artifact_safe"] is True
    assert report["body_text_excerpt_public"] == "Session started. <redacted>?<redacted>secret"
    assert probe.public_redaction_errors(report) == []
