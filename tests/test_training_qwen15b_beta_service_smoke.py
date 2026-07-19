from __future__ import annotations

import json

from scripts.training_qwen15b_beta_service_smoke import build


def test_beta_service_smoke_covers_authenticated_persistent_api(tmp_path) -> None:
    report = build(tmp_path / "smoke")
    assert report["ok"] is True
    assert report["authentication_required"] is True
    assert report["submit_idempotent"] is True
    assert report["persistent_process_restart_recovery_verified"] is True
    assert report["recovered_global_step"] == 4
    assert report["cancel_route_ready"] is True
    assert report["running_cancel_marker_ready"] is True
    assert report["cleanup_route_ready"] is True
    assert report["live_gpu_run_performed"] is False
    encoded = json.dumps(report, sort_keys=True)
    assert str(tmp_path) not in encoded
