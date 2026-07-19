from __future__ import annotations

from scripts.volunteer_training_public_demo import _parse_cell_output, run_demo
from scripts.volunteer_training_public_demo_check import check


def test_cell_output_parser_handles_pretty_json_and_redacts_private_fields() -> None:
    raw = '{\n  "ok": true,\n  "work_completed": true,\n  "process_returncode": 0,\n  "last_report": {"real_transformers_peft_lora": true}\n}\n'
    result = _parse_cell_output(raw, returncode=0, cell_index=0)
    assert result["ok"] is True
    assert result["work_completed"] is True
    assert result["real_transformers_peft_lora"] is True
    assert result["credential_values_public"] is False
    assert result["private_paths_public"] is False


def test_public_demo_runs_two_cells_and_cleans_private_runtime(tmp_path) -> None:
    report = run_demo(tmp_path / "demo", max_runtime_seconds=180.0)
    assert report["ok"] is True
    assert report["independent_cell_process_count"] == 2
    assert report["cleanup"]["cleanup_verified"] is True
    assert not (tmp_path / "demo" / "private").exists()
    checked = check(tmp_path / "demo" / "volunteer_training_public_demo.json", require_verified=True)
    assert checked["ok"] is True, checked
