import json

from scripts.training_qwen15b_elastic_check import check_report
from scripts.training_qwen15b_elastic_pack import pack_report
from tests.test_training_qwen15b_elastic_check import _ready_report


def test_pack_recomputes_retained_live_evidence_without_rerun(tmp_path) -> None:
    source = _ready_report()
    source["ok"] = False
    source["elastic_volunteer_training_ready"] = False
    source["evidence"]["verified"] = False
    source["evidence"]["bounded_no_miner_pause_verified"] = False
    source["evidence"]["stage_reassignment_to_new_miners_verified"] = False
    source["blockers"] = ["elastic_live_acceptance_incomplete"]
    source["cleanup"]["all_four_kernels_deleted"] = False
    source["cleanup"]["live_resources_left_running"] = True
    source["final_cleanup_deletions"] = [
        {"deleted_or_absent": False},
        {"deleted_or_absent": False},
        {"deleted_or_absent": False},
        {"deleted_or_absent": False},
    ]
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    packed = pack_report(source_path, tmp_path / "packed")
    assert packed["ok"] is True
    assert packed["elastic_volunteer_training_ready"] is True
    assert packed["evidence"]["bounded_no_miner_pause_verified"] is True
    assert packed["evidence"]["stage_reassignment_to_new_miners_verified"] is True
    assert packed["cleanup"]["all_four_kernels_deleted"] is True
    assert packed["cleanup"]["live_resources_left_running"] is False
    assert packed["artifact_repack"]["fresh_live_run_performed_for_repack"] is False
    assert check_report(packed, require_ready=True)["ok"] is True
