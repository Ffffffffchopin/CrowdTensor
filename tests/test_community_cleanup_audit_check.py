import json

from crowdtensor.model_adapter import stable_hash
from scripts.community_cleanup_audit import SCHEMA
from scripts.community_cleanup_audit_check import check


def valid_report() -> dict:
    value = {
        "schema": SCHEMA,
        "ok": True,
        "source_cleanup_verified": True,
        "kaggle_query_authenticated": True,
        "all_community_kaggle_resources_deleted": True,
        "all_evidence_private_runtimes_removed": True,
        "community_docker_resources_removed": True,
        "evidence_source_count": 3,
        "matching_kaggle_resource_count": 0,
        "local_private_runtime_count": 0,
        "community_docker_container_count": 0,
        "community_docker_image_count": 0,
        "live_resources_left_running": False,
        "credential_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    value["content_hash"] = stable_hash(value)
    return value


def test_cleanup_checker_accepts_zero_resource_audit(tmp_path) -> None:
    path = tmp_path / "cleanup.json"
    path.write_text(json.dumps(valid_report()), encoding="utf-8")
    assert check(path)["ok"] is True


def test_cleanup_checker_rejects_retained_resource(tmp_path) -> None:
    value = valid_report()
    value["matching_kaggle_resource_count"] = 1
    value["all_community_kaggle_resources_deleted"] = False
    value["live_resources_left_running"] = True
    value["content_hash"] = stable_hash(
        {key: item for key, item in value.items() if key != "content_hash"}
    )
    path = tmp_path / "cleanup.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    result = check(path)
    assert result["ok"] is False
    assert "community_cleanup_resource_count_nonzero" in result["errors"]
