from scripts.community_docs_check import check


def test_repository_community_docs_are_complete_and_linked() -> None:
    result = check(".")
    assert result["ok"] is True
    assert result["missing_files"] == []
    assert result["broken_links"] == []
    assert all(result["required_phrases"].values())
