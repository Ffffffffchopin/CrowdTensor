from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from scripts import kaggle_gpu_token_weekly_quota_probe as probe


def test_parse_token_sections_reads_exported_api_tokens(tmp_path: Path) -> None:
    token_file = tmp_path / "kaggle_token.md"
    token_file.write_text(
        "\n".join(
            [
                "# tpuowner",
                "export KAGGLE_API_TOKEN=KGAabc",
                "export MY_KAGGLE_TOKEN=KGAabc",
                "",
                "# primary Kaggle account",
                "export KAGGLE_API_TOKEN='KGAdef'",
                "export MY_KAGGLE_TOKEN='KGAdef'",
            ]
        ),
        encoding="utf-8",
    )

    sections = probe.parse_token_sections(token_file)

    assert [item["label"] for item in sections] == ["tpuowner", "primary Kaggle account"]
    assert sections[0]["env"]["KAGGLE_API_TOKEN"] == "KGAabc"
    assert sections[1]["env"]["MY_KAGGLE_TOKEN"] == "KGAdef"


def test_parse_raw_token_file_accepts_key_with_username_hint(tmp_path: Path) -> None:
    raw_token = tmp_path / "crowdtensor_token.md"
    raw_token.write_text("KGAabc", encoding="utf-8")

    section = probe.parse_raw_token_file(raw_token, username_hint="gpuowner", label="crowdtensor")

    assert section["label"] == "crowdtensor"
    assert section["raw_token_file"] is True
    assert section["env"]["KAGGLE_USERNAME"] == "gpuowner"
    assert section["env"]["KAGGLE_KEY"] == "KGAabc"
    assert section["env"]["KAGGLE_API_TOKEN"] == "KGAabc"


def test_parse_raw_token_file_accepts_json_token(tmp_path: Path) -> None:
    raw_token = tmp_path / "kaggle.json"
    raw_token.write_text(json.dumps({"username": "owner", "key": "KGAjson"}), encoding="utf-8")

    section = probe.parse_raw_token_file(raw_token)

    assert section["label"] == "kaggle"
    assert section["env"]["KAGGLE_USERNAME"] == "owner"
    assert section["env"]["KAGGLE_KEY"] == "KGAjson"


def test_classify_push_distinguishes_weekly_quota_and_accepted() -> None:
    accepted = {"ok": True, "output_tail": "Kernel version 1 successfully pushed"}
    exhausted = {"ok": False, "output_tail": "Kernel push error: Maximum weekly GPU quota of 30.00 hours reached."}
    session = {"ok": False, "output_tail": "Maximum batch GPU session count reached"}

    assert probe.classify_push(accepted) == "gpu_submission_accepted"
    assert probe.classify_push(exhausted) == "weekly_gpu_quota_exhausted"
    assert probe.classify_push(session) == "gpu_session_limit_rejected"


def test_report_redaction_catches_raw_kga_tokens() -> None:
    assert probe.public_redaction_errors({"bad": "KGAsecret-token"})
    assert not probe.public_redaction_errors({"good": "KGA<redacted>"})


def test_infer_owner_from_kernel_list_uses_ref_owner() -> None:
    output = """ref title author lastRunTime totalVotes
tpuowner/minimalaif minimalaif tpuowner 2026-01-01 0
"""
    assert probe.infer_owner_from_kernel_list(output, "Fallback User") == "tpuowner"
    assert probe.infer_owner_from_kernel_list("", "Fallback User") == "fallback-user"


def test_build_summary_counts_quota_classes() -> None:
    summary = probe.build_summary(
        [
            {
                "label": "a",
                "auth_ok": True,
                "push_accepted": True,
                "weekly_gpu_quota_exhausted": False,
                "weekly_gpu_quota_exhausted_by_api": False,
                "gpu_reserved_exceeds_remaining_by_api": False,
                "quota_class": "gpu_submission_accepted",
            },
            {
                "label": "b",
                "auth_ok": True,
                "push_accepted": False,
                "weekly_gpu_quota_exhausted": True,
                "weekly_gpu_quota_exhausted_by_api": True,
                "gpu_reserved_exceeds_remaining_by_api": True,
                "quota_class": "weekly_gpu_quota_exhausted",
            },
        ]
    )

    assert summary["auth_ok_count"] == 2
    assert summary["gpu_submission_accepted_count"] == 1
    assert summary["weekly_gpu_quota_exhausted_count"] == 1
    assert summary["weekly_gpu_quota_exhausted_by_api_count"] == 1
    assert summary["gpu_reserved_exceeds_remaining_by_api_count"] == 1
    assert json.dumps(summary, sort_keys=True)


def test_quota_to_public_dict_records_reserved_effective_remaining() -> None:
    quota = SimpleNamespace(
        time_used=timedelta(seconds=90),
        time_reserved=timedelta(seconds=20),
        total_time_allowed=timedelta(seconds=100),
        minimum_time_allowed=timedelta(seconds=100),
        has_ever_run=True,
    )

    public = probe._quota_to_public_dict(quota)

    assert public["present"] is True
    assert public["time_used_seconds"] == 90
    assert public["time_reserved_seconds"] == 20
    assert public["remaining_seconds"] == 10
    assert public["effective_remaining_after_reserved_seconds"] == 0
    assert public["quota_exhausted_by_used"] is False
    assert public["reserved_exceeds_remaining"] is True
