from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from crowdtensor.adapters.text_data import (
    create_instruction_data_pack,
    load_data_pack,
    load_instruction_records,
    validate_instruction_data_pack,
)
from crowdtensor.cli import main
from crowdtensor.core import DataPack
from crowdtensor.core.data_packs import DataPackError


def _records(path) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=True) + "\n"
            for value in (
                {
                    "record_id": "linux-001",
                    "prompt": "How do I inspect disk usage?",
                    "response": "Run df -h for filesystems and du -sh for directories.",
                    "language": "en",
                    "source_ref": "urn:crowdtensor:authored:linux-001",
                },
                {
                    "record_id": "linux-002",
                    "prompt": "How do I inspect memory usage?",
                    "response": "Use free -h and inspect /proc/meminfo when needed.",
                    "language": "en",
                },
            )
        ),
        encoding="utf-8",
    )


def _create(source, output, **overrides):
    values = {
        "pack_id": "open-linux-basics-v1",
        "license_spdx": "CC-BY-4.0",
        "source_kind": "contributor_authored",
        "languages": ("en",),
        "domains": ("linux", "open-source"),
        "contributor_id": "public-contributor-alias",
        "redistribution_allowed": True,
        "training_allowed": True,
        "personal_data_reviewed": True,
        "copyright_reviewed": True,
        "benchmark_contamination_reviewed": True,
        "moderation_status": "approved",
        "public_records": True,
    }
    values.update(overrides)
    return create_instruction_data_pack(source, output, **values)


def test_data_pack_is_canonical_reviewed_and_contains_no_raw_text(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    _records(source)
    output = tmp_path / "pack"
    created = _create(source, output)
    repeated = _create(source, output)
    report = validate_instruction_data_pack(output)
    pack = load_data_pack(output)

    assert created["created"] is True
    assert repeated["created"] is False
    assert report["ok"] is True
    assert report["admission_ready"] is True
    assert DataPack.from_dict(pack.to_dict()) == pack
    assert len(load_instruction_records(output)) == 2
    public = json.dumps(pack.to_dict(), sort_keys=True)
    assert "inspect disk usage" not in public
    assert "public-contributor-alias" not in public
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)

    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / "data_pack_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(pack.to_dict(), schema)
    jsonschema.validate(replace(pack, pack_id="x").to_dict(), schema)


def test_data_pack_fails_closed_for_mutation_and_obvious_secrets(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    _records(source)
    output = tmp_path / "pack"
    _create(source, output)
    with (output / "records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"prompt": "extra", "response": "changed"}) + "\n"
        )
    report = validate_instruction_data_pack(output)
    assert report["ok"] is False
    assert "data_pack_records_hash_mismatch" in report["errors"]

    secret = tmp_path / "secret.jsonl"
    secret.write_text(
        json.dumps(
            {"prompt": "credential", "response": "hf_" + "a" * 30}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DataPackError, match="obvious_secret"):
        _create(secret, tmp_path / "secret-pack")

    with pytest.raises(DataPackError, match="boolean_required"):
        _create(
            source,
            tmp_path / "string-attestation-pack",
            redistribution_allowed="false",
        )


def test_data_pack_cli_requires_explicit_review_for_campaign_admission(
    tmp_path, capsys
) -> None:
    source = tmp_path / "source.jsonl"
    _records(source)
    output = tmp_path / "pending-pack"
    with pytest.raises(SystemExit) as created:
        main(
            [
                "train",
                "data-pack",
                "create",
                str(source),
                str(output),
                "--pack-id",
                "pending-linux-v1",
                "--license",
                "CC-BY-4.0",
                "--source-kind",
                "contributor_authored",
                "--language",
                "en",
                "--domain",
                "linux",
                "--contributor-id",
                "private-alias",
                "--json",
            ]
        )
    assert created.value.code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["admission_ready"] is False
    assert "private-alias" not in json.dumps(report)

    with pytest.raises(SystemExit) as validated:
        main(["train", "data-pack", "validate", str(output), "--json"])
    assert validated.value.code == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["records_verified"] is True
    assert validation["admission_ready"] is False
