from __future__ import annotations

import json

from crowdtensor.qwen15b_training import (
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
)
from scripts.training_qwen15b_dataset_prepare import build


def test_dataset_prepare_keeps_private_payload_out_of_public_report(tmp_path) -> None:
    private = tmp_path / "output" / "private-tokenized.json"

    def preparer(output, **_kwargs):
        private.parent.mkdir(parents=True, exist_ok=True)
        private.write_text('{"train":[[1,2]],"validation":[[3,4]]}', encoding="utf-8")
        return {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "sequence_length": 64,
            "train_sequence_count": 32,
            "validation_sequence_count": 8,
            "private_tokenized_payload_hash": "sha256:private-payload",
            "raw_text_public": False,
            "token_ids_public": False,
            "private_paths_public": False,
            "private_tokenized_path": str(private),
        }

    report = build(tmp_path / "output", preparer=preparer)
    assert report["ok"] is True
    assert report["private_payload_present"] is True
    encoded = json.dumps(report, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert "[[1,2]]" not in encoded
    assert report["token_ids_public"] is False
