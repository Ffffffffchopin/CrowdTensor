from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crowdtensor.micro_llm_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    build_default_micro_llm_artifact,
    encode_prompt_text,
    inspect_micro_llm_artifact,
    load_micro_llm_artifact,
)
from crowdtensor.micro_transformer import default_micro_transformer_model, micro_transformer_artifact_hash


class MicroLlmArtifactTests(unittest.TestCase):
    def test_build_load_and_inspect_default_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_default_micro_llm_artifact(tmp)
            loaded = load_micro_llm_artifact(tmp)
            inspected = inspect_micro_llm_artifact(Path(tmp) / "manifest.json")

            self.assertEqual(report["schema"], ARTIFACT_SCHEMA_VERSION)
            self.assertTrue(report["ok"])
            self.assertEqual(report["artifact_hash"], loaded["artifact_hash"])
            self.assertEqual(inspected["artifact_hash"], loaded["artifact_hash"])
            self.assertEqual(loaded["model"]["artifact_hash"], loaded["artifact_hash"])
            self.assertNotEqual(
                loaded["artifact_hash"],
                micro_transformer_artifact_hash(default_micro_transformer_model()),
            )

    def test_prompt_tokenizer_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_default_micro_llm_artifact(tmp)
            artifact = load_micro_llm_artifact(tmp)
            config = artifact["config"]

            self.assertEqual(len(encode_prompt_text("arn", config)), 3)
            with self.assertRaises(ValueError):
                encode_prompt_text("ar", config)
            with self.assertRaises(ValueError):
                encode_prompt_text("zzz", config)

if __name__ == "__main__":
    unittest.main()
