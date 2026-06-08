from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "remote_real_llm_sharded_beta_pack.py"
SPEC = importlib.util.spec_from_file_location("remote_real_llm_sharded_beta_pack", SCRIPT_PATH)
pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pack)


class _HttpErrorWithBody(HTTPError):
    def __init__(self, body: str) -> None:
        super().__init__(
            url="http://127.0.0.1:8951/admin/inference-sessions",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=None,
        )
        self._body = body.encode("utf-8")

    def read(self, *args: object, **kwargs: object) -> bytes:
        return self._body


class RemoteRealLlmShardedBetaPackTests(unittest.TestCase):
    def test_parse_args_reads_and_forwards_prompt_texts_file(self) -> None:
        prompt_file = Path(tempfile.mkdtemp(prefix="crowdtensor_remote_real_llm_prompts_test_")) / "prompts.txt"
        prompt_file.write_text("first, comma prompt\nsecond prompt\n", encoding="utf-8")

        args = pack.parse_args([
            "--mode",
            "remote-loopback",
            "--prompt-texts-file",
            str(prompt_file),
        ])

        self.assertEqual(args.prompt_texts, "")
        self.assertEqual(args.prompt_texts_file, str(prompt_file))
        self.assertEqual(args.prompt_texts_list, ["first, comma prompt", "second prompt"])
        self.assertEqual(pack.prompt_list_from_args(args), ["first, comma prompt", "second prompt"])

    def test_parse_args_rejects_inline_and_file_prompt_batch(self) -> None:
        prompt_file = Path(tempfile.mkdtemp(prefix="crowdtensor_remote_real_llm_prompts_test_")) / "prompts.txt"
        prompt_file.write_text("first prompt\nsecond prompt\n", encoding="utf-8")

        with self.assertRaises(SystemExit) as raised:
            pack.parse_args([
                "--mode",
                "remote-loopback",
                "--prompt-texts",
                "first prompt,second prompt",
                "--prompt-texts-file",
                str(prompt_file),
            ])

        self.assertEqual(
            str(raised.exception),
            "remote_real_llm_sharded_beta accepts either --prompt-texts or --prompt-texts-file, not both",
        )

    def test_payload_summary_extracts_nested_generation_hash(self) -> None:
        summary = pack.payload_summary({
            "schema": "remote_real_llm_sharded_beta_v1",
            "ok": True,
            "diagnosis_codes": ["remote_real_llm_sharded_ready", "multi_token_generation_ready"],
            "payload_summaries": {
                "remote_existing_real_llm_sharded_inference": {
                    "generation": {
                        "max_new_tokens": 4,
                        "generated_token_count": 4,
                        "generated_text_hash": "sha256:nested",
                        "generated_text_redacted": True,
                        "multi_token_generation_ready": True,
                    }
                }
            },
        })

        self.assertEqual(summary["generation"]["generated_token_count"], 4)
        self.assertEqual(summary["generation"]["generated_text_hash"], "sha256:nested")

    def test_remote_existing_reports_missing_hf_dependencies(self) -> None:
        output_dir = Path(tempfile.mkdtemp(prefix="crowdtensor_remote_real_llm_beta_test_"))
        args = pack.parse_args([
            "--mode",
            "remote-existing",
            "--coordinator-url",
            "http://127.0.0.1:8951",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
            "--output-dir",
            str(output_dir),
            "--request-count",
            "1",
        ])
        error = _HttpErrorWithBody(
            "real_llm_sharded_infer requires optional Hugging Face dependencies: transformers. "
            "Install with: python -m pip install -e .[hf]"
        )

        with patch.object(pack.base, "request_json", side_effect=error):
            report = pack.build_report(args)

        self.assertFalse(report["ok"])
        self.assertEqual(report["schema"], "remote_real_llm_sharded_beta_v1")
        self.assertIn("hf_dependencies_missing", report["diagnosis_codes"])
        self.assertIn("session_create_failed", report["diagnosis_codes"])
        self.assertIn("remote_real_llm_sharded_failed", report["diagnosis_codes"])
        step = report["steps"][0]
        self.assertEqual(step["http_status"], 500)
        self.assertIn("python -m pip install -e .[hf]", step["error"])


if __name__ == "__main__":
    unittest.main()
