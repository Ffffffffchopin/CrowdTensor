from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
import unittest

from crowdtensor.auth import hash_token, token_matches, validate_token_verifier


ROOT = Path(__file__).resolve().parents[1]


class HashTokenTests(unittest.TestCase):
    def test_hash_token_format_and_matching(self) -> None:
        token_hash = hash_token("local-miner")

        self.assertEqual(token_hash, "sha256:" + hashlib.sha256(b"local-miner").hexdigest())
        self.assertTrue(token_matches("local-miner", token_hash))
        self.assertFalse(token_matches("wrong", token_hash))
        self.assertTrue(token_matches("plain", "plain"))

    def test_validate_rejects_bad_verifiers(self) -> None:
        with self.assertRaises(ValueError):
            validate_token_verifier("")
        with self.assertRaises(ValueError):
            validate_token_verifier("sha256:abc")
        with self.assertRaises(ValueError):
            validate_token_verifier("sha256:" + "z" * 64)

    def test_hash_token_script_outputs_plaintext_and_json(self) -> None:
        plain = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "hash_token.py"), "local-miner"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(plain.returncode, 0, plain.stderr)
        self.assertEqual(plain.stdout.strip(), hash_token("local-miner"))

        structured = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "hash_token.py"), "local-miner", "--json"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(structured.returncode, 0, structured.stderr)
        payload = json.loads(structured.stdout)
        self.assertEqual(payload["algorithm"], "sha256")
        self.assertEqual(payload["token_hash"], hash_token("local-miner"))


if __name__ == "__main__":
    unittest.main()
