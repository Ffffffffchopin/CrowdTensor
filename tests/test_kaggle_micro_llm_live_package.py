from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts import kaggle_micro_llm_live_package as pack


class KaggleMicroLlmLivePackageTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_kaggle_micro_llm_live_package_test_"))

    def test_build_package_excludes_operator_material_and_writes_kernel_metadata(self) -> None:
        root = self._tmp_dir()
        kaggle_dir = root / "external-real"
        kaggle_dir.mkdir()
        (kaggle_dir / "kaggle_real_runtime_acceptance.json").write_text(json.dumps({
            "schema": "kaggle_real_runtime_acceptance_v1",
            "ok": True,
            "coordinator_url": "http://24.199.118.54:9180",
            "miner_id": "kaggle-cpu-1",
        }), encoding="utf-8")
        for stage in ["stage0", "stage1"]:
            upload = kaggle_dir / f"kaggle-upload-{stage}"
            upload.mkdir()
            (upload / "miner.private.env").write_text("export CROWDTENSOR_MINER_TOKEN='miner-secret'\n", encoding="utf-8")
            (upload / "kaggle_remote_miner.py").write_text("# launcher\n", encoding="utf-8")
            (upload / "KAGGLE_RUN.md").write_text("# run\n", encoding="utf-8")
        (kaggle_dir / "operator.private.env").write_text("bad\n", encoding="utf-8")
        (kaggle_dir / "miner_registry.json").write_text("bad\n", encoding="utf-8")

        output_dir = root / "package"
        report = pack.build_package(argparse.Namespace(
            kaggle_dir=str(kaggle_dir),
            output_dir=str(output_dir),
            owner="xuyuhaosuyi",
            dataset_slug="crowdtensor-micro-llm-live-test",
            dataset_title="CrowdTensor Micro LLM Live Test",
            kernel_slug_prefix="crowdtensor-micro-llm-live-test",
            kernel_title_prefix="CrowdTensor Micro LLM Live Test",
            coordinator_url="",
            miner_id="",
            inline_kernel_payload=False,
        ))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["dataset_ref"], "xuyuhaosuyi/crowdtensor-micro-llm-live-test")
        self.assertTrue((output_dir / "dataset" / "crowdtensor_source.tar.gz").is_file())
        self.assertFalse((output_dir / "dataset" / "operator.private.env").exists())
        self.assertFalse((output_dir / "dataset" / "miner_registry.json").exists())
        for stage in ["stage0", "stage1"]:
            self.assertTrue((output_dir / "dataset" / stage / "miner.private.env").is_file())
            self.assertFalse((output_dir / "dataset" / stage / "operator.private.env").exists())
            metadata = json.loads((output_dir / "kernels" / stage / "kernel-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["dataset_sources"], ["xuyuhaosuyi/crowdtensor-micro-llm-live-test"])
            self.assertEqual(metadata["enable_internet"], "true")
            code = (output_dir / "kernels" / stage / "kernel.py").read_text(encoding="utf-8")
            self.assertIn("crowdtensor_source.tar.gz", code)
            self.assertIn("crowdtensor_source", code)
            self.assertIn(f'STAGE = "{stage}"', code)
            self.assertIn("miner_cli.py", code)

    def test_inline_kernel_payload_embeds_private_stage_payload_without_dataset_source(self) -> None:
        root = self._tmp_dir()
        kaggle_dir = root / "external-real"
        kaggle_dir.mkdir()
        (kaggle_dir / "kaggle_real_runtime_acceptance.json").write_text(json.dumps({
            "schema": "kaggle_real_runtime_acceptance_v1",
            "ok": True,
            "coordinator_url": "http://24.199.118.54:9180",
            "miner_id": "kaggle-cpu-1",
        }), encoding="utf-8")
        for stage in ["stage0", "stage1"]:
            upload = kaggle_dir / f"kaggle-upload-{stage}"
            upload.mkdir()
            (upload / "miner.private.env").write_text(f"export CROWDTENSOR_MINER_TOKEN='{stage}-secret'\n", encoding="utf-8")
            (upload / "kaggle_remote_miner.py").write_text("# launcher\n", encoding="utf-8")
            (upload / "KAGGLE_RUN.md").write_text("# run\n", encoding="utf-8")

        output_dir = root / "package"
        report = pack.build_package(argparse.Namespace(
            kaggle_dir=str(kaggle_dir),
            output_dir=str(output_dir),
            owner="xuyuhaosuyi",
            dataset_slug="crowdtensor-micro-llm-live-test",
            dataset_title="CrowdTensor Micro LLM Live Test",
            kernel_slug_prefix="crowdtensor-micro-llm-live-test",
            kernel_title_prefix="CrowdTensor Micro LLM Live Test",
            coordinator_url="",
            miner_id="",
            inline_kernel_payload=True,
        ))

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["safety"]["private_kernel_payload_contains_miner_env"])
        for stage in ["stage0", "stage1"]:
            metadata = json.loads((output_dir / "kernels" / stage / "kernel-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["dataset_sources"], [])
            code = (output_dir / "kernels" / stage / "kernel.py").read_text(encoding="utf-8")
            self.assertIn("SOURCE_TARBALL_B64", code)
            self.assertIn(f"{stage}-secret", code)
            self.assertIn("miner_cli.py", code)


if __name__ == "__main__":
    unittest.main()
