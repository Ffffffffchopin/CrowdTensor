from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts import kaggle_real_llm_live_package as pack


class KaggleRealLlmLivePackageTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_kaggle_real_llm_live_package_test_"))

    def _write_prepare(self, root: Path, *, extra_uploads: list[str] | None = None) -> Path:
        real_llm_dir = root / "real-llm-live"
        real_llm_dir.mkdir()
        (real_llm_dir / "real_llm_live_rc.json").write_text(json.dumps({
            "schema": "real_llm_live_rc_v1",
            "ok": True,
            "coordinator_url": "http://24.199.118.54:9184",
            "miner_id": "kaggle-real-llm",
            "workload": {"hf_model_id": "sshleifer/tiny-gpt2"},
        }), encoding="utf-8")
        uploads = ["stage0", "stage1", *(extra_uploads or [])]
        for upload_key in uploads:
            upload = real_llm_dir / f"kaggle-upload-real-llm-{upload_key}"
            upload.mkdir()
            (upload / "miner.private.env").write_text(f"export CROWDTENSOR_MINER_TOKEN='{upload_key}-secret'\n", encoding="utf-8")
            (upload / "kaggle_remote_miner.py").write_text("# launcher\n", encoding="utf-8")
            (upload / "KAGGLE_RUN.md").write_text("# run\n", encoding="utf-8")
        (real_llm_dir / "operator.private.env").write_text("bad\n", encoding="utf-8")
        (real_llm_dir / "miner_registry.json").write_text("bad\n", encoding="utf-8")
        return real_llm_dir

    def test_build_package_excludes_operator_material_and_writes_real_llm_kernel_metadata(self) -> None:
        root = self._tmp_dir()
        real_llm_dir = self._write_prepare(root)
        output_dir = root / "package"

        report = pack.build_package(argparse.Namespace(
            real_llm_dir=str(real_llm_dir),
            output_dir=str(output_dir),
            owner="xuyuhaosuyi",
            dataset_slug="crowdtensor-real-llm-live-test",
            dataset_title="CrowdTensor Real LLM Live Test",
            kernel_slug_prefix="crowdtensor-real-llm-live-test",
            kernel_title_prefix="CrowdTensor Real LLM Live Test",
            coordinator_url="",
            miner_id="",
            hf_model_id="",
            hf_cache_dir="",
            real_llm_backend="hf_transformers_cpu",
            max_tasks=2,
            max_request_attempts=240,
            inline_kernel_payload=False,
        ))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["dataset_ref"], "xuyuhaosuyi/crowdtensor-real-llm-live-test")
        self.assertEqual(report["hf_model_id"], "sshleifer/tiny-gpt2")
        self.assertTrue((output_dir / "dataset" / "crowdtensor_source.tar.gz").is_file())
        self.assertFalse((output_dir / "dataset" / "operator.private.env").exists())
        self.assertFalse((output_dir / "dataset" / "miner_registry.json").exists())
        for stage in ["stage0", "stage1"]:
            self.assertTrue((output_dir / "dataset" / stage / "miner.private.env").is_file())
            metadata = json.loads((output_dir / "kernels" / stage / "kernel-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["id"], f"xuyuhaosuyi/crowdtensor-real-llm-live-test-{stage}")
            self.assertEqual(metadata["title"], f"Crowdtensor Real Llm Live Test {stage.title()}")
            self.assertEqual(metadata["dataset_sources"], ["xuyuhaosuyi/crowdtensor-real-llm-live-test"])
            self.assertEqual(metadata["enable_internet"], "true")
            code = (output_dir / "kernels" / stage / "kernel.py").read_text(encoding="utf-8")
            self.assertIn("crowdtensor_source.tar.gz", code)
            self.assertIn("miner_cli.py", code)
            self.assertIn("--enable-hf-tiny-gpt-runtime", code)
            self.assertIn("--real-llm-stage-role", code)
            self.assertIn(f'STAGE = "{stage}"', code)
            self.assertIn("transformers==4.40.2", code)
            self.assertIn('"2"', code)
            self.assertIn('"240"', code)

    def test_failure_mode_builds_victim_and_rescue_stage_kernels(self) -> None:
        root = self._tmp_dir()
        real_llm_dir = self._write_prepare(root, extra_uploads=["stage1-victim", "stage1-rescue"])
        output_dir = root / "package"

        report = pack.build_package(argparse.Namespace(
            real_llm_dir=str(real_llm_dir),
            output_dir=str(output_dir),
            owner="xuyuhaosuyi",
            dataset_slug="crowdtensor-real-llm-live-test",
            dataset_title="CrowdTensor Real LLM Live Test",
            kernel_slug_prefix="crowdtensor-real-llm-live-test",
            kernel_title_prefix="CrowdTensor Real LLM Live Test",
            coordinator_url="",
            miner_id="",
            hf_model_id="",
            hf_cache_dir="",
            real_llm_backend="hf_transformers_cpu",
            max_tasks=2,
            max_request_attempts=240,
            compute_seconds=0.3,
            victim_compute_seconds=42.0,
            heartbeat_interval=0.2,
            idle_sleep=0.4,
            failure_mode="kill-stage1-after-claim",
            inline_kernel_payload=False,
        ))

        self.assertTrue(report["ok"], report)
        keys = {item["key"]: item for item in report["stages"]}
        self.assertEqual(set(keys), {"stage0", "stage1-victim", "stage1-rescue"})
        self.assertEqual(keys["stage1-victim"]["role"], "victim")
        self.assertEqual(keys["stage1-victim"]["max_tasks"], 1)
        self.assertEqual(keys["stage1-victim"]["compute_seconds"], 42.0)
        self.assertEqual(keys["stage1-rescue"]["miner_id"], "kaggle-real-llm-stage1-rescue")
        victim_metadata = json.loads((output_dir / "kernels" / "stage1-victim" / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(victim_metadata["id"], "xuyuhaosuyi/crowdtensor-real-llm-live-test-stage1-victim")
        self.assertEqual(victim_metadata["title"], "Crowdtensor Real Llm Live Test Stage1 Victim")
        victim_code = (output_dir / "kernels" / "stage1-victim" / "kernel.py").read_text(encoding="utf-8")
        rescue_code = (output_dir / "kernels" / "stage1-rescue" / "kernel.py").read_text(encoding="utf-8")
        self.assertIn('"42.0"', victim_code)
        self.assertIn('"0.3"', rescue_code)
        self.assertIn('"0.2"', rescue_code)
        self.assertIn('"0.4"', rescue_code)

    def test_inline_kernel_payload_embeds_private_stage_payload_without_dataset_source(self) -> None:
        root = self._tmp_dir()
        real_llm_dir = self._write_prepare(root)
        output_dir = root / "package"

        report = pack.build_package(argparse.Namespace(
            real_llm_dir=str(real_llm_dir),
            output_dir=str(output_dir),
            owner="xuyuhaosuyi",
            dataset_slug="crowdtensor-real-llm-live-test",
            dataset_title="CrowdTensor Real LLM Live Test",
            kernel_slug_prefix="crowdtensor-real-llm-live-test",
            kernel_title_prefix="CrowdTensor Real LLM Live Test",
            coordinator_url="",
            miner_id="",
            hf_model_id="",
            hf_cache_dir="",
            real_llm_backend="hf_transformers_cpu",
            max_tasks=2,
            max_request_attempts=240,
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
            self.assertIn("--enable-hf-tiny-gpt-runtime", code)
            self.assertIn("--real-llm-stage-role", code)
            self.assertIn('"2"', code)
            self.assertIn('"240"', code)

    def test_cuda_backend_writes_gpu_kernel_metadata_and_preflight(self) -> None:
        root = self._tmp_dir()
        real_llm_dir = self._write_prepare(root)
        output_dir = root / "package"

        report = pack.build_package(argparse.Namespace(
            real_llm_dir=str(real_llm_dir),
            output_dir=str(output_dir),
            owner="xuyuhaosuyi",
            dataset_slug="crowdtensor-real-llm-live-test",
            dataset_title="CrowdTensor Real LLM Live Test",
            kernel_slug_prefix="crowdtensor-real-llm-live-test",
            kernel_title_prefix="CrowdTensor Real LLM Live Test",
            coordinator_url="",
            miner_id="",
            hf_model_id="",
            hf_cache_dir="",
            real_llm_backend="hf_transformers_cuda",
            max_tasks=1,
            max_request_attempts=120,
            compute_seconds=0.2,
            victim_compute_seconds=30.0,
            heartbeat_interval=0.1,
            idle_sleep=1.0,
            failure_mode="none",
            inline_kernel_payload=True,
        ))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["real_llm_backend"], "hf_transformers_cuda")
        self.assertEqual(report["torch_spec"], "torch==2.7.1+cu118 torchvision==0.22.1+cu118")
        self.assertEqual(report["torch_index_url"], "https://download.pytorch.org/whl/cu118")
        self.assertEqual(report["transformers_spec"], "transformers==4.40.2")
        self.assertTrue(report["safety"]["gpu_backend_selected"])
        self.assertTrue(report["safety"]["cuda_torch_wheel_pinned"])
        for stage in ["stage0", "stage1"]:
            metadata = json.loads((output_dir / "kernels" / stage / "kernel-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["enable_gpu"], "true")
            code = (output_dir / "kernels" / stage / "kernel.py").read_text(encoding="utf-8")
            self.assertIn("torch.cuda.is_available()", code)
            self.assertIn("torch==2.7.1+cu118", code)
            self.assertIn("torchvision==0.22.1+cu118", code)
            self.assertIn("https://download.pytorch.org/whl/cu118", code)
            self.assertIn("transformers==4.40.2", code)
            self.assertIn("--force-reinstall", code)
            self.assertIn("--index-url", code)
            self.assertIn("--real-llm-backend", code)
            self.assertIn("hf_transformers_cuda", code)
        stage_summaries = {item["stage"]: item for item in report["stages"]}
        self.assertTrue(stage_summaries["stage0"]["gpu_accelerator_enabled"])
        self.assertEqual(stage_summaries["stage0"]["torch_spec"], "torch==2.7.1+cu118 torchvision==0.22.1+cu118")
        self.assertEqual(stage_summaries["stage0"]["transformers_spec"], "transformers==4.40.2")
        self.assertTrue(stage_summaries["stage1"]["cuda_preflight_present"])

    def test_rejects_kernel_slug_prefix_that_exceeds_kaggle_limit(self) -> None:
        root = self._tmp_dir()
        real_llm_dir = self._write_prepare(root)
        output_dir = root / "package"

        with self.assertRaises(SystemExit) as raised:
            pack.build_package(argparse.Namespace(
                real_llm_dir=str(real_llm_dir),
                output_dir=str(output_dir),
                owner="xuyuhaosuyi",
                dataset_slug="crowdtensor-real-llm-live-test",
                dataset_title="CrowdTensor Real LLM Live Test",
                kernel_slug_prefix="crowdtensor-stage-local-gpu-beta-20260528064229",
                kernel_title_prefix="CrowdTensor Real LLM Live Test",
                coordinator_url="",
                miner_id="",
                hf_model_id="",
                hf_cache_dir="",
                real_llm_backend="hf_transformers_cuda",
                real_llm_partition_mode="stage-local",
                max_tasks=1,
                max_request_attempts=120,
                compute_seconds=0.2,
                victim_compute_seconds=30.0,
                heartbeat_interval=0.1,
                idle_sleep=1.0,
                failure_mode="none",
                inline_kernel_payload=True,
            ))

        message = str(raised.exception)
        self.assertIn("--kernel-slug-prefix", message)
        self.assertIn("limit is", message)


if __name__ == "__main__":
    unittest.main()
