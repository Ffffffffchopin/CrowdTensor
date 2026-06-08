from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import real_p2p_swarm_inference_core_rc_check as check
from scripts import real_p2p_swarm_inference_core_rc_pack as pack


class RealP2PSwarmInferenceCoreRCPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_real_p2p_rc_test_"))

    def test_missing_hf_dependency_blocks_local_smoke_without_false_ready(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args(["local-smoke", "--output-dir", str(output_dir)])

        with patch.object(pack, "missing_hf_dependencies", return_value=["transformers"]):
            report = pack.build_report(args)

        self.assertFalse(report["ok"], report)
        self.assertTrue(report["degraded"])
        self.assertIn("real_p2p_core_rc_hf_runtime_missing", report["diagnosis_codes"])
        self.assertNotIn("real_p2p_swarm_inference_core_rc_ready", report["diagnosis_codes"])
        self.assertEqual(report["user_status"]["state"], "blocked")
        self.assertEqual(report["recommended_next_command"]["reason"], "install_missing_runtime")
        self.assertIn("pip install", report["recommended_next_command"]["command_line"])
        self.assertEqual(report["review_summary"]["state"], "blocked")
        self.assertEqual(report["review_summary"]["next_step"], "fix_real_p2p_core_rc_blockers")
        self.assertTrue((output_dir / "real_p2p_swarm_inference_core_rc.json").is_file())

    def test_package_writes_runbook(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_report(pack.parse_args(["package", "--output-dir", str(output_dir)]))

        self.assertTrue(report["ok"], report)
        self.assertIn("real_p2p_core_rc_runbook_ready", report["diagnosis_codes"])
        self.assertEqual(report["user_status"]["state"], "package-ready")
        self.assertEqual(report["review_summary"]["state"], "package-ready")
        self.assertFalse(report["review_summary"]["ready"])
        self.assertEqual(report["recommended_next_command"]["reason"], "verify_local_real_p2p_path")
        self.assertIn("real-p2p-rc local-smoke", report["recommended_next_command"]["command_line"])
        self.assertTrue(report["artifact_summary"]["public_artifact_safe"])
        self.assertGreaterEqual(report["artifact_summary"]["present_artifact_count"], 3)
        self.assertTrue((output_dir / "REAL_P2P_SWARM_INFERENCE_CORE_RC.md").is_file())

    def test_evidence_import_requires_real_ready_code(self) -> None:
        output_dir = self._tmp_dir()
        source = check.fake_ready_report("local-smoke", output_dir / "source")
        source["external"] = {
            "external_runtime_verified": True,
            "external_generate_verified": True,
        }
        source_path = output_dir / "source" / "real_p2p_swarm_inference_core_rc.json"
        source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "import"),
            "--real-p2p-report",
            str(source_path),
        ]))

        self.assertTrue(report["ok"], report)
        self.assertIn("real_p2p_core_rc_evidence_import_ready", report["diagnosis_codes"])
        self.assertEqual(source["schema"], "real_p2p_swarm_inference_core_rc_v1")
        self.assertTrue(report["external"]["external_runtime_verified"])
        self.assertTrue(report["external"]["external_generate_verified"])
        self.assertEqual(report["hf_model_id"], pack.DEFAULT_HF_MODEL_ID)
        self.assertEqual(report["expected_hf_model_id"], pack.DEFAULT_HF_MODEL_ID)
        self.assertEqual(report["imported"]["hf_model_id"], pack.DEFAULT_HF_MODEL_ID)
        self.assertTrue(report["imported"]["model"]["compatible"])
        self.assertIn("real_p2p_core_rc_model_metadata_ready", report["diagnosis_codes"])
        self.assertFalse(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertTrue(report["output_request"]["public_artifact_safe"])
        self.assertEqual(report["prompt_scope"]["source"], "prompt-text")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 1)
        self.assertTrue(report["prompt_scope"]["inline_prompt_text"])
        self.assertTrue(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertTrue(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_prompt_placeholders"])
        self.assertTrue(report["prompt_scope"]["saved_artifacts_public_safe"])
        self.assertTrue(report["prompt_scope"]["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertFalse(report["prompt_scope"]["prompt_file_path_public"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(report["prompt_scope"]["public_artifact_safe"])
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertFalse(report["answer_scope"]["visible_in_terminal"])
        self.assertFalse(report["answer_scope"]["terminal_only"])
        self.assertEqual(report["answer_scope"]["saved_json_display"], "hash-only")
        self.assertEqual(report["answer_scope"]["saved_markdown_display"], "hash-only")
        self.assertTrue(report["answer_scope"]["public_artifact_safe"])
        self.assertTrue(report["shareable_summary"]["saved_artifacts_public_safe"])
        self.assertFalse(report["shareable_summary"]["raw_prompt_public"])
        self.assertFalse(report["shareable_summary"]["raw_generated_text_public"])
        self.assertFalse(report["shareable_summary"]["generated_token_ids_public"])
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertFalse(report["shareable_summary"]["local_answer_terminal_only"])
        self.assertEqual(report["review_summary"]["schema"], "real_p2p_swarm_inference_core_rc_review_summary_v1")
        self.assertEqual(report["review_summary"]["state"], "ready")
        self.assertTrue(report["review_summary"]["ready"])
        self.assertEqual(report["review_summary"]["next_step"], "review_artifacts")
        self.assertEqual(report["review_summary"]["recommended_next_command"], report["recommended_next_command"])
        self.assertEqual(report["review_summary"]["next_command"], report["recommended_next_command"]["command_line"])
        self.assertEqual(report["user_status"]["state"], "ready")
        self.assertEqual(report["user_status"]["recommended_label"], report["recommended_next_command"]["label"])
        self.assertEqual(report["recommended_next_command"]["reason"], "review_artifacts")
        self.assertTrue(report["recommended_next_command"]["public_artifact_safe"])
        self.assertGreaterEqual(len(report["next_commands"]), 2)
        self.assertTrue(any(item["label"] == "inspect support bundle" for item in report["next_commands"]))
        self.assertEqual(report["artifact_summary"]["schema"], "real_p2p_swarm_inference_core_rc_artifact_summary_v1")
        self.assertTrue(report["artifact_summary"]["inspect_first"].endswith("real_p2p_swarm_inference_core_rc.md"))
        self.assertTrue(report["artifact_summary"]["summary_json"].endswith("real_p2p_swarm_inference_core_rc.json"))
        self.assertTrue(report["artifact_summary"]["support_bundle"].endswith("support_bundle.json"))
        self.assertGreaterEqual(report["artifact_summary"]["present_artifact_count"], 3)
        self.assertTrue(report["artifact_summary"]["public_artifact_safe"])
        markdown = (output_dir / "import" / "real_p2p_swarm_inference_core_rc.md").read_text(encoding="utf-8")
        self.assertIn("## Review", markdown)
        self.assertIn("## What To Do Next", markdown)
        self.assertIn("## Artifact Summary", markdown)
        self.assertIn("- recommended next:", markdown)
        self.assertIn("inspect support bundle", markdown)
        self.assertIn("## Output Scope", markdown)
        self.assertIn("- output request note:", markdown)
        self.assertIn("answer text", markdown)
        self.assertIn("prompt scope: `source=prompt-text count=1", markdown)
        self.assertIn("- prompt scope note:", markdown)
        self.assertIn("raw prompt text", markdown)
        self.assertIn("state=no-local-answer", markdown)
        self.assertIn("- answer scope note:", markdown)
        self.assertIn("not an answer transcript", markdown)
        self.assertIn("raw_generated_text_public=False", markdown)
        support = json.loads((output_dir / "import" / "support_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(support["review_summary"], report["review_summary"])
        self.assertEqual(support["user_status"], report["user_status"])
        self.assertEqual(support["recommended_next_command"], report["recommended_next_command"])
        self.assertEqual(support["next_commands"], report["next_commands"])
        self.assertEqual(support["artifact_summary"], report["artifact_summary"])
        self.assertEqual(support["prompt_scope"], report["prompt_scope"])
        self.assertEqual(support["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(support["shareable_summary"]["answer_scope_state"], "no-local-answer")

    def test_evidence_import_without_source_prompt_scope_uses_safe_fallback(self) -> None:
        output_dir = self._tmp_dir()
        source = check.fake_ready_report("local-smoke", output_dir / "source")
        source.pop("prompt_scope", None)
        source_path = output_dir / "source" / "real_p2p_swarm_inference_core_rc.json"
        source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "import"),
            "--real-p2p-report",
            str(source_path),
        ]))
        encoded = json.dumps(report, sort_keys=True)

        self.assertEqual(report["prompt_scope"]["source"], "imported-or-built-in-validation-prompts")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 1)
        self.assertFalse(report["prompt_scope"]["inline_prompt_text"])
        self.assertFalse(report["prompt_scope"]["terminal_next_commands_local_private"])
        self.assertFalse(report["prompt_scope"]["terminal_logs_local_private"])
        self.assertFalse(report["prompt_scope"]["prefer_prompt_file_or_stdin_for_shareable_logs"])
        self.assertFalse(report["prompt_scope"]["prompt_file_path_public"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(report["prompt_scope"]["public_artifact_safe"])
        self.assertNotIn("CrowdTensor real P2P core RC", encoded)

    def test_evidence_import_blocks_non_default_model_without_matching_real_p2p_report(self) -> None:
        output_dir = self._tmp_dir()
        source = check.fake_ready_report("local-smoke", output_dir / "source")
        source_path = output_dir / "source" / "real_p2p_swarm_inference_core_rc.json"
        source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "import"),
            "--real-p2p-report",
            str(source_path),
            "--hf-model-id",
            "distilgpt2",
        ]))

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["hf_model_id"], "distilgpt2")
        self.assertEqual(report["imported"]["model"]["observed_hf_model_id"], pack.DEFAULT_HF_MODEL_ID)
        self.assertFalse(report["imported"]["model"]["compatible"])
        self.assertIn("real_p2p_core_rc_model_metadata_mismatch", report["diagnosis_codes"])
        self.assertIn("real_p2p_core_rc_evidence_import_blocked", report["diagnosis_codes"])

    def test_evidence_import_blocks_default_model_without_observed_model_metadata(self) -> None:
        output_dir = self._tmp_dir()
        source = check.fake_ready_report("local-smoke", output_dir / "source")
        source.pop("hf_model_id", None)
        source_path = output_dir / "source" / "real_p2p_swarm_inference_core_rc.json"
        source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "import"),
            "--real-p2p-report",
            str(source_path),
        ]))

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["hf_model_id"], pack.DEFAULT_HF_MODEL_ID)
        self.assertEqual(report["imported"]["model"]["observed_hf_model_id"], "")
        self.assertFalse(report["imported"]["model"]["model_id_present"])
        self.assertFalse(report["imported"]["model"]["model_id_match"])
        self.assertFalse(report["imported"]["model"]["compatible"])
        self.assertFalse(report["imported"]["model"]["default_model_retained_evidence"])
        self.assertIn("real_p2p_core_rc_model_metadata_mismatch", report["diagnosis_codes"])
        self.assertIn("real_p2p_core_rc_evidence_import_blocked", report["diagnosis_codes"])

    def test_external_existing_verify_generate_forwards_and_reports_hf_model_id(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "external-existing",
            "--output-dir",
            str(output_dir),
            "--peer-bootstrap",
            "http://p2p.example",
            "--admin-token",
            "admin-secret",
            "--verify-generate",
            "--hf-model-id",
            "distilgpt2",
            "--prompt-texts",
            "first prompt,second prompt",
            "--stream-generation",
        ])
        catalog = {
            "schema": "real_p2p_provider_catalog_v1",
            "provider_count": 3,
            "peers": [
                {"role": "coordinator", "peer_id": "coord"},
                {"role": "miner", "peer_id": "stage0", "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]}},
                {"role": "miner", "peer_id": "stage1", "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]}},
            ],
        }
        seen_commands: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> object:
            seen_commands.append(command)
            return pack.subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({
                    "schema": "public_swarm_generate_v1",
                    "ok": True,
                    "generation": {
                        "generated_token_count": 2,
                        "request_count": 2,
                        "batch_generation_ready": True,
                        "results": [
                            {
                                "request_id": "req-0",
                                "prompt_hash": "sha256:p0",
                                "generated_token_count": 2,
                                "generated_text_hash": "sha256:g0",
                                "decoded_tokens_match": True,
                                "multi_token_generation_ready": True,
                            },
                            {
                                "request_id": "req-1",
                                "prompt_hash": "sha256:p1",
                                "generated_token_count": 2,
                                "generated_text_hash": "sha256:g1",
                                "decoded_tokens_match": True,
                                "multi_token_generation_ready": True,
                            },
                        ],
                    },
                    "stream": {
                        "enabled": True,
                        "event_count": 4,
                        "endpoint_ready": True,
                        "source": "admin-session-stream",
                        "progress": {
                            "stream_progress_complete": True,
                            "all_token_events_ready": True,
                            "monotonic_progress": True,
                            "expected_request_count": 2,
                            "per_request_progress": [
                                {
                                    "request_key": "req-0",
                                    "request_id": "req-0",
                                    "prompt_hash": "sha256:p0",
                                    "event_count": 2,
                                    "observed_token_counts": [1, 2],
                                    "max_observed_token_count": 2,
                                    "target_token_count": 2,
                                    "monotonic_progress": True,
                                    "stream_progress_complete": True,
                                },
                                {
                                    "request_key": "req-1",
                                    "request_id": "req-1",
                                    "prompt_hash": "sha256:p1",
                                    "event_count": 2,
                                    "observed_token_counts": [1, 2],
                                    "max_observed_token_count": 2,
                                    "target_token_count": 2,
                                    "monotonic_progress": True,
                                    "stream_progress_complete": True,
                                },
                            ],
                            "per_request_progress_complete": True,
                            "per_request_monotonic_progress": True,
                            "observed_token_counts": [1, 2],
                            "max_observed_token_count": 2,
                            "max_new_tokens": 2,
                            "source": "admin-session-stream",
                        },
                        "events": [],
                    },
                }) + "\n",
                stderr="",
            )

        with patch.object(pack, "request_json", return_value=catalog):
            report = pack.run_external_existing(args, output_dir=output_dir, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertTrue(seen_commands)
        command = seen_commands[0]
        self.assertIn("--hf-model-id", command)
        self.assertEqual(command[command.index("--hf-model-id") + 1], "distilgpt2")
        self.assertIn("--prompt-texts", command)
        self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
        self.assertIn("--stream", command)
        self.assertEqual(report["hf_model_id"], "distilgpt2")
        self.assertEqual(report["prompt_scope"]["source"], "prompt-texts")
        self.assertEqual(report["prompt_scope"]["prompt_count"], 2)
        self.assertTrue(report["prompt_scope"]["inline_prompt_text"])
        self.assertFalse(report["prompt_scope"]["raw_prompt_public"])
        self.assertTrue(report["prompt_scope"]["public_artifact_safe"])
        self.assertTrue(report["batch"]["batch_generation_ready"])
        self.assertTrue(report["stream"]["stream_generation_ready"])
        self.assertIn("real_p2p_core_rc_model_metadata_ready", report["diagnosis_codes"])
        self.assertIn("external_real_p2p_generate_batch_ready", report["diagnosis_codes"])
        self.assertIn("external_real_p2p_generate_stream_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_endpoint_ready", report["diagnosis_codes"])
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("first prompt", encoded)
        self.assertNotIn("second prompt", encoded)

    def test_prompt_batch_requires_external_verify_generate(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args(["local-smoke", "--prompt-texts", "a,b"])
        with self.assertRaises(SystemExit):
            pack.parse_args([
                "external-existing",
                "--peer-bootstrap",
                "http://p2p.example",
                "--prompt-texts",
                "a,b",
            ])

    def test_real_p2p_source_tarball_includes_libp2p_runtime_files(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.build_real_p2p_source_tarball(output_dir / "crowdtensor_source.tar.gz")

        self.assertTrue(report["libp2p_runtime_files_included"], report)
        with tarfile.open(output_dir / "crowdtensor_source.tar.gz", "r:gz") as archive:
            names = set(archive.getnames())
        self.assertIn("package.json", names)
        self.assertIn("package-lock.json", names)
        self.assertIn("scripts/real_p2p_daemon.py", names)
        self.assertIn("scripts/libp2p_node20_polyfill.mjs", names)
        self.assertIn("scripts/libp2p_kad_daemon.mjs", names)

    def test_kaggle_kernel_installs_node_dependencies_for_libp2p_sidecar(self) -> None:
        code = pack.render_real_p2p_kaggle_kernel(
            stage="stage0",
            p2p_url="http://24.199.118.54:9760",
            p2p_http_port=9870,
            p2p_libp2p_bootstrap="/ip4/24.199.118.54/tcp/10760/p2p/12D3KooWtest",
            swarm_id="swarm-test",
            miner_id="miner-test",
            backend="cpu",
            hf_model_id="sshleifer/tiny-gpt2",
            hf_cache_dir="/tmp/hf",
            miner_token="miner-token",
            peer_secret="peer-secret",
            max_tasks=2,
            source_tarball_b64="",
        )

        self.assertIn('"npm"', code)
        self.assertIn('"install"', code)
        self.assertIn("libp2p-kad", code)
        self.assertIn("P2P_LIBP2P_BOOTSTRAP", code)
        self.assertIn("KAGGLE_STAGE_STATUS", code)
        self.assertIn("real_p2p_kaggle_stage_status_v1", code)
        self.assertIn("status_path.write_text", code)
        self.assertIn("redact_value(\" \".join(command))", code)
        self.assertIn("redact_value(\" \".join(sidecar_cmd))", code)

    def test_kaggle_package_failure_mode_builds_victim_and_rescue_kernels(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "kaggle-auto",
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "owner",
            "--kernel-slug-prefix",
            "ct-real-p2p-requeue-test",
            "--failure-mode",
            "kill-stage1-after-claim",
            "--max-new-tokens",
            "8",
            "--compute-seconds",
            "0.4",
            "--victim-compute-seconds",
            "42",
            "--max-request-attempts",
            "77",
        ])

        report = pack.build_real_p2p_kaggle_package(
            args,
            output_dir=output_dir,
            miner_token="miner-token",
            peer_secret="peer-secret",
            p2p_libp2p_bootstrap="/ip4/24.199.118.54/tcp/10760/p2p/12D3KooWtest",
        )
        stages = {item["key"]: item for item in report["stages"]}

        self.assertTrue(report["ok"], report)
        self.assertEqual(set(stages), {"stage0", "stage1-victim", "stage1-rescue"})
        self.assertEqual(stages["stage1-victim"]["role"], "victim")
        self.assertEqual(stages["stage1-victim"]["max_tasks"], 1)
        self.assertEqual(stages["stage1-victim"]["compute_seconds"], 42.0)
        self.assertEqual(stages["stage1-rescue"]["miner_id"], "real-p2p-rc-kaggle-stage1-rescue")
        self.assertLessEqual(len(stages["stage1-victim"]["kernel_ref"].split("/", 1)[1]), 49)
        self.assertLessEqual(len(stages["stage1-rescue"]["kernel_ref"].split("/", 1)[1]), 49)
        self.assertTrue(stages["stage1-victim"]["kernel_ref"].endswith("stage1-victim"))
        self.assertTrue(stages["stage1-rescue"]["kernel_ref"].endswith("stage1-rescue"))
        victim_code = (Path(stages["stage1-victim"]["kernel_dir"]) / "kernel.py").read_text(encoding="utf-8")
        rescue_code = (Path(stages["stage1-rescue"]["kernel_dir"]) / "kernel.py").read_text(encoding="utf-8")
        self.assertIn("COMPUTE_SECONDS = 42.0", victim_code)
        self.assertIn("MAX_REQUEST_ATTEMPTS = 77", rescue_code)
        self.assertIn('"--compute-seconds"', rescue_code)
        self.assertIn('"--max-request-attempts"', rescue_code)

    def test_kaggle_connectivity_package_writes_read_only_probe(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "kaggle-connectivity",
            "--output-dir",
            str(output_dir),
            "--public-host",
            "24.199.118.54",
            "--p2p-port",
            "9760",
            "--coordinator-port",
            "9761",
            "--libp2p-port",
            "10760",
            "--kernel-slug-prefix",
            "ct-connectivity-test",
        ])

        report = pack.build_kaggle_connectivity_package(args, output_dir=output_dir)
        kernel = Path(report["kernel_dir"]) / "kernel.py"
        text = kernel.read_text(encoding="utf-8")

        self.assertTrue(report["ok"], report)
        self.assertIn("real_p2p_kaggle_connectivity_package_ready", report["diagnosis_codes"])
        self.assertIn("real_p2p_kaggle_connectivity_probe_v1", text)
        self.assertIn("p2p_http_health", text)
        self.assertIn("libp2p_tcp", text)
        self.assertNotIn("MINER_TOKEN", text)

    def test_kaggle_runtime_smoke_package_writes_runtime_probe(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "kaggle-runtime-smoke",
            "--output-dir",
            str(output_dir),
            "--discovery-backend",
            "libp2p-kad",
            "--public-host",
            "24.199.118.54",
            "--p2p-port",
            "9760",
            "--libp2p-port",
            "10760",
            "--kernel-slug-prefix",
            "ct-runtime-test",
        ])

        report = pack.build_kaggle_runtime_smoke_package(
            args,
            output_dir=output_dir,
            p2p_libp2p_bootstrap="/ip4/24.199.118.54/tcp/10760/p2p/12D3KooWtest",
        )
        kernel = Path(report["kernel_dir"]) / "kernel.py"
        text = kernel.read_text(encoding="utf-8")

        self.assertTrue(report["ok"], report)
        self.assertIn("real_p2p_kaggle_runtime_smoke_package_ready", report["diagnosis_codes"])
        self.assertTrue(report["source"]["libp2p_runtime_files_included"])
        self.assertIn("real_p2p_kaggle_runtime_smoke_status_v1", text)
        self.assertIn("npm_install", text)
        self.assertIn("node_check_libp2p_daemon", text)
        self.assertIn("hf_dependency_check", text)
        self.assertIn("libp2p_sidecar_ready", text)
        self.assertIn("libp2p_sidecar_failed", text)
        self.assertIn("sidecar_returncode", text)
        self.assertIn("import shutil", text)
        self.assertIn("cleanup_large_runtime_outputs", text)
        self.assertNotIn("MINER_TOKEN", text)

    def test_finalize_kaggle_connectivity_accepts_ready_probe(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "kaggle-connectivity",
            "--output-dir",
            str(output_dir),
            "--p2p-port",
            "9760",
            "--coordinator-port",
            "9761",
            "--libp2p-port",
            "10760",
        ])
        package = {
            "schema": "real_p2p_kaggle_connectivity_package_v1",
            "ok": True,
            "output_dir": str(output_dir / "kaggle-connectivity"),
            "kernel_ref": "owner/probe",
        }

        report = pack.finalize_kaggle_connectivity(
            args,
            output_dir=output_dir,
            steps=[
                {"name": "real_p2p_daemon_public", "ok": True},
                {"name": "serve_public_real_p2p", "ok": True},
                {"name": "kaggle_push_connectivity_probe", "ok": True},
                {"name": "kaggle_connectivity_probe_status", "ok": True, "stdout_tail": "KernelWorkerStatus.COMPLETE"},
            ],
            payloads={"connectivity_package": package},
            pushed_refs={"probe": "owner/probe"},
            cleanup_steps=[{"name": "kaggle_delete_probe", "ok": True}],
            secret_values=["secret"],
            p2p_process={},
            serve_process={},
        )

        self.assertTrue(report["ok"], report)
        self.assertIn("real_p2p_kaggle_connectivity_probe_ready", report["diagnosis_codes"])
        self.assertIn("libp2p_external_tcp_reachable", report["diagnosis_codes"])

    def test_finalize_kaggle_runtime_smoke_accepts_ready_probe(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "kaggle-runtime-smoke",
            "--discovery-backend",
            "libp2p-kad",
            "--output-dir",
            str(output_dir),
            "--p2p-port",
            "9760",
            "--libp2p-port",
            "10760",
        ])
        package = {
            "schema": "real_p2p_kaggle_runtime_smoke_package_v1",
            "ok": True,
            "output_dir": str(output_dir / "kaggle-runtime-smoke"),
            "kernel_ref": "owner/smoke",
            "p2p_libp2p_bootstrap": "/ip4/24.199.118.54/tcp/10760/p2p/12D3KooWtest",
        }

        report = pack.finalize_kaggle_runtime_smoke(
            args,
            output_dir=output_dir,
            steps=[
                {"name": "real_p2p_daemon_public", "ok": True},
                {"name": "public_libp2p_bootstrap_multiaddr", "ok": True},
                {"name": "kaggle_push_runtime_smoke", "ok": True},
                {"name": "kaggle_runtime_smoke_status", "ok": True, "stdout_tail": "KernelWorkerStatus.COMPLETE"},
            ],
            payloads={"runtime_smoke_package": package},
            pushed_refs={"runtime-smoke": "owner/smoke"},
            cleanup_steps=[{"name": "kaggle_delete_runtime-smoke", "ok": True}],
            secret_values=["secret"],
            p2p_process={},
        )

        self.assertTrue(report["ok"], report)
        self.assertIn("real_p2p_kaggle_runtime_smoke_ready", report["diagnosis_codes"])
        self.assertIn("real_p2p_kaggle_hf_runtime_ready", report["diagnosis_codes"])
        self.assertIn("kaggle_libp2p_sidecar_start_ready", report["diagnosis_codes"])
        self.assertNotIn("hivemind_petals_class_alpha_ready", report["diagnosis_codes"])

    def test_finalize_report_accepts_ready_real_p2p_generation(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args(["local-smoke", "--output-dir", str(output_dir), "--max-new-tokens", "2"])
        catalog = {
            "schema": "real_p2p_provider_catalog_v1",
            "provider_count": 3,
            "registry": {"signed_provider_record_count": 3},
            "peers": [
                {"role": "coordinator", "peer_id": "coord", "urls": {"coordinator": "http://127.0.0.1:9761"}},
                {
                    "role": "miner",
                    "peer_id": "stage0",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                },
                {
                    "role": "miner",
                    "peer_id": "stage1",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]},
                },
            ],
        }
        rows = [
            {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 0}},
            {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": 0}},
            {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 1}},
            {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": 1}},
        ]
        payloads = {
            "generate": {
                "ok": True,
                "session": {"session_id": "real-p2p-session"},
                "generation": {
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "generated_text_hash": "sha256:generated",
                    "decoded_tokens_match": True,
                },
            },
            "route_dry_run": {
                "ok": True,
                "route": {
                    "route_source": "real-p2p-discovery",
                    "usable_now": True,
                    "matched_capabilities": {
                        "real_llm_sharded_stage0": "stage0",
                        "real_llm_sharded_stage1": "stage1",
                    },
                },
            },
        }

        with patch.object(pack, "request_json", side_effect=[catalog, {"schema": "real_p2p_nat_relay_diagnostics_v1", "ok": True}]), patch.object(
            pack.product_mvp,
            "request_json",
            return_value={"results": rows},
        ):
            report = pack.finalize_report(
                args,
                output_dir=output_dir,
                steps=[
                    {"name": "p2p_daemon", "ok": True},
                    {"name": "serve_real_p2p", "ok": True},
                    {"name": "join_real_p2p_stage0_step_0", "ok": True, "duration_seconds": 0.2},
                    {"name": "join_real_p2p_stage1_step_0", "ok": True, "duration_seconds": 0.3},
                    {"name": "join_real_p2p_stage0_step_1", "ok": True, "duration_seconds": 0.2},
                    {"name": "join_real_p2p_stage1_step_1", "ok": True, "duration_seconds": 0.3},
                    {"name": "generate_real_p2p", "ok": True},
                    {"name": "generate_real_p2p_route_dry_run", "ok": True},
                ],
                payloads=payloads,
                p2p_url="http://127.0.0.1:9760",
                coordinator_url="http://127.0.0.1:9761",
                p2p_process={},
                serve_process={},
                secret_values=["secret"],
            )

        encoded = json.dumps(report, sort_keys=True)
        self.assertTrue(report["ok"], report)
        self.assertIn("real_p2p_swarm_inference_core_rc_ready", report["diagnosis_codes"])
        self.assertIn("libp2p_or_real_p2p_discovery_ready", report["diagnosis_codes"])
        self.assertIn("real_p2p_local_generate_ready", report["diagnosis_codes"])
        self.assertNotIn('"generated_token_ids":', encoded)
        self.assertNotIn('"generated_text":', encoded)

    def test_finalize_report_accepts_local_stage_requeue_summary(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "local-smoke",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "2",
            "--failure-mode",
            "kill-stage1-after-claim",
            "--lease-seconds",
            "2",
            "--victim-compute-seconds",
            "8",
        ])
        catalog = {
            "schema": "real_p2p_provider_catalog_v1",
            "provider_count": 3,
            "registry": {"signed_provider_record_count": 3},
            "peers": [
                {"role": "coordinator", "peer_id": "coord", "urls": {"coordinator": "http://127.0.0.1:9761"}},
                {
                    "role": "miner",
                    "peer_id": "stage0",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                },
                {
                    "role": "miner",
                    "peer_id": "stage1-rescue",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]},
                },
            ],
        }
        rows = [
            {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 0}},
            {"status": "completed", "stage_id": 1, "miner_id": "stage1-rescue", "validation": {"generation_step": 0}},
            {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 1}},
            {"status": "completed", "stage_id": 1, "miner_id": "stage1-rescue", "validation": {"generation_step": 1}},
        ]
        payloads = {
            "generate": {
                "ok": True,
                "session": {"session_id": "real-p2p-session"},
                "generation": {
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "generated_text_hash": "sha256:generated",
                    "decoded_tokens_match": True,
                },
            },
            "route_dry_run": {
                "ok": True,
                "route": {
                    "route_source": "real-p2p-discovery",
                    "usable_now": True,
                    "matched_capabilities": {
                        "real_llm_sharded_stage0": "stage0",
                        "real_llm_sharded_stage1": "stage1-rescue",
                    },
                },
            },
            "live_requeue_summary": {
                "enabled": True,
                "scope": "local-smoke",
                "failure_mode": "kill-stage1-after-claim",
                "target_stage": "stage1",
                "victim_miner_id": "real-p2p-rc-local-stage1-victim",
                "rescue_miner_id": "real-p2p-rc-local-stage1-rescue",
                "victim_task_id": "task-stage1-0",
                "claim_observed": True,
                "victim_kernel_deleted": True,
                "victim_process_terminated": True,
                "lease_expired": True,
                "rescue_miner_used": True,
                "rescued_result": True,
                "accepted_result_after_requeue": True,
                "victim_result_accepted": False,
                "claim": {"ok": True, "task_id": "task-stage1-0", "stage": "stage1", "miner_id": "real-p2p-rc-local-stage1-victim"},
                "requeue_observation": {"ok": True, "task_id": "task-stage1-0", "status": "queued", "attempt": 2},
                "rescue_observation": {"ok": True, "task_id": "task-stage1-0", "status": "completed", "miner_id": "real-p2p-rc-local-stage1-rescue"},
            },
        }

        with patch.object(pack, "request_json", side_effect=[catalog, {"schema": "real_p2p_nat_relay_diagnostics_v1", "ok": True}]), patch.object(
            pack.product_mvp,
            "request_json",
            return_value={"results": rows},
        ):
            report = pack.finalize_report(
                args,
                output_dir=output_dir,
                steps=[
                    {"name": "p2p_daemon", "ok": True},
                    {"name": "serve_real_p2p", "ok": True},
                    {"name": "local_stage1_victim_claim_observed", "ok": True},
                    {"name": "local_stage1_victim_process_terminated", "ok": True},
                    {"name": "local_stage1_victim_task_requeued", "ok": True},
                    {"name": "local_stage1_rescue_result_accepted", "ok": True},
                    {"name": "generate_real_p2p", "ok": True},
                    {"name": "generate_real_p2p_route_dry_run", "ok": True},
                ],
                payloads=payloads,
                p2p_url="http://127.0.0.1:9760",
                coordinator_url="http://127.0.0.1:9761",
                p2p_process={},
                serve_process={},
                secret_values=["secret"],
            )

        encoded = json.dumps(report, sort_keys=True)
        self.assertTrue(report["ok"], report)
        self.assertIn("real_p2p_local_stage_requeue_ready", report["diagnosis_codes"])
        self.assertIn("local_stage_requeue_ready", report["diagnosis_codes"])
        self.assertIn("stage_requeue_ready", report["diagnosis_codes"])
        self.assertIn("live_stage1_requeue_ready", report["diagnosis_codes"])
        self.assertNotIn("external_stage_requeue_ready", report["diagnosis_codes"])
        self.assertNotIn("lease_token", encoded)

    def test_finalize_report_accepts_route_ready_after_coordinator_ttl_eviction(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "local-smoke",
            "--discovery-backend",
            "libp2p-kad",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "2",
        ])
        catalog = {
            "schema": "real_p2p_provider_catalog_v1",
            "provider_count": 2,
            "registry": {
                "signed_provider_record_count": 2,
                "provider_record_transport": "libp2p-stream",
            },
            "libp2p": {
                "ok": True,
                "peer_id": "12D3KooWtest",
                "diagnosis_codes": [
                    "libp2p_discovery_backend_ready",
                    "p2p_peer_identity_ready",
                    "p2p_provider_dht_ready",
                ],
            },
            "peers": [
                {
                    "role": "miner",
                    "peer_id": "stage0",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                },
                {
                    "role": "miner",
                    "peer_id": "stage1",
                    "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]},
                },
            ],
        }
        rows = [
            {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 0}},
            {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": 0}},
            {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 1}},
            {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": 1}},
        ]
        payloads = {
            "generate": {
                "ok": True,
                "session": {"session_id": "real-p2p-session"},
                "generation": {
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "generated_text_hash": "sha256:generated",
                    "decoded_tokens_match": True,
                },
            },
            "route_dry_run": {
                "ok": True,
                "route": {
                    "route_source": "real-p2p-discovery",
                    "usable_now": True,
                    "matched_capabilities": {
                        "real_llm_sharded_stage0": "stage0",
                        "real_llm_sharded_stage1": "stage1",
                    },
                },
            },
        }

        with patch.object(pack, "request_json", side_effect=[catalog, {"schema": "real_p2p_nat_relay_diagnostics_v1", "ok": True}]), patch.object(
            pack.product_mvp,
            "request_json",
            return_value={"results": rows},
        ):
            report = pack.finalize_report(
                args,
                output_dir=output_dir,
                steps=[
                    {"name": "p2p_daemon", "ok": True},
                    {"name": "libp2p_bootstrap_multiaddr", "ok": True},
                    {"name": "serve_real_p2p", "ok": True},
                    {"name": "join_real_p2p_stage0_step_0", "ok": True, "duration_seconds": 1.2},
                    {"name": "join_real_p2p_stage1_step_0", "ok": True, "duration_seconds": 1.3},
                    {"name": "join_real_p2p_stage0_step_1", "ok": True, "duration_seconds": 1.1},
                    {"name": "join_real_p2p_stage1_step_1", "ok": True, "duration_seconds": 1.4},
                    {"name": "generate_real_p2p", "ok": True},
                    {"name": "generate_real_p2p_route_dry_run", "ok": True},
                ],
                payloads=payloads,
                p2p_url="http://127.0.0.1:9760",
                coordinator_url="http://127.0.0.1:9761",
                p2p_process={},
                serve_process={},
                secret_values=["secret"],
            )

        self.assertTrue(report["ok"], report)
        self.assertIn("real_p2p_swarm_inference_core_rc_ready", report["diagnosis_codes"])
        self.assertIn("hivemind_petals_class_alpha_local_ready", report["diagnosis_codes"])
        self.assertIn("stage_latency_ready", report["diagnosis_codes"])
        self.assertEqual(report["p2p"]["coordinator_provider_count"], 0)

    def test_finalize_kaggle_auto_keeps_kernel_failure_diagnostics(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            "kaggle-auto",
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "owner",
            "--kernel-slug-prefix",
            "ct-kdiag-test",
            "--max-new-tokens",
            "2",
        ])
        terminal_steps = [
            {
                "name": "kaggle_stage0_terminal_status",
                "key": "stage0",
                "kernel_ref": "owner/stage0",
                "ok": False,
                "terminal": True,
                "stdout_tail": "KernelWorkerStatus.ERROR",
            },
            {
                "name": "kaggle_stage1_terminal_status",
                "key": "stage1",
                "kernel_ref": "owner/stage1",
                "ok": False,
                "terminal": True,
                "stdout_tail": "KernelWorkerStatus.ERROR",
            },
        ]
        output_steps = [
            {
                "name": "kaggle_output_stage0",
                "key": "stage0",
                "kernel_ref": "owner/stage0",
                "ok": True,
                "artifact_count": 2,
                "status_files": ["crowdtensor_real_p2p_rc_stage0_status.json"],
                "log_files": ["crowdtensor_real_p2p_rc_stage0.log"],
            },
            {
                "name": "kaggle_output_stage1",
                "key": "stage1",
                "kernel_ref": "owner/stage1",
                "ok": True,
                "artifact_count": 2,
                "status_files": ["crowdtensor_real_p2p_rc_stage1_status.json"],
                "log_files": ["crowdtensor_real_p2p_rc_stage1.log"],
            },
        ]

        with patch.object(pack, "request_json", return_value={}), patch.object(
            pack.product_mvp,
            "request_json",
            return_value={"results": []},
        ):
            report = pack.finalize_kaggle_auto(
                args,
                output_dir=output_dir,
                steps=[
                    {"name": "kaggle_push_stage0", "ok": True},
                    {"name": "kaggle_push_stage1", "ok": True},
                    {"name": "wait_kaggle_stage_miners_real_p2p", "ok": False, "error": "stage discovery timed out"},
                    *terminal_steps,
                ],
                payloads={
                    "kaggle_package": {"ok": True},
                    "kaggle_terminal_status_steps": terminal_steps,
                    "kaggle_output_steps": output_steps,
                },
                pushed_refs={"stage0": "owner/stage0", "stage1": "owner/stage1"},
                cleanup_steps=[
                    {"name": "kaggle_delete_stage0", "ok": True, "kernel_ref": "owner/stage0"},
                    {"name": "kaggle_delete_stage1", "ok": True, "kernel_ref": "owner/stage1"},
                ],
                secret_values=["secret"],
                p2p_process={},
                serve_process={},
            )

        self.assertFalse(report["ok"], report)
        self.assertIn("real_p2p_kaggle_auto_blocked", report["diagnosis_codes"])
        self.assertIn("external_real_p2p_stage_discovery_blocked", report["diagnosis_codes"])
        self.assertIn("kaggle_stage_terminal_status_ready", report["diagnosis_codes"])
        self.assertIn("kaggle_stage_kernel_terminal_blocked", report["diagnosis_codes"])
        self.assertIn("kaggle_stage_output_probe_ready", report["diagnosis_codes"])
        self.assertEqual(len(report["kaggle_lifecycle"]["terminal_status_steps"]), 2)
        self.assertEqual(len(report["kaggle_lifecycle"]["output_steps"]), 2)
        self.assertNotIn("public_report_safety_failed", report["diagnosis_codes"])

    def test_check_builds_ready_contract(self) -> None:
        output_dir = self._tmp_dir()
        result = check.run_check(check.parse_args(["--output-dir", str(output_dir)]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "real_p2p_swarm_inference_core_rc_check_v1")
        self.assertIn("real_p2p_swarm_inference_core_rc_check_ready", result["diagnosis_codes"])


if __name__ == "__main__":
    unittest.main()
