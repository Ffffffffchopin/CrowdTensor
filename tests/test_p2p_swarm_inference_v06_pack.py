from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import p2p_swarm_inference_v06_check as check
from scripts import p2p_swarm_inference_v06_pack as pack


class P2PSwarmInferenceV06PackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_p2p_v06_test_"))

    def test_check_builds_ready_evidence_import(self) -> None:
        output_dir = self._tmp_dir()
        result = check.run_check(check.parse_args([
            "--mode",
            "evidence-import",
            "--output-dir",
            str(output_dir),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "p2p_swarm_inference_v06_check_v1")
        report = json.loads((output_dir / "v06" / "p2p_swarm_inference_v06.json").read_text(encoding="utf-8"))
        self.assertIn("p2p_swarm_inference_v06_ready", report["diagnosis_codes"])
        self.assertEqual(report["p2p"]["hf_model_id"], pack.DEFAULT_HF_MODEL_ID)
        self.assertEqual(report["p2p"]["observed_hf_model_id"], pack.DEFAULT_HF_MODEL_ID)
        self.assertTrue(report["p2p"]["model_id_present"])
        self.assertTrue(report["p2p"]["model_id_match"])
        self.assertIn("p2p_v06_model_metadata_ready", report["diagnosis_codes"])
        self.assertIn("Production NAT traversal", report["not_completed"])
        self.assertFalse(report["output_request"]["include_output"])
        self.assertFalse(report["output_request"]["raw_prompt_public"])
        self.assertFalse(report["output_request"]["raw_generated_text_public"])
        self.assertFalse(report["output_request"]["generated_token_ids_public"])
        self.assertTrue(report["output_request"]["public_artifact_safe"])
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
        markdown = (output_dir / "v06" / "p2p_swarm_inference_v06.md").read_text(encoding="utf-8")
        self.assertIn("## Output Scope", markdown)
        self.assertIn("state=no-local-answer", markdown)
        self.assertIn("raw_generated_text_public=False", markdown)
        support = json.loads((output_dir / "v06" / "support_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(support["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(support["shareable_summary"]["answer_scope_state"], "no-local-answer")

    def test_check_builds_ready_non_default_model_evidence_import(self) -> None:
        output_dir = self._tmp_dir()
        result = check.run_check(check.parse_args([
            "--mode",
            "evidence-import",
            "--hf-model-id",
            "distilgpt2",
            "--output-dir",
            str(output_dir),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["hf_model_id"], "distilgpt2")
        report = json.loads((output_dir / "v06" / "p2p_swarm_inference_v06.json").read_text(encoding="utf-8"))
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["p2p"]["hf_model_id"], "distilgpt2")
        self.assertEqual(report["p2p"]["observed_hf_model_id"], "distilgpt2")
        self.assertTrue(report["p2p"]["model_id_match"])
        self.assertIn("p2p_v06_model_metadata_ready", report["diagnosis_codes"])
        self.assertNotIn("p2p_v06_model_metadata_mismatch", report["diagnosis_codes"])

    def test_evidence_import_preserves_nested_local_p2p_details(self) -> None:
        output_dir = self._tmp_dir()
        source_dir = output_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        preview_report = source_dir / "preview.json"
        product_report = source_dir / "product.json"
        optional_report = source_dir / "optional.json"
        local_report = source_dir / "local_v06.json"
        preview_report.write_text(json.dumps(check.fake_preview_payload()), encoding="utf-8")
        product_report.write_text(json.dumps(check.fake_product_payload()), encoding="utf-8")
        optional_report.write_text(json.dumps(check.fake_product_payload("distilgpt2")), encoding="utf-8")
        local_report.write_text(json.dumps({
            "schema": pack.SCHEMA,
            "ok": True,
            "hf_model_id": pack.DEFAULT_HF_MODEL_ID,
            "p2p": {
                "ready": True,
                "hf_model_id": pack.DEFAULT_HF_MODEL_ID,
                "p2p_url": "http://127.0.0.1:9560",
                "catalog_peer_count": 4,
                "coordinator_peer_count": 1,
                "stage0_peer_count": 1,
                "stage1_peer_count": 1,
                "generate_route": {
                    "usable_now": True,
                    "route_source": "p2p-discovery",
                    "matched_capabilities": {
                        "real_llm_sharded_stage0": "stage0",
                        "real_llm_sharded_stage1": "stage1",
                    },
                },
            },
            "payload_summaries": {
                "local_p2p_discovery": {
                    "rescue_probe": {"ok": True, "diagnosis_codes": ["p2p_stage_rescue_ready"]},
                    "real_generate_probe": {
                        "ok": True,
                        "kv_cache": {
                            "schema": "p2p_real_generate_dual_stage_kv_cache_v1",
                            "ready": True,
                            "stage0": {"ready": True, "ready_count": 2, "hit_count": 1},
                            "stage1": {"ready": True, "ready_count": 2, "hit_count": 1},
                        },
                        "diagnosis_codes": [
                            "p2p_real_generate_ready",
                            "p2p_real_generate_kv_cache_ready",
                        ],
                    },
                    "real_stage_rescue_probe": {
                        "ok": True,
                        "diagnosis_codes": [
                            "p2p_real_stage_rescue_ready",
                            "p2p_rescue_generation_completed",
                            "stage0_rescue_generation_completed",
                            "stage1_rescue_generation_completed",
                        ],
                    },
                }
            },
            "diagnosis_codes": [
                "p2pd_daemon_ready",
                "local_three_process_p2p_discovery_ready",
                "p2p_stage_discovery_ready",
                "p2p_generate_route_ready",
                "p2p_stage_rescue_ready",
                "p2p_real_generate_ready",
                "p2p_real_generate_kv_cache_ready",
                "p2p_real_stage_rescue_ready",
                "p2p_rescue_generation_completed",
                "stage0_rescue_generation_completed",
                "stage1_rescue_generation_completed",
                "coordinator_to_p2p_transition_ready",
                "coordinator_result_fallback_ready",
            ],
        }), encoding="utf-8")
        report = pack.build_report(pack.parse_args([
            pack.MODE_EVIDENCE_IMPORT,
            "--output-dir",
            str(output_dir / "final"),
            "--preview-v04-report",
            str(preview_report),
            "--product-mvp-report",
            str(product_report),
            "--optional-model-report",
            str(optional_report),
            "--p2p-discovery-report",
            str(local_report),
            "--json",
        ]))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["p2p"]["catalog_peer_count"], 4)
        self.assertEqual(report["p2p"]["stage0_peer_count"], 1)
        self.assertEqual(report["p2p"]["stage1_peer_count"], 1)
        self.assertTrue(report["p2p"]["generate_route"]["usable_now"])
        self.assertTrue(report["p2p"]["stage_rescue_ready"])
        self.assertTrue(report["p2p"]["real_generate_ready"])
        self.assertTrue(report["p2p"]["kv_cache"]["ready"])
        self.assertEqual(report["p2p"]["hf_model_id"], pack.DEFAULT_HF_MODEL_ID)
        self.assertEqual(report["p2p"]["observed_hf_model_id"], pack.DEFAULT_HF_MODEL_ID)
        self.assertTrue(report["p2p"]["model_id_present"])
        self.assertTrue(report["p2p"]["model_id_match"])
        self.assertIn("p2p_real_generate_kv_cache_ready", report["diagnosis_codes"])
        self.assertTrue(report["p2p"]["real_stage_rescue_ready"])

    def test_evidence_import_blocks_non_default_model_without_matching_p2p_report(self) -> None:
        output_dir = self._tmp_dir()
        source_dir = output_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        preview_report = source_dir / "preview.json"
        product_report = source_dir / "product.json"
        optional_report = source_dir / "optional.json"
        local_report = source_dir / "local_v06.json"
        preview_report.write_text(json.dumps(check.fake_preview_payload()), encoding="utf-8")
        product_report.write_text(json.dumps(check.fake_product_payload()), encoding="utf-8")
        optional_report.write_text(json.dumps(check.fake_product_payload("distilgpt2")), encoding="utf-8")
        payload = check.fake_local_p2p()
        payload["hf_model_id"] = pack.DEFAULT_HF_MODEL_ID
        local_report.write_text(json.dumps(payload), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            pack.MODE_EVIDENCE_IMPORT,
            "--output-dir",
            str(output_dir / "final"),
            "--preview-v04-report",
            str(preview_report),
            "--product-mvp-report",
            str(product_report),
            "--optional-model-report",
            str(optional_report),
            "--p2p-discovery-report",
            str(local_report),
            "--hf-model-id",
            "distilgpt2",
            "--json",
        ]))

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["p2p"]["hf_model_id"], "distilgpt2")
        self.assertEqual(report["p2p"]["observed_hf_model_id"], pack.DEFAULT_HF_MODEL_ID)
        self.assertTrue(report["p2p"]["model_id_present"])
        self.assertFalse(report["p2p"]["model_id_match"])
        self.assertIn("p2p_v06_model_metadata_mismatch", report["diagnosis_codes"])
        self.assertIn("p2p_swarm_inference_v06_blocked", report["diagnosis_codes"])

    def test_evidence_import_blocks_default_model_without_observed_model_metadata(self) -> None:
        output_dir = self._tmp_dir()
        source_dir = output_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        preview_report = source_dir / "preview.json"
        product_report = source_dir / "product.json"
        optional_report = source_dir / "optional.json"
        local_report = source_dir / "local_v06.json"
        preview_report.write_text(json.dumps(check.fake_preview_payload()), encoding="utf-8")
        product_report.write_text(json.dumps(check.fake_product_payload()), encoding="utf-8")
        optional_report.write_text(json.dumps(check.fake_product_payload("distilgpt2")), encoding="utf-8")
        payload = check.fake_local_p2p()
        payload.pop("hf_model_id", None)
        local_report.write_text(json.dumps(payload), encoding="utf-8")

        report = pack.build_report(pack.parse_args([
            pack.MODE_EVIDENCE_IMPORT,
            "--output-dir",
            str(output_dir / "final"),
            "--preview-v04-report",
            str(preview_report),
            "--product-mvp-report",
            str(product_report),
            "--optional-model-report",
            str(optional_report),
            "--p2p-discovery-report",
            str(local_report),
            "--json",
        ]))

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["p2p"]["hf_model_id"], pack.DEFAULT_HF_MODEL_ID)
        self.assertEqual(report["p2p"]["observed_hf_model_id"], "")
        self.assertFalse(report["p2p"]["model_id_present"])
        self.assertFalse(report["p2p"]["model_id_match"])
        self.assertIn("p2p_v06_model_metadata_mismatch", report["diagnosis_codes"])
        self.assertIn("p2p_swarm_inference_v06_blocked", report["diagnosis_codes"])

    def test_real_generate_kv_cache_summary_extracts_safe_rows(self) -> None:
        output_dir = self._tmp_dir()
        events = [
            {
                "type": "task_completed",
                "miner_id": "stage0",
                "validation": {"stage_id": 0, "generation_step": 0},
                "activation_results": [
                    {
                        "kv_cache_schema": "real_llm_stage0_kv_cache_v1",
                        "kv_cache_stage": "stage0_prefix",
                        "kv_cache_ready": True,
                        "kv_cache_hit": False,
                        "kv_cache_tokens_before": 0,
                        "kv_cache_tokens_after": 7,
                        "generated_prefix_token_count": 0,
                        "hidden_state": [[0.1]],
                        "input_ids": [1, 2],
                    }
                ],
            },
            {
                "type": "task_completed",
                "miner_id": "stage0",
                "validation": {"stage_id": 0, "generation_step": 1},
                "activation_results": [
                    {
                        "kv_cache_schema": "real_llm_stage0_kv_cache_v1",
                        "kv_cache_stage": "stage0_prefix",
                        "kv_cache_ready": True,
                        "kv_cache_hit": True,
                        "kv_cache_tokens_before": 7,
                        "kv_cache_tokens_after": 8,
                        "generated_prefix_token_count": 1,
                    }
                ],
            },
            {
                "type": "task_completed",
                "miner_id": "stage1",
                "validation": {"stage_id": 1, "generation_step": 0},
                "inference_results": [
                    {
                        "kv_cache_schema": "real_llm_stage1_kv_cache_v1",
                        "kv_cache_stage": "stage1_suffix",
                        "kv_cache_ready": True,
                        "kv_cache_hit": False,
                        "kv_cache_tokens_before": 0,
                        "kv_cache_tokens_after": 7,
                        "generated_token_count": 1,
                        "generated_token_ids": [1],
                        "generated_text": "raw",
                    }
                ],
            },
            {
                "type": "task_completed",
                "miner_id": "stage1",
                "validation": {"stage_id": 1, "generation_step": 1},
                "inference_results": [
                    {
                        "kv_cache_schema": "real_llm_stage1_kv_cache_v1",
                        "kv_cache_stage": "stage1_suffix",
                        "kv_cache_ready": True,
                        "kv_cache_hit": True,
                        "kv_cache_tokens_before": 7,
                        "kv_cache_tokens_after": 8,
                        "generated_token_count": 2,
                    }
                ],
            },
        ]
        (output_dir / "tasks.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )

        summary = pack.real_generate_kv_cache_summary(output_dir, required_tokens=2)
        encoded = json.dumps(summary, sort_keys=True)

        self.assertTrue(summary["ready"], summary)
        self.assertEqual(summary["stage0"]["hit_count"], 1)
        self.assertEqual(summary["stage1"]["hit_count"], 1)
        self.assertNotIn("hidden_state", encoded)
        self.assertNotIn("input_ids", encoded)
        self.assertNotIn("generated_token_ids", encoded)
        self.assertNotIn("generated_text", encoded)

    def test_finalize_real_generate_report_preserves_safe_batch_summary(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            pack.MODE_LOCAL_SMOKE,
            "--output-dir",
            str(output_dir),
            "--prompt-texts",
            "first private p2p prompt,second private p2p prompt",
            "--max-new-tokens",
            "2",
        ])
        payloads = {
            "generate": {
                "ok": True,
                "diagnosis_codes": ["public_swarm_generate_ready", "public_swarm_generate_batch_ready"],
                "session": {"session_id": "session-1"},
                "route": {"usable_now": True, "route_source": "p2p-discovery"},
                "generation": {
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "request_count": 2,
                    "batch_generation_ready": True,
                    "generated_text_hash": "sha256:batch",
                    "decoded_tokens_match": True,
                    "results": [
                        {
                            "request_id": "req-1",
                            "prompt_hash": "sha256:p1",
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g1",
                            "decoded_tokens_match": True,
                            "multi_token_generation_ready": True,
                            "raw_generated_text_public": False,
                            "generated_token_ids_public": False,
                        },
                        {
                            "request_id": "req-2",
                            "prompt_hash": "sha256:p2",
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:g2",
                            "decoded_tokens_match": True,
                            "multi_token_generation_ready": True,
                            "raw_generated_text_public": False,
                            "generated_token_ids_public": False,
                        },
                    ],
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                },
            }
        }
        rows = [
            {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 0, "request_count": 2}},
            {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": 0, "request_count": 2}},
            {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 1, "request_count": 2}},
            {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": 1, "request_count": 2}},
        ]

        original_results = pack.product_mvp.request_json
        try:
            pack.product_mvp.request_json = lambda *_args, **_kwargs: {"results": rows}  # type: ignore[assignment]
            report = pack.finalize_real_generate_report(
                args,
                output_dir=output_dir,
                base_url="http://coordinator.example",
                steps=[
                    {"name": "p2pd", "ok": True},
                    {"name": "serve", "ok": True},
                    {"name": "join_stage0", "ok": True},
                    {"name": "join_stage1", "ok": True},
                    {"name": "generate", "ok": True},
                ],
                payloads=payloads,
                p2pd_process={},
                serve_process={},
            )
        finally:
            pack.product_mvp.request_json = original_results  # type: ignore[assignment]

        encoded = json.dumps(report, sort_keys=True)
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["batch"]["enabled"])
        self.assertTrue(report["batch"]["batch_generation_ready"])
        self.assertEqual(report["batch"]["expected_request_count"], 2)
        self.assertIn("p2p_real_generate_batch_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])
        self.assertNotIn("first private p2p prompt", encoded)
        self.assertNotIn("second private p2p prompt", encoded)
        self.assertNotIn('"generated_token_ids":', encoded)
        self.assertNotIn('"generated_text":', encoded)

    def test_finalize_real_generate_report_preserves_safe_stream_summary(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            pack.MODE_LOCAL_SMOKE,
            "--output-dir",
            str(output_dir),
            "--stream-generation",
            "--max-new-tokens",
            "2",
        ])
        payloads = {
            "generate": {
                "ok": True,
                "diagnosis_codes": [
                    "public_swarm_generate_ready",
                    "public_swarm_generate_stream_ready",
                    "public_swarm_generate_stream_endpoint_ready",
                ],
                "session": {"session_id": "session-1"},
                "route": {"usable_now": True, "route_source": "p2p-discovery"},
                "generation": {
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "generated_text_hash": "sha256:stream",
                    "decoded_tokens_match": True,
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                },
                "stream": {
                    "enabled": True,
                    "event_count": 2,
                    "source": "admin-session-stream",
                    "endpoint_ready": True,
                    "progress": {
                        "stream_progress_complete": True,
                        "all_token_events_ready": True,
                        "monotonic_progress": True,
                        "observed_token_counts": [1, 2],
                        "max_observed_token_count": 2,
                        "max_new_tokens": 2,
                        "source": "admin-session-stream",
                    },
                    "events": [
                        {
                            "schema": "session_stream_event_v1",
                            "session_id": "session-1",
                            "task_id": "task-1",
                            "miner_id": "stage1",
                            "stage_id": 1,
                            "generated_token_count": 1,
                            "max_new_tokens": 2,
                            "generation_step": 0,
                            "generated_text_hash": "sha256:s1",
                            "raw_generated_text_public": False,
                            "generated_token_ids_public": False,
                        },
                        {
                            "schema": "session_stream_event_v1",
                            "session_id": "session-1",
                            "task_id": "task-2",
                            "miner_id": "stage1",
                            "stage_id": 1,
                            "generated_token_count": 2,
                            "max_new_tokens": 2,
                            "generation_step": 1,
                            "generated_text_hash": "sha256:s2",
                            "raw_generated_text_public": False,
                            "generated_token_ids_public": False,
                        },
                    ],
                },
            }
        }
        rows = [
            {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 0}},
            {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": 0}},
            {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 1}},
            {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": 1}},
        ]

        original_results = pack.product_mvp.request_json
        try:
            pack.product_mvp.request_json = lambda *_args, **_kwargs: {"results": rows}  # type: ignore[assignment]
            report = pack.finalize_real_generate_report(
                args,
                output_dir=output_dir,
                base_url="http://coordinator.example",
                steps=[
                    {"name": "p2pd", "ok": True},
                    {"name": "serve", "ok": True},
                    {"name": "join_stage0", "ok": True},
                    {"name": "join_stage1", "ok": True},
                    {"name": "generate", "ok": True},
                ],
                payloads=payloads,
                p2pd_process={},
                serve_process={},
            )
        finally:
            pack.product_mvp.request_json = original_results  # type: ignore[assignment]

        encoded = json.dumps(report, sort_keys=True)
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["stream"]["stream_generation_ready"])
        self.assertTrue(report["stream"]["endpoint_ready"])
        self.assertIn("p2p_real_generate_stream_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_endpoint_ready", report["diagnosis_codes"])
        self.assertNotIn('"generated_token_ids":', encoded)
        self.assertNotIn('"generated_text":', encoded)

    def test_parse_args_rejects_unbounded_prompt_batches(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args([
                pack.MODE_LOCAL_SMOKE,
                "--prompt-texts",
                "one,two,three,four,five",
            ])

    def test_parse_args_accepts_prompt_texts_file(self) -> None:
        output_dir = self._tmp_dir()
        prompt_file = output_dir / "prompts.txt"
        prompts = ["one prompt, with comma", "second prompt"]
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")

        args = pack.parse_args([
            pack.MODE_LOCAL_SMOKE,
            "--prompt-texts-file",
            str(prompt_file),
        ])

        self.assertEqual(args.prompt_texts_file, str(prompt_file))
        self.assertEqual(args.prompt_texts_list, prompts)

    def test_normalize_local_p2p_report_requires_real_stage_rescue(self) -> None:
        normalized = pack.normalize_local_p2p_report({
            "schema": pack.SCHEMA,
            "ok": True,
            "p2p": {
                "ready": True,
                "catalog_peer_count": 3,
                "coordinator_peer_count": 1,
                "stage0_peer_count": 1,
                "stage1_peer_count": 1,
                "generate_route": {"usable_now": True},
            },
            "payload_summaries": {
                "local_p2p_discovery": {
                    "rescue_probe": {"ok": True},
                    "real_generate_probe": {"ok": True},
                }
            },
            "diagnosis_codes": [
                "p2pd_daemon_ready",
                "local_three_process_p2p_discovery_ready",
                "p2p_stage_discovery_ready",
                "p2p_generate_route_ready",
                "p2p_stage_rescue_ready",
                "p2p_real_generate_ready",
            ],
        })

        self.assertFalse(normalized["ok"], normalized)

    def test_run_p2p_join_miner_step_uses_p2p_join_command(self) -> None:
        seen: dict[str, object] = {}

        def fake_run_step(name: str, command: list[str], *, timeout: float) -> tuple[dict[str, object], dict[str, object]]:
            seen["name"] = name
            seen["command"] = command
            seen["timeout"] = timeout
            return {"name": name, "ok": True, "returncode": 0}, {
                "ok": True,
                "peer_bootstrap_used": True,
                "p2p": {"enabled": True},
                "diagnosis_codes": ["join_command_ready", "p2p_stage_miner_announce_ready"],
            }

        original = pack.product_mvp.run_step
        try:
            pack.product_mvp.run_step = fake_run_step  # type: ignore[assignment]
            step, payload = pack.run_p2p_join_miner_step(
                p2p_url="http://p2p.example",
                swarm_id="swarm",
                miner_id="miner-stage0",
                stage="stage0",
                backend="cpu",
                hf_model_id="sshleifer/tiny-gpt2",
                hf_cache_dir="/tmp/hf-cache",
                timeout=12.0,
                http_timeout=3.0,
            )
        finally:
            pack.product_mvp.run_step = original  # type: ignore[assignment]

        command = seen["command"]
        self.assertTrue(step["ok"])
        self.assertTrue(payload["ok"])
        self.assertIn("join", command)
        self.assertIn("--p2p", command)
        self.assertIn("--peer-bootstrap", command)
        self.assertIn("http://p2p.example", command)
        self.assertIn("--run", command)
        self.assertIn("--hf-model-id", command)
        self.assertEqual(command[command.index("--hf-model-id") + 1], "sshleifer/tiny-gpt2")
        self.assertIn("--hf-cache-dir", command)
        self.assertIn("--max-runtime-seconds", command)
        self.assertEqual(command[command.index("--max-runtime-seconds") + 1], "7.0")
        self.assertIn("--idle-sleep", command)
        self.assertEqual(command[command.index("--idle-sleep") + 1], "0.2")

    def test_p2p_generate_command_forwards_non_default_model(self) -> None:
        command = pack.p2p_generate_command(
            p2p_url="http://p2p.example",
            admin_token="admin-secret",
            backend="cuda",
            hf_model_id="distilgpt2",
            max_new_tokens=3,
            timeout_seconds=12.0,
            http_timeout=7.0,
            prompt_text="hello",
        )

        self.assertIn("generate", command)
        self.assertIn("--p2p", command)
        self.assertEqual(command[command.index("--peer-bootstrap") + 1], "http://p2p.example")
        self.assertEqual(command[command.index("--prompt") + 1], "hello")
        self.assertEqual(command[command.index("--admin-token") + 1], "admin-secret")
        self.assertEqual(command[command.index("--backend") + 1], "cuda")
        self.assertEqual(command[command.index("--hf-model-id") + 1], "distilgpt2")
        self.assertEqual(command[command.index("--max-new-tokens") + 1], "3")
        self.assertEqual(command[command.index("--timeout-seconds") + 1], "12.0")
        self.assertEqual(command[command.index("--http-timeout") + 1], "7.0")
        self.assertNotIn("--prompt-texts", command)
        self.assertNotIn("--dry-run", command)
        self.assertNotIn("--stream", command)
        self.assertEqual(command[-1], "--json")

    def test_p2p_generate_command_preserves_batch_stream_and_dry_run(self) -> None:
        command = pack.p2p_generate_command(
            p2p_url="http://p2p.example",
            backend="cpu",
            hf_model_id="gpt2",
            max_new_tokens="2",
            prompt_text="ignored single prompt",
            prompt_texts="first prompt,second prompt",
            dry_run=True,
            stream=True,
        )

        self.assertEqual(command[command.index("--hf-model-id") + 1], "gpt2")
        self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
        self.assertNotIn("--prompt", command)
        self.assertIn("--dry-run", command)
        self.assertIn("--stream", command)
        self.assertNotIn("--admin-token", command)

    def test_p2p_generate_command_preserves_prompt_texts_file(self) -> None:
        command = pack.p2p_generate_command(
            p2p_url="http://p2p.example",
            backend="cpu",
            hf_model_id="gpt2",
            max_new_tokens="2",
            prompt_text="ignored single prompt",
            prompt_texts="ignored batch",
            prompt_texts_file="/tmp/prompts.txt",
        )

        self.assertEqual(command[command.index("--prompt-texts-file") + 1], "/tmp/prompts.txt")
        self.assertNotIn("--prompt-texts", command)
        self.assertNotIn("--prompt", command)
        self.assertEqual(command[-1], "--json")

    def test_check_builds_ready_local_smoke_contract(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "local-smoke",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_check_builds_ready_package(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "package",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_kaggle_auto_preflights_host_hf_runtime_before_public_side_effects(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            pack.MODE_KAGGLE_AUTO,
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "owner",
        ])
        original_missing = pack.missing_hf_dependencies
        original_package = pack.build_p2p_v06_kaggle_package
        try:
            pack.missing_hf_dependencies = lambda: ["transformers"]  # type: ignore[assignment]

            def fail_package(*_args: object, **_kwargs: object) -> dict[str, object]:
                raise AssertionError("kaggle package should not be generated without host HF runtime")

            pack.build_p2p_v06_kaggle_package = fail_package  # type: ignore[assignment]
            report = pack.run_kaggle_auto(args, output_dir=output_dir)
        finally:
            pack.missing_hf_dependencies = original_missing  # type: ignore[assignment]
            pack.build_p2p_v06_kaggle_package = original_package  # type: ignore[assignment]

        self.assertFalse(report["ok"], report)
        self.assertIn("hf_dependencies_missing", report["diagnosis_codes"])
        self.assertIn("host_hf_runtime_missing", report["diagnosis_codes"])
        self.assertTrue(report["kaggle_lifecycle"]["kernels_deleted"])
        self.assertFalse(report["kaggle_lifecycle"]["token_rotation_required"])

    def test_kaggle_package_includes_signed_join_when_required(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args([
            pack.MODE_KAGGLE_AUTO,
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "owner",
            "--peer-secret",
            "peer-secret-value",
            "--require-signed",
        ])

        report = pack.build_p2p_v06_kaggle_package(args, output_dir=output_dir, miner_token="miner-token-value")

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["safety"]["private_kernel_payload_contains_peer_secret"])
        for stage in ["stage0", "stage1"]:
            kernel = output_dir / "kaggle-package" / "kernels" / stage / "kernel.py"
            text = kernel.read_text(encoding="utf-8")
            self.assertIn('PEER_SECRET = "peer-secret-value"', text)
            self.assertIn('"--peer-secret"', text)
            self.assertNotIn('"--require-signed"', text)
        for stage_report in report["stages"]:
            self.assertTrue(stage_report["signed_join_present"], stage_report)
            self.assertTrue(stage_report["inline_private_peer_secret"], stage_report)

    def test_kaggle_auto_uses_discovery_catalog_when_final_catalog_ttl_expires(self) -> None:
        output_dir = self._tmp_dir()
        package_dir = output_dir / "package"
        kernel_dir = package_dir / "kernels" / "stage0"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.py").write_text("MINER_TOKEN='secret'", encoding="utf-8")
        args = pack.parse_args([
            pack.MODE_KAGGLE_AUTO,
            "--output-dir",
            str(output_dir),
            "--kaggle-owner",
            "owner",
        ])
        payloads = {
            "kaggle_package": {
                "ok": True,
                "output_dir": str(package_dir),
                "diagnosis_codes": ["p2p_v06_kaggle_package_ready"],
            },
            "runtime": {"admin_token": pack.ADMIN_TOKEN},
            "stage_catalog": {
                "schema": "p2p_lite_catalog_v1",
                "peers": [
                    {"role": "coordinator", "capabilities": {"backend": "cpu"}},
                    {"role": "miner", "capabilities": {"backend": "cpu", "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]}},
                    {"role": "miner", "capabilities": {"backend": "cpu", "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]}},
                ],
            },
            "generate": {
                "ok": True,
                "diagnosis_codes": ["public_swarm_generate_ready", "p2p_generate_route_ready"],
                "session": {"session_id": "session-1"},
                "route": {"usable_now": True, "route_source": "p2p-discovery"},
                "generation": {"generated_token_count": 2, "multi_token_generation_ready": True},
            },
        }
        rows = [
            {"status": "completed", "stage_id": 0, "miner_id": "stage0", "validation": {"generation_step": 0}},
            {"status": "completed", "stage_id": 1, "miner_id": "stage1", "validation": {"generation_step": 0}},
        ]

        def final_catalog_expired(*_args: object, **_kwargs: object) -> dict[str, object]:
            return {"schema": "p2p_lite_catalog_v1", "peers": []}

        original_request_json = pack.request_json
        original_results = pack.v06_admin_results
        try:
            pack.request_json = final_catalog_expired  # type: ignore[assignment]
            pack.v06_admin_results = lambda *_args, **_kwargs: rows  # type: ignore[assignment]
            report = pack.finish_kaggle_auto_report(
                args,
                output_dir=output_dir,
                steps=[{"name": "generate_p2p_kaggle", "ok": True}],
                payloads=payloads,
                pushed_refs={"stage0": "owner/stage0", "stage1": "owner/stage1"},
                cleanup_steps=[{"name": "kaggle_delete_stage0", "ok": True}, {"name": "kaggle_delete_stage1", "ok": True}],
                p2pd_proc=None,
                serve_proc=None,
                generate_proc=None,
            )
        finally:
            pack.request_json = original_request_json  # type: ignore[assignment]
            pack.v06_admin_results = original_results  # type: ignore[assignment]

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["stage0_peer_count"], 0)
        self.assertEqual(report["stage1_peer_count"], 0)
        self.assertEqual(report["discovery_stage0_peer_count"], 1)
        self.assertEqual(report["discovery_stage1_peer_count"], 1)
        self.assertIn("p2p_swarm_inference_v06_kaggle_auto_ready", report["diagnosis_codes"])
        self.assertIn("p2p_v06_kaggle_private_artifacts_cleaned", report["diagnosis_codes"])
        self.assertFalse((package_dir / "kernels").exists())
        self.assertTrue(report["kaggle_lifecycle"]["local_private_artifacts_cleaned"])

    def test_wait_v06_workload_queued_uses_v06_observer_token(self) -> None:
        seen: dict[str, str] = {}

        def fake_request_json(method: str, base_url: str, path: str, **kwargs: object) -> dict[str, object]:
            seen["method"] = method
            seen["base_url"] = base_url
            seen["path"] = path
            seen["observer_token"] = str(kwargs.get("observer_token") or "")
            return {
                "tasks": [
                    {
                        "status": "queued",
                        "workload_type": pack.WORKLOAD_TYPE,
                        "workload_metadata": {"stage_id": 0},
                    }
                ]
            }

        original = pack.product_mvp.request_json
        try:
            pack.product_mvp.request_json = fake_request_json  # type: ignore[assignment]
            ready, error = pack.wait_v06_workload_queued("http://coordinator.example", timeout=0.1)
        finally:
            pack.product_mvp.request_json = original  # type: ignore[assignment]

        self.assertTrue(ready, error)
        self.assertEqual(seen["observer_token"], pack.OBSERVER_TOKEN)
        self.assertEqual(seen["path"], "/state")

    def test_wait_v06_workload_started_accepts_leased_fast_task(self) -> None:
        def fake_request_json(method: str, base_url: str, path: str, **kwargs: object) -> dict[str, object]:
            self.assertEqual(kwargs.get("observer_token"), pack.OBSERVER_TOKEN)
            return {
                "tasks": [
                    {
                        "status": "leased",
                        "workload_type": pack.WORKLOAD_TYPE,
                        "workload_metadata": {"stage_id": 0},
                    }
                ]
            }

        original = pack.product_mvp.request_json
        try:
            pack.product_mvp.request_json = fake_request_json  # type: ignore[assignment]
            ready, error = pack.wait_v06_workload_started("http://coordinator.example", timeout=0.1)
        finally:
            pack.product_mvp.request_json = original  # type: ignore[assignment]

        self.assertTrue(ready, error)

    def test_wait_v06_stage_queued_matches_requested_stage(self) -> None:
        def fake_request_json(method: str, base_url: str, path: str, **kwargs: object) -> dict[str, object]:
            self.assertEqual(kwargs.get("observer_token"), pack.OBSERVER_TOKEN)
            return {
                "tasks": [
                    {
                        "status": "queued",
                        "workload_type": pack.WORKLOAD_TYPE,
                        "workload_metadata": {"stage_id": 0},
                    },
                    {
                        "status": "queued",
                        "workload_type": pack.WORKLOAD_TYPE,
                        "workload_metadata": {"stage_id": 1},
                    },
                ]
            }

        original = pack.product_mvp.request_json
        try:
            pack.product_mvp.request_json = fake_request_json  # type: ignore[assignment]
            ready, error = pack.wait_v06_stage_queued("http://coordinator.example", stage_id=1, timeout=0.1)
        finally:
            pack.product_mvp.request_json = original  # type: ignore[assignment]

        self.assertTrue(ready, error)

    def test_external_existing_validates_p2p_catalog_route(self) -> None:
        output_dir = self._tmp_dir()
        source_dir = output_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        preview_report = source_dir / "preview.json"
        product_report = source_dir / "product.json"
        optional_report = source_dir / "optional.json"
        preview_report.write_text(json.dumps(check.fake_preview_payload()), encoding="utf-8")
        product_report.write_text(json.dumps(check.fake_product_payload()), encoding="utf-8")
        optional_report.write_text(json.dumps(check.fake_product_payload("distilgpt2")), encoding="utf-8")

        def fake_request_json(base_url: str, path: str, *, timeout: float = 5.0) -> dict:
            self.assertEqual(base_url, "http://p2p.example")
            self.assertEqual(path, "/peer/catalog")
            return {
                "schema": "p2p_lite_catalog_v1",
                "ok": True,
                "peers": [
                    {
                        "peer_id": "coord",
                        "role": "coordinator",
                        "urls": {"coordinator": "http://coord.example"},
                        "capabilities": {"backend": "cpu"},
                    },
                    {
                        "peer_id": "stage0",
                        "role": "miner",
                        "capabilities": {"backend": "cpu", "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                    },
                    {
                        "peer_id": "stage1",
                        "role": "miner",
                        "capabilities": {"backend": "cpu", "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]},
                    },
                ],
            }

        original = pack.request_json
        try:
            pack.request_json = fake_request_json  # type: ignore[assignment]
            report = pack.build_report(pack.parse_args([
                pack.MODE_EXTERNAL_EXISTING,
                "--output-dir",
                str(output_dir / "external"),
                "--peer-bootstrap",
                "http://p2p.example",
                "--preview-v04-report",
                str(preview_report),
                "--product-mvp-report",
                str(product_report),
                "--optional-model-report",
                str(optional_report),
                "--json",
            ]))
        finally:
            pack.request_json = original  # type: ignore[assignment]

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["p2p"]["external_runtime_verified"])
        self.assertEqual(report["p2p"]["catalog_peer_count"], 3)
        self.assertEqual(report["p2p"]["generate_route"]["route_source"], "p2p-discovery")
        self.assertIn("external_existing_p2p_verified", report["diagnosis_codes"])

    def test_external_existing_filters_p2p_catalog_route_by_model_id(self) -> None:
        output_dir = self._tmp_dir()
        source_dir = output_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        preview_report = source_dir / "preview.json"
        product_report = source_dir / "product.json"
        optional_report = source_dir / "optional.json"
        preview_report.write_text(json.dumps(check.fake_preview_payload()), encoding="utf-8")
        product_report.write_text(json.dumps(check.fake_product_payload("distilgpt2")), encoding="utf-8")
        optional_report.write_text(json.dumps(check.fake_product_payload("distilgpt2")), encoding="utf-8")

        def fake_request_json(base_url: str, path: str, *, timeout: float = 5.0) -> dict:
            del timeout
            self.assertEqual(base_url, "http://p2p.example")
            self.assertEqual(path, "/peer/catalog")
            return {
                "schema": "p2p_lite_catalog_v1",
                "ok": True,
                "peers": [
                    {
                        "peer_id": "coord-tiny",
                        "role": "coordinator",
                        "urls": {"coordinator": "http://coord.example"},
                        "capabilities": {"backend": "cpu", "hf_model_id": "sshleifer/tiny-gpt2"},
                    },
                    {
                        "peer_id": "stage0-tiny",
                        "role": "miner",
                        "capabilities": {
                            "backend": "cpu",
                            "hf_model_id": "sshleifer/tiny-gpt2",
                            "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"],
                        },
                    },
                    {
                        "peer_id": "stage1-tiny",
                        "role": "miner",
                        "capabilities": {
                            "backend": "cpu",
                            "hf_model_id": "sshleifer/tiny-gpt2",
                            "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"],
                        },
                    },
                ],
            }

        original = pack.request_json
        try:
            pack.request_json = fake_request_json  # type: ignore[assignment]
            report = pack.build_report(pack.parse_args([
                pack.MODE_EXTERNAL_EXISTING,
                "--output-dir",
                str(output_dir / "external"),
                "--peer-bootstrap",
                "http://p2p.example",
                "--hf-model-id",
                "distilgpt2",
                "--preview-v04-report",
                str(preview_report),
                "--product-mvp-report",
                str(product_report),
                "--optional-model-report",
                str(optional_report),
                "--json",
            ]))
        finally:
            pack.request_json = original  # type: ignore[assignment]

        self.assertFalse(report["ok"], report)
        route = report["p2p"]["generate_route"]
        self.assertFalse(route["usable_now"], route)
        self.assertIn("real_llm_sharded_stage0", route["missing_capabilities"])
        self.assertIn("session_route_model_mismatch", route["diagnosis_codes"])
        self.assertIn("external_p2p_route_blocked", report["diagnosis_codes"])

    def test_external_existing_verify_generate_forwards_model_batch_and_stream(self) -> None:
        output_dir = self._tmp_dir()
        source_dir = output_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        preview_report = source_dir / "preview.json"
        product_report = source_dir / "product.json"
        optional_report = source_dir / "optional.json"
        preview_report.write_text(json.dumps(check.fake_preview_payload()), encoding="utf-8")
        product_report.write_text(json.dumps(check.fake_product_payload()), encoding="utf-8")
        optional_report.write_text(json.dumps(check.fake_product_payload("distilgpt2")), encoding="utf-8")

        def fake_request_json(base_url: str, path: str, *, timeout: float = 5.0) -> dict:
            self.assertEqual(base_url, "http://p2p.example")
            self.assertEqual(path, "/peer/catalog")
            return {
                "schema": "p2p_lite_catalog_v1",
                "ok": True,
                "peers": [
                    {"peer_id": "coord", "role": "coordinator", "urls": {"coordinator": "http://coord.example"}},
                    {
                        "peer_id": "stage0",
                        "role": "miner",
                        "capabilities": {"backend": "cpu", "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage0"]},
                    },
                    {
                        "peer_id": "stage1",
                        "role": "miner",
                        "capabilities": {"backend": "cpu", "real_llm_sharded_stage_capabilities": ["real_llm_sharded_stage1"]},
                    },
                ],
            }

        captured: list[list[str]] = []

        def fake_run(command: list[str], **_: object):
            captured.append(command)
            self.assertIn("--hf-model-id", command)
            self.assertEqual(command[command.index("--hf-model-id") + 1], "distilgpt2")
            self.assertIn("--prompt-texts", command)
            self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
            self.assertIn("--stream", command)
            return type("Completed", (), {
                "returncode": 0,
                "stdout": json.dumps({
                    "schema": "public_swarm_generate_v1",
                    "ok": True,
                    "generation": {
                        "generated_token_count": 2,
                        "max_new_tokens": 2,
                        "generated_text_hash": "sha256:batch",
                        "request_count": 2,
                        "batch_generation_ready": True,
                        "results": [
                            {
                                "request_id": "req-1",
                                "prompt_hash": "sha256:p1",
                                "generated_token_count": 2,
                                "max_new_tokens": 2,
                                "generated_text_hash": "sha256:g1",
                                "decoded_tokens_match": True,
                            },
                            {
                                "request_id": "req-2",
                                "prompt_hash": "sha256:p2",
                                "generated_token_count": 2,
                                "max_new_tokens": 2,
                                "generated_text_hash": "sha256:g2",
                                "decoded_tokens_match": True,
                            },
                        ],
                    },
                    "stream": {
                        "enabled": True,
                        "event_count": 4,
                        "endpoint_ready": True,
                        "progress": {
                            "stream_progress_complete": True,
                            "all_token_events_ready": True,
                            "monotonic_progress": True,
                            "expected_request_count": 2,
                            "per_request_progress": [
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
                                {
                                    "request_key": "req-2",
                                    "request_id": "req-2",
                                    "prompt_hash": "sha256:p2",
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
                        },
                        "events": [],
                    },
                }),
                "stderr": "",
            })()

        original_request_json = pack.request_json
        original_run = pack.subprocess.run
        try:
            pack.request_json = fake_request_json  # type: ignore[assignment]
            pack.subprocess.run = fake_run  # type: ignore[assignment]
            report = pack.build_report(pack.parse_args([
                pack.MODE_EXTERNAL_EXISTING,
                "--output-dir",
                str(output_dir / "external"),
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
                "--preview-v04-report",
                str(preview_report),
                "--product-mvp-report",
                str(product_report),
                "--optional-model-report",
                str(optional_report),
                "--json",
            ]))
        finally:
            pack.request_json = original_request_json  # type: ignore[assignment]
            pack.subprocess.run = original_run  # type: ignore[assignment]

        self.assertTrue(captured)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["p2p"]["hf_model_id"], "distilgpt2")
        self.assertEqual(report["p2p"]["observed_hf_model_id"], "distilgpt2")
        self.assertTrue(report["p2p"]["model_id_match"])
        self.assertTrue(report["p2p"]["external_generate_verified"])
        self.assertIn("p2p_v06_model_metadata_ready", report["diagnosis_codes"])
        self.assertIn("p2p_external_generate_batch_ready", report["diagnosis_codes"])
        self.assertIn("p2p_external_generate_stream_ready", report["diagnosis_codes"])

    def test_external_existing_failure_does_not_pass_from_imported_evidence(self) -> None:
        output_dir = self._tmp_dir()
        source_dir = output_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        preview_report = source_dir / "preview.json"
        product_report = source_dir / "product.json"
        optional_report = source_dir / "optional.json"
        preview_report.write_text(json.dumps(check.fake_preview_payload()), encoding="utf-8")
        product_report.write_text(json.dumps(check.fake_product_payload()), encoding="utf-8")
        optional_report.write_text(json.dumps(check.fake_product_payload("distilgpt2")), encoding="utf-8")

        def failing_request_json(base_url: str, path: str, *, timeout: float = 5.0) -> dict:
            raise OSError("unreachable")

        original = pack.request_json
        try:
            pack.request_json = failing_request_json  # type: ignore[assignment]
            report = pack.build_report(pack.parse_args([
                pack.MODE_EXTERNAL_EXISTING,
                "--output-dir",
                str(output_dir / "external-failed"),
                "--peer-bootstrap",
                "http://p2p.example",
                "--preview-v04-report",
                str(preview_report),
                "--product-mvp-report",
                str(product_report),
                "--optional-model-report",
                str(optional_report),
                "--json",
            ]))
        finally:
            pack.request_json = original  # type: ignore[assignment]

        self.assertFalse(report["ok"], report)
        self.assertIn("p2p_swarm_inference_v06_blocked", report["diagnosis_codes"])
        self.assertIn("external_p2p_catalog_unreachable", report["diagnosis_codes"])

    def test_public_report_redacts_sensitive_fragments(self) -> None:
        output_dir = self._tmp_dir()
        report = pack.persist_report(
            {
                "schema": pack.SCHEMA,
                "ok": True,
                "mode": pack.MODE_EVIDENCE_IMPORT,
                "p2p": {},
                "inference": {},
                "steps": [],
                "payload_summaries": {},
                "diagnosis_codes": ["p2p_swarm_inference_v06_ready"],
                "artifacts": {
                    "p2p_swarm_inference_v06_json": pack.artifact_entry(
                        output_dir / "p2p_swarm_inference_v06.json",
                        output_dir,
                        kind="p2p_swarm_inference_v06",
                    )
                },
                "safety": {},
                "completed": [],
                "not_completed": [],
                "secret": "CROWDTENSOR_ADMIN_TOKEN=secret",
            },
            output_dir=output_dir,
        )

        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("CROWDTENSOR_ADMIN_TOKEN=secret", encoded)


if __name__ == "__main__":
    unittest.main()
