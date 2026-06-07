from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from crowdtensor import cli
from scripts import public_real_llm_swarm_beta_check as check
from scripts import public_real_llm_swarm_beta_pack as pack


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


def write_default_sources(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    source_dir = output_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    external_path = source_dir / "external.json"
    p2p_path = source_dir / "p2p.json"
    usable_path = source_dir / "usable.json"
    public_swarm_v2_path = source_dir / "public_swarm_v2.json"
    external_path.write_text(json.dumps(check.fake_external_payload()) + "\n", encoding="utf-8")
    p2p_path.write_text(json.dumps(check.fake_p2p_payload()) + "\n", encoding="utf-8")
    usable_path.write_text(json.dumps(check.fake_usable_payload()) + "\n", encoding="utf-8")
    public_swarm_v2_path.write_text(json.dumps(check.fake_public_swarm_v2_payload()) + "\n", encoding="utf-8")
    return external_path, p2p_path, usable_path, public_swarm_v2_path


def mark_usable_model(payload: dict, model_id: str) -> dict:
    p2p = payload["readiness"]["p2p_product_path"]
    p2p["model"] = {
        "expected_hf_model_id": model_id,
        "observed_hf_model_id": model_id,
        "model_id_present": True,
        "model_id_match": True,
        "compatible": True,
        "default_model_retained_evidence": False,
    }
    return payload


def mark_public_swarm_v2_model(payload: dict, model_id: str) -> dict:
    payload["public_swarm_v2"]["hf_model_id"] = model_id
    for key in ["local_p2p_generate", "external_validation", "p2p_route_hardening"]:
        payload["readiness"][key]["model"] = {
            "expected_hf_model_id": model_id,
            "observed_hf_model_id": model_id,
            "model_id_present": True,
            "model_id_match": True,
            "compatible": True,
            "default_model_retained_evidence": False,
        }
    return payload


def mark_public_swarm_v2_local_model_variant(payload: dict, model_id: str) -> dict:
    payload = mark_public_swarm_v2_model(payload, model_id)
    payload["mode"] = pack.MODE_LOCAL_MODEL_VARIANT
    payload["public_swarm_v2"]["hf_model_id"] = model_id
    payload["public_swarm_v2"]["local_model_variant_only"] = True
    payload["public_swarm_v2"]["external_validation_claimed"] = False
    external_model = payload["readiness"]["external_validation"]["model"]
    external_model["observed_hf_model_id"] = pack.DEFAULT_HF_MODEL_ID
    external_model["model_id_match"] = False
    external_model["compatible"] = False
    payload["readiness"]["external_validation"]["ready"] = False
    payload["readiness"]["external_validation"]["retained_external_evidence_ready"] = False
    payload["diagnosis_codes"] = [
        code
        for code in payload["diagnosis_codes"]
        if code != "public_swarm_inference_v2_ready"
    ]
    payload["diagnosis_codes"].extend([
        "public_swarm_inference_v2_local_model_variant_ready",
        "public_swarm_v2_local_model_variant_ready",
        "public_swarm_v2_external_validation_not_claimed",
        "public_swarm_v2_local_model_variant_model_match_ready",
    ])
    return payload


def fake_v2_if_requested(command: list[str], payload: dict | None = None) -> subprocess.CompletedProcess[str] | None:
    joined = " ".join(command)
    if "public_swarm_inference_v2_pack.py" not in joined:
        return None
    self_payload = payload or check.fake_public_swarm_v2_payload()
    return completed(self_payload)


def write_fresh_p2p_if_requested(command: list[str], output_dir: Path, payload: dict | None = None) -> subprocess.CompletedProcess[str] | None:
    joined = " ".join(command)
    if "petals_class_p2p_candidate_pack.py" not in joined:
        return None
    self_payload = payload or check.fake_p2p_payload()
    if "--output-dir" in command:
        fresh_dir = Path(command[command.index("--output-dir") + 1])
    else:
        fresh_dir = output_dir / "beta" / "p2p-candidate"
    fresh_dir.mkdir(parents=True, exist_ok=True)
    (fresh_dir / "petals_class_p2p_candidate.json").write_text(json.dumps(self_payload) + "\n", encoding="utf-8")
    return completed(self_payload)


def p2p_candidate_commands(commands: list[list[str]]) -> list[list[str]]:
    return [command for command in commands if "petals_class_p2p_candidate_pack.py" in " ".join(command)]


def public_swarm_v2_commands(commands: list[list[str]]) -> list[list[str]]:
    return [command for command in commands if "public_swarm_inference_v2_pack.py" in " ".join(command)]


class PublicRealLlmSwarmBetaPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_public_real_llm_beta_test_"))

    def test_release_aggregates_product_external_p2p_and_cuda_fail_closed(self) -> None:
        output_dir = self._tmp_dir()
        external_path, p2p_path, usable_path, public_swarm_v2_path = write_default_sources(output_dir)

        args = pack.parse_args([
            "release",
            "--output-dir",
            str(output_dir / "beta"),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--base-port",
            "9440",
            "--port",
            "9440",
            "--timeout-seconds",
            "60",
        ])

        commands: list[list[str]] = []

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            commands.append(command)
            joined = " ".join(command)
            p2p_completed = write_fresh_p2p_if_requested(command, output_dir)
            if p2p_completed is not None:
                return p2p_completed
            v2_completed = fake_v2_if_requested(command)
            if v2_completed is not None:
                return v2_completed
            if "public_swarm_product_beta_pack.py" in joined:
                return completed(check.fake_product_payload())
            if "public_swarm_gpu_inference_beta_pack.py" in joined:
                return completed(check.fake_gpu_payload())
            raise AssertionError(command)

        report = pack.build_report(args, runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], pack.SCHEMA)
        self.assertTrue(report["beta"]["cpu_default_ready"])
        self.assertTrue(report["beta"]["external_two_stage_ready"])
        self.assertTrue(report["beta"]["external_stage_requeue_ready"])
        self.assertTrue(report["beta"]["p2p_ready_product_beta"])
        self.assertTrue(report["beta"]["p2p_live_requeue_ready"])
        self.assertTrue(report["beta"]["p2p_victim_result_not_accepted"])
        self.assertTrue(report["beta"]["p2p_batch_ready"])
        self.assertTrue(report["beta"]["p2p_stream_ready"])
        self.assertTrue(report["beta"]["public_swarm_v2_ready"])
        self.assertTrue(report["beta"]["public_swarm_v2_batch_ready"])
        self.assertTrue(report["beta"]["public_swarm_v2_stream_ready"])
        self.assertTrue(report["beta"]["public_swarm_v2_real_p2p_local_ready"])
        self.assertTrue(report["beta"]["public_swarm_v2_real_p2p_local_requeue_ready"])
        self.assertTrue(report["beta"]["kv_cache_ready"])
        self.assertTrue(report["beta"]["cuda_optional_fail_closed_ready"])
        self.assertIn("public_real_llm_swarm_beta_ready", report["diagnosis_codes"])
        self.assertIn("p2p_live_requeue_rescue_ready", report["diagnosis_codes"])
        self.assertIn("p2p_victim_result_not_accepted", report["diagnosis_codes"])
        self.assertIn("public_real_llm_swarm_beta_p2p_batch_ready", report["diagnosis_codes"])
        self.assertIn("public_real_llm_swarm_beta_p2p_stream_ready", report["diagnosis_codes"])
        self.assertIn("public_real_llm_swarm_beta_kv_cache_ready", report["diagnosis_codes"])
        self.assertIn("public_real_llm_swarm_beta_product_model_match_ready", report["diagnosis_codes"])
        self.assertIn("public_real_llm_swarm_beta_kv_cache_model_match_ready", report["diagnosis_codes"])
        self.assertIn("public_real_llm_swarm_beta_public_swarm_v2_ready", report["diagnosis_codes"])
        self.assertIn("public_real_llm_swarm_beta_p2p_user_path_ready", report["diagnosis_codes"])
        self.assertIn("public_real_llm_swarm_beta_v2_real_p2p_local_ready", report["diagnosis_codes"])
        self.assertIn("public_real_llm_swarm_beta_v2_real_p2p_local_requeue_ready", report["diagnosis_codes"])
        self.assertIn("public_real_llm_swarm_beta_v2_batch_ready", report["diagnosis_codes"])
        self.assertIn("public_real_llm_swarm_beta_v2_stream_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_inference_v2_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_real_p2p_local_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_dual_stage_kv_cache_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_model_match_ready", report["diagnosis_codes"])
        self.assertIn("stage0_kv_cache_hits_ready", report["diagnosis_codes"])
        self.assertIn("stage1_kv_cache_hits_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_endpoint_ready", report["diagnosis_codes"])
        self.assertFalse(
            [code for code in report["diagnosis_codes"] if code.endswith("_blocked")],
            report["diagnosis_codes"],
        )
        self.assertTrue(report["readiness"]["product_path"]["model"]["compatible"])
        self.assertTrue(report["readiness"]["external_kaggle"]["model"]["model_id_match"])
        self.assertTrue(report["readiness"]["p2p_candidate"]["model"]["model_id_match"])
        self.assertTrue(report["readiness"]["public_swarm_v2"]["ready"])
        self.assertTrue(report["readiness"]["public_swarm_v2"]["model"]["compatible"])
        self.assertEqual(report["readiness"]["public_swarm_v2"]["generated_token_count"], 16)
        self.assertEqual(report["readiness"]["public_swarm_v2"]["accepted_rows"], 32)
        self.assertEqual(report["readiness"]["public_swarm_v2"]["stage0_kv_cache_hits"], 15)
        self.assertEqual(report["readiness"]["public_swarm_v2"]["stage1_kv_cache_hits"], 15)
        self.assertTrue(report["readiness"]["public_swarm_v2"]["real_p2p_local_route_hardening_ready"])
        self.assertTrue(report["readiness"]["public_swarm_v2"]["real_p2p_local_stage_requeue_ready"])
        self.assertEqual(report["readiness"]["public_swarm_v2"]["real_p2p_local_stage_requeue_target"], "stage1")
        self.assertEqual(report["readiness"]["public_swarm_v2"]["real_p2p_local_generated_token_count"], "<redacted>")
        self.assertEqual(report["readiness"]["public_swarm_v2"]["real_p2p_local_discovery_backend"], "http-provider-store")
        self.assertTrue(report["readiness"]["usable_p2p_kv_cache"]["model"]["model_id_match"])
        self.assertTrue(report["readiness"]["p2p_candidate"]["batch"]["batch_generation_ready"])
        self.assertTrue(report["readiness"]["p2p_candidate"]["stream"]["stream_generation_ready"])
        self.assertTrue(report["readiness"]["usable_p2p_kv_cache"]["ready"])
        self.assertTrue(report["release_private_artifact_cleanup"]["private_artifacts_cleaned"])
        self.assertIn("public_real_llm_swarm_beta_private_artifacts_cleaned", report["diagnosis_codes"])
        self.assertEqual(report["readiness"]["usable_p2p_kv_cache"]["generated_token_count"], 16)
        self.assertEqual(report["readiness"]["usable_p2p_kv_cache"]["stage0"]["hit_count"], 15)
        self.assertEqual(report["readiness"]["usable_p2p_kv_cache"]["stage1"]["hit_count"], 15)
        self.assertEqual(len([command for command in report["steps"] if command["name"] == "public_swarm_v2_local_p2p_generate"]), 1)
        self.assertEqual(len([command for command in report["steps"] if command["name"] == "petals_class_p2p_candidate_local_smoke"]), 1)
        [v2_command] = public_swarm_v2_commands(commands)
        self.assertIn("--real-p2p-port", v2_command)
        self.assertEqual(v2_command[v2_command.index("--real-p2p-port") + 1], "9890")
        self.assertIn("--real-p2p-coordinator-port", v2_command)
        self.assertEqual(v2_command[v2_command.index("--real-p2p-coordinator-port") + 1], "9891")
        self.assertIn("--real-p2p-libp2p-port", v2_command)
        self.assertEqual(v2_command[v2_command.index("--real-p2p-libp2p-port") + 1], "0")
        self.assertIn("--real-p2p-discovery-backend", v2_command)
        self.assertEqual(v2_command[v2_command.index("--real-p2p-discovery-backend") + 1], "http-provider-store")
        [p2p_command] = p2p_candidate_commands(commands)
        self.assertIn("--timeout-seconds", p2p_command)
        self.assertGreater(float(p2p_command[p2p_command.index("--timeout-seconds") + 1]), 1800.0)
        self.assertIn("/p2p-candidate/petals_class_p2p_candidate.json", report["source_reports"]["p2p_report"])
        self.assertIn("/public-swarm-v2/public_swarm_inference_v2.json", report["source_reports"]["public_swarm_v2_report"])
        self.assertIn("/public-swarm-v2/public_swarm_inference_v2.json", report["source_reports"]["usable_report"])
        self.assertFalse(report["readiness"]["p2p_candidate"]["live_requeue_summary"]["victim_result_accepted"])
        self.assertTrue((output_dir / "beta" / "public_real_llm_swarm_beta.json").is_file())
        self.assertTrue((output_dir / "beta" / "PUBLIC_REAL_LLM_SWARM_BETA.md").is_file())
        runbook = (output_dir / "beta" / "PUBLIC_REAL_LLM_SWARM_BETA.md").read_text(encoding="utf-8")
        self.assertIn("## Verify The Full Beta Contract", runbook)
        self.assertIn("crowdtensor public-real-llm-swarm-beta release", runbook)
        self.assertIn("python scripts/public_real_llm_swarm_beta_check.py", runbook)
        self.assertIn("## Review The Result", runbook)
        self.assertIn("public_real_llm_swarm_beta.md", runbook)
        self.assertIn("`Review`, `Operator Action`, and `Not Completed`", runbook)
        self.assertIn("support_bundle.json", runbook)
        self.assertIn("## Share Safely", runbook)
        self.assertIn("raw prompts, generated text, generated token ids, credentials, activations, and lease tokens are excluded", runbook)
        self.assertIn("## Troubleshooting", runbook)
        self.assertIn("stage0/stage1 hit counts", runbook)
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
        self.assertEqual(report["artifact_summary"]["schema"], pack.ARTIFACT_SUMMARY_SCHEMA)
        self.assertEqual(report["artifact_summary"]["inspect_first"], "public_real_llm_swarm_beta.md")
        self.assertEqual(report["artifact_summary"]["machine_readable"], "public_real_llm_swarm_beta.json")
        self.assertEqual(report["artifact_summary"]["support_bundle"], "support_bundle.json")
        self.assertEqual(report["artifact_summary"]["runbook"], "PUBLIC_REAL_LLM_SWARM_BETA.md")
        self.assertEqual(
            report["artifact_summary"]["shareable_paths"],
            ["public_real_llm_swarm_beta.json", "public_real_llm_swarm_beta.md", "support_bundle.json"],
        )
        self.assertTrue(report["artifact_summary"]["public_artifact_safe"])
        self.assertFalse(report["artifact_summary"]["raw_prompt_public"])
        self.assertFalse(report["artifact_summary"]["raw_generated_text_public"])
        self.assertFalse(report["artifact_summary"]["generated_token_ids_public"])
        self.assertEqual(report["review_summary"]["schema"], pack.REVIEW_SUMMARY_SCHEMA)
        self.assertEqual(report["review_summary"]["state"], "ready")
        self.assertTrue(report["review_summary"]["ready"])
        self.assertEqual(report["review_summary"]["next_step"], "share_public_artifacts")
        self.assertEqual(report["review_summary"]["inspect_first"], "public_real_llm_swarm_beta.md")
        self.assertEqual(report["review_summary"]["support_bundle"], "support_bundle.json")
        self.assertEqual(report["review_summary"]["not_completed_count"], 0)
        self.assertTrue(report["review_summary"]["public_artifact_safe"])
        self.assertFalse(report["review_summary"]["raw_prompt_public"])
        self.assertFalse(report["review_summary"]["raw_generated_text_public"])
        self.assertFalse(report["review_summary"]["generated_token_ids_public"])
        markdown = (output_dir / "beta" / "public_real_llm_swarm_beta.md").read_text(encoding="utf-8")
        self.assertIn("## Review", markdown)
        self.assertIn("- state: `ready`", markdown)
        self.assertIn("- next step: `share_public_artifacts`", markdown)
        self.assertIn("- inspect first: `public_real_llm_swarm_beta.md`", markdown)
        self.assertIn("## Operator Action", markdown)
        self.assertIn(report["operator_action"][0], markdown)
        self.assertIn("## Artifacts", markdown)
        self.assertIn("- machine readable: `public_real_llm_swarm_beta.json`", markdown)
        self.assertIn("- support bundle: `support_bundle.json`", markdown)
        self.assertIn("## Output Scope", markdown)
        self.assertIn("- answer scope: `no-local-answer`", markdown)
        self.assertIn(
            "- shareable: `saved_artifacts=True raw_prompt_public=False raw_generated_text_public=False generated_token_ids_public=False answer_scope_state=no-local-answer local_answer_terminal_only=False`",
            markdown,
        )
        support = json.loads((output_dir / "beta" / "support_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(support["artifact_summary"]["inspect_first"], "public_real_llm_swarm_beta.md")
        self.assertEqual(support["review_summary"]["next_step"], "share_public_artifacts")
        self.assertEqual(support["operator_action"], report["operator_action"])
        self.assertEqual(support["not_completed"], report["not_completed"])
        self.assertTrue(support["release_private_artifact_cleanup"]["private_artifacts_cleaned"])
        self.assertEqual(support["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(support["shareable_summary"]["answer_scope_state"], "no-local-answer")
        self.assertNotIn('"generated_text":', encoded)
        self.assertNotIn('"generated_token_ids":', encoded)
        self.assertNotIn("CROWDTENSOR_ADMIN_TOKEN=", encoded)

    def test_release_filters_superseded_product_child_blockers_when_product_path_ready(self) -> None:
        output_dir = self._tmp_dir()
        external_path, p2p_path, usable_path, public_swarm_v2_path = write_default_sources(output_dir)
        args = pack.parse_args([
            "release",
            "--output-dir",
            str(output_dir / "beta"),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--base-port",
            "9447",
            "--port",
            "9447",
            "--timeout-seconds",
            "60",
        ])
        product_payload = check.fake_product_payload()
        product_payload["payload_summaries"] = {
            "legacy_rc": {
                "diagnosis_codes": [
                    "p2p_lite_route_blocked",
                    "p2p_lite_discovery_blocked",
                    "public_swarm_inference_beta_blocked",
                    "public_swarm_inference_beta_rc_blocked",
                    "public_swarm_product_beta_blocked",
                    "public_swarm_product_rc_blocked",
                ],
            }
        }

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            p2p_completed = write_fresh_p2p_if_requested(command, output_dir)
            if p2p_completed is not None:
                return p2p_completed
            v2_completed = fake_v2_if_requested(command)
            if v2_completed is not None:
                return v2_completed
            if "public_swarm_product_beta_pack.py" in joined:
                return completed(product_payload)
            if "public_swarm_gpu_inference_beta_pack.py" in joined:
                return completed(check.fake_gpu_payload())
            raise AssertionError(command)

        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertFalse(
            [code for code in report["diagnosis_codes"] if code.endswith("_blocked")],
            report["diagnosis_codes"],
        )

    def test_persist_report_prunes_generated_private_runtime_files(self) -> None:
        output_dir = self._tmp_dir() / "beta"
        (output_dir / "child" / "state").mkdir(parents=True, exist_ok=True)
        (output_dir / "child" / "state" / "tasks.jsonl").write_text('{"generated_text":"raw"}\n', encoding="utf-8")
        (output_dir / "child" / "libp2p-bootstrap-peer-key.json").write_text('{"private":"key"}\n', encoding="utf-8")
        (output_dir / "child" / "miner.private.env").write_text("TOKEN=secret\n", encoding="utf-8")
        report = {
            "schema": pack.SCHEMA,
            "generated_at": "2026-06-02T00:00:00+00:00",
            "ok": True,
            "mode": "release",
            "beta": {"ready": True},
            "readiness": {},
            "diagnosis_codes": [],
            "artifacts": {},
            "safety": {},
            "limitations": [],
            "not_completed": [],
        }

        persisted = pack.persist_report(report, output_dir=output_dir)

        self.assertTrue(persisted["ok"], persisted)
        self.assertTrue(persisted["release_private_artifact_cleanup"]["private_artifacts_cleaned"])
        self.assertEqual(persisted["release_private_artifact_cleanup"]["removed_private_artifact_count"], 4)
        self.assertFalse((output_dir / "child" / "state").exists())
        self.assertFalse((output_dir / "child" / "libp2p-bootstrap-peer-key.json").exists())
        self.assertFalse((output_dir / "child" / "miner.private.env").exists())

    def test_release_prefers_fresh_usable_report_written_by_public_swarm_v2(self) -> None:
        output_dir = self._tmp_dir()
        external_path, p2p_path, usable_path, public_swarm_v2_path = write_default_sources(output_dir)
        external = check.fake_external_payload()
        external["workload"]["max_new_tokens"] = 16
        external["generation"]["generated_token_count"] = 16
        external["generation"]["max_new_tokens"] = 16
        p2p = check.fake_p2p_payload()
        p2p["candidate"]["external_generated_token_count"] = 16
        p2p["candidate"]["accepted_rows"] = 32
        p2p["generation"]["generated_token_count"] = 16
        p2p["generation"]["max_new_tokens"] = 16
        external_path.write_text(json.dumps(external) + "\n", encoding="utf-8")
        p2p_path.write_text(json.dumps(p2p) + "\n", encoding="utf-8")
        fresh_usable = check.fake_usable_payload()
        fresh_usable["readiness"]["p2p_product_path"]["generated_token_count"] = 16
        fresh_usable["readiness"]["p2p_product_path"]["generation"]["generated_token_count"] = 16
        fresh_usable["readiness"]["p2p_product_path"]["kv_cache"]["stage0"]["hit_count"] = 15
        fresh_usable["readiness"]["p2p_product_path"]["kv_cache"]["stage0"]["expected_hit_count"] = 15
        fresh_usable["readiness"]["p2p_product_path"]["kv_cache"]["stage1"]["hit_count"] = 15
        fresh_usable["readiness"]["p2p_product_path"]["kv_cache"]["stage1"]["expected_hit_count"] = 15

        args = pack.parse_args([
            "release",
            "--output-dir",
            str(output_dir / "beta"),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--max-new-tokens",
            "16",
            "--base-port",
            "9441",
            "--port",
            "9441",
            "--timeout-seconds",
            "60",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            p2p_completed = write_fresh_p2p_if_requested(command, output_dir, p2p)
            if p2p_completed is not None:
                return p2p_completed
            if "public_swarm_inference_v2_pack.py" in joined:
                fresh_path = output_dir / "beta" / "public-swarm-v2" / "usable-v1-local" / "usable_swarm_inference.json"
                fresh_path.parent.mkdir(parents=True, exist_ok=True)
                fresh_path.write_text(json.dumps(fresh_usable) + "\n", encoding="utf-8")
                return completed(check.fake_public_swarm_v2_payload(16))
            if "public_swarm_product_beta_pack.py" in joined:
                product = check.fake_product_payload()
                product["product_beta"]["max_new_tokens"] = 16
                return completed(product)
            if "public_swarm_gpu_inference_beta_pack.py" in joined:
                return completed(check.fake_gpu_payload())
            raise AssertionError(command)

        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertIn("/p2p-candidate/petals_class_p2p_candidate.json", report["source_reports"]["p2p_report"])
        self.assertIn("/public-swarm-v2/usable-v1-local/usable_swarm_inference.json", report["source_reports"]["usable_report"])
        self.assertEqual(report["readiness"]["usable_p2p_kv_cache"]["generated_token_count"], 16)
        self.assertEqual(report["readiness"]["usable_p2p_kv_cache"]["stage0"]["hit_count"], 15)
        self.assertEqual(report["readiness"]["usable_p2p_kv_cache"]["stage1"]["hit_count"], 15)

    def test_evidence_import_blocks_when_usable_kv_cache_report_missing(self) -> None:
        output_dir = self._tmp_dir()
        external_path, p2p_path, _usable_path, public_swarm_v2_path = write_default_sources(output_dir)
        product_path = output_dir / "sources" / "product.json"
        gpu_path = output_dir / "sources" / "gpu.json"
        product_path.write_text(json.dumps(check.fake_product_payload()) + "\n", encoding="utf-8")
        gpu_path.write_text(json.dumps(check.fake_gpu_payload()) + "\n", encoding="utf-8")

        args = pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "beta"),
            "--product-report",
            str(product_path),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(output_dir / "missing-usable.json"),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--gpu-report",
            str(gpu_path),
            "--base-port",
            "9441",
            "--port",
            "9441",
            "--timeout-seconds",
            "60",
        ])

        report = pack.build_report(args)

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["beta"]["kv_cache_ready"])
        self.assertFalse(report["readiness"]["usable_p2p_kv_cache"]["ready"])
        self.assertIn("public_real_llm_swarm_beta_kv_cache_missing", report["diagnosis_codes"])
        self.assertIn("persistent dual-stage KV-cache reuse", report["not_completed"])
        self.assertEqual(report["review_summary"]["state"], "blocked")
        self.assertFalse(report["review_summary"]["ready"])
        self.assertEqual(report["review_summary"]["next_step"], "review_not_completed")
        self.assertEqual(report["review_summary"]["not_completed_count"], len(report["not_completed"]))
        self.assertIn("persistent dual-stage KV-cache reuse", report["review_summary"]["not_completed_preview"])
        markdown = (output_dir / "beta" / "public_real_llm_swarm_beta.md").read_text(encoding="utf-8")
        self.assertIn("## Review", markdown)
        self.assertIn("- state: `blocked`", markdown)
        self.assertIn("- next step: `review_not_completed`", markdown)
        self.assertIn("## Operator Action", markdown)
        self.assertIn(report["operator_action"][0], markdown)
        self.assertIn("## Not Completed", markdown)
        self.assertIn("- persistent dual-stage KV-cache reuse", markdown)
        support = json.loads((output_dir / "beta" / "support_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(support["review_summary"]["next_step"], "review_not_completed")
        self.assertEqual(support["not_completed"], report["not_completed"])
        self.assertIn("persistent dual-stage KV-cache reuse", support["not_completed"])
        self.assertEqual(support["operator_action"], report["operator_action"])

    def test_evidence_import_blocks_when_public_swarm_v2_report_missing(self) -> None:
        output_dir = self._tmp_dir()
        product_path = output_dir / "sources" / "product.json"
        external_path, p2p_path, usable_path, _public_swarm_v2_path = write_default_sources(output_dir)
        gpu_path = output_dir / "sources" / "gpu.json"
        product_path.write_text(json.dumps(check.fake_product_payload()) + "\n", encoding="utf-8")
        gpu_path.write_text(json.dumps(check.fake_gpu_payload()) + "\n", encoding="utf-8")

        args = pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "beta"),
            "--product-report",
            str(product_path),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(output_dir / "missing-v2.json"),
            "--gpu-report",
            str(gpu_path),
        ])

        report = pack.build_report(args)

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["beta"]["public_swarm_v2_ready"])
        self.assertFalse(report["readiness"]["public_swarm_v2"]["ready"])
        self.assertIn("public_swarm_v2_token_target_missing", report["diagnosis_codes"])
        self.assertIn("Public Swarm v2 ordinary P2P user path", report["not_completed"])
        markdown = (output_dir / "beta" / "public_real_llm_swarm_beta.md").read_text(encoding="utf-8")
        self.assertIn("## Not Completed", markdown)
        self.assertIn("- Public Swarm v2 ordinary P2P user path", markdown)

    def test_evidence_import_blocks_when_public_swarm_v2_batch_or_stream_missing(self) -> None:
        output_dir = self._tmp_dir()
        product_path = output_dir / "sources" / "product.json"
        external_path, p2p_path, usable_path, public_swarm_v2_path = write_default_sources(output_dir)
        gpu_path = output_dir / "sources" / "gpu.json"
        product_path.write_text(json.dumps(check.fake_product_payload()) + "\n", encoding="utf-8")
        gpu_path.write_text(json.dumps(check.fake_gpu_payload()) + "\n", encoding="utf-8")
        v2_payload = check.fake_public_swarm_v2_payload()
        v2_payload["readiness"]["local_p2p_generate"]["batch_ready"] = False
        v2_payload["readiness"]["local_p2p_generate"]["stream_ready"] = False
        v2_payload["readiness"]["local_p2p_generate"].pop("batch", None)
        v2_payload["readiness"]["local_p2p_generate"].pop("stream", None)
        v2_payload["diagnosis_codes"] = [
            code for code in v2_payload["diagnosis_codes"]
            if code not in {"public_swarm_v2_batch_generation_ready", "public_swarm_v2_stream_generation_ready"}
        ]
        public_swarm_v2_path.write_text(json.dumps(v2_payload) + "\n", encoding="utf-8")

        args = pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "beta"),
            "--product-report",
            str(product_path),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--gpu-report",
            str(gpu_path),
        ])

        report = pack.build_report(args)

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["beta"]["public_swarm_v2_ready"])
        self.assertFalse(report["readiness"]["public_swarm_v2"]["batch_ready"])
        self.assertFalse(report["readiness"]["public_swarm_v2"]["stream_ready"])
        self.assertIn("public_swarm_v2_batch_generation_missing", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_stream_generation_missing", report["diagnosis_codes"])
        self.assertIn("Public Swarm v2 batch generation", report["not_completed"])
        self.assertIn("Public Swarm v2 stream generation", report["not_completed"])
        markdown = (output_dir / "beta" / "public_real_llm_swarm_beta.md").read_text(encoding="utf-8")
        self.assertIn("## Not Completed", markdown)
        self.assertIn("- Public Swarm v2 batch generation", markdown)
        self.assertIn("- Public Swarm v2 stream generation", markdown)

    def test_evidence_import_blocks_when_public_swarm_v2_real_p2p_local_route_missing(self) -> None:
        output_dir = self._tmp_dir()
        product_path = output_dir / "sources" / "product.json"
        external_path, p2p_path, usable_path, public_swarm_v2_path = write_default_sources(output_dir)
        gpu_path = output_dir / "sources" / "gpu.json"
        product_path.write_text(json.dumps(check.fake_product_payload()) + "\n", encoding="utf-8")
        gpu_path.write_text(json.dumps(check.fake_gpu_payload()) + "\n", encoding="utf-8")
        v2_payload = check.fake_public_swarm_v2_payload()
        v2_payload["readiness"]["real_p2p_local_route_hardening"]["ready"] = False
        v2_payload["readiness"]["real_p2p_local_route_hardening"]["stage_requeue_ready"] = True
        public_swarm_v2_path.write_text(json.dumps(v2_payload) + "\n", encoding="utf-8")

        args = pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "beta"),
            "--product-report",
            str(product_path),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--gpu-report",
            str(gpu_path),
        ])

        report = pack.build_report(args)

        self.assertFalse(report["ok"], report)
        self.assertTrue(report["readiness"]["public_swarm_v2"]["ready"])
        self.assertFalse(report["beta"]["public_swarm_v2_real_p2p_local_ready"])
        self.assertFalse(report["readiness"]["public_swarm_v2"]["real_p2p_local_route_hardening_ready"])
        self.assertFalse(report["readiness"]["public_swarm_v2"]["real_p2p_local_stage_requeue_ready"])
        self.assertIn("public_swarm_v2_real_p2p_local_missing", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_real_p2p_local_requeue_missing", report["diagnosis_codes"])
        self.assertIn("Public Swarm v2 fresh real-P2P local route hardening", report["not_completed"])
        self.assertIn("Public Swarm v2 fresh real-P2P local stage requeue", report["not_completed"])

    def test_release_preserves_product_batch_readiness(self) -> None:
        output_dir = self._tmp_dir()
        external_path, p2p_path, usable_path, public_swarm_v2_path = write_default_sources(output_dir)

        product_payload = check.fake_product_payload()
        product_payload["product_beta"]["batch"] = {
            "enabled": True,
            "request_count": 2,
            "prompt_hashes": ["sha256:a", "sha256:b"],
            "prompt_char_counts": [12, 13],
            "results": [
                {
                    "request_id": "req-1",
                    "prompt_hash": "sha256:a",
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "generated_text_hash": "sha256:g1",
                    "multi_token_generation_ready": True,
                },
                {
                    "request_id": "req-2",
                    "prompt_hash": "sha256:b",
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "generated_text_hash": "sha256:g2",
                    "multi_token_generation_ready": True,
                },
            ],
            "batch_generation_ready": True,
        }
        product_payload["diagnosis_codes"].append("public_swarm_generate_batch_ready")

        args = pack.parse_args([
            "release",
            "--output-dir",
            str(output_dir / "beta"),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--prompt-texts",
            "first prompt,second prompt",
            "--base-port",
            "9445",
            "--port",
            "9445",
            "--timeout-seconds",
            "60",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            p2p_completed = write_fresh_p2p_if_requested(command, output_dir, json.loads(p2p_path.read_text(encoding="utf-8")))
            if p2p_completed is not None:
                return p2p_completed
            v2_completed = fake_v2_if_requested(command)
            if v2_completed is not None:
                return v2_completed
            if "public_swarm_product_beta_pack.py" in joined:
                self.assertIn("--prompt-texts", command)
                self.assertEqual(command[command.index("--prompt-texts") + 1], "first prompt,second prompt")
                self.assertNotIn("--prompt-text", command)
                return completed(product_payload)
            if "public_swarm_gpu_inference_beta_pack.py" in joined:
                return completed(check.fake_gpu_payload())
            raise AssertionError(command)

        report = pack.build_report(args, runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["beta"]["batch"]["batch_generation_ready"])
        self.assertIn("public_real_llm_swarm_beta_batch_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_batch_ready", report["diagnosis_codes"])
        self.assertNotIn("first prompt", encoded)
        self.assertNotIn("second prompt", encoded)

    def test_release_blocks_v2_batch_ready_code_without_batch_evidence(self) -> None:
        output_dir = self._tmp_dir()
        external_path, p2p_path, usable_path, public_swarm_v2_path = write_default_sources(output_dir)

        public_swarm_v2 = check.fake_public_swarm_v2_payload(16)
        local = public_swarm_v2["readiness"]["local_p2p_generate"]
        local["batch_ready"] = False
        local["batch"] = {
            "enabled": True,
            "expected_request_count": 2,
            "observed_request_count": 1,
            "result_count": 1,
            "batch_generation_ready": False,
            "raw_prompts_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        }
        public_swarm_v2["diagnosis_codes"].extend([
            "public_swarm_v2_batch_generation_ready",
            "public_swarm_generate_batch_ready",
        ])
        public_swarm_v2_path.write_text(json.dumps(public_swarm_v2) + "\n", encoding="utf-8")

        args = pack.parse_args([
            "release",
            "--output-dir",
            str(output_dir / "beta"),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--base-port",
            "9446",
            "--port",
            "9446",
            "--timeout-seconds",
            "60",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            p2p_completed = write_fresh_p2p_if_requested(command, output_dir, json.loads(p2p_path.read_text(encoding="utf-8")))
            if p2p_completed is not None:
                return p2p_completed
            v2_completed = fake_v2_if_requested(command, public_swarm_v2)
            if v2_completed is not None:
                return v2_completed
            if "public_swarm_product_beta_pack.py" in joined:
                return completed(check.fake_product_payload())
            if "public_swarm_gpu_inference_beta_pack.py" in joined:
                return completed(check.fake_gpu_payload())
            raise AssertionError(command)

        report = pack.build_report(args, runner=fake_runner)

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["readiness"]["public_swarm_v2"]["batch_ready"])
        self.assertNotIn("public_real_llm_swarm_beta_v2_batch_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_v2_batch_generation_ready", report["diagnosis_codes"])
        self.assertIn("public_real_llm_swarm_beta_p2p_batch_ready", report["diagnosis_codes"])
        self.assertIn("Public Swarm v2 batch generation", report["not_completed"])

    def test_public_swarm_v2_summary_does_not_trust_local_batch_ready_without_identity(self) -> None:
        payload = check.fake_public_swarm_v2_payload(16)
        local = payload["readiness"]["local_p2p_generate"]
        local["batch_ready"] = True
        local["batch"] = {
            "enabled": True,
            "expected_request_count": 2,
            "observed_request_count": 2,
            "result_count": 2,
            "results": [
                {
                    "request_id": "req-1",
                    "prompt_hash": "sha256:p1",
                    "generated_token_count": 16,
                    "max_new_tokens": 16,
                    "generated_text_hash": "sha256:g1",
                    "multi_token_generation_ready": True,
                },
                {
                    "request_id": "req-1",
                    "prompt_hash": "sha256:p1",
                    "generated_token_count": 16,
                    "max_new_tokens": 16,
                    "generated_text_hash": "sha256:g1-dup",
                    "multi_token_generation_ready": True,
                },
            ],
            "batch_generation_ready": True,
        }
        payload["diagnosis_codes"].extend([
            "public_swarm_v2_batch_generation_ready",
            "public_swarm_generate_batch_ready",
        ])

        summary = pack.public_swarm_v2_summary(payload, min_generated_tokens=16)

        self.assertFalse(summary["ready"], summary)
        self.assertFalse(summary["batch_ready"])
        self.assertFalse(summary["batch"]["batch_identity_ready"])
        self.assertFalse(summary["batch"]["batch_generation_ready"])

    def test_public_swarm_v2_summary_does_not_trust_local_stream_ready_without_per_request_progress(self) -> None:
        payload = check.fake_public_swarm_v2_payload(16)
        local = payload["readiness"]["local_p2p_generate"]
        local["stream_ready"] = True
        local["stream"] = {
            "enabled": True,
            "requested": True,
            "event_count": 32,
            "source": "admin-session-stream",
            "endpoint_ready": True,
            "stream_generation_ready": True,
            "progress": {
                "stream_progress_complete": True,
                "all_token_events_ready": True,
                "monotonic_progress": True,
                "expected_request_count": 2,
                "observed_token_counts": list(range(1, 17)),
                "max_observed_token_count": 16,
                "max_new_tokens": 16,
                "source": "admin-session-stream",
            },
            "events": [],
        }
        payload["diagnosis_codes"].extend([
            "public_swarm_v2_stream_generation_ready",
            "public_swarm_generate_stream_ready",
        ])

        summary = pack.public_swarm_v2_summary(payload, min_generated_tokens=16)

        self.assertFalse(summary["ready"], summary)
        self.assertFalse(summary["stream_ready"])
        self.assertFalse(summary["stream"]["stream_generation_ready"])

    def test_release_preserves_product_stream_readiness(self) -> None:
        output_dir = self._tmp_dir()
        external_path, p2p_path, usable_path, public_swarm_v2_path = write_default_sources(output_dir)

        product_payload = check.fake_product_payload()
        product_payload["product_beta"]["stream"] = {
            "enabled": True,
            "requested": True,
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
                    "generated_token_count": 2,
                    "max_new_tokens": 2,
                    "generation_step": 1,
                    "generated_text_hash": "sha256:stream",
                    "decoded_tokens_match": True,
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                }
            ],
            "stream_generation_ready": True,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        }
        product_payload["diagnosis_codes"].extend([
            "public_swarm_generate_stream_ready",
            "public_swarm_generate_stream_endpoint_ready",
        ])

        args = pack.parse_args([
            "release",
            "--output-dir",
            str(output_dir / "beta"),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--stream-generation",
            "--base-port",
            "9446",
            "--port",
            "9446",
            "--timeout-seconds",
            "60",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            p2p_completed = write_fresh_p2p_if_requested(command, output_dir, json.loads(p2p_path.read_text(encoding="utf-8")))
            if p2p_completed is not None:
                return p2p_completed
            v2_completed = fake_v2_if_requested(command)
            if v2_completed is not None:
                return v2_completed
            if "public_swarm_product_beta_pack.py" in joined:
                self.assertIn("--stream-generation", command)
                return completed(product_payload)
            if "public_swarm_gpu_inference_beta_pack.py" in joined:
                return completed(check.fake_gpu_payload())
            raise AssertionError(command)

        report = pack.build_report(args, runner=fake_runner)
        encoded = json.dumps(report, sort_keys=True)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["beta"]["stream"]["stream_generation_ready"])
        self.assertTrue(report["readiness"]["product_path"]["stream"]["stream_generation_ready"])
        self.assertIn("public_real_llm_swarm_beta_stream_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_generate_stream_endpoint_ready", report["diagnosis_codes"])
        self.assertNotIn('"generated_text":', encoded)
        self.assertNotIn('"generated_token_ids":', encoded)

    def test_prompt_batch_rejects_more_than_four_prompts(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args([
                "release",
                "--prompt-texts",
                "one,two,three,four,five",
            ])

    def test_p2p_summary_does_not_accept_batch_ready_code_without_batch_evidence(self) -> None:
        payload = check.fake_p2p_payload()
        payload["candidate"]["batch"] = {
            "enabled": True,
            "expected_request_count": 2,
            "observed_request_count": 1,
            "result_count": 1,
            "batch_generation_ready": False,
            "raw_prompts_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        }
        payload["diagnosis_codes"].extend([
            "public_swarm_generate_batch_ready",
            "public_real_llm_swarm_beta_p2p_batch_ready",
        ])

        p2p = pack.p2p_summary(payload, min_generated_tokens=8)

        self.assertTrue(p2p["ready"], p2p)
        self.assertFalse(p2p["batch_ready"])
        self.assertFalse(p2p["batch"]["batch_generation_ready"])

    def test_p2p_summary_does_not_accept_batch_with_duplicate_request_identity(self) -> None:
        payload = check.fake_p2p_payload()
        payload["candidate"]["batch"] = {
            "enabled": True,
            "expected_request_count": 2,
            "observed_request_count": 2,
            "result_count": 2,
            "results": [
                {
                    "request_id": "req-1",
                    "prompt_hash": "sha256:p1",
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "generated_text_hash": "sha256:g1",
                    "multi_token_generation_ready": True,
                },
                {
                    "request_id": "req-1",
                    "prompt_hash": "sha256:p1",
                    "generated_token_count": 8,
                    "max_new_tokens": 8,
                    "generated_text_hash": "sha256:g1-dup",
                    "multi_token_generation_ready": True,
                },
            ],
            "batch_generation_ready": True,
            "raw_prompts_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        }
        payload["diagnosis_codes"].extend([
            "public_swarm_generate_batch_ready",
            "public_real_llm_swarm_beta_p2p_batch_ready",
        ])

        p2p = pack.p2p_summary(payload, min_generated_tokens=8)

        self.assertTrue(p2p["ready"], p2p)
        self.assertFalse(p2p["batch_ready"])
        self.assertFalse(p2p["batch"]["batch_identity_ready"])

    def test_evidence_import_blocks_when_product_report_missing(self) -> None:
        output_dir = self._tmp_dir()
        external_path = output_dir / "sources" / "external.json"
        p2p_path = output_dir / "sources" / "p2p.json"
        usable_path = output_dir / "sources" / "usable.json"
        public_swarm_v2_path = output_dir / "sources" / "public_swarm_v2.json"
        gpu_path = output_dir / "sources" / "gpu.json"
        external_path.parent.mkdir(parents=True, exist_ok=True)
        external_path.write_text(json.dumps(check.fake_external_payload()) + "\n", encoding="utf-8")
        p2p_path.write_text(json.dumps(check.fake_p2p_payload()) + "\n", encoding="utf-8")
        usable_path.write_text(json.dumps(check.fake_usable_payload()) + "\n", encoding="utf-8")
        public_swarm_v2_path.write_text(json.dumps(check.fake_public_swarm_v2_payload()) + "\n", encoding="utf-8")
        gpu_path.write_text(json.dumps(check.fake_gpu_payload()) + "\n", encoding="utf-8")

        args = pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "beta"),
            "--product-report",
            str(output_dir / "missing-product.json"),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--gpu-report",
            str(gpu_path),
        ])

        report = pack.build_report(args)

        self.assertFalse(report["ok"])
        self.assertIn("public_real_llm_swarm_beta_blocked", report["diagnosis_codes"])
        self.assertIn("local serve/join/generate product path", report["not_completed"])

    def test_release_blocks_p2p_requeue_without_victim_rejection_detail(self) -> None:
        output_dir = self._tmp_dir()
        external_path = output_dir / "sources" / "external.json"
        p2p_path = output_dir / "sources" / "p2p.json"
        usable_path = output_dir / "sources" / "usable.json"
        public_swarm_v2_path = output_dir / "sources" / "public_swarm_v2.json"
        external_path.parent.mkdir(parents=True, exist_ok=True)
        external_path.write_text(json.dumps(check.fake_external_payload()) + "\n", encoding="utf-8")
        legacy_p2p = check.fake_p2p_payload()
        legacy_p2p["diagnosis_codes"] = [
            code for code in legacy_p2p["diagnosis_codes"]
            if code not in {"p2p_live_requeue_rescue_ready", "p2p_victim_result_not_accepted"}
        ]
        legacy_p2p["candidate"]["live_requeue_summary"]["victim_result_accepted"] = True
        p2p_path.write_text(json.dumps(legacy_p2p) + "\n", encoding="utf-8")
        usable_path.write_text(json.dumps(check.fake_usable_payload()) + "\n", encoding="utf-8")
        public_swarm_v2_path.write_text(json.dumps(check.fake_public_swarm_v2_payload()) + "\n", encoding="utf-8")

        args = pack.parse_args([
            "release",
            "--output-dir",
            str(output_dir / "beta"),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--base-port",
            "9450",
            "--port",
            "9450",
            "--timeout-seconds",
            "60",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            p2p_completed = write_fresh_p2p_if_requested(command, output_dir, json.loads(p2p_path.read_text(encoding="utf-8")))
            if p2p_completed is not None:
                return p2p_completed
            v2_completed = fake_v2_if_requested(command)
            if v2_completed is not None:
                return v2_completed
            if "public_swarm_product_beta_pack.py" in joined:
                return completed(check.fake_product_payload())
            if "public_swarm_gpu_inference_beta_pack.py" in joined:
                return completed(check.fake_gpu_payload())
            raise AssertionError(command)

        report = pack.build_report(args, runner=fake_runner)

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["beta"]["p2p_ready_product_beta"])
        self.assertFalse(report["beta"]["p2p_live_requeue_ready"])
        self.assertFalse(report["beta"]["p2p_victim_result_not_accepted"])
        self.assertIn("public_real_llm_swarm_beta_blocked", report["diagnosis_codes"])
        self.assertIn("real-P2P discovery candidate with live requeue rescue", report["not_completed"])

    def test_release_recovers_redacted_p2p_lease_timeout_when_diagnosed(self) -> None:
        output_dir = self._tmp_dir()
        external_path = output_dir / "sources" / "external.json"
        p2p_path = output_dir / "sources" / "p2p.json"
        usable_path = output_dir / "sources" / "usable.json"
        public_swarm_v2_path = output_dir / "sources" / "public_swarm_v2.json"
        external_path.parent.mkdir(parents=True, exist_ok=True)
        external_path.write_text(json.dumps(check.fake_external_payload()) + "\n", encoding="utf-8")
        p2p_payload = check.fake_p2p_payload()
        p2p_payload["candidate"]["live_requeue_summary"]["lease_expired"] = "<redacted>"
        p2p_payload["diagnosis_codes"].append("live_requeue_lease_timeout_observed")
        p2p_path.write_text(json.dumps(p2p_payload) + "\n", encoding="utf-8")
        usable_path.write_text(json.dumps(check.fake_usable_payload()) + "\n", encoding="utf-8")
        public_swarm_v2_path.write_text(json.dumps(check.fake_public_swarm_v2_payload()) + "\n", encoding="utf-8")

        args = pack.parse_args([
            "release",
            "--output-dir",
            str(output_dir / "beta"),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--base-port",
            "9451",
            "--port",
            "9451",
            "--timeout-seconds",
            "60",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            p2p_completed = write_fresh_p2p_if_requested(command, output_dir, json.loads(p2p_path.read_text(encoding="utf-8")))
            if p2p_completed is not None:
                return p2p_completed
            v2_completed = fake_v2_if_requested(command)
            if v2_completed is not None:
                return v2_completed
            if "public_swarm_product_beta_pack.py" in joined:
                return completed(check.fake_product_payload())
            if "public_swarm_gpu_inference_beta_pack.py" in joined:
                return completed(check.fake_gpu_payload())
            raise AssertionError(command)

        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["beta"]["p2p_ready_product_beta"])
        self.assertIs(report["readiness"]["p2p_candidate"]["live_requeue_summary"]["lease_expired"], True)
        self.assertIn("p2p_live_requeue_rescue_ready", report["diagnosis_codes"])

    def test_release_blocks_non_default_model_when_imported_evidence_mismatches(self) -> None:
        output_dir = self._tmp_dir()
        external_path, p2p_path, usable_path, public_swarm_v2_path = write_default_sources(output_dir)

        args = pack.parse_args([
            "release",
            "--output-dir",
            str(output_dir / "beta"),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--hf-model-id",
            "distilgpt2",
            "--base-port",
            "9455",
            "--port",
            "9455",
            "--timeout-seconds",
            "60",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            p2p_completed = write_fresh_p2p_if_requested(command, output_dir, json.loads(p2p_path.read_text(encoding="utf-8")))
            if p2p_completed is not None:
                return p2p_completed
            v2_completed = fake_v2_if_requested(command)
            if v2_completed is not None:
                return v2_completed
            if "public_swarm_product_beta_pack.py" in joined:
                product = check.fake_product_payload()
                product["product_beta"]["hf_model_id"] = "distilgpt2"
                return completed(product)
            if "public_swarm_gpu_inference_beta_pack.py" in joined:
                return completed(check.fake_gpu_payload())
            raise AssertionError(command)

        report = pack.build_report(args, runner=fake_runner)

        self.assertFalse(report["ok"], report)
        self.assertIn("external_model_mismatch", report["diagnosis_codes"])
        self.assertIn("p2p_model_mismatch", report["diagnosis_codes"])
        self.assertIn("external evidence model match", report["not_completed"])
        self.assertIn("real-P2P evidence model match", report["not_completed"])
        self.assertEqual(report["readiness"]["external_kaggle"]["model"]["expected_hf_model_id"], "distilgpt2")
        self.assertEqual(report["readiness"]["external_kaggle"]["model"]["observed_hf_model_id"], "sshleifer/tiny-gpt2")

    def test_local_model_variant_accepts_non_default_v2_without_release_claims(self) -> None:
        output_dir = self._tmp_dir()
        calls: list[list[str]] = []
        public_swarm_v2 = mark_public_swarm_v2_local_model_variant(check.fake_public_swarm_v2_payload(16), "distilgpt2")
        usable = mark_usable_model(check.fake_usable_payload(), "distilgpt2")
        usable["readiness"]["p2p_product_path"]["generated_token_count"] = 16
        usable["readiness"]["p2p_product_path"]["generation"]["generated_token_count"] = 16
        usable["readiness"]["p2p_product_path"]["generation"]["max_new_tokens"] = 16
        usable["readiness"]["p2p_product_path"]["kv_cache"]["stage0"]["hit_count"] = 15
        usable["readiness"]["p2p_product_path"]["kv_cache"]["stage1"]["hit_count"] = 15

        args = pack.parse_args([
            "local-model-variant",
            "--output-dir",
            str(output_dir / "beta"),
            "--hf-model-id",
            "distilgpt2",
            "--stream-generation",
            "--max-new-tokens",
            "16",
            "--base-port",
            "9459",
            "--port",
            "9459",
            "--timeout-seconds",
            "60",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            calls.append(command)
            if "public_swarm_product_beta_pack.py" in joined:
                product = check.fake_product_payload()
                product["product_beta"]["hf_model_id"] = "distilgpt2"
                product["product_beta"]["max_new_tokens"] = 16
                child_dir = Path(command[command.index("--output-dir") + 1])
                child_dir.mkdir(parents=True, exist_ok=True)
                (child_dir / "public_swarm_product_beta.json").write_text(json.dumps(product) + "\n", encoding="utf-8")
                return completed(product)
            if "public_swarm_inference_v2_pack.py" in joined:
                self.assertEqual(command[2], "local-model-variant")
                self.assertEqual(command[command.index("--hf-model-id") + 1], "distilgpt2")
                child_dir = Path(command[command.index("--output-dir") + 1])
                child_dir.mkdir(parents=True, exist_ok=True)
                (child_dir / "public_swarm_inference_v2.json").write_text(json.dumps(public_swarm_v2) + "\n", encoding="utf-8")
                usable_dir = child_dir / "usable-v1-local"
                usable_dir.mkdir(parents=True, exist_ok=True)
                (usable_dir / "usable_swarm_inference.json").write_text(json.dumps(usable) + "\n", encoding="utf-8")
                return completed(public_swarm_v2)
            if "public_swarm_gpu_inference_beta_pack.py" in joined:
                gpu = check.fake_gpu_payload()
                child_dir = Path(command[command.index("--output-dir") + 1])
                child_dir.mkdir(parents=True, exist_ok=True)
                (child_dir / "public_swarm_gpu_inference_beta_local_smoke.json").write_text(json.dumps(gpu) + "\n", encoding="utf-8")
                return completed(gpu)
            raise AssertionError(command)

        report = pack.build_report(args, runner=fake_runner)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["mode"], "local-model-variant")
        self.assertTrue(report["beta"]["local_model_variant_only"])
        self.assertTrue(report["beta"]["public_swarm_v2_local_model_variant_ready"])
        self.assertFalse(report["beta"]["release_evidence_ready"])
        self.assertFalse(report["beta"]["external_two_stage_ready"])
        self.assertFalse(report["beta"]["p2p_ready_product_beta"])
        self.assertTrue(report["readiness"]["public_swarm_v2"]["model"]["compatible"])
        self.assertTrue(report["readiness"]["public_swarm_v2"]["local_model_variant_only"])
        self.assertFalse(report["readiness"]["public_swarm_v2"]["external_validation_claimed"])
        self.assertTrue(report["readiness"]["usable_p2p_kv_cache"]["model"]["compatible"])
        self.assertIn("public_real_llm_swarm_beta_local_model_variant_ready", report["diagnosis_codes"])
        self.assertIn("public_swarm_inference_v2_local_model_variant_ready", report["diagnosis_codes"])
        self.assertIn("external_validation_not_claimed", report["diagnosis_codes"])
        self.assertNotIn("external_runtime_verified", report["diagnosis_codes"])
        self.assertNotIn("external_stage_requeue_ready", report["diagnosis_codes"])
        self.assertNotIn("cuda_runtime_available", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_gpu_beta_ready", report["diagnosis_codes"])
        self.assertNotIn("public_swarm_v2_external_stage_rows_ready", report["diagnosis_codes"])
        self.assertNotIn("cuda_runtime_available", report["readiness"]["public_swarm_v2"]["diagnosis_codes"])
        self.assertNotIn("public_real_llm_swarm_beta_ready", report["diagnosis_codes"])
        self.assertNotIn("release_evidence_ready", report["diagnosis_codes"])
        self.assertEqual(report["not_completed"], [])
        self.assertEqual(len([command for command in calls if "public_swarm_inference_v2_pack.py" in " ".join(command)]), 1)

    def test_local_model_variant_blocks_when_real_p2p_local_route_missing(self) -> None:
        output_dir = self._tmp_dir()
        public_swarm_v2 = mark_public_swarm_v2_local_model_variant(check.fake_public_swarm_v2_payload(16), "distilgpt2")
        public_swarm_v2["readiness"]["real_p2p_local_route_hardening"]["ready"] = False
        public_swarm_v2["readiness"]["real_p2p_local_route_hardening"]["stage_requeue_ready"] = True
        usable = mark_usable_model(check.fake_usable_payload(), "distilgpt2")
        usable["readiness"]["p2p_product_path"]["generated_token_count"] = 16
        usable["readiness"]["p2p_product_path"]["generation"]["generated_token_count"] = 16
        usable["readiness"]["p2p_product_path"]["generation"]["max_new_tokens"] = 16
        usable["readiness"]["p2p_product_path"]["kv_cache"]["stage0"]["hit_count"] = 15
        usable["readiness"]["p2p_product_path"]["kv_cache"]["stage1"]["hit_count"] = 15

        args = pack.parse_args([
            "local-model-variant",
            "--output-dir",
            str(output_dir / "beta"),
            "--hf-model-id",
            "distilgpt2",
            "--stream-generation",
            "--max-new-tokens",
            "16",
            "--base-port",
            "9462",
            "--port",
            "9462",
            "--timeout-seconds",
            "60",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            if "public_swarm_product_beta_pack.py" in joined:
                product = check.fake_product_payload()
                product["product_beta"]["hf_model_id"] = "distilgpt2"
                product["product_beta"]["max_new_tokens"] = 16
                child_dir = Path(command[command.index("--output-dir") + 1])
                child_dir.mkdir(parents=True, exist_ok=True)
                (child_dir / "public_swarm_product_beta.json").write_text(json.dumps(product) + "\n", encoding="utf-8")
                return completed(product)
            if "public_swarm_inference_v2_pack.py" in joined:
                child_dir = Path(command[command.index("--output-dir") + 1])
                child_dir.mkdir(parents=True, exist_ok=True)
                (child_dir / "public_swarm_inference_v2.json").write_text(json.dumps(public_swarm_v2) + "\n", encoding="utf-8")
                usable_dir = child_dir / "usable-v1-local"
                usable_dir.mkdir(parents=True, exist_ok=True)
                (usable_dir / "usable_swarm_inference.json").write_text(json.dumps(usable) + "\n", encoding="utf-8")
                return completed(public_swarm_v2)
            if "public_swarm_gpu_inference_beta_pack.py" in joined:
                gpu = check.fake_gpu_payload()
                child_dir = Path(command[command.index("--output-dir") + 1])
                child_dir.mkdir(parents=True, exist_ok=True)
                (child_dir / "public_swarm_gpu_inference_beta_local_smoke.json").write_text(json.dumps(gpu) + "\n", encoding="utf-8")
                return completed(gpu)
            raise AssertionError(command)

        report = pack.build_report(args, runner=fake_runner)

        self.assertFalse(report["ok"], report)
        self.assertTrue(report["readiness"]["public_swarm_v2"]["ready"])
        self.assertFalse(report["beta"]["public_swarm_v2_real_p2p_local_ready"])
        self.assertFalse(report["readiness"]["public_swarm_v2"]["real_p2p_local_route_hardening_ready"])
        self.assertFalse(report["readiness"]["public_swarm_v2"]["real_p2p_local_stage_requeue_ready"])
        self.assertIn("public_real_llm_swarm_beta_local_model_variant_blocked", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_real_p2p_local_missing", report["diagnosis_codes"])
        self.assertIn("public_swarm_v2_real_p2p_local_requeue_missing", report["diagnosis_codes"])
        self.assertIn("Public Swarm v2 fresh real-P2P local route hardening", report["not_completed"])
        self.assertIn("Public Swarm v2 fresh real-P2P local stage requeue", report["not_completed"])

    def test_release_blocks_when_product_evidence_model_mismatches(self) -> None:
        output_dir = self._tmp_dir()
        source_dir = output_dir / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        external_path = source_dir / "external.json"
        p2p_path = source_dir / "p2p.json"
        usable_path = source_dir / "usable.json"
        public_swarm_v2_path = source_dir / "public_swarm_v2.json"
        external = check.fake_external_payload()
        external["workload"]["hf_model_id"] = "distilgpt2"
        p2p = check.fake_p2p_payload()
        p2p["candidate"]["hf_model_id"] = "distilgpt2"
        usable = mark_usable_model(check.fake_usable_payload(), "distilgpt2")
        public_swarm_v2 = mark_public_swarm_v2_model(check.fake_public_swarm_v2_payload(), "distilgpt2")
        external_path.write_text(json.dumps(external) + "\n", encoding="utf-8")
        p2p_path.write_text(json.dumps(p2p) + "\n", encoding="utf-8")
        usable_path.write_text(json.dumps(usable) + "\n", encoding="utf-8")
        public_swarm_v2_path.write_text(json.dumps(public_swarm_v2) + "\n", encoding="utf-8")

        args = pack.parse_args([
            "release",
            "--output-dir",
            str(output_dir / "beta"),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--hf-model-id",
            "distilgpt2",
            "--base-port",
            "9456",
            "--port",
            "9456",
            "--timeout-seconds",
            "60",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            p2p_completed = write_fresh_p2p_if_requested(command, output_dir, json.loads(p2p_path.read_text(encoding="utf-8")))
            if p2p_completed is not None:
                return p2p_completed
            v2_completed = fake_v2_if_requested(command, public_swarm_v2)
            if v2_completed is not None:
                return v2_completed
            if "public_swarm_product_beta_pack.py" in joined:
                return completed(check.fake_product_payload())
            if "public_swarm_gpu_inference_beta_pack.py" in joined:
                return completed(check.fake_gpu_payload())
            raise AssertionError(command)

        report = pack.build_report(args, runner=fake_runner)

        self.assertFalse(report["ok"], report)
        self.assertTrue(report["readiness"]["product_path"]["path_ready"])
        self.assertFalse(report["readiness"]["product_path"]["model"]["compatible"])
        self.assertTrue(report["readiness"]["external_kaggle"]["model"]["compatible"])
        self.assertTrue(report["readiness"]["p2p_candidate"]["model"]["compatible"])
        self.assertTrue(report["readiness"]["usable_p2p_kv_cache"]["model"]["compatible"])
        self.assertIn("product_model_mismatch", report["diagnosis_codes"])
        self.assertNotIn("external_model_mismatch", report["diagnosis_codes"])
        self.assertNotIn("p2p_model_mismatch", report["diagnosis_codes"])
        self.assertNotIn("kv_cache_model_mismatch", report["diagnosis_codes"])
        self.assertIn("local product evidence model match", report["not_completed"])

    def test_evidence_import_blocks_when_kv_cache_evidence_model_mismatches(self) -> None:
        output_dir = self._tmp_dir()
        source_dir = output_dir / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        product_path = source_dir / "product.json"
        external_path = source_dir / "external.json"
        p2p_path = source_dir / "p2p.json"
        usable_path = source_dir / "usable.json"
        public_swarm_v2_path = source_dir / "public_swarm_v2.json"
        gpu_path = source_dir / "gpu.json"
        product = check.fake_product_payload()
        product["product_beta"]["hf_model_id"] = "distilgpt2"
        external = check.fake_external_payload()
        external["workload"]["hf_model_id"] = "distilgpt2"
        p2p = check.fake_p2p_payload()
        p2p["candidate"]["hf_model_id"] = "distilgpt2"
        public_swarm_v2 = mark_public_swarm_v2_model(check.fake_public_swarm_v2_payload(), "distilgpt2")
        product_path.write_text(json.dumps(product) + "\n", encoding="utf-8")
        external_path.write_text(json.dumps(external) + "\n", encoding="utf-8")
        p2p_path.write_text(json.dumps(p2p) + "\n", encoding="utf-8")
        usable_path.write_text(json.dumps(check.fake_usable_payload()) + "\n", encoding="utf-8")
        public_swarm_v2_path.write_text(json.dumps(public_swarm_v2) + "\n", encoding="utf-8")
        gpu_path.write_text(json.dumps(check.fake_gpu_payload()) + "\n", encoding="utf-8")

        args = pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "beta"),
            "--product-report",
            str(product_path),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--gpu-report",
            str(gpu_path),
            "--hf-model-id",
            "distilgpt2",
            "--base-port",
            "9458",
            "--port",
            "9458",
            "--timeout-seconds",
            "60",
        ])

        report = pack.build_report(args)

        self.assertFalse(report["ok"], report)
        self.assertTrue(report["readiness"]["product_path"]["model"]["compatible"])
        self.assertTrue(report["readiness"]["external_kaggle"]["model"]["compatible"])
        self.assertTrue(report["readiness"]["p2p_candidate"]["model"]["compatible"])
        self.assertTrue(report["readiness"]["usable_p2p_kv_cache"]["cache_ready"])
        self.assertFalse(report["readiness"]["usable_p2p_kv_cache"]["model"]["compatible"])
        self.assertIn("kv_cache_model_mismatch", report["diagnosis_codes"])
        self.assertNotIn("product_model_mismatch", report["diagnosis_codes"])
        self.assertNotIn("public_real_llm_swarm_beta_kv_cache_missing", report["diagnosis_codes"])
        self.assertIn("KV-cache evidence model match", report["not_completed"])
        self.assertNotIn("persistent dual-stage KV-cache reuse", report["not_completed"])

    def test_release_blocks_when_imported_evidence_misses_token_target(self) -> None:
        output_dir = self._tmp_dir()
        external_path, p2p_path, usable_path, public_swarm_v2_path = write_default_sources(output_dir)
        external_path.write_text(json.dumps(check.fake_external_payload(tokens=8)) + "\n", encoding="utf-8")
        p2p_path.write_text(json.dumps(check.fake_p2p_payload(tokens=8)) + "\n", encoding="utf-8")

        args = pack.parse_args([
            "release",
            "--output-dir",
            str(output_dir / "beta"),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--max-new-tokens",
            "16",
            "--base-port",
            "9457",
            "--port",
            "9457",
            "--timeout-seconds",
            "60",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            p2p_completed = write_fresh_p2p_if_requested(command, output_dir, json.loads(p2p_path.read_text(encoding="utf-8")))
            if p2p_completed is not None:
                return p2p_completed
            v2_completed = fake_v2_if_requested(command)
            if v2_completed is not None:
                return v2_completed
            if "public_swarm_product_beta_pack.py" in joined:
                product = check.fake_product_payload()
                product["product_beta"]["max_new_tokens"] = 16
                return completed(product)
            if "public_swarm_gpu_inference_beta_pack.py" in joined:
                return completed(check.fake_gpu_payload())
            raise AssertionError(command)

        report = pack.build_report(args, runner=fake_runner)

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["readiness"]["external_kaggle"]["token_target_ready"])
        self.assertFalse(report["readiness"]["p2p_candidate"]["token_target_ready"])
        self.assertEqual(report["readiness"]["external_kaggle"]["required_generated_token_count"], 16)
        self.assertEqual(report["readiness"]["p2p_candidate"]["required_generated_token_count"], 16)
        self.assertIn("external_generated_token_target_missing", report["diagnosis_codes"])
        self.assertIn("p2p_generated_token_target_missing", report["diagnosis_codes"])
        self.assertIn("external generated token target", report["not_completed"])
        self.assertIn("real-P2P generated token target", report["not_completed"])

    def test_evidence_import_accepts_sixteen_token_external_and_p2p_targets(self) -> None:
        output_dir = self._tmp_dir()
        product_path = output_dir / "sources" / "product.json"
        external_path = output_dir / "sources" / "external.json"
        p2p_path = output_dir / "sources" / "p2p.json"
        usable_path = output_dir / "sources" / "usable.json"
        public_swarm_v2_path = output_dir / "sources" / "public_swarm_v2.json"
        gpu_path = output_dir / "sources" / "gpu.json"
        product_path.parent.mkdir(parents=True, exist_ok=True)
        product = check.fake_product_payload()
        product["product_beta"]["max_new_tokens"] = 16
        external = check.fake_external_payload()
        external["workload"]["max_new_tokens"] = 16
        external["generation"]["generated_token_count"] = 16
        external["generation"]["max_new_tokens"] = 16
        p2p = check.fake_p2p_payload()
        p2p["candidate"]["external_generated_token_count"] = 16
        p2p["candidate"]["accepted_rows"] = 32
        p2p["generation"]["generated_token_count"] = 16
        p2p["generation"]["max_new_tokens"] = 16
        product_path.write_text(json.dumps(product) + "\n", encoding="utf-8")
        external_path.write_text(json.dumps(external) + "\n", encoding="utf-8")
        p2p_path.write_text(json.dumps(p2p) + "\n", encoding="utf-8")
        usable = mark_usable_model(check.fake_usable_payload(), "sshleifer/tiny-gpt2")
        usable["readiness"]["p2p_product_path"]["generated_token_count"] = 16
        usable["readiness"]["p2p_product_path"]["generation"]["generated_token_count"] = 16
        usable["readiness"]["p2p_product_path"]["generation"]["max_new_tokens"] = 16
        usable["readiness"]["p2p_product_path"]["kv_cache"]["stage0"]["hit_count"] = 15
        usable["readiness"]["p2p_product_path"]["kv_cache"]["stage0"]["expected_hit_count"] = 15
        usable["readiness"]["p2p_product_path"]["kv_cache"]["stage1"]["hit_count"] = 15
        usable["readiness"]["p2p_product_path"]["kv_cache"]["stage1"]["expected_hit_count"] = 15
        usable_path.write_text(json.dumps(usable) + "\n", encoding="utf-8")
        public_swarm_v2_path.write_text(json.dumps(check.fake_public_swarm_v2_payload(16)) + "\n", encoding="utf-8")
        gpu_path.write_text(json.dumps(check.fake_gpu_payload()) + "\n", encoding="utf-8")

        args = pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "beta"),
            "--product-report",
            str(product_path),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--gpu-report",
            str(gpu_path),
            "--max-new-tokens",
            "16",
        ])

        report = pack.build_report(args)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["readiness"]["external_kaggle"]["token_target_ready"])
        self.assertTrue(report["readiness"]["p2p_candidate"]["token_target_ready"])
        self.assertEqual(report["readiness"]["external_kaggle"]["generated_token_count"], 16)
        self.assertEqual(report["readiness"]["p2p_candidate"]["generated_token_count"], 16)
        self.assertIn("external_generated_token_target_ready", report["diagnosis_codes"])
        self.assertIn("p2p_generated_token_target_ready", report["diagnosis_codes"])
        self.assertIn("public_real_llm_swarm_beta_ready", report["diagnosis_codes"])
        self.assertNotIn("external_generated_token_target_missing", report["diagnosis_codes"])
        self.assertNotIn("p2p_generated_token_target_missing", report["diagnosis_codes"])

    def test_release_blocks_default_model_without_imported_model_metadata(self) -> None:
        output_dir = self._tmp_dir()
        external_path = output_dir / "sources" / "external.json"
        p2p_path = output_dir / "sources" / "p2p.json"
        usable_path = output_dir / "sources" / "usable.json"
        public_swarm_v2_path = output_dir / "sources" / "public_swarm_v2.json"
        external_path.parent.mkdir(parents=True, exist_ok=True)
        external_payload = check.fake_external_payload()
        external_payload["workload"].pop("hf_model_id", None)
        p2p_payload = check.fake_p2p_payload()
        p2p_payload["candidate"].pop("hf_model_id", None)
        external_path.write_text(json.dumps(external_payload) + "\n", encoding="utf-8")
        p2p_path.write_text(json.dumps(p2p_payload) + "\n", encoding="utf-8")
        usable_path.write_text(json.dumps(check.fake_usable_payload()) + "\n", encoding="utf-8")
        public_swarm_v2_path.write_text(json.dumps(check.fake_public_swarm_v2_payload()) + "\n", encoding="utf-8")

        args = pack.parse_args([
            "release",
            "--output-dir",
            str(output_dir / "beta"),
            "--external-report",
            str(external_path),
            "--p2p-report",
            str(p2p_path),
            "--usable-report",
            str(usable_path),
            "--public-swarm-v2-report",
            str(public_swarm_v2_path),
            "--base-port",
            "9460",
            "--port",
            "9460",
            "--timeout-seconds",
            "60",
        ])

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(command)
            p2p_completed = write_fresh_p2p_if_requested(command, output_dir, json.loads(p2p_path.read_text(encoding="utf-8")))
            if p2p_completed is not None:
                return p2p_completed
            v2_completed = fake_v2_if_requested(command)
            if v2_completed is not None:
                return v2_completed
            if "public_swarm_product_beta_pack.py" in joined:
                return completed(check.fake_product_payload())
            if "public_swarm_gpu_inference_beta_pack.py" in joined:
                return completed(check.fake_gpu_payload())
            raise AssertionError(command)

        report = pack.build_report(args, runner=fake_runner)

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["readiness"]["external_kaggle"]["model"]["model_id_present"])
        self.assertFalse(report["readiness"]["external_kaggle"]["model"]["compatible"])
        self.assertFalse(report["readiness"]["p2p_candidate"]["model"]["model_id_present"])
        self.assertFalse(report["readiness"]["p2p_candidate"]["model"]["compatible"])
        self.assertIn("external_model_mismatch", report["diagnosis_codes"])
        self.assertIn("p2p_model_mismatch", report["diagnosis_codes"])
        self.assertIn("external evidence model match", report["not_completed"])
        self.assertIn("real-P2P evidence model match", report["not_completed"])

    def test_check_script_validates_release_contract(self) -> None:
        output_dir = self._tmp_dir()

        result = check.run_check(check.parse_args(["--output-dir", str(output_dir)]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], check.SCHEMA)
        self.assertEqual(result["max_new_tokens"], 16)
        self.assertIn("public_real_llm_swarm_beta_check_ready", result["diagnosis_codes"])
        self.assertEqual(result["artifact_summary"]["schema"], check.ARTIFACT_SUMMARY_SCHEMA)
        self.assertTrue(result["artifact_summary"]["inspect_first"].endswith("public_real_llm_swarm_beta.md"))
        self.assertTrue(result["artifact_summary"]["machine_readable"].endswith("public_real_llm_swarm_beta.json"))
        self.assertTrue(result["artifact_summary"]["support_bundle"].endswith("support_bundle.json"))
        self.assertTrue(result["artifact_summary"]["check_json"].endswith("public_real_llm_swarm_beta_check.json"))
        self.assertTrue(result["artifact_summary"]["public_artifact_safe"])
        self.assertFalse(result["artifact_summary"]["raw_prompt_public"])
        self.assertFalse(result["artifact_summary"]["raw_generated_text_public"])
        self.assertFalse(result["artifact_summary"]["generated_token_ids_public"])
        self.assertEqual(result["review_summary"]["schema"], check.REVIEW_SUMMARY_SCHEMA)
        self.assertEqual(result["review_summary"]["state"], "ready")
        self.assertTrue(result["review_summary"]["ready"])
        self.assertEqual(result["review_summary"]["next_step"], "review_checked_artifacts")
        self.assertEqual(result["review_summary"]["error_count"], 0)
        self.assertTrue(result["review_summary"]["public_artifact_safe"])
        self.assertEqual(result["operator_action"], "Open the checked Markdown and support bundle.")
        report_path = output_dir / "beta" / "public_real_llm_swarm_beta.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["beta"]["max_new_tokens"], 16)
        self.assertEqual(report["readiness"]["external_kaggle"]["generated_token_count"], 16)
        self.assertEqual(report["readiness"]["external_kaggle"]["required_generated_token_count"], 16)
        self.assertEqual(report["readiness"]["p2p_candidate"]["generated_token_count"], 16)
        self.assertEqual(report["readiness"]["p2p_candidate"]["required_generated_token_count"], 16)
        self.assertEqual(report["readiness"]["public_swarm_v2"]["generated_token_count"], 16)
        self.assertEqual(report["readiness"]["public_swarm_v2"]["required_generated_token_count"], 16)
        self.assertEqual(report["readiness"]["public_swarm_v2"]["accepted_rows"], 32)
        self.assertEqual(report["readiness"]["usable_p2p_kv_cache"]["generated_token_count"], 16)
        self.assertEqual(report["readiness"]["usable_p2p_kv_cache"]["required_generated_token_count"], 16)
        self.assertEqual(report["readiness"]["usable_p2p_kv_cache"]["stage0"]["hit_count"], 15)
        self.assertEqual(report["readiness"]["usable_p2p_kv_cache"]["stage1"]["hit_count"], 15)
        self.assertFalse(report["output_request"]["include_output"])
        self.assertEqual(report["answer_scope"]["scope_state"], "no-local-answer")
        self.assertEqual(report["shareable_summary"]["answer_scope_state"], "no-local-answer")
        artifacts = result["artifacts"]
        for artifact_name in [
            "public_real_llm_swarm_beta_json",
            "public_real_llm_swarm_beta_markdown",
            "support_bundle_json",
            "runbook",
        ]:
            self.assertTrue(Path(artifacts[artifact_name]).is_file(), artifacts[artifact_name])
        self.assertTrue(artifacts["public_real_llm_swarm_beta_markdown"].endswith("public_real_llm_swarm_beta.md"))
        self.assertTrue(artifacts["support_bundle_json"].endswith("support_bundle.json"))
        self.assertTrue(artifacts["runbook"].endswith("PUBLIC_REAL_LLM_SWARM_BETA.md"))
        persisted_check = json.loads((output_dir / "public_real_llm_swarm_beta_check.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted_check["review_summary"]["next_step"], "review_checked_artifacts")
        self.assertTrue(persisted_check["artifact_summary"]["check_json"].endswith("public_real_llm_swarm_beta_check.json"))

    def test_pack_human_summary_shows_final_status_and_artifacts(self) -> None:
        output_dir = self._tmp_dir()

        report = check.build_fake_release(output_dir, tokens=16)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            pack.print_human_summary(report)
        output = stdout.getvalue()

        self.assertIn("CrowdTensor Public Real-LLM Swarm Inference Beta", output)
        self.assertIn("  model: sshleifer/tiny-gpt2 tokens=16", output)
        self.assertIn("  external tokens: 16/16", output)
        self.assertIn("  p2p tokens: 16/16", output)
        self.assertIn("  public_swarm_v2 tokens: 16/16 accepted_rows=32/32", output)
        self.assertIn("  batch ready: product=False p2p=True v2=True", output)
        self.assertIn("  stream ready: product=False p2p=True v2=True", output)
        self.assertIn("  kv_cache_ready: True", output)
        self.assertIn("  kv_cache hits: stage0=15 stage1=15", output)
        self.assertIn("  operator_action:", output)
        self.assertIn("    - Use `crowdtensor serve`, `crowdtensor join --stage stage0`, `crowdtensor join --stage stage1`, and `crowdtensor generate` as the primary user path.", output)
        self.assertIn("  artifact public_real_llm_swarm_beta_markdown: public_real_llm_swarm_beta.md present=True", output)
        self.assertIn("  artifact support_bundle_json: support_bundle.json present=True", output)
        self.assertNotIn("  not_completed:", output)

    def test_pack_human_summary_shows_blockers(self) -> None:
        output_dir = self._tmp_dir()

        report = check.build_fake_release(output_dir, tokens=16)
        report["ok"] = False
        report["beta"]["ready"] = False
        report["readiness"]["public_swarm_v2"]["generated_token_count"] = 8
        report["readiness"]["public_swarm_v2"]["accepted_rows"] = 16
        report["not_completed"] = [
            "external generated token target",
            "Public Swarm v2 generated token target",
        ]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            pack.print_human_summary(report)
        output = stdout.getvalue()

        self.assertIn("  ready: False", output)
        self.assertIn("  public_swarm_v2 tokens: 8/16 accepted_rows=16/32", output)
        self.assertIn("  operator_action:", output)
        self.assertIn("    - Use `crowdtensor serve`, `crowdtensor join --stage stage0`, `crowdtensor join --stage stage1`, and `crowdtensor generate` as the primary user path.", output)
        self.assertIn("  not_completed:", output)
        self.assertIn("    - external generated token target", output)
        self.assertIn("    - Public Swarm v2 generated token target", output)

    def test_check_human_summary_shows_artifacts(self) -> None:
        output_dir = self._tmp_dir()

        result = check.run_check(check.parse_args(["--output-dir", str(output_dir)]))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            check.print_human_summary(result)
        output = stdout.getvalue()

        self.assertIn("Public Real-LLM Swarm Beta check ready: True", output)
        self.assertIn("  max_new_tokens: 16", output)
        self.assertIn("  review: state=ready next=review_checked_artifacts", output)
        self.assertIn("errors=0", output)
        self.assertIn("  artifacts: inspect=", output)
        self.assertIn("check=", output)
        self.assertIn("  action: Open the checked Markdown and support bundle.", output)
        self.assertIn("  diagnosis: public_real_llm_swarm_beta_check_ready", output)
        self.assertIn("  artifact public_real_llm_swarm_beta_markdown:", output)
        self.assertIn("public_real_llm_swarm_beta.md", output)
        self.assertIn("  artifact support_bundle_json:", output)
        self.assertIn("support_bundle.json", output)
        self.assertIn("  artifact runbook:", output)
        self.assertIn("PUBLIC_REAL_LLM_SWARM_BETA.md", output)
        self.assertIn("  artifact public_real_llm_swarm_beta_check_json:", output)

    def test_pack_cli_and_check_default_to_final_16_token_contract(self) -> None:
        pack_args = pack.parse_args(["release"])
        cli_args = cli.parse_args(["public-real-llm-swarm-beta", "release"])
        check_args = check.parse_args([])

        self.assertEqual(pack_args.max_new_tokens, 16)
        self.assertEqual(cli_args.max_new_tokens, 16)
        self.assertEqual(check_args.max_new_tokens, 16)

    def test_check_validation_rejects_eight_token_release_when_expect_16(self) -> None:
        output_dir = self._tmp_dir()

        payload = check.build_fake_release(output_dir, tokens=8)
        errors = check.validate_report(payload, mode="release", expected_tokens=16)

        self.assertIn("beta_token_target_below_16", errors)
        self.assertIn("external_token_target_below_16", errors)
        self.assertIn("external_required_token_target_below_16", errors)
        self.assertIn("p2p_token_target_below_16", errors)
        self.assertIn("p2p_required_token_target_below_16", errors)
        self.assertIn("public_swarm_v2_token_target_below_16", errors)
        self.assertIn("public_swarm_v2_required_token_target_below_16", errors)
        self.assertIn("usable_kv_cache_token_target_below_16", errors)
        self.assertIn("usable_kv_cache_required_token_target_below_16", errors)
        self.assertIn("stage0_kv_cache_hits_below_15", errors)
        self.assertIn("stage1_kv_cache_hits_below_15", errors)

    def test_check_script_blocked_result_has_review_guidance(self) -> None:
        output_dir = self._tmp_dir()
        payload = check.build_fake_release(output_dir, tokens=8)
        errors = check.validate_report(payload, mode="release", expected_tokens=16)
        result = {
            "schema": check.SCHEMA,
            "ok": False,
            "mode": "release",
            "max_new_tokens": 16,
            "output_dir": str(output_dir),
            "errors": errors,
            "diagnosis_codes": ["public_real_llm_swarm_beta_check_blocked"],
            "artifacts": {
                "public_real_llm_swarm_beta_json": check.artifact_path(
                    payload,
                    "public_real_llm_swarm_beta_json",
                    "public_real_llm_swarm_beta.json",
                ),
                "public_real_llm_swarm_beta_markdown": check.artifact_path(
                    payload,
                    "public_real_llm_swarm_beta_markdown",
                    "public_real_llm_swarm_beta.md",
                ),
                "support_bundle_json": check.artifact_path(payload, "support_bundle_json", "support_bundle.json"),
                "runbook": check.artifact_path(payload, "runbook", "PUBLIC_REAL_LLM_SWARM_BETA.md"),
            },
        }
        result["artifact_summary"] = check.check_artifact_summary(result)
        result["review_summary"] = check.check_review_summary(result)
        result["operator_action"] = result["review_summary"]["operator_action"]

        self.assertFalse(result["ok"], result)
        self.assertIn("public_real_llm_swarm_beta_check_blocked", result["diagnosis_codes"])
        self.assertEqual(result["review_summary"]["state"], "blocked")
        self.assertFalse(result["review_summary"]["ready"])
        self.assertEqual(result["review_summary"]["next_step"], "fix_validation_errors")
        self.assertGreater(result["review_summary"]["error_count"], 0)
        self.assertIn("beta_token_target_below_16", result["review_summary"]["error_preview"])
        self.assertIn("Fix the listed validation errors", result["operator_action"])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            check.print_human_summary(result)
        output = stdout.getvalue()
        self.assertIn("Public Real-LLM Swarm Beta check ready: False", output)
        self.assertIn("  review: state=blocked next=fix_validation_errors", output)
        self.assertIn("  errors:", output)
        self.assertIn("    - beta_token_target_below_16", output)

    def test_check_script_validates_local_model_variant_contract(self) -> None:
        output_dir = self._tmp_dir()

        result = check.run_check(check.parse_args([
            "--mode",
            "local-model-variant",
            "--hf-model-id",
            "distilgpt2",
            "--output-dir",
            str(output_dir),
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["mode"], "local-model-variant")
        self.assertEqual(result["schema"], check.SCHEMA)
        self.assertIn("public_real_llm_swarm_beta_check_ready", result["diagnosis_codes"])


if __name__ == "__main__":
    unittest.main()
