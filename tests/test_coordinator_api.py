from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from coordinator import (
    create_app,
    enforce_miner_join_policy,
    load_miner_token_registry,
    miner_registry_policy_summary,
    parse_task_lane,
)
from crowdtensor.auth import hash_token
from crowdtensor.diloco import run_inner_loop
from crowdtensor.external_llm import WORKLOAD_TYPE as WORKLOAD_EXTERNAL_LLM_INFER
from crowdtensor.external_llm import run_mock_external_llm_inference
from crowdtensor.model_bundle import INFERENCE_WORKLOAD_TYPE as WORKLOAD_MODEL_BUNDLE_INFER
from crowdtensor.model_bundle import run_model_bundle_inference
from crowdtensor.model_bundle import run_model_bundle_inner_loop
from crowdtensor.outer_optimizer import (
    DELTA_FORMAT_SIGN_COMPRESSED_EF,
    OPTIMIZER_DILOCO_NESTEROV,
    compress_sign_delta,
    compress_sign_delta_with_error_feedback,
)
from crowdtensor.protocol import REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY, REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY
from crowdtensor.real_llm import WORKLOAD_TYPE as WORKLOAD_REAL_LLM_SHARDED_INFER
from crowdtensor.toy_compute import compute_pseudo_gradient


def endpoint_for(app, path: str, method: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method.upper() in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"route {method} {path} not found")


def request_model(endpoint, name: str = "request"):
    parameter = inspect.signature(endpoint).parameters[name]
    return parameter.annotation


def write_registry(path: Path, miners: list[dict]) -> Path:
    path.write_text(json.dumps({"miners": miners}), encoding="utf-8")
    return path


class CoordinatorApiTests(unittest.TestCase):
    def test_lifespan_starts_and_stops_reaper(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10)
                async with app.router.lifespan_context(app):
                    task = app.state.reaper_task
                    self.assertFalse(task.done())
                self.assertTrue(app.state.reaper_task.done())

        asyncio.run(run_case())

    def test_claim_result_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10)
            health = endpoint_for(app, "/health", "GET")
            version = endpoint_for(app, "/version", "GET")
            ready = endpoint_for(app, "/ready", "GET")
            state = endpoint_for(app, "/state", "GET")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            claim_request = request_model(claim_task)(miner_id="api-miner")

            self.assertTrue(health()["ok"])
            self.assertEqual(health()["version"], "0.1.0a0")
            version_payload = version()
            self.assertEqual(version_payload["service"], "crowdtensord-coordinator")
            self.assertEqual(version_payload["version"], "0.1.0a0")
            self.assertEqual(version_payload["protocol_version"], "runtime_contract_v1")
            self.assertEqual(version_payload["default_workload_type"], "diloco_train")
            ready_payload = ready()
            self.assertTrue(ready_payload["ok"])
            self.assertEqual(ready_payload["auth"]["miner_required"], False)
            self.assertEqual(ready_payload["auth"]["observer_required"], False)
            self.assertIn("queued", ready_payload["task_counts"])
            claim = claim_task(claim_request)
            self.assertEqual(claim["optimizer_spec"]["contract_version"], "outer_optimizer_contract_v1")
            self.assertEqual(claim["optimizer_spec"]["delta_format"], "dense_float")
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="api-miner",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )
            result_request = request_model(result_task)(
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                local_delta=inner_result["local_delta"],
                metrics=inner_result,
            )

            result = result_task(claim["task_id"], result_request)
            self.assertEqual(result["global_step"], 1)
            self.assertEqual(result["optimizer_step"], 1)
            self.assertEqual(result["optimizer"]["optimizer_type"], "diloco_momentum")
            self.assertEqual(result["optimizer"]["optimizer_step_after"], 1)
            summary = state()
            self.assertEqual(summary["task_counts"]["completed"], 1)
            self.assertEqual(summary["task_counts"]["queued"], 1)
            self.assertEqual(summary["model"]["schema_version"], "diloco_mock_v1")
            self.assertEqual(summary["accepted_results"], 1)
            self.assertEqual(summary["max_staleness"], 0)
            self.assertEqual(summary["last_completed"]["task_id"], claim["task_id"])

    def test_miner_token_protects_task_endpoints_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10, miner_token="miner-secret")
            state = endpoint_for(app, "/state", "GET")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            heartbeat = endpoint_for(app, "/tasks/{task_id}/heartbeat", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            claim_request = request_model(claim_task)(miner_id="token-miner")

            with self.assertRaises(HTTPException) as missing_claim:
                claim_task(claim_request)
            with self.assertRaises(HTTPException) as bad_claim:
                claim_task(claim_request, x_crowdtensor_miner_token="bad")

            self.assertEqual(missing_claim.exception.status_code, 401)
            self.assertEqual(bad_claim.exception.status_code, 401)
            self.assertEqual(state()["task_counts"]["leased"], 0)

            claim = claim_task(claim_request, x_crowdtensor_miner_token="miner-secret")
            heartbeat_request = request_model(heartbeat)(
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                runtime_status={"phase": "unit"},
            )
            with self.assertRaises(HTTPException) as bad_heartbeat:
                heartbeat(claim["task_id"], heartbeat_request, x_crowdtensor_miner_token="bad")
            self.assertEqual(bad_heartbeat.exception.status_code, 401)

            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="token-miner",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )
            result_request = request_model(result_task)(
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                local_delta=inner_result["local_delta"],
                metrics=inner_result,
            )
            with self.assertRaises(HTTPException) as missing_result:
                result_task(claim["task_id"], result_request)
            self.assertEqual(missing_result.exception.status_code, 401)

            result = result_task(
                claim["task_id"],
                result_request,
                x_crowdtensor_miner_token="miner-secret",
            )
            self.assertTrue(result["accepted"])
            summary = state()
            self.assertEqual(summary["accepted_results"], 1)
            self.assertEqual(summary["rejected_results"], 0)

    def test_hashed_shared_tokens_protect_miner_observer_and_admin_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                miner_token=hash_token("miner-secret"),
                observer_token=hash_token("observer-secret"),
                admin_token=hash_token("admin-secret"),
            )
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            state = endpoint_for(app, "/state", "GET")
            admin_events = endpoint_for(app, "/admin/events", "GET")

            with self.assertRaises(HTTPException) as bad_claim:
                claim_task(
                    request_model(claim_task)(miner_id="hashed-miner"),
                    x_crowdtensor_miner_token="bad",
                )
            self.assertEqual(bad_claim.exception.status_code, 401)

            claim = claim_task(
                request_model(claim_task)(miner_id="hashed-miner"),
                x_crowdtensor_miner_token="miner-secret",
            )
            self.assertEqual(claim["attempt"], 1)

            with self.assertRaises(HTTPException) as bad_state:
                state(x_crowdtensor_observer_token="bad")
            self.assertEqual(bad_state.exception.status_code, 401)
            self.assertIn("task_counts", state(x_crowdtensor_observer_token="observer-secret"))

            with self.assertRaises(HTTPException) as bad_admin:
                admin_events(limit=1, x_crowdtensor_admin_token="bad")
            self.assertEqual(bad_admin.exception.status_code, 403)
            self.assertIn("events", admin_events(limit=1, x_crowdtensor_admin_token="admin-secret"))

    def test_miner_token_registry_loader_validates_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            valid = write_registry(
                base / "valid.json",
                [
                    {"miner_id": "miner-a", "token": "token-a", "enabled": True, "label": "alpha"},
                    {"miner_id": "miner-b", "token": hash_token("token-b"), "enabled": False},
                ],
            )
            registry = load_miner_token_registry(valid)
            self.assertEqual(registry["miner-a"]["token"], "token-a")
            self.assertEqual(registry["miner-b"]["token"], hash_token("token-b"))
            self.assertTrue(registry["miner-a"]["enabled"])
            self.assertFalse(registry["miner-b"]["enabled"])

            cases = [
                ("missing-miners.json", {}),
                ("duplicate.json", {"miners": [{"miner_id": "a", "token": "1"}, {"miner_id": "a", "token": "2"}]}),
                ("missing-id.json", {"miners": [{"token": "1"}]}),
                ("missing-token.json", {"miners": [{"miner_id": "a"}]}),
                ("bad-enabled.json", {"miners": [{"miner_id": "a", "token": "1", "enabled": "yes"}]}),
                ("bad-hash-length.json", {"miners": [{"miner_id": "a", "token": "sha256:abc"}]}),
                ("bad-hash-hex.json", {"miners": [{"miner_id": "a", "token": "sha256:" + "z" * 64}]}),
            ]
            for filename, payload in cases:
                path = base / filename
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError, msg=filename):
                    load_miner_token_registry(path)

    def test_miner_token_registry_preserves_safe_join_policy_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = write_registry(
                Path(tmp) / "miners.json",
                [
                    {
                        "miner_id": "stage0-gpu",
                        "token": hash_token("registered-token"),
                        "enabled": True,
                        "label": "alpha",
                        "join_policy": {
                            "schema": "crowdtensor_miner_join_policy_v1",
                            "coordinator_url": "https://coord.example",
                            "stage": "stage0",
                            "backend": "cuda",
                            "hf_model_id": "sshleifer/tiny-gpt2",
                            "max_tasks": 4,
                            "max_runtime_seconds": 120.0,
                            "trust_tier": "probation",
                            "quota_task_limit": 25,
                            "claim_rate_limit": 2,
                            "claim_rate_window_seconds": 60.0,
                            "reward_account": "acct-secret",
                            "read_only_workload": "real_llm_sharded_infer",
                            "not_production": True,
                        },
                    },
                ],
            )

            registry = load_miner_token_registry(registry_path)
            summary = miner_registry_policy_summary(registry)
            encoded = json.dumps(summary, sort_keys=True)

            self.assertEqual(summary["schema"], "crowdtensor_miner_registry_policy_summary_v1")
            self.assertEqual(summary["miner_count"], 1)
            self.assertEqual(summary["policy_count"], 1)
            policy = summary["miners"][0]["policy"]
            self.assertEqual(policy["stage"], "stage0")
            self.assertEqual(policy["backend"], "cuda")
            self.assertEqual(policy["trust_tier"], "probation")
            self.assertEqual(policy["quota_task_limit"], 25)
            self.assertEqual(policy["claim_rate_limit"], 2)
            self.assertEqual(policy["claim_rate_window_seconds"], 60.0)
            self.assertTrue(policy["coordinator_url_present"])
            self.assertTrue(policy["reward_account_present"])
            self.assertNotIn("registered-token", encoded)
            self.assertNotIn("acct-secret", encoded)

    def test_ready_exposes_safe_miner_policy_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = write_registry(
                Path(tmp) / "miners.json",
                [
                    {
                        "miner_id": "stage1-cpu",
                        "token": hash_token("registered-token"),
                        "join_policy": {
                            "coordinator_url": "https://coord.example",
                            "stage": "stage1",
                            "backend": "cpu",
                            "hf_model_id": "sshleifer/tiny-gpt2",
                            "trust_tier": "new",
                            "quota_task_limit": 5,
                            "reward_account": "acct-secret",
                        },
                    },
                ],
            )
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                miner_token_registry=registry_path,
            )
            ready = endpoint_for(app, "/ready", "GET")

            payload = ready()
            encoded = json.dumps(payload, sort_keys=True)

            self.assertTrue(payload["ok"])
            self.assertTrue(payload["auth"]["miner_registry_configured"])
            policy_summary = payload["miner_policy_summary"]
            self.assertEqual(policy_summary["policy_count"], 1)
            self.assertEqual(policy_summary["miners"][0]["miner_id"], "stage1-cpu")
            self.assertEqual(policy_summary["miners"][0]["policy"]["stage"], "stage1")
            self.assertNotIn("registered-token", encoded)
            self.assertNotIn("acct-secret", encoded)

    def test_join_policy_scopes_claim_capabilities(self) -> None:
        registry = {
            "stage0-gpu": {
                "enabled": True,
                "token": hash_token("registered-token"),
                "join_policy": {
                    "stage": "stage0",
                    "backend": "cuda",
                    "hf_model_id": "sshleifer/tiny-gpt2",
                    "read_only_workload": WORKLOAD_REAL_LLM_SHARDED_INFER,
                },
            },
        }
        capabilities = {
            "runtime": "python-cli",
            "backend": "cuda",
            "protocol_version": "runtime_contract_v1",
            "supported_workloads": [WORKLOAD_REAL_LLM_SHARDED_INFER, "diloco_train"],
            "real_llm_runtime": {"model_id": "sshleifer/tiny-gpt2"},
            "real_llm_sharded_stage_capabilities": [
                REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
                REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
            ],
        }

        scoped, reason = enforce_miner_join_policy(
            miner_id="stage0-gpu",
            capabilities=capabilities,
            registry=registry,
        )

        self.assertEqual(reason, "")
        self.assertEqual(scoped["supported_workloads"], [WORKLOAD_REAL_LLM_SHARDED_INFER])
        self.assertEqual(scoped["real_llm_sharded_stage_capabilities"], [REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY])
        self.assertEqual(scoped["real_llm_runtime"]["adapter_kind"], "hf_transformers_cuda")

    def test_join_policy_rejects_mismatched_stage_claim_and_records_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = write_registry(
                Path(tmp) / "miners.json",
                [
                    {
                        "miner_id": "stage0-only",
                        "token": hash_token("registered-token"),
                        "join_policy": {
                            "stage": "stage0",
                            "backend": "cuda",
                            "hf_model_id": "sshleifer/tiny-gpt2",
                            "read_only_workload": WORKLOAD_REAL_LLM_SHARDED_INFER,
                        },
                    },
                ],
            )
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                miner_token_registry=registry_path,
                real_llm_backend="hf_transformers_cuda",
            )
            app.state.store.create_real_llm_sharded_inference_session(
                request_count=1,
                max_new_tokens=1,
                required_runtime="python-cli",
                required_backend="cuda",
                model_id="sshleifer/tiny-gpt2",
                llm_backend="hf_transformers_cuda",
            )
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            state = endpoint_for(app, "/state", "GET")

            with self.assertRaises(HTTPException) as blocked:
                claim_task(
                    request_model(claim_task)(
                        miner_id="stage0-only",
                        capabilities={
                            "runtime": "python-cli",
                            "backend": "cuda",
                            "protocol_version": "runtime_contract_v1",
                            "supported_workloads": [WORKLOAD_REAL_LLM_SHARDED_INFER],
                            "real_llm_runtime": {"model_id": "sshleifer/tiny-gpt2"},
                            "real_llm_sharded_stage_capabilities": [REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY],
                        },
                    ),
                    x_crowdtensor_miner_token="registered-token",
                )

            self.assertEqual(blocked.exception.status_code, 503)
            self.assertEqual(blocked.exception.detail, "join_policy_stage_mismatch")
            summary = state()
            self.assertEqual(summary["blocked_claims"], 1)
            self.assertEqual(summary["last_blocked_claim"]["reason"], "join_policy_stage_mismatch")

    def test_join_policy_allows_matching_stage_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = write_registry(
                Path(tmp) / "miners.json",
                [
                    {
                        "miner_id": "stage0-only",
                        "token": hash_token("registered-token"),
                        "join_policy": {
                            "stage": "stage0",
                            "backend": "cuda",
                            "hf_model_id": "sshleifer/tiny-gpt2",
                            "read_only_workload": WORKLOAD_REAL_LLM_SHARDED_INFER,
                        },
                    },
                ],
            )
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                miner_token_registry=registry_path,
                real_llm_backend="hf_transformers_cuda",
            )
            app.state.store.create_real_llm_sharded_inference_session(
                request_count=1,
                max_new_tokens=1,
                required_runtime="python-cli",
                required_backend="cuda",
                model_id="sshleifer/tiny-gpt2",
                llm_backend="hf_transformers_cuda",
            )
            claim_task = endpoint_for(app, "/tasks/claim", "POST")

            claim = claim_task(
                request_model(claim_task)(
                    miner_id="stage0-only",
                    capabilities={
                        "runtime": "python-cli",
                        "backend": "cuda",
                        "protocol_version": "runtime_contract_v1",
                        "supported_workloads": [WORKLOAD_REAL_LLM_SHARDED_INFER, "diloco_train"],
                        "real_llm_runtime": {"model_id": "sshleifer/tiny-gpt2"},
                        "real_llm_sharded_stage_capabilities": [
                            REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
                            REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
                        ],
                    },
                ),
                x_crowdtensor_miner_token="registered-token",
            )

            self.assertEqual(claim["workload_type"], WORKLOAD_REAL_LLM_SHARDED_INFER)
            self.assertEqual(claim["task_requirements"]["backend"], "cuda")

    def test_join_policy_quota_blocks_after_claim_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = write_registry(
                Path(tmp) / "miners.json",
                [
                    {
                        "miner_id": "quota-stage0",
                        "token": hash_token("registered-token"),
                        "join_policy": {
                            "stage": "stage0",
                            "backend": "cuda",
                            "hf_model_id": "sshleifer/tiny-gpt2",
                            "quota_task_limit": 1,
                            "read_only_workload": WORKLOAD_REAL_LLM_SHARDED_INFER,
                        },
                    },
                ],
            )
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                miner_token_registry=registry_path,
                real_llm_backend="hf_transformers_cuda",
            )
            for _ in range(2):
                app.state.store.create_real_llm_sharded_inference_session(
                    request_count=1,
                    max_new_tokens=1,
                    required_runtime="python-cli",
                    required_backend="cuda",
                    model_id="sshleifer/tiny-gpt2",
                    llm_backend="hf_transformers_cuda",
                )
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            state = endpoint_for(app, "/state", "GET")
            claim_request = request_model(claim_task)(
                miner_id="quota-stage0",
                capabilities={
                    "runtime": "python-cli",
                    "backend": "cuda",
                    "protocol_version": "runtime_contract_v1",
                    "supported_workloads": [WORKLOAD_REAL_LLM_SHARDED_INFER],
                    "real_llm_runtime": {"model_id": "sshleifer/tiny-gpt2"},
                    "real_llm_sharded_stage_capabilities": [REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY],
                },
            )

            first = claim_task(claim_request, x_crowdtensor_miner_token="registered-token")
            with self.assertRaises(HTTPException) as blocked:
                claim_task(claim_request, x_crowdtensor_miner_token="registered-token")

            self.assertEqual(first["workload_type"], WORKLOAD_REAL_LLM_SHARDED_INFER)
            self.assertEqual(blocked.exception.status_code, 503)
            self.assertEqual(blocked.exception.detail, "join_policy_quota_exhausted")
            summary = state()
            self.assertEqual(summary["blocked_claims"], 1)
            self.assertEqual(summary["last_blocked_claim"]["reason"], "join_policy_quota_exhausted")

    def test_join_policy_rate_limits_registered_miner_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = write_registry(
                Path(tmp) / "miners.json",
                [
                    {
                        "miner_id": "rate-stage0",
                        "token": hash_token("registered-token"),
                        "join_policy": {
                            "stage": "stage0",
                            "backend": "cuda",
                            "hf_model_id": "sshleifer/tiny-gpt2",
                            "claim_rate_limit": 1,
                            "claim_rate_window_seconds": 60.0,
                            "read_only_workload": WORKLOAD_REAL_LLM_SHARDED_INFER,
                        },
                    },
                ],
            )
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                miner_token_registry=registry_path,
                real_llm_backend="hf_transformers_cuda",
            )
            for _ in range(2):
                app.state.store.create_real_llm_sharded_inference_session(
                    request_count=1,
                    max_new_tokens=1,
                    required_runtime="python-cli",
                    required_backend="cuda",
                    model_id="sshleifer/tiny-gpt2",
                    llm_backend="hf_transformers_cuda",
                )
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            state = endpoint_for(app, "/state", "GET")
            claim_request = request_model(claim_task)(
                miner_id="rate-stage0",
                capabilities={
                    "runtime": "python-cli",
                    "backend": "cuda",
                    "protocol_version": "runtime_contract_v1",
                    "supported_workloads": [WORKLOAD_REAL_LLM_SHARDED_INFER],
                    "real_llm_runtime": {"model_id": "sshleifer/tiny-gpt2"},
                    "real_llm_sharded_stage_capabilities": [REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY],
                },
            )

            first = claim_task(claim_request, x_crowdtensor_miner_token="registered-token")
            with self.assertRaises(HTTPException) as blocked:
                claim_task(claim_request, x_crowdtensor_miner_token="registered-token")

            self.assertEqual(first["workload_type"], WORKLOAD_REAL_LLM_SHARDED_INFER)
            self.assertEqual(blocked.exception.status_code, 429)
            self.assertEqual(blocked.exception.detail, "join_policy_rate_limited")
            summary = state()
            self.assertEqual(summary["blocked_claims"], 1)
            self.assertEqual(summary["last_blocked_claim"]["reason"], "join_policy_rate_limited")

    def test_hashed_miner_token_registry_claims_with_plaintext_request_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = write_registry(
                Path(tmp) / "miners.json",
                [
                    {"miner_id": "registered", "token": hash_token("registered-token")},
                ],
            )
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=1,
                miner_token=hash_token("shared-token"),
                miner_token_registry=registry_path,
            )
            claim_task = endpoint_for(app, "/tasks/claim", "POST")

            with self.assertRaises(HTTPException) as shared_for_registered:
                claim_task(
                    request_model(claim_task)(miner_id="registered"),
                    x_crowdtensor_miner_token="shared-token",
                )
            self.assertEqual(shared_for_registered.exception.status_code, 401)

            registered_claim = claim_task(
                request_model(claim_task)(miner_id="registered"),
                x_crowdtensor_miner_token="registered-token",
            )
            self.assertEqual(registered_claim["attempt"], 1)

    def test_miner_token_registry_claim_rules_and_shared_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = write_registry(
                Path(tmp) / "miners.json",
                [
                    {"miner_id": "registered", "token": "registered-token"},
                    {"miner_id": "disabled", "token": "disabled-token", "enabled": False},
                ],
            )
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=2,
                miner_token="shared-token",
                miner_token_registry=registry_path,
            )
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            heartbeat = endpoint_for(app, "/tasks/{task_id}/heartbeat", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")

            with self.assertRaises(HTTPException) as shared_for_registered:
                claim_task(
                    request_model(claim_task)(miner_id="registered"),
                    x_crowdtensor_miner_token="shared-token",
                )
            with self.assertRaises(HTTPException) as disabled:
                claim_task(
                    request_model(claim_task)(miner_id="disabled"),
                    x_crowdtensor_miner_token="disabled-token",
                )
            self.assertEqual(shared_for_registered.exception.status_code, 401)
            self.assertEqual(disabled.exception.detail, "miner token is disabled")

            registered_claim = claim_task(
                request_model(claim_task)(miner_id="registered"),
                x_crowdtensor_miner_token="registered-token",
            )
            unknown_claim = claim_task(
                request_model(claim_task)(miner_id="unknown"),
                x_crowdtensor_miner_token="shared-token",
            )
            with self.assertRaises(HTTPException) as bad_heartbeat:
                heartbeat(
                    registered_claim["task_id"],
                    request_model(heartbeat)(
                        lease_token=registered_claim["lease_token"],
                        attempt=registered_claim["attempt"],
                    ),
                    x_crowdtensor_miner_token="bad",
                )
            self.assertEqual(bad_heartbeat.exception.status_code, 401)
            heartbeat(
                registered_claim["task_id"],
                request_model(heartbeat)(
                    lease_token=registered_claim["lease_token"],
                    attempt=registered_claim["attempt"],
                ),
                x_crowdtensor_miner_token="registered-token",
            )

            inner_result = run_inner_loop(
                registered_claim["weights"],
                task_id=registered_claim["task_id"],
                miner_id="registered",
                model_version=registered_claim["model_version"],
                inner_steps=registered_claim["inner_steps"],
            )
            with self.assertRaises(HTTPException) as bad_result:
                result_task(
                    registered_claim["task_id"],
                    request_model(result_task)(
                        lease_token=registered_claim["lease_token"],
                        attempt=registered_claim["attempt"],
                        local_delta=inner_result["local_delta"],
                        metrics=inner_result,
                    ),
                    x_crowdtensor_miner_token="bad",
                )
            self.assertEqual(bad_result.exception.status_code, 401)
            result = result_task(
                registered_claim["task_id"],
                request_model(result_task)(
                    lease_token=registered_claim["lease_token"],
                    attempt=registered_claim["attempt"],
                    local_delta=inner_result["local_delta"],
                    metrics=inner_result,
                ),
                x_crowdtensor_miner_token="registered-token",
            )

            self.assertTrue(result["accepted"])
            self.assertEqual(unknown_claim["attempt"], 1)

    def test_miner_token_cors_header_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                cors_origins=["http://browser.test"],
                miner_token="miner-secret",
            )

            cors = [
                middleware
                for middleware in app.user_middleware
                if middleware.cls.__name__ == "CORSMiddleware"
            ]
            self.assertEqual(len(cors), 1)
            allow_headers = cors[0].kwargs["allow_headers"]
            self.assertIn("x-crowdtensor-miner-token", allow_headers)

    def test_observer_token_protects_state_and_metrics_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                admin_token="admin-secret",
                observer_token="observer-secret",
            )
            health = endpoint_for(app, "/health", "GET")
            ready = endpoint_for(app, "/ready", "GET")
            version = endpoint_for(app, "/version", "GET")
            state = endpoint_for(app, "/state", "GET")
            metrics = endpoint_for(app, "/metrics", "GET")

            self.assertTrue(health()["ok"])
            self.assertEqual(version()["api_status"], "alpha")
            ready_payload = ready()
            self.assertTrue(ready_payload["ok"])
            self.assertEqual(ready_payload["auth"]["observer_required"], True)
            self.assertEqual(ready_payload["auth"]["admin_configured"], True)
            with self.assertRaises(HTTPException) as missing_state:
                state()
            with self.assertRaises(HTTPException) as bad_state:
                state(x_crowdtensor_observer_token="bad")
            with self.assertRaises(HTTPException) as admin_state:
                state(x_crowdtensor_observer_token="admin-secret")
            with self.assertRaises(HTTPException) as bad_metrics:
                metrics(x_crowdtensor_observer_token="bad")

            self.assertEqual(missing_state.exception.status_code, 401)
            self.assertEqual(bad_state.exception.status_code, 401)
            self.assertEqual(admin_state.exception.status_code, 401)
            self.assertEqual(bad_metrics.exception.status_code, 401)

            summary = state(x_crowdtensor_observer_token="observer-secret")
            response = metrics(x_crowdtensor_observer_token="observer-secret")
            self.assertIn("task_counts", summary)
            self.assertIn("text/plain; version=0.0.4", response.headers["content-type"])

    def test_observer_token_cors_header_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                cors_origins=["http://browser.test"],
                observer_token="observer-secret",
            )

            cors = [
                middleware
                for middleware in app.user_middleware
                if middleware.cls.__name__ == "CORSMiddleware"
            ]
            self.assertEqual(len(cors), 1)
            allow_headers = cors[0].kwargs["allow_headers"]
            self.assertIn("x-crowdtensor-observer-token", allow_headers)

    def test_metrics_endpoint_exposes_safe_aggregate_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10, admin_token="secret")
            metrics = endpoint_for(app, "/metrics", "GET")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            admin_trust = endpoint_for(app, "/admin/trust-overrides", "POST")

            claim = claim_task(request_model(claim_task)(miner_id="metrics-miner"))
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="metrics-miner",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )
            result_task(
                claim["task_id"],
                request_model(result_task)(
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    local_delta=inner_result["local_delta"],
                    metrics=inner_result,
                ),
            )
            admin_trust(
                request_model(admin_trust)(
                    miner_id="metrics-miner",
                    workload_type="diloco_train",
                    mode="block",
                    reason="metrics test",
                ),
                x_crowdtensor_admin_token="secret",
            )

            response = metrics()
            text = response.body.decode("utf-8")

            self.assertIn("text/plain; version=0.0.4", response.headers["content-type"])
            self.assertIn('crowdtensord_task_count{status="completed"} 1', text)
            self.assertIn('crowdtensord_results_total{result="accepted"} 1', text)
            self.assertIn('crowdtensord_model_step{step="global"} 1', text)
            self.assertIn('crowdtensord_model_step{step="micro_transformer"} 0', text)
            self.assertIn('crowdtensord_model_step{step="model_bundle"} 0', text)
            self.assertIn('crowdtensord_model_updates_total{target="micro_transformer"} 0', text)
            self.assertIn('crowdtensord_model_updates_total{target="model_bundle"} 0', text)
            self.assertIn('crowdtensord_trust_overrides{mode="block"} 1', text)
            self.assertIn('crowdtensord_miner_workload_blocks{source="manual"} 1', text)
            self.assertNotIn("metrics-miner", text)
            self.assertNotIn("lease_token", text)
            self.assertNotIn(claim["lease_token"], text)

    def test_state_reports_async_staleness_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10, backlog=2)
            state = endpoint_for(app, "/state", "GET")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")

            claims = [
                claim_task(request_model(claim_task)(miner_id=f"api-miner-{index}"))
                for index in range(2)
            ]
            for index, claim in enumerate(claims):
                inner_result = run_inner_loop(
                    claim["weights"],
                    task_id=claim["task_id"],
                    miner_id=f"api-miner-{index}",
                    model_version=claim["model_version"],
                    inner_steps=claim["inner_steps"],
                )
                result_task(
                    claim["task_id"],
                    request_model(result_task)(
                        lease_token=claim["lease_token"],
                        attempt=claim["attempt"],
                        local_delta=inner_result["local_delta"],
                        metrics=inner_result,
                    ),
                )

            summary = state()
            self.assertEqual(summary["accepted_results"], 2)
            self.assertEqual(summary["max_staleness"], 1)
            self.assertAlmostEqual(summary["avg_staleness"], 0.5)
            completed = [
                task for task in summary["tasks"]
                if task["status"] == "completed"
            ]
            self.assertEqual([task["staleness"] for task in completed], [0, 1])
            self.assertEqual([task["result_model_version"] for task in completed], [1, 2])

    def test_quality_rejection_returns_422_and_updates_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10)
            state = endpoint_for(app, "/state", "GET")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            claim = claim_task(request_model(claim_task)(miner_id="bad-api-miner"))

            with self.assertRaises(HTTPException) as raised:
                result_task(
                    claim["task_id"],
                    request_model(result_task)(
                        lease_token=claim["lease_token"],
                        attempt=claim["attempt"],
                        local_delta=[1000.0 for _ in claim["weights"]],
                    ),
                )

            self.assertEqual(raised.exception.status_code, 422)
            self.assertEqual(raised.exception.detail["code"], "delta_norm_too_large")
            summary = state()
            self.assertEqual(summary["accepted_results"], 0)
            self.assertEqual(summary["rejected_results"], 1)
            self.assertEqual(summary["miner_scores"]["bad-api-miner"]["rejected"], 1)

    def test_replay_audit_rejection_returns_422_and_updates_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                replay_audit=True,
            )
            state = endpoint_for(app, "/state", "GET")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            claim = claim_task(request_model(claim_task)(miner_id="bad-audit-api-miner"))

            with self.assertRaises(HTTPException) as raised:
                result_task(
                    claim["task_id"],
                    request_model(result_task)(
                        lease_token=claim["lease_token"],
                        attempt=claim["attempt"],
                        local_delta=[0.0 for _ in claim["weights"]],
                    ),
                )

            self.assertEqual(claim["audit_mode"], "replay")
            self.assertEqual(raised.exception.status_code, 422)
            self.assertEqual(raised.exception.detail["code"], "local_delta_replay_mismatch")
            self.assertEqual(raised.exception.detail["audit_code"], "local_delta_replay_mismatch")
            summary = state()
            self.assertEqual(summary["audit_results"], 1)
            self.assertEqual(summary["audit_rejections"], 1)
            self.assertEqual(summary["last_rejected"]["validation"]["audit_code"], "local_delta_replay_mismatch")

    def test_pseudo_gradient_alias_is_still_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10)
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            claim = claim_task(request_model(claim_task)(miner_id="compat-miner"))
            gradient = compute_pseudo_gradient(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="compat-miner",
                inner_steps=claim["inner_steps"],
            )

            result = result_task(
                claim["task_id"],
                request_model(result_task)(
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    pseudo_gradient=gradient,
                ),
            )
            self.assertEqual(result["global_step"], 1)

    def test_timeout_requeue_and_stale_result_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=0.05, inner_steps=10, reaper_interval=0.01)
            state = endpoint_for(app, "/state", "GET")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")

            stale = claim_task(request_model(claim_task)(miner_id="stale-miner"))
            time.sleep(0.09)
            self.assertEqual(state()["task_counts"]["queued"], 1)

            stale_gradient = compute_pseudo_gradient(
                stale["weights"],
                task_id=stale["task_id"],
                miner_id="stale-miner",
                inner_steps=stale["inner_steps"],
            )
            stale_result_request = request_model(result_task)(
                lease_token=stale["lease_token"],
                attempt=stale["attempt"],
                pseudo_gradient=stale_gradient,
            )
            with self.assertRaises(HTTPException) as raised:
                result_task(stale["task_id"], stale_result_request)
            self.assertEqual(raised.exception.status_code, 409)

            fresh = claim_task(request_model(claim_task)(miner_id="fresh-miner"))
            self.assertEqual(fresh["task_id"], stale["task_id"])
            self.assertEqual(fresh["attempt"], stale["attempt"] + 1)

    def test_heartbeat_extends_live_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=1.0, inner_steps=10, reaper_interval=0.01)
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            heartbeat_task = endpoint_for(app, "/tasks/{task_id}/heartbeat", "POST")

            claim = claim_task(request_model(claim_task)(miner_id="heartbeat-miner"))
            time.sleep(0.05)
            heartbeat = heartbeat_task(
                claim["task_id"],
                request_model(heartbeat_task)(
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                ),
            )
            self.assertGreater(heartbeat["lease_expires_at"], claim["lease_expires_at"])

    def test_claim_and_heartbeat_accept_miner_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10)
            state = endpoint_for(app, "/state", "GET")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            heartbeat_task = endpoint_for(app, "/tasks/{task_id}/heartbeat", "POST")
            claim = claim_task(
                request_model(claim_task)(
                    miner_id="metadata-miner",
                    capabilities={"runtime": "browser", "backend": "js-worker"},
                )
            )
            heartbeat_task(
                claim["task_id"],
                request_model(heartbeat_task)(
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    runtime_status={"phase": "training"},
                ),
            )

            task = state()["tasks"][0]
            self.assertEqual(task["capabilities"]["runtime"], "browser")
            self.assertEqual(task["runtime_status"]["phase"], "training")

    def test_incompatible_claim_returns_503_and_updates_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10, backlog=0)
            state = endpoint_for(app, "/state", "GET")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            app.state.store._create_task(  # noqa: SLF001 - scheduler fixture
                required_runtime="python-cli",
                required_backend="cpu",
            )

            with self.assertRaises(HTTPException) as raised:
                claim_task(
                    request_model(claim_task)(
                        miner_id="browser-api-miner",
                        capabilities={
                            "runtime": "browser",
                            "backend": "js-worker",
                            "protocol_version": "runtime_contract_v1",
                        },
                    )
                )

            self.assertEqual(raised.exception.status_code, 503)
            self.assertEqual(raised.exception.detail, "no compatible queued task available")
            summary = state()
            self.assertEqual(summary["incompatible_claims"], 1)
            self.assertEqual(
                summary["task_counts_by_requirement"]["python-cli/cpu/runtime_contract_v1"]["queued"],
                1,
            )

    def test_quarantined_claim_returns_503_and_updates_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "cpu_lora_mock",
                    },
                ],
            )
            state = endpoint_for(app, "/state", "GET")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            capabilities = {
                "runtime": "python-cli",
                "backend": "cpu",
                "protocol_version": "runtime_contract_v1",
                "supported_workloads": ["cpu_lora_mock"],
            }

            for _ in range(2):
                claim = claim_task(
                    request_model(claim_task)(
                        miner_id="bad-lora-api-miner",
                        capabilities=capabilities,
                    )
                )
                with self.assertRaises(HTTPException) as rejected:
                    result_task(
                        claim["task_id"],
                        request_model(result_task)(
                            lease_token=claim["lease_token"],
                            attempt=claim["attempt"],
                            adapter_delta={"values": [100.0, 0.0, 0.0]},
                        ),
                    )
                self.assertEqual(rejected.exception.status_code, 422)

            with self.assertRaises(HTTPException) as blocked:
                claim_task(
                    request_model(claim_task)(
                        miner_id="bad-lora-api-miner",
                        capabilities=capabilities,
                    )
                )

            self.assertEqual(blocked.exception.status_code, 503)
            self.assertEqual(blocked.exception.detail, "miner quarantined for workload")
            summary = state()
            self.assertEqual(summary["blocked_claims"], 1)
            self.assertIn("bad-lora-api-miner", summary["quarantined_miners"])
            self.assertTrue(
                summary["miner_workload_scores"]["bad-lora-api-miner"]["cpu_lora_mock"]["quarantined"]
            )

    def test_admin_token_required_for_control_plane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10, admin_token="")
            admin_trust = endpoint_for(app, "/admin/trust-overrides", "POST")
            override_request = request_model(admin_trust)(
                miner_id="api-miner",
                workload_type="diloco_train",
                mode="block",
                reason="test",
            )

            with self.assertRaises(HTTPException) as missing:
                admin_trust(override_request, x_crowdtensor_admin_token="secret")
            self.assertEqual(missing.exception.status_code, 403)
            self.assertEqual(missing.exception.detail, "admin token is not configured")

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10, admin_token="secret")
            admin_trust = endpoint_for(app, "/admin/trust-overrides", "POST")
            override_request = request_model(admin_trust)(
                miner_id="api-miner",
                workload_type="diloco_train",
                mode="block",
                reason="test",
            )

            with self.assertRaises(HTTPException) as invalid:
                admin_trust(override_request, x_crowdtensor_admin_token="wrong")
            self.assertEqual(invalid.exception.status_code, 403)
            self.assertEqual(invalid.exception.detail, "invalid admin token")

    def test_admin_trust_override_blocks_allows_and_events_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                admin_token="secret",
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "diloco_train",
                    },
                ],
            )
            state = endpoint_for(app, "/state", "GET")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            admin_events = endpoint_for(app, "/admin/events", "GET")
            admin_trust = endpoint_for(app, "/admin/trust-overrides", "POST")
            request_type = request_model(admin_trust)

            blocked = admin_trust(
                request_type(
                    miner_id="operator-api-miner",
                    workload_type="diloco_train",
                    mode="block",
                    reason="operator block",
                ),
                x_crowdtensor_admin_token="secret",
            )
            self.assertTrue(blocked["accepted"])
            with self.assertRaises(HTTPException) as blocked_claim:
                claim_task(
                    request_model(claim_task)(
                        miner_id="operator-api-miner",
                        capabilities={
                            "runtime": "python-cli",
                            "backend": "cpu",
                            "protocol_version": "runtime_contract_v1",
                            "supported_workloads": ["diloco_train"],
                        },
                    )
                )
            self.assertEqual(blocked_claim.exception.status_code, 503)
            self.assertEqual(blocked_claim.exception.detail, "miner manually blocked for workload")

            allowed = admin_trust(
                request_type(
                    miner_id="operator-api-miner",
                    workload_type="diloco_train",
                    mode="allow",
                    reason="operator allow",
                ),
                x_crowdtensor_admin_token="secret",
            )
            self.assertTrue(allowed["accepted"])
            claim = claim_task(
                request_model(claim_task)(
                    miner_id="operator-api-miner",
                    capabilities={
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "protocol_version": "runtime_contract_v1",
                        "supported_workloads": ["diloco_train"],
                    },
                )
            )
            summary = state()
            events = admin_events(limit=20, x_crowdtensor_admin_token="secret")

            self.assertEqual(claim["workload_type"], "diloco_train")
            self.assertEqual(summary["miner_trust_overrides"]["operator-api-miner"]["diloco_train"]["mode"], "allow")
            claimed_events = [event for event in events["events"] if event["type"] == "task_claimed"]
            self.assertTrue(claimed_events)
            self.assertEqual(claimed_events[-1]["lease_token"], "<redacted>")

    def test_admin_results_reports_safe_result_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10, admin_token="secret")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            admin_results = endpoint_for(app, "/admin/results", "GET")
            claim = claim_task(request_model(claim_task)(miner_id="ledger-api-miner"))
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="ledger-api-miner",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )
            result = result_task(
                claim["task_id"],
                request_model(result_task)(
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    idempotency_key="api-ledger-key",
                    local_delta=inner_result["local_delta"],
                    metrics=inner_result,
                ),
            )
            self.assertTrue(result["accepted"])

            with self.assertRaises(HTTPException) as bad_token:
                admin_results(limit=10, x_crowdtensor_admin_token="bad")
            self.assertEqual(bad_token.exception.status_code, 403)
            with self.assertRaises(HTTPException) as bad_status:
                admin_results(limit=10, status="broken", x_crowdtensor_admin_token="secret")
            self.assertEqual(bad_status.exception.status_code, 422)

            ledger = admin_results(
                limit=10,
                status="accepted",
                miner_id="ledger-api-miner",
                workload_type="diloco_train",
                task_id=claim["task_id"],
                x_crowdtensor_admin_token="secret",
            )
            self.assertEqual(ledger["status"], "accepted")
            self.assertEqual(ledger["task_id"], claim["task_id"])
            self.assertEqual(len(ledger["results"]), 1)
            row = ledger["results"][0]
            self.assertEqual(row["task_id"], claim["task_id"])
            self.assertTrue(row["idempotent"])
            self.assertEqual(row["validation"]["code"], "ok")
            self.assertEqual(row["audit"], {})
            self.assertTrue(row["model_updated"])
            public_text = json.dumps(ledger, sort_keys=True)
            self.assertNotIn("api-ledger-key", public_text)
            self.assertNotIn("lease_token", public_text)
            self.assertNotIn("result_idempotency_key_hash", public_text)

    def test_admin_accounting_reports_safe_miner_summary_with_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = write_registry(
                Path(tmp) / "miners.json",
                [
                    {
                        "miner_id": "accounting-api-miner",
                        "token": hash_token("registered-token"),
                        "join_policy": {
                            "stage": "both",
                            "backend": "cpu",
                            "hf_model_id": "sshleifer/tiny-gpt2",
                            "trust_tier": "probation",
                            "quota_task_limit": 2,
                            "claim_rate_limit": 1,
                            "claim_rate_window_seconds": 30.0,
                            "reward_account": "acct-secret",
                            "read_only_workload": "diloco_train",
                        },
                    },
                ],
            )
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                admin_token="secret",
                miner_token_registry=registry_path,
            )
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            admin_accounting = endpoint_for(app, "/admin/accounting", "GET")
            claim = claim_task(
                request_model(claim_task)(miner_id="accounting-api-miner"),
                x_crowdtensor_miner_token="registered-token",
            )
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="accounting-api-miner",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )
            result_task(
                claim["task_id"],
                request_model(result_task)(
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    idempotency_key="accounting-api-secret",
                    local_delta=inner_result["local_delta"],
                    metrics={**inner_result, "elapsed_ms": 12.0},
                ),
                x_crowdtensor_miner_token="registered-token",
            )

            with self.assertRaises(HTTPException) as bad_token:
                admin_accounting(limit=10, x_crowdtensor_admin_token="bad")
            self.assertEqual(bad_token.exception.status_code, 403)
            with self.assertRaises(HTTPException) as bad_status:
                admin_accounting(status="broken", x_crowdtensor_admin_token="secret")
            self.assertEqual(bad_status.exception.status_code, 422)

            accounting = admin_accounting(
                limit=10,
                status="accepted",
                miner_id="accounting-api-miner",
                workload_type="diloco_train",
                x_crowdtensor_admin_token="secret",
            )
            payload = json.dumps(accounting, sort_keys=True)

            self.assertEqual(accounting["schema"], "miner_accounting_summary_v1")
            self.assertEqual(accounting["row_count"], 1)
            row = accounting["rows"][0]
            self.assertEqual(row["miner_id"], "accounting-api-miner")
            self.assertEqual(row["accounting_status"], "accepted")
            self.assertEqual(row["work_units"]["inner_steps"], 10)
            self.assertEqual(row["join_policy"]["trust_tier"], "probation")
            self.assertEqual(row["join_policy"]["quota_task_limit"], 2)
            self.assertEqual(row["join_policy"]["claim_rate_limit"], 1)
            self.assertEqual(row["join_policy"]["claim_rate_window_seconds"], 30.0)
            self.assertTrue(row["join_policy"]["reward_account_present"])
            totals = accounting["miner_totals"]["accounting-api-miner/diloco_train"]
            self.assertEqual(totals["accepted"], 1)
            self.assertEqual(totals["work_units"]["inner_steps"], 10)
            self.assertEqual(totals["join_policy"]["claim_rate_limit"], 1)
            self.assertEqual(totals["join_policy"]["claim_rate_window_seconds"], 30.0)
            self.assertTrue(totals["join_policy"]["reward_account_present"])
            self.assertNotIn("acct-secret", payload)
            self.assertNotIn("accounting-api-secret", payload)
            self.assertNotIn("lease_token", payload)

    def test_admin_inference_session_enqueues_read_only_task_and_filters_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                admin_token="secret",
            )
            admin_inference = endpoint_for(app, "/admin/inference-sessions", "POST")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            admin_results = endpoint_for(app, "/admin/results", "GET")
            state = endpoint_for(app, "/state", "GET")
            request_type = request_model(admin_inference)

            with self.assertRaises(HTTPException) as missing_admin:
                admin_inference(request_type(request_count=3))
            self.assertEqual(missing_admin.exception.status_code, 403)
            session = admin_inference(
                request_type(request_count=3, scenario_id="route-baseline"),
                x_crowdtensor_admin_token="secret",
            )
            self.assertEqual(session["schema"], "inference_session_request_v1")
            self.assertEqual(session["workload_type"], WORKLOAD_MODEL_BUNDLE_INFER)
            self.assertEqual(session["request_count"], 3)
            self.assertEqual(session["scenario_id"], "route-baseline")
            self.assertEqual(session["status"], "queued")
            self.assertIn(session["task_id"], session["result_query"])
            self.assertEqual(state()["task_counts"]["queued"], 1)

            claim = claim_task(
                request_model(claim_task)(
                    miner_id="admin-infer-miner",
                    capabilities={
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "protocol_version": "runtime_contract_v1",
                        "supported_workloads": [WORKLOAD_MODEL_BUNDLE_INFER],
                    },
                )
            )
            self.assertEqual(claim["task_id"], session["task_id"])
            self.assertEqual(claim["workload_spec"]["request_count"], 3)
            self.assertEqual(claim["workload_spec"]["scenario_id"], "route-baseline")
            inner_result = run_model_bundle_inference(claim["workload_spec"])
            result = result_task(
                claim["task_id"],
                request_model(result_task)(
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    idempotency_key="admin-inference-key",
                    inference_result=inner_result["inference_result"],
                    inference_results=inner_result["inference_results"],
                    metrics=inner_result,
                ),
            )
            self.assertTrue(result["accepted"])
            self.assertFalse(result["model_updated"])
            self.assertFalse(result["model_bundle_updated"])

            ledger = admin_results(
                limit=10,
                status="accepted",
                miner_id="",
                workload_type=WORKLOAD_MODEL_BUNDLE_INFER,
                task_id=session["task_id"],
                x_crowdtensor_admin_token="secret",
            )
            self.assertEqual(len(ledger["results"]), 1)
            row = ledger["results"][0]
            self.assertEqual(row["task_id"], session["task_id"])
            self.assertEqual(row["validation"]["request_count"], 3)
            self.assertEqual(row["validation"]["scenario_id"], "route-baseline")
            self.assertEqual(row["session_metrics"]["request_count"], 3)
            self.assertEqual(row["session_metrics"]["scenario_id"], "route-baseline")
            self.assertFalse(row["model_updated"])
            self.assertFalse(row["model_bundle_updated"])
            public_text = json.dumps(ledger, sort_keys=True)
            self.assertNotIn("admin-inference-key", public_text)
            self.assertNotIn("lease_token", public_text)
            self.assertNotIn("inference_results", public_text)

    def test_admin_inference_session_can_enqueue_external_llm_read_only_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                admin_token="secret",
            )
            admin_inference = endpoint_for(app, "/admin/inference-sessions", "POST")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            admin_results = endpoint_for(app, "/admin/results", "GET")
            request_type = request_model(admin_inference)

            session = admin_inference(
                request_type(request_count=3, workload_type=WORKLOAD_EXTERNAL_LLM_INFER),
                x_crowdtensor_admin_token="secret",
            )
            self.assertEqual(session["schema"], "inference_session_request_v1")
            self.assertEqual(session["workload_type"], WORKLOAD_EXTERNAL_LLM_INFER)
            self.assertEqual(session["request_count"], 3)
            self.assertEqual(session["scenario_id"], "")
            self.assertIn(WORKLOAD_EXTERNAL_LLM_INFER, session["result_query"])

            claim = claim_task(
                request_model(claim_task)(
                    miner_id="admin-external-llm-miner",
                    capabilities={
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "protocol_version": "runtime_contract_v1",
                        "supported_workloads": [WORKLOAD_EXTERNAL_LLM_INFER],
                        "external_llm_runtime": {"adapter_kind": "mock", "model_id": "mock-llm"},
                    },
                )
            )
            self.assertEqual(claim["task_id"], session["task_id"])
            self.assertEqual(claim["workload_type"], WORKLOAD_EXTERNAL_LLM_INFER)
            self.assertEqual(claim["workload_spec"]["request_count"], 3)
            inner_result = run_mock_external_llm_inference(claim["workload_spec"])
            result = result_task(
                claim["task_id"],
                request_model(result_task)(
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    idempotency_key="admin-external-llm-key",
                    external_llm_result=inner_result["external_llm_result"],
                    external_llm_results=inner_result["external_llm_results"],
                    metrics=inner_result,
                ),
            )
            self.assertTrue(result["accepted"])
            self.assertFalse(result["model_updated"])
            self.assertFalse(result["model_bundle_updated"])

            ledger = admin_results(
                limit=10,
                status="accepted",
                miner_id="",
                workload_type=WORKLOAD_EXTERNAL_LLM_INFER,
                task_id=session["task_id"],
                x_crowdtensor_admin_token="secret",
            )
            self.assertEqual(len(ledger["results"]), 1)
            row = ledger["results"][0]
            self.assertEqual(row["task_id"], session["task_id"])
            self.assertEqual(row["validation"]["request_count"], 3)
            self.assertEqual(row["session_metrics"]["completion_count"], 3)
            public_text = json.dumps(ledger, sort_keys=True)
            self.assertNotIn("admin-external-llm-key", public_text)
            self.assertNotIn("lease_token", public_text)
            self.assertNotIn("external_llm_results", public_text)
            self.assertNotIn("output_text", public_text)

    def test_admin_inference_session_rejects_unsupported_runtime_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10, backlog=0, admin_token="secret")
            admin_inference = endpoint_for(app, "/admin/inference-sessions", "POST")
            request_type = request_model(admin_inference)

            with self.assertRaises(HTTPException) as rejected:
                admin_inference(
                    request_type(request_count=1, runtime="browser"),
                    x_crowdtensor_admin_token="secret",
                )
            self.assertEqual(rejected.exception.status_code, 422)
            with self.assertRaises(HTTPException) as bad_scenario:
                admin_inference(
                    request_type(request_count=1, scenario_id="freeform-prompt"),
                    x_crowdtensor_admin_token="secret",
                )
            self.assertEqual(bad_scenario.exception.status_code, 422)
            with self.assertRaises(HTTPException) as bad_workload:
                admin_inference(
                    request_type(request_count=1, workload_type="public_chat"),
                    x_crowdtensor_admin_token="secret",
                )
            self.assertEqual(bad_workload.exception.status_code, 422)

    def test_real_llm_inference_session_reports_missing_hf_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10, backlog=0, admin_token="secret")
            admin_inference = endpoint_for(app, "/admin/inference-sessions", "POST")
            request_type = request_model(admin_inference)

            with patch(
                "crowdtensor.state_store.inspect_real_llm_artifact",
                side_effect=RuntimeError(
                    "real_llm_sharded_infer requires optional Hugging Face dependencies: transformers. "
                    "Install with: python -m pip install -e .[hf]"
                ),
            ), self.assertRaises(HTTPException) as missing:
                admin_inference(
                    request_type(request_count=1, workload_type="real_llm_sharded_infer"),
                    x_crowdtensor_admin_token="secret",
                )
            self.assertEqual(missing.exception.status_code, 503)
            self.assertIn("requires optional Hugging Face dependencies", str(missing.exception.detail))

    def test_real_llm_cuda_session_uses_cpu_coordinator_metadata_only(self) -> None:
        artifact = {
            "schema": "real_llm_artifact_v1",
            "artifact_hash": "sha256:test-real-llm-cuda-artifact",
            "model_id": "sshleifer/tiny-gpt2",
            "backend": "hf_transformers_cuda",
            "split_index": 1,
            "num_hidden_layers": 2,
            "hidden_size": 2,
            "vocab_size": 128,
            "read_only": True,
            "metadata_only": True,
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "crowdtensor.state_store.inspect_real_llm_artifact",
            return_value=artifact,
        ) as inspect_artifact:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10, backlog=0, admin_token="secret")
            admin_inference = endpoint_for(app, "/admin/inference-sessions", "POST")
            request_type = request_model(admin_inference)

            session = admin_inference(
                request_type(
                    request_count=1,
                    workload_type="real_llm_sharded_infer",
                    backend="cuda",
                    runtime="python-cli",
                ),
                x_crowdtensor_admin_token="secret",
            )

            inspect_artifact.assert_called_once()
            self.assertFalse(inspect_artifact.call_args.kwargs["require_runtime"])
            self.assertEqual(session["backend"], "hf_transformers_cuda")
            self.assertEqual(session["task_requirements"]["backend"], "cuda")
            self.assertEqual(session["task_requirements"]["stage_capability"], "real_llm_sharded_cuda_stage0")

    def test_real_llm_session_uses_requested_hf_model_id(self) -> None:
        artifact = {
            "schema": "real_llm_artifact_v1",
            "artifact_hash": "sha256:test-distilgpt2-artifact",
            "model_id": "distilgpt2",
            "backend": "hf_transformers_cpu",
            "split_index": 1,
            "num_hidden_layers": 2,
            "hidden_size": 2,
            "vocab_size": 128,
            "read_only": True,
            "metadata_only": True,
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "crowdtensor.state_store.inspect_real_llm_artifact",
            return_value=artifact,
        ) as inspect_artifact:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10, backlog=0, admin_token="secret")
            admin_inference = endpoint_for(app, "/admin/inference-sessions", "POST")
            request_type = request_model(admin_inference)

            session = admin_inference(
                request_type(
                    request_count=1,
                    workload_type="real_llm_sharded_infer",
                    backend="cpu",
                    runtime="python-cli",
                    hf_model_id="distilgpt2",
                ),
                x_crowdtensor_admin_token="secret",
            )

            self.assertEqual(inspect_artifact.call_args.kwargs["model_id"], "distilgpt2")
            self.assertEqual(session["model_id"], "distilgpt2")

    def test_admin_session_stream_returns_safe_progress_events(self) -> None:
        artifact = {
            "schema": "real_llm_artifact_v1",
            "artifact_hash": "sha256:test-real-llm-stream-artifact",
            "model_id": "sshleifer/tiny-gpt2",
            "backend": "hf_transformers_cuda",
            "split_index": 1,
            "num_hidden_layers": 2,
            "hidden_size": 2,
            "vocab_size": 128,
            "read_only": True,
            "metadata_only": True,
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "crowdtensor.state_store.inspect_real_llm_artifact",
            return_value=artifact,
        ):
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10, backlog=0, admin_token="secret")
            admin_inference = endpoint_for(app, "/admin/inference-sessions", "POST")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            session_stream = endpoint_for(app, "/admin/session-stream", "GET")
            request_type = request_model(admin_inference)

            session = admin_inference(
                request_type(
                    request_count=1,
                    workload_type="real_llm_sharded_infer",
                    backend="cuda",
                    runtime="python-cli",
                    max_new_tokens=1,
                ),
                x_crowdtensor_admin_token="secret",
            )
            stage0_claim = claim_task(
                request_model(claim_task)(
                    miner_id="stream-stage0",
                    capabilities={
                        "runtime": "python-cli",
                        "backend": "cuda",
                        "protocol_version": "runtime_contract_v1",
                        "supported_workloads": ["real_llm_sharded_infer"],
                        "real_llm_runtime": {"adapter_kind": "hf_transformers_cuda"},
                        "real_llm_sharded_stage_role": "stage0",
                        "real_llm_sharded_stage_capabilities": [REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY],
                    },
                )
            )
            activation = {
                "schema_version": "real_llm_activation_v1",
                "session_id": session["session_id"],
                "request_id": "req-1",
                "prompt_hash": stage0_claim["workload_spec"]["requests"][0]["prompt_hash"],
                "model_id": "sshleifer/tiny-gpt2",
                "artifact_hash": artifact["artifact_hash"],
                "split_index": 1,
                "generation_step": 0,
                "max_new_tokens": 1,
                "generated_token_ids": [],
                "generated_text": "",
                "input_ids": [1, 2],
                "position_ids": [0, 1],
                "hidden_shape": [1, 2, 2],
                "hidden_state": [[[0.1, 0.2], [0.3, 0.4]]],
            }
            activation_hash = "sha256:" + hashlib.sha256(
                json.dumps({
                    "artifact_hash": activation["artifact_hash"],
                    "hidden_shape": activation["hidden_shape"],
                    "hidden_state": activation["hidden_state"],
                    "input_ids": activation["input_ids"],
                    "model_id": activation["model_id"],
                    "position_ids": activation["position_ids"],
                    "request_id": activation["request_id"],
                    "schema_version": activation["schema_version"],
                    "session_id": activation["session_id"],
                    "split_index": activation["split_index"],
                }, separators=(",", ":"), sort_keys=True).encode("utf-8")
            ).hexdigest()
            activation["activation_hash"] = activation_hash
            result_task(
                stage0_claim["task_id"],
                request_model(result_task)(
                    lease_token=stage0_claim["lease_token"],
                    attempt=stage0_claim["attempt"],
                    sharded_inference_result={
                        "schema_version": "real_llm_sharded_infer_v1",
                        "type": "real_llm_sharded_infer",
                        "session_id": session["session_id"],
                        "stage_id": 0,
                        "stage_count": 2,
                        "model_id": "sshleifer/tiny-gpt2",
                        "backend": "hf_transformers_cuda",
                        "partition_mode": "full",
                        "artifact_schema": "real_llm_artifact_v1",
                        "artifact_hash": artifact["artifact_hash"],
                        "split_index": 1,
                        "max_new_tokens": 1,
                        "generation_step": 0,
                        "request_count": 1,
                        "activation_count": 1,
                        "activation_bytes": 10,
                        "activation_hashes": [activation_hash],
                        "activation_transport_ready": True,
                        "activation_results": [activation],
                        "real_llm_artifact_ready": True,
                    },
                ),
            )
            stage1_claim = claim_task(
                request_model(claim_task)(
                    miner_id="stream-stage1",
                    capabilities={
                        "runtime": "python-cli",
                        "backend": "cuda",
                        "protocol_version": "runtime_contract_v1",
                        "supported_workloads": ["real_llm_sharded_infer"],
                        "real_llm_runtime": {"adapter_kind": "hf_transformers_cuda"},
                        "real_llm_sharded_stage_role": "stage1",
                        "real_llm_sharded_stage_capabilities": [REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY],
                    },
                )
            )
            generated_hash = "sha256:" + "a" * 64
            output_hash = "sha256:" + hashlib.sha256(
                json.dumps({
                    "activation_hash": activation_hash,
                    "artifact_hash": artifact["artifact_hash"],
                    "baseline_match": True,
                    "baseline_next_token_id": 42,
                    "model_id": "sshleifer/tiny-gpt2",
                    "next_token_id": 42,
                    "request_id": "req-1",
                }, separators=(",", ":"), sort_keys=True).encode("utf-8")
            ).hexdigest()
            inference_result = {
                "request_id": "req-1",
                "prompt_hash": stage1_claim["workload_spec"]["requests"][0]["prompt_hash"],
                "model_id": "sshleifer/tiny-gpt2",
                "artifact_hash": artifact["artifact_hash"],
                "activation_hash": activation_hash,
                "generation_step": 0,
                "max_new_tokens": 1,
                "next_token_id": 42,
                "baseline_next_token_id": 42,
                "generated_token_ids": [42],
                "generated_token_count": 1,
                "generated_text": " raw streamed token",
                "generated_text_hash": generated_hash,
                "baseline_match": True,
                "output_hash": output_hash,
            }
            result_task(
                stage1_claim["task_id"],
                request_model(result_task)(
                    lease_token=stage1_claim["lease_token"],
                    attempt=stage1_claim["attempt"],
                    sharded_inference_result={
                        "schema_version": "real_llm_sharded_infer_v1",
                        "type": "real_llm_sharded_infer",
                        "session_id": session["session_id"],
                        "stage_id": 1,
                        "stage_count": 2,
                        "model_id": "sshleifer/tiny-gpt2",
                        "backend": "hf_transformers_cuda",
                        "partition_mode": "full",
                        "artifact_schema": "real_llm_artifact_v1",
                        "artifact_hash": artifact["artifact_hash"],
                        "split_index": 1,
                        "max_new_tokens": 1,
                        "generation_step": 0,
                        "request_count": 1,
                        "activation_count": 1,
                        "activation_hashes": [activation_hash],
                        "activation_transport_ready": True,
                        "inference_result": inference_result,
                        "inference_results": [inference_result],
                        "baseline_match": True,
                        "decoded_tokens_match": True,
                        "generated_token_ids": [42],
                        "generated_token_count": 1,
                        "generated_text": " raw streamed token",
                        "generated_text_hash": generated_hash,
                        "real_llm_artifact_ready": True,
                    },
                ),
            )

            with self.assertRaises(HTTPException) as bad_token:
                session_stream(session_id=session["session_id"], x_crowdtensor_admin_token="bad")
            self.assertEqual(bad_token.exception.status_code, 403)

            payload = session_stream(
                session_id=session["session_id"],
                max_new_tokens=1,
                x_crowdtensor_admin_token="secret",
            )
            self.assertEqual(payload["schema"], "admin_session_stream_v1")
            self.assertEqual(payload["event_count"], 1)
            self.assertTrue(payload["stream_progress_complete"])
            self.assertIn("session_stream_progress_complete", payload["diagnosis_codes"])
            event = payload["events"][0]
            self.assertEqual(event["schema"], "session_stream_event_v1")
            self.assertEqual(event["generated_token_count"], 1)
            self.assertEqual(event["generated_text_hash"], generated_hash)
            public_text = json.dumps(payload, sort_keys=True)
            self.assertNotIn("raw streamed token", public_text)
            self.assertNotIn('"generated_token_ids":', public_text)

    def test_sign_compressed_result_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, lease_seconds=5, inner_steps=10, delta_format="sign_compressed")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            state = endpoint_for(app, "/state", "GET")
            claim = claim_task(
                request_model(claim_task)(
                    miner_id="api-compressed-miner",
                    capabilities={"supported_delta_formats": ["sign_compressed"]},
                )
            )
            self.assertEqual(claim["optimizer_spec"]["delta_format"], "sign_compressed")
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="api-compressed-miner",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )
            result = result_task(
                claim["task_id"],
                request_model(result_task)(
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    compressed_delta=compress_sign_delta(inner_result["local_delta"]),
                    metrics={"transport": "sign_compressed"},
                ),
            )

            self.assertTrue(result["accepted"])
            self.assertEqual(result["optimizer"]["delta_format"], "sign_compressed")
            self.assertEqual(state()["accepted_results"], 1)

    def test_sign_compressed_error_feedback_result_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                delta_format=DELTA_FORMAT_SIGN_COMPRESSED_EF,
            )
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            state = endpoint_for(app, "/state", "GET")
            claim = claim_task(
                request_model(claim_task)(
                    miner_id="api-compressed-ef-miner",
                    capabilities={"supported_delta_formats": [DELTA_FORMAT_SIGN_COMPRESSED_EF]},
                )
            )
            self.assertEqual(claim["optimizer_spec"]["delta_format"], DELTA_FORMAT_SIGN_COMPRESSED_EF)
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="api-compressed-ef-miner",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )
            compressed, _residual = compress_sign_delta_with_error_feedback(
                inner_result["local_delta"],
                residual=[0.01, -0.02, 0.03],
            )
            result = result_task(
                claim["task_id"],
                request_model(result_task)(
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    compressed_delta=compressed,
                    metrics={"transport": DELTA_FORMAT_SIGN_COMPRESSED_EF},
                ),
            )

            self.assertTrue(result["accepted"])
            self.assertEqual(result["optimizer"]["delta_format"], DELTA_FORMAT_SIGN_COMPRESSED_EF)
            self.assertTrue(result["optimizer"]["error_feedback"])
            self.assertEqual(state()["accepted_results"], 1)

    def test_nesterov_outer_optimizer_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                outer_optimizer=OPTIMIZER_DILOCO_NESTEROV,
            )
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            state = endpoint_for(app, "/state", "GET")

            claim = claim_task(request_model(claim_task)(miner_id="api-nesterov-miner"))
            self.assertEqual(claim["optimizer_spec"]["optimizer_type"], OPTIMIZER_DILOCO_NESTEROV)
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="api-nesterov-miner",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )
            result = result_task(
                claim["task_id"],
                request_model(result_task)(
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    local_delta=inner_result["local_delta"],
                    metrics=inner_result,
                ),
            )

            self.assertEqual(result["optimizer"]["optimizer_type"], OPTIMIZER_DILOCO_NESTEROV)
            self.assertIn("outer_update_norm", result["optimizer"])
            self.assertEqual(state()["model"]["outer_optimizer_type"], OPTIMIZER_DILOCO_NESTEROV)

    def test_create_app_accepts_task_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {"runtime": "python-cli", "backend": "cpu", "count": 2},
                    {"runtime": "browser", "backend": "js-worker", "count": 1},
                ],
            )
            state = endpoint_for(app, "/state", "GET")

            summary = state()

            self.assertEqual(summary["task_counts"]["queued"], 3)
            self.assertEqual(summary["task_lanes"][0]["runtime"], "python-cli")
            self.assertEqual(
                summary["task_counts_by_requirement"]["browser/js-worker/runtime_contract_v1"]["queued"],
                1,
            )

    def test_parse_task_lane(self) -> None:
        lane = parse_task_lane("browser:js-worker:3")

        self.assertEqual(lane["runtime"], "browser")
        self.assertEqual(lane["backend"], "js-worker")
        self.assertEqual(lane["count"], 3)
        self.assertEqual(lane["protocol_version"], "runtime_contract_v1")
        self.assertEqual(lane["workload_type"], "diloco_train")

        probe_lane = parse_task_lane("browser:js-worker:2:browser_probe")
        self.assertEqual(probe_lane["workload_type"], "browser_probe")
        self.assertEqual(probe_lane["count"], 2)

        lora_lane = parse_task_lane("python-cli:cpu:1:cpu_lora_mock")
        self.assertEqual(lora_lane["workload_type"], "cpu_lora_mock")
        self.assertEqual(lora_lane["runtime"], "python-cli")

        lm_lane = parse_task_lane("python-cli:cpu:1:micro_transformer_lm")
        self.assertEqual(lm_lane["workload_type"], "micro_transformer_lm")
        self.assertEqual(lm_lane["runtime"], "python-cli")

        bundle_lane = parse_task_lane("python-cli:cpu:1:model_bundle_lm")
        self.assertEqual(bundle_lane["workload_type"], "model_bundle_lm")
        self.assertEqual(bundle_lane["runtime"], "python-cli")

    def test_browser_probe_result_does_not_update_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "browser",
                        "backend": "js-worker",
                        "count": 1,
                        "workload_type": "browser_probe",
                    },
                ],
            )
            state = endpoint_for(app, "/state", "GET")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")

            claim = claim_task(
                request_model(claim_task)(
                    miner_id="probe-api-miner",
                    capabilities={
                        "runtime": "browser",
                        "backend": "js-worker",
                        "protocol_version": "runtime_contract_v1",
                        "supported_workloads": ["browser_probe"],
                    },
                )
            )
            result = result_task(
                claim["task_id"],
                request_model(result_task)(
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    probe_result={
                        "verified": True,
                        "ops": 4096,
                        "elapsed_ms": 2.0,
                        "hash": "feedbeef",
                    },
                ),
            )
            summary = state()

            self.assertTrue(result["accepted"])
            self.assertFalse(result["model_updated"])
            self.assertEqual(summary["accepted_results"], 1)
            self.assertEqual(summary["model_updates"], 0)
            self.assertEqual(summary["model"]["global_step"], 0)

    def test_cpu_lora_mock_result_updates_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=20,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "cpu_lora_mock",
                    },
                ],
            )
            state = endpoint_for(app, "/state", "GET")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            claim = claim_task(
                request_model(claim_task)(
                    miner_id="lora-api-miner",
                    capabilities={
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "protocol_version": "runtime_contract_v1",
                        "supported_workloads": ["cpu_lora_mock"],
                    },
                )
            )
            from crowdtensor.lora_mock import run_lora_inner_loop
            inner_result = run_lora_inner_loop(claim["workload_spec"], inner_steps=claim["inner_steps"])

            result = result_task(
                claim["task_id"],
                request_model(result_task)(
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    adapter_delta=inner_result["adapter_delta"],
                    metrics=inner_result,
                ),
            )
            summary = state()

            self.assertTrue(result["adapter_updated"])
            self.assertFalse(result["model_updated"])
            self.assertEqual(summary["adapter_updates"], 1)
            self.assertEqual(summary["model_updates"], 0)
            self.assertEqual(summary["model"]["adapter_step"], 1)

    def test_micro_transformer_result_updates_nested_lm_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=4,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "micro_transformer_lm",
                    },
                ],
            )
            state = endpoint_for(app, "/state", "GET")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            claim = claim_task(
                request_model(claim_task)(
                    miner_id="lm-api-miner",
                    capabilities={
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "protocol_version": "runtime_contract_v1",
                        "supported_workloads": ["micro_transformer_lm"],
                    },
                )
            )
            from crowdtensor.micro_transformer import run_micro_transformer_inner_loop
            inner_result = run_micro_transformer_inner_loop(claim["workload_spec"], inner_steps=claim["inner_steps"])

            result = result_task(
                claim["task_id"],
                request_model(result_task)(
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    local_delta=inner_result["local_delta"],
                    metrics=inner_result,
                ),
            )
            summary = state()

            self.assertTrue(result["micro_transformer_updated"])
            self.assertFalse(result["model_updated"])
            self.assertEqual(result["workload_type"], "micro_transformer_lm")
            self.assertEqual(summary["model"]["global_step"], 0)
            self.assertEqual(summary["model"]["micro_transformer"]["version"], 1)
            self.assertEqual(summary["micro_transformer_updates"], 1)

    def test_model_bundle_result_updates_nested_bundle_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                state_dir=tmp,
                lease_seconds=5,
                inner_steps=6,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "model_bundle_lm",
                    },
                ],
            )
            state = endpoint_for(app, "/state", "GET")
            claim_task = endpoint_for(app, "/tasks/claim", "POST")
            result_task = endpoint_for(app, "/tasks/{task_id}/result", "POST")
            claim = claim_task(
                request_model(claim_task)(
                    miner_id="bundle-api-miner",
                    capabilities={
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "protocol_version": "runtime_contract_v1",
                        "supported_workloads": ["model_bundle_lm"],
                    },
                )
            )
            inner_result = run_model_bundle_inner_loop(
                claim["workload_spec"],
                inner_steps=claim["inner_steps"],
            )

            result = result_task(
                claim["task_id"],
                request_model(result_task)(
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    bundle_delta=inner_result["bundle_delta"],
                    metrics=inner_result,
                ),
            )
            summary = state()

            self.assertTrue(result["model_bundle_updated"])
            self.assertFalse(result["model_updated"])
            self.assertEqual(result["workload_type"], "model_bundle_lm")
            self.assertEqual(summary["model"]["global_step"], 0)
            self.assertEqual(summary["model"]["model_bundle"]["version"], 1)
            self.assertEqual(summary["model_bundle_updates"], 1)
            self.assertNotIn("bundle_delta", json.dumps(summary["tasks"], sort_keys=True))


if __name__ == "__main__":
    unittest.main()
