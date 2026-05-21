from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
import tempfile
import time
import unittest

from fastapi import HTTPException

from coordinator import create_app, load_miner_token_registry, parse_task_lane
from crowdtensor.auth import hash_token
from crowdtensor.diloco import run_inner_loop
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
            self.assertIn('crowdtensord_model_updates_total{target="micro_transformer"} 0', text)
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
                x_crowdtensor_admin_token="secret",
            )
            self.assertEqual(ledger["status"], "accepted")
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
            self.assertNotIn("result_response", public_text)
            self.assertNotIn("local_delta", public_text)

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


if __name__ == "__main__":
    unittest.main()
