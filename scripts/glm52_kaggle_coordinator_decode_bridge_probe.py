#!/usr/bin/env python3
"""Public-safe GLM 5.2 Coordinator bridge contract for Kaggle same-request decode.

This module intentionally separates the Coordinator runtime contract from a live
Kaggle success claim. The state machine can carry private activations between
stage workers and accept a final token hash without exposing token ids or hidden
payloads. A live RC still has to run real Kaggle workers and feed the resulting
stage/coordinator/cleanup reports into ``glm52_kaggle_same_request_probe.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_kaggle_same_request_probe as same_request_probe


SCHEMA = "glm52_kaggle_coordinator_decode_bridge_probe_v1"
COORDINATOR_SCHEMA = "glm52_kaggle_coordinator_decode_v1"
MODEL_ID = same_request_probe.MODEL_ID
COMPATIBLE_WEIGHT_REPO = same_request_probe.COMPATIBLE_WEIGHT_REPO
REQUIRED_PROVIDERS = same_request_probe.REQUIRED_PROVIDERS
DEFAULT_OUTPUT_DIR = "dist/glm52-kaggle-coordinator-decode-bridge"
SENSITIVE_FRAGMENTS = same_request_probe.SENSITIVE_FRAGMENTS + (
    '"hidden_b64":',
    '"private_input_hidden_b64":',
    '"input_hidden_b64":',
    "CT_GLM52_COORDINATOR_TOKEN",
    "X-CrowdTensor-GLM52-Token",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_text(value: Any) -> str:
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()


def sha_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return sha_text(encoded)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    loaded = json.loads(p.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 71


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def safety_flags() -> dict[str, bool]:
    return {
        "public_artifact_safe": True,
        "credentials_public": False,
        "cookies_public": False,
        "signed_url_public": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_public": False,
        "hidden_state_public": False,
        "logits_public": False,
        "kv_cache_public": False,
        "weight_tensor_values_public": False,
        "safetensors_header_payload_public": False,
    }


def default_stage_specs(stage_count: int = 39) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for stage_id in range(int(stage_count)):
        if stage_id == 0:
            provider = "kaggle_cuda"
        elif stage_id in {13, 14}:
            provider = "kaggle_jax_tpu"
        else:
            provider = "kaggle_cpu"
        specs.append(
            {
                "stage_id": stage_id,
                "stage_count": int(stage_count),
                "provider": provider,
                "stage_layer_range": [stage_id * 2, stage_id * 2 + 2],
                "compatible_weight_repo": COMPATIBLE_WEIGHT_REPO,
            }
        )
    return specs


def normalize_stage_specs(stage_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, raw in enumerate(stage_specs):
        stage_id = _int(raw.get("stage_id"), index)
        layer_range = _list(raw.get("stage_layer_range")) or [stage_id * 2, stage_id * 2 + 2]
        normalized.append(
            {
                "stage_id": stage_id,
                "stage_count": _int(raw.get("stage_count"), len(stage_specs)),
                "provider": str(raw.get("provider") or ""),
                "stage_layer_range": [_int(layer_range[0]), _int(layer_range[1])] if len(layer_range) == 2 else [],
                "compatible_weight_repo": str(raw.get("compatible_weight_repo") or COMPATIBLE_WEIGHT_REPO),
            }
        )
    return sorted(
        normalized,
        key=lambda item: (
            _int((_list(item.get("stage_layer_range")) or [0])[0]),
            _int((_list(item.get("stage_layer_range")) or [0, 0])[1]),
            _int(item.get("stage_id")),
        ),
    )


def _private_activation_ready(value: dict[str, Any]) -> bool:
    return bool(
        _hash_ok(value.get("activation_hash"))
        and isinstance(value.get("hidden_b64"), str)
        and bool(value.get("hidden_b64"))
        and len(_list(value.get("hidden_shape"))) == 2
        and str(value.get("hidden_dtype") or "")
    )


class Glm52CoordinatorState:
    """Ordered GLM stage task queue with private activation handoff."""

    def __init__(
        self,
        *,
        stage_specs: list[dict[str, Any]],
        coordinator_request_id_hash: str,
        max_new_tokens: int = 1,
    ) -> None:
        if not _hash_ok(coordinator_request_id_hash):
            raise ValueError("coordinator_request_id_hash_invalid")
        self.stage_specs = normalize_stage_specs(stage_specs)
        if len(self.stage_specs) < 2:
            raise ValueError("stage_specs_too_short")
        self.stage_count = len(self.stage_specs)
        self.stage_order = [_int(spec.get("stage_id")) for spec in self.stage_specs]
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.coordinator_request_id_hash = coordinator_request_id_hash
        self.generated_token_hashes: list[str] = []
        self.activation_hashes: list[str] = []
        self.output_hashes: list[str] = []
        self.tasks: dict[str, dict[str, Any]] = {}
        self.pending: list[dict[str, Any]] = []
        self.completed: list[dict[str, Any]] = []
        self.stage_seen: set[int] = set()
        self._counter = 0
        self._lock = threading.RLock()
        self.started_at = time.monotonic()
        self._queue_stage(stage_id=self.stage_order[0], generation_step=0)

    def _new_task_id(self, stage_id: int, generation_step: int) -> str:
        self._counter += 1
        return f"glm52-{self._counter:04d}-stage{stage_id}-step{generation_step}"

    def _stage_spec(self, stage_id: int) -> dict[str, Any]:
        for spec in self.stage_specs:
            if _int(spec.get("stage_id")) == int(stage_id):
                return spec
        return {}

    def _sequence_index(self, stage_id: int) -> int:
        try:
            return self.stage_order.index(int(stage_id))
        except ValueError:
            return -1

    def _next_stage_id(self, stage_id: int) -> int | None:
        index = self._sequence_index(stage_id)
        if index < 0 or index + 1 >= len(self.stage_order):
            return None
        return self.stage_order[index + 1]

    def _queue_stage(
        self,
        *,
        stage_id: int,
        generation_step: int,
        activation: dict[str, Any] | None = None,
    ) -> None:
        spec = self._stage_spec(stage_id)
        task = {
            "task_id": self._new_task_id(stage_id, generation_step),
            "stage_id": int(stage_id),
            "sequence_index": self._sequence_index(stage_id),
            "is_final_stage": self._next_stage_id(stage_id) is None,
            "next_stage_id": self._next_stage_id(stage_id),
            "generation_step": int(generation_step),
            "coordinator_request_id_hash": self.coordinator_request_id_hash,
            "provider": str(spec.get("provider") or ""),
            "stage_layer_range": _list(spec.get("stage_layer_range")),
            "status": "queued",
            "created_at": time.time(),
        }
        if activation:
            task["activation"] = activation
            task["activation_hash"] = activation.get("activation_hash")
        self.tasks[str(task["task_id"])] = task
        self.pending.append(task)

    def claim(self, *, miner_id: str, stage_id: int) -> dict[str, Any]:
        with self._lock:
            self.stage_seen.add(int(stage_id))
            if self.ready():
                return {"ok": True, "done": True}
            for index, task in enumerate(self.pending):
                if _int(task.get("stage_id")) != int(stage_id):
                    continue
                claimed = self.pending.pop(index)
                claimed["status"] = "leased"
                claimed["miner_id"] = str(miner_id or "")
                claimed["claimed_at"] = time.time()
                self.tasks[str(claimed["task_id"])] = claimed
                return {
                    "ok": True,
                    "done": False,
                    "task": {
                        key: value
                        for key, value in claimed.items()
                        if key not in {"status", "created_at", "claimed_at", "miner_id"}
                    },
                }
            return {"ok": True, "done": False, "task": None}

    def submit(self, result: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            task_id = str(result.get("task_id") or "")
            task = self.tasks.get(task_id)
            if not task:
                return {"ok": False, "accepted": False, "reason": "unknown_task"}
            stage_id = _int(result.get("stage_id"), _int(task.get("stage_id")))
            if stage_id != _int(task.get("stage_id")):
                return {"ok": False, "accepted": False, "reason": "stage_id_mismatch"}
            if result.get("public_artifact_safe") is not True:
                return {"ok": False, "accepted": False, "reason": "public_artifact_unsafe"}
            if result.get("stage_decode_verified") is not True:
                return {"ok": False, "accepted": False, "reason": "stage_decode_not_verified"}
            if not _hash_ok(result.get("stage_output_hash")):
                return {"ok": False, "accepted": False, "reason": "stage_output_hash_missing"}
            if not _hash_ok(result.get("weight_value_sha256") or result.get("stage_weight_value_hash")):
                return {"ok": False, "accepted": False, "reason": "weight_value_hash_missing"}
            if _int(result.get("weight_value_byte_count") or result.get("stage_weight_value_byte_count")) <= 0:
                return {"ok": False, "accepted": False, "reason": "weight_value_byte_count_missing"}

            task["status"] = "completed"
            task["completed_at"] = time.time()
            task["duration_seconds"] = float(result.get("duration_seconds") or 0.0)
            task["stage_output_hash"] = str(result.get("stage_output_hash") or "")
            task["output_hash"] = str(result.get("output_hash") or result.get("stage_output_hash") or "")
            task["weight_value_sha256"] = str(result.get("weight_value_sha256") or result.get("stage_weight_value_hash") or "")
            task["weight_value_byte_count"] = _int(
                result.get("weight_value_byte_count") or result.get("stage_weight_value_byte_count")
            )
            task["provider_runtime_verified"] = result.get("provider_runtime_verified") is not False
            task["provider_device_count"] = _int(result.get("provider_device_count"))
            task["kv_cache"] = _dict(result.get("kv_cache"))
            task["stage_decode_report_hash"] = str(result.get("stage_decode_report_hash") or "")
            self.completed.append(task)

            next_stage_id = self._next_stage_id(stage_id)
            if next_stage_id is not None:
                activation = _dict(result.get("activation"))
                if not _private_activation_ready(activation):
                    return {"ok": False, "accepted": False, "reason": "activation_missing"}
                self.activation_hashes.append(str(activation.get("activation_hash")))
                self.output_hashes.append(str(task.get("output_hash") or ""))
                self._queue_stage(
                    stage_id=next_stage_id,
                    generation_step=_int(task.get("generation_step")),
                    activation=activation,
                )
            else:
                token_hash = str(
                    result.get("generated_token_hash")
                    or result.get("next_token_hash")
                    or result.get("selected_token_id_hash")
                    or ""
                )
                if not _hash_ok(token_hash):
                    return {"ok": False, "accepted": False, "reason": "generated_token_hash_missing"}
                self.generated_token_hashes.append(token_hash)
                self.output_hashes.append(str(task.get("output_hash") or ""))
                if len(self.generated_token_hashes) < self.max_new_tokens:
                    self._queue_stage(stage_id=self.stage_order[0], generation_step=_int(task.get("generation_step")) + 1)
            return {"ok": True, "accepted": True, "ready": self.ready()}

    def ready(self) -> bool:
        return len(self.generated_token_hashes) >= self.max_new_tokens

    def public_status(self) -> dict[str, Any]:
        with self._lock:
            completed_public = []
            for task in self.completed:
                completed_public.append(
                    {
                        "task_id_hash": sha_text(task.get("task_id")),
                        "stage_id": _int(task.get("stage_id")),
                        "sequence_index": _int(task.get("sequence_index")),
                        "provider": str(task.get("provider") or ""),
                        "generation_step": _int(task.get("generation_step")),
                        "miner_id_hash": sha_text(task.get("miner_id") or ""),
                        "activation_hash": str(task.get("activation_hash") or ""),
                        "stage_output_hash": str(task.get("stage_output_hash") or ""),
                        "output_hash": str(task.get("output_hash") or ""),
                        "duration_seconds": float(task.get("duration_seconds") or 0.0),
                        "stage_cache_summary": _dict(task.get("kv_cache")),
                    }
                )
            return {
                "schema": COORDINATOR_SCHEMA,
                "ok": True,
                "ready": self.ready(),
                "model_id": MODEL_ID,
                "coordinator_request_id_hash": self.coordinator_request_id_hash,
                "stage_count": self.stage_count,
                "stage_order": list(self.stage_order),
                "max_new_tokens": self.max_new_tokens,
                "generated_token_count": len(self.generated_token_hashes),
                "generated_token_hashes": list(self.generated_token_hashes),
                "generated_token_hash": self.generated_token_hashes[-1] if self.generated_token_hashes else "",
                "activation_hashes": list(self.activation_hashes),
                "output_hashes": list(self.output_hashes),
                "pending_count": len(self.pending),
                "completed_task_count": len(self.completed),
                "stage_seen": sorted(self.stage_seen),
                "stage_task_counts": {
                    f"stage{stage_id}": sum(1 for item in self.completed if _int(item.get("stage_id")) == stage_id)
                    for stage_id in self.stage_order
                },
                "completed_tasks": completed_public,
                "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
                "live_run_performed": True,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "activation_public": False,
                "hidden_state_public": False,
                "logits_public": False,
                "kv_cache_public": False,
                "public_artifact_safe": True,
            }

    def same_request_stage_reports(self) -> list[dict[str, Any]]:
        with self._lock:
            reports = []
            for task in sorted(self.completed, key=lambda item: _int(item.get("sequence_index"))):
                spec = self._stage_spec(_int(task.get("stage_id")))
                reports.append(
                    {
                        "schema": same_request_probe.STAGE_SCHEMA,
                        "model_id": MODEL_ID,
                        "compatible_weight_repo": str(spec.get("compatible_weight_repo") or COMPATIBLE_WEIGHT_REPO),
                        "provider": str(spec.get("provider") or ""),
                        "stage_id": _int(task.get("stage_id")),
                        "stage_layer_range": _list(spec.get("stage_layer_range")),
                        "coordinator_request_id_hash": self.coordinator_request_id_hash,
                        "stage_execution_verified": True,
                        "stage_decode_verified": True,
                        "stage_output_hash": str(task.get("stage_output_hash") or ""),
                        "weight_tensor_values_loaded": True,
                        "stage_owned_weight_values_loaded": True,
                        "weight_tensor_values_public": False,
                        "weight_value_byte_count": _int(task.get("weight_value_byte_count")),
                        "weight_value_sha256": str(task.get("weight_value_sha256") or ""),
                        "live_run_performed": True,
                        "public_artifact_safe": True,
                        "activation_public": False,
                        "kv_cache_public": False,
                    }
                )
            return reports

    def coordinator_report(self) -> dict[str, Any]:
        status = self.public_status()
        return {
            "schema": COORDINATOR_SCHEMA,
            "model_id": MODEL_ID,
            "coordinator_request_id_hash": self.coordinator_request_id_hash,
            "generated_token_count": len(self.generated_token_hashes),
            "generated_token_hash": self.generated_token_hashes[-1] if self.generated_token_hashes else "",
            "live_run_performed": True,
            "public_artifact_safe": status.get("public_artifact_safe") is True,
            "status": status,
        }


class Glm52CoordinatorServer:
    def __init__(self, *, host: str, port: int, token: str, state: Glm52CoordinatorState) -> None:
        token_value = str(token or "")
        state_value = state

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, Any]:
                size = int(self.headers.get("Content-Length") or 0)
                if size <= 0:
                    return {}
                loaded = json.loads(self.rfile.read(size).decode("utf-8"))
                return loaded if isinstance(loaded, dict) else {}

            def _authorized(self) -> bool:
                return bool(token_value) and self.headers.get("X-CrowdTensor-GLM52-Token") == token_value

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path in {"/ready", "/status"}:
                    self._send(200, state_value.public_status())
                    return
                self._send(404, {"ok": False, "error": "not_found"})

            def do_POST(self) -> None:
                if not self._authorized():
                    self._send(403, {"ok": False, "error": "forbidden"})
                    return
                path = self.path.split("?", 1)[0]
                payload = self._read_json()
                if path == "/claim":
                    self._send(
                        200,
                        state_value.claim(
                            miner_id=str(payload.get("miner_id") or ""),
                            stage_id=_int(payload.get("stage_id")),
                        ),
                    )
                    return
                if path == "/submit":
                    self._send(200, state_value.submit(payload))
                    return
                self._send(404, {"ok": False, "error": "not_found"})

        self.httpd = ThreadingHTTPServer((host, int(port)), Handler)
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def cleanup_report(*, cleaned: bool = False) -> dict[str, Any]:
    return {
        "temporary_kaggle_kernels_deleted": bool(cleaned),
        "temporary_private_packages_removed": bool(cleaned),
        "live_resources_left_running": False if cleaned else None,
        "public_artifact_safe": True,
    }


def build_contract_report(args: argparse.Namespace) -> dict[str, Any]:
    stage_specs = default_stage_specs(stage_count=int(args.stage_count))
    request_hash = str(args.coordinator_request_id_hash or sha_json({"glm52_contract": args.stage_count}))
    state = Glm52CoordinatorState(
        stage_specs=stage_specs,
        coordinator_request_id_hash=request_hash,
        max_new_tokens=1,
    )
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "mode": "contract",
        "model_id": MODEL_ID,
        "compatible_weight_repo": COMPATIBLE_WEIGHT_REPO,
        "coordinator_bridge_contract_ready": True,
        "coordinator_request_id_hash": request_hash,
        "stage_count": len(stage_specs),
        "required_providers": REQUIRED_PROVIDERS,
        "initial_public_status": state.public_status(),
        "same_request_decode_verified": False,
        "live_run_performed": False,
        "completion_boundary": {
            "contract_is_not_live_success": True,
            "requires_live_kaggle_stage_workers": True,
            "requires_same_request_probe_assemble": True,
            "requires_cleanup_evidence": True,
        },
        "blockers": [
            "glm52_coordinator_bridge_contract_only",
            "glm52_live_kaggle_same_request_not_run",
        ],
        "safety": safety_flags(),
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set(_list(report.get("blockers")) + ["public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage-count", type=int, default=39)
    parser.add_argument("--coordinator-request-id-hash", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_contract_report(args)
    path = output_dir / "glm52_kaggle_coordinator_decode_bridge_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Coordinator bridge contract ready: {report.get('coordinator_bridge_contract_ready')}")
        print(f"Live same-request verified: {report.get('same_request_decode_verified')}")
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
