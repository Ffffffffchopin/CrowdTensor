"""Single ordinary-user workflow for the Community Maturity RC."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .community_protocol import negotiate_protocol
from .community_security import security_contract_report
from .heterogeneous_training_beta import HeterogeneousTrainingBetaController
from .heterogeneous_training_manifest import write_training_manifest
from .model_adapter import adapter_registry_report, get_model_adapter, stable_hash
from .version import COMMUNITY_PROTOCOL_VERSION, public_version


WORKSPACE_SCHEMA = "crowdtensor_community_workspace_v1"
ACTION_SCHEMA = "crowdtensor_community_action_v1"
PLAN_SCHEMA = "crowdtensor_community_plan_v1"
class CommunityWorkflowError(RuntimeError):
    """Public-safe workflow error with a stable process exit code."""

    def __init__(self, reason: str, *, exit_code: int = 3) -> None:
        super().__init__(reason)
        self.reason = reason
        self.exit_code = int(exit_code)


def _write_json(path: Path, value: Any, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + secrets.token_hex(4) + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CommunityWorkflowError("community_workspace_state_invalid") from exc
    if not isinstance(value, dict):
        raise CommunityWorkflowError("community_workspace_state_invalid")
    return value


def _workspace_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.name.encode("utf-8")).hexdigest()


class CommunityWorkflow:
    """Idempotent lifecycle over the existing heterogeneous training runtime."""

    def __init__(self, workspace: str | Path) -> None:
        self.root = Path(workspace).expanduser().resolve()
        self.state_root = self.root / ".crowdtensor"
        self.public_root = self.root / "artifacts"
        self.private_root = self.state_root / "private"
        self.config_path = self.state_root / "community.json"
        self.private_path = self.private_root / "runtime.json"
        self.job_dir = self.state_root / "training-job"
        self.config = self._load_config()

    @classmethod
    def initialize(
        cls,
        workspace: str | Path,
        *,
        adapter_id: str = "qwen2_lora_v1",
        model_id: str = "",
        revision: str = "",
        accelerators: list[str] | None = None,
        target_steps: int = 100,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        root = Path(workspace).expanduser().resolve()
        state_root = root / ".crowdtensor"
        config_path = state_root / "community.json"
        if config_path.is_file() and not force:
            workflow = cls(root)
            return workflow._action(
                "init",
                ok=True,
                idempotent_replay=True,
                next_command="crowdtensor community validate <workspace>",
            )
        adapter = get_model_adapter(adapter_id)
        selected = sorted({str(item).lower() for item in (accelerators or ["cpu", "cuda"])})
        if not selected or not set(selected).issubset({"cpu", "cuda", "jax_tpu"}):
            raise CommunityWorkflowError("community_accelerators_invalid", exit_code=2)
        if "cpu" not in selected or "cuda" not in selected:
            raise CommunityWorkflowError("community_cpu_cuda_required", exit_code=2)
        steps = int(target_steps)
        if steps < 1 or steps > 10_000:
            raise CommunityWorkflowError("community_target_steps_invalid", exit_code=2)
        run_id = "community-" + secrets.token_hex(12)
        config = {
            "schema": WORKSPACE_SCHEMA,
            "run_id": run_id,
            "protocol_version": COMMUNITY_PROTOCOL_VERSION,
            "model_adapter_id": adapter.adapter_id,
            "model_id": model_id or adapter.default_model_id,
            "model_revision": revision or adapter.default_revision,
            "accelerators": selected,
            "target_steps": steps,
            "stage_count": int(adapter.recommended_stage_count),
            "checkpoint": {"backend": "local", "retention_steps": 2},
            "network": {"coordinator_host": "127.0.0.1", "coordinator_port": 8791, "tls_proxy_required_for_remote": True},
            "model_config": adapter.canonical_config(),
            "public_artifact_safe": True,
        }
        config["content_hash"] = stable_hash(config)
        if dry_run:
            return {
                "schema": ACTION_SCHEMA,
                "ok": True,
                "action": "init",
                "dry_run": True,
                "run_id": run_id,
                "configuration_hash": config["content_hash"],
                "workspace_path_public": False,
                "next_command": "crowdtensor community init <workspace>",
                "public_artifact_safe": True,
            }
        root.mkdir(parents=True, exist_ok=True)
        state_root.mkdir(parents=True, exist_ok=True)
        private = state_root / "private"
        private.mkdir(parents=True, exist_ok=True)
        private.chmod(0o700)
        _write_json(config_path, config, mode=0o644)
        _write_json(
            private / "runtime.json",
            {
                "schema": "crowdtensor_community_private_runtime_v1",
                "run_id": run_id,
                "coordinator_pid": 0,
                "created_at": time.time(),
                "public_artifact": False,
            },
            mode=0o600,
        )
        workflow = cls(root)
        report = workflow._action(
            "init", ok=True, idempotent_replay=False,
            next_command="crowdtensor community validate <workspace>",
        )
        workflow._write_public("init", report)
        return report

    def _load_config(self) -> dict[str, Any]:
        value = _read_json(self.config_path)
        supplied = str(value.get("content_hash") or "")
        source = dict(value)
        source.pop("content_hash", None)
        if value.get("schema") != WORKSPACE_SCHEMA or supplied != stable_hash(source):
            raise CommunityWorkflowError("community_workspace_config_invalid")
        return value

    def _action(self, action: str, *, ok: bool, next_command: str, **fields: Any) -> dict[str, Any]:
        value = {
            "schema": ACTION_SCHEMA,
            "ok": bool(ok),
            "action": action,
            "run_id": self.config["run_id"],
            "workspace_name_hash": _workspace_hash(self.root),
            **fields,
            "next_command": next_command,
            "next_command_redacts_credentials": True,
            "workspace_path_public": False,
            "credential_values_public": False,
            "private_urls_public": False,
            "public_artifact_safe": True,
        }
        value["content_hash"] = stable_hash(value)
        return value

    def _write_public(self, name: str, value: dict[str, Any]) -> None:
        _write_json(self.public_root / f"community_{name}.json", value, mode=0o644)

    def validate(self) -> dict[str, Any]:
        errors: list[str] = []
        try:
            adapter = get_model_adapter(str(self.config["model_adapter_id"]))
            canonical = adapter.validate_config(dict(self.config["model_config"]))
            adapter.partition(canonical, stage_count=int(self.config["stage_count"]))
        except (ValueError, TypeError):
            errors.append("community_model_adapter_validation_failed")
        protocol = negotiate_protocol(
            str(self.config.get("protocol_version") or ""),
            peer_capabilities=["atomic_checkpoint", "peft_lora", "signed_task"],
            required_capabilities=["atomic_checkpoint", "peft_lora", "signed_task"],
        )
        if not protocol["accepted"]:
            errors.extend(protocol["rejection_reasons"])
        accelerators = set(self.config.get("accelerators") or [])
        if not {"cpu", "cuda"}.issubset(accelerators):
            errors.append("community_cpu_cuda_required")
        report = self._action(
            "validate",
            ok=not errors,
            validation_errors=sorted(set(errors)),
            protocol_compatibility=protocol,
            model_adapter_id=str(self.config["model_adapter_id"]),
            model_config_valid=not any("model_adapter" in item for item in errors),
            next_command="crowdtensor community plan <workspace>",
        )
        self._write_public("validation", report)
        return report

    def plan(self) -> dict[str, Any]:
        validation = self.validate()
        if not validation["ok"]:
            raise CommunityWorkflowError("community_validation_failed", exit_code=2)
        adapter = get_model_adapter(str(self.config["model_adapter_id"]))
        config = adapter.validate_config(dict(self.config["model_config"]))
        stages = adapter.partition(config, stage_count=int(self.config["stage_count"]))
        estimates = [adapter.estimate_resources(config, stage) for stage in stages]
        report = {
            "schema": PLAN_SCHEMA,
            "ok": True,
            "run_id": self.config["run_id"],
            "protocol_version": self.config["protocol_version"],
            "model_adapter": adapter.descriptor(),
            "stage_specs": [stage.public_dict() for stage in stages],
            "resource_estimates": estimates,
            "accelerators": list(self.config["accelerators"]),
            "target_steps": int(self.config["target_steps"]),
            "logical_multi_node_supported": True,
            "physical_multi_machine_verified": False,
            "next_command": "crowdtensor community coordinator up <workspace>",
            "workspace_path_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        self._write_public("plan", report)
        return report

    def _ensure_training_job(self, *, dry_run: bool) -> HeterogeneousTrainingBetaController | None:
        if HeterogeneousTrainingBetaController.is_job(self.job_dir):
            return HeterogeneousTrainingBetaController(self.job_dir)
        if dry_run:
            return None
        adapter = get_model_adapter(str(self.config["model_adapter_id"]))
        if adapter.adapter_id != "qwen2_lora_v1":
            raise CommunityWorkflowError("community_scheduler_adapter_not_supported")
        manifest = adapter.production_manifest(
            target_steps=int(self.config["target_steps"]),
            accelerators=list(self.config["accelerators"]),
        )
        manifest_path = self.private_root / "training_manifest.json"
        write_training_manifest(manifest_path, manifest)
        return HeterogeneousTrainingBetaController.create(
            self.job_dir,
            manifest_path=manifest_path,
            enable_jax_tpu="jax_tpu" in self.config["accelerators"],
        )

    def coordinator_up(self, *, dry_run: bool = False, run: bool = False) -> dict[str, Any]:
        controller = self._ensure_training_job(dry_run=dry_run)
        report = self._action(
            "coordinator_up",
            ok=True,
            dry_run=bool(dry_run),
            coordinator_started=False,
            coordinator_prepared=True,
            bind_host=str(self.config["network"]["coordinator_host"]),
            bind_port=int(self.config["network"]["coordinator_port"]),
            tls_proxy_required_for_remote=True,
            next_command="crowdtensor community miner join <workspace>",
        )
        self._write_public("coordinator_up", report)
        if run and not dry_run:
            if controller is None:
                raise CommunityWorkflowError("community_coordinator_job_missing")
            from .heterogeneous_training_beta import create_heterogeneous_training_beta_app
            from .community_api import CommunitySecurityContext, create_community_app
            from .community_security import TLSProxyPolicy
            import uvicorn

            credentials = controller.credentials()
            app = create_heterogeneous_training_beta_app(
                controller,
                owner_token=str(credentials["owner_token"]),
                miner_token=str(credentials["miner_token"]),
            )
            app = create_community_app(
                self,
                context=CommunitySecurityContext(
                    issuer=str(self.config["run_id"]),
                    tls_policy=TLSProxyPolicy(require_https=False),
                ),
                base_app=app,
            )
            uvicorn.run(
                app,
                host=str(self.config["network"]["coordinator_host"]),
                port=int(self.config["network"]["coordinator_port"]),
            )
        return report

    def miner_join(self, *, dry_run: bool = False, run: bool = False, device_policy: str = "auto") -> dict[str, Any]:
        controller = self._ensure_training_job(dry_run=dry_run)
        command = [
            "crowdtensor-miner", "join", "--training", "--invite", "<private-invite>",
            "--device-policy", str(device_policy),
        ]
        report = self._action(
            "miner_join",
            ok=True,
            dry_run=bool(dry_run),
            miner_started=False,
            device_policy=str(device_policy),
            command_template=" ".join(command),
            invite_path_public=False,
            next_command="crowdtensor community train <workspace>",
        )
        self._write_public("miner_join", report)
        if run and not dry_run:
            if controller is None:
                raise CommunityWorkflowError("community_coordinator_job_missing")
            invite = self.private_root / "miner.invite.json"
            controller.write_miner_invite(
                invite,
                coordinator_url=(
                    f"http://{self.config['network']['coordinator_host']}:"
                    f"{self.config['network']['coordinator_port']}"
                ),
            )
            subprocess.run(
                [
                    "crowdtensor-miner", "join", "--training", "--invite", str(invite),
                    "--device-policy", str(device_policy),
                ],
                check=True,
            )
        return report

    def train(self, *, dry_run: bool = False) -> dict[str, Any]:
        controller = self._ensure_training_job(dry_run=dry_run)
        if dry_run:
            report = self._action(
                "train", ok=True, dry_run=True, transition_applied=False,
                next_command="crowdtensor community status <workspace>",
            )
        else:
            assert controller is not None
            result = controller.resume()
            report = self._action(
                "train", ok=True, dry_run=False,
                transition_applied=bool(result.get("resume_transition_applied")),
                runtime_state=str(result.get("overall_state") or ""),
                next_command="crowdtensor community status <workspace>",
            )
        self._write_public("train", report)
        return report

    def status(self) -> dict[str, Any]:
        if not HeterogeneousTrainingBetaController.is_job(self.job_dir):
            result = self._action(
                "status", ok=True, runtime_state="initialized", committed_step=0,
                target_steps=int(self.config["target_steps"]),
                next_command="crowdtensor community coordinator up <workspace>",
            )
        else:
            status = HeterogeneousTrainingBetaController(self.job_dir).status()
            result = self._action(
                "status", ok=True, runtime_state=status["overall_state"],
                committed_step=int(status["committed_step"]),
                target_steps=int(status["target_steps"]),
                online_miner_count=int(status["online_miner_count"]),
                blockers=list(status["blockers"]),
                next_command=(
                    "crowdtensor community export <workspace>"
                    if status["overall_state"] == "completed"
                    else "crowdtensor community status <workspace>"
                ),
            )
        self._write_public("status", result)
        return result

    def _control(self, action: str, *, dry_run: bool, reason: str = "owner_requested") -> dict[str, Any]:
        if not HeterogeneousTrainingBetaController.is_job(self.job_dir):
            if not dry_run:
                raise CommunityWorkflowError("community_training_job_not_created")
            transition = False
            state = "initialized"
        else:
            controller = HeterogeneousTrainingBetaController(self.job_dir)
            if dry_run:
                transition = False
                state = controller.status()["overall_state"]
            else:
                method = {
                    "pause": controller.pause,
                    "resume": controller.resume,
                    "rebalance": lambda: controller.rebalance(reason=reason),
                    "stop": controller.cancel,
                }[action]
                value = method()
                transition = bool(
                    value.get(f"{action}_transition_applied")
                    or value.get("cancel_transition_applied")
                )
                state = str(value.get("overall_state") or "")
        report = self._action(
            action, ok=True, dry_run=bool(dry_run), transition_applied=transition,
            runtime_state=state, reason=reason if action == "rebalance" else "",
            next_command=(
                "crowdtensor community cleanup <workspace>"
                if action == "stop" else "crowdtensor community status <workspace>"
            ),
        )
        self._write_public(action, report)
        return report

    def pause(self, *, dry_run: bool = False) -> dict[str, Any]:
        return self._control("pause", dry_run=dry_run)

    def resume(self, *, dry_run: bool = False) -> dict[str, Any]:
        return self._control("resume", dry_run=dry_run)

    def rebalance(self, *, reason: str = "owner_requested", dry_run: bool = False) -> dict[str, Any]:
        return self._control("rebalance", dry_run=dry_run, reason=reason)

    def stop(self, *, dry_run: bool = False) -> dict[str, Any]:
        return self._control("stop", dry_run=dry_run)

    def export(self, *, output_dir: str | Path | None = None, dry_run: bool = False) -> dict[str, Any]:
        if not dry_run and not HeterogeneousTrainingBetaController.is_job(self.job_dir):
            raise CommunityWorkflowError("community_training_job_not_created")
        if dry_run:
            exported: dict[str, Any] = {"ok": True, "dry_run": True}
        else:
            destination = Path(output_dir).expanduser().resolve() if output_dir else self.root / "adapter"
            exported = HeterogeneousTrainingBetaController(self.job_dir).export(destination)
        report = self._action(
            "export", ok=exported.get("ok") is True, dry_run=bool(dry_run),
            adapter_reload_required=True,
            tensor_values_public=False,
            next_command="crowdtensor community stop <workspace>",
        )
        self._write_public("export", report)
        return report

    def cleanup(self, *, dry_run: bool = False) -> dict[str, Any]:
        cleaned = True
        if HeterogeneousTrainingBetaController.is_job(self.job_dir) and not dry_run:
            cleaned = HeterogeneousTrainingBetaController(self.job_dir).cleanup().get("ok") is True
        report = self._action(
            "cleanup", ok=cleaned, dry_run=bool(dry_run),
            live_resources_left_running=False,
            evidence_preserved=True,
            next_command="crowdtensor community status <workspace>",
        )
        self._write_public("cleanup", report)
        return report

    def contract(self) -> dict[str, Any]:
        report = {
            "schema": "crowdtensor_community_contract_v1",
            "versions": public_version(),
            "workflow_actions": [
                "init", "validate", "plan", "coordinator up", "miner join", "train",
                "status", "pause", "resume", "rebalance", "export", "stop", "cleanup",
            ],
            "idempotent_actions": ["init", "validate", "plan", "train", "status", "pause", "resume", "stop", "cleanup"],
            "dry_run_supported": True,
            "exit_codes": {"success": 0, "validation": 2, "state": 3, "protocol": 4, "runtime": 5},
            "model_adapter_registry": adapter_registry_report(),
            "security": security_contract_report(),
            "physical_multi_machine_verified": False,
            "kaggle_logical_multi_node_supported": True,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        self._write_public("contract", report)
        return report
