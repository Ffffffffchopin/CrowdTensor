"""Accelerate/FSDP2 launch adapter for stable rank-group training windows."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from crowdtensor.core.contracts import ContractError, TrainingMode, TrainingProject, stable_hash
from crowdtensor.core.execution import (
    ProviderSnapshot,
    ResourceAvailability,
    StableShardedLaunchSpec,
    TrainingExecutionPlan,
)
from crowdtensor.core.plugins import BackendCapabilities
from crowdtensor.core.workspace import load_project


def accelerate_executable() -> str:
    """Resolve the console script belonging to the active Python environment."""

    sibling = Path(sys.executable).with_name("accelerate")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return str(shutil.which("accelerate") or "")


def probe_accelerate_fsdp2_runtime() -> dict[str, Any]:
    """Explicitly verify the installed upstream exposes FSDP2 launch support."""

    if importlib.util.find_spec("accelerate") is None or not accelerate_executable():
        return {"available": False, "version": ""}
    try:
        accelerate_version = importlib.metadata.version("accelerate")
        torch_version = importlib.metadata.version("torch")
        utilities = importlib.import_module("accelerate.utils")
        plugin = getattr(utilities, "FullyShardedDataParallelPlugin")
        if "fsdp_version" not in inspect.signature(plugin).parameters:
            return {"available": False, "version": ""}
        fsdp = importlib.import_module("torch.distributed.fsdp")
        if not callable(getattr(fsdp, "fully_shard", None)):
            return {"available": False, "version": ""}
    except (ImportError, AttributeError, TypeError, importlib.metadata.PackageNotFoundError):
        return {"available": False, "version": ""}
    return {
        "available": True,
        "version": f"accelerate-{accelerate_version}+torch-{torch_version}",
    }


def _atomic_write(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def accelerate_config_text(spec: StableShardedLaunchSpec) -> str:
    """Render the small documented Accelerate FSDP2 configuration subset."""

    wrap_policy = "TRANSFORMER_BASED_WRAP" if spec.transformer_layer_class else "NO_WRAP"
    lines = [
        "compute_environment: LOCAL_MACHINE",
        "debug: false",
        "distributed_type: FSDP",
        "downcast_bf16: 'no'",
        "enable_cpu_affinity: false",
        "fsdp_config:",
        "  fsdp_activation_checkpointing: false",
        f"  fsdp_auto_wrap_policy: {wrap_policy}",
        "  fsdp_cpu_offload: false",
        "  fsdp_cpu_ram_efficient_loading: true",
        "  fsdp_reshard_after_forward: true",
        "  fsdp_state_dict_type: SHARDED_STATE_DICT",
        "  fsdp_sync_module_states: true",
    ]
    if spec.transformer_layer_class:
        lines.append(
            "  fsdp_transformer_layer_cls_to_wrap: " + spec.transformer_layer_class
        )
    lines.extend(
        [
            "  fsdp_version: 2",
            "machine_rank: 0",
            "main_training_function: main",
            f"mixed_precision: {spec.mixed_precision}",
            f"num_machines: {spec.num_machines}",
            f"num_processes: {spec.num_processes}",
            "rdzv_backend: static",
            "same_network: false",
            "use_cpu: false",
            "",
        ]
    )
    return "\n".join(lines)


class AccelerateFSDP2Backend:
    """Build and materialize a stable FSDP2 rank-group launch contract."""

    backend_id = "accelerate_fsdp2"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            modes=frozenset({TrainingMode.STABLE_SHARDED}),
            checkpoint_formats=("torch_distributed_checkpoint", "safetensors"),
            supports_full_parameters=True,
            supports_peft=True,
        )

    def validate_project(self, project: TrainingProject) -> tuple[str, ...]:
        blockers = []
        if project.mode is not TrainingMode.STABLE_SHARDED:
            blockers.append("stable_sharded_mode_required")
        if project.training_backend != self.backend_id:
            blockers.append("training_backend_mismatch")
        return tuple(sorted(blockers))

    def build_plan(
        self,
        project: TrainingProject,
        providers: Sequence[ProviderSnapshot],
        **options: Any,
    ) -> TrainingExecutionPlan:
        allow_cpu_validation = options.get("allow_cpu_validation") is True
        groups: dict[tuple[str, str], list[ProviderSnapshot]] = {}
        for resource in providers:
            if (
                resource.device_type
                in ({"cuda", "cpu"} if allow_cpu_validation else {"cuda"})
                and resource.availability is ResourceAvailability.STABLE_WINDOW
                and "distributed_collective" in resource.capabilities
                and resource.stable_group_id
            ):
                groups.setdefault(
                    (resource.stable_group_id, resource.device_type), []
                ).append(resource)
        selected: tuple[ProviderSnapshot, ...] = ()
        if groups:
            ranked = sorted(
                groups.items(),
                key=lambda item: (
                    -sum(resource.device_count for resource in item[1]),
                    -sum(resource.free_memory_bytes for resource in item[1]),
                    item[0][1] != "cuda",
                    item[0],
                ),
            )
            selected = tuple(ranked[0][1])
        blockers = list(self.validate_project(project))
        if sum(item.device_count for item in selected) < 2:
            blockers.append(
                "stable_rank_group_insufficient"
                if allow_cpu_validation
                else "stable_cuda_rank_group_insufficient"
            )
        probe = dict(options.get("runtime_probe") or probe_accelerate_fsdp2_runtime())
        required_capabilities = ["distributed_collective", "stable_sharded"]
        if not selected or selected[0].device_type == "cuda":
            required_capabilities.insert(0, "cuda")
        return TrainingExecutionPlan(
            project_hash=project.content_hash,
            mode=project.mode,
            backend_id=self.backend_id,
            selected_resources=selected,
            required_capabilities=tuple(required_capabilities),
            restart_semantics="restart_rank_group_from_committed_checkpoint",
            runtime_name="accelerate_fsdp2",
            runtime_available=probe.get("available") is True,
            runtime_version=str(probe.get("version") or ""),
            blockers=tuple(blockers),
        )

    def build_launch_spec(
        self,
        plan: TrainingExecutionPlan,
        **options: Any,
    ) -> StableShardedLaunchSpec:
        if plan.mode is not TrainingMode.STABLE_SHARDED or plan.backend_id != self.backend_id:
            raise ContractError("accelerate_launch_plan_backend_mismatch")
        device_types = {item.device_type for item in plan.selected_resources}
        cpu_validation = (
            device_types == {"cpu"} and options.get("allow_cpu_validation") is True
        )
        if device_types != {"cuda"} and not cpu_validation:
            raise ContractError("accelerate_launch_cuda_resources_required")
        machines = len({item.machine_id_hash for item in plan.selected_resources})
        processes = sum(item.device_count for item in plan.selected_resources)
        trainer = str(options.get("trainer_entrypoint") or "train.py")
        blockers = list(plan.blockers)
        if not options.get("trainer_entrypoint"):
            blockers.append("trainer_entrypoint_required")
        if options.get("trainer_contract_verified") is not True:
            blockers.append("trainer_contract_unverified")
        environment_names = (
            "CROWDTENSOR_BASE_CHECKPOINT_PATH",
            "CROWDTENSOR_BASE_PAYLOAD_PATH",
            "CROWDTENSOR_MACHINE_RANK",
            "CROWDTENSOR_MAIN_PROCESS_IP",
            "CROWDTENSOR_MAIN_PROCESS_PORT",
            "CROWDTENSOR_OUTPUT_CHECKPOINT_PATH",
            "CROWDTENSOR_TRAINER_RESULT_PATH",
            "CROWDTENSOR_WORK_UNIT_PATH",
        )
        max_restarts = int(options.get("max_restarts", 1))
        config_path = ".crowdtensor/state/accelerate-fsdp2.yaml"
        checkpoint_path = ".crowdtensor/checkpoints/stable-sharded"
        command = (
            "accelerate",
            "launch",
            "--config_file",
            config_path,
            "--num_machines",
            str(max(1, machines)),
            "--num_processes",
            str(max(1, processes)),
            "--machine_rank",
            "${CROWDTENSOR_MACHINE_RANK}",
            "--main_process_ip",
            "${CROWDTENSOR_MAIN_PROCESS_IP}",
            "--main_process_port",
            "${CROWDTENSOR_MAIN_PROCESS_PORT}",
            "--max_restarts",
            "0",
            trainer,
            "--crowdtensor-project",
            ".crowdtensor/project.json",
            "--crowdtensor-checkpoint-dir",
            checkpoint_path,
            "--crowdtensor-work-unit",
            "${CROWDTENSOR_WORK_UNIT_PATH}",
            "--crowdtensor-base-checkpoint",
            "${CROWDTENSOR_BASE_CHECKPOINT_PATH}",
            "--crowdtensor-base-payload",
            "${CROWDTENSOR_BASE_PAYLOAD_PATH}",
            "--crowdtensor-output-checkpoint",
            "${CROWDTENSOR_OUTPUT_CHECKPOINT_PATH}",
            "--crowdtensor-result",
            "${CROWDTENSOR_TRAINER_RESULT_PATH}",
        )
        return StableShardedLaunchSpec(
            plan_hash=plan.content_hash,
            project_hash=plan.project_hash,
            backend_id=self.backend_id,
            distributed_type="fsdp2",
            num_machines=max(1, machines),
            num_processes=max(1, processes),
            mixed_precision=str(options.get("mixed_precision") or "bf16"),
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            trainer_entrypoint=trainer,
            transformer_layer_class=options.get("transformer_layer_class"),
            command_template=command,
            environment_names=environment_names,
            max_restarts=max_restarts,
            blockers=tuple(blockers),
        )

    def materialize_launch(
        self,
        spec: StableShardedLaunchSpec,
        *,
        workspace: Path,
    ) -> dict[str, Any]:
        root = Path(workspace).expanduser().resolve()
        project = load_project(root)
        if project.content_hash != spec.project_hash:
            raise ContractError("accelerate_launch_workspace_project_mismatch")
        config = accelerate_config_text(spec).encode("utf-8")
        config_path = root / spec.config_path
        launch_path = root / ".crowdtensor/state/stable-sharded-launch.json"
        (root / spec.checkpoint_path).mkdir(parents=True, exist_ok=True)
        launch_payload = spec.to_dict()
        launch_bytes = (
            json.dumps(launch_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        _atomic_write(config_path, config)
        _atomic_write(launch_path, launch_bytes)
        report = {
            "schema": "crowdtensor_stable_sharded_materialization_v2",
            "project_hash": project.content_hash,
            "launch_spec_hash": spec.content_hash,
            "config_path": spec.config_path,
            "config_sha256": _sha256_bytes(config),
            "launch_path": ".crowdtensor/state/stable-sharded-launch.json",
            "launch_sha256": _sha256_bytes(launch_bytes),
            "execution_ready": spec.execution_ready,
            "command_executed": False,
            "checkpoint_restart_policy": "restart_rank_group_from_latest_committed_checkpoint",
            "controller_restart_limit": spec.max_restarts,
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        return report

    def run_session(self, workspace: str | Path, **options: Any) -> dict[str, Any]:
        """Execute bounded rank-group work through the v2 Session Controller."""

        from .stable_session import run_stable_sharded_session

        return run_stable_sharded_session(workspace, backend=self, **options)
