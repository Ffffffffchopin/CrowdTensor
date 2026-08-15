"""Training Architecture v2 bridge for the proven Volunteer PEFT path."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import fcntl
import re
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from crowdtensor.core.contracts import (
    ArtifactRef,
    CheckpointLineage,
    CheckpointRef,
    ContractError,
    ContributionReceipt,
    ReceiptOutcome,
    TrainingMode,
    TrainingProject,
    WorkUnit,
    stable_hash,
    validate_receipt_binding,
)
from crowdtensor.core.controller import SessionController, SessionControllerError
from crowdtensor.core.execution import ProviderSnapshot, TrainingExecutionPlan
from crowdtensor.core.plugins import BackendCapabilities
from crowdtensor.core.workspace import CONTROL_DIR, init_project_contract


def _identifier(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9._-]+", "-", str(value).lower()).strip("-._")
    if not normalized:
        normalized = fallback
    if len(normalized) == 1:
        normalized = fallback + "-" + normalized
    if normalized != str(value):
        suffix = stable_hash(str(value)).split(":", 1)[1][:10]
        normalized = f"{normalized[:116].rstrip('-._')}-{suffix}"
    return normalized[:128].rstrip("-._")


def _runtime_probe() -> dict[str, Any]:
    packages = ("torch", "transformers", "peft", "safetensors")
    versions: list[str] = []
    for package in packages:
        if importlib.util.find_spec(package) is None:
            return {"available": False, "version": ""}
        try:
            versions.append(f"{package}-{importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            return {"available": False, "version": ""}
    return {"available": True, "version": "+".join(versions)}


def project_from_campaign(campaign: Mapping[str, Any]) -> TrainingProject:
    """Translate an immutable Volunteer campaign into v2 training intent."""

    from crowdtensor.volunteer_training_protocol import validate_campaign_manifest

    canonical = validate_campaign_manifest(dict(campaign))
    model_source = canonical.get("model_source") or {}
    dataset_source = canonical.get("dataset_source") or {}
    model_id = str(model_source.get("model_id") or canonical["model_id"])
    dataset_id = str(dataset_source.get("dataset_id") or canonical["dataset_id"])
    model_revision = str(
        model_source.get("revision") or f"campaign-revision-{canonical['model_revision']}"
    )
    dataset_revision = str(
        dataset_source.get("revision")
        or f"campaign-revision-{canonical['dataset_revision']}"
    )
    optimizer = str(canonical["outer_optimizer"]["optimizer_type"])
    optimization_plugin = {
        "diloco_momentum": "diloco_v1",
        "local_sgd_mean": "local_sgd_v1",
    }[optimizer]
    return TrainingProject(
        project_id=_identifier(str(canonical["campaign_id"]), fallback="campaign"),
        mode=TrainingMode.ELASTIC_DELTA,
        model=ArtifactRef(
            model_id,
            model_revision,
            str(canonical.get("base_model_hash") or canonical["model_manifest_hash"]),
        ),
        dataset=ArtifactRef(
            dataset_id,
            dataset_revision,
            str(canonical["dataset_snapshot_hash"]),
        ),
        model_adapter=_identifier(
            str(canonical.get("model_adapter_id") or "legacy_lora_v1"),
            fallback="legacy-lora-v1",
        ),
        training_backend="volunteer_peft",
        target_steps=int(canonical["round_policy"]["target_rounds"]),
        optimization_plugins=("peft_lora_v1", optimization_plugin),
    )


def checkpoint_lineage_from_campaign(
    project: TrainingProject,
    campaign: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> CheckpointLineage:
    """Convert public Volunteer lineage without exposing campaign paths."""

    canonical_project = project_from_campaign(campaign)
    if canonical_project.content_hash != project.content_hash:
        raise ContractError("volunteer_bridge_project_campaign_mismatch")
    from crowdtensor.training_contract import sha256_json

    report = dict(lineage)
    supplied_hash = str(report.get("content_hash") or "")
    expected_hash = sha256_json(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    if (
        report.get("schema") != "crowdtensor_volunteer_checkpoint_lineage_v1"
        or supplied_hash != expected_hash
        or report.get("public_artifact_safe") is not True
        or report.get("ok") is not True
        or report.get("campaign_id") != campaign.get("campaign_id")
        or report.get("campaign_manifest_hash") != campaign.get("manifest_hash")
    ):
        raise ContractError("volunteer_bridge_lineage_invalid")
    initial_hash = str(report.get("initial_adapter_hash") or "")
    genesis = CheckpointRef(
        checkpoint_id="adapter-v0",
        project_hash=project.content_hash,
        step=0,
        generation=0,
        artifact=ArtifactRef(
            f"crowdtensor://campaign/{project.project_id}/adapter/v0",
            "adapter-v0",
            initial_hash,
        ),
        adapter_only=True,
    )
    checkpoints = [genesis]
    previous = genesis
    for raw in report.get("entries") or []:
        if raw.get("lineage_link_verified") is not True:
            raise ContractError("volunteer_bridge_lineage_link_invalid")
        before = int(raw["adapter_version_before"])
        after = int(raw["adapter_version_after"])
        if before != previous.step or after != before + 1:
            raise ContractError("volunteer_bridge_lineage_version_invalid")
        checkpoint = CheckpointRef(
            checkpoint_id=f"adapter-v{after}",
            project_hash=project.content_hash,
            step=after,
            generation=after,
            artifact=ArtifactRef(
                f"crowdtensor://campaign/{project.project_id}/adapter/v{after}",
                f"adapter-v{after}",
                str(raw["canonical_adapter_hash"]),
            ),
            parent_hash=previous.content_hash,
            created_by_work_id=None,
            adapter_only=True,
        )
        checkpoints.append(checkpoint)
        previous = checkpoint
    if previous.artifact.digest != report.get("canonical_adapter_hash"):
        raise ContractError("volunteer_bridge_lineage_head_mismatch")
    return CheckpointLineage(project.content_hash, tuple(checkpoints))


def work_from_claim(
    project: TrainingProject,
    campaign: Mapping[str, Any],
    claim: Mapping[str, Any],
    *,
    base_checkpoint: CheckpointRef,
) -> WorkUnit:
    """Strip lease secrets and translate one Volunteer claim into a v2 Work Unit."""

    from crowdtensor.volunteer_training_protocol import (
        validate_campaign_manifest,
        validate_work_unit,
    )

    canonical_campaign = validate_campaign_manifest(dict(campaign))
    canonical_work = validate_work_unit(dict(claim), campaign=canonical_campaign)
    if project_from_campaign(canonical_campaign).content_hash != project.content_hash:
        raise ContractError("volunteer_bridge_project_campaign_mismatch")
    if base_checkpoint.project_hash != project.content_hash:
        raise ContractError("volunteer_bridge_checkpoint_project_mismatch")
    if base_checkpoint.artifact.digest != canonical_work["base_adapter_hash"]:
        raise ContractError("volunteer_bridge_checkpoint_claim_mismatch")
    return WorkUnit(
        work_id=_identifier(str(canonical_work["work_id"]), fallback="work"),
        project_hash=project.content_hash,
        mode=TrainingMode.ELASTIC_DELTA,
        generation=int(canonical_work["lease_generation"]),
        backend="volunteer_peft",
        base_checkpoint_hash=base_checkpoint.content_hash,
        data_shard_hash=str(canonical_work["dataset_shard_hash"]),
        step_start=int(canonical_work["step_start"]),
        step_count=int(canonical_work["local_steps"]),
        required_capabilities=("peft_lora",),
    )


def receipt_from_submission(
    project: TrainingProject,
    work: WorkUnit,
    base_checkpoint: CheckpointRef,
    *,
    contributor_id_hash: str,
    delta_manifest: Mapping[str, Any],
    response: Mapping[str, Any],
    completed_at: str | float,
    output_checkpoint: CheckpointRef | None = None,
    recovery_replay: bool = False,
) -> ContributionReceipt:
    """Issue a v2 receipt after the v1 Coordinator has admitted the delta."""

    accepted = response.get("accepted") is True
    if response.get("idempotent_replay") is True and not recovery_replay:
        raise ContractError("volunteer_bridge_existing_receipt_required")
    aggregated = accepted and response.get("round_aggregated") is True
    if aggregated != (output_checkpoint is not None):
        raise ContractError("volunteer_bridge_receipt_checkpoint_state_mismatch")
    if isinstance(completed_at, (int, float)):
        timestamp = datetime.fromtimestamp(float(completed_at), UTC).isoformat()
    else:
        timestamp = str(completed_at)
    result_id = str(response.get("result_id") or delta_manifest.get("result_id") or "")
    if accepted and (
        not result_id or result_id != str(delta_manifest.get("result_id") or "")
    ):
        raise ContractError("volunteer_bridge_result_id_mismatch")
    metrics = []
    for name in ("loss_start", "loss_end"):
        if name in delta_manifest:
            metrics.append((name, float(delta_manifest[name])))
    for name in ("delta_norm_before_clip", "delta_norm_after_clip"):
        if name in response:
            metrics.append((name, float(response[name])))
    receipt = ContributionReceipt(
        receipt_id=_identifier(result_id, fallback="receipt"),
        project_hash=project.content_hash,
        work_id=work.work_id,
        work_generation=work.generation,
        contributor_id_hash=contributor_id_hash,
        base_checkpoint_hash=base_checkpoint.content_hash,
        submitted_artifact_hash=str(delta_manifest.get("delta_file_hash") or ""),
        outcome=ReceiptOutcome.ACCEPTED if accepted else ReceiptOutcome.REJECTED,
        completed_at=timestamp,
        steps=work.step_count if accepted else 0,
        samples=int(delta_manifest.get("samples_seen") or 0) if accepted else 0,
        tokens=int(delta_manifest.get("tokens_seen") or 0) if accepted else 0,
        checkpoint_committed=aggregated,
        output_checkpoint_hash=(output_checkpoint.content_hash if output_checkpoint else None),
        rejection_code=(None if accepted else str(response.get("code") or "submission_rejected")),
        metrics=tuple(metrics),
    )
    validate_receipt_binding(
        receipt,
        work=work,
        base_checkpoint=base_checkpoint,
        output_checkpoint=output_checkpoint,
    )
    return receipt


class VolunteerControllerTransport:
    """Mirror Volunteer claim/submit transitions into a v2 SessionController.

    Artifact transfer, heartbeat, numerical training, and submission remain on
    the wrapped Volunteer transport. Only public contracts enter v2 state.
    """

    def __init__(self, transport: Any, workspace: str | Path) -> None:
        required = (
            "campaign",
            "checkpoint_lineage",
            "status",
            "claim",
            "heartbeat",
            "download_artifact",
            "submit",
        )
        if any(not callable(getattr(transport, name, None)) for name in required):
            raise TypeError("volunteer_controller_transport_protocol_required")
        self.transport = transport
        self.campaign_document = transport.campaign()
        self.project = project_from_campaign(self.campaign_document)
        init_project_contract(workspace, self.project)
        self.controller = SessionController(workspace)
        remote_lineage = checkpoint_lineage_from_campaign(
            self.project,
            self.campaign_document,
            transport.checkpoint_lineage(),
        )
        if self.controller.state_path.is_file():
            local_lineage = self.controller.lineage()
            local_hashes = [item.content_hash for item in local_lineage.checkpoints]
            remote_hashes = [item.content_hash for item in remote_lineage.checkpoints]
            remote_extends_local = remote_hashes[: len(local_hashes)] == local_hashes
            pending_remote_commit = bool(
                self.controller.active_works()
                and len(remote_hashes) == len(local_hashes) + 1
            )
            if not remote_extends_local or len(remote_hashes) not in {
                len(local_hashes),
                len(local_hashes) + (1 if pending_remote_commit else 0),
            }:
                raise SessionControllerError(
                    "volunteer_bridge_remote_lineage_conflict"
                )
            self.controller.initialize(local_lineage)
        else:
            self.controller.initialize(remote_lineage)
        self.transaction_lock_path = (
            self.controller.workspace
            / CONTROL_DIR
            / "state/volunteer-bridge-transaction.lock"
        )

    @contextmanager
    def _transaction_guard(self):
        self.transaction_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction_lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def campaign(self) -> dict[str, Any]:
        return dict(self.campaign_document)

    def checkpoint_lineage(self) -> dict[str, Any]:
        return self.transport.checkpoint_lineage()

    def status(self) -> dict[str, Any]:
        return self.transport.status()

    def _base_for_adapter_hash(self, adapter_hash: str) -> CheckpointRef:
        matches = [
            checkpoint
            for checkpoint in self.controller.lineage().checkpoints
            if checkpoint.artifact.digest == str(adapter_hash)
        ]
        if len(matches) != 1:
            raise SessionControllerError(
                "volunteer_bridge_claim_checkpoint_missing"
            )
        return matches[0]

    def _record_claim_response(
        self, *, cell_id: str, response: Mapping[str, Any]
    ) -> dict[str, Any]:
        from crowdtensor.volunteer_training_protocol import hash_cell_id

        result = dict(response)
        raw_work = result.get("work_unit")
        if not isinstance(raw_work, dict):
            return result
        base = self._base_for_adapter_hash(
            str(raw_work.get("base_adapter_hash") or "")
        )
        work = work_from_claim(
            self.project,
            self.campaign_document,
            raw_work,
            base_checkpoint=base,
        )
        active = self.controller.active_lease(work.work_id)
        replace = bool(active is not None and active[0].content_hash != work.content_hash)
        session = self.controller.issue(
            work,
            contributor_id_hash=hash_cell_id(cell_id),
            replace_active=replace,
            lease_expires_at=float(raw_work["lease_expires_at"]),
        )
        return {**result, "v2_session": session}

    def record_claim_response(
        self, *, cell_id: str, response: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Mirror an already-authenticated Coordinator claim into v2 state."""

        with self._transaction_guard():
            return self._record_claim_response(cell_id=cell_id, response=response)

    def record_claim_operation(
        self,
        *,
        cell_id: str,
        operation: Callable[[], Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Serialize Coordinator claim acceptance and its v2 transition."""

        with self._transaction_guard():
            return self._record_claim_response(
                cell_id=cell_id, response=operation()
            )

    def claim(self, *, cell_id: str, capability: dict[str, Any]) -> dict[str, Any]:
        with self._transaction_guard():
            response = self.transport.claim(cell_id=cell_id, capability=capability)
            return self._record_claim_response(
                cell_id=cell_id, response=response
            )

    def heartbeat(self, *, cell_id: str, work: dict[str, Any]) -> dict[str, Any]:
        with self._transaction_guard():
            response = self.transport.heartbeat(cell_id=cell_id, work=work)
            return self.record_heartbeat_response(
                cell_id=cell_id,
                work_id=str(work.get("work_id") or ""),
                generation=int(work.get("lease_generation") or 0),
                response=response,
            )

    def record_heartbeat_response(
        self,
        *,
        cell_id: str,
        work_id: str,
        generation: int,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist a successful authenticated heartbeat's lease extension."""

        from crowdtensor.volunteer_training_protocol import hash_cell_id

        result = dict(response)
        if result.get("ok") is not True:
            return result
        expiry = result.get("lease_expires_at")
        if not isinstance(expiry, (int, float)) or isinstance(expiry, bool):
            raise SessionControllerError(
                "volunteer_bridge_heartbeat_expiry_missing"
            )
        session = self.controller.renew(
            _identifier(str(work_id), fallback="work"),
            int(generation),
            contributor_id_hash=hash_cell_id(cell_id),
            lease_expires_at=float(expiry),
        )
        return {**result, "v2_session": session}

    def record_heartbeat_operation(
        self,
        *,
        cell_id: str,
        work_id: str,
        generation: int,
        operation: Callable[[], Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Serialize Coordinator heartbeat acceptance and lease persistence."""

        with self._transaction_guard():
            return self.record_heartbeat_response(
                cell_id=cell_id,
                work_id=work_id,
                generation=generation,
                response=operation(),
            )

    def download_artifact(
        self, ref: dict[str, Any], destination: Path, *, max_bytes: int
    ) -> int:
        return int(
            self.transport.download_artifact(
                ref, destination, max_bytes=max_bytes
            )
        )

    def submit(
        self,
        *,
        cell_id: str,
        work: dict[str, Any],
        delta_manifest: dict[str, Any],
    ) -> dict[str, Any]:
        work_id = _identifier(str(work.get("work_id") or ""), fallback="work")
        generation = int(work.get("lease_generation") or 0)
        with self._transaction_guard():
            response = self.transport.submit(
                cell_id=cell_id,
                work=work,
                delta_manifest=delta_manifest,
            )
            return self.record_submission_response(
                cell_id=cell_id,
                work_id=work_id,
                generation=generation,
                delta_manifest=delta_manifest,
                response=response,
                raw_work=work,
            )

    def record_submission_response(
        self,
        *,
        cell_id: str,
        work_id: str,
        generation: int,
        delta_manifest: Mapping[str, Any],
        response: Mapping[str, Any],
        raw_work: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mirror an already-authenticated Coordinator submission into v2."""

        work_id = _identifier(str(work_id), fallback="work")
        generation = int(generation)
        terminal = self.controller.terminal(work_id, generation)
        active_lease = self.controller.active_lease(work_id, generation)
        if active_lease is None and terminal is None:
            raise SessionControllerError("session_controller_active_work_missing")
        if terminal is not None:
            translated, existing_receipt, base, existing_output = terminal
            if raw_work is not None:
                replay_work = work_from_claim(
                    self.project,
                    self.campaign_document,
                    raw_work,
                    base_checkpoint=base,
                )
                if replay_work.content_hash != translated.content_hash:
                    raise SessionControllerError(
                        "session_controller_terminal_work_mismatch"
                    )
        else:
            existing_receipt = None
            existing_output = None
            assert active_lease is not None
            active = active_lease[0]
            lineage = self.controller.lineage()
            base = next(
                (
                    checkpoint
                    for checkpoint in lineage.checkpoints
                    if checkpoint.content_hash == active.base_checkpoint_hash
                ),
                None,
            )
            if base is None:
                raise SessionControllerError(
                    "session_controller_base_checkpoint_missing"
                )
            if raw_work is None:
                translated = active
            else:
                translated = work_from_claim(
                    self.project,
                    self.campaign_document,
                    raw_work,
                    base_checkpoint=base,
                )
                if translated.content_hash != active.content_hash:
                    raise SessionControllerError(
                        "session_controller_active_work_mismatch"
                    )
        result = dict(response)
        if result.get("accepted") is not True:
            return result
        committed_during_submit = self.controller.terminal(work_id, generation)
        if committed_during_submit is not None:
            committed_work, committed_receipt, committed_base, committed_output = (
                committed_during_submit
            )
            if committed_work.content_hash != translated.content_hash:
                raise SessionControllerError(
                    "session_controller_terminal_work_mismatch"
                )
            terminal = committed_during_submit
            translated = committed_work
            existing_receipt = committed_receipt
            base = committed_base
            existing_output = committed_output
        output = None
        if result.get("round_aggregated") is True:
            remote_lineage = checkpoint_lineage_from_campaign(
                self.project,
                self.campaign_document,
                self.transport.checkpoint_lineage(),
            )
            local_lineage = self.controller.lineage()
            if terminal is not None:
                if (
                    existing_output is None
                    or remote_lineage.content_hash != local_lineage.content_hash
                ):
                    raise SessionControllerError(
                        "volunteer_bridge_aggregated_lineage_replay_invalid"
                    )
                output = existing_output
            elif (
                len(remote_lineage.checkpoints)
                != len(local_lineage.checkpoints) + 1
                or any(
                    left.content_hash != right.content_hash
                    for left, right in zip(
                        local_lineage.checkpoints,
                        remote_lineage.checkpoints,
                        strict=False,
                    )
                )
            ):
                raise SessionControllerError(
                    "volunteer_bridge_aggregated_lineage_advance_invalid"
                )
            else:
                output = remote_lineage.checkpoints[-1]
        elif existing_output is not None:
            raise SessionControllerError(
                "volunteer_bridge_aggregated_replay_metadata_missing"
            )
        completed_at = result.get("accepted_at")
        if not isinstance(completed_at, (str, int, float)):
            raise SessionControllerError("volunteer_bridge_accepted_at_missing")
        from crowdtensor.volunteer_training_protocol import hash_cell_id

        receipt = receipt_from_submission(
            self.project,
            translated,
            base,
            contributor_id_hash=hash_cell_id(cell_id),
            delta_manifest=delta_manifest,
            response=result,
            completed_at=completed_at,
            output_checkpoint=output,
            recovery_replay=True,
        )
        if existing_receipt is not None and receipt.content_hash != existing_receipt.content_hash:
            raise SessionControllerError("volunteer_bridge_receipt_replay_conflict")
        session = self.controller.commit(
            translated,
            receipt,
            base_checkpoint=base,
            output_checkpoint=output,
        )
        return {**result, "v2_session": session}

    def record_submission_operation(
        self,
        *,
        cell_id: str,
        work_id: str,
        generation: int,
        delta_manifest: Mapping[str, Any],
        operation: Callable[[], Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Serialize Coordinator admission/aggregation and the v2 commit."""

        with self._transaction_guard():
            return self.record_submission_response(
                cell_id=cell_id,
                work_id=work_id,
                generation=generation,
                delta_manifest=delta_manifest,
                response=operation(),
            )


class VolunteerPEFTBackend:
    """Adapter that delegates numerical work to VolunteerTrainingCell."""

    backend_id = "volunteer_peft"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            modes=frozenset({TrainingMode.ELASTIC_DELTA}),
            checkpoint_formats=("peft_safetensors",),
            supports_full_parameters=False,
            supports_peft=True,
        )

    def validate_project(self, project: TrainingProject) -> tuple[str, ...]:
        blockers = []
        if project.mode is not TrainingMode.ELASTIC_DELTA:
            blockers.append("elastic_delta_mode_required")
        if project.training_backend != self.backend_id:
            blockers.append("training_backend_mismatch")
        if "peft_lora_v1" not in project.optimization_plugins:
            blockers.append("peft_lora_plugin_required")
        return tuple(sorted(blockers))

    def build_plan(
        self,
        project: TrainingProject,
        providers: Sequence[ProviderSnapshot],
        **options: Any,
    ) -> TrainingExecutionPlan:
        selected = tuple(
            item for item in providers if "peft_lora" in item.capabilities
        )
        blockers = list(self.validate_project(project))
        if not selected:
            blockers.append("no_eligible_resource")
        probe = dict(options.get("runtime_probe") or _runtime_probe())
        return TrainingExecutionPlan(
            project_hash=project.content_hash,
            mode=project.mode,
            backend_id=self.backend_id,
            selected_resources=selected,
            required_capabilities=("peft_lora",),
            restart_semantics="reissue_work_from_committed_checkpoint",
            runtime_name="transformers_peft",
            runtime_available=probe.get("available") is True,
            runtime_version=str(probe.get("version") or ""),
            blockers=tuple(blockers),
        )

    def run_next_work_unit(self, worker_runtime: Any) -> dict[str, Any]:
        if not callable(getattr(worker_runtime, "join_once", None)):
            raise TypeError("volunteer_peft_cell_join_once_required")
        report = worker_runtime.join_once()
        if not isinstance(report, dict):
            raise RuntimeError("volunteer_peft_cell_report_invalid")
        return report

    def run_volunteer_cell_once(self, cell: Any) -> dict[str, Any]:
        return self.run_next_work_unit(cell)

    def bind_controller_transport(
        self, transport: Any, *, workspace: str | Path
    ) -> VolunteerControllerTransport:
        return VolunteerControllerTransport(transport, workspace)
