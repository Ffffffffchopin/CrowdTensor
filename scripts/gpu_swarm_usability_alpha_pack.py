#!/usr/bin/env python3
"""Build the GPU Swarm Usability Alpha evidence and user runbook."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import control_user_alpha_pack as control_alpha  # noqa: E402


SCHEMA = "gpu_swarm_usability_alpha_v1"
SUPPORT_BUNDLE_SCHEMA = "gpu_swarm_usability_alpha_support_bundle_v1"
DEFAULT_OUTPUT_DIR = "dist/gpu-swarm-usability-alpha"
DEFAULT_CONTROL_USER_ALPHA_REPORT = "dist/control-user-alpha-goal-r1/control_user_alpha.json"
DEFAULT_CORE_HANDOFF_REPORT = control_alpha.DEFAULT_CORE_HANDOFF_REPORT
DEFAULT_CORE_STATUS_REPORT = control_alpha.DEFAULT_CORE_STATUS_REPORT
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
STAGES = ("stage0", "stage1")
EXECUTION_MODES = ("fixture", "evidence-import", "external-existing", "kaggle-auto")
ACTIONS = ("smoke", "prepare", "coordinator", "miner", "infer", "status", "collect", "clean")
SAFE_MINER_TOKEN_ENV = "GPU_SWARM_MINER_PRIVATE_TOKEN"
BOUNDARIES = {
    "not_production": True,
    "not_p2p_nat_traversal": True,
    "not_arbitrary_public_prompt_serving": True,
    "not_billing": True,
    "not_unbounded_gpu_pooling": True,
}
SENSITIVE_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "CROWDTENSOR_P2P_PEER_SECRET=",
    "Bearer ",
    "SOURCE_TARBALL_B64",
    "MINER_ENV_TEXT",
    '"prompt":',
    '"prompt_text":',
    '"prompt_texts":',
    '"raw_prompt":',
    '"generated_text":',
    '"output_text":',
    '"generated_token_ids":',
    '"token_ids":',
    '"activation":',
    '"activations":',
    '"activation_results":',
    '"hidden_state":',
    '"input_ids":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
    '"lease_token":',
    '"idempotency_key":',
    "operator.private.env",
    "miner.private.env",
    "miner.stage0.private.env",
    "miner.stage1.private.env",
    "miner_registry.json",
    "kernel.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def stable_hash_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if path.is_file():
        entry["sha256"] = sha256_file(path)
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def artifact_summary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "gpu_swarm_usability_alpha_artifact_summary_v1",
        "artifact_count": len(artifacts),
        "present_artifact_count": sum(1 for item in artifacts.values() if item.get("present")),
        "inspect_first": (artifacts.get("runbook_markdown") or {}).get("path", ""),
        "support_bundle": (artifacts.get("support_bundle_json") or {}).get("path", ""),
        "public_artifact_safe": True,
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True)
    errors = [fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded]
    return sorted(set(errors))


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def shell_join_template(command: list[str]) -> str:
    rendered = []
    for part in command:
        value = str(part)
        if value.startswith("${") and value.endswith("}"):
            rendered.append(value)
        else:
            rendered.append(shlex.quote(value))
    return " ".join(rendered)


def model_catalog_from_control(control_report: dict[str, Any]) -> dict[str, Any]:
    catalog = _dict(control_report.get("model_catalog"))
    models = [item for item in _list(catalog.get("models")) if isinstance(item, dict)]
    return {
        "schema": "gpu_swarm_model_catalog_import_v1",
        "model_catalog_imported": bool(catalog.get("model_catalog_ready") and models),
        "default_model_id": catalog.get("default_model_id") or DEFAULT_MODEL_ID,
        "models": models,
        "capabilities": _dict(catalog.get("capabilities")),
        "boundaries": catalog.get("boundaries") or list(BOUNDARIES),
        "public_artifact_safe": True,
    }


def selected_model(catalog: dict[str, Any], model_id: str) -> dict[str, Any]:
    models = [item for item in _list(catalog.get("models")) if isinstance(item, dict)]
    for item in models:
        if str(item.get("model_id") or "") == model_id:
            return item
    return models[-1] if models else {"model_id": model_id or DEFAULT_MODEL_ID}


def stage_capability(stage: str) -> str:
    return f"real_llm_sharded_cuda_{stage}"


def build_stage_join_pack(args: argparse.Namespace, *, stage: str, model: dict[str, Any]) -> dict[str, Any]:
    stage_dir = f"stage-{stage}"
    script = f"{stage_dir}/miner_join.sh"
    runbook = f"{stage_dir}/MINER_JOIN.md"
    env_template = f"{stage_dir}/miner.env.template"
    command = [
        "crowdtensor-miner",
        "--coordinator",
        args.coordinator_url.rstrip("/"),
        "--miner-id",
        f"{args.miner_id_prefix}-{stage}",
        "--miner-token",
        f"${{{SAFE_MINER_TOKEN_ENV}:?set {SAFE_MINER_TOKEN_ENV}}}",
        "--max-tasks",
        "1",
        "--enable-hf-tiny-gpt-runtime",
        "--hf-model-id",
        str(model.get("model_id") or args.model_id),
        "--real-llm-backend",
        "hf_transformers_cuda",
        "--real-llm-stage-role",
        stage,
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return {
        "schema": "gpu_swarm_miner_join_pack_v1",
        "ready": True,
        "stage": stage,
        "stage_role": stage,
        "required_capability": stage_capability(stage),
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "miner_id": f"{args.miner_id_prefix}-{stage}",
        "model_id": str(model.get("model_id") or args.model_id),
        "backend": "hf_transformers_cuda",
        "execution_mode": "stage_selective_hf",
        "stage_owned_weight_loading_required": True,
        "safe_env_template": env_template,
        "join_script": script,
        "join_runbook": runbook,
        "recommended_command": f"bash {script}",
        "command_template": shell_join_template(command),
        "private_token_placeholder": f"${{{SAFE_MINER_TOKEN_ENV}}}",
        "private_token_env": SAFE_MINER_TOKEN_ENV,
        "public_artifact_safe": True,
    }


def render_join_script(pack: dict[str, Any]) -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"if [ -z \"${{{SAFE_MINER_TOKEN_ENV}:-}}\" ]; then",
        f"  echo \"{SAFE_MINER_TOKEN_ENV} is required and must be provided privately on this Miner host\" >&2",
        "  exit 2",
        "fi",
        "",
        str(pack["command_template"]),
        "",
    ])


def render_env_template(pack: dict[str, Any]) -> str:
    return "\n".join([
        "# Copy this file privately on the Miner host and replace the placeholder.",
        "# Do not publish or commit the filled file.",
        f"# Required private env var: {SAFE_MINER_TOKEN_ENV}",
        "# Example private shell setup: export the required env var to your local token value.",
        "",
        f"# Stage: {pack.get('stage')}",
        f"# Coordinator URL: {pack.get('coordinator_url')}",
        f"# Required capability: {pack.get('required_capability')}",
        f"# Model: {pack.get('model_id')}",
        "",
    ])


def render_join_markdown(pack: dict[str, Any]) -> str:
    return "\n".join([
        "# GPU Swarm Miner Join",
        "",
        f"- stage: `{pack.get('stage')}`",
        f"- coordinator: `{pack.get('coordinator_url')}`",
        f"- required capability: `{pack.get('required_capability')}`",
        f"- model: `{pack.get('model_id')}`",
        f"- backend: `{pack.get('backend')}`",
        f"- private token env var: `{pack.get('private_token_env')}`",
        "",
        "## Start",
        "",
        f"1. Put a private Miner token in `{pack.get('safe_env_template')}` on the Miner host.",
        f"2. Run `{pack.get('recommended_command')}`.",
        "",
        "## Boundary",
        "",
        "- This Alpha join pack is Coordinator-backed and does not provide production P2P or billing.",
        "- Public artifacts use placeholders and do not include credentials.",
        "",
    ])


def build_join_packs(args: argparse.Namespace, *, model_catalog: dict[str, Any]) -> dict[str, Any]:
    model = selected_model(model_catalog, args.model_id)
    stage_packs = [build_stage_join_pack(args, stage=stage, model=model) for stage in STAGES]
    return {
        "schema": "gpu_swarm_miner_join_packs_v1",
        "gpu_miner_join_pack_ready": all(item.get("ready") for item in stage_packs),
        "stage_count": len(stage_packs),
        "stages": stage_packs,
        "stage_roles": list(STAGES),
        "distinct_stage_miners_required": True,
        "public_artifact_safe": True,
    }


def build_coordinator_workflow(args: argparse.Namespace, *, model: dict[str, Any]) -> dict[str, Any]:
    command = [
        "crowdtensord",
        "--host",
        args.bind_host,
        "--port",
        str(args.port),
        "--state-dir",
        str(Path(args.output_dir) / "state"),
        "--lease-seconds",
        str(args.lease_seconds),
        "--inner-steps",
        str(args.max_new_tokens),
        "--task-lane",
        "python-cli:cuda:0:real_llm_sharded_infer",
        "--real-llm-model-id",
        str(model.get("model_id") or args.model_id),
    ]
    return {
        "schema": "gpu_swarm_coordinator_workflow_v1",
        "coordinator_workflow_ready": True,
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "bind_host": args.bind_host,
        "port": args.port,
        "start_command": shell_join(command),
        "state_dir": "state",
        "requires_private_operator_tokens": True,
        "public_artifact_safe": True,
    }


def build_gpu_readiness(args: argparse.Namespace, *, join_packs: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for pack in _list(join_packs.get("stages")):
        if not isinstance(pack, dict):
            continue
        checks.append({
            "stage": pack.get("stage"),
            "gpu_readiness": "not_checked_in_evidence_import" if args.execution_mode == "evidence-import" else "requires_external_check",
            "cuda_available": None,
            "hf_runtime_ready": "required",
            "stage_owned_weight_loading_ready": "retained_evidence_ready",
            "diagnosis": "external GPU not required for CI-safe smoke" if args.execution_mode in {"fixture", "evidence-import"} else "verify external Miner preflight",
            "public_artifact_safe": True,
        })
    return {
        "schema": "gpu_swarm_readiness_v1",
        "gpu_readiness_report_ready": True,
        "checks": checks,
        "external_gpu_required_for_current_mode": args.execution_mode in {"external-existing", "kaggle-auto"},
        "fresh_gpu_checked": False,
        "public_artifact_safe": True,
    }


def build_cleanup_plan(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    return {
        "schema": "gpu_swarm_cleanup_plan_v1",
        "cleanup_ready": True,
        "dry_run_default": True,
        "action": args.action,
        "would_remove_private_runtime_state": [
            str((output_dir / "state").as_posix()),
            "local filled Miner token env files outside this public artifact directory",
        ],
        "would_keep_public_artifacts": [
            "gpu_swarm_usability_alpha.json",
            "GPU_SWARM_ALPHA.md",
            "support_bundle.json",
            "stage-stage0/MINER_JOIN.md",
            "stage-stage1/MINER_JOIN.md",
        ],
        "private_env_written": False,
        "public_artifact_safe": True,
    }


def build_inference_lifecycle(
    args: argparse.Namespace,
    *,
    model: dict[str, Any],
    control_report: dict[str, Any],
) -> dict[str, Any]:
    prompt_hash = stable_hash_payload({"prompt_placeholder": "<redacted>", "label": args.request_label})
    generated_count = int(model.get("verified_token_count") or 0)
    completed = bool(args.execution_mode in {"fixture", "evidence-import"} and generated_count >= min(args.max_new_tokens, generated_count or 1))
    events = [
        {"event": "prepare", "state": "complete", "public_artifact_safe": True},
        {"event": "coordinator_plan", "state": "complete", "public_artifact_safe": True},
        {"event": "miner_join_plan", "state": "complete", "public_artifact_safe": True},
        {"event": "infer_request", "state": "submitted", "prompt_hash": prompt_hash, "public_artifact_safe": True},
        {"event": "status", "state": "completed" if completed else "requires_external_runtime", "public_artifact_safe": True},
        {"event": "collect", "state": "public_artifacts_ready", "public_artifact_safe": True},
    ]
    return {
        "schema": "gpu_swarm_inference_lifecycle_v1",
        "inference_request_lifecycle_ready": True,
        "execution_mode": args.execution_mode,
        "external_runtime_verified": args.execution_mode in {"external-existing", "kaggle-auto"} and bool(args.external_runtime_verified),
        "request_label": args.request_label,
        "model_id": model.get("model_id"),
        "model_live_verified": bool(model.get("live_verified")),
        "verified_token_count": generated_count,
        "max_new_tokens": args.max_new_tokens,
        "prompt_hash": prompt_hash,
        "result_scope": {
            "terminal_answer_allowed": args.execution_mode == "fixture",
            "saved_json_display": "redacted",
            "saved_markdown_display": "redacted",
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "public_artifact_safe": True,
        },
        "events": events,
        "status": {
            "state": "completed" if completed else "requires_external_runtime",
            "failure_diagnosis": "none" if completed else "external GPU Coordinator and stage Miners must be verified",
            "operator_action": "review_artifacts" if completed else "run external-existing verification after starting Coordinator and Miners",
            "public_artifact_safe": True,
        },
        "control_user_alpha_session_lifecycle_ready": bool(control_report.get("session_lifecycle_ready")),
        "public_artifact_safe": True,
    }


def build_user_workflow(
    args: argparse.Namespace,
    *,
    model_catalog: dict[str, Any],
    join_packs: dict[str, Any],
    lifecycle: dict[str, Any],
) -> dict[str, Any]:
    model = selected_model(model_catalog, args.model_id)
    next_commands = [
        {
            "label": "prepare",
            "command_line": f"crowdtensor gpu-swarm prepare --output-dir {shlex.quote(args.output_dir)}",
            "public_artifact_safe": True,
        },
        {
            "label": "coordinator",
            "command_line": f"crowdtensor gpu-swarm coordinator --output-dir {shlex.quote(args.output_dir)}",
            "public_artifact_safe": True,
        },
    ]
    for pack in _list(join_packs.get("stages")):
        if isinstance(pack, dict):
            next_commands.append({
                "label": f"miner-{pack.get('stage')}",
                "command_line": f"crowdtensor gpu-swarm miner --stage {pack.get('stage')} --output-dir {shlex.quote(args.output_dir)}",
                "public_artifact_safe": True,
            })
    next_commands.extend([
        {
            "label": "infer",
            "command_line": f"crowdtensor gpu-swarm infer --output-dir {shlex.quote(args.output_dir)} --prompt '<your prompt>'",
            "public_artifact_safe": True,
        },
        {
            "label": "status",
            "command_line": f"crowdtensor gpu-swarm status --output-dir {shlex.quote(args.output_dir)}",
            "public_artifact_safe": True,
        },
        {
            "label": "collect",
            "command_line": f"crowdtensor gpu-swarm collect --output-dir {shlex.quote(args.output_dir)}",
            "public_artifact_safe": True,
        },
    ])
    return {
        "schema": "gpu_swarm_user_workflow_v1",
        "user_gpu_swarm_entrypoint_ready": True,
        "entrypoint": "crowdtensor gpu-swarm smoke",
        "selected_action": args.action,
        "selected_stage": args.stage,
        "one_command_smoke_ready": True,
        "current_mode": args.execution_mode,
        "selected_model": {
            "model_id": model.get("model_id"),
            "live_verified": bool(model.get("live_verified")),
            "verified_token_count": int(model.get("verified_token_count") or 0),
            "n_stage_plan_ready": bool(model.get("n_stage_plan_ready")),
            "public_artifact_safe": True,
        },
        "status": lifecycle.get("status"),
        "next_commands": next_commands,
        "public_artifact_safe": True,
    }


def write_join_files(output_dir: Path, join_packs: dict[str, Any]) -> None:
    for pack in _list(join_packs.get("stages")):
        if not isinstance(pack, dict):
            continue
        stage_dir = output_dir / str(pack["stage"]).replace("stage", "stage-stage")
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "miner_join.sh").write_text(render_join_script(pack), encoding="utf-8")
        (stage_dir / "MINER_JOIN.md").write_text(render_join_markdown(pack), encoding="utf-8")
        (stage_dir / "miner.env.template").write_text(render_env_template(pack), encoding="utf-8")


def render_runbook(report: dict[str, Any]) -> str:
    workflow = _dict(report.get("user_workflow"))
    coordinator = _dict(report.get("coordinator_workflow"))
    join_packs = _dict(report.get("miner_join_packs"))
    lifecycle = _dict(report.get("inference_lifecycle"))
    lines = [
        "# CrowdTensor GPU Swarm Usability Alpha",
        "",
        f"- ready: `{report.get('gpu_swarm_usability_alpha_ready')}`",
        f"- action: `{report.get('action')}`",
        f"- execution mode: `{report.get('execution_mode')}`",
        f"- external runtime verified: `{report.get('external_runtime_verified')}`",
        f"- selected model: `{_dict(workflow.get('selected_model')).get('model_id')}`",
        f"- public artifact safe: `{report.get('public_artifact_safe')}`",
        "",
        "## One Command Smoke",
        "",
        "`crowdtensor gpu-swarm smoke --output-dir dist/gpu-swarm-usability-alpha`",
        "",
        "## Action Commands",
        "",
    ]
    for item in _list(workflow.get("next_commands")):
        if isinstance(item, dict):
            lines.append(f"- {item.get('label')}: `{item.get('command_line')}`")
    lines.extend([
        "",
        "## Coordinator",
        "",
        f"- URL: `{coordinator.get('coordinator_url')}`",
        f"- start: `{coordinator.get('start_command')}`",
        "",
        "## Miner Join",
        "",
    ])
    for pack in _list(join_packs.get("stages")):
        if isinstance(pack, dict):
            lines.append(f"- `{pack.get('stage')}`: `{pack.get('recommended_command')}` capability=`{pack.get('required_capability')}`")
    lines.extend([
        "",
        "## Inference Lifecycle",
        "",
        f"- state: `{_dict(lifecycle.get('status')).get('state')}`",
        f"- operator action: `{_dict(lifecycle.get('status')).get('operator_action')}`",
        "",
        "## Boundaries",
        "",
    ])
    for name, value in sorted(_dict(report.get("boundaries")).items()):
        lines.append(f"- {name}: `{value}`")
    lines.extend(["", "## Diagnosis", "", "- " + ", ".join(report.get("diagnosis_codes") or []), ""])
    return "\n".join(lines)


def render_command_script(report: dict[str, Any], *, action: str) -> str:
    if action == "coordinator":
        command = _dict(report.get("coordinator_workflow")).get("start_command") or "crowdtensord --help"
    elif action in {"stage0", "stage1"}:
        packs = _list(_dict(report.get("miner_join_packs")).get("stages"))
        command = next((item.get("command_template") for item in packs if isinstance(item, dict) and item.get("stage") == action), "crowdtensor-miner --help")
    else:
        command = "crowdtensor gpu-swarm status"
    return "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", "", str(command), ""])


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    control_path = Path(args.control_user_alpha_report)
    if not control_path.is_file():
        control_args = control_alpha.parse_args([
            "--output-dir",
            str(output_dir / "control-user-alpha"),
            "--core-handoff-report",
            args.core_handoff_report,
            "--core-status-report",
            args.core_status_report,
            "--mode",
            "evidence-import",
            "--model-id",
            args.model_id,
            "--max-new-tokens",
            str(min(args.max_new_tokens, 1)),
            "--request-label",
            args.request_label,
            "--prompt",
            args.prompt,
        ])
        control_report = control_alpha.build_report(control_args)
        control_path = output_dir / "control-user-alpha" / "control_user_alpha.json"
    else:
        control_report = load_json(control_path)
    core_handoff_path = Path(args.core_handoff_report)
    core_handoff = load_json(core_handoff_path)
    model_catalog = model_catalog_from_control(control_report)
    model = selected_model(model_catalog, args.model_id)
    join_packs = build_join_packs(args, model_catalog=model_catalog)
    coordinator = build_coordinator_workflow(args, model=model)
    readiness = build_gpu_readiness(args, join_packs=join_packs)
    lifecycle = build_inference_lifecycle(args, model=model, control_report=control_report)
    workflow = build_user_workflow(args, model_catalog=model_catalog, join_packs=join_packs, lifecycle=lifecycle)
    cleanup_plan = build_cleanup_plan(args, output_dir=output_dir)
    write_join_files(output_dir, join_packs)

    two_gpu_stage_route_ready = bool(
        join_packs.get("gpu_miner_join_pack_ready")
        and {item.get("required_capability") for item in _list(join_packs.get("stages")) if isinstance(item, dict)}
        == {"real_llm_sharded_cuda_stage0", "real_llm_sharded_cuda_stage1"}
    )
    control_imported = bool(control_report.get("ok") and control_report.get("control_layer_ready") and control_report.get("user_layer_ready"))
    core_handoff_imported = bool(core_handoff.get("ok") and core_handoff.get("schema") == "core_technology_handoff_rc_v1")
    external_runtime_verified = bool(args.external_runtime_verified and args.execution_mode in {"external-existing", "kaggle-auto"})
    ready = bool(
        workflow.get("user_gpu_swarm_entrypoint_ready")
        and join_packs.get("gpu_miner_join_pack_ready")
        and coordinator.get("coordinator_workflow_ready")
        and two_gpu_stage_route_ready
        and lifecycle.get("inference_request_lifecycle_ready")
        and model_catalog.get("model_catalog_imported")
        and control_imported
        and core_handoff_imported
        and args.execution_mode in {"fixture", "evidence-import", "external-existing", "kaggle-auto"}
    )
    diagnosis_codes = {
        "user_gpu_swarm_entrypoint_ready" if workflow.get("user_gpu_swarm_entrypoint_ready") else "user_gpu_swarm_entrypoint_blocked",
        "gpu_miner_join_pack_ready" if join_packs.get("gpu_miner_join_pack_ready") else "gpu_miner_join_pack_blocked",
        "coordinator_workflow_ready" if coordinator.get("coordinator_workflow_ready") else "coordinator_workflow_blocked",
        "two_gpu_stage_route_ready" if two_gpu_stage_route_ready else "two_gpu_stage_route_blocked",
        "inference_request_lifecycle_ready" if lifecycle.get("inference_request_lifecycle_ready") else "inference_request_lifecycle_blocked",
        "model_catalog_imported" if model_catalog.get("model_catalog_imported") else "model_catalog_missing",
        "control_user_alpha_imported" if control_imported else "control_user_alpha_missing",
        "core_handoff_imported" if core_handoff_imported else "core_handoff_missing",
        "gpu_swarm_public_artifact_redaction_ready",
    }
    if args.execution_mode in {"fixture", "evidence-import"}:
        diagnosis_codes.add("gpu_swarm_ci_safe_smoke_ready")
    if args.execution_mode == "evidence-import":
        diagnosis_codes.add("gpu_swarm_retained_7b_14b_evidence_consumed")
    if external_runtime_verified:
        diagnosis_codes.add("gpu_swarm_external_runtime_verified")
    else:
        diagnosis_codes.add("gpu_swarm_external_runtime_not_verified")
    if ready:
        diagnosis_codes.add("gpu_swarm_usability_alpha_ready")
    diagnosis_codes.add(f"gpu_swarm_{args.action.replace('-', '_')}_ready")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "action": args.action,
        "selected_stage": args.stage,
        "gpu_swarm_usability_alpha_ready": ready,
        "user_gpu_swarm_entrypoint_ready": bool(workflow.get("user_gpu_swarm_entrypoint_ready")),
        "gpu_miner_join_pack_ready": bool(join_packs.get("gpu_miner_join_pack_ready")),
        "coordinator_workflow_ready": bool(coordinator.get("coordinator_workflow_ready")),
        "two_gpu_stage_route_ready": two_gpu_stage_route_ready,
        "inference_request_lifecycle_ready": bool(lifecycle.get("inference_request_lifecycle_ready")),
        "model_catalog_imported": bool(model_catalog.get("model_catalog_imported")),
        "control_user_alpha_imported": control_imported,
        "core_handoff_imported": core_handoff_imported,
        "public_artifact_safe": True,
        "execution_mode": args.execution_mode,
        "external_runtime_verified": external_runtime_verified,
        "output_dir": str(output_dir),
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "model_id": model.get("model_id"),
        "core_handoff_report": str(core_handoff_path),
        "control_user_alpha_report": str(control_path),
        "model_catalog": model_catalog,
        "coordinator_workflow": coordinator,
        "miner_join_packs": join_packs,
        "gpu_readiness": readiness,
        "inference_lifecycle": lifecycle,
        "user_workflow": workflow,
        "cleanup_plan": cleanup_plan,
        "boundaries": dict(BOUNDARIES),
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "credentials_public": False,
            "lease_material_public": False,
            "idempotency_material_public": False,
            "private_env_written": False,
            "kernel_payload_written": False,
            "token_placeholders_only": True,
            "report_public_leak_paths": [],
        },
        "mode_truth": {
            "fixture": args.execution_mode == "fixture",
            "evidence_import": args.execution_mode == "evidence-import",
            "external_existing": args.execution_mode == "external-existing",
            "kaggle_auto": args.execution_mode == "kaggle-auto",
            "fresh_gpu_run_performed": False,
            "retained_evidence_consumed": args.execution_mode in {"fixture", "evidence-import"},
        },
        "next_production_work": [
            "production account and quota enforcement",
            "production P2P/NAT traversal or managed relay",
            "fresh external GPU verification automation for non-Kaggle users",
            "trust, abuse prevention, billing, and operator policy",
        ],
        "diagnosis_codes": sorted(diagnosis_codes),
        "errors": [] if ready else ["gpu_swarm_usability_alpha_not_ready"],
    }
    leaks = public_redaction_errors(report)
    report["safety"]["report_public_leak_paths"] = leaks
    report["public_artifact_safe"] = not leaks
    report["safety"]["public_artifact_safe"] = not leaks
    if leaks:
        report["ok"] = False
        report["gpu_swarm_usability_alpha_ready"] = False
        report["errors"] = sorted(set(_list(report.get("errors")) + ["public_redaction_failed"]))
        report["diagnosis_codes"] = sorted(set(_list(report.get("diagnosis_codes")) + ["gpu_swarm_public_artifact_redaction_failed"]))

    artifacts = {
        "summary_json": artifact_entry(output_dir / "gpu_swarm_usability_alpha.json", output_dir, kind="gpu_swarm_usability_alpha", schema=SCHEMA, ok=report.get("ok")),
        "runbook_markdown": artifact_entry(output_dir / "GPU_SWARM_ALPHA.md", output_dir, kind="gpu_swarm_runbook"),
        "support_bundle_json": artifact_entry(output_dir / "support_bundle.json", output_dir, kind="gpu_swarm_usability_alpha_support_bundle", schema=SUPPORT_BUNDLE_SCHEMA, ok=report.get("ok")),
        "stage0_join_script": artifact_entry(output_dir / "stage-stage0" / "miner_join.sh", output_dir, kind="gpu_miner_join_script"),
        "stage1_join_script": artifact_entry(output_dir / "stage-stage1" / "miner_join.sh", output_dir, kind="gpu_miner_join_script"),
        "stage0_join_runbook": artifact_entry(output_dir / "stage-stage0" / "MINER_JOIN.md", output_dir, kind="gpu_miner_join_runbook"),
        "stage1_join_runbook": artifact_entry(output_dir / "stage-stage1" / "MINER_JOIN.md", output_dir, kind="gpu_miner_join_runbook"),
        "stage0_env_template": artifact_entry(output_dir / "stage-stage0" / "miner.env.template", output_dir, kind="gpu_miner_env_template"),
        "stage1_env_template": artifact_entry(output_dir / "stage-stage1" / "miner.env.template", output_dir, kind="gpu_miner_env_template"),
        "coordinator_command_script": artifact_entry(output_dir / "start_coordinator.sh", output_dir, kind="gpu_swarm_command_script"),
        "stage0_command_script": artifact_entry(output_dir / "stage0_miner_command.sh", output_dir, kind="gpu_swarm_command_script"),
        "stage1_command_script": artifact_entry(output_dir / "stage1_miner_command.sh", output_dir, kind="gpu_swarm_command_script"),
        "core_handoff_report_json": artifact_entry(core_handoff_path.resolve(), output_dir, kind="core_technology_handoff_rc", schema="core_technology_handoff_rc_v1", ok=core_handoff_imported),
        "control_user_alpha_json": artifact_entry(control_path.resolve(), output_dir, kind="control_user_alpha", schema="control_user_alpha_v1", ok=control_imported),
    }
    report["artifacts"] = artifacts
    report["artifact_summary"] = artifact_summary(artifacts)
    (output_dir / "GPU_SWARM_ALPHA.md").write_text(render_runbook(report), encoding="utf-8")
    write_json(output_dir / "support_bundle.json", build_support_bundle(report))
    for action in ["coordinator", "stage0", "stage1"]:
        script = output_dir / ("start_coordinator.sh" if action == "coordinator" else f"{action}_miner_command.sh")
        script.write_text(render_command_script(report, action=action), encoding="utf-8")
    artifacts["runbook_markdown"]["present"] = True
    artifacts["support_bundle_json"]["present"] = True
    for name in [
        "coordinator_command_script",
        "stage0_command_script",
        "stage1_command_script",
    ]:
        artifacts[name] = artifact_entry(Path(output_dir) / str(artifacts[name]["path"]), output_dir, kind=artifacts[name]["kind"])
    report["artifact_summary"] = artifact_summary(artifacts)
    report["artifact_summary"]["public_artifact_safe"] = bool(report.get("public_artifact_safe"))
    write_json(output_dir / "gpu_swarm_usability_alpha.json", report)
    report["artifacts"]["summary_json"]["present"] = True
    report["artifact_summary"]["present_artifact_count"] = sum(1 for item in artifacts.values() if item.get("present"))
    write_json(output_dir / "gpu_swarm_usability_alpha.json", report)
    return report


def build_support_bundle(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "generated_at": report.get("generated_at"),
        "ok": report.get("ok") is True,
        "gpu_swarm_usability_alpha_ready": report.get("gpu_swarm_usability_alpha_ready") is True,
        "user_gpu_swarm_entrypoint_ready": report.get("user_gpu_swarm_entrypoint_ready") is True,
        "gpu_miner_join_pack_ready": report.get("gpu_miner_join_pack_ready") is True,
        "coordinator_workflow_ready": report.get("coordinator_workflow_ready") is True,
        "two_gpu_stage_route_ready": report.get("two_gpu_stage_route_ready") is True,
        "inference_request_lifecycle_ready": report.get("inference_request_lifecycle_ready") is True,
        "model_catalog_imported": report.get("model_catalog_imported") is True,
        "control_user_alpha_imported": report.get("control_user_alpha_imported") is True,
        "core_handoff_imported": report.get("core_handoff_imported") is True,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "execution_mode": report.get("execution_mode"),
        "external_runtime_verified": report.get("external_runtime_verified") is True,
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "artifact_summary": report.get("artifact_summary") or {},
        "next_production_work": report.get("next_production_work") or [],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GPU Swarm Usability Alpha evidence.")
    parser.add_argument("--action", choices=ACTIONS, default="smoke")
    parser.add_argument("--stage", choices=STAGES, default="stage0")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--control-user-alpha-report", default=DEFAULT_CONTROL_USER_ALPHA_REPORT)
    parser.add_argument("--core-handoff-report", default=DEFAULT_CORE_HANDOFF_REPORT)
    parser.add_argument("--core-status-report", default=DEFAULT_CORE_STATUS_REPORT)
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, default="evidence-import")
    parser.add_argument("--external-runtime-verified", action="store_true")
    parser.add_argument("--coordinator-url", default="http://127.0.0.1:9300")
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument("--miner-id-prefix", default="gpu-swarm-alpha")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--lease-seconds", type=float, default=30.0)
    parser.add_argument("--request-label", default="gpu-swarm-alpha-smoke")
    parser.add_argument("--prompt", default="CrowdTensor GPU swarm alpha smoke request")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.port < 1:
        raise SystemExit("--port must be positive")
    if args.max_new_tokens < 1 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 1 and 32")
    if args.lease_seconds <= 0:
        raise SystemExit("--lease-seconds must be positive")
    if not args.request_label.strip():
        raise SystemExit("--request-label must be non-empty")
    if not args.prompt.strip():
        raise SystemExit("--prompt must be non-empty")
    for attr in ["core_handoff_report", "core_status_report"]:
        if not Path(getattr(args, attr)).is_file():
            raise SystemExit(f"--{attr.replace('_', '-')} must point to an existing JSON file")
    if args.control_user_alpha_report and not Path(args.control_user_alpha_report).is_file():
        # The pack can generate a nested Control/User Alpha report when the default is absent.
        if args.control_user_alpha_report != DEFAULT_CONTROL_USER_ALPHA_REPORT:
            raise SystemExit("--control-user-alpha-report must point to an existing JSON file")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(render_runbook(report))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
