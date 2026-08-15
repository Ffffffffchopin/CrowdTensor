from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from crowdtensor.core.workspace import export_workspace, init_project, inspect_workspace


ROOT = Path(__file__).resolve().parents[1]


def test_archive_manifest_has_a_recoverable_baseline_and_active_boundary() -> None:
    manifest = json.loads(
        (ROOT / "architecture/archive-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "crowdtensor_repository_archive_manifest_v1"
    assert manifest["archive_ref"] == "e332a7b"
    assert manifest["direction"] == "training_first_v2"
    assert "crowdtensor/core/**" in manifest["active_roots"]
    assert "crowdtensor/backends/**" in manifest["active_roots"]
    assert manifest["historical_evidence"]["source_of_truth"].startswith("Git history")


def test_retired_top_level_surfaces_do_not_return_to_the_active_tree() -> None:
    forbidden = (
        "coordinator.py",
        "miner_cli.py",
        "compose.yaml",
        "package.json",
        "package-lock.json",
        "web",
        "site",
        "deploy",
        "examples",
        "requirements",
        "crowdtensor/real_llm.py",
        "crowdtensor/p2p_lite.py",
        "crowdtensor/qwen15b_training.py",
        "crowdtensor/glm52_kaggle_alpha.py",
        "crowdtensor/community_cli.py",
        "build",
        "crowdtensord.egg-info",
    )
    assert [item for item in forbidden if (ROOT / item).exists()] == []
    assert sorted(path.name for path in (ROOT / "scripts").iterdir() if path.is_file()) == [
        "check_repository.py",
        "install_contributor.sh",
    ]


def test_public_cli_is_lazy_and_contains_only_training_first_commands() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from crowdtensor.cli import _parser; "
                "text=_parser().format_help(); "
                "assert all(x in text for x in ('train','volunteer','adapters')); "
                "assert all(x not in text for x in ('p2p-daemon','glm52','cpu-infer')); "
                "blocked={'torch','jax','transformers','accelerate','deepspeed'}; "
                "assert not blocked.intersection(sys.modules)"
            ),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_compatibility_imports_forward_to_v2_adapter_modules() -> None:
    from crowdtensor.adapters.capabilities import CAPABILITY_SCHEMA as current_capability
    from crowdtensor.adapters.manifests import MANIFEST_SCHEMA as current_manifest
    from crowdtensor.heterogeneous_training_manifest import MANIFEST_SCHEMA
    from crowdtensor.heterogeneous_training_scheduler import CAPABILITY_SCHEMA

    assert MANIFEST_SCHEMA == current_manifest
    assert CAPABILITY_SCHEMA == current_capability


def test_existing_v2_workspace_remains_inspectable_and_exportable(tmp_path) -> None:
    workspace = tmp_path / "existing-workspace"
    init_project(
        workspace,
        model="org/model",
        model_revision="model-revision",
        dataset="org/data",
        dataset_revision="data-revision",
        model_adapter="qwen2_lora_v1",
        training_backend="volunteer_peft",
        target_steps=3,
    )
    before = inspect_workspace(workspace)
    destination = tmp_path / "public-export"
    exported = export_workspace(workspace, destination)
    after = inspect_workspace(workspace)

    assert before["project_hash"] == after["project_hash"]
    assert exported["command_ok"] is True
    assert (destination / "project.json").is_file()
    assert str(tmp_path) not in json.dumps(exported, sort_keys=True)
