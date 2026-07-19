from __future__ import annotations

import json
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts import kaggle_model_attach_probe as probe


def _write_fake_safetensors(path: Path, keys: list[str]) -> None:
    header = {
        key: {
            "dtype": "BF16",
            "shape": [2, 2],
            "data_offsets": [index * 8, (index + 1) * 8],
        }
        for index, key in enumerate(keys)
    }
    raw = json.dumps(header, sort_keys=True).encode("utf-8")
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"0" * 64)


def test_rendered_stage_plan_kernel_compiles_and_runs_on_fake_attached_model() -> None:
    base = Path(tempfile.mkdtemp(prefix="crowdtensor_attach_probe_test_"))
    model_dir = base / "models" / "qwen-lm" / "qwen2.5" / "transformers" / "tiny" / "1"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 8,
                "model_type": "qwen2",
                "num_hidden_layers": 2,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    weight_map = {
        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
        "model.layers.1.input_layernorm.weight": "model-00002-of-00002.safetensors",
        "model.norm.weight": "model-00002-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors",
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    _write_fake_safetensors(
        model_dir / "model-00001-of-00002.safetensors",
        ["model.embed_tokens.weight", "model.layers.0.input_layernorm.weight"],
    )
    _write_fake_safetensors(
        model_dir / "model-00002-of-00002.safetensors",
        ["model.layers.1.input_layernorm.weight", "model.norm.weight", "lm_head.weight"],
    )

    kernel = base / "kernel.py"
    kernel.write_text(
        probe.kernel_code(
            expected_path=str(model_dir),
            expected_paths=[str(base / "missing"), str(model_dir)],
            model_source="qwen-lm/qwen2.5/Transformers/tiny/1",
            stage_plan_enabled=True,
            stage_count=2,
            stage_backends=["cuda", "jax_tpu"],
        ),
        encoding="utf-8",
    )
    py_compile.compile(str(kernel), doraise=True)

    proc = subprocess.run(
        [sys.executable, str(kernel)],
        check=False,
        cwd=base,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout
    runtime = json.loads((base / "kaggle_model_attach_runtime_report.json").read_text(encoding="utf-8"))
    assert runtime["ok"] is True
    assert runtime["attach_ok"] is True
    assert runtime["path_present"] is True
    assert runtime["resolved_attached_path"] == str(model_dir)
    assert len(runtime["attached_path_probes"]) == 2
    assert runtime["stage_owned_preflight_verified"] is True
    assert runtime["weight_tensor_values_public"] is False
    assert runtime["stage_plan"]["schema"] == "kaggle_model_attach_stage_plan_v1"
    assert runtime["stage_plan"]["stage_count"] == 2
    assert runtime["stage_plan"]["stage_backends"] == ["cuda", "jax_tpu"]
    assert runtime["stage_plan"]["assigned_key_count_total"] == 5
    assert runtime["stage_plan"]["present_key_count_total"] == 5
    assert all(item["stage_owned_header_verified"] for item in runtime["stage_plan"]["stage_plans"])
    assert probe.public_redaction_errors(runtime) == []


def test_stage_plan_probe_args_are_bounded() -> None:
    args = probe.parse_args(["--stage-plan", "--stage-count", "10", "--max-header-bytes", "4096"])

    assert args.stage_plan is True
    assert args.stage_count == 10
    assert args.max_header_bytes == 4096


def test_default_expected_paths_include_models_and_short_kaggle_mounts() -> None:
    candidate = {
        "owner_slug": "metaresearch",
        "model_slug": "llama-3.1",
        "framework": "Transformers",
        "instance_slug": "405b",
        "version_number": 1,
    }

    paths = probe.default_expected_paths(candidate)

    assert "/kaggle/input/models/metaresearch/llama-3.1/transformers/405b/1" in paths
    assert "/kaggle/input/metaresearch/llama-3.1/transformers/405b" in paths
