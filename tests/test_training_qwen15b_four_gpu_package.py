from __future__ import annotations

import json
import stat

from scripts.training_qwen15b_four_gpu_package import build_package


def _private_payload(path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "crowdtensor_qwen15b_tokenized_private_v1",
                "model_id": "Qwen/Qwen2.5-1.5B",
                "model_revision": "8faed761d45a263340a0528343f099c05c9a4323",
                "sequence_length": 64,
                "train": [[1] * 64 for _ in range(32)],
                "validation": [[2] * 64 for _ in range(8)],
            }
        ),
        encoding="utf-8",
    )


def test_private_packages_pin_two_same_account_t4x2_roles_without_public_secrets(tmp_path) -> None:
    payload = tmp_path / "tokens.json"
    _private_payload(payload)
    config = {
        "model_type": "qwen2",
        "num_hidden_layers": 28,
        "vocab_size": 151936,
        "hidden_size": 1536,
    }
    reports = []
    for role in ("kernel_a", "kernel_b"):
        report = build_package(
            tmp_path / role,
            owner="same-owner",
            slug=f"qwen15b-alpha-{role}",
            role=role,
            config=config,
            tokenized_payload_path=payload,
            coordinator_url="https://private-route.invalid",
            coordinator_token="private-coordinator-token",
            run_id="private-run-id",
        )
        reports.append(report)
        package = tmp_path / role / "private-kernel"
        source = (package / "kernel.py").read_text(encoding="utf-8")
        assert stat.S_IMODE(package.stat().st_mode) == 0o700
        assert stat.S_IMODE((package / "kernel.py").stat().st_mode) == 0o600
        compile(source, str(package / "kernel.py"), "exec")
        metadata = json.loads((package / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert metadata["id"].startswith("same-owner/")
        assert metadata["enable_gpu"] == "true"
        assert metadata["machine_shape"] == "NvidiaTeslaT4"
        assert metadata["is_private"] == "true"
        assert "private-coordinator-token" not in source
        assert "https://private-route.invalid" not in source
        assert 'Version(torchao_version) < Version("0.16.0")' in source
        assert '"uninstall", "-y", "torchao"' in source
        assert "def run_dependency_smoke():" in source
        assert "def run_cuda_mixed_precision_smoke():" in source
        assert "inject_adapter_in_model" in source
        assert '"qwen15b_dependency_smoke_failed:backward_invalid"' in source
        assert source.index("report[\"dependency_smoke\"] = run_dependency_smoke()") < source.index(
            "report[\"stage_runtime_started_ns\"] = time.time_ns()"
        )
        assert source.index(
            "report[\"cuda_mixed_precision_smoke\"] = run_cuda_mixed_precision_smoke()"
        ) < source.index("report[\"stage_runtime_started_ns\"] = time.time_ns()")
        assert source.index("report[\"stage_runtime_started_ns\"] = time.time_ns()") < source.index(
            "worker = run_kernel_role("
        )
        public = json.dumps({key: value for key, value in report.items() if key != "package_dir"})
        assert "private-coordinator-token" not in public
        assert "private-route" not in public
        assert report["model_id"] == "Qwen/Qwen2.5-1.5B"
        assert report["steps"] == 8
        assert report["microbatches_per_step"] == 4
        assert report["dependency_smoke_before_stage_materialization"] is True
        assert report["cuda_mixed_precision_smoke_before_stage_materialization"] is True
        assert report["token_ids_public"] is False
    assert reports[0]["owned_stage_ids"] == [0, 1]
    assert reports[1]["owned_stage_ids"] == [2, 3]
    assert reports[0]["kernel_ref"].split("/", 1)[0] == reports[1]["kernel_ref"].split("/", 1)[0]


def test_elastic_package_embeds_private_resume_contract_without_nonce_leak(tmp_path) -> None:
    payload = tmp_path / "tokens.json"
    _private_payload(payload)
    nonce = "private-new-miner-registration-nonce"
    report = build_package(
        tmp_path / "elastic",
        owner="same-owner",
        slug="qwen15b-elastic-new-kernel-a",
        role="kernel_a",
        config={
            "model_type": "qwen2",
            "num_hidden_layers": 28,
            "vocab_size": 151936,
            "hidden_size": 1536,
        },
        tokenized_payload_path=payload,
        coordinator_url="https://private-route.invalid",
        coordinator_token="private-coordinator-token",
        run_id="private-elastic-run",
        elastic_mode=True,
        miner_id_hash="sha256:" + "a" * 64,
        registration_nonce=nonce,
        expected_start_step=4,
        segment_end_step=8,
        target_steps=8,
    )
    source = (
        tmp_path / "elastic" / "private-kernel" / "kernel.py"
    ).read_text(encoding="utf-8")
    compile(source, "kernel.py", "exec")
    public = json.dumps({key: value for key, value in report.items() if key != "package_dir"})
    assert report["elastic_mode"] is True
    assert report["expected_start_step"] == 4
    assert report["segment_end_step"] == 8
    assert report["central_checkpoint_barrier"] is True
    assert "run_elastic_kernel_role" in source
    assert "elastic_training_runtime.py" not in source  # bundled, not exposed as a path
    assert nonce not in source
    assert nonce not in public
    assert "private-coordinator-token" not in source
    assert "private-route.invalid" not in source


def test_product_miner_package_runs_public_join_path_with_auto_role_and_drain(tmp_path) -> None:
    payload = tmp_path / "tokens.json"
    _private_payload(payload)
    report = build_package(
        tmp_path / "product",
        owner="same-owner",
        slug="qwen15b-product-miner",
        role="kernel_a",
        config={
            "model_type": "qwen2",
            "num_hidden_layers": 28,
            "vocab_size": 151936,
            "hidden_size": 1536,
        },
        tokenized_payload_path=payload,
        coordinator_url="https://private-route.invalid",
        coordinator_token="private-coordinator-token",
        run_id="private-elastic-run",
        elastic_mode=True,
        miner_id_hash="sha256:" + "b" * 64,
        registration_nonce="private-product-nonce",
        expected_start_step=0,
        segment_end_step=8,
        target_steps=8,
        product_miner_mode=True,
        product_role="auto",
        max_steps_per_session=4,
    )
    source = (
        tmp_path / "product" / "private-kernel" / "kernel.py"
    ).read_text(encoding="utf-8")
    compile(source, "kernel.py", "exec")
    public = json.dumps({key: value for key, value in report.items() if key != "package_dir"})
    assert report["product_miner_mode"] is True
    assert report["product_role"] == "auto"
    assert report["max_steps_per_session"] == 4
    assert "run_training_join" in source
    assert 'role=str(private_env.get("product_role") or "auto")' in source
    assert "private-coordinator-token" not in source
    assert "private-coordinator-token" not in public
