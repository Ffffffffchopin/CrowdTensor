import json
from types import SimpleNamespace

import pytest

import crowdtensor.model_adapter as registry
from crowdtensor.community_cli import run as run_community_cli
from crowdtensor.community_live_training import CommunityLiveCoordinator


class ExamplePluginAdapter(registry.ModelAdapter):
    adapter_id = "example_lora_v1"
    family = "example"
    model_types = ("example",)
    architectures = ("ExampleForCausalLM",)
    default_model_id = "example/model"
    default_revision = "a" * 40
    default_model_license = "apache-2.0"
    target_modules = ("q_proj", "v_proj")
    default_config = {
        "model_type": "example",
        "architectures": ["ExampleForCausalLM"],
        "num_hidden_layers": 4,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 256,
    }

    def production_manifest(self, *, target_steps, accelerators):
        raise registry.ModelAdapterError("example_scheduler_not_supported")


class FakeEntryPoint:
    def __init__(self, name, value, loaded, *, distribution="example-adapter", version="1.2.3"):
        self.name = name
        self.value = value
        self._loaded = loaded
        self.dist = SimpleNamespace(metadata={"Name": distribution}, version=version)

    def load(self):
        return self._loaded


class FakeEntryPoints(list):
    def select(self, *, group):
        return self if group == registry.MODEL_ADAPTER_ENTRY_POINT_GROUP else []


def install_fake_points(monkeypatch, *points) -> None:
    monkeypatch.setattr(
        registry.importlib.metadata,
        "entry_points",
        lambda: FakeEntryPoints(points),
    )


def test_entry_point_plugin_is_discovered_with_public_provenance(monkeypatch) -> None:
    install_fake_points(
        monkeypatch,
        FakeEntryPoint("example_lora_v1", "example:Adapter", ExamplePluginAdapter),
    )
    adapter = registry.get_model_adapter("example_lora_v1")
    assert isinstance(adapter, ExamplePluginAdapter)
    report = registry.adapter_registry_report()
    assert report["schema"] == "crowdtensor_model_adapter_registry_v2"
    assert report["plugin_adapter_count"] == 1
    assert report["supported_model_families"] == ["example", "qwen2", "smollm2"]
    descriptor = next(item for item in report["adapters"] if item["adapter_id"] == adapter.adapter_id)
    assert descriptor["registration"]["kind"] == "entry_point_plugin"
    assert descriptor["registration"]["distribution_name"] == "example-adapter"
    assert descriptor["registration"]["module_path_public"] is False


def test_plugin_loading_can_be_disabled_without_affecting_builtins(monkeypatch) -> None:
    install_fake_points(
        monkeypatch,
        FakeEntryPoint("example_lora_v1", "example:Adapter", ExamplePluginAdapter),
    )
    monkeypatch.setenv("CROWDTENSOR_DISABLE_MODEL_ADAPTER_PLUGINS", "1")
    assert registry.adapter_registry_report()["plugin_adapter_count"] == 0
    assert registry.get_model_adapter("qwen2_lora_v1").family == "qwen2"
    with pytest.raises(registry.ModelAdapterError, match="id_unsupported"):
        registry.get_model_adapter("example_lora_v1")


def test_plugin_drives_live_coordinator_worker_metadata(monkeypatch, tmp_path) -> None:
    install_fake_points(
        monkeypatch,
        FakeEntryPoint("example_lora_v1", "example:Adapter", ExamplePluginAdapter),
    )
    coordinator = CommunityLiveCoordinator(
        tmp_path / "state.json",
        run_id="plugin-runtime",
        target_steps=1,
        model_adapter_id="example_lora_v1",
    )
    registration = coordinator.register(
        worker_id_hash="sha256:" + "a" * 64,
        role="stage0",
        backend="cuda",
    )
    assert registration["model_adapter_id"] == "example_lora_v1"
    assert registration["model_id"] == ExamplePluginAdapter.default_model_id
    assert registration["model_revision"] == ExamplePluginAdapter.default_revision
    assert registration["model_config"]["model_type"] == "example"
    assert registration["split_index"] == 2
    status = coordinator.public_status()
    assert status["model_adapter_id"] == "example_lora_v1"
    assert status["model_id"] == ExamplePluginAdapter.default_model_id


def test_community_cli_lists_and_checks_discovered_plugin(
    monkeypatch, capsys
) -> None:
    install_fake_points(
        monkeypatch,
        FakeEntryPoint("example_lora_v1", "example:Adapter", ExamplePluginAdapter),
    )
    assert run_community_cli(["adapters", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["ok"] is True
    assert listed["plugin_adapter_count"] == 1
    assert "example" in listed["supported_model_families"]
    assert (
        run_community_cli(
            ["adapters", "check", "example_lora_v1", "--json"]
        )
        == 0
    )
    checked = json.loads(capsys.readouterr().out)
    assert checked["ok"] is True
    assert checked["adapter_id"] == "example_lora_v1"


@pytest.mark.parametrize(
    "point",
    [
        FakeEntryPoint("wrong_name", "example:Adapter", ExamplePluginAdapter),
        FakeEntryPoint("qwen2_lora_v1", "example:Adapter", ExamplePluginAdapter),
        FakeEntryPoint("example_lora_v1", "example:not-an-adapter", object()),
        FakeEntryPoint(
            "example_lora_v1",
            "example:Adapter",
            ExamplePluginAdapter,
            distribution="",
        ),
    ],
)
def test_invalid_or_conflicting_plugins_fail_closed(monkeypatch, point) -> None:
    install_fake_points(monkeypatch, point)
    with pytest.raises(registry.ModelAdapterError, match="plugin_discovery_failed"):
        registry.adapter_registry_report()
