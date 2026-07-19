import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab_cuda_session_probe.py"
spec = importlib.util.spec_from_file_location("colab_cuda_session_probe", MODULE_PATH)
probe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(probe)


def test_allocate_gpu_uses_gpu_variant_and_authuser(monkeypatch):
    urls = []

    def fake_http_request(url, **kwargs):
        urls.append(url)
        if kwargs.get("method") == "POST":
            return 200, '{"endpoint":"endpoint","accelerator":"T4","variant":"GPU","runtimeProxyInfo":{"token":"proxy","url":"https://proxy.example"}}'
        return 200, '{"token":"xsrf"}'

    monkeypatch.setattr(probe.base, "http_request", fake_http_request)

    assigned = probe.allocate_gpu("token", "T4", authuser="2")

    assert assigned["endpoint"] == "endpoint"
    assert all("authuser=2" in url for url in urls)
    assert all("variant=GPU" in url for url in urls)
    assert all("accelerator=T4" in url for url in urls)


def test_save_gpu_session_marks_variant_gpu(tmp_path):
    state_path = tmp_path / "sessions.json"
    assignment = {
        "endpoint": "endpoint",
        "accelerator": "T4",
        "runtimeProxyInfo": {"token": "proxy-token", "url": "https://proxy.example"},
    }

    session = probe.save_gpu_session(state_path, "gpu-session", assignment)

    assert session["variant"] == "GPU"
    assert session["accelerator"] == "T4"
    assert session["url"] == "https://proxy.example"
    assert state_path.stat().st_mode & 0o777 == 0o600


def test_assignment_is_gpu_accepts_known_accelerators():
    assert probe.assignment_is_gpu({"variant": "GPU"}) is True
    assert probe.assignment_is_gpu({"accelerator": "T4"}) is True
    assert probe.assignment_is_gpu({"accelerator": "V5E1"}) is False
