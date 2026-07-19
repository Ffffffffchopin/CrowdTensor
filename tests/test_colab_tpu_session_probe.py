import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab_tpu_session_probe.py"
spec = importlib.util.spec_from_file_location("colab_tpu_session_probe", MODULE_PATH)
probe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(probe)


def test_list_assignments_uses_requested_authuser(monkeypatch):
    seen = {}

    def fake_http_request(url, **kwargs):
        seen["url"] = url
        return 200, "{}"

    monkeypatch.setattr(probe, "http_request", fake_http_request)

    probe.list_assignments("token", authuser="2")

    assert "authuser=2" in seen["url"]


def test_allocate_uses_requested_authuser(monkeypatch):
    urls = []

    def fake_http_request(url, **kwargs):
        urls.append(url)
        if kwargs.get("method") == "POST":
            return 200, '{"endpoint":"endpoint","runtimeProxyInfo":{"token":"proxy","url":"https://proxy.example"}}'
        return 200, '{"token":"xsrf"}'

    monkeypatch.setattr(probe, "http_request", fake_http_request)

    assigned = probe.allocate("token", "V5E1", authuser="1")

    assert assigned["endpoint"] == "endpoint"
    assert all("authuser=1" in url for url in urls)


def test_unassign_uses_requested_authuser(monkeypatch):
    urls = []

    def fake_http_request(url, **kwargs):
        urls.append(url)
        return 200, '{"token":"xsrf"}'

    monkeypatch.setattr(probe, "http_request", fake_http_request)

    assert probe.unassign("token", "endpoint", authuser="3") == 200
    assert all("authuser=3" in url for url in urls)
