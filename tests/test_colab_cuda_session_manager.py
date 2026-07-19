from __future__ import annotations

from pathlib import Path

from scripts import colab_cuda_session_manager as manager


def test_is_stale_error_detects_colab_kernel_404() -> None:
    assert manager.is_stale_error("HTTPError: 404 Client Error: Not Found for url: /api/kernels") is True
    assert manager.is_stale_error("Kernel not found at proxy") is True
    assert manager.is_stale_error("CUDA out of memory") is False


def test_clear_runtime_ids_preserves_private_proxy_fields(tmp_path: Path) -> None:
    state = tmp_path / "sessions.json"
    manager.write_state(state, {
        "gpu": {
            "url": "https://proxy.example",
            "token": "secret",
            "endpoint": "endpoint",
            "kernel_id": "kid",
            "session_id": "sid",
        }
    })

    manager.clear_runtime_ids(state, "gpu")
    session = manager.load_session(state, "gpu")

    assert session["url"] == "https://proxy.example"
    assert session["token"] == "secret"
    assert session["kernel_id"] is None
    assert session["session_id"] is None


def test_execute_with_retry_reacquires_after_stale(monkeypatch, tmp_path: Path) -> None:
    state = tmp_path / "sessions.json"
    manager.write_state(state, {
        "gpu": {
            "url": "https://old.example",
            "token": "old-token",
            "endpoint": "old-endpoint",
            "kernel_id": "old-kid",
            "session_id": "old-sid",
        }
    })
    calls: list[str] = []

    class FakeRuntime:
        def __init__(self, url, token, kernel_id=None, session_id=None):
            self.url = url
            self.token = token
            self.kernel_id = "new-kid"
            self.session_id = "new-sid"

        def execute_code(self, code: str, timeout: float):
            calls.append(self.url)
            if self.url == "https://old.example":
                raise RuntimeError("HTTPError: 404 Client Error: Not Found for url: /api/kernels")
            return [{"text": "ok"}]

        def stop(self):
            return None

    def fake_reacquire(**kwargs):
        session = {
            "url": "https://new.example",
            "token": "new-token",
            "endpoint": "new-endpoint",
            "kernel_id": None,
            "session_id": None,
        }
        manager.write_state(state, {"gpu": session})
        return session, {"ok": True, "reacquired": True}

    monkeypatch.setattr(manager, "build_runtime", lambda session: FakeRuntime(session["url"], session["token"]))
    monkeypatch.setattr(manager, "reacquire_gpu_session", fake_reacquire)

    outputs, session, result = manager.execute_with_retry(
        "print('x')",
        session_name="gpu",
        state_path=state,
        max_attempts=2,
        timeout=60,
    )

    assert outputs == [{"text": "ok"}]
    assert session["url"] == "https://new.example"
    assert result["ok"] is True
    assert calls == ["https://old.example", "https://new.example"]
    assert any(item.get("event") == "reacquire_after_stale" for item in result["attempts"])


def test_execute_with_retry_can_keep_runtime_alive(monkeypatch, tmp_path: Path) -> None:
    state = tmp_path / "sessions.json"
    manager.write_state(state, {
        "gpu": {
            "url": "https://runtime.example",
            "token": "token",
            "endpoint": "endpoint",
        }
    })
    stops = 0

    class FakeRuntime:
        kernel_id = "kid"
        session_id = "sid"

        def execute_code(self, code: str, timeout: float):
            return [{"text": "ok"}]

        def stop(self):
            nonlocal stops
            stops += 1

    monkeypatch.setattr(manager, "build_runtime", lambda session: FakeRuntime())

    outputs, session, result = manager.execute_with_retry(
        "print('x')",
        session_name="gpu",
        state_path=state,
        max_attempts=1,
        timeout=60,
        stop_runtime_after_success=False,
    )

    assert outputs == [{"text": "ok"}]
    saved = manager.load_session(state, "gpu")
    assert saved["kernel_id"] == "kid"
    assert session["url"] == "https://runtime.example"
    assert result["ok"] is True
    assert stops == 0
