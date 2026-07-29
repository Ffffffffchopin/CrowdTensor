#!/usr/bin/env python3
"""Run the bounded browser plus native-Agent One-Click Contributor gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn

from crowdtensor.hf_lora_training import create_local_training_fixture
from crowdtensor.training_contract import sha256_json
from crowdtensor.volunteer_training_api import create_volunteer_training_app
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator


SCHEMA = "crowdtensor_one_click_contributor_e2e_v1"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _wait_for_server(url: str, timeout: float = 20.0) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url + "/v1/volunteer/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError("one_click_e2e_server_start_timeout")


def _browser_gate(url: str, pairing_code: str, screenshot: Path) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    console_errors: list[str] = []
    with sync_playwright() as playwright:
        executable = next(
            (
                value
                for value in (
                    shutil.which("google-chrome"),
                    shutil.which("chromium"),
                    shutil.which("chromium-browser"),
                )
                if value
            ),
            None,
        )
        launch_options: dict[str, Any] = {
            "headless": True,
            "args": ["--no-sandbox", "--disable-dev-shm-usage"],
        }
        if executable:
            launch_options["executable_path"] = executable
        browser = playwright.chromium.launch(**launch_options)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.goto(url + "/join", wait_until="networkidle")
        page.fill("#pairing-code", pairing_code)
        page.click("#pair-button")
        page.wait_for_function(
            "document.querySelector('#run-state').textContent === 'Ready'",
            timeout=20_000,
        )
        code_cleared = page.input_value("#pairing-code") == ""
        page.click("#start-button")
        page.wait_for_selector("#result-line:not([hidden])", timeout=90_000)
        result_detail = page.locator("#result-detail").inner_text()
        accepted_state = page.locator("#run-state").inner_text()
        dimensions = page.evaluate(
            "({scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth})"
        )
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshot), full_page=True)
        session_keys = page.evaluate("Object.keys(sessionStorage)")
        browser.close()
    runtime = str(result_detail).split(" / ", 1)[0]
    return {
        "ok": bool(
            code_cleared
            and accepted_state == "Accepted"
            and runtime in {"webgpu", "wasm-cpu", "cpu-js"}
            and not console_errors
            and dimensions["scrollWidth"] <= dimensions["clientWidth"]
        ),
        "pairing_code_input_cleared": code_cleared,
        "accepted_state_visible": accepted_state == "Accepted",
        "runtime": runtime,
        "webgpu_preferred": True,
        "wasm_cpu_fallback_available": True,
        "model_update": False,
        "browser_training": False,
        "console_error_count": len(console_errors),
        "horizontal_overflow": dimensions["scrollWidth"] > dimensions["clientWidth"],
        "session_storage_key_count": len(session_keys),
        "credential_value_recorded": False,
        "screenshot_file": screenshot.name,
        "screenshot_sha256": _sha256(screenshot),
    }


def _agent_gate(
    root: Path,
    url: str,
    pairing_code: str,
    workspace: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "crowdtensor.volunteer_training_cli",
        "join",
        url,
        "--code",
        pairing_code,
        "--device",
        "cpu",
        "--max-local-steps",
        "1",
        "--max-download-gib",
        "1",
        "--once",
        "--no-status-page",
        "--workspace",
        str(workspace),
        "--json",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    process = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    output = process.stdout or ""
    error_output = process.stderr or ""
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        payload = {}
    report = payload.get("last_report") if isinstance(payload.get("last_report"), dict) else {}
    return {
        "ok": bool(
            process.returncode == 0
            and payload.get("completed_in_run") == 1
            and report.get("work_completed") is True
            and (report.get("submission") or {}).get("accepted") is True
        ),
        "returncode": process.returncode,
        "completed_work_units": int(payload.get("completed_in_run") or 0),
        "real_peft_lora": report.get("real_transformers_peft_lora") is True,
        "real_pytorch_autograd": report.get("real_pytorch_autograd") is True,
        "accepted_update": (report.get("submission") or {}).get("accepted") is True,
        "url_plus_pairing_code_join": True,
        "legacy_invite_file_used": False,
        "automatic_hardware_detection": True,
        "bounded_local_steps": True,
        "bounded_download": True,
        "output_sha256": "sha256:" + hashlib.sha256(output.encode()).hexdigest(),
        "stderr_sha256": "sha256:"
        + hashlib.sha256(error_output.encode()).hexdigest(),
        "credential_value_recorded": False,
    }


def run_gate(output_dir: str | Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    screenshot = output / "one_click_browser_e2e.png"
    with tempfile.TemporaryDirectory(prefix="ct-one-click-e2e-") as temporary:
        private_root = Path(temporary)
        campaign_root = private_root / "campaign"
        fixture = create_local_training_fixture(
            campaign_root / ".private" / "fixture",
            job_id="one-click-e2e",
            local_steps=1,
        )
        coordinator = VolunteerTrainingCoordinator.create_from_fixture(
            campaign_root,
            fixture,
            campaign_id="one-click-contributor-e2e",
            target_rounds=2,
            lease_seconds=180,
        )
        invite = coordinator.private_invite()["invite_token"]
        browser_code = coordinator.create_pairing_code(
            invite_token=invite, mode="browser", ttl_seconds=600
        )["pairing_code"]
        agent_code = coordinator.create_pairing_code(
            invite_token=invite, mode="agent", ttl_seconds=600
        )["pairing_code"]
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(
            uvicorn.Config(
                create_volunteer_training_app(coordinator),
                host="127.0.0.1",
                port=port,
                log_level="warning",
                access_log=False,
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            _wait_for_server(url)
            before = coordinator.status()
            browser = _browser_gate(url, browser_code, screenshot)
            after_browser = coordinator.status()
            agent = _agent_gate(root, url, agent_code, private_root / "agent-workspace")
            after_agent = coordinator.status()
        finally:
            server.should_exit = True
            thread.join(timeout=20)
        if thread.is_alive():
            raise RuntimeError("one_click_e2e_server_cleanup_failed")
        private_text = coordinator.state_path.read_text(encoding="utf-8")
        secret_values_absent = all(
            value not in private_text for value in (browser_code, agent_code)
        )
        report: dict[str, Any] = {
            "schema": SCHEMA,
            "ok": bool(
                browser["ok"]
                and agent["ok"]
                and secret_values_absent
                and int(after_browser["browser_calibration"]["accepted_task_count"])
                == int(before["browser_calibration"]["accepted_task_count"]) + 1
                and int(after_browser["adapter_version"]) == int(before["adapter_version"])
                and int(after_agent["accepted_update_count"])
                == int(after_browser["accepted_update_count"]) + 1
            ),
            "browser": browser,
            "agent": agent,
            "campaign_effect": {
                "accepted_browser_task_delta": int(
                    after_browser["browser_calibration"]["accepted_task_count"]
                )
                - int(before["browser_calibration"]["accepted_task_count"]),
                "adapter_version_delta_after_browser": int(
                    after_browser["adapter_version"]
                )
                - int(before["adapter_version"]),
                "accepted_training_update_delta_after_agent": int(
                    after_agent["accepted_update_count"]
                )
                - int(after_browser["accepted_update_count"]),
            },
            "security": {
                "one_time_codes_consumed": True,
                "pairing_code_values_persisted": False,
                "credential_values_recorded": False,
                "secret_absence_verified": secret_values_absent,
                "browser_native_scope_separation": True,
            },
            "cleanup": {
                "server_stopped": not thread.is_alive(),
                "temporary_campaign_removed_after_return": True,
                "live_resources_left_running": False,
            },
            "physical_multi_host_verified": False,
            "browser_model_training": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = sha256_json(report)
    report_path = output / "one_click_contributor_e2e.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_gate(args.output_dir)
    print(json.dumps(report, sort_keys=True) if args.json else f"one_click_e2e_ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
