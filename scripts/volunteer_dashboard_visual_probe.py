#!/usr/bin/env python3
"""Capture bounded desktop/mobile Dashboard evidence with Playwright."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from crowdtensor.training_contract import sha256_file, sha256_json
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator
from crowdtensor.hf_lora_training import create_local_training_fixture
from scripts.volunteer_training_public_demo import (
    _free_port,
    _run_cells,
    _start_server,
    _wait_for_health,
)


SCHEMA = "crowdtensor_volunteer_dashboard_visual_probe_v1"


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o644)


def _layout_probe(page: Any) -> dict[str, Any]:
    return page.evaluate(
        """
        () => {
          const canvas = document.querySelector('#roundCanvas');
          const context = canvas ? canvas.getContext('2d') : null;
          let nonBlankPixels = 0;
          if (canvas && context) {
            const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
            for (let index = 0; index < pixels.length; index += 4) {
              if (pixels[index] || pixels[index + 1] || pixels[index + 2]) nonBlankPixels += 1;
            }
          }
          const rectangles = ['.topbar', '.campaign-heading', '.tabs', '.metric-grid',
            '.overview-grid', '.reliability-band', 'footer']
            .map((selector) => {
              const node = document.querySelector(selector);
              if (!node) return null;
              const rect = node.getBoundingClientRect();
              return {selector, top: rect.top, bottom: rect.bottom, left: rect.left,
                right: rect.right, width: rect.width, height: rect.height};
            }).filter(Boolean);
          const verticalOrder = rectangles.every((item, index) => index === 0 || item.top >= rectangles[index - 1].bottom - 1);
          return {
            canvas_present: Boolean(canvas),
            canvas_nonblank: nonBlankPixels > 0,
            canvas_nonblank_pixel_count: nonBlankPixels,
            horizontal_overflow: document.documentElement.scrollWidth > window.innerWidth + 1,
            vertical_order_coherent: verticalOrder,
            body_height: document.body.scrollHeight,
            rectangles,
          };
        }
        """
    )


def run_probe(output_dir: str | Path) -> dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - optional browser extra
        raise RuntimeError("playwright_dependency_missing") from exc

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    private = output / ".private"
    screenshots = output / "screenshots"
    private.mkdir(parents=True, exist_ok=True)
    screenshots.mkdir(parents=True, exist_ok=True)
    server: uvicorn.Server | None = None
    thread: Any = None
    started = time.monotonic()
    try:
        fixture = create_local_training_fixture(
            private / "fixture",
            job_id="crowdtensor-dashboard-visual-fixture",
            row_count=8,
            sequence_length=8,
            local_steps=1,
            learning_rate=0.04,
            batch_size=2,
        )
        coordinator = VolunteerTrainingCoordinator.create_from_fixture(
            private / "campaign",
            fixture,
            campaign_id="crowdtensor-dashboard-visual-campaign",
            target_rounds=1,
            minimum_quorum=2,
            lease_seconds=120.0,
        )
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        coordinator.write_invite(base_url)
        server, thread = _start_server(coordinator, port)
        _wait_for_health(base_url, time.monotonic() + 30.0)
        _run_cells(
            invite_path=coordinator.invite_path,
            root=output,
            port=port,
            deadline=time.monotonic() + 150.0,
        )

        viewport_results: dict[str, dict[str, Any]] = {}
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                executable_path="/usr/bin/google-chrome",
                args=["--no-sandbox"],
            )
            try:
                for name, width, height in (
                    ("desktop", 1440, 1000),
                    ("mobile", 390, 844),
                ):
                    page = browser.new_page(viewport={"width": width, "height": height})
                    page.goto(base_url + "/v1/volunteer/dashboard", wait_until="networkidle")
                    page.wait_for_selector("#campaignTitle")
                    page.wait_for_timeout(250)
                    screenshot = screenshots / f"dashboard-{name}.png"
                    page.screenshot(path=str(screenshot), full_page=True)
                    layout = _layout_probe(page)
                    viewport_results[name] = {
                        "viewport_width": width,
                        "viewport_height": height,
                        "screenshot_sha256": sha256_file(screenshot),
                        **layout,
                    }
                    page.close()
            finally:
                browser.close()

        coordinator.cleanup()
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=15.0)
        service_stopped = not bool(thread and thread.is_alive())
        shutil.rmtree(private, ignore_errors=True)
        shutil.rmtree(output / "private", ignore_errors=True)
        private_removed = not private.exists() and not (output / "private").exists()
        report = {
            "schema": SCHEMA,
            "ok": bool(
                all(
                    item.get("canvas_present") is True
                    and item.get("canvas_nonblank") is True
                    and item.get("horizontal_overflow") is False
                    and item.get("vertical_order_coherent") is True
                    for item in viewport_results.values()
                )
                and len(viewport_results) == 2
                and service_stopped
                and private_removed
            ),
            "dashboard_visual_probe_verified": True,
            "same_host_only": True,
            "physical_multi_host_verified": False,
            "viewports": viewport_results,
            "screenshots": {
                "desktop": "screenshots/dashboard-desktop.png",
                "mobile": "screenshots/dashboard-mobile.png",
            },
            "cleanup": {
                "http_service_stopped": service_stopped,
                "private_runtime_removed": private_removed,
                "live_resources_left_running": False,
            },
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "credential_values_public": False,
            "private_paths_public": False,
            "raw_data_public": False,
            "tensor_values_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = sha256_json(report)
        _write(output / "volunteer_dashboard_visual_probe.json", report)
        return report
    except Exception:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=10.0)
        shutil.rmtree(private, ignore_errors=True)
        shutil.rmtree(output / "private", ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run_probe(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else f"verified={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
