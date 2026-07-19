#!/usr/bin/env python3
"""Probe Lightning AI account/runtime availability with public-safe output."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


SCHEMA = "lightning_gpu_provider_probe_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha16(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def parse_cookie_expiry(value: str) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    try:
        return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def load_cookie_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, dict) and isinstance(loaded.get("cookies"), list):
        rows = loaded["cookies"]
    elif isinstance(loaded, list):
        rows = loaded
    else:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            # Browser extensions commonly export: name, value, domain, path, expires.
            rows.append({
                "name": parts[0],
                "value": parts[1],
                "domain": parts[2],
                "path": parts[3],
                "expires": parts[4],
            })
    cookies: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        value = str(row.get("value") or "")
        domain = str(row.get("domain") or "").strip()
        if not name or not domain:
            continue
        cookie: dict[str, Any] = {
            "name": name,
            "value": value,
            "domain": domain,
            "path": str(row.get("path") or "/"),
            "httpOnly": bool(row.get("httpOnly", False)),
            "secure": bool(row.get("secure", True)),
            "sameSite": row.get("sameSite") if row.get("sameSite") in {"Strict", "Lax", "None"} else "Lax",
        }
        expires = parse_cookie_expiry(str(row.get("expires") or row.get("expirationDate") or row.get("expiry") or ""))
        if expires:
            cookie["expires"] = expires
        cookies.append(cookie)
    return cookies


def public_cookie_summary(cookies: list[dict[str, Any]]) -> dict[str, Any]:
    auth_name_re = re.compile(r"(auth|session|token|jwt|access|refresh|id)", re.IGNORECASE)
    return {
        "cookie_count": len(cookies),
        "domains": sorted({str(item.get("domain") or "") for item in cookies}),
        "names_hashes": sorted({sha16(str(item.get("name") or "")) for item in cookies}),
        "auth_cookie_name_signal": any(auth_name_re.search(str(item.get("name") or "")) for item in cookies),
        "values_public": False,
    }


def extract_credit_signals(text: str) -> dict[str, Any]:
    lowered = text.lower()
    credit_matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(?:free\s*)?(?:credits?|gpu\s*hours?)", lowered)
    zero_credit_visible = bool(re.search(r"\b0\s*(?:free\s*)?credits?\b", lowered) or "credits 0" in lowered or "0 credits" in lowered)
    return {
        "credit_numbers_visible": sorted(set(credit_matches))[:10],
        "zero_credit_visible": zero_credit_visible,
        "billing_text_visible": any(term in lowered for term in ["billing", "payment", "upgrade", "add credits", "buy credits"]),
        "free_text_visible": "free" in lowered,
        "gpu_text_visible": "gpu" in lowered,
        "studio_text_visible": "studio" in lowered,
    }


def public_page_summary(url: str, title: str, text: str) -> dict[str, Any]:
    return {
        "url_public": url.split("?")[0][:200],
        "title_hash": sha16(title),
        "text_chars": len(text),
        "text_hash": sha16(text),
        "credit_signals": extract_credit_signals(text),
    }


def public_action_summary(page: Any) -> list[dict[str, Any]]:
    try:
        actions = page.locator("a,button,[role=button]").evaluate_all(
            """els => els.slice(0, 80).map((el, index) => ({
                index,
                tag: el.tagName,
                role: el.getAttribute('role') || '',
                text: (el.innerText || el.textContent || '').trim().slice(0, 120),
                href: el.href || el.getAttribute('href') || '',
                disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true'
            }))"""
        )
    except Exception:
        return []
    public: list[dict[str, Any]] = []
    for item in actions if isinstance(actions, list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "")
        href = str(item.get("href") or "")
        lowered = f"{text} {href}".lower()
        public.append({
            "index": item.get("index"),
            "tag": str(item.get("tag") or ""),
            "role": str(item.get("role") or ""),
            "text_public": text[:80],
            "href_public": href.split("?")[0][:160],
            "disabled": item.get("disabled") is True,
            "create_or_start_signal": any(term in lowered for term in ["new studio", "create studio", "start", "launch", "open in studio", "run on lightning"]),
            "gpu_signal": "gpu" in lowered,
            "paid_signal": any(term in lowered for term in ["billing", "upgrade", "buy", "credits", "payment"]),
        })
    return public


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cookie_path = Path(args.cookies)
    cookies = load_cookie_rows(cookie_path)
    started = time.monotonic()
    pages: list[dict[str, Any]] = []
    blockers: list[str] = []
    logged_in = False
    studio_candidate_visible = False
    free_gpu_candidate_visible = False
    paid_or_zero_credit_blocker = False
    login_or_signup_visible = False
    action_summaries: list[dict[str, Any]] = []
    screenshot_path = ""
    storage_state_path = ""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        if cookies:
            context.add_cookies(cookies)
        page = context.new_page()
        for url in args.urls:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=int(args.navigation_timeout_ms))
                page.wait_for_timeout(int(args.settle_ms))
            except PlaywrightTimeoutError:
                blockers.append("lightning_navigation_timeout")
            title = ""
            text = ""
            try:
                title = page.title()
                text = page.locator("body").inner_text(timeout=5000)
            except Exception:
                blockers.append("lightning_page_text_unavailable")
            pages.append(public_page_summary(page.url, title, text))
            action_summaries.extend(public_action_summary(page))
            lowered = text.lower()
            if any(term in lowered for term in ["log in", "create account", "sign up"]):
                login_or_signup_visible = True
            if any(term in lowered for term in ["log out", "sign out", "account settings", "workspace settings"]):
                logged_in = True
            if any(term in lowered for term in ["new studio", "create studio", "studio"]):
                studio_candidate_visible = True
            if "gpu" in lowered and "free" in lowered:
                free_gpu_candidate_visible = True
            if extract_credit_signals(text)["zero_credit_visible"] or any(term in lowered for term in ["add credits", "buy credits"]):
                paid_or_zero_credit_blocker = True
        screenshot = output_dir / "lightning_probe_page.png"
        try:
            page.screenshot(path=str(screenshot), full_page=True)
            screenshot_path = str(screenshot)
        except Exception:
            blockers.append("lightning_screenshot_failed")
        storage = output_dir / "lightning_storage_state.private.json"
        try:
            context.storage_state(path=str(storage))
            storage_state_path = str(storage)
        except Exception:
            blockers.append("lightning_storage_state_failed")
        context.close()
        browser.close()

    if not cookies:
        blockers.append("lightning_cookies_missing_or_unparsed")
    if login_or_signup_visible:
        blockers.append("lightning_login_or_signup_visible")
        logged_in = False
    if not logged_in:
        blockers.append("lightning_login_not_verified")
    if paid_or_zero_credit_blocker:
        blockers.append("lightning_zero_or_paid_credit_signal_visible")
    if not studio_candidate_visible:
        blockers.append("lightning_studio_entry_not_verified")
    if not free_gpu_candidate_visible:
        blockers.append("lightning_free_gpu_entry_not_verified")

    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "ok": True,
        "lightning_login_verified": logged_in,
        "login_or_signup_visible": login_or_signup_visible,
        "studio_candidate_visible": studio_candidate_visible,
        "free_gpu_candidate_visible": free_gpu_candidate_visible,
        "safe_to_attempt_free_gpu_start": bool(logged_in and studio_candidate_visible and free_gpu_candidate_visible and not paid_or_zero_credit_blocker),
        "paid_or_zero_credit_blocker": paid_or_zero_credit_blocker,
        "cookie_summary": public_cookie_summary(cookies),
        "pages": pages,
        "actions": action_summaries[:120],
        "create_or_start_action_visible": any(item.get("create_or_start_signal") for item in action_summaries),
        "paid_action_visible": any(item.get("paid_signal") for item in action_summaries),
        "artifact_paths": {
            "screenshot": screenshot_path,
            "storage_state_private": storage_state_path,
        },
        "blockers": sorted(set(blockers)),
        "public_artifact_safe": True,
        "credentials_public": False,
        "cookies_public": False,
        "private_runtime_state_public": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cookies", default=str(Path.home() / "lightning-cookies.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--url", dest="urls", action="append", default=[])
    parser.add_argument("--navigation-timeout-ms", type=int, default=45000)
    parser.add_argument("--settle-ms", type=int, default=5000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.urls:
        args.urls = [
            "https://lightning.ai/",
            "https://lightning.ai/studios",
            "https://lightning.ai/me/studios",
        ]
    report = build_report(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "lightning_gpu_provider_probe.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
