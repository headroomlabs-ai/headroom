from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from headroom.dashboard import STATIC_DIR, get_dashboard_html

playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect
sync_playwright = playwright.sync_playwright


def test_dashboard_includes_active_sessions_tab() -> None:
    html = get_dashboard_html()

    assert "Active Sessions" in html
    assert "All Active Sessions" in html
    assert "activeSessionRows" in html
    assert "stats.cluster?.enabled" in html


def _active_session_stats() -> dict:
    return {
        "cost": {"savings_usd": 18.42, "compression_savings_usd": 18.42},
        "requests": {"total": 46, "cached": 0, "failed": 0, "rate_limited": 0},
        "tokens": {"input": 128_000, "output": 31_000, "saved": 84_500},
        "overhead": {},
        "ttfb": {},
        "latency": {},
        "waste_signals": {},
        "savings_history": [],
        "persistent_savings": {"display_session": {}, "lifetime": {}},
        "pipeline_timing": {},
        "compression_cache": {},
        "prefix_cache": {},
        "active_sessions": {
            "local_summary": {"count": 2},
            "local": [
                {
                    "session_id": "session-local-codex-001",
                    "instance_id": "instance-macbook-001",
                    "agent_type": "codex",
                    "age_seconds": 18,
                    "metrics": {"requests": 19, "tokens_saved": 42_300},
                },
                {
                    "session_id": "session-local-claude-002",
                    "instance_id": "instance-linux-002",
                    "agent_type": "claude-code",
                    "age_seconds": 43,
                    "metrics": {"requests": 11, "tokens_saved": 21_700},
                },
            ],
        },
        "cluster": {
            "enabled": True,
            "cluster_id": "engineering-prod",
            "summary": {"count": 1},
            "active_sessions": [
                {
                    "session_id": "session-cluster-copilot-003",
                    "instance_id": "instance-runner-003",
                    "agent_type": "copilot",
                    "age_seconds": 67,
                    "metrics": {"requests": 16, "tokens_saved": 20_500},
                }
            ],
        },
    }


def test_dashboard_renders_active_sessions_and_can_capture_screenshot() -> None:
    stats = _active_session_stats()
    dashboard_html = get_dashboard_html()
    artifact_dir = os.environ.get("PLAYWRIGHT_ARTIFACT_DIR")

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1600, "height": 1000},
            color_scheme="dark",
            ignore_https_errors=True,
        )
        page = context.new_page()

        def handler(route) -> None:  # type: ignore[no-untyped-def]
            path = urlsplit(route.request.url).path
            if path in ("/dashboard", "/"):
                route.fulfill(status=200, content_type="text/html", body=dashboard_html)
                return
            if path.startswith("/dashboard/static/"):
                route.fulfill(
                    status=200,
                    content_type="text/javascript",
                    body=(STATIC_DIR / Path(path).name).read_bytes(),
                )
                return
            if "/stats-history" in path:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"history": [], "series": {}, "lifetime": {}}),
                )
                return
            if path.endswith("/stats"):
                route.fulfill(status=200, content_type="application/json", body=json.dumps(stats))
                return
            if path.endswith("/health"):
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "healthy", "version": "0.3.0"}),
                )
                return
            route.continue_()

        page.route("**/*", handler)
        page.goto("http://headroom.local/dashboard", wait_until="networkidle")
        page.get_by_role("button", name="Active Sessions").click()

        expect(page.get_by_text("All Active Sessions", exact=True)).to_be_visible()
        expect(page.get_by_text("2", exact=True).first).to_be_visible()
        expect(page.get_by_text("engineering-prod", exact=True)).to_be_visible()
        expect(page.get_by_text("Enabled", exact=True)).to_be_visible()
        expect(page.get_by_text("3 visible", exact=True)).to_be_visible()
        expect(page.get_by_text("codex", exact=True)).to_be_visible()
        expect(page.get_by_text("claude-code", exact=True)).to_be_visible()
        expect(page.get_by_text("copilot", exact=True)).to_be_visible()

        if artifact_dir:
            screenshot_path = Path(artifact_dir) / "pr-563-active-sessions-dashboard.png"
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot_path), full_page=True)

        context.close()
        browser.close()
