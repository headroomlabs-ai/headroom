"""Playwright validation for the persisted lifetime Cache Reads tile.

The Prefix Cache Impact card historically rendered only from in-memory
session counters, so every proxy restart blanked the operator's cache
savings. These tests pin the durable behavior: the card renders from
``persistent_savings.lifetime.cache_read_tokens`` alone after a restart
with zero traffic, session-scoped tiles read "no activity since restart",
and the card stays hidden when neither session nor lifetime data exists.
"""

from __future__ import annotations

import copy
import json
from urllib.parse import urlsplit

import pytest

from headroom.dashboard import get_dashboard_html
from tests.test_dashboard_cache_ttl_playwright import _sample_history, _sample_stats

playwright = pytest.importorskip("playwright.sync_api")
Page = playwright.Page
expect = playwright.expect
sync_playwright = playwright.sync_playwright


def _stats_lifetime_only() -> dict:
    """Post-restart shape: zero session cache traffic, persisted lifetime present."""
    stats = copy.deepcopy(_sample_stats())
    totals = stats.setdefault("prefix_cache", {}).setdefault("totals", {})
    totals.update({"requests": 0, "cache_read_tokens": 0, "cache_write_tokens": 0})
    stats.setdefault("persistent_savings", {})["lifetime"] = {
        "requests": 6088,
        "tokens_saved": 42_181,
        "compression_savings_usd": 0.5,
        "cache_read_tokens": 629_537_547,
        "cache_savings_usd": 7.2,
        "total_input_tokens": 1_294_591_655,
        "total_input_cost_usd": 12.5,
    }
    return stats


def _install_dashboard_routes(page: Page, stats: dict) -> None:
    history = _sample_history()
    health = {"status": "healthy", "version": "0.3.0"}
    dashboard_html = get_dashboard_html()

    def handler(route) -> None:  # type: ignore[no-untyped-def]
        path = urlsplit(route.request.url).path
        if path in ("/dashboard", "/"):
            route.fulfill(status=200, content_type="text/html", body=dashboard_html)
            return
        if "/stats-history" in path:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(history))
            return
        if path.endswith("/stats"):
            route.fulfill(status=200, content_type="application/json", body=json.dumps(stats))
            return
        if path.endswith("/health"):
            route.fulfill(status=200, content_type="application/json", body=json.dumps(health))
            return
        route.continue_()

    page.route("**/*", handler)


def _open_dashboard(page: Page, stats: dict) -> None:
    _install_dashboard_routes(page, stats)
    page.goto("http://headroom.local/dashboard")
    page.wait_for_load_state("networkidle")


def test_card_renders_lifetime_cache_reads_after_zero_traffic_restart() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1600})
        _open_dashboard(page, _stats_lifetime_only())

        expect(page.get_by_text("Prefix Cache Impact", exact=True)).to_be_visible()
        expect(page.get_by_text("Cache Reads (lifetime)", exact=True)).to_be_visible()
        expect(page.get_by_text("629.5M", exact=True)).to_be_visible()
        expect(page.get_by_text("$7.20 saved")).to_be_visible()
        # Session-scoped siblings read as inactive, not as literal zeros.
        expect(page.get_by_text("no activity since restart").first).to_be_visible()
        assert page.get_by_text("no activity since restart").count() >= 5
        # x-show hides via CSS (element stays in the DOM), so assert
        # visibility, not count — unlike the x-if card gate below.
        expect(page.get_by_text("Cache Efficiency", exact=True)).to_be_hidden()

        browser.close()


def test_card_hidden_when_no_session_and_no_lifetime_data() -> None:
    stats = _stats_lifetime_only()
    stats["persistent_savings"]["lifetime"]["cache_read_tokens"] = 0
    stats["persistent_savings"]["lifetime"]["cache_savings_usd"] = 0.0

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1600})
        _open_dashboard(page, stats)

        expect(page.get_by_text("Prefix Cache Impact", exact=True)).to_have_count(0)

        browser.close()
