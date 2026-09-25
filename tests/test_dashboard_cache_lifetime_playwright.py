"""Playwright validation for persisted lifetime Prefix Cache metrics.

The Session dashboard now reports only current-process cache activity,
while durable cache totals live behind the Lifetime view. These tests pin
that split after a restart with zero session traffic.
"""

from __future__ import annotations

import json
from urllib.parse import urlsplit

import pytest

from headroom.dashboard import get_dashboard_html
from headroom.proxy.savings_tracker import _empty_display_session
from tests.test_dashboard_cache_ttl_playwright import (
    _fulfill_static_asset,
    _sample_history,
)

playwright = pytest.importorskip("playwright.sync_api")
Page = playwright.Page
expect = playwright.expect
sync_playwright = playwright.sync_playwright


def _session_stats_no_cache() -> dict:
    """Post-restart session payload: zero current-process cache traffic."""
    return {
        "cost": {},
        "requests": {},
        "tokens": {},
        "overhead": {},
        "ttfb": {},
        "latency": {},
        "waste_signals": {},
        "savings_history": [],
        "persistent_savings": {"display_session": {}, "lifetime": {}},
        "pipeline_timing": {},
        "compression_cache": {},
        "prefix_cache": {
            "by_provider": {},
            "totals": {
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "requests": 0,
                "hit_requests": 0,
                "observed_ttl_buckets": {},
                "observed_ttl_mix": {"active_buckets": []},
            },
        },
    }


def _lifetime_cache_payload(
    *,
    cache_read_tokens: int = 629_537_547,
    cache_savings_usd: float = 7.2,
) -> dict:
    return {
        "requests": {"total": 6088, "failed": 0, "rate_limited": 0, "cached": 0},
        "tokens": {"saved": 42_181},
        "cost": {
            "input_usd": 12.5,
            "compression_savings_usd": 0.5,
            "cache_savings_usd": cache_savings_usd,
        },
        "prefix_cache": {
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": 0,
            "hit_requests": 0,
            "requests": 0,
            "bust_count": 0,
            "bust_tokens": 0,
        },
        "projects": {},
        "by_model": {},
        "by_stack": {},
        "waste_signals": {},
        "persistence": {"healthy": True, "enabled": True, "pending_records": 0},
    }


def _install_dashboard_routes(page: Page, stats: dict, lifetime: dict) -> None:
    history = _sample_history()
    health = {"status": "healthy", "version": "0.3.0"}
    dashboard_html = get_dashboard_html()

    def handler(route) -> None:  # type: ignore[no-untyped-def]
        path = urlsplit(route.request.url).path
        if path in ("/dashboard", "/"):
            route.fulfill(status=200, content_type="text/html", body=dashboard_html)
            return
        if _fulfill_static_asset(route, path):
            return
        if "/stats-history" in path:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(history))
            return
        if path.endswith("/stats-lifetime"):
            route.fulfill(status=200, content_type="application/json", body=json.dumps(lifetime))
            return
        if path.endswith("/stats"):
            route.fulfill(status=200, content_type="application/json", body=json.dumps(stats))
            return
        if path.endswith("/health"):
            route.fulfill(status=200, content_type="application/json", body=json.dumps(health))
            return
        route.continue_()

    page.route("**/*", handler)


def _open_dashboard(page: Page, stats: dict, lifetime: dict) -> None:
    _install_dashboard_routes(page, stats, lifetime)
    page.goto("http://headroom.local/dashboard")
    page.wait_for_load_state("networkidle")


def test_card_renders_lifetime_cache_reads_after_zero_traffic_restart() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1600})
        _open_dashboard(page, _session_stats_no_cache(), _lifetime_cache_payload())

        expect(page.get_by_text("Prefix Cache Impact", exact=True)).to_have_count(0)
        page.get_by_role("button", name="Lifetime", exact=True).click()
        expect(page.get_by_role("heading", name="Prefix Cache", exact=True)).to_be_visible()
        expect(page.get_by_text("629.5M / 0", exact=True)).to_be_visible()
        expect(page.get_by_text("$7.20", exact=True)).to_be_visible()

        browser.close()


def test_card_hidden_when_no_session_and_no_lifetime_data() -> None:
    stats = _session_stats_no_cache()
    lifetime = _lifetime_cache_payload(cache_read_tokens=0, cache_savings_usd=0.0)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1600})
        _open_dashboard(page, stats, lifetime)

        expect(page.get_by_text("Prefix Cache Impact", exact=True)).to_have_count(0)

        browser.close()


def test_expired_display_session_preserves_session_lifetime_fallback() -> None:
    stats = _session_stats_no_cache()
    stats["persistent_savings"]["display_session"] = _empty_display_session()
    stats["persistent_savings"]["lifetime"] = {
        "cache_read_tokens": 629_537_547,
        "cache_savings_usd": 7.2,
    }
    lifetime = _lifetime_cache_payload()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1600})
        _open_dashboard(page, stats, lifetime)
        expect(page.get_by_test_id("rolling-display-session-economics")).to_have_count(0)
        expect(page.get_by_text("Prefix Cache Impact", exact=True)).to_be_visible()
        expect(page.get_by_text("Cache Reads (lifetime)", exact=True)).to_be_visible()
        page.get_by_role("button", name="Lifetime", exact=True).click()
        expect(page.get_by_text("$7.20", exact=True)).to_be_visible()
        browser.close()
