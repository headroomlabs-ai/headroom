"""Behavior-driven Playwright validation for the Compression vs Cache panel.

The /stats endpoint has long exposed ``prefix_cache.compression_vs_cache``
and ``prefix_cache.prefix_freeze`` (built in ``headroom/proxy/cost.py``)
but the dashboard never rendered them. These tests pin the new section:
net tokens saved by compression against cached-prefix tokens its mutations
invalidated, plus the prefix-freeze net benefit.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from headroom.dashboard import get_dashboard_html
from tests.test_dashboard_cache_ttl_playwright import (
    _fulfill_static_asset,
    _sample_history,
    _sample_stats,
)

playwright = pytest.importorskip("playwright.sync_api")
Page = playwright.Page
expect = playwright.expect
sync_playwright = playwright.sync_playwright


def _stats_with_compression_vs_cache() -> dict:
    stats = copy.deepcopy(_sample_stats())
    stats["prefix_cache"]["compression_vs_cache"] = {
        "tokens_saved_by_compression": 143_000,
        "tokens_lost_to_cache_bust": 36_000,
        "cache_bust_count": 4,
        "net_tokens": 107_000,
    }
    stats["prefix_cache"]["prefix_freeze"] = {
        "busts_avoided": 6,
        "tokens_preserved": 88_000,
        "compression_foregone_tokens": 21_000,
        "net_benefit_tokens": 67_000,
    }
    return stats


def _install_dashboard_routes(page: Page, stats: dict) -> None:
    history = _sample_history()
    health = {"status": "healthy", "version": "0.3.0"}
    dashboard_html = get_dashboard_html()

    def handler(route) -> None:  # type: ignore[no-untyped-def]
        # Match on the URL path only: the dashboard fetches /stats?cached=1,
        # so suffix checks against the full URL miss it and the request
        # escapes the harness to the real network.
        path = urlsplit(route.request.url).path
        if path in ("/dashboard", "/"):
            route.fulfill(status=200, content_type="text/html", body=dashboard_html)
            return
        if _fulfill_static_asset(route, path):
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


def test_dashboard_renders_compression_vs_cache_net_metrics(tmp_path: Path) -> None:
    configured_artifact_dir = os.environ.get("HEADROOM_PLAYWRIGHT_ARTIFACT_DIR", "").strip()
    artifact_dir = (
        Path(configured_artifact_dir) if configured_artifact_dir else tmp_path / "artifacts"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1600})
        _open_dashboard(page, _stats_with_compression_vs_cache())

        expect(page.get_by_text("Compression vs Cache", exact=True)).to_be_visible()
        expect(page.get_by_test_id("cvc-net-headline")).to_have_text("Net positive")
        expect(page.get_by_test_id("cvc-saved-value")).to_have_text("143.0k")
        expect(page.get_by_test_id("cvc-bust-value")).to_have_text("36.0k")
        expect(page.get_by_text("4 busts observed")).to_be_visible()
        expect(page.get_by_test_id("cvc-net-value")).to_have_text("+107.0k")
        expect(page.get_by_test_id("cvc-net-value")).to_have_class(
            "mt-2 text-3xl font-light text-emerald-400"
        )
        expect(page.get_by_test_id("freeze-net-value")).to_have_text("+67.0k")
        expect(page.get_by_text("6 busts avoided, 21.0k foregone")).to_be_visible()
        expect(page.get_by_text("Savings Over Time", exact=True)).to_be_visible()
        expect(page.get_by_text("143.0k tokens total", exact=True)).to_have_count(0)
        token_usage = page.get_by_test_id("token-usage")
        expect(token_usage.get_by_text("Proxy Removed", exact=True)).to_be_visible()
        expect(token_usage.get_by_text("91.0k", exact=True)).to_be_visible()
        expect(page.get_by_test_id("proxy-share-of-total")).to_have_text(
            "23.5% of before-compression tokens"
        )

        artifact_dir.mkdir(parents=True, exist_ok=True)
        page.screenshot(
            path=str(artifact_dir / "dashboard-compression-vs-cache.png"),
            full_page=True,
        )

        browser.close()


def test_dashboard_marks_negative_compression_vs_cache_net_in_red() -> None:
    stats = _stats_with_compression_vs_cache()
    stats["prefix_cache"]["compression_vs_cache"] = {
        "tokens_saved_by_compression": 12_000,
        "tokens_lost_to_cache_bust": 48_000,
        "cache_bust_count": 9,
        "net_tokens": -36_000,
    }

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1600})
        _open_dashboard(page, stats)

        expect(page.get_by_test_id("cvc-net-headline")).to_have_text("Net negative")
        expect(page.get_by_test_id("cvc-net-value")).to_have_text("-36.0k")
        expect(page.get_by_test_id("cvc-net-value")).to_have_class(
            "mt-2 text-3xl font-light text-red-400"
        )

        browser.close()


def test_dashboard_hides_compression_vs_cache_section_without_data() -> None:
    stats = copy.deepcopy(_sample_stats())
    stats["prefix_cache"]["compression_vs_cache"] = {
        "tokens_saved_by_compression": 0,
        "tokens_lost_to_cache_bust": 0,
        "cache_bust_count": 0,
        "net_tokens": 0,
    }
    stats["prefix_cache"]["prefix_freeze"] = {
        "busts_avoided": 0,
        "tokens_preserved": 0,
        "compression_foregone_tokens": 0,
        "net_benefit_tokens": 0,
    }

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1600})
        _open_dashboard(page, stats)

        expect(page.get_by_test_id("cvc-net-headline")).to_have_count(0)

        browser.close()


def test_dashboard_canonical_metric_homes_at_responsive_viewports(tmp_path: Path) -> None:
    configured_artifact_dir = os.environ.get("HEADROOM_PLAYWRIGHT_ARTIFACT_DIR", "").strip()
    artifact_dir = (
        Path(configured_artifact_dir)
        if configured_artifact_dir
        else tmp_path / "responsive-artifacts"
    )

    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for theme in ("light", "dark"):
            for width in (1280, 768, 400):
                page = browser.new_page(
                    viewport={"width": width, "height": 1400}, color_scheme=theme
                )
                _open_dashboard(page, _stats_with_compression_vs_cache())
                session_metrics = page.get_by_test_id("session-optimization")
                expect(session_metrics.get_by_text("Tokens Saved", exact=True)).to_be_visible()
                expect(session_metrics.get_by_text("143.0k", exact=True)).to_be_visible()
                expect(page.get_by_test_id("proxy-share-of-total")).to_have_text(
                    "23.5% of before-compression tokens"
                )
                expect(
                    page.get_by_test_id("token-usage").get_by_text("91.0k", exact=True)
                ).to_be_visible()
                performance = page.get_by_text("Performance", exact=True).locator("..")
                expect(performance.get_by_text("Overhead Range", exact=True)).to_be_visible()
                expect(performance.get_by_text("5 - 43ms", exact=True)).to_be_visible()
                expect(performance.get_by_text("TTFB Range", exact=True)).to_be_visible()
                expect(performance.get_by_text("0.42 - 2.90s", exact=True)).to_be_visible()
                page.get_by_test_id("session-view").screenshot(
                    path=str(Path(artifact_dir) / f"dashboard-session-{theme}-{width}.png")
                )
                ttl_metrics = page.get_by_test_id("ttl-metrics")
                expect(ttl_metrics.get_by_text("Bucket Mix", exact=True)).to_be_visible()
                expect(ttl_metrics.get_by_test_id("ttl-bucket-mix-1h-total")).to_have_text("235.0k")
                expect(ttl_metrics.get_by_test_id("ttl-bucket-mix-5m-total")).to_have_text("185.0k")
                expect(ttl_metrics.get_by_test_id("ttl-bucket-1h-value")).to_have_count(0)
                expect(ttl_metrics.get_by_test_id("ttl-bucket-5m-value")).to_have_count(0)
                expect(ttl_metrics.get_by_test_id("ttl-bucket-1h-requests")).to_have_text("24")
                expect(ttl_metrics.get_by_test_id("ttl-bucket-5m-requests")).to_have_text("18")
                ttl_metrics.screenshot(
                    path=str(Path(artifact_dir) / f"dashboard-ttl-{theme}-{width}.png")
                )
                page.get_by_role("button", name="Historical", exact=True).click()
                history_summary = page.get_by_test_id("history-summary")
                expect(
                    history_summary.get_by_text("Lifetime Tokens Saved", exact=True)
                ).to_be_visible()
                expect(page.get_by_test_id("history-total-value")).to_have_text("143.0k")
                expect(page.get_by_test_id("history-total-value")).to_have_count(1)
                expect(page.get_by_text("Latest total", exact=True)).to_have_count(0)
                expect(
                    page.get_by_text("Recent Historical Checkpoints", exact=True)
                ).to_be_visible()
                historical_card = page.get_by_test_id("historical-summary-card")
                expect(historical_card).to_be_visible()
                historical_checkpoints = page.get_by_text(
                    "Recent Historical Checkpoints", exact=True
                ).locator("..")
                expect(historical_checkpoints).to_be_visible()
                historical_checkpoints.screenshot(
                    path=str(Path(artifact_dir) / f"dashboard-history-{theme}-{width}.png")
                )
                page.close()

        browser.close()
