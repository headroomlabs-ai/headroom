"""Behavior-driven Playwright validation for dashboard TTL cache metrics."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from headroom.dashboard import STATIC_DIR, get_dashboard_html

playwright = pytest.importorskip("playwright.sync_api")
Page = playwright.Page
expect = playwright.expect
sync_playwright = playwright.sync_playwright


def _sample_stats() -> dict:
    return {
        "cost": {
            "savings_usd": 12.34,
            "compression_savings_usd": 12.34,
            "cache_savings_usd": 5.25,
        },
        "requests": {
            "total": 128,
            "cached": 96,
            "rate_limited": 0,
            "failed": 0,
            "by_provider": {"anthropic": 128},
            "by_model": {"claude-opus-4-6": 128},
        },
        "tokens": {
            "input": 245_000,
            "output": 88_000,
            "saved": 143_000,
            "proxy_compression_saved": 91_000,
            "total_before_compression": 388_000,
            "savings_percent": 36.86,
        },
        "overhead": {"average_ms": 14.2, "min_ms": 4.5, "max_ms": 42.7},
        "ttfb": {"average_ms": 1320.0, "min_ms": 420.0, "max_ms": 2900.0},
        "latency": {"average_ms": 1510.0, "min_ms": 520.0, "max_ms": 3300.0},
        "waste_signals": {"json_bloat": 95_000, "repetition": 48_000},
        "savings_history": [
            ["2026-04-01T00:00:00Z", 12_000],
            ["2026-04-02T00:00:00Z", 38_000],
            ["2026-04-03T00:00:00Z", 57_000],
            ["2026-04-04T00:00:00Z", 102_000],
            ["2026-04-05T00:00:00Z", 143_000],
        ],
        "persistent_savings": {
            "display_session": {},
            "lifetime": {"tokens_saved": 143_000, "compression_savings_usd": 12.34},
        },
        "pipeline_timing": {},
        "compression_cache": {"mode": "cache"},
        "prefix_cache": {
            "by_provider": {
                "anthropic": {
                    "cache_read_tokens": 9_800_000,
                    "cache_write_tokens": 420_000,
                    "cache_write_5m_tokens": 185_000,
                    "cache_write_1h_tokens": 235_000,
                    "cache_write_5m_requests": 18,
                    "cache_write_1h_requests": 24,
                    "requests": 128,
                    "hit_requests": 96,
                    "hit_rate": 75.0,
                    "bust_count": 0,
                    "bust_write_tokens": 0,
                    "read_discount": "90%",
                    "write_premium": "25%",
                    "savings_usd": 5.67,
                    "write_premium_usd": 0.42,
                    "net_savings_usd": 5.25,
                    "label": "Explicit breakpoints, 5-min TTL",
                    "observed_ttl_buckets": {
                        "5m": {"tokens": 185_000, "requests": 18},
                        "1h": {"tokens": 235_000, "requests": 24},
                    },
                    "observed_ttl_mix": {
                        "5m_pct": 44.0,
                        "1h_pct": 56.0,
                        "active_buckets": ["5m", "1h"],
                    },
                }
            },
            "totals": {
                "cache_read_tokens": 9_800_000,
                "cache_write_tokens": 420_000,
                "cache_write_5m_tokens": 185_000,
                "cache_write_1h_tokens": 235_000,
                "cache_write_5m_requests": 18,
                "cache_write_1h_requests": 24,
                "requests": 128,
                "hit_requests": 96,
                "bust_count": 0,
                "bust_write_tokens": 0,
                "savings_usd": 5.67,
                "write_premium_usd": 0.42,
                "net_savings_usd": 5.25,
                "hit_rate": 75.0,
                "observed_ttl_buckets": {
                    "5m": {"tokens": 185_000, "requests": 18},
                    "1h": {"tokens": 235_000, "requests": 24},
                },
                "observed_ttl_mix": {
                    "5m_pct": 44.0,
                    "1h_pct": 56.0,
                    "active_buckets": ["5m", "1h"],
                },
            },
            "prefix_freeze": {
                "busts_avoided": 0,
                "tokens_preserved": 0,
                "compression_foregone_tokens": 0,
                "net_benefit_tokens": 0,
            },
            "attribution": "Observed provider TTL buckets.",
        },
    }


def _sample_history() -> dict:
    return {
        "history": [
            {
                "timestamp": "2026-04-01T00:00:00Z",
                "total_tokens_saved": 12_000,
                "compression_savings_usd": 0.6,
            },
            {
                "timestamp": "2026-04-05T00:00:00Z",
                "total_tokens_saved": 143_000,
                "compression_savings_usd": 12.34,
            },
        ],
        "series": {
            "daily": [
                {
                    "timestamp": "2026-04-05T00:00:00Z",
                    "tokens_saved": 20_000,
                    "total_tokens_saved": 143_000,
                    "compression_savings_usd_delta": 1.7,
                }
            ],
            "weekly": [],
            "monthly": [],
        },
        "lifetime": {"tokens_saved": 143_000, "compression_savings_usd": 12.34},
    }


def _fulfill_static_asset(route, path: str) -> bool:  # type: ignore[no-untyped-def]
    """Serve the vendored dashboard JS from disk; True when it handled the route.

    The harnesses in this file and its siblings intercept every request, so the
    dashboard's relative asset URLs would otherwise fall through to a real fetch
    against a fake origin with nothing listening. Alpine has to load: every
    section of <main> lives inside a `<template x-if>`, which renders nothing at
    all without it, so an empty <main> is the symptom to look for here.
    """
    if not path.startswith("/dashboard/static/"):
        return False
    route.fulfill(
        status=200,
        content_type="text/javascript",
        body=(STATIC_DIR / Path(path).name).read_bytes(),
    )
    return True


def _install_dashboard_routes(page: Page, stats: dict | None = None) -> None:
    stats = copy.deepcopy(stats) if stats is not None else _sample_stats()
    history = _sample_history()
    lifetime = {"projects": {}}
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
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(history),
            )
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


def test_dashboard_per_project_setup_url_uses_current_origin() -> None:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1720, "height": 1400}, color_scheme="dark")
        _install_dashboard_routes(page)

        page.goto("http://127.0.0.1:8788/dashboard", wait_until="load")
        page.get_by_role("button", name="Lifetime", exact=True).click()
        expect(
            page.get_by_text(
                "ANTHROPIC_BASE_URL: http://127.0.0.1:8788/p/<project-name>", exact=True
            )
        ).to_be_visible()
        expect(
            page.get_by_text(
                "ANTHROPIC_BASE_URL: http://127.0.0.1:8787/p/<project-name>", exact=True
            )
        ).to_have_count(0)

        page.goto("http://headroom.local:9393/dashboard", wait_until="load")
        page.get_by_role("button", name="Lifetime", exact=True).click()
        expect(
            page.get_by_text(
                "ANTHROPIC_BASE_URL: http://headroom.local:9393/p/<project-name>", exact=True
            )
        ).to_be_visible()
        expect(
            page.get_by_text(
                "ANTHROPIC_BASE_URL: http://127.0.0.1:8787/p/<project-name>", exact=True
            )
        ).to_have_count(0)

        browser.close()


def test_dashboard_renders_observed_ttl_metrics_and_can_capture_screenshot(tmp_path: Path) -> None:
    configured_artifact_dir = os.environ.get("HEADROOM_PLAYWRIGHT_ARTIFACT_DIR", "").strip()

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1720, "height": 1400}, color_scheme="dark")
        _install_dashboard_routes(page)
        page.goto("http://headroom.local/dashboard", wait_until="load")

        expect(page.get_by_text("Observed TTL Buckets")).to_be_visible()
        expect(page.get_by_text("Provider-reported cache write mix")).to_be_visible()
        expect(page.get_by_test_id("ttl-bucket-headline")).to_have_text("1h leaning")
        expect(page.get_by_test_id("ttl-bucket-mix-1h-pct")).to_have_text("1h 56.0%")
        expect(page.get_by_test_id("ttl-bucket-mix-5m-pct")).to_have_text("5m 44.0%")
        expect(page.get_by_test_id("ttl-bucket-mix-1h-total")).to_have_text("235.0k")
        expect(page.get_by_test_id("ttl-bucket-mix-5m-total")).to_have_text("185.0k")
        expect(page.get_by_test_id("ttl-bucket-1h-requests")).to_have_text("24")
        expect(page.get_by_test_id("ttl-bucket-5m-requests")).to_have_text("18")
        expect(page.get_by_text("TTL 1h 56.0% / 5m 44.0%")).to_be_visible()

        screenshot_path = (
            Path(configured_artifact_dir) / "dashboard-cache-ttl-main.png"
            if configured_artifact_dir
            else tmp_path / "dashboard-cache-ttl-main.png"
        )
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshot_path), full_page=True)
        browser.close()


def test_dashboard_renders_ttl_request_counts_when_token_totals_are_zero() -> None:
    stats = _sample_stats()
    buckets = stats["prefix_cache"]["totals"]["observed_ttl_buckets"]
    buckets["5m"]["tokens"] = 0
    buckets["1h"]["tokens"] = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1400})
        _install_dashboard_routes(page, stats)
        page.goto("http://headroom.local/dashboard", wait_until="load")

        ttl_metrics = page.get_by_test_id("ttl-metrics")
        expect(ttl_metrics).to_be_visible()
        expect(ttl_metrics.get_by_test_id("ttl-bucket-mix-1h-total")).to_have_text("235.0k")
        expect(ttl_metrics.get_by_test_id("ttl-bucket-mix-5m-total")).to_have_text("185.0k")
        expect(ttl_metrics.get_by_test_id("ttl-bucket-1h-requests")).to_have_text("24")
        expect(ttl_metrics.get_by_test_id("ttl-bucket-5m-requests")).to_have_text("18")

        browser.close()


def test_dashboard_hides_ttl_metrics_when_request_counts_are_zero() -> None:
    stats = _sample_stats()
    buckets = stats["prefix_cache"]["totals"]["observed_ttl_buckets"]
    for bucket in buckets.values():
        bucket["requests"] = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1400})
        _install_dashboard_routes(page, stats)
        page.goto("http://headroom.local/dashboard", wait_until="load")

        expect(page.get_by_test_id("ttl-metrics")).to_have_count(0)

        browser.close()
