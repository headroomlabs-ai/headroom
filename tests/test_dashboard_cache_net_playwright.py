"""Behavior-driven Playwright validation for the Compression vs Cache panel.

The /stats endpoint has long exposed ``prefix_cache.compression_vs_cache``
and ``prefix_cache.prefix_freeze`` (built in ``headroom/proxy/cost.py``)
but the dashboard never rendered them. These tests pin the new section:
net tokens saved by compression against cached-prefix tokens its mutations
invalidated, plus the prefix-freeze net benefit.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import pytest

from headroom.dashboard import get_dashboard_html
from headroom.proxy.cost import CostTracker, build_prefix_cache_stats
from headroom.proxy.prometheus_metrics import PrometheusMetrics
from headroom.proxy.savings_tracker import SavingsTracker, _empty_display_session
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


def _neutral_stats() -> dict:
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
        "prefix_cache": {"by_provider": {}, "totals": {}},
    }


def _openai_fixture_stats(
    *,
    cache_read_tokens: int = 99_000,
    tokens_saved: int = 25_000,
    include_current_process: bool = True,
    include_rolling: bool = True,
    reload_tracker: bool = False,
) -> dict:
    metrics = PrometheusMetrics(stateless=True)
    cost_tracker = CostTracker()
    if include_current_process:
        for _ in range(40):
            asyncio.run(
                metrics.record_request(
                    "openai",
                    "deepseek-v4-flash",
                    100_000,
                    0,
                    0,
                    0,
                    cached=True,
                    cache_read_tokens=99_000,
                    uncached_input_tokens=1_000,
                )
            )
        for _ in range(40):
            cost_tracker.record_tokens(
                model="deepseek-v4-flash",
                tokens_saved=0,
                tokens_sent=100_000,
                cache_read_tokens=99_000,
                uncached_tokens=1_000,
            )
    prefix_cache = build_prefix_cache_stats(metrics, cost_tracker)
    assert set(prefix_cache["by_provider"]) == ({"openai"} if include_current_process else set())
    assert prefix_cache["totals"]["requests"] == (40 if include_current_process else 0)
    if include_current_process:
        assert prefix_cache["totals"]["cache_read_tokens"] == 3_960_000
        assert prefix_cache["totals"]["uncached_input_tokens"] == 40_000
        assert prefix_cache["totals"]["hit_rate"] == 99.0
    assert prefix_cache["totals"]["cache_write_tokens"] == 0
    assert prefix_cache["totals"]["net_savings_usd"] == 0.0

    with TemporaryDirectory(prefix="headroom-960-2-") as directory:
        tracker_path = str(Path(directory) / "savings.json")
        tracker = SavingsTracker(path=tracker_path, stateless=False)
        if include_rolling:
            for _ in range(40):
                tracker.record_request(
                    model="deepseek-v4-flash",
                    provider="openai",
                    input_tokens=100_000,
                    tokens_saved=tokens_saved,
                    cache_read_tokens=cache_read_tokens,
                    uncached_input_tokens=100_000 - cache_read_tokens,
                    total_input_tokens=100_000,
                )
        if reload_tracker:
            tracker = SavingsTracker(path=tracker_path, stateless=False)
        display_session = tracker.stats_preview()["display_session"]
        assert display_session["requests"] == (40 if include_rolling else 0)
        assert display_session["cache_read_tokens"] == (
            40 * cache_read_tokens if include_rolling else 0
        )
        assert display_session["total_input_tokens"] == (4_000_000 if include_rolling else 0)
        if not include_rolling:
            assert display_session == _empty_display_session()

    stats = _neutral_stats()
    stats["prefix_cache"] = prefix_cache
    stats["persistent_savings"] = {"display_session": display_session, "lifetime": {}}
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


def test_dashboard_renders_compression_vs_cache_net_metrics() -> None:
    artifact_dir = os.environ.get("HEADROOM_PLAYWRIGHT_ARTIFACT_DIR")

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

        if artifact_dir:
            Path(artifact_dir).mkdir(parents=True, exist_ok=True)
            page.screenshot(
                path=str(Path(artifact_dir) / "dashboard-compression-vs-cache.png"),
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


def test_dashboard_preserves_negative_prefix_freeze_diagnostic_in_red() -> None:
    stats = _stats_with_compression_vs_cache()
    stats["prefix_cache"]["prefix_freeze"] = {
        "busts_avoided": 1,
        "tokens_preserved": 12_000,
        "compression_foregone_tokens": 48_000,
        "net_benefit_tokens": -36_000,
    }

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1600})
        _open_dashboard(page, stats)
        expect(page.get_by_test_id("freeze-net-value")).to_have_text("-36.0k")
        expect(page.get_by_test_id("freeze-net-value")).to_have_class(
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


def test_dashboard_separates_openai_zero_current_process_from_rolling_economics() -> None:
    stats = _openai_fixture_stats()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1600})
        _open_dashboard(page, stats)

        expect(page.get_by_text("Rolling display session economics", exact=True)).to_be_visible()
        session = stats["persistent_savings"]["display_session"]
        expect(page.get_by_test_id("rolling-provider-cache")).to_contain_text(
            f"${session['cache_savings_usd']:.2f}"
        )
        expect(page.get_by_test_id("rolling-compression-savings")).to_contain_text(
            f"${session['compression_savings_usd']:.2f}"
        )
        expect(
            page.get_by_text("Provider cache discount, current process: $0.00", exact=True)
        ).to_be_visible()
        expect(page.get_by_text("openai", exact=True)).to_be_visible()
        expect(page.get_by_text("Net savings", exact=True)).to_have_count(0)
        rendered_values = page.locator(
            ".rolling-provider-value, .rolling-compression-value"
        ).all_text_contents()
        numeric_values = [float(value.removeprefix("$")) for value in rendered_values]
        rendered_sum = f"${sum(numeric_values):.2f}"
        assert rendered_sum not in page.locator("body").inner_text()
        browser.close()


def test_dashboard_shows_persisted_rolling_session_without_current_process_cache() -> None:
    stats = _openai_fixture_stats(include_current_process=False, reload_tracker=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1600})
        _open_dashboard(page, stats)
        expect(page.get_by_test_id("rolling-display-session-economics")).to_be_visible()
        expect(page.get_by_test_id("rolling-provider-cache")).to_contain_text(
            f"${stats['persistent_savings']['display_session']['cache_savings_usd']:.2f}"
        )
        expect(page.get_by_text("Prefix Cache Impact", exact=True)).to_have_count(0)
        browser.close()


def test_dashboard_keeps_current_process_card_when_rolling_session_is_empty() -> None:
    stats = _openai_fixture_stats(include_rolling=False)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1600})
        _open_dashboard(page, stats)
        expect(page.get_by_test_id("rolling-display-session-economics")).to_have_count(0)
        expect(page.get_by_text("Prefix Cache Impact", exact=True)).to_be_visible()
        expect(
            page.get_by_text("Provider cache discount, current process: $0.00", exact=True)
        ).to_be_visible()
        browser.close()


@pytest.mark.parametrize(("cache_read_tokens", "tokens_saved"), [(0, 25_000), (99_000, 0)])
def test_dashboard_keeps_zero_rolling_peer_visible(
    cache_read_tokens: int, tokens_saved: int
) -> None:
    stats = _openai_fixture_stats(cache_read_tokens=cache_read_tokens, tokens_saved=tokens_saved)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 768, "height": 1600})
        _open_dashboard(page, stats)
        rolling = page.get_by_test_id("rolling-display-session-economics")
        expect(rolling).to_be_visible()
        session = stats["persistent_savings"]["display_session"]
        expect(rolling.get_by_test_id("rolling-provider-cache")).to_contain_text(
            f"${session['cache_savings_usd']:.2f}"
        )
        expect(rolling.get_by_test_id("rolling-compression-savings")).to_contain_text(
            f"${session['compression_savings_usd']:.2f}"
        )
        browser.close()


def test_dashboard_rolling_economics_layout_at_required_viewports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = os.environ.get("HEADROOM_PLAYWRIGHT_ARTIFACT_DIR")
    if artifact_dir is None:
        artifact_dir = str(Path.cwd().parent / ".claude/pr-sweep/proof/960-2-screenshots")
    monkeypatch.setenv("HEADROOM_PLAYWRIGHT_ARTIFACT_DIR", artifact_dir)
    stats = _openai_fixture_stats()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for theme in ("light", "dark"):
            for width in (1280, 768, 400):
                page = browser.new_page(viewport={"width": width, "height": 1600})
                page.emulate_media(color_scheme=theme)
                _open_dashboard(page, stats)
                rolling = page.get_by_test_id("rolling-display-session-economics")
                provider = page.get_by_test_id("rolling-provider-cache")
                compression = page.get_by_test_id("rolling-compression-savings")
                expect(rolling).to_be_visible()
                assert awaitable_width(page, rolling) <= width
                assert awaitable_width(page, provider) <= width
                assert awaitable_width(page, compression) <= width
                for locator in (
                    provider.locator(".rolling-provider-label"),
                    provider.locator(".rolling-provider-value"),
                    provider.locator(".rolling-scope-copy"),
                    compression.locator(".rolling-compression-label"),
                    compression.locator(".rolling-compression-value"),
                    compression.locator(".rolling-scope-copy"),
                ):
                    assert locator.evaluate(
                        """(element) => {
                            const rgb = getComputedStyle(element).color.match(/\\d+/g).map(Number);
                            const bg = getComputedStyle(document.documentElement).getPropertyValue('--color-surface').trim().match(/#[0-9a-f]{6}/i)[0];
                            const channels = bg.slice(1).match(/../g).map(value => parseInt(value, 16));
                            const luminance = values => values.map(value => value / 255).map(value => value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4).reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
                            const ratio = (a, b) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
                            return ratio(luminance(rgb), luminance(channels)) >= 4.5;
                        }""",
                    )
                if width == 400:
                    assert provider.bounding_box()["y"] < compression.bounding_box()["y"]
                else:
                    assert provider.bounding_box()["x"] < compression.bounding_box()["x"]
                overflow = page.evaluate(
                    """() => ({
                        bodyScrollWidth: document.body.scrollWidth,
                        bodyClientWidth: document.body.clientWidth,
                        viewportWidth: document.documentElement.clientWidth,
                        overflowing: [...document.querySelectorAll('*')]
                            .filter(element => element.getBoundingClientRect().right > window.innerWidth)
                            .slice(0, 5)
                            .map(element => [element.tagName, element.className, element.getBoundingClientRect().right])
                    })"""
                )
                assert (
                    overflow["bodyScrollWidth"] <= overflow["viewportWidth"]
                    and overflow["bodyScrollWidth"] <= overflow["bodyClientWidth"]
                ), overflow
                if artifact_dir:
                    output = Path(artifact_dir)
                    output.mkdir(parents=True, exist_ok=True)
                    page.screenshot(
                        path=str(output / f"rolling-{theme}-{width}.png"), full_page=True
                    )
                page.close()
        browser.close()


def awaitable_width(page: Page, locator) -> float:
    box = locator.bounding_box()
    assert box is not None
    return box["width"]
