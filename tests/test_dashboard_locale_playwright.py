"""Browser coverage for the dashboard locale selector."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.test_dashboard_cache_net_playwright import (
    _install_dashboard_routes,
    _stats_with_compression_vs_cache,
)

playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect
sync_playwright = playwright.sync_playwright


def test_zh_locale_tracks_live_alpine_text_and_can_switch_back() -> None:
    artifact_dir = os.environ.get("HEADROOM_PLAYWRIGHT_ARTIFACT_DIR")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        _install_dashboard_routes(page, _stats_with_compression_vs_cache())
        page.goto("http://headroom.local/dashboard")
        page.wait_for_load_state("networkidle")

        page.locator("select[title='Dashboard language']").select_option("zh-CN")
        expect(page.locator("html")).to_have_attribute("lang", "zh-CN")
        expect(page.get_by_role("button", name="本次会话")).to_be_visible()

        page.evaluate(
            """() => {
                const node = document.createElement('span');
                node.id = 'changing-stat';
                node.textContent = '0 requests processed';
                document.body.appendChild(node);
                applyHeadroomTranslations(node);
            }"""
        )
        expect(page.locator("#changing-stat")).to_have_text("已处理 0 个请求")

        page.evaluate(
            """() => {
                const node = document.querySelector('#changing-stat');
                node.textContent = '10 requests processed';
                applyHeadroomTranslations(node);
            }"""
        )
        expect(page.locator("#changing-stat")).to_have_text("已处理 10 个请求")

        if artifact_dir:
            Path(artifact_dir).mkdir(parents=True, exist_ok=True)
            page.screenshot(
                path=str(Path(artifact_dir) / "pr-1243-dashboard-zh-cn.png"),
                full_page=True,
            )

        page.locator("select[title='仪表盘语言']").select_option("en")
        expect(page.locator("html")).to_have_attribute("lang", "en")
        expect(page.locator("#changing-stat")).to_have_text("10 requests processed")
        browser.close()
