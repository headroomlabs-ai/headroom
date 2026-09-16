"""Playwright coverage for the dashboard CO2 savings card."""

from __future__ import annotations

import copy

import pytest

from tests.test_dashboard_cache_lifetime_playwright import _open_dashboard
from tests.test_dashboard_cache_ttl_playwright import _sample_stats

playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect
sync_playwright = playwright.sync_playwright


def test_co2_card_renders_estimated_savings() -> None:
    stats = copy.deepcopy(_sample_stats())
    stats["co2"] = {
        "co2_saved_mg": 120_000,
        "co2_saved_g": 120,
        "methodology": "EcoLogits energy model x IEA 2023 global carbon intensity",
    }

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        _open_dashboard(page, stats)

        expect(page.get_by_text("CO2 Saved", exact=True)).to_be_visible()
        expect(page.get_by_text("120.00 g", exact=True)).to_be_visible()
        expect(page.get_by_text("EcoLogits est.", exact=True)).to_be_visible()

        browser.close()
