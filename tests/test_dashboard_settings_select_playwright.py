"""The settings page's enum selects show the effective value.

Each select's ``<option>`` elements come from an ``x-for`` nested inside the
select, so they do not exist yet when ``x-model`` first writes the value.
Without a per-option ``:selected`` binding the browser falls back to the first
option: a proxy running ``mode=cache``/``savings_profile=coding`` displayed
``token``/``agent-90``, the first entry of each choices tuple.
"""

from __future__ import annotations

import json
from urllib.parse import urlsplit

import pytest

from headroom.dashboard import get_settings_html
from tests.test_dashboard_cache_ttl_playwright import _fulfill_static_asset

playwright = pytest.importorskip("playwright.sync_api")
Page = playwright.Page
expect = playwright.expect
sync_playwright = playwright.sync_playwright


def _enum_field(key: str, label: str, choices: list[str], default: str) -> dict:
    return {
        "key": key,
        "env": f"HEADROOM_{key.upper()}",
        "label": label,
        "group": "Compression",
        "type": "enum",
        "choices": choices,
        "default": default,
        "help": "",
        "secret": False,
        "manifest_managed": False,
        "minimum": None,
        "maximum": None,
        "tier": "basic",
        "env_override": False,
        "runtime_override": False,
        "stored": None,
    }


# Effective values deliberately differ from the first choice of each field.
_SCHEMA = {
    "groups": ["Compression"],
    "fields": [
        _enum_field("mode", "Proxy mode", ["token", "cache"], "cache"),
        _enum_field(
            "savings_profile",
            "Savings profile",
            ["agent-90", "balanced", "coding", "general"],
            "coding",
        ),
    ],
    "values": {"mode": "cache", "savings_profile": "coding"},
    "supervised": False,
}


def _open_settings(page: Page) -> None:  # type: ignore[valid-type]
    settings_html = get_settings_html()

    def handler(route) -> None:  # type: ignore[no-untyped-def]
        path = urlsplit(route.request.url).path
        if path == "/settings/ui":
            route.fulfill(status=200, content_type="text/html", body=settings_html)
            return
        if _fulfill_static_asset(route, path):
            return
        if path == "/settings/schema":
            route.fulfill(status=200, content_type="application/json", body=json.dumps(_SCHEMA))
            return
        route.fulfill(status=404, body="")

    page.route("**/*", handler)
    page.goto("http://headroom.local/settings/ui")
    page.wait_for_load_state("networkidle")


def _select_for(page: Page, label: str):  # type: ignore[no-untyped-def,valid-type]
    return page.locator("label", has_text=label).locator("select")


def test_enum_selects_display_the_effective_value() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open_settings(page)

        expect(_select_for(page, "Proxy mode")).to_have_value("cache")
        expect(_select_for(page, "Savings profile")).to_have_value("coding")
        browser.close()


def test_choosing_an_option_still_updates_the_model() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open_settings(page)

        _select_for(page, "Savings profile").select_option("balanced")

        values = page.evaluate("() => Alpine.$data(document.body).values")
        assert values["savings_profile"] == "balanced"
        assert values["mode"] == "cache"
        browser.close()
