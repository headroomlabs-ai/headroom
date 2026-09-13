"""Keyboard operability and accessible-name invariants for the operator dashboard.

The static tests run everywhere and pin the template contract: every control
reachable by mouse is reachable by keyboard, every icon-only control has an
accessible name, charts have text alternatives, and keyboard focus is visible.
The Playwright tests drive a real browser through the same workflows.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from headroom.dashboard import get_dashboard_html

TEMPLATES = Path(__file__).resolve().parents[1] / "headroom" / "dashboard" / "templates"
DASHBOARD = (TEMPLATES / "dashboard.html").read_text()
SETTINGS = (TEMPLATES / "settings.html").read_text()

_TAG = re.compile(r"<(\w+)\b([^>]*)>", re.S)


def _tags(html: str, name: str) -> list[str]:
    return [attrs for tag, attrs in _TAG.findall(html) if tag == name]


@pytest.mark.parametrize("html", [DASHBOARD, SETTINGS], ids=["dashboard", "settings"])
def test_focus_visible_style_is_defined(html: str) -> None:
    assert ":focus-visible {" in html
    assert "prefers-reduced-motion" in html


@pytest.mark.parametrize("html", [DASHBOARD, SETTINGS], ids=["dashboard", "settings"])
def test_every_clickable_non_button_is_keyboard_operable(html: str) -> None:
    """A div/tr with @click must expose a role, a tabindex and an Enter handler."""
    offenders = []
    for tag, attrs in _TAG.findall(html):
        if tag in ("button", "a", "input", "select", "textarea", "label"):
            continue
        if "@click=" not in attrs and "onclick=" not in attrs:
            continue
        if "@click.away" in attrs and "@click=" not in attrs:
            continue
        ok = 'tabindex="0"' in attrs and 'role="button"' in attrs and "@keydown.enter" in attrs
        if not ok:
            offenders.append(f"<{tag} {attrs.strip()[:90]}>")
    assert not offenders, offenders


def test_icon_only_buttons_have_accessible_names() -> None:
    """Buttons whose visible content is only an SVG need aria-label."""
    for m in re.finditer(r"<button\b([^>]*)>(.*?)</button>", DASHBOARD, re.S):
        attrs, inner = m.groups()
        text = re.sub(r"<svg\b.*?</svg>", "", inner, flags=re.S)
        text = re.sub(r"<[^>]+>", "", text).strip()
        if not text and "x-text=" not in attrs and "x-text=" not in inner:
            assert "aria-label=" in attrs, (
                f"icon-only button lacks aria-label: {attrs.strip()[:80]}"
            )


def test_svgs_are_named_or_hidden() -> None:
    for attrs in _tags(DASHBOARD, "svg"):
        assert 'aria-hidden="true"' in attrs or 'role="img"' in attrs, attrs.strip()[:80]
    for attrs in _tags(DASHBOARD, "svg"):
        if 'role="img"' in attrs:
            assert "aria-label" in attrs, attrs.strip()[:80]


def test_feed_drawer_has_dialog_semantics_and_escape() -> None:
    assert 'id="feed-drawer"' in DASHBOARD
    assert 'role="dialog"' in DASHBOARD
    assert "@keydown.escape.window" in DASHBOARD
    assert 'aria-controls="feed-drawer"' in DASHBOARD
    assert ':aria-expanded="feedOpen"' in DASHBOARD


# --------------------------------------------------------------------------- browser
playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect
sync_playwright = playwright.sync_playwright

from tests.test_dashboard_cache_ttl_playwright import (  # noqa: E402
    _fulfill_static_asset,
    _sample_history,
    _sample_stats,
)


def _stats_with_requests() -> dict:
    stats = copy.deepcopy(_sample_stats())
    stats["log_full_messages"] = True
    stats["recent_requests"] = [
        {
            "request_id": "req-1",
            "timestamp": 1_756_000_000.0,
            "model": "claude-opus-4-6",
            "input_tokens_optimized": 1200,
            "output_tokens": 300,
            "savings_percent": 41.0,
            "total_latency_ms": 1510.0,
            "transformations": [],
        }
    ]
    return stats


def _open(page, stats: dict) -> None:  # type: ignore[no-untyped-def]
    history = _sample_history()
    html = get_dashboard_html()

    def handler(route) -> None:  # type: ignore[no-untyped-def]
        path = urlsplit(route.request.url).path
        if path in ("/dashboard", "/"):
            route.fulfill(status=200, content_type="text/html", body=html)
        elif _fulfill_static_asset(route, path):
            return
        elif "/stats-history" in path:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(history))
        elif path.endswith("/stats"):
            route.fulfill(status=200, content_type="application/json", body=json.dumps(stats))
        elif path.endswith("/health"):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "healthy", "version": "0.3.0"}),
            )
        elif "/transformations/feed" in path:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"transformations": []}),
            )
        else:
            route.continue_()

    page.route("**/*", handler)
    page.goto("http://headroom.local/dashboard")
    page.wait_for_load_state("networkidle")


def test_request_row_expands_with_keyboard() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1600})
        _open(page, _stats_with_requests())

        row = page.get_by_role("button", name=re.compile("claude-opus-4-6.*Toggle details"))
        expect(row).to_have_attribute("aria-expanded", "false")
        row.focus()
        page.keyboard.press("Enter")
        expect(row).to_have_attribute("aria-expanded", "true")
        page.keyboard.press("Space")
        expect(row).to_have_attribute("aria-expanded", "false")

        # Keyboard focus is visibly indicated (outline drawn by :focus-visible).
        outline = row.evaluate("el => { el.focus(); return getComputedStyle(el).outlineStyle }")
        assert outline != "none"
        browser.close()


def test_live_feed_drawer_keyboard_flow() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1600})
        _open(page, _stats_with_requests())

        toggle = page.get_by_role("button", name="Live Feed", exact=True)
        expect(toggle).to_have_attribute("aria-expanded", "false")
        toggle.focus()
        page.keyboard.press("Enter")
        drawer = page.get_by_role("dialog", name="Message Transformations live feed")
        expect(drawer).to_be_visible()
        expect(toggle).to_have_attribute("aria-expanded", "true")
        # Focus moves into the drawer, onto its close control.
        expect(page.get_by_role("button", name="Close live feed")).to_be_focused()
        # Escape closes it and returns focus to the control that opened it.
        page.keyboard.press("Escape")
        expect(drawer).to_be_hidden()
        expect(toggle).to_be_focused()
        browser.close()


def test_header_controls_have_names_and_pressed_state() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1600})
        _open(page, _stats_with_requests())

        expect(page.get_by_role("button", name="Toggle light or dark mode")).to_be_visible()
        session = page.get_by_role("button", name="Session", exact=True)
        lifetime = page.get_by_role("button", name="Lifetime", exact=True)
        expect(session).to_have_attribute("aria-pressed", "true")
        expect(lifetime).to_have_attribute("aria-pressed", "false")
        lifetime.focus()
        page.keyboard.press("Enter")
        expect(lifetime).to_have_attribute("aria-pressed", "true")
        expect(page.get_by_role("status")).to_contain_text("Healthy")
        browser.close()
