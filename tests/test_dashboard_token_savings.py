"""Regression tests for dashboard token savings copy."""

from __future__ import annotations

import re

from headroom.dashboard import get_dashboard_html


def _getter_body(html: str, name: str) -> str:
    match = re.search(rf"get {name}\(\) \{{(?P<body>.*?)\n\s*\}},", html, re.S)
    assert match is not None
    return match.group("body")


def test_token_savings_headline_leads_with_total_wire_then_new_input() -> None:
    html = get_dashboard_html()

    # Whole wire keeps the headline slot (#1653): it is the conservative,
    # screenshot-able figure. The old attempted-denominator must stay gone.
    headline_body = _getter_body(html, "headlineSavingsPercent")
    assert "stats.tokens?.savings_percent" in headline_body
    assert "stats.tokens?.proxy_savings_percent" in headline_body
    assert "stats.tokens?.new_input_savings_percent" not in headline_body
    assert "stats.tokens?.active_savings_percent" not in headline_body
    assert "stats.tokens?.proxy_attempted_tokens" not in headline_body

    title_body = _getter_body(html, "headlineSavingsTitle")
    assert "Of total wire input tokens" in title_body
    assert "Of compressible tokens attempted" not in title_body

    # New input rides the line below, gated on a provider cache breakdown, and
    # replaces the old "Of total wire" sublabel that duplicated the headline.
    assert "'Of new input: '" in html
    assert "'Of total wire: '" not in html
    assert 'x-show="(stats.tokens?.new_input_tokens || 0) > 0"' in html


def test_new_input_explanation_lives_in_the_tooltip_not_the_card() -> None:
    """The card shows a number and its basis; the "why" is hover-only."""
    html = get_dashboard_html()
    card_line = next(line for line in html.splitlines() if "'Of new input: '" in line)
    label, _, tooltip = card_line.partition("title=")
    assert "what compression can touch" not in label
    assert "what compression can touch" in tooltip
