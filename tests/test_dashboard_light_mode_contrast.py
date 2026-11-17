"""Regression tests for light-mode dashboard contrast."""

from __future__ import annotations

import re

from headroom.dashboard import get_dashboard_html


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def test_savings_profile_badge_is_readable_in_light_mode() -> None:
    html = get_dashboard_html()

    assert 'class="text-xs text-cyan-100"' in html
    override = re.search(
        r"html:not\(\.dark\) \.text-cyan-100 \{ color: (#[0-9a-fA-F]{6}); \}",
        html,
    )
    assert override is not None

    # The badge is Tailwind cyan-500 at 10% opacity over the white light-mode
    # surface (#e6f8fb when composited). WCAG AA requires at least 4.5:1
    # contrast for normal-sized text.
    assert _contrast_ratio(override.group(1), "#e6f8fb") >= 4.5
