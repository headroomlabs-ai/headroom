from __future__ import annotations

import pytest

from headroom.tray import core
from headroom.tray.core import Snapshot


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "0"),
        (950, "950"),
        (1_000, "1.00K"),
        (12_345, "12.3K"),
        (999_999, "1.00M"),
        (4_560_000, "4.56M"),
        (1_200_000_000, "1.20B"),
    ],
)
def test_format_tokens(n: int, expected: str) -> None:
    assert core.format_tokens(n) == expected


def test_format_usd() -> None:
    assert core.format_usd(0) == "$0.00"
    assert core.format_usd(1234.5) == "$1,234.50"
    assert core.format_usd(25_000) == "$25.0K"


def test_parse_stats_sums_input_and_output_savings() -> None:
    snap = core.parse_stats(
        {
            "persistent_savings": {
                "lifetime": {
                    "tokens_saved": 1000,
                    "output_tokens_saved": 500,
                    "compression_savings_usd": 1.5,
                    "output_savings_usd": 0.25,
                    "requests": 7,
                }
            }
        }
    )
    assert snap == Snapshot(online=True, tokens_saved=1500, usd_saved=1.75, requests=7)


def test_parse_stats_tolerates_missing_keys() -> None:
    assert core.parse_stats({}) == Snapshot(online=True)


def test_fetch_snapshot_offline_never_raises() -> None:
    assert core.fetch_snapshot("http://127.0.0.1:1", timeout=0.2).online is False


def test_base_url_maps_wildcard_and_ipv6() -> None:
    assert core.base_url("0.0.0.0", 8787) == "http://127.0.0.1:8787"
    assert core.base_url("::1", 9) == "http://[::1]:9"


def test_tooltip_capped_and_mentions_update() -> None:
    text = core.tooltip(Snapshot(online=True, tokens_saved=2000, usd_saved=3), "9.9.9")
    assert "2.00K" in text and "9.9.9" in text and len(text) <= 127
    assert "offline" in core.tooltip(Snapshot())


def test_available_update(monkeypatch: pytest.MonkeyPatch) -> None:
    import headroom.update_check as uc

    monkeypatch.setattr(uc, "is_update_check_enabled", lambda: True)
    monkeypatch.setattr(uc, "installed_version", lambda: "1.0.0")
    monkeypatch.setattr(uc, "read_cache", lambda: {"latest_version": "1.2.0"})
    assert core.available_update() == "1.2.0"
    monkeypatch.setattr(uc, "read_cache", lambda: {"latest_version": "1.0.0"})
    assert core.available_update() is None


def test_icon_renders_all_states() -> None:
    pytest.importorskip("PIL")
    from headroom.tray.icon import render_icon

    for online in (True, False):
        for update in (True, False):
            assert render_icon(online=online, update=update, size=32).size == (32, 32)
