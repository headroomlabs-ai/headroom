from __future__ import annotations

from headroom.dashboard import get_dashboard_html


def test_dashboard_includes_active_sessions_tab() -> None:
    html = get_dashboard_html()

    assert "Active Sessions" in html
    assert "All Active Sessions" in html
    assert "activeSessionRows" in html
    assert "stats.cluster?.enabled" in html
