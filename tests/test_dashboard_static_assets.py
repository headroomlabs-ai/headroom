"""The dashboard must not depend on third-party CDNs.

Edge's Tracking Prevention (and corporate proxies) block unpkg.com and
cdn.tailwindcss.com, which left the dashboard unstyled and dataless on some
Windows machines. Tailwind/htmx/alpine are vendored and served locally instead.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.dashboard import get_dashboard_html, get_settings_html  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

ASSETS = ["tailwind.min.js", "htmx.min.js", "alpine.min.js"]


@pytest.fixture
def client():
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
        )
    )
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("html", [get_dashboard_html(), get_settings_html()])
def test_templates_reference_no_cdn(html: str):
    assert "unpkg.com" not in html
    assert "cdn.tailwindcss.com" not in html


@pytest.mark.parametrize("asset", ASSETS)
def test_asset_is_served(client, asset: str):
    resp = client.get(f"/dashboard/static/{asset}")
    assert resp.status_code == 200, resp.text
    assert len(resp.content) > 10_000


def test_dashboard_only_references_served_assets(client):
    html = client.get("/dashboard").text
    for asset in ASSETS:
        assert f"/dashboard/static/{asset}" in html
