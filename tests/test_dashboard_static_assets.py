"""The dashboard must not depend on third-party CDNs.

Edge's Tracking Prevention (and corporate proxies) block unpkg.com and
cdn.tailwindcss.com, which left the dashboard unstyled and dataless on some
Windows machines. Tailwind/htmx/alpine are vendored and served locally instead.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi.responses import Response  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from headroom.dashboard import get_dashboard_html, get_settings_html  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

ASSETS = ["tailwind.min.js", "htmx.min.js", "alpine.min.js"]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            http2=False,
        )
    )
    with TestClient(app) as c:
        yield c


@pytest.fixture
def passthrough(client):
    with patch.object(
        client.app.state.proxy,
        "handle_passthrough",
        new=AsyncMock(return_value=Response("upstream", status_code=418)),
    ) as handler:
        yield handler


@pytest.mark.parametrize("html", [get_dashboard_html(), get_settings_html()])
def test_templates_reference_no_cdn(html: str):
    assert "unpkg.com" not in html
    assert "cdn.tailwindcss.com" not in html


@pytest.mark.parametrize("asset", ASSETS)
def test_asset_is_served(client, passthrough, asset: str):
    resp = client.get(f"/dashboard/static/{asset}")
    assert resp.status_code == 200, resp.text
    assert len(resp.content) > 10_000
    passthrough.assert_not_called()


def test_dashboard_trailing_slash_is_served_locally(client, passthrough):
    resp = client.get("/dashboard/")

    assert resp.status_code == 200
    assert resp.text == get_dashboard_html()
    passthrough.assert_not_called()


@pytest.mark.parametrize("path", ["/dashboard/static", "/dashboard/static/"])
def test_static_directory_boundaries_are_not_forwarded(client, passthrough, path: str):
    resp = client.get(path)

    assert resp.status_code == 404
    assert resp.text == "Not Found"
    passthrough.assert_not_called()


def test_unrelated_unknown_path_still_reaches_passthrough(client, passthrough):
    resp = client.get("/unrelated-unknown-path")

    assert resp.status_code == 418
    passthrough.assert_awaited_once()


def test_dashboard_only_references_served_assets(client):
    html = client.get("/dashboard").text
    for asset in ASSETS:
        assert f"/dashboard/static/{asset}" in html
