from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


def test_favicon_is_served_locally_and_never_reaches_passthrough(monkeypatch) -> None:
    """GH #1787: /favicon.ico must not be tunneled to the upstream provider."""
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)

    with TestClient(app) as client:
        with patch.object(
            client.app.state.proxy, "handle_passthrough", new=AsyncMock()
        ) as passthrough:
            response = client.get("/favicon.ico")

    assert response.status_code == 204
    passthrough.assert_not_called()
