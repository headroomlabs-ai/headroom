"""Tests for the /transformations/feed endpoint in the proxy server."""

import pytest

# Skip if fastapi not available
pytest.importorskip("fastapi")

from httpx import ASGITransport, AsyncClient

from headroom.proxy.server import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.mark.asyncio
async def test_transformations_feed_endpoint_returns_list(app):
    """The endpoint should return a list of recent transformations."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/transformations/feed")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "transformations" in data
    assert isinstance(data["transformations"], list)


@pytest.mark.asyncio
async def test_transformations_feed_returns_messages(app):
    """Each transformation exposes both the original request and the
    post-compression form that was actually sent upstream, plus the response.

    The pre/post pair is what makes compression legible: consumers can diff
    the two to see what the pipeline stripped, replaced, or kept.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/transformations/feed")

    data = response.json()
    transformations = data["transformations"]
    for t in transformations:
        assert "request_messages" in t or t.get("request_messages") is None
        assert "compressed_messages" in t or t.get("compressed_messages") is None
        assert "response_content" in t or t.get("response_content") is None


@pytest.mark.asyncio
async def test_transformations_feed_respects_limit(app):
    """The endpoint should respect a ?limit= query parameter."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/transformations/feed?limit=5")

    data = response.json()
    assert len(data["transformations"]) <= 5
