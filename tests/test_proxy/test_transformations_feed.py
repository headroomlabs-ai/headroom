"""Tests for the /transformations/feed endpoint in the proxy server."""

import pytest

# Skip if fastapi not available
pytest.importorskip("fastapi")

from httpx import ASGITransport, AsyncClient

from headroom.proxy.models import RequestLog
from headroom.proxy.server import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.mark.asyncio
async def test_transformations_feed_endpoint_returns_list(app):
    """The endpoint should return a list of recent transformations."""
    async with AsyncClient(
        transport=ASGITransport(app=app, client=("127.0.0.1", 12345)),
        base_url="http://127.0.0.1",
    ) as client:
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
    async with AsyncClient(
        transport=ASGITransport(app=app, client=("127.0.0.1", 12345)),
        base_url="http://127.0.0.1",
    ) as client:
        response = await client.get("/transformations/feed")

    data = response.json()
    transformations = data["transformations"]
    for t in transformations:
        assert "request_messages" in t
        assert t["request_messages"] is None or isinstance(t["request_messages"], list)
        assert "compressed_messages" in t
        assert t["compressed_messages"] is None or isinstance(t["compressed_messages"], list)
        assert "response_content" in t
        assert t["response_content"] is None or isinstance(t["response_content"], str)


@pytest.mark.asyncio
async def test_transformations_feed_respects_limit(app):
    """The endpoint should respect a ?limit= query parameter."""
    async with AsyncClient(
        transport=ASGITransport(app=app, client=("127.0.0.1", 12345)),
        base_url="http://127.0.0.1",
    ) as client:
        response = await client.get("/transformations/feed?limit=5")

    data = response.json()
    assert len(data["transformations"]) <= 5


@pytest.mark.asyncio
async def test_transformations_feed_can_omit_message_bodies(app):
    """``?include_messages=0`` is for pollers that only read the numbers: the
    three body fields are absent (not null) and every other field is the same
    as the full response."""
    app.state.proxy.logger.log(
        RequestLog(
            request_id="r1",
            timestamp="2026-04-24T10:00:00Z",
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens_original=100,
            input_tokens_optimized=40,
            output_tokens=10,
            tokens_saved=60,
            savings_percent=60.0,
            optimization_latency_ms=1.0,
            total_latency_ms=20.0,
            tags={},
            cache_hit=False,
            transforms_applied=["kompress:user:0.4"],
            cache_read_tokens=1000,
            cache_write_tokens=5,
            uncached_input_tokens=30,
            request_messages=[{"role": "user", "content": "hi"}],
            compressed_messages=[{"role": "user", "content": "hi"}],
            response_content="ok",
        )
    )
    async with AsyncClient(
        transport=ASGITransport(app=app, client=("127.0.0.1", 12345)),
        base_url="http://127.0.0.1",
    ) as client:
        full = (await client.get("/transformations/feed?limit=1")).json()
        slim = (await client.get("/transformations/feed?limit=1&include_messages=0")).json()

    full_item = full["transformations"][0]
    slim_item = slim["transformations"][0]
    assert full_item["request_messages"] == [{"role": "user", "content": "hi"}]
    bodies = {"request_messages", "compressed_messages", "response_content"}
    assert not bodies & slim_item.keys()
    # The prefix-cache split rides along, so a poller can rate tokens_saved
    # against new input (uncached + cache_write) like /stats does.
    assert (
        slim_item["uncached_input_tokens"],
        slim_item["cache_write_tokens"],
        slim_item["cache_read_tokens"],
    ) == (30, 5, 1000)
    assert {k: v for k, v in full_item.items() if k not in bodies} == slim_item
    assert slim["log_full_messages"] == full["log_full_messages"]
