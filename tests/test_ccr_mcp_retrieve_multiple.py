"""Batch retrieval through the MCP callbacks and real compression store."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

pytest.importorskip("mcp")

from mcp.types import CallToolRequestParams  # noqa: E402

from headroom.cache.compression_store import CompressionStore  # noqa: E402
from headroom.ccr.mcp_server import (  # noqa: E402
    MAX_RETRIEVAL_HASHES,
    RETRIEVE_MULTIPLE_TOOL_NAME,
    HeadroomMCPServer,
)


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    srv = HeadroomMCPServer(check_proxy=False)
    srv._local_store = CompressionStore()
    yield srv
    srv._local_store.clear()


async def _call(server, arguments):
    result = await server._on_call_tool(
        None,
        CallToolRequestParams(name=RETRIEVE_MULTIPLE_TOOL_NAME, arguments=arguments),
    )
    assert not result.is_error, result.content
    return json.loads(result.content[0].text)


async def test_multiple_tool_is_advertised_with_batch_schema(server):
    response = await server._on_list_tools(None)
    tools = response.tools
    tool = next(t for t in tools if t.name == RETRIEVE_MULTIPLE_TOOL_NAME)
    assert tool.input_schema["required"] == ["hashes"]
    hashes = tool.input_schema["properties"]["hashes"]
    assert hashes["type"] == "array"
    assert hashes["items"]["type"] == "string"
    assert hashes["minItems"] == 1
    assert hashes["maxItems"] == MAX_RETRIEVAL_HASHES


async def test_multiple_returns_full_originals_in_order_including_duplicates(server):
    store = server._local_store
    originals = ["first original\nwith formatting", "第二个原文\n"]
    keys = [store.store(text, "compressed") for text in originals]

    payload = await _call(server, {"hashes": [keys[1], keys[0], keys[1]]})

    results = payload["results"]
    assert [r["hash"] for r in results] == [keys[1], keys[0], keys[1]]
    assert [r["original_content"] for r in results] == [originals[1], originals[0], originals[1]]
    assert all(r["source"] == "local" for r in results)
    assert server._stats.retrievals == 3


async def test_missing_and_expired_hashes_do_not_hide_successes(server, monkeypatch):
    monkeypatch.setattr("headroom.cache.compression_store.time.time", lambda: 1000.0)
    expired = server._local_store.store("expired original", "small", ttl=1)
    available = server._local_store.store("still available", "small", ttl=100)
    monkeypatch.setattr("headroom.cache.compression_store.time.time", lambda: 1002.0)

    results = (await _call(server, {"hashes": ["missing", expired, available]}))["results"]

    assert results[0]["hash"] == "missing"
    assert "error" in results[0]
    assert results[1]["status"] == "expired"
    assert "do not retry the same hash" in results[1]["error"].lower()
    assert results[2]["original_content"] == "still available"
    assert server._stats.retrievals == 1


async def test_batch_retrieval_does_not_extend_entry_ttl(server, monkeypatch):
    monkeypatch.setattr("headroom.cache.compression_store.time.time", lambda: 1000.0)
    key = server._local_store.store("original", "compressed", ttl=10)
    monkeypatch.setattr("headroom.cache.compression_store.time.time", lambda: 1005.0)

    results = (await _call(server, {"hashes": [key, key]}))["results"]
    assert [result["original_content"] for result in results] == ["original", "original"]

    monkeypatch.setattr("headroom.cache.compression_store.time.time", lambda: 1011.0)
    result = (await _call(server, {"hashes": [key]}))["results"][0]
    assert result["status"] == "expired"
    assert "error" in result


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"hashes": None},
        {"hashes": "abc123"},
        {"hashes": []},
        {"hashes": {"hash": "abc123"}},
        {"hashes": ["valid", ""]},
        {"hashes": ["valid", " \t\n"]},
        {"hashes": ["valid", 42]},
        {"hashes": ["valid", None]},
        {"hashes": ["valid", ["abc123"]]},
        {"hashes": ["abc123"] * (MAX_RETRIEVAL_HASHES + 1)},
    ],
)
async def test_invalid_batch_is_rejected_before_any_lookup(server, monkeypatch, arguments):
    async def unexpected_lookup(hash_key):
        pytest.fail("invalid input must not trigger retrieval")

    monkeypatch.setattr(server, "_retrieve_content", unexpected_lookup)

    content = await server._handle_retrieve_multiple(arguments)
    assert "error" in json.loads(content[0].text)
    assert server._stats.retrievals == 0


async def test_backend_failure_is_reported_per_hash(server, monkeypatch):
    available = server._local_store.store("original", "compressed")
    retrieve = server._retrieve_content

    async def failing_retrieve(hash_key):
        if hash_key == "broken":
            raise RuntimeError("backend unavailable")
        return await retrieve(hash_key)

    monkeypatch.setattr(server, "_retrieve_content", failing_retrieve)

    results = (await _call(server, {"hashes": ["broken", available]}))["results"]

    assert results[0]["hash"] == "broken"
    assert "error" in results[0]
    assert results[1]["original_content"] == "original"


async def test_batch_uses_local_store_and_http_proxy_fallback(server):
    local = server._local_store.store("local original", "compressed")
    requests = []

    def proxy(request):
        requests.append(request)
        assert request.url.path == "/v1/retrieve"
        hash_key = json.loads(request.content)["hash"]
        if hash_key == "remote":
            return httpx.Response(
                200, json={"hash": hash_key, "original_content": "proxy original"}
            )
        return httpx.Response(404)

    server.check_proxy = True
    async with httpx.AsyncClient(transport=httpx.MockTransport(proxy)) as client:
        server._http_client = client
        results = (await _call(server, {"hashes": [local, "remote", "missing"]}))["results"]

    assert results[0]["source"] == "local"
    assert results[1]["source"] == "proxy"
    assert results[1]["original_content"] == "proxy original"
    assert results[2]["hash"] == "missing"
    assert "error" in results[2]
    assert len(requests) == 2
    assert server._stats.retrievals == 2


async def test_proxy_lookups_are_concurrent_but_bounded_and_results_stay_ordered(
    server, monkeypatch
):
    active = peak = 0

    async def retrieve(hash_key):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return {"hash": hash_key, "original_content": hash_key}

    monkeypatch.setattr(server, "_retrieve_content", retrieve)
    hashes = [str(i) for i in range(MAX_RETRIEVAL_HASHES)]

    results = (await _call(server, {"hashes": hashes}))["results"]

    assert 1 < peak <= 8
    assert [r["hash"] for r in results] == hashes
