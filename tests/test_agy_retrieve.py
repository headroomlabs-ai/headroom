"""Tests for headroom.proxy.agy_retrieve.AgyRetrieveServer.

The retrieve server is a PLAIN-HTTP loopback listener that serves the same
FastAPI app (``create_app()``) as the HTTPS dispatch server.  Its load-bearing
property: it shares the *process-global* compression store, so a marker stored
on the dispatch side resolves via ``GET /v1/retrieve/{hash}`` on this side.

All tests use ephemeral loopback ports; no TLS, no real network, no
``~/.headroom`` mutation beyond the in-memory process-global store (which is
reset around each test).
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from headroom.cache.compression_store import (
    get_compression_store,
    reset_compression_store,
)
from headroom.proxy.agy_retrieve import AgyRetrieveServer


@pytest.fixture(autouse=True)
def _clean_compression_store():
    """Isolate the process-global compression store around each test."""
    reset_compression_store()
    yield
    reset_compression_store()


async def test_retrieve_server_starts_on_loopback_plain_http() -> None:
    """Server binds loopback and answers plain HTTP (no TLS handshake)."""
    srv = AgyRetrieveServer(port=0)
    await srv.start()
    try:
        host, port = srv.address
        assert host == "127.0.0.1"
        assert isinstance(port, int) and port > 0

        # Plain HTTP (http://) must succeed — proving there is NO TLS layer.
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{port}/v1/retrieve/stats")
        assert resp.status_code == 200
    finally:
        await srv.stop()


async def test_get_retrieve_returns_store_populated_content() -> None:
    """LOAD-BEARING: a hash stored via the process-global store resolves over
    plain HTTP from a SECOND create_app() — proving the cache is shared.

    This is exactly the dispatch-populates / retrieve-resolves contract: the
    HTTPS dispatch server stores markers into the same process-global singleton
    that this plain-HTTP listener serves.
    """
    original = '{"rows": [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]}'
    compressed = '{"rows": "[Retrieve more]"}'

    # Populate the process-global store DIRECTLY (as the dispatch side would,
    # via the same get_compression_store() singleton) — the server is a
    # *separate* create_app() instance and must still see this entry.
    store = get_compression_store()
    hash_key = store.store(
        original=original,
        compressed=compressed,
        original_tokens=42,
        compressed_tokens=7,
        tool_name="search_api",
    )

    srv = AgyRetrieveServer(port=0)
    await srv.start()
    try:
        _, port = srv.address
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{port}/v1/retrieve/{hash_key}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["hash"] == hash_key
        assert body["original_content"] == original
        assert body["tool_name"] == "search_api"
    finally:
        await srv.stop()


async def test_get_unknown_hash_returns_404() -> None:
    """An unknown marker hash returns 404 (not a 500/hang)."""
    srv = AgyRetrieveServer(port=0)
    await srv.start()
    try:
        _, port = srv.address
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{port}/v1/retrieve/deadbeefdeadbeefdeadbeef")
        assert resp.status_code == 404
    finally:
        await srv.stop()


async def test_retrieve_server_binds_loopback_only() -> None:
    """The listener socket family/host must be loopback (127.0.0.1)."""
    srv = AgyRetrieveServer(port=0)
    await srv.start()
    try:
        host, _ = srv.address
        assert host == "127.0.0.1"
    finally:
        await srv.stop()


async def test_retrieve_server_clean_start_stop_no_leaked_server_tasks() -> None:
    """start()/stop() leaves no server-owned tasks (lifespan / connection).

    The only surviving task may be the FastAPI app's *periodic* TOIN-stats
    background task — an app-level concern that the production
    ``_start_agy_servers`` reaps at loop teardown (it cancels all pending tasks
    in its ``finally`` before ``loop.close()``).  This test mirrors that final
    sweep and asserts every leftover is cancellable (i.e. no task wedges the
    shutdown), and that the server's OWN lifespan task is gone.
    """
    loop = asyncio.get_running_loop()
    before = {t for t in asyncio.all_tasks(loop) if not t.done()}

    srv = AgyRetrieveServer(port=0)
    await srv.start()
    await srv.stop()
    assert srv._lifespan_task is None, "stop() must clear the lifespan task"

    await asyncio.sleep(0)
    after = {t for t in asyncio.all_tasks(loop) if not t.done()}
    leaked = after - before

    # Any leftover must be ONLY the app-level periodic stats task; no hypercorn
    # connection / lifespan task may survive stop().
    offending = [t for t in leaked if "_log_toin_stats_periodically" not in repr(t.get_coro())]
    assert not offending, f"retrieve server leaked server-owned tasks: {offending}"

    # Model the production loop-teardown sweep: every leftover cancels cleanly.
    for task in leaked:
        task.cancel()
    if leaked:
        await asyncio.gather(*leaked, return_exceptions=True)


async def test_retrieve_server_stop_idempotent() -> None:
    """stop() after stop() does not raise."""
    srv = AgyRetrieveServer(port=0)
    await srv.start()
    await srv.stop()
    await srv.stop()  # idempotent


def test_retrieve_server_address_raises_before_start() -> None:
    """address property raises RuntimeError before start()."""
    srv = AgyRetrieveServer(port=0)
    with pytest.raises(RuntimeError):
        _ = srv.address
