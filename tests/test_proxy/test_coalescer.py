"""Tests for request coalescer."""
import asyncio
import pytest
from headroom.proxy.coalescer import RequestCoalescer


@pytest.mark.asyncio
async def test_single_request_not_batched():
    c = RequestCoalescer(window_sec=0.1, max_batch=3)
    result = await c.submit({"model": "test", "messages": [{"role": "user", "content": "hi"}]})
    assert result["coalesced"] is False
    await c.close()


@pytest.mark.asyncio
async def test_rapid_requests_batched():
    c = RequestCoalescer(window_sec=0.3, max_batch=3)
    tasks = [c.submit({"model": "test", "messages": [{"role": "user", "content": f"msg {i}"}]}) for i in range(3)]
    results = await asyncio.gather(*tasks)
    for r in results:
        assert r["coalesced"] is True
        assert r["batch_size"] == 3
    await c.close()


@pytest.mark.asyncio
async def test_max_batch_triggers_flush():
    c = RequestCoalescer(window_sec=10.0, max_batch=2)
    tasks = [c.submit({"model": "test", "messages": [{"role": "user", "content": f"msg {i}"}]}) for i in range(2)]
    results = await asyncio.gather(*tasks)
    for r in results:
        assert r["coalesced"] is True
        assert r["batch_size"] == 2
    await c.close()


@pytest.mark.asyncio
async def test_deduplication():
    c = RequestCoalescer(window_sec=0.1, max_batch=3)
    tasks = [c.submit({"model": "test", "messages": [
        {"role": "user", "content": "same message"},
        {"role": "user", "content": f"unique {i}"},
    ]}) for i in range(3)]
    results = await asyncio.gather(*tasks)
    merged = results[0]["payload"]
    messages = merged.get("messages", [])
    same_count = sum(1 for m in messages if m.get("content") == "same message")
    assert same_count == 1
    await c.close()


@pytest.mark.asyncio
async def test_solo_after_window():
    c = RequestCoalescer(window_sec=0.1, max_batch=5)
    r1 = await c.submit({"model": "test", "messages": [{"role": "user", "content": "first"}]})
    assert r1["coalesced"] is False
    await asyncio.sleep(0.2)
    r2 = await c.submit({"model": "test", "messages": [{"role": "user", "content": "second"}]})
    assert r2["coalesced"] is False
    await c.close()
