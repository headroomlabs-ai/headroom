"""Fuzzing tests for RequestCoalescer."""
import asyncio
import pytest
from hypothesis import given, strategies as st, settings
from headroom.proxy.coalescer import RequestCoalescer


@settings(max_examples=20, deadline=2000)
@given(
    n_requests=st.integers(min_value=1, max_value=5),
    window=st.floats(min_value=0.05, max_value=0.5),
)
@pytest.mark.asyncio
async def test_batch_size_never_exceeds_max(n_requests, window):
    """Batch size should never exceed max_batch."""
    max_batch = 3
    c = RequestCoalescer(window_sec=window, max_batch=max_batch)
    tasks = [
        c.submit({"model": "test", "messages": [{"role": "user", "content": f"msg {i}"}]})
        for i in range(n_requests)
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        batch_size = r.get("batch_size", 1)
        assert batch_size <= max_batch
    await c.close()


@settings(max_examples=20, deadline=1000)
@given(window=st.floats(min_value=0.05, max_value=0.3))
@pytest.mark.asyncio
async def test_solo_request_never_batched(window):
    """Single request should never be batched."""
    c = RequestCoalescer(window_sec=window, max_batch=5)
    r = await c.submit({"model": "test", "messages": []})
    assert r["coalesced"] is False
    await c.close()
