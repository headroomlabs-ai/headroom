"""Tests for the request coalescer.

Contract under test:
- only requests with *complete-input identity* share a dispatch
- two different prompts under one credential never trade answers
- the shared dispatch is owned by no single waiter, so cancelling one
  caller cannot strand its peers
"""

import asyncio

import pytest

from headroom.proxy.coalescer import RequestCoalescer


def _payload(text: str, *, model: str = "test", system: str | None = None) -> dict:
    body: dict = {"model": model, "messages": [{"role": "user", "content": text}]}
    if system is not None:
        body["system"] = system
    return body


@pytest.mark.asyncio
async def test_single_request_not_batched():
    c = RequestCoalescer(window_sec=0.1, max_batch=3)
    result = await c.submit(_payload("hi"))
    assert result["coalesced"] is False
    await c.close()


@pytest.mark.asyncio
async def test_identical_rapid_requests_batched():
    c = RequestCoalescer(window_sec=0.3, max_batch=3)
    tasks = [c.submit(_payload("same question")) for _ in range(3)]
    results = await asyncio.gather(*tasks)
    for r in results:
        assert r["coalesced"] is True
        assert r["batch_size"] == 3
    assert c.get_stats()["api_calls_saved"] == 2
    await c.close()


@pytest.mark.asyncio
async def test_max_batch_triggers_flush():
    c = RequestCoalescer(window_sec=10.0, max_batch=2)
    results = await asyncio.gather(*(c.submit(_payload("same")) for _ in range(2)))
    for r in results:
        assert r["coalesced"] is True
        assert r["batch_size"] == 2
    await c.close()


@pytest.mark.asyncio
async def test_different_prompts_never_share_a_dispatch_or_an_answer():
    """Two prompts under one credential are independent sessions.

    Merging their histories was the defect: one outbound conversation, one
    upstream answer, handed to both callers.
    """
    seen: list[dict] = []

    async def dispatch(payload):
        prompt = payload["messages"][-1]["content"]
        seen.append(payload)
        # Deterministic upstream: answer the final prompt it was given.
        return {"id": f"msg_{prompt}", "text": prompt}

    c = RequestCoalescer(window_sec=0.2, max_batch=8, dispatch_fn=dispatch)
    alpha, beta = await asyncio.gather(
        c.submit(_payload("Return ALPHA only")),
        c.submit(_payload("Return BETA only")),
    )

    assert alpha["coalesced"] is False and beta["coalesced"] is False
    assert alpha["response"]["text"] == "Return ALPHA only"
    assert beta["response"]["text"] == "Return BETA only"
    assert len(seen) == 2, "each prompt needs its own upstream call"
    assert seen[0]["messages"] == [{"role": "user", "content": "Return ALPHA only"}]
    assert seen[1]["messages"] == [{"role": "user", "content": "Return BETA only"}]
    assert c.get_stats()["api_calls_saved"] == 0
    await c.close()


@pytest.mark.asyncio
async def test_identical_conversations_preserve_every_turn():
    """Repeated turns inside one request are never deduplicated away."""
    c = RequestCoalescer(window_sec=0.1, max_batch=2)
    req = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "shared intro"},
            {"role": "user", "content": "retry"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "retry"},
        ],
    }
    r1, r2 = await asyncio.gather(c.submit(req), c.submit(dict(req)))
    assert r1["coalesced"] is True
    assert r1["payload"]["messages"] == req["messages"]
    assert r2["payload"]["messages"] == req["messages"]
    assert sum(1 for m in r2["payload"]["messages"] if m["content"] == "retry") == 2
    await c.close()


@pytest.mark.asyncio
async def test_solo_after_window():
    c = RequestCoalescer(window_sec=0.1, max_batch=5)
    r1 = await c.submit(_payload("first"))
    assert r1["coalesced"] is False
    await asyncio.sleep(0.2)
    r2 = await c.submit(_payload("second"))
    assert r2["coalesced"] is False
    await c.close()


@pytest.mark.asyncio
async def test_async_context_manager_closes_cleanly():
    async with RequestCoalescer(window_sec=0.01, max_batch=2) as coalescer:
        result = await coalescer.submit(_payload("hi"))
    assert result["coalesced"] is False


@pytest.mark.asyncio
async def test_cancelled_waiter_before_flush_does_not_strand_batch_members():
    c = RequestCoalescer(window_sec=0.2, max_batch=2)
    alice_task = asyncio.create_task(c.submit(_payload("alice")))
    await asyncio.sleep(0.02)
    alice_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await alice_task

    bob_res, carol_res = await asyncio.gather(
        c.submit(_payload("same")),
        c.submit(_payload("same")),
    )
    assert bob_res["coalesced"] is True
    assert carol_res["coalesced"] is True
    assert bob_res["batch_size"] == 2
    assert c.get_stats()["cancelled_requests"] >= 1
    await c.close()


@pytest.mark.asyncio
async def test_cancelling_waiter_during_in_flight_full_batch_flush_keeps_peers():
    """A batch-triggering caller must not own, or be able to cancel, dispatch.

    max_batch is reached by the second submit; dispatch is then held. Because
    the second submit triggers the flush *inline*, cancelling it used to
    cancel the dispatch and leave the first caller permanently unresolved.
    """
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[dict] = []

    async def slow_dispatch(payload):
        calls.append(payload)
        started.set()
        await release.wait()
        return {"ok": True}

    c = RequestCoalescer(window_sec=10.0, max_batch=2, dispatch_fn=slow_dispatch)
    first = asyncio.create_task(c.submit(_payload("same")))
    second = asyncio.create_task(c.submit(_payload("same")))

    await asyncio.wait_for(started.wait(), timeout=2)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second

    release.set()
    result = await asyncio.wait_for(first, timeout=2)
    assert result["response"] == {"ok": True}
    assert len(calls) == 1
    assert c.get_stats()["dispatch_errors"] == 0
    assert not any(not r.future.done() for r in c._pending)
    await c.close()


@pytest.mark.asyncio
async def test_incompatible_settings_partitioned_into_separate_batches():
    """Different model, system, tools, or scope never inherit each other's settings."""
    c = RequestCoalescer(window_sec=0.05, max_batch=3)
    r_model_a, r_model_b, r_sys_b = await asyncio.gather(
        c.submit(_payload("a", model="claude-sonnet-4-5", system="sys-a")),
        c.submit(_payload("a", model="claude-haiku-4-5", system="sys-a")),
        c.submit(_payload("a", model="claude-sonnet-4-5", system="sys-b")),
    )
    assert r_model_a["coalesced"] is False
    assert r_model_a["payload"]["model"] == "claude-sonnet-4-5"
    assert r_model_a["payload"]["system"] == "sys-a"
    assert r_model_b["coalesced"] is False
    assert r_model_b["payload"]["model"] == "claude-haiku-4-5"
    assert r_sys_b["coalesced"] is False
    assert r_sys_b["payload"]["system"] == "sys-b"
    await c.close()


@pytest.mark.asyncio
async def test_dispatch_errors_reach_every_identical_waiter():
    dispatched: list[dict] = []

    async def fake_dispatch(payload):
        dispatched.append(payload)
        return {"id": "msg_batch_1"}

    c = RequestCoalescer(window_sec=0.2, max_batch=2, dispatch_fn=fake_dispatch)
    r1, r2 = await asyncio.gather(c.submit(_payload("one")), c.submit(_payload("one")))
    assert len(dispatched) == 1
    assert r1["response"]["id"] == "msg_batch_1"
    assert r2["response"]["id"] == "msg_batch_1"
    assert r1["response"] is r2["response"], "identical inputs may share one decoded answer"
    await c.close()

    async def failing_dispatch(_payload):
        raise RuntimeError("upstream unavailable")

    c_err = RequestCoalescer(window_sec=0.2, max_batch=2, dispatch_fn=failing_dispatch)
    with pytest.raises(RuntimeError, match="upstream unavailable"):
        await asyncio.gather(c_err.submit(_payload("one")), c_err.submit(_payload("one")))
    assert c_err.get_stats()["dispatch_errors"] == 1
    await c_err.close()
