"""Proxy-path CCR handler records retrieved-token volume and dropped-response overhead.

Sync tests (driven via asyncio.run) so they run without pytest-asyncio: the
handler must accumulate the retrieved payload size and the usage of the response
it drops when it makes a continuation call.
"""

from __future__ import annotations

import asyncio

from headroom.ccr.response_handler import CCRResponseHandler, CCRToolResult
from headroom.ccr.tool_injection import CCR_TOOL_NAME


def test_handle_response_accumulates_retrieved_tokens_and_overhead(monkeypatch):
    handler = CCRResponseHandler()
    initial = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "t1", "name": CCR_TOOL_NAME, "input": {"hash": "a" * 24}}
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1200, "output_tokens": 40},
    }
    monkeypatch.setattr(
        handler,
        "_execute_retrieval",
        lambda call: CCRToolResult(
            tool_call_id=call.tool_call_id,
            content="{}",
            success=True,
            items_retrieved=3,
            tokens_retrieved=800,
        ),
    )

    async def api_call(messages, tools):
        # Continuation with no further tool calls ends the loop.
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }

    asyncio.run(
        handler.handle_response(
            initial, [{"role": "user", "content": "hi"}], [], api_call, "anthropic"
        )
    )
    stats = handler.get_stats()
    assert stats["total_retrievals"] == 1
    assert stats["tokens_retrieved"] == 800
    # Overhead = usage of the response we dropped when making the continuation.
    assert stats["ccr_overhead_input_tokens"] == 1200
    assert stats["ccr_overhead_output_tokens"] == 40


def test_extract_usage_tokens_tolerates_shapes():
    f = CCRResponseHandler._extract_usage_tokens
    assert f({"usage": {"input_tokens": 10, "output_tokens": 5}}) == (10, 5)
    assert f({"usage": {"prompt_tokens": 7, "completion_tokens": 3}}) == (7, 3)
    assert f({}) == (0, 0)
    assert f("not-a-dict") == (0, 0)
    assert f({"usage": None}) == (0, 0)
