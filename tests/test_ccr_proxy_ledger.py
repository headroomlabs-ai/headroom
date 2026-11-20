"""Proxy in-process CCR retrievals are written to the durable savings ledger.

Without this, `headroom savings` net would equal gross for proxy deployments —
the proxy CCR handler never routes through the MCP tool path that writes the
ledger. Sync tests (asyncio.run) so they run without pytest-asyncio.
"""

from __future__ import annotations

import asyncio
import json

from headroom import savings_ledger as L
from headroom.ccr.response_handler import CCRResponseHandler, CCRToolResult
from headroom.ccr.tool_injection import CCR_TOOL_NAME


def _tool_use_response(usage: dict) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "t1", "name": CCR_TOOL_NAME, "input": {"hash": "a" * 24}}
        ],
        "stop_reason": "tool_use",
        "usage": usage,
    }


async def _final(messages, tools):
    return {"role": "assistant", "content": [{"type": "text", "text": "done"}]}


def test_proxy_ccr_round_writes_retrieve_ledger_event(monkeypatch, tmp_path):
    ledger = tmp_path / "savings_events.jsonl"
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(ledger))

    handler = CCRResponseHandler()
    monkeypatch.setattr(
        handler,
        "_execute_retrieval",
        lambda call: CCRToolResult(
            tool_call_id=call.tool_call_id, content="{}", success=True, tokens_retrieved=800
        ),
    )
    asyncio.run(
        handler.handle_response(
            _tool_use_response({"input_tokens": 1200, "output_tokens": 40}),
            [{"role": "user", "content": "hi"}],
            [],
            _final,
            "anthropic",
        )
    )

    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["kind"] == "retrieve"
    assert event["source"] == "proxy"
    # payload 800 + 1 retrieval * 50 overhead
    assert event["tokens_retrieved"] == 850

    life = L.aggregate_savings().lifetime
    assert life["tokens_retrieved"] == 850
    assert life["net_tokens_saved"] == -850  # pure drawback, unclamped


def test_proxy_ccr_miss_writes_no_ledger_event(monkeypatch, tmp_path):
    ledger = tmp_path / "savings_events.jsonl"
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(ledger))

    handler = CCRResponseHandler()
    monkeypatch.setattr(
        handler,
        "_execute_retrieval",
        lambda call: CCRToolResult(
            tool_call_id=call.tool_call_id, content="{}", success=False, tokens_retrieved=0
        ),
    )
    asyncio.run(
        handler.handle_response(
            _tool_use_response({"input_tokens": 100, "output_tokens": 10}),
            [{"role": "user", "content": "hi"}],
            [],
            _final,
            "anthropic",
        )
    )
    assert not ledger.exists() or ledger.read_text(encoding="utf-8") == ""


def test_proxy_ccr_continuation_failure_writes_no_retrieve_event(monkeypatch, tmp_path):
    ledger = tmp_path / "savings_events.jsonl"
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(ledger))

    handler = CCRResponseHandler()
    monkeypatch.setattr(
        handler,
        "_execute_retrieval",
        lambda call: CCRToolResult(
            tool_call_id=call.tool_call_id, content="{}", success=True, tokens_retrieved=800
        ),
    )

    async def failing_continuation(messages, tools):
        raise RuntimeError("upstream failed")

    asyncio.run(
        handler.handle_response(
            _tool_use_response({"input_tokens": 1200, "output_tokens": 40}),
            [{"role": "user", "content": "hi"}],
            [],
            failing_continuation,
            "anthropic",
        )
    )

    assert handler.get_stats()["tokens_retrieved"] == 0
    assert not ledger.exists() or ledger.read_text(encoding="utf-8") == ""
