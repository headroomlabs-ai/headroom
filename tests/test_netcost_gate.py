"""Net-cost mutation gate in ContentRouter (#856 P2, flag-gated).

``HEADROOM_NET_COST_POLICY=1`` routes every router mutation candidate
through ``CompressionPolicy.net_mutation_gain`` with the issue's v1
estimators (exact ΔT, S = token total after the slot, env-tunable R and
P_alive). Flag off (default) preserves exact current behavior.
"""

from __future__ import annotations

import json

import pytest

from headroom import OpenAIProvider, Tokenizer
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

_provider = OpenAIProvider()


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer(_provider.get_token_counter("gpt-4o"), "gpt-4o")


@pytest.fixture
def router() -> ContentRouter:
    return ContentRouter(ContentRouterConfig())


def _tool_json(rows: int) -> str:
    return json.dumps(
        [{"id": i, "name": f"item_{i}", "status": "ok", "score": i * 3.14} for i in range(rows)]
    )


def _messages(tool_content: str, suffix_filler_words: int) -> list[dict]:
    suffix = "analysis context word " * suffix_filler_words
    return [
        {"role": "user", "content": "fetch the records"},
        {"role": "tool", "content": tool_content},
        {"role": "user", "content": suffix},
        {"role": "user", "content": "summarize"},
    ]


def _tool_slot_compressed(result, messages) -> bool:
    return result.messages[1]["content"] != messages[1]["content"]


class TestNetCostGate:
    def test_flag_off_compresses_as_before(self, router, tokenizer, monkeypatch):
        monkeypatch.delenv("HEADROOM_NET_COST_POLICY", raising=False)
        messages = _messages(_tool_json(300), suffix_filler_words=4000)
        result = router.apply([dict(m) for m in messages], tokenizer)
        assert _tool_slot_compressed(result, messages)
        assert not any(t.startswith("netcost:") for t in result.transforms_applied)

    def test_flag_on_blocks_when_suffix_dominates(self, router, tokenizer, monkeypatch):
        # Big suffix after a modest shave: corrected formula says the cache
        # invalidation outweighs the saving -> slot left untouched.
        monkeypatch.setenv("HEADROOM_NET_COST_POLICY", "1")
        messages = _messages(_tool_json(300), suffix_filler_words=40000)
        result = router.apply([dict(m) for m in messages], tokenizer)
        assert not _tool_slot_compressed(result, messages)
        assert any(t.startswith("netcost:skip:") for t in result.transforms_applied)

    def test_flag_on_allows_when_shave_dominates(self, router, tokenizer, monkeypatch):
        # Tiny suffix after a huge shave -> gate allows, compression applies.
        monkeypatch.setenv("HEADROOM_NET_COST_POLICY", "1")
        messages = _messages(_tool_json(2000), suffix_filler_words=5)
        result = router.apply([dict(m) for m in messages], tokenizer)
        assert _tool_slot_compressed(result, messages)
        assert not any(t.startswith("netcost:skip:") for t in result.transforms_applied)

    def test_flag_on_gates_cached_results_too(self, router, tokenizer, monkeypatch):
        # First apply warms the result cache with the flag off; second apply
        # with the flag on must still gate the cache-hit path.
        messages = _messages(_tool_json(300), suffix_filler_words=40000)
        monkeypatch.delenv("HEADROOM_NET_COST_POLICY", raising=False)
        warm = router.apply([dict(m) for m in messages], tokenizer)
        assert _tool_slot_compressed(warm, messages)

        monkeypatch.setenv("HEADROOM_NET_COST_POLICY", "1")
        gated = router.apply([dict(m) for m in messages], tokenizer)
        assert not _tool_slot_compressed(gated, messages)
        assert any(t.startswith("netcost:skip:") for t in gated.transforms_applied)

    def test_malformed_env_falls_back_to_defaults(self, router, tokenizer, monkeypatch):
        monkeypatch.setenv("HEADROOM_NET_COST_POLICY", "1")
        monkeypatch.setenv("HEADROOM_NET_COST_EXPECTED_READS", "lots")
        monkeypatch.setenv("HEADROOM_NET_COST_P_ALIVE", "warm")
        messages = _messages(_tool_json(300), suffix_filler_words=40000)
        # Must not raise; defaults (R=10, P=1) still block this scenario.
        result = router.apply([dict(m) for m in messages], tokenizer)
        assert not _tool_slot_compressed(result, messages)

    def test_p_alive_zero_disables_penalty(self, router, tokenizer, monkeypatch):
        # Cold cache (P_alive=0): no suffix penalty, mutation always wins.
        monkeypatch.setenv("HEADROOM_NET_COST_POLICY", "1")
        monkeypatch.setenv("HEADROOM_NET_COST_P_ALIVE", "0")
        messages = _messages(_tool_json(300), suffix_filler_words=40000)
        result = router.apply([dict(m) for m in messages], tokenizer)
        assert _tool_slot_compressed(result, messages)
