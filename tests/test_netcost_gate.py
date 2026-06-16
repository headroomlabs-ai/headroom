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

    def test_nonfinite_env_falls_back_to_defaults(self, router, tokenizer, monkeypatch):
        # ``float("inf")``/``float("nan")`` parse without ValueError; the gate
        # must reject them and fall back to defaults so telemetry isn't
        # poisoned. With R=10/P=1 defaults this scenario still skips.
        monkeypatch.setenv("HEADROOM_NET_COST_POLICY", "1")
        monkeypatch.setenv("HEADROOM_NET_COST_EXPECTED_READS", "inf")
        monkeypatch.setenv("HEADROOM_NET_COST_P_ALIVE", "nan")
        messages = _messages(_tool_json(300), suffix_filler_words=40000)
        result = router.apply([dict(m) for m in messages], tokenizer)
        assert not _tool_slot_compressed(result, messages)
        # Marker must be a bounded band, never a raw float / "nan".
        skip_markers = [t for t in result.transforms_applied if t.startswith("netcost:skip:")]
        assert skip_markers
        assert all(m.split(":")[-1] in _GAIN_BANDS for m in skip_markers)


_GAIN_BANDS = {
    "0",
    "lt100",
    "lt1k",
    "lt10k",
    "gte10k",
    "neg_lt100",
    "neg_lt1k",
    "neg_lt10k",
    "neg_gte10k",
    "nan",
}


class TestNetCostHelpers:
    def test_gain_bucket_bands_and_sign(self):
        from headroom.transforms.content_router import _gain_bucket

        assert _gain_bucket(0) == "0"
        assert _gain_bucket(50) == "lt100"
        assert _gain_bucket(500) == "lt1k"
        assert _gain_bucket(5000) == "lt10k"
        assert _gain_bucket(50000) == "gte10k"
        assert _gain_bucket(-50) == "neg_lt100"
        assert _gain_bucket(-50000) == "neg_gte10k"
        assert _gain_bucket(float("nan")) == "nan"
        assert _gain_bucket(float("inf")) == "nan"

    def test_message_tokens_block_list_beats_repr(self, tokenizer):
        # str(content) over a block list counts repr punctuation/type names;
        # the block-aware helper counts only the text-bearing payload.
        from headroom.transforms.content_router import _netcost_message_tokens

        text = "word " * 200
        block_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image", "source": {"data": "x" * 500}},
            ],
        }
        helper = _netcost_message_tokens(block_msg, tokenizer)
        text_only = tokenizer.count_text(text)
        # Helper tracks the text payload closely; the image block adds only a
        # small repr proxy, far less than stringifying the whole list.
        assert abs(helper - text_only) < text_only * 0.5
        assert helper < tokenizer.count_text(str(block_msg["content"]))

    def test_message_tokens_tool_result_blocks(self, tokenizer):
        from headroom.transforms.content_router import _netcost_message_tokens

        payload = "log line " * 100
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": payload}],
                }
            ],
        }
        assert _netcost_message_tokens(msg, tokenizer) >= tokenizer.count_text(payload) * 0.8

    def test_message_tokens_string_content(self, tokenizer):
        from headroom.transforms.content_router import _netcost_message_tokens

        s = "plain string content " * 50
        assert _netcost_message_tokens({"role": "user", "content": s}, tokenizer) == (
            tokenizer.count_text(s)
        )


def _frozen_messages(tool_content: str, suffix_filler_words: int) -> list[dict]:
    """A short conversation whose compressible tool dump sits *inside* the
    frozen prefix (index 1, with frozen_message_count=2)."""
    suffix = "analysis context word " * suffix_filler_words
    return [
        {"role": "user", "content": "fetch the records"},
        {"role": "tool", "content": tool_content},
        {"role": "user", "content": suffix},
        {"role": "user", "content": "summarize"},
    ]


class TestNetCostFrozenUnlock:
    """#856 P2b: let formula-positive deep edits through the frozen floor."""

    def test_flag_off_frozen_stays_frozen(self, router, tokenizer, monkeypatch):
        # Default (flag off): a message in the prefix cache is never mutated,
        # however compressible it is — the binary floor wins.
        monkeypatch.delenv("HEADROOM_NET_COST_POLICY", raising=False)
        messages = _frozen_messages(_tool_json(2000), suffix_filler_words=5)
        result = router.apply([dict(m) for m in messages], tokenizer, frozen_message_count=2)
        assert not _tool_slot_compressed(result, messages)
        assert "router:netcost_frozen_unlock" not in result.transforms_applied

    def test_flag_on_unlocks_when_shave_dominates(self, router, tokenizer, monkeypatch):
        # Huge shave deep in the frozen zone, tiny suffix after -> the
        # break-even gate clears the deep edit and it proceeds (the "50K
        # stale dump, 10K suffix" user story).
        monkeypatch.setenv("HEADROOM_NET_COST_POLICY", "1")
        messages = _frozen_messages(_tool_json(2000), suffix_filler_words=5)
        result = router.apply([dict(m) for m in messages], tokenizer, frozen_message_count=2)
        assert _tool_slot_compressed(result, messages)
        assert "router:netcost_frozen_unlock" in result.transforms_applied

    def test_flag_on_keeps_frozen_when_suffix_dominates(self, router, tokenizer, monkeypatch):
        # Modest shave, big cached suffix -> gate runs on the unlocked slot
        # but rejects it. The frozen message is left byte-identical and no
        # unlock marker is emitted, proving the floor opened yet the formula
        # still protected the cache.
        monkeypatch.setenv("HEADROOM_NET_COST_POLICY", "1")
        messages = _frozen_messages(_tool_json(300), suffix_filler_words=40000)
        result = router.apply([dict(m) for m in messages], tokenizer, frozen_message_count=2)
        assert not _tool_slot_compressed(result, messages)
        assert "router:netcost_frozen_unlock" not in result.transforms_applied
        assert any(t.startswith("netcost:skip:") for t in result.transforms_applied)

    def test_flag_on_block_content_frozen_stays_frozen(self, router, tokenizer, monkeypatch):
        # The gate is wired into the string and parallel-merge paths only;
        # block-list frozen content (whose per-block cache_control contract
        # is not net-cost aware) stays frozen even with a tiny suffix.
        monkeypatch.setenv("HEADROOM_NET_COST_POLICY", "1")
        big = "log line of output " * 400
        messages = [
            {"role": "user", "content": "fetch"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": big}],
                    }
                ],
            },
            {"role": "user", "content": "summarize"},
        ]
        original = [dict(m) for m in messages]
        result = router.apply([dict(m) for m in messages], tokenizer, frozen_message_count=2)
        assert result.messages[1]["content"] == original[1]["content"]
        assert "router:netcost_frozen_unlock" not in result.transforms_applied
