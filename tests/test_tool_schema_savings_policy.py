from __future__ import annotations

from headroom.proxy.tool_schema_savings_policy import (
    TOOL_SCHEMA_SAVINGS_TAGS,
    headline_tokens_saved,
    tool_schema_saved_from_tags,
)


def test_tool_schema_saved_from_tags_sums_headroom_deferral_tags() -> None:
    assert (
        tool_schema_saved_from_tags(
            {
                "tool_search_deferred_tokens": "120",
                "turn_hook_tools_saved_tokens": 30,
                "unrelated": 999,
            }
        )
        == 150
    )


def test_tool_schema_saved_from_tags_ignores_invalid_values() -> None:
    assert (
        tool_schema_saved_from_tags(
            {
                "tool_search_deferred_tokens": "not-an-int",
                "turn_hook_tools_saved_tokens": None,
            }
        )
        == 0
    )


def test_tool_schema_saved_from_tags_rejects_non_mapping_tags() -> None:
    assert tool_schema_saved_from_tags(None) == 0
    assert tool_schema_saved_from_tags([("tool_search_deferred_tokens", 10)]) == 0


def test_tool_schema_savings_tags_are_stable() -> None:
    assert TOOL_SCHEMA_SAVINGS_TAGS == (
        "tool_search_deferred_tokens",
        "turn_hook_tools_saved_tokens",
    )


# ── headline_tokens_saved: the one figure every surface reports ────────────────
# Headroom saves tool-definition tokens in two accounting shapes — compaction
# folds into tokens_saved, deferral is tagged and additive. Both existed before
# but the rule was never written down, so two harnesses dropped their compaction
# savings and three surfaces open-coded the sum. These cases pin the contract.


def test_folded_compaction_is_not_counted_twice() -> None:
    """Compaction is ALREADY inside tokens_saved (handlers fold both endpoints).

    Adding an attribution amount back on top would inflate every tool-heavy turn.
    """
    assert headline_tokens_saved(420, {}) == 420


def test_deferral_tags_are_additive_to_tokens_saved() -> None:
    """Deferral removes schemas count_messages never saw, so it can't be folded."""
    tags = {"tool_search_deferred_tokens": 9639}
    assert headline_tokens_saved(0, tags) == 9639
    assert headline_tokens_saved(1_000, tags) == 10_639


def test_headline_identical_across_harnesses_for_equivalent_work() -> None:
    """A 500-token saving reports as 500 whichever accounting shape produced it.

    Anthropic/Claude Code folds its compaction; a Codex deferral is tagged. Same
    real saving, same headline — that equivalence is the point of the helper.
    """
    anthropic_folded = headline_tokens_saved(500, {})
    codex_tagged = headline_tokens_saved(0, {"tool_search_deferred_tokens": 500})
    assert anthropic_folded == codex_tagged == 500


def test_headline_survives_malformed_tags() -> None:
    for tags in (None, {}, "not-a-dict", {"tool_search_deferred_tokens": None}):
        assert headline_tokens_saved(10, tags) == 10
    assert headline_tokens_saved(10, {"tool_search_deferred_tokens": "abc"}) == 10
    assert headline_tokens_saved(None, None) == 0


def test_headline_clamps_negative_message_savings() -> None:
    """Handlers revert inflation before forwarding, so a negative is a count artifact."""
    assert headline_tokens_saved(-5, {}) == 0
    assert headline_tokens_saved(-5, {"tool_search_deferred_tokens": 100}) == 95


def test_tool_schema_compaction_saves_real_tokens_not_just_bytes() -> None:
    """The premise of folding compaction into tokens_saved on every handler.

    Compaction strips annotation keys ($schema/title/examples). If that only moved
    bytes that tokenize to nothing, the fold would be worthless — so pin a positive
    TOKEN delta on a realistically-shaped tool array, and pin that folding it into
    both endpoints keeps ``tok_before - tok_after == tok_saved`` coherent.
    """
    import json

    from headroom.providers.anthropic import AnthropicProvider
    from headroom.proxy.tool_schema_compaction import compact_tools

    tok = AnthropicProvider().get_token_counter("claude-sonnet-4-6")
    payload = {
        "tools": [
            {
                "name": f"tool_{i}",
                "description": "Does   a    thing.\n\n  Returns text.",
                "input_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "title": f"tool_{i}_schema",
                    "examples": [{"path": "/tmp/x"}, {"path": "/tmp/y"}],
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "File path"}},
                    "required": ["path"],
                },
            }
            for i in range(14)
        ]
    }
    before_tools = payload["tools"]
    body, modified, _bytes_before, _bytes_after = compact_tools(payload)
    assert modified is True

    tool_before = tok.count_text(json.dumps(before_tools, default=str))
    tool_after = tok.count_text(json.dumps(body["tools"], default=str))
    assert tool_after < tool_before, "compaction must shrink tool TOKENS, not only bytes"

    # Mirrors the fold each handler applies at its final recount, with zero message
    # compression — the shape that used to report tok_saved=0 on Claude Code.
    original_tokens = optimized_tokens = 5_000
    if 0 < tool_after < tool_before:
        original_tokens += tool_before
        optimized_tokens += tool_after
    tokens_saved = max(0, original_tokens - optimized_tokens)

    assert tokens_saved == tool_before - tool_after
    assert original_tokens - optimized_tokens == tokens_saved
    assert headline_tokens_saved(tokens_saved, {}) == tokens_saved
