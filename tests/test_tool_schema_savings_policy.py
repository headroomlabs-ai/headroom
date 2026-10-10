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


def test_reconcile_failure_warns_once_per_type_without_exception_text() -> None:
    import logging

    from headroom.proxy import tool_schema_savings_policy as policy

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    def _broken_count(tools: object) -> int:
        raise RuntimeError("tokenizer exploded on secret-schema-text")

    def _other_broken_count(tools: object) -> int:
        raise ValueError("different failure")

    tool = {"name": "Read", "defer_loading": True, "description": "x"}
    tags = {"tool_search_deferred_tokens": 100}
    booking = (100, {"Read": dict(tool)}, None)

    handler = _Capture(level=logging.DEBUG)
    policy.logger.addHandler(handler)
    old_level = policy.logger.level
    policy.logger.setLevel(logging.DEBUG)
    policy._reconcile_warned_types.clear()
    try:
        # A healthy reconcile logs nothing.
        policy.reconcile_deferred_tokens(dict(tags), booking, [dict(tool)], lambda t: 10)
        assert records == []

        policy.reconcile_deferred_tokens(dict(tags), booking, [dict(tool)], _broken_count)
        policy.reconcile_deferred_tokens(dict(tags), booking, [dict(tool)], _broken_count)
        policy.reconcile_deferred_tokens(dict(tags), booking, [dict(tool)], _other_broken_count)
    finally:
        policy.logger.removeHandler(handler)
        policy.logger.setLevel(old_level)
        policy._reconcile_warned_types.clear()

    warnings = [r for r in records if r.levelno == logging.WARNING]
    debugs = [r for r in records if r.levelno == logging.DEBUG]
    # One warning per exception type, a debug record (with traceback) per failure.
    assert [w.getMessage().split("(")[1].split(")")[0] for w in warnings] == [
        "RuntimeError",
        "ValueError",
    ]
    assert "100 booked tokens" in warnings[0].getMessage()
    assert all(w.exc_info is None for w in warnings)
    assert not any("secret-schema-text" in w.getMessage() for w in warnings)
    assert len(debugs) == 3
    assert all(d.exc_info is not None for d in debugs)


def test_reconcile_warned_set_is_bounded() -> None:
    from headroom.proxy import tool_schema_savings_policy as policy

    policy._reconcile_warned_types.clear()
    try:
        for i in range(policy._RECONCILE_WARNED_MAX + 5):
            policy._log_reconcile_failure(type(f"Err{i}", (Exception,), {})(), 1)
        assert len(policy._reconcile_warned_types) <= policy._RECONCILE_WARNED_MAX
    finally:
        policy._reconcile_warned_types.clear()
