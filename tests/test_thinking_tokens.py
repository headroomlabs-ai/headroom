"""Tests for the thinking/visible output-token split.

The property under test throughout is that **unknown is not zero**. Every
provider reports this differently and one reports it not at all, so the failure
mode to guard against is a fabricated zero quietly asserting "no thinking
happened" and corrupting every average computed over mixed traffic.
"""

from __future__ import annotations

from headroom.proxy.outcome import RequestOutcome
from headroom.proxy.thinking_tokens import (
    ThinkingTokens,
    anthropic_thinking_text,
    extract_from_usage,
    extract_thinking_tokens,
)


class TestOpenAI:
    def test_responses_details(self):
        payload = {
            "usage": {"output_tokens": 900, "output_tokens_details": {"reasoning_tokens": 700}}
        }
        assert extract_thinking_tokens(payload) == ThinkingTokens(tokens=700, inferred=False)

    def test_chat_details(self):
        payload = {
            "usage": {
                "completion_tokens": 500,
                "completion_tokens_details": {"reasoning_tokens": 320},
            }
        }
        assert extract_thinking_tokens(payload).tokens == 320

    def test_zero_reasoning_is_a_real_zero_not_unknown(self):
        """A reported 0 is a claim: the provider says no thinking happened.
        It must not be collapsed into None."""
        payload = {
            "usage": {"completion_tokens": 40, "completion_tokens_details": {"reasoning_tokens": 0}}
        }
        result = extract_thinking_tokens(payload)
        assert result.tokens == 0
        assert result.known is True

    def test_usage_without_details_is_unknown(self):
        payload = {"usage": {"completion_tokens": 40}}
        assert extract_thinking_tokens(payload).tokens is None

    def test_bare_usage_dict_entry_point(self):
        usage = {"output_tokens_details": {"reasoning_tokens": 128}}
        assert extract_from_usage(usage).tokens == 128
        assert extract_from_usage({}).tokens is None
        assert extract_from_usage(None).tokens is None

    def test_booleans_are_not_counts(self):
        """``isinstance(True, int)`` is True in Python; a bool here means a
        malformed payload, not a count of one."""
        usage = {"completion_tokens_details": {"reasoning_tokens": True}}
        assert extract_from_usage(usage).tokens is None


class TestGemini:
    def test_thoughts_token_count(self):
        payload = {"usageMetadata": {"candidatesTokenCount": 200, "thoughtsTokenCount": 1500}}
        assert extract_thinking_tokens(payload).tokens == 1500

    def test_absent_thoughts_is_unknown(self):
        payload = {"usageMetadata": {"candidatesTokenCount": 200}}
        assert extract_thinking_tokens(payload).tokens is None


class TestAnthropic:
    """Anthropic reports no thinking count at all — the interesting case."""

    def test_no_estimator_means_unknown_never_zero(self):
        payload = {
            "content": [
                {"type": "thinking", "thinking": "a b c d e f"},
                {"type": "text", "text": "done"},
            ]
        }
        result = extract_thinking_tokens(payload)
        assert result.tokens is None, "without a tokenizer we do not know, and must say so"
        assert result.inferred is False

    def test_estimator_produces_an_inferred_count(self):
        payload = {"content": [{"type": "thinking", "thinking": "one two three four"}]}
        result = extract_thinking_tokens(payload, estimator=lambda t: len(t.split()))
        assert result.tokens == 4
        assert result.inferred is True, "a derived number must never look reported"

    def test_content_present_but_no_thinking_is_a_real_zero(self):
        """Distinguishable from 'not an Anthropic response': a response that
        carries content blocks genuinely did no thinking."""
        payload = {"content": [{"type": "text", "text": "hello"}]}
        assert extract_thinking_tokens(payload).tokens == 0

    def test_unrecognised_payload_is_unknown(self):
        assert extract_thinking_tokens({"foo": "bar"}).tokens is None
        assert extract_thinking_tokens(None).tokens is None
        assert extract_thinking_tokens("not a dict").tokens is None

    def test_redacted_thinking_still_counts(self):
        """Redacted blocks carry no readable text but were still billed."""
        payload = {"content": [{"type": "redacted_thinking", "data": "xxxx yyyy"}]}
        result = extract_thinking_tokens(payload, estimator=lambda t: len(t.split()))
        assert result.tokens == 2

    def test_thinking_text_skips_malformed_blocks(self):
        payload = {
            "content": [
                "not a dict",
                {"type": "thinking"},
                {"type": "thinking", "thinking": None},
                {"type": "thinking", "thinking": "real"},
            ]
        }
        assert anthropic_thinking_text(payload) == "real"

    def test_a_raising_estimator_degrades_to_unknown(self):
        """Accounting must never cost a caller their response."""

        def boom(_: str) -> int:
            raise RuntimeError("tokenizer exploded")

        payload = {"content": [{"type": "thinking", "thinking": "x"}]}
        assert extract_thinking_tokens(payload, estimator=boom).tokens is None


class TestVisibleSplit:
    def test_visible_from(self):
        assert ThinkingTokens(tokens=700).visible_from(900) == 200

    def test_visible_from_is_none_when_unknown(self):
        assert ThinkingTokens().visible_from(900) is None

    def test_inferred_overshoot_clamps_at_zero(self):
        """An inferred count is on Headroom's tokenizer scale while
        output_tokens is on the provider's; they can disagree on a short
        response and the difference must not go negative."""
        assert ThinkingTokens(tokens=120, inferred=True).visible_from(100) == 0


class TestRequestOutcome:
    def _outcome(self, **kw) -> RequestOutcome:
        base = {
            "request_id": "r1",
            "provider": "anthropic",
            "model": "claude-sonnet-4",
            "original_tokens": 1000,
            "optimized_tokens": 800,
            "output_tokens": 900,
            "tokens_saved": 200,
            "attempted_input_tokens": 1000,
        }
        base.update(kw)
        return RequestOutcome(**base)

    def test_defaults_preserve_every_existing_emit_site(self):
        """The fields are optional so none of the existing construction sites
        need to change; an outcome that says nothing must report unknown."""
        outcome = self._outcome()
        assert outcome.thinking_tokens is None
        assert outcome.thinking_inferred is False
        assert outcome.turn_index == 0
        assert outcome.visible_output_tokens is None

    def test_visible_output_tokens_splits_the_total(self):
        outcome = self._outcome(output_tokens=900, thinking_tokens=700)
        assert outcome.visible_output_tokens == 200

    def test_reported_zero_yields_full_visible(self):
        outcome = self._outcome(output_tokens=900, thinking_tokens=0)
        assert outcome.visible_output_tokens == 900

    def test_turn_index_round_trips(self):
        assert self._outcome(turn_index=7).turn_index == 7


class TestHandlerWiring:
    """The handler helper must actually produce a count.

    These exist because the first implementation called ``Tokenizer()`` with no
    arguments — a TypeError, swallowed by the helper's own ``except``, so every
    Anthropic response silently reported "unknown" forever. Unit tests passed
    throughout, because they inject their own estimator and never exercise the
    real one. Only an end-to-end assertion on the helper catches that.
    """

    def test_helper_returns_an_inferred_count_for_thinking_blocks(self):
        from headroom.proxy.handlers.anthropic import _thinking_tokens_for

        payload = {
            "content": [
                {"type": "thinking", "thinking": "Let me read the parser before editing it."},
                {"type": "text", "text": "done"},
            ]
        }
        result = _thinking_tokens_for(payload)
        assert result.tokens is not None, "the real estimator must be wired, not silently absent"
        assert result.tokens > 0
        assert result.inferred is True

    def test_helper_reports_a_real_zero_when_nothing_was_thought(self):
        from headroom.proxy.handlers.anthropic import _thinking_tokens_for

        result = _thinking_tokens_for({"content": [{"type": "text", "text": "hi"}]})
        assert result.tokens == 0
        assert result.inferred is False

    def test_estimator_is_built_once(self):
        """AnthropicTokenCounter loads a tiktoken encoding in __init__, so
        constructing one per request would put a vocab load in the response
        path."""
        from headroom.proxy.handlers.anthropic import _thinking_estimator

        assert _thinking_estimator() is _thinking_estimator()

    def test_helper_never_raises_on_junk(self):
        from headroom.proxy.handlers.anthropic import _thinking_tokens_for

        for junk in (None, "string", 42, [], {"content": "not a list"}):
            assert _thinking_tokens_for(junk).tokens is None or True
