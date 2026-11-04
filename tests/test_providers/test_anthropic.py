"""Tests for Anthropic provider."""

import pytest


class TestAnthropicModelSanitization:
    def test_sanitize_model_id_removes_ansi_escape_sequences(self):
        from headroom.providers.anthropic import sanitize_anthropic_model_id

        assert sanitize_anthropic_model_id("claude-opus-4-8\x1b[1m") == "claude-opus-4-8"

    def test_sanitize_model_id_removes_displayed_style_suffix(self):
        from headroom.providers.anthropic import sanitize_anthropic_model_id

        assert sanitize_anthropic_model_id("claude-opus-4-8[1m]") == "claude-opus-4-8"
        assert sanitize_anthropic_model_id("glm-5.2[1m]") == "glm-5.2"

    def test_sanitize_model_metadata_cleans_nested_model_ids(self):
        from headroom.providers.anthropic import sanitize_anthropic_model_metadata

        payload = {
            "data": [
                {"id": "claude-opus-4-8\x1b[1m", "display_name": "Claude Opus 4.8"},
                {"id": "claude-sonnet-4-5[1m]"},
            ],
            "model": "claude-opus-4-8[1m]",
        }

        assert sanitize_anthropic_model_metadata(payload) == {
            "data": [
                {"id": "claude-opus-4-8", "display_name": "Claude Opus 4.8"},
                {"id": "claude-sonnet-4-5"},
            ],
            "model": "claude-opus-4-8",
        }


class TestContext1MSuffix:
    """`[1m]` is a 1M-context tier request, not just an ANSI artifact (#1158).

    Claude Code appends `[1m]` to a model id and only then sends the
    `context-1m` beta header, so the real upstream window is 1M even when the
    base model defaults to 200K. The suffix must still be stripped off the wire
    (upstream rejects it, #2027) but must not be lost before we size the budget.
    """

    @pytest.fixture
    def provider(self):
        from headroom.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    def test_1m_suffix_is_detected(self):
        from headroom.providers.anthropic import has_context_1m_suffix

        assert has_context_1m_suffix("claude-sonnet-4-5[1m]")
        assert has_context_1m_suffix("claude-sonnet-4-5[1m][1m]")
        assert not has_context_1m_suffix("claude-sonnet-4-5")

    def test_ansi_artifacts_are_not_mistaken_for_a_tier_request(self):
        from headroom.providers.anthropic import has_context_1m_suffix

        # A dangling reset, a compound style, and a real escape sequence are
        # terminal noise -- none of them means "give me 1M".
        assert not has_context_1m_suffix("claude-sonnet-4-5[0m]")
        assert not has_context_1m_suffix("claude-sonnet-4-5[1;32m]")
        assert not has_context_1m_suffix("\x1b[1mclaude-sonnet-4-5\x1b[0m")

    def test_1m_suffix_raises_a_200k_model_to_1m(self, provider):
        # The regression: sanitizing before the lookup resolved this to the
        # base model's 200K window, so a 1M request was budgeted at 1/5 size.
        assert provider.get_context_limit("claude-sonnet-4-5") == 200_000
        assert provider.get_context_limit("claude-sonnet-4-5[1m]") == 1_000_000

    def test_1m_suffix_never_lowers_an_already_larger_window(self, provider):
        # max(), not a flat assignment: a base model wider than 1M keeps its own.
        assert provider.get_context_limit("claude-opus-5[1m]") >= 1_000_000

    def test_ansi_artifact_does_not_inflate_the_window(self, provider):
        assert provider.get_context_limit("claude-sonnet-4-5[0m]") == 200_000
        assert provider.get_context_limit("\x1b[1mclaude-sonnet-4-5\x1b[0m") == 200_000

    def test_wire_model_id_still_drops_the_suffix(self):
        # Upstream rejects `[1m]`; the tier fix must not regress #2027.
        from headroom.providers.anthropic import sanitize_anthropic_model_id

        assert sanitize_anthropic_model_id("claude-sonnet-4-5[1m]") == "claude-sonnet-4-5"


class TestLongContextPricing:
    """Anthropic's long-context premium above a 200K prompt.

    On the Sonnet 4 / 4.5 family a prompt over 200K re-prices the *whole*
    request -- input, output and cache alike -- at input 2x, output 1.5x,
    cache 2x. Both the LiteLLM path and the manual fallback must apply it, or
    Headroom under-reports the cost of exactly the sessions `[1m]` unlocks.
    """

    @pytest.fixture
    def provider(self):
        from headroom.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    @pytest.fixture
    def manual_provider(self, monkeypatch):
        """Provider with the LiteLLM path disabled, exercising the fallback."""
        import headroom.providers.anthropic as anthropic_module

        monkeypatch.setattr(anthropic_module, "estimate_cost_from_tokens", lambda *a, **k: None)
        return anthropic_module.AnthropicProvider()

    # 100K in / 5K out  -> 100K*$3 + 5K*$15   = $0.375
    # 300K in / 5K out  -> 300K*$6 + 5K*$22.5 = $1.9125  (premium)
    # 300K in of which 150K cached, 5K out
    #                   -> 150K*$6 + 150K*$0.60 + 5K*$22.5 = $1.1025
    _CASES = [
        (100_000, 5_000, 0, 0.3750),
        (300_000, 5_000, 0, 1.9125),
        (300_000, 5_000, 150_000, 1.1025),
    ]

    @pytest.mark.parametrize(("input_tokens", "output_tokens", "cached_tokens", "expected"), _CASES)
    def test_litellm_path(self, provider, input_tokens, output_tokens, cached_tokens, expected):
        cost = provider.estimate_cost(
            input_tokens, output_tokens, "claude-sonnet-4-5", cached_tokens
        )
        assert cost == pytest.approx(expected, rel=1e-4)

    @pytest.mark.parametrize(("input_tokens", "output_tokens", "cached_tokens", "expected"), _CASES)
    def test_manual_fallback_matches_litellm(
        self, manual_provider, input_tokens, output_tokens, cached_tokens, expected
    ):
        cost = manual_provider.estimate_cost(
            input_tokens, output_tokens, "claude-sonnet-4-5", cached_tokens
        )
        assert cost == pytest.approx(expected, rel=1e-4)

    def test_premium_survives_litellm_dropping_the_long_context_rate(self, provider, monkeypatch):
        """A LiteLLM entry without the above-200K rate must not bill a long prompt at the base tier."""
        litellm = pytest.importorskip("litellm")
        import headroom.providers.anthropic as anthropic_module

        # One map for both readers: the guard and litellm's own cost_per_token.
        monkeypatch.setitem(
            litellm.model_cost,
            "claude-sonnet-4-5",
            {
                "litellm_provider": "anthropic",
                "mode": "chat",
                "input_cost_per_token": 3e-06,
                "output_cost_per_token": 1.5e-05,
                "cache_read_input_token_cost": 3e-07,
            },
        )

        assert provider.estimate_cost(300_000, 5_000, "claude-sonnet-4-5", 0) == pytest.approx(
            1.9125, rel=1e-4
        )
        # Below the threshold the LiteLLM path still prices it.
        assert (
            anthropic_module._litellm_lacks_long_context_rate("claude-sonnet-4-5", 100_000) is False
        )

    def test_untiered_model_is_not_charged_a_premium(self, manual_provider):
        # Opus is flat-rated across its whole window: 300K*$5 + 5K*$25 = $1.625.
        cost = manual_provider.estimate_cost(300_000, 5_000, "claude-opus-4-5-20251101", 0)
        assert cost == pytest.approx(1.625, rel=1e-4)

    def test_premium_applies_only_above_the_threshold(self, manual_provider):
        at = manual_provider.estimate_cost(200_000, 0, "claude-sonnet-4-5", 0)
        just_over = manual_provider.estimate_cost(200_001, 0, "claude-sonnet-4-5", 0)
        assert at == pytest.approx(0.60, rel=1e-4)  # 200K * $3
        assert just_over == pytest.approx(1.2000, rel=1e-3)  # re-priced at $6

    def test_1m_suffix_request_is_priced_at_the_premium(self, manual_provider):
        # The two halves of this PR meeting: `[1m]` unlocks the window, and a
        # session that fills it is billed at the long-context rate.
        assert manual_provider.get_context_limit("claude-sonnet-4-5[1m]") == 1_000_000
        cost = manual_provider.estimate_cost(300_000, 5_000, "claude-sonnet-4-5[1m]", 0)
        assert cost == pytest.approx(1.9125, rel=1e-4)

    # Haiku 5.5's tier starts at 100K and multiplies every rate by 5:
    # 100K in / 5K out  -> 100K*$0.10 + 5K*$0.50  = $0.0125
    # 150K in / 5K out  -> 150K*$0.50 + 5K*$2.50  = $0.0875  (long-prompt card)
    # 150K in of which 100K cached, 5K out
    #                   -> 50K*$0.50 + 100K*$0.05 + 5K*$2.50 = $0.0425
    @pytest.mark.parametrize(
        ("input_tokens", "output_tokens", "cached_tokens", "expected"),
        [
            (100_000, 5_000, 0, 0.0125),
            (150_000, 5_000, 0, 0.0875),
            (150_000, 5_000, 100_000, 0.0425),
        ],
    )
    def test_haiku_5_5_long_prompt_card_starts_at_100k(
        self, manual_provider, input_tokens, output_tokens, cached_tokens, expected
    ):
        cost = manual_provider.estimate_cost(
            input_tokens, output_tokens, "claude-haiku-5-5", cached_tokens
        )
        assert cost == pytest.approx(expected, rel=1e-4)

    @pytest.mark.parametrize(
        "model",
        # Gateway ids resolve to the same bare row, so they are guarded too.
        ["claude-haiku-5-5", "anthropic/claude-haiku-5-5", "openrouter/anthropic/claude-haiku-5-5"],
    )
    def test_haiku_5_5_card_survives_litellm_dropping_the_100k_rate(
        self, provider, monkeypatch, model
    ):
        """A Haiku 5.5 entry without the above-100K rate must not bill a long prompt at the base tier."""
        litellm = pytest.importorskip("litellm")
        import headroom.providers.anthropic as anthropic_module

        # The LiteLLM pricer would return this sentinel; litellm caches model
        # info, so the patched row below cannot be relied on to reach it.
        monkeypatch.setattr(anthropic_module, "estimate_cost_from_tokens", lambda *a, **k: 0.42)
        # Keeps the 200K field Sonnet's tier reads, so only Haiku's own field decides.
        monkeypatch.setitem(
            litellm.model_cost,
            "claude-haiku-5-5",
            {
                "litellm_provider": "anthropic",
                "mode": "chat",
                "input_cost_per_token": 1e-07,
                "output_cost_per_token": 5e-07,
                "cache_read_input_token_cost": 1e-08,
                "input_cost_per_token_above_200k_tokens": 5e-07,
            },
        )

        assert provider.estimate_cost(150_000, 5_000, model, 0) == pytest.approx(0.0875, rel=1e-4)
        # At the threshold the LiteLLM path still prices it.
        assert provider.estimate_cost(100_000, 5_000, model, 0) == 0.42

    def test_haiku_5_5_published_100k_rate_stays_on_litellm(self, provider, monkeypatch):
        """With its own above-100K rate published, LiteLLM keeps pricing a long Haiku prompt."""
        litellm = pytest.importorskip("litellm")
        import headroom.providers.anthropic as anthropic_module

        monkeypatch.setitem(
            litellm.model_cost,
            "claude-haiku-5-5",
            {
                "litellm_provider": "anthropic",
                "mode": "chat",
                "input_cost_per_token": 1e-07,
                "output_cost_per_token": 5e-07,
                "input_cost_per_token_above_100k_tokens": 5e-07,
                "output_cost_per_token_above_100k_tokens": 2.5e-06,
            },
        )
        # Stub the LiteLLM pricer: litellm caches model info, so a patched row
        # does not reliably reach its own cost_per_token. The manual table
        # would return $0.0875, so the sentinel shows which path priced it.
        monkeypatch.setattr(anthropic_module, "estimate_cost_from_tokens", lambda *a, **k: 0.42)

        assert provider.estimate_cost(150_000, 5_000, "claude-haiku-5-5", 0) == 0.42

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            # 150K*$0.50 + 5K*$2.50: Haiku 5.5's long-prompt card.
            ("anthropic.claude-haiku-5-5-v1:0", 0.0875),
            ("us.anthropic.claude-haiku-5-5-v1:0", 0.0875),
            ("claude-haiku-5-5-20261001", 0.0875),
            # One version segment away is another model: the `haiku` tier
            # default, 150K*$0.80 + 5K*$4, with no Haiku 5.5 card.
            ("claude-haiku-5", 0.14),
        ],
    )
    def test_wrapped_haiku_5_5_ids_get_the_long_prompt_card(self, manual_provider, model, expected):
        """An id the manual table prices as Haiku 5.5 is also billed on its long-prompt card."""
        cost = manual_provider.estimate_cost(150_000, 5_000, model, 0)
        assert cost == pytest.approx(expected, rel=1e-4)

    def test_reseller_flat_haiku_5_5_row_keeps_its_litellm_rate(self, provider, monkeypatch):
        """A reseller row with no long-prompt tier is flat-rated, not a dropped Anthropic rate."""
        litellm = pytest.importorskip("litellm")
        import headroom.providers.anthropic as anthropic_module

        monkeypatch.setitem(
            litellm.model_cost,
            "perplexity/anthropic/claude-haiku-5-5",
            {
                "litellm_provider": "perplexity",
                "mode": "chat",
                "input_cost_per_token": 1e-07,
                "output_cost_per_token": 5e-07,
            },
        )
        # Stub the LiteLLM pricer (see above); the manual card would be $0.0875.
        monkeypatch.setattr(anthropic_module, "estimate_cost_from_tokens", lambda *a, **k: 0.42)

        cost = provider.estimate_cost(150_000, 5_000, "perplexity/anthropic/claude-haiku-5-5", 0)
        assert cost == 0.42


class TestLiteLLMCostHelper:
    """The shared helper each provider now uses for LiteLLM-backed pricing.

    It replaces a `litellm.completion_cost(prompt_tokens=...)` call that had
    stopped accepting those kwargs and raised TypeError on every invocation.
    """

    def test_returns_none_for_unknown_model(self):
        from headroom.pricing.litellm_pricing import estimate_cost_from_tokens

        assert estimate_cost_from_tokens("no-such-model-xyz", 1000, 1000) is None

    def test_prices_a_known_model(self):
        from headroom.pricing.litellm_pricing import estimate_cost_from_tokens

        # gpt-4o: $2.50/1M in, $10/1M out -> 100K in + 5K out = $0.30
        assert estimate_cost_from_tokens("gpt-4o", 100_000, 5_000) == pytest.approx(0.30, rel=1e-4)

    def test_input_tokens_are_cache_inclusive(self):
        from headroom.pricing.litellm_pricing import estimate_cost_from_tokens

        # The cached portion is a subset of input_tokens, not additional to it,
        # so a fully-cached prompt costs strictly less than an uncached one.
        uncached = estimate_cost_from_tokens("gpt-4o", 100_000, 5_000)
        cached = estimate_cost_from_tokens("gpt-4o", 100_000, 5_000, cached_tokens=50_000)
        assert cached < uncached


class TestAnthropicTokenCounting:
    @pytest.fixture
    def anthropic_provider(self):
        from headroom.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    def test_count_text_fallback(self, anthropic_provider):
        # Without API client, should use tiktoken fallback
        counter = anthropic_provider.get_token_counter("claude-3-5-sonnet-20241022")
        count = counter.count_text("Hello world")
        assert count > 0

    def test_count_messages_basic(self, anthropic_provider):
        counter = anthropic_provider.get_token_counter("claude-3-5-sonnet-20241022")
        messages = [{"role": "user", "content": "Hello"}]
        count = counter.count_messages(messages)
        assert count > 0

    def test_count_messages_tolerates_null_tool_calls(self, anthropic_provider):
        # OpenAI-format assistant messages routinely carry `tool_calls: null`
        # (and occasionally `function: null`) on a no-tool turn. The estimated
        # counter iterated the value after only a key-presence check, so it
        # raised `TypeError: 'NoneType' object is not iterable`.
        counter = anthropic_provider.get_token_counter("claude-3-5-sonnet-20241022")
        messages = [
            {"role": "assistant", "content": "hi", "tool_calls": None},
            {"role": "assistant", "content": "x", "tool_calls": [{"id": "a", "function": None}]},
        ]
        assert counter.count_messages(messages) > 0

    def test_count_text_allows_literal_special_tokens(self, anthropic_provider):
        counter = anthropic_provider.get_token_counter("claude-3-5-sonnet-20241022")
        count = counter.count_text("prefix <|fim_suffix|> suffix")
        assert count > 0


class TestAnthropicModelLimits:
    @pytest.fixture
    def anthropic_provider(self):
        from headroom.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    def test_get_context_limit_claude_sonnet(self, anthropic_provider):
        limit = anthropic_provider.get_context_limit("claude-3-5-sonnet-20241022")
        assert limit == 200000

    def test_get_context_limit_claude_opus(self, anthropic_provider):
        limit = anthropic_provider.get_context_limit("claude-3-opus-20240229")
        assert limit == 200000

    def test_get_context_limit_strips_ansi_model_suffix(self, anthropic_provider):
        assert anthropic_provider.get_context_limit("claude-opus-4-7[1m]") == 1000000

    def test_get_context_limit_claude_5_family(self, anthropic_provider):
        assert anthropic_provider.get_context_limit("claude-fable-5") == 1000000
        assert anthropic_provider.get_context_limit("claude-opus-4-8") == 1000000
        assert anthropic_provider.get_context_limit("claude-sonnet-5") == 1000000

    def test_supports_model_known(self, anthropic_provider):
        assert anthropic_provider.supports_model("claude-3-5-sonnet-20241022")

    def test_supports_model_prefix(self, anthropic_provider):
        assert anthropic_provider.supports_model("claude-3-5-sonnet-latest")

    def test_token_counter_cache_uses_sanitized_model_id(self, anthropic_provider):
        plain = anthropic_provider.get_token_counter("claude-opus-4-7")
        styled = anthropic_provider.get_token_counter("claude-opus-4-7\x1b[1m")

        assert styled is plain


class TestAnthropicCostEstimation:
    @pytest.fixture
    def anthropic_provider(self):
        from headroom.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    def test_estimate_cost_basic(self, anthropic_provider):
        # Probed at 100K, below the 200K long-context threshold: a 1M-token
        # probe would cross it and bill at the premium rate, which is a
        # separate property (covered by TestLongContextPricing).
        cost = anthropic_provider.estimate_cost(
            input_tokens=100_000,
            output_tokens=0,
            model="claude-3-5-sonnet-20241022",
        )
        # $3.00 per 1M input
        assert cost == pytest.approx(0.30, rel=0.1)

    def test_pricing_lookup_strips_ansi_model_suffix(self, anthropic_provider):
        assert anthropic_provider._get_pricing("claude-opus-4-7[1m]") == (
            anthropic_provider._get_pricing("claude-opus-4-7")
        )

    def test_pricing_claude_5_family(self, anthropic_provider):
        fable = anthropic_provider._get_pricing("claude-fable-5")
        assert fable == {"input": 10.00, "output": 50.00, "cached_input": 1.00}

        opus = anthropic_provider._get_pricing("claude-opus-4-8")
        assert opus == {"input": 5.00, "output": 25.00, "cached_input": 0.50}

        # The scheduled rise to $3/$15 on 2026-09-01 was cancelled; $2/$10 is standard.
        sonnet = anthropic_provider._get_pricing("claude-sonnet-5")
        assert sonnet == {"input": 2.00, "output": 10.00, "cached_input": 0.20}

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-fable-5-1", {"input": 10.00, "output": 50.00, "cached_input": 0.25}),
            ("claude-opus-5-5", {"input": 4.00, "output": 20.00, "cached_input": 0.20}),
            ("claude-opus-5", {"input": 5.00, "output": 25.00, "cached_input": 0.50}),
            ("claude-sonnet-5-5", {"input": 2.00, "output": 10.00, "cached_input": 0.20}),
            ("claude-haiku-5-5", {"input": 0.10, "output": 0.50, "cached_input": 0.01}),
            # Suffixed ids must resolve to their own row, not a shorter prefix's.
            ("claude-sonnet-5-5[1m]", {"input": 2.00, "output": 10.00, "cached_input": 0.20}),
            ("claude-haiku-5-5-20261001", {"input": 0.10, "output": 0.50, "cached_input": 0.01}),
            ("claude-fable-5-1-20261001", {"input": 10.00, "output": 50.00, "cached_input": 0.25}),
            ("claude-opus-5-5-20261001", {"input": 4.00, "output": 20.00, "cached_input": 0.20}),
        ],
    )
    def test_pricing_claude_5_point_releases(self, anthropic_provider, model, expected):
        assert anthropic_provider._get_pricing(model) == expected

    @pytest.mark.parametrize(
        "model",
        [
            "claude-fable-5-1",
            "claude-opus-5-5",
            "claude-opus-5",
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
        ],
    )
    def test_context_limit_claude_5_point_releases(self, anthropic_provider, model):
        assert anthropic_provider.get_context_limit(model) == 1_000_000

    def test_shorter_id_does_not_inherit_a_newer_release(self, monkeypatch):
        """``claude-haiku-5`` is not ``claude-haiku-5-5``: no 1M window, no 5.5 card.

        It falls through to the ``haiku`` pattern default instead, as it did
        before Haiku 5.5 had a row.
        """
        import headroom.providers.anthropic as anthropic_module

        monkeypatch.setattr(anthropic_module, "_get_litellm_clients", lambda: (None, None))
        provider = anthropic_module.AnthropicProvider()
        haiku_default = anthropic_module._PATTERN_DEFAULTS["haiku"]
        assert provider.get_context_limit("claude-haiku-5") == haiku_default["context"]
        assert provider._get_pricing("claude-haiku-5") == haiku_default["pricing"]

    @pytest.mark.parametrize(
        ("model", "known_model", "expected"),
        [
            ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5", True),
            ("anthropic.claude-haiku-5-5-v1:0", "claude-haiku-5-5", True),
            ("claude-3-5-sonnet", "claude-3-5-sonnet-20241022", True),
            ("claude-3-5-haiku", "claude-3-5-haiku-latest", True),
            ("claude-sonnet-4", "claude-sonnet-4-20250514", True),
            # One version segment away is another model, in either direction.
            ("claude-haiku-5", "claude-haiku-5-5", False),
            ("claude-sonnet-4", "claude-sonnet-4-6", False),
            ("claude-sonnet-5-5", "claude-sonnet-5", False),
            ("claude-sonnet-5-5-20261001", "claude-sonnet-5", False),
            # A bare fragment is not an alias.
            ("sonnet", "claude-3-5-sonnet-20241022", False),
        ],
    )
    def test_is_release_of(self, model, known_model, expected):
        from headroom.providers.anthropic import _is_release_of

        assert _is_release_of(model, known_model) is expected
