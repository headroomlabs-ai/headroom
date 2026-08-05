"""Tests for the pluggable tokenizer system."""

from __future__ import annotations

import pytest

from headroom.tokenizers import (
    BaseTokenizer,
    CharacterCounter,
    EstimatingTokenCounter,
    TiktokenCounter,
    TokenCounter,
    TokenizerRegistry,
    get_mistral_tokenizer,
    get_tokenizer,
    is_mistral_tokenizer_available,
    list_supported_models,
    register_tokenizer,
)


class TestTiktokenCounter:
    """Tests for TiktokenCounter."""

    def test_init_default_model(self):
        """Test initialization with default model."""
        counter = TiktokenCounter()
        assert counter.model == "gpt-4o"
        assert counter.encoding_name == "o200k_base"

    def test_init_gpt4_model(self):
        """Test initialization with GPT-4."""
        counter = TiktokenCounter("gpt-4")
        assert counter.model == "gpt-4"
        assert counter.encoding_name == "cl100k_base"

    def test_unknown_gpt4_snapshot_uses_cl100k(self):
        """Unknown gpt-4 (non-o, non-turbo) snapshots must use cl100k_base.

        Regression: the prefix matcher scanned MODEL_TO_ENCODING for the
        first key starting with the prefix. For prefix "gpt-4" that matched
        the "gpt-4o" entry first and wrongly returned o200k_base for any
        gpt-4 snapshot not in the table (e.g. a future dated build).
        """
        from headroom.tokenizers.tiktoken_counter import get_encoding_for_model

        assert get_encoding_for_model("gpt-4-2025-01-01") == "cl100k_base"
        assert get_encoding_for_model("gpt-4-future") == "cl100k_base"
        # gpt-4o snapshots still resolve to o200k_base (most-specific first).
        assert get_encoding_for_model("gpt-4o-2099-12-31") == "o200k_base"
        # gpt-4-turbo snapshots use cl100k_base.
        assert get_encoding_for_model("gpt-4-turbo-2099") == "cl100k_base"

    def test_gpt41_and_o4_families_use_o200k(self):
        """gpt-4.1 / gpt-4.5 / o4 use o200k_base, not cl100k_base.

        Regression: gpt-4.1* and gpt-4.5* matched the broad "gpt-4" prefix and
        resolved to cl100k_base, and o4* matched no prefix and fell to the
        cl100k_base default — both wrong encodings for those models.
        """
        from headroom.tokenizers.tiktoken_counter import get_encoding_for_model

        assert get_encoding_for_model("gpt-4.1") == "o200k_base"
        assert get_encoding_for_model("gpt-4.1-mini") == "o200k_base"
        assert get_encoding_for_model("gpt-4.5-preview") == "o200k_base"
        assert get_encoding_for_model("o4-mini") == "o200k_base"
        # A plain gpt-4 snapshot must still use cl100k_base (not shadowed).
        assert get_encoding_for_model("gpt-4-0613") == "cl100k_base"

    def test_count_text_empty(self):
        """Test counting empty text."""
        counter = TiktokenCounter()
        assert counter.count_text("") == 0

    def test_count_text_simple(self):
        """Test counting simple text."""
        counter = TiktokenCounter()
        count = counter.count_text("Hello, world!")
        assert count > 0
        assert count < 10  # Should be a few tokens

    def test_count_text_unicode(self):
        """Test counting text with unicode."""
        counter = TiktokenCounter()
        count = counter.count_text("Hello, 世界!")
        assert count > 0

    def test_count_messages_single(self):
        """Test counting single message."""
        counter = TiktokenCounter()
        messages = [{"role": "user", "content": "Hello!"}]
        count = counter.count_messages(messages)
        assert count > 0

    def test_count_messages_with_tool_calls(self):
        """Test counting messages with tool calls."""
        counter = TiktokenCounter()
        messages = [
            {"role": "user", "content": "Search for Python"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "Python"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "Results...",
            },
        ]
        count = counter.count_messages(messages)
        assert count > 0

    def test_count_messages_image_block_is_not_stringified(self):
        """An Anthropic-style image block must be priced as an image, not text.

        Over the wire the image arrives as a base64 string inside list content.
        The old count_messages else-branch stringified any non-text/non-image_url
        part and tokenized it as text, so a 1MB image counted as ~330K phantom
        tokens. The base handler prices image blocks by a bounded estimate, so the
        count must stay small regardless of the base64 payload size.
        """
        import base64

        counter = TiktokenCounter()
        blob = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200_000).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this screenshot?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": blob,
                        },
                    },
                ],
            }
        ]

        count = counter.count_messages(messages)

        # The base64 blob alone would be tens of thousands of text tokens; a
        # bounded image estimate keeps the whole message well under that.
        assert count < 5000, count
        assert count < len(blob) // 10

    def test_encode_decode_roundtrip(self):
        """Test encode/decode roundtrip."""
        counter = TiktokenCounter()
        text = "Hello, world!"
        tokens = counter.encode(text)
        decoded = counter.decode(tokens)
        assert decoded == text

    def test_count_text_allows_literal_special_tokens(self):
        """count_text must not raise on literal tiktoken special-token strings.

        Regression: passthrough/tool content containing "<|endoftext|>" (or FIM
        markers) made tiktoken raise ValueError under its default
        disallowed_special="all", aborting token counting for the whole request.
        Through the proxy this surfaced as an HTTP 413 compression_refused.
        """
        counter = TiktokenCounter("gpt-4o")
        text = "before <|endoftext|> after <|fim_prefix|> end"
        # Must not raise; markers are counted as ordinary text.
        count = counter.count_text(text)
        assert count > counter.count_text("before  after  end")

    def test_encode_allows_literal_special_tokens(self):
        """encode must treat literal special-token strings as ordinary text."""
        counter = TiktokenCounter("gpt-4o")
        text = "x <|endoftext|> y"
        tokens = counter.encode(text)
        assert isinstance(tokens, list) and len(tokens) > 0
        # Encoding as ordinary text round-trips back to the original literal.
        assert counter.decode(tokens) == text

    def test_repr(self):
        """Test string representation."""
        counter = TiktokenCounter("gpt-4o")
        assert "TiktokenCounter" in repr(counter)
        assert "gpt-4o" in repr(counter)


class TestEstimatingTokenCounter:
    """Tests for EstimatingTokenCounter."""

    def test_init_default(self):
        """Test initialization with defaults."""
        counter = EstimatingTokenCounter()
        assert counter._fixed_ratio is None

    def test_init_fixed_ratio(self):
        """Test initialization with fixed ratio."""
        counter = EstimatingTokenCounter(chars_per_token=3.5)
        assert counter._fixed_ratio == 3.5

    def test_count_text_empty(self):
        """Test counting empty text."""
        counter = EstimatingTokenCounter()
        assert counter.count_text("") == 0

    def test_count_text_simple(self):
        """Test counting simple text."""
        counter = EstimatingTokenCounter()
        text = "Hello, world!"
        count = counter.count_text(text)
        assert count > 0
        # Rough estimate: 13 chars / 4 chars per token ≈ 3-4 tokens
        assert 2 <= count <= 6

    def test_count_text_fixed_ratio(self):
        """Test counting with fixed ratio."""
        counter = EstimatingTokenCounter(chars_per_token=5.0)
        text = "x" * 50  # 50 chars
        count = counter.count_text(text)
        assert count == 10  # 50 / 5 = 10

    def test_count_text_fixed_ratio_prices_cjk(self):
        """The fixed-ratio path must price dense scripts (CJK) like the auto path.

        The registry builds provider counters (Anthropic 3.5, Google 4.0, ...)
        with a fixed ratio; before the fix CJK was priced at the Latin ratio, so
        a large CJK context read as ~40-55% of its true size and could skip
        compression."""
        counter = EstimatingTokenCounter(chars_per_token=3.5)
        cjk = "これはテストです" * 100  # 800 dense-script chars, no ASCII

        count = counter.count_text(cjk)

        # ~1 token per 1.5 chars (CHARS_PER_TOKEN_CJK), not 1 per 3.5.
        assert count == pytest.approx(len(cjk) / 1.5, rel=0.05)
        # Far higher than the old Latin-ratio estimate.
        assert count > len(cjk) / 3.5 * 2

        # ASCII is unaffected by the fixed ratio.
        assert counter.count_text("x" * 35) == 10

    def test_count_text_minimum_one(self):
        """Test minimum of 1 token."""
        counter = EstimatingTokenCounter()
        assert counter.count_text("x") >= 1

    def test_count_messages(self):
        """Test counting messages."""
        counter = EstimatingTokenCounter()
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = counter.count_messages(messages)
        assert count > 0

    def test_json_detection(self):
        """Test JSON content detection."""
        counter = EstimatingTokenCounter()
        json_text = '{"name": "test", "value": 123}'
        # Should use JSON ratio
        count = counter.count_text(json_text)
        assert count > 0

    def test_code_detection(self):
        """Test code content detection."""
        counter = EstimatingTokenCounter()
        code_text = """
def hello():
    return "Hello, world!"
"""
        count = counter.count_text(code_text)
        assert count > 0

    def test_count_text_cjk_not_underestimated(self):
        """CJK text must not be priced at the Latin ~4-chars/token ratio.

        Regression: count_text divided the whole string length by the Latin
        ratio (4.0), so 100 Chinese characters estimated ~25 tokens while real
        tokenizers (cl100k_base / DeepSeek / Qwen) produce ~60-150. Dense
        scripts tokenize at roughly one token per character, so the estimate
        must be far above len/4 and on the order of the character count.
        """
        counter = EstimatingTokenCounter()
        text = "你好世界" * 25  # 100 CJK characters
        count = counter.count_text(text)
        # Old behavior returned len/4 == 25; require clearly above that floor.
        assert count > len(text) / 3
        # And in the right ballpark for one-token-per-char scripts.
        assert count >= int(len(text) * 0.6)

    def test_count_text_cjk_japanese_and_korean(self):
        """Japanese (Kana) and Korean (Hangul) are also dense scripts."""
        counter = EstimatingTokenCounter()
        for text in ("こんにちは世界" * 10, "안녕하세요" * 10):
            count = counter.count_text(text)
            assert count >= int(len(text) * 0.6)

    def test_count_text_mixed_latin_cjk(self):
        """Mixed text prices the Latin part and the CJK part independently."""
        counter = EstimatingTokenCounter()
        latin = "The quick brown fox jumps over the lazy dog. "  # 45 chars
        cjk = "今天天气很好"  # 6 CJK chars
        mixed = counter.count_text(latin + cjk)
        # Must exceed the all-Latin estimate of the same length, since the CJK
        # tail is priced denser than 4 chars/token.
        latin_only = counter.count_text(latin + "x" * len(cjk))
        assert mixed > latin_only

    def test_count_text_latin_unchanged(self):
        """Pure-Latin estimates are unchanged by the CJK adjustment."""
        counter = EstimatingTokenCounter()
        text = "Hello, world!"
        assert 2 <= counter.count_text(text) <= 6

    def test_repr(self):
        """Test string representation."""
        counter = EstimatingTokenCounter()
        assert "EstimatingTokenCounter" in repr(counter)


class TestCharacterCounter:
    """Tests for CharacterCounter."""

    def test_init_default(self):
        """Test initialization with default ratio."""
        counter = CharacterCounter()
        assert counter.chars_per_token == 4.0

    def test_init_custom_ratio(self):
        """Test initialization with custom ratio."""
        counter = CharacterCounter(chars_per_token=3.5)
        assert counter.chars_per_token == 3.5

    def test_count_text(self):
        """Test counting text."""
        counter = CharacterCounter(chars_per_token=4.0)
        text = "x" * 40  # 40 chars
        count = counter.count_text(text)
        assert count == 10  # 40 / 4 = 10

    def test_count_text_empty(self):
        """Test counting empty text."""
        counter = CharacterCounter()
        assert counter.count_text("") == 0


class TestTokenizerRegistry:
    """Tests for TokenizerRegistry."""

    def test_get_openai_model(self):
        """Test getting tokenizer for OpenAI model."""
        tokenizer = get_tokenizer("gpt-4o")
        assert isinstance(tokenizer, TiktokenCounter)

    def test_get_anthropic_model(self):
        """Anthropic (Claude) has no public tokenizer, so it is priced against a
        real BPE proxy (tiktoken o200k_base) rather than a character estimate —
        for consistent, monotone before/after counts. Falls back to the estimator
        only when the tiktoken vocab can't be loaded."""
        tokenizer = get_tokenizer("claude-3-sonnet")
        if isinstance(tokenizer, TiktokenCounter):
            assert tokenizer.encoding_name == "o200k_base"
        else:  # vocab unavailable in this environment → documented fallback
            assert isinstance(tokenizer, EstimatingTokenCounter)

    def test_get_unknown_model_fallback(self):
        """Test fallback for unknown model."""
        tokenizer = get_tokenizer("unknown-model-xyz")
        assert isinstance(tokenizer, EstimatingTokenCounter)

    def test_get_kimi_moonshot_calibrated_estimator(self):
        """Kimi/Moonshot resolves to the calibrated (3.1 chars/tok) estimator
        across every serving form — Fireworks body, litellm slug, native — so
        the size-gates aren't starved by the ~20% under-count of the default
        adaptive estimator (measured on a SWE-bench Kimi-K2.7-code run)."""
        for m in (
            "accounts/fireworks/models/kimi-k2p7-code",  # Fireworks body model
            "fireworks_ai/kimi-k2p7-code-high",  # litellm slug
            "moonshotai/Kimi-K2-Instruct",  # native
            "KIMI-K2P7-CODE",  # case-insensitive
        ):
            tk = get_tokenizer(m)
            assert isinstance(tk, EstimatingTokenCounter), m
            assert tk._fixed_ratio == 3.1, f"{m}: expected 3.1, got {tk._fixed_ratio}"
        # calibrated estimate must beat the default adaptive on Kimi-like code
        # (which the default under-counts): denser ratio -> more tokens.
        code = 'def f(x):\n    return {"a": 1, "b": [2, 3]}\n' * 200
        kimi = get_tokenizer("fireworks_ai/kimi-k2p7-code-high").count_text(code)
        default = get_tokenizer("unknown-model-xyz").count_text(code)
        assert kimi > default, (kimi, default)

    def test_get_with_specific_backend(self):
        """Test forcing specific backend."""
        tokenizer = get_tokenizer("any-model", backend="estimation")
        assert isinstance(tokenizer, EstimatingTokenCounter)

    def test_register_custom_tokenizer(self):
        """Test registering custom tokenizer."""
        custom = EstimatingTokenCounter(chars_per_token=3.0)
        register_tokenizer("my-custom-model", tokenizer=custom)
        retrieved = get_tokenizer("my-custom-model")
        assert retrieved is custom

    def test_list_supported_models(self):
        """Test listing supported models."""
        models = list_supported_models()
        assert isinstance(models, dict)
        assert "gpt-4o" in str(models) or "^gpt-4o" in str(models)

    def test_clear_cache(self):
        """Test clearing tokenizer cache."""
        # Get a tokenizer to populate cache
        get_tokenizer("gpt-4o")
        # Clear cache
        TokenizerRegistry.clear_cache()
        # Should still work after clearing
        tokenizer = get_tokenizer("gpt-4o")
        assert tokenizer is not None


class TestTokenCounterProtocol:
    """Tests for TokenCounter protocol."""

    def test_tiktoken_implements_protocol(self):
        """Test TiktokenCounter implements protocol."""
        counter = TiktokenCounter()
        assert isinstance(counter, TokenCounter)

    def test_estimating_implements_protocol(self):
        """Test EstimatingTokenCounter implements protocol."""
        counter = EstimatingTokenCounter()
        assert isinstance(counter, TokenCounter)

    def test_character_implements_protocol(self):
        """Test CharacterCounter implements protocol."""
        counter = CharacterCounter()
        assert isinstance(counter, TokenCounter)


class TestBaseTokenizer:
    """Tests for BaseTokenizer base class."""

    def test_message_overhead_constant(self):
        """Test message overhead constant."""
        assert BaseTokenizer.MESSAGE_OVERHEAD == 4

    def test_reply_overhead_constant(self):
        """Test reply overhead constant."""
        assert BaseTokenizer.REPLY_OVERHEAD == 3


class TestMistralTokenizer:
    """Tests for Mistral tokenizer using official mistral-common."""

    def test_is_available(self):
        """Test availability check."""
        result = is_mistral_tokenizer_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_get_mistral_tokenizer_class(self):
        """Test getting MistralTokenizer class."""
        MistralTokenizer = get_mistral_tokenizer()
        assert MistralTokenizer is not None
        assert hasattr(MistralTokenizer, "count_text")

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_init_default_model(self):
        """Test initialization with default model."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        assert counter.model == "mistral-large"
        assert counter.version == "v3"

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_init_mixtral_model(self):
        """Test initialization with Mixtral model (uses v1)."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer("mixtral-8x7b")
        assert counter.version == "v1"

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_count_text_empty(self):
        """Test counting empty text."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        assert counter.count_text("") == 0

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_count_text_simple(self):
        """Test counting simple text."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        count = counter.count_text("Hello, world!")
        assert count > 0
        assert count < 10

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_count_text_unicode(self):
        """Test counting text with unicode."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        count = counter.count_text("Hello, 世界!")
        assert count > 0

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_count_messages(self):
        """Test counting messages."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = counter.count_messages(messages)
        assert count > 0

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_count_messages_with_system(self):
        """Test counting messages with system prompt."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        count = counter.count_messages(messages)
        assert count > 0

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_encode_decode_roundtrip(self):
        """Test encode/decode roundtrip."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        text = "Hello, world!"
        tokens = counter.encode(text)
        decoded = counter.decode(tokens)
        assert decoded == text

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_implements_protocol(self):
        """Test MistralTokenizer implements TokenCounter protocol."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer()
        assert isinstance(counter, TokenCounter)

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_repr(self):
        """Test string representation."""
        MistralTokenizer = get_mistral_tokenizer()
        counter = MistralTokenizer("mistral-large")
        assert "MistralTokenizer" in repr(counter)
        assert "mistral-large" in repr(counter)

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_registry_returns_mistral_for_mistral_models(self):
        """Test registry returns Mistral tokenizer for Mistral models."""
        tokenizer = get_tokenizer("mistral-large")
        MistralTokenizer = get_mistral_tokenizer()
        assert isinstance(tokenizer, MistralTokenizer)

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_registry_returns_mistral_for_mixtral(self):
        """Test registry returns Mistral tokenizer for Mixtral models."""
        tokenizer = get_tokenizer("mixtral-8x7b")
        MistralTokenizer = get_mistral_tokenizer()
        assert isinstance(tokenizer, MistralTokenizer)

    @pytest.mark.skipif(
        not is_mistral_tokenizer_available(),
        reason="mistral-common not installed",
    )
    def test_registry_returns_mistral_for_codestral(self):
        """Test registry returns Mistral tokenizer for Codestral models."""
        tokenizer = get_tokenizer("codestral")
        MistralTokenizer = get_mistral_tokenizer()
        assert isinstance(tokenizer, MistralTokenizer)


class TestLargeToolBlobEstimation:
    """Oversized tool blobs are token-estimated without serializing them in full."""

    def test_oversized_tool_blob_count_text_is_bounded(self, monkeypatch):
        """Regression: count_text over a multi-megabyte serialized blob froze the
        event loop (~seconds). json.dumps itself is cheap; count_text over the
        whole string is the cost, so its input must stay bounded for oversized
        blobs.
        """
        tok = EstimatingTokenCounter()
        sizes: list[int] = []
        real_count_text = tok.count_text

        def spy(text):
            sizes.append(len(text))
            return real_count_text(text)

        monkeypatch.setattr(tok, "count_text", spy)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": {"small": "x"}},
                    {"type": "tool_result", "content": {"data": "A" * 4_000_000}},
                ],
            }
        ]
        tok.count_messages(messages)

        assert sizes, "count_text should be exercised"
        # the 4 MB blob must never be counted whole — only its bounded sample
        assert max(sizes) <= tok.SAMPLE_CHARS + tok.SAMPLE_CHUNK

    def test_count_serialized_is_model_accurate_and_keeps_small_exact(self):
        """Small blobs stay exact; large ones track the active counter, not a flat ratio."""
        import json

        tok = EstimatingTokenCounter(chars_per_token=3.5)  # Claude-like ratio
        small = {"k": "v"}
        assert tok._count_serialized(small) == tok.count_text(json.dumps(small))

        # Within 10% of the exact full count (a flat ratio would be ~15% off for 3.5).
        big = {"k": "A" * 200_000}
        exact = tok.count_text(json.dumps(big))
        assert abs(tok._count_serialized(big) - exact) / exact < 0.10

    def test_tool_result_list_recurses_into_image_block(self):
        """A tool that returns an image nests a base64 block inside a
        `tool_result` list. Serializing it prices the base64 as text (a 50-200x
        overcount); recursing into the block prices the image at ~1600 tokens."""
        tok = EstimatingTokenCounter()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "A" * 280_000,  # ~280KB base64
                                },
                            }
                        ],
                    }
                ],
            }
        ]

        count = tok.count_messages(messages)

        # The image is priced structurally (~1600), not as a huge text blob.
        assert count < 5_000

    def test_oversized_estimate_never_overcounts(self):
        """R4 (prefer false negatives): a token-dense head + sparse tail must not
        over-count. Counting per leaf cannot extrapolate a dense front slice to the
        whole the way scaling one sample could.
        """
        import json

        tok = EstimatingTokenCounter()  # content-aware, the hardest case
        blob = {"head": "x1y2-z3w4 " * 4_000, "tail": "A" * 2_000_000}
        exact = tok.count_text(json.dumps(blob))
        assert tok._count_serialized(blob) <= exact

    def test_deeply_nested_blob_does_not_recurse(self):
        """Iterative walk: a deeply nested blob must not raise RecursionError on the
        request path (the earlier recursive helpers died near depth 500).
        """
        deep: dict = {}
        cur = deep
        for _ in range(2_000):
            cur["n"] = {}
            cur = cur["n"]
        cur["leaf"] = "x" * 60_000
        assert EstimatingTokenCounter()._count_serialized(deep) >= 0
