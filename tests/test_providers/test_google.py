"""Tests for Google (Gemini) provider."""

from __future__ import annotations

import pytest

from headroom.providers import GoogleProvider


class TestGoogleProvider:
    """Tests for GoogleProvider."""

    @pytest.fixture
    def provider(self):
        """Create Google provider without client (estimation mode)."""
        return GoogleProvider()

    def test_name(self, provider):
        assert provider.name == "google"

    def test_get_token_counter(self, provider):
        counter = provider.get_token_counter("gemini-2.0-flash")
        assert counter is not None
        assert counter.count_text("Hello, world!") > 0

    def test_get_token_counter_is_cached_per_model(self, provider):
        """Counters are memoized per model so their internal token-count cache
        survives across requests, matching the Anthropic/OpenAI providers."""
        first = provider.get_token_counter("gemini-2.0-flash")
        assert provider.get_token_counter("gemini-2.0-flash") is first
        # A different model gets its own counter.
        other = provider.get_token_counter("gemini-1.5-pro")
        assert other is not first
        assert provider.get_token_counter("gemini-1.5-pro") is other

    def test_get_token_counter_rejects_unknown_model(self, provider):
        with pytest.raises(ValueError):
            provider.get_token_counter("gpt-4o")
