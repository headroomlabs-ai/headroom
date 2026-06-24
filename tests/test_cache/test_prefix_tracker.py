"""Tests for PrefixCacheTracker — cache-aware compression."""

import time

import pytest

from headroom.cache.prefix_tracker import (
    MISS_COLD_START,
    MISS_PREFIX_CHANGE,
    MISS_TTL_EXPIRY,
    MISS_UNKNOWN,
    FreezeStats,
    PrefixCacheTracker,
    PrefixFreezeConfig,
    SessionTrackerStore,
)


class TestPrefixCacheTracker:
    """Test PrefixCacheTracker core functionality."""

    @pytest.fixture
    def tracker(self):
        return PrefixCacheTracker("anthropic")

    @pytest.fixture
    def openai_tracker(self):
        return PrefixCacheTracker("openai")

    def test_turn_0_no_freeze(self, tracker):
        """First turn should never freeze — no cache state yet."""
        assert tracker.get_frozen_message_count() == 0

    def test_turn_1_with_cache_hit_freezes(self, tracker):
        """After turn 1 with cache hits, turn 2 should freeze."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant." * 100},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        # Simulate: provider cached 2000 tokens (system + user)
        token_counts = [1500, 50, 500]

        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=2050,
            messages=messages,
            message_token_counts=token_counts,
        )

        # On turn 2, the first 2 messages (1500 + 50 = 1550 <= 2050) are frozen
        assert tracker.get_frozen_message_count() == 3  # All 3 fit within 2050

    def test_partial_freeze(self, tracker):
        """Only messages that fit within cached tokens are frozen."""
        messages = [
            {"role": "system", "content": "System prompt" * 50},
            {"role": "user", "content": "First question" * 50},
            {"role": "assistant", "content": "First answer" * 50},
            {"role": "user", "content": "Second question"},
        ]
        token_counts = [2000, 500, 500, 50]

        tracker.update_from_response(
            cache_read_tokens=2500,
            cache_write_tokens=0,
            messages=messages,
            message_token_counts=token_counts,
        )

        # 2000 + 500 = 2500 <= 2500, but 2000 + 500 + 500 = 3000 > 2500
        assert tracker.get_frozen_message_count() == 2

    def test_cold_start_no_freeze(self, tracker):
        """If cache_read=0 and cache_write=0, don't freeze."""
        messages = [{"role": "user", "content": "Hello"}]

        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=0,
            messages=messages,
        )

        assert tracker.get_frozen_message_count() == 0

    def test_cache_write_freezes_next_turn(self, tracker):
        """Cache writes (new cache entries) should be frozen on the next turn."""
        messages = [
            {"role": "system", "content": "System" * 200},
            {"role": "user", "content": "Hello"},
        ]
        token_counts = [1500, 50]

        # Turn 1: provider writes to cache (above min threshold)
        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=1550,
            messages=messages,
            message_token_counts=token_counts,
        )

        # Turn 2: should freeze what was written
        assert tracker.get_frozen_message_count() == 2

    def test_min_cached_tokens_threshold(self):
        """Below min_cached_tokens, no freeze."""
        config = PrefixFreezeConfig(min_cached_tokens=2000)
        tracker = PrefixCacheTracker("anthropic", config)

        messages = [{"role": "user", "content": "Hello"}]

        # Turn 1: only 500 tokens cached — below threshold
        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=500,
            messages=messages,
            message_token_counts=[500],
        )

        assert tracker.get_frozen_message_count() == 0

    def test_disabled_config(self):
        """Disabled config always returns 0."""
        config = PrefixFreezeConfig(enabled=False)
        tracker = PrefixCacheTracker("anthropic", config)

        messages = [{"role": "system", "content": "System" * 500}]

        tracker.update_from_response(
            cache_read_tokens=5000,
            cache_write_tokens=0,
            messages=messages,
            message_token_counts=[5000],
        )

        assert tracker.get_frozen_message_count() == 0

    def test_turn_number_increments(self, tracker):
        """Turn number should increment on each update."""
        messages = [{"role": "user", "content": "Hello"}]

        assert tracker._turn_number == 0

        tracker.update_from_response(0, 0, messages)
        assert tracker._turn_number == 1

        tracker.update_from_response(0, 0, messages)
        assert tracker._turn_number == 2

    def test_stats_tracking(self, tracker):
        """Stats should reflect tracker state."""
        stats = tracker.stats
        assert isinstance(stats, FreezeStats)
        assert stats.busts_avoided == 0
        assert stats.tokens_preserved == 0
        assert stats.turn_number == 0

    def test_record_bust_avoided(self, tracker):
        """Recording bust avoided should update stats."""
        tracker.record_bust_avoided(tokens_preserved=5000, compression_foregone=500)
        tracker.record_bust_avoided(tokens_preserved=3000, compression_foregone=200)

        stats = tracker.stats
        assert stats.busts_avoided == 2
        assert stats.tokens_preserved == 8000
        assert stats.compression_foregone_tokens == 700
        assert stats.net_benefit_tokens == 7300

    def test_should_force_compress_outside_frozen(self, tracker):
        """Messages outside frozen prefix should always be compressed."""
        tracker._cached_message_count = 3
        assert tracker.should_force_compress(5, 1000, 200) is True

    def test_should_force_compress_when_savings_exceed_discount(self, tracker):
        """For Anthropic (90% discount), compression must save >90% to be worth it."""
        tracker._cached_message_count = 5

        # 95% savings > 90% discount — should force compress
        assert tracker.should_force_compress(2, 1000, 50) is True

        # 50% savings < 90% discount — should NOT force compress
        assert tracker.should_force_compress(2, 1000, 500) is False

    def test_should_force_compress_openai(self, openai_tracker):
        """For OpenAI (50% discount), compression must save >50% to be worth it."""
        openai_tracker._cached_message_count = 5

        # 60% savings > 50% discount — should force compress
        assert openai_tracker.should_force_compress(2, 1000, 400) is True

        # 40% savings < 50% discount — should NOT force compress
        assert openai_tracker.should_force_compress(2, 1000, 600) is False

    def test_estimate_message_tokens(self):
        """Token estimation should roughly match character / 3.5."""
        messages = [
            {"role": "system", "content": "A" * 350},  # ~100 tokens
            {"role": "user", "content": "B" * 70},  # ~20 tokens
        ]
        counts = PrefixCacheTracker._estimate_message_tokens(messages)
        assert len(counts) == 2
        assert counts[0] > counts[1]  # System should have more tokens

    def test_estimate_content_blocks(self):
        """Token estimation should handle Anthropic content blocks."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "A" * 350},
                    {"type": "text", "text": "B" * 350},
                ],
            },
        ]
        counts = PrefixCacheTracker._estimate_message_tokens(messages)
        assert len(counts) == 1
        assert counts[0] > 100

    def test_estimate_tool_result_content(self):
        """Token estimation should count tool_result content field."""
        tool_content = "x" * 3500  # ~1000 tokens
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": tool_content,
                    }
                ],
            },
        ]
        counts = PrefixCacheTracker._estimate_message_tokens(messages)
        assert len(counts) == 1
        # Should be ~1000 tokens, definitely > 100
        assert counts[0] > 100

    def test_estimate_tool_use_input(self):
        """Token estimation should count tool_use input field."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Read",
                        "input": {"file_path": "/very/long/path/" + "x" * 700},
                    }
                ],
            },
        ]
        counts = PrefixCacheTracker._estimate_message_tokens(messages)
        assert len(counts) == 1
        # Should count the serialized input dict
        assert counts[0] > 50

    def test_estimate_tool_result_nested_blocks(self):
        """Token estimation should handle nested content blocks in tool_result."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "A" * 3500},
                        ],
                    }
                ],
            },
        ]
        counts = PrefixCacheTracker._estimate_message_tokens(messages)
        assert len(counts) == 1
        assert counts[0] > 100

    def test_session_ttl_expiry(self):
        """Tracker should report as expired after TTL."""
        config = PrefixFreezeConfig(session_ttl_seconds=1)
        tracker = PrefixCacheTracker("anthropic", config)

        assert tracker.is_expired is False

        # Simulate time passing
        tracker._last_activity = time.time() - 2
        assert tracker.is_expired is True


class TestSessionTrackerStore:
    """Test SessionTrackerStore management."""

    @pytest.fixture
    def store(self):
        return SessionTrackerStore()

    def test_get_or_create_new(self, store):
        """Should create a new tracker for unknown session."""
        tracker = store.get_or_create("session-1", "anthropic")
        assert isinstance(tracker, PrefixCacheTracker)
        assert tracker.provider == "anthropic"

    def test_get_or_create_existing(self, store):
        """Should return the same tracker for the same session."""
        tracker1 = store.get_or_create("session-1", "anthropic")
        tracker2 = store.get_or_create("session-1", "anthropic")
        assert tracker1 is tracker2

    def test_different_sessions(self, store):
        """Different sessions should get different trackers."""
        tracker1 = store.get_or_create("session-1", "anthropic")
        tracker2 = store.get_or_create("session-2", "openai")
        assert tracker1 is not tracker2
        assert tracker1.provider == "anthropic"
        assert tracker2.provider == "openai"

    def test_active_sessions_count(self, store):
        """Should track the number of active sessions."""
        assert store.active_sessions == 0

        store.get_or_create("s1", "anthropic")
        assert store.active_sessions == 1

        store.get_or_create("s2", "openai")
        assert store.active_sessions == 2

    def test_cleanup_expired(self, store):
        """Should remove expired sessions on cleanup."""
        config = PrefixFreezeConfig(session_ttl_seconds=1)
        store = SessionTrackerStore(default_config=config)

        tracker = store.get_or_create("expired-session", "anthropic")
        tracker._last_activity = time.time() - 2

        # Force cleanup
        store._last_cleanup = 0
        store._maybe_cleanup()

        assert store.active_sessions == 0

    def test_compute_session_id_from_header(self, store):
        """Should use x-headroom-session-id header if present."""

        class MockRequest:
            headers = {"x-headroom-session-id": "explicit-id-123"}

        session_id = store.compute_session_id(
            MockRequest(), "claude-3", [{"role": "user", "content": "Hi"}]
        )
        assert session_id == "explicit-id-123"

    def test_compute_session_id_from_hash(self, store):
        """Should hash model + system prompt as fallback."""

        class MockRequest:
            headers = {}

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

        id1 = store.compute_session_id(MockRequest(), "claude-3", messages)
        id2 = store.compute_session_id(MockRequest(), "claude-3", messages)
        assert id1 == id2  # Stable hash
        assert len(id1) == 16

        # Different model = different session
        id3 = store.compute_session_id(MockRequest(), "gpt-4", messages)
        assert id3 != id1

    def test_compute_session_id_no_system(self, store):
        """Should work without system messages."""

        class MockRequest:
            headers = {}

        messages = [{"role": "user", "content": "Hi"}]
        session_id = store.compute_session_id(MockRequest(), "claude-3", messages)
        assert isinstance(session_id, str)
        assert len(session_id) == 16


class TestMultiTurnScenario:
    """Integration-style tests simulating multi-turn conversations."""

    def test_five_turn_conversation(self):
        """Simulate a 5-turn conversation with growing prefix."""
        tracker = PrefixCacheTracker("anthropic")

        # Turn 1: System + User (cold start, no cache)
        messages_t1 = [
            {"role": "system", "content": "System prompt" * 200},
            {"role": "user", "content": "Question 1"},
        ]
        token_counts_t1 = [2000, 50]

        assert tracker.get_frozen_message_count() == 0  # No freeze on turn 1

        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=2050,
            messages=messages_t1,
            message_token_counts=token_counts_t1,
        )

        # Turn 2: Previous messages cached, new user message added
        messages_t2 = messages_t1 + [
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
        ]
        token_counts_t2 = [2000, 50, 200, 50]

        frozen = tracker.get_frozen_message_count()
        assert frozen == 2  # System + User1 frozen

        tracker.update_from_response(
            cache_read_tokens=2050,
            cache_write_tokens=250,
            messages=messages_t2,
            message_token_counts=token_counts_t2,
        )

        # Turn 3: Even more cached
        messages_t3 = messages_t2 + [
            {"role": "assistant", "content": "Answer 2"},
            {"role": "user", "content": "Question 3"},
        ]
        token_counts_t3 = [2000, 50, 200, 50, 200, 50]

        frozen = tracker.get_frozen_message_count()
        assert frozen == 4  # System + User1 + Asst1 + User2 frozen

        tracker.update_from_response(
            cache_read_tokens=2300,
            cache_write_tokens=250,
            messages=messages_t3,
            message_token_counts=token_counts_t3,
        )

        # Verify turn count
        assert tracker._turn_number == 3

    def test_cache_bust_resets_freeze(self):
        """If cache is busted (0 read, 0 write), freeze should reset."""
        tracker = PrefixCacheTracker("anthropic")

        messages = [
            {"role": "system", "content": "System" * 200},
            {"role": "user", "content": "Hello"},
        ]

        # Turn 1: Cache established
        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=2000,
            messages=messages,
            message_token_counts=[1500, 500],
        )
        assert tracker.get_frozen_message_count() == 2  # Both fit within 2000

        # Turn 2: Cache bust (0 reads, system prompt changed)
        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=0,
            messages=messages,
            message_token_counts=[1500, 500],
        )

        # After a bust with 0 total, freeze should reset
        assert tracker.get_frozen_message_count() == 0


class TestClassifyCacheMiss:
    """Cache-miss attribution (#1313): TTL lapse vs prefix change vs unknown."""

    BASE = [
        {"role": "system", "content": "x" * 4000},
        {"role": "user", "content": "hello"},
    ]
    CHANGED = [
        {"role": "system", "content": "DIFFERENT" * 400},
        {"role": "user", "content": "hello"},
    ]

    def _warm(self, tracker, messages, read=500, write=500):
        """Simulate a turn that left `messages` cached."""
        tracker.update_from_response(
            cache_read_tokens=read, cache_write_tokens=write, messages=messages
        )

    def test_cold_start_is_not_a_miss(self):
        """No prior cached prefix → cold start, is_miss False."""
        tracker = PrefixCacheTracker("anthropic")
        result = tracker.classify_cache_miss(0, self.BASE)
        assert result.is_miss is False
        assert result.reason == MISS_COLD_START

    def test_cache_read_is_a_hit(self):
        """A non-zero read on an expected-cached prefix is a hit, not a miss."""
        tracker = PrefixCacheTracker("anthropic")
        self._warm(tracker, self.BASE)
        result = tracker.classify_cache_miss(800, self.BASE)
        assert result.is_miss is False
        assert result.reason == "hit"

    def test_ttl_expiry_when_idle_exceeds_ttl(self):
        """Idle longer than the cache TTL → ttl_expiry."""
        tracker = PrefixCacheTracker("anthropic")
        self._warm(tracker, self.BASE)
        result = tracker.classify_cache_miss(0, self.BASE, idle_seconds=400)
        assert result.is_miss is True
        assert result.reason == MISS_TTL_EXPIRY
        assert result.ttl_exceeded is True
        assert result.cache_ttl_seconds == 300

    def test_ttl_wins_tie_when_prefix_also_changed(self):
        """When idle past TTL AND prefix changed, TTL expiry wins (docstring)."""
        tracker = PrefixCacheTracker("anthropic")
        self._warm(tracker, self.BASE)
        result = tracker.classify_cache_miss(0, self.CHANGED, idle_seconds=400)
        assert result.reason == MISS_TTL_EXPIRY
        assert result.ttl_exceeded is True
        assert result.prefix_changed is True

    def test_prefix_change_within_ttl(self):
        """Within TTL but the forwarded prefix differs → prefix_change."""
        tracker = PrefixCacheTracker("anthropic")
        self._warm(tracker, self.BASE)
        result = tracker.classify_cache_miss(0, self.CHANGED, idle_seconds=10)
        assert result.is_miss is True
        assert result.reason == MISS_PREFIX_CHANGE
        assert result.prefix_changed is True
        assert result.ttl_exceeded is False

    def test_unknown_when_stable_prefix_within_ttl(self):
        """Within TTL, prefix unchanged, but still no read → unknown."""
        tracker = PrefixCacheTracker("anthropic")
        self._warm(tracker, self.BASE)
        result = tracker.classify_cache_miss(0, self.BASE, idle_seconds=10)
        assert result.is_miss is True
        assert result.reason == MISS_UNKNOWN

    def test_growing_prefix_is_stable(self):
        """A turn that appends to last turn's forwarded prefix is not a change."""
        tracker = PrefixCacheTracker("anthropic")
        self._warm(tracker, self.BASE)
        grown = self.BASE + [{"role": "assistant", "content": "hi back"}]
        result = tracker.classify_cache_miss(0, grown, idle_seconds=10)
        # Prefix preserved (only appended) → not a prefix_change.
        assert result.prefix_changed is False
        assert result.reason == MISS_UNKNOWN

    def test_one_hour_ttl_override(self):
        """cache_ttl_seconds override widens the TTL window (1h breakpoint)."""
        tracker = PrefixCacheTracker("anthropic", PrefixFreezeConfig(cache_ttl_seconds=3600))
        self._warm(tracker, self.BASE)
        # 400s idle is past the 300s default but within 3600s → not TTL expiry.
        result = tracker.classify_cache_miss(0, self.BASE, idle_seconds=400)
        assert result.cache_ttl_seconds == 3600
        assert result.ttl_exceeded is False
        assert result.reason == MISS_UNKNOWN

    def test_resolved_ttl_falls_back_to_provider_default(self):
        assert PrefixCacheTracker("anthropic").resolved_cache_ttl_seconds() == 300
        assert (
            PrefixCacheTracker(
                "anthropic", PrefixFreezeConfig(cache_ttl_seconds=3600)
            ).resolved_cache_ttl_seconds()
            == 3600
        )
