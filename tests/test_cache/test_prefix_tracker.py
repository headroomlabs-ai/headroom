"""Tests for PrefixCacheTracker — cache-aware compression."""

import time

import pytest

from headroom.cache.prefix_tracker import (
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


# =============================================================================
# Shallow-copy tests (Problem 1 fix): no deep-copy of large message lists
# =============================================================================


class TestPrefixTrackerShallowCopy:
    """PrefixCacheTracker stores shallow list snapshots, not deep copies."""

    def _make_tracker_with_messages(
        self,
    ) -> tuple[PrefixCacheTracker, list[dict]]:
        """Return a tracker that has processed one turn with 3 messages."""
        tracker = PrefixCacheTracker("anthropic")
        messages = [
            {"role": "system", "content": "System " * 200},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=2000,
            messages=messages,
            message_token_counts=[1500, 50, 50],
        )
        return tracker, messages

    def test_stored_list_is_not_identical_to_original(self) -> None:
        """Stored list is a separate list object (shallow copy), not the same ref."""
        tracker, messages = self._make_tracker_with_messages()
        stored = tracker.get_last_original_messages()
        # Should be equal in value but distinct list objects.
        assert stored == messages
        assert stored is not messages

    def test_stored_list_shares_dict_references(self) -> None:
        """Message dicts inside the stored list are the same objects (shallow copy)."""
        tracker, messages = self._make_tracker_with_messages()
        stored = tracker.get_last_original_messages()
        # Shallow copy means each dict is the same object as the original.
        for original_msg, stored_msg in zip(messages, stored):
            assert stored_msg is original_msg

    def test_large_message_list_does_not_deep_copy(self) -> None:
        """Storing 200 large messages does not allocate 200 independent copies."""
        large_msg = {"role": "user", "content": "x" * 50_000}
        messages = [large_msg] * 200  # 200 refs to the same large dict
        tracker = PrefixCacheTracker("anthropic")
        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=5000,
            messages=messages,
            message_token_counts=[100] * 200,
        )
        stored = tracker.get_last_original_messages()
        # Every dict in the stored list must be the exact same object as the
        # original (i.e. no deep-copy happened — identity, not equality).
        assert all(m is large_msg for m in stored)
        # The stored list itself should be a fresh list (not the original).
        assert stored is not messages

    def test_accessor_returns_fresh_list_each_call(self) -> None:
        """Each call to get_last_*_messages returns a new list instance."""
        tracker, _ = self._make_tracker_with_messages()
        a = tracker.get_last_original_messages()
        b = tracker.get_last_original_messages()
        assert a is not b
        assert a == b


# =============================================================================
# LRU cap tests for SessionTrackerStore (Problem 1 fix)
# =============================================================================


class TestSessionTrackerStoreLRU:
    """SessionTrackerStore enforces a max-sessions cap with LRU eviction."""

    def test_sessions_below_cap_are_retained(self) -> None:
        store = SessionTrackerStore(max_sessions=5)
        for i in range(5):
            store.get_or_create(f"session-{i}", "anthropic")
        assert store.active_sessions == 5

    def test_session_at_cap_plus_one_evicts_oldest(self) -> None:
        store = SessionTrackerStore(max_sessions=3)
        t1 = store.get_or_create("s1", "anthropic")
        store.get_or_create("s2", "anthropic")
        store.get_or_create("s3", "anthropic")
        # Access s1 to promote it to MRU.
        store.get_or_create("s1", "anthropic")
        # Adding s4 should evict s2 (now LRU after s1's promotion).
        store.get_or_create("s4", "anthropic")
        assert store.active_sessions == 3
        # s1 must still be alive (was promoted).
        assert store.get_or_create("s1", "anthropic") is t1

    def test_cap_is_enforced_with_many_sessions(self) -> None:
        cap = 10
        store = SessionTrackerStore(max_sessions=cap)
        for i in range(50):
            store.get_or_create(f"session-{i}", "anthropic")
        assert store.active_sessions == cap

    def test_existing_session_is_returned_unchanged(self) -> None:
        store = SessionTrackerStore(max_sessions=10)
        t1 = store.get_or_create("s1", "anthropic")
        t1_again = store.get_or_create("s1", "anthropic")
        assert t1 is t1_again
