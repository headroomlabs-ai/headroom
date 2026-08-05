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

    def test_compute_session_id_distinguishes_leading_system_run(self, store):
        """Different dynamic LEADING system messages should not collide."""

        class MockRequest:
            headers = {}

        static_prompt = "framework prompt " * 80
        conv_a = [
            {"role": "system", "content": [{"type": "text", "text": static_prompt}]},
            {"role": "system", "content": [{"type": "text", "text": "context: session A"}]},
            {"role": "user", "content": "hello"},
        ]
        conv_b = [
            {"role": "system", "content": [{"type": "text", "text": static_prompt}]},
            {"role": "system", "content": [{"type": "text", "text": "context: session B"}]},
            {"role": "user", "content": "hello"},
        ]

        id_a = store.compute_session_id(MockRequest(), "claude-3", conv_a)
        id_b = store.compute_session_id(MockRequest(), "claude-3", conv_b)

        assert id_a != id_b

    def test_compute_session_id_distinguishes_top_level_system(self, store):
        """Anthropic carries the system prompt as a top-level field (not a
        role:'system' message). The handler folds it in as a synthetic system
        message so two conversations with the same model and turns but different
        system prompts get distinct ids — otherwise they share one tracker and
        their sticky state cross-contaminates. This exercises that mechanism."""

        class MockRequest:
            headers = {}

        turns = [{"role": "user", "content": "hello"}]

        def with_system(system):
            # Mirror what handlers/anthropic.py does for the top-level system.
            return [{"role": "system", "content": system}, *turns]

        id_a = store.compute_session_id(
            MockRequest(), "claude-3", with_system("You are a Python expert.")
        )
        id_b = store.compute_session_id(
            MockRequest(), "claude-3", with_system("You are a Rust expert.")
        )
        assert id_a != id_b

        # A list-of-text-blocks system folds the same text as the string form.
        id_a_list = store.compute_session_id(
            MockRequest(),
            "claude-3",
            with_system([{"type": "text", "text": "You are a Python expert."}]),
        )
        assert id_a_list == id_a

    def test_compute_session_id_is_stable_when_only_non_system_turns_change(self, store):
        """Appending non-system turns should keep the same fallback session id."""

        class MockRequest:
            headers = {}

        base_messages = [
            {"role": "system", "content": [{"type": "text", "text": "framework prompt"}]},
            {"role": "system", "content": [{"type": "text", "text": "context: session A"}]},
            {"role": "user", "content": "hello"},
        ]
        extended_messages = base_messages + [{"role": "assistant", "content": "hi there"}]

        id1 = store.compute_session_id(MockRequest(), "claude-3", base_messages)
        id2 = store.compute_session_id(MockRequest(), "claude-3", extended_messages)

        assert id1 == id2

    def test_compute_session_id_no_system(self, store):
        """Should work without system messages."""

        class MockRequest:
            headers = {}

        messages = [{"role": "user", "content": "Hi"}]
        session_id = store.compute_session_id(MockRequest(), "claude-3", messages)
        assert isinstance(session_id, str)
        assert len(session_id) == 16

    def test_mid_conversation_system_turns_do_not_rotate_session_id(self, store):
        """Claude Code sends <system-reminder> turns as role:"system" MESSAGES
        interleaved into the history (hook outputs, skills lists, truncation
        notices). Hashing those into the fallback id rotates the session id
        mid-conversation — orphaning the prefix tracker and every other
        session-sticky subsystem (beta headers, CCR/memory registries, the
        compression cache) each time a reminder lands. Only the LEADING run of
        system messages is session identity; later system turns are content.
        """

        class MockRequest:
            headers = {}

        leading = {"role": "system", "content": "You are an agent. " * 40}
        turn1 = [leading, {"role": "user", "content": "read file A"}]
        turn2 = turn1 + [
            {"role": "assistant", "content": "read it"},
            {"role": "user", "content": "tool result ..."},
            {
                "role": "system",
                "content": "<system-reminder>Truncated: PARTIAL view</system-reminder>",
            },
            {"role": "user", "content": "continue"},
        ]

        id1 = store.compute_session_id(MockRequest(), "claude-sonnet-5", turn1)
        id2 = store.compute_session_id(MockRequest(), "claude-sonnet-5", turn2)
        assert id1 == id2


class TestConversationLineageResolution:
    """resolve_tracker: one PrefixCacheTracker per conversation lineage (#2085).

    Concurrent conversations that share a fallback session id (same model +
    same system prompt — e.g. a Claude Code session and its parallel subagents)
    must not thrash one tracker's frozen-prefix state: interleaved turns would
    each see the *other* conversation's prefix, freeze never stabilizes, and
    the provider prompt cache is re-written on every call.

    resolve_tracker keys trackers by message lineage instead: an incoming
    history that extends a known lineage reuses its tracker; a diverging or
    rewritten history gets a fresh one. The session id itself is never changed,
    so session-sticky state keyed on it elsewhere (beta headers, CCR/memory
    registries, the compression cache) is unaffected.
    """

    @pytest.fixture
    def store(self):
        return SessionTrackerStore()

    @staticmethod
    def _history(name: str, turn: int) -> list[dict]:
        """Client-shaped request messages for `turn` (1-based): u0,a0,...,u_{turn-1}."""
        messages: list[dict] = []
        for t in range(turn):
            messages.append({"role": "user", "content": f"[{name}] user {t} " + "x" * 200})
            if t < turn - 1:
                messages.append(
                    {"role": "assistant", "content": f"[{name}] assistant {t} " + "y" * 200}
                )
        return messages

    @staticmethod
    def _block_history(turn: int, cc_on: int | None) -> list[dict]:
        """Block-content history; cache_control breakpoint on message `cc_on` (or none)."""
        messages: list[dict] = []
        for t in range(turn):
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": f"user {t} " + "x" * 200}]}
            )
            if t < turn - 1:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"assistant {t} " + "y" * 200}],
                    }
                )
        if cc_on is not None:
            msg = messages[cc_on]
            blocks = [dict(b) for b in msg["content"]]
            blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
            messages[cc_on] = {**msg, "content": blocks}
        return messages

    def test_interleaved_conversations_resolve_to_independent_trackers(self, store):
        """The #2085 production shape: two conversations, one session id,
        alternating requests. Each must keep its own tracker and per-turn state
        (on a shared tracker, _turn_number would count both conversations)."""
        sid = "shared-fallback-id"
        trackers: dict[str, PrefixCacheTracker] = {}
        for turn in range(1, 5):
            for name in ("A", "B"):
                history = self._history(name, turn)
                tracker = store.resolve_tracker(sid, "anthropic", messages=history)
                trackers.setdefault(name, tracker)
                assert tracker is trackers[name], f"[{name}] turn {turn} switched trackers"
                tracker.update_from_response(
                    cache_read_tokens=1000 * turn,
                    cache_write_tokens=500,
                    messages=history,
                )
        assert trackers["A"] is not trackers["B"]
        assert trackers["A"]._turn_number == 4
        assert trackers["B"]._turn_number == 4

    def test_identical_first_turns_share_until_divergence_then_split(self, store):
        """Templated fan-outs send byte-identical first turns. While histories
        are identical, sharing a tracker is harmless (the provider cache line
        is identical too); they must split as soon as the histories diverge."""
        sid = "shared"
        first = [{"role": "user", "content": "verify the fix " + "p" * 300}]
        t_a1 = store.resolve_tracker(sid, "anthropic", messages=first)
        t_b1 = store.resolve_tracker(sid, "anthropic", messages=first)
        assert t_b1 is t_a1

        a2 = first + [
            {"role": "assistant", "content": "answer A"},
            {"role": "user", "content": "next A"},
        ]
        b2 = first + [
            {"role": "assistant", "content": "answer B"},
            {"role": "user", "content": "next B"},
        ]
        t_a2 = store.resolve_tracker(sid, "anthropic", messages=a2)
        t_b2 = store.resolve_tracker(sid, "anthropic", messages=b2)
        assert t_a2 is not t_b2

        a3 = a2 + [
            {"role": "assistant", "content": "answer A2"},
            {"role": "user", "content": "next A2"},
        ]
        assert store.resolve_tracker(sid, "anthropic", messages=a3) is t_a2

    @pytest.mark.parametrize(
        "cc_turn2",
        [0, -1, None],
        ids=["breakpoint-stays", "breakpoint-moved-to-last", "breakpoint-removed"],
    )
    def test_cache_control_movement_does_not_split_lineage(self, store, cc_turn2):
        """Clients move the cache_control breakpoint every turn; that must not
        read as a rewritten history."""
        sid = "shared"
        t1 = store.resolve_tracker(sid, "anthropic", messages=self._block_history(1, cc_on=0))
        t2 = store.resolve_tracker(
            sid, "anthropic", messages=self._block_history(2, cc_on=cc_turn2)
        )
        assert t2 is t1

    @pytest.mark.parametrize(
        "requote",
        [
            lambda m: {**m, "content": [{"type": "text", "text": m["content"]}]},
            lambda m: {
                **m,
                "content": [{"type": "text", "text": m["content"], "index": 0}],
            },
            lambda m: {
                **m,
                "content": [
                    {"type": "text", "text": m["content"]},
                    {"cachePoint": {"type": "default"}},
                ],
            },
        ],
        ids=["string-to-block-sugar", "streaming-index-annotation", "bedrock-cachepoint-block"],
    )
    def test_representation_churn_does_not_split_lineage(self, store, requote):
        """Clients re-encode history turn-to-turn without changing content
        (litellm flips string<->block sugar, streaming assembly adds `index`,
        Bedrock moves its cachePoint block). Lineage matching must use the
        same canonical equivalence as the cache-stable delta path."""
        sid = "shared"
        first = {"role": "user", "content": "hello " + "x" * 200}
        t1 = store.resolve_tracker(sid, "anthropic", messages=[first])
        grown = [
            requote(first),
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "next"},
        ]
        assert store.resolve_tracker(sid, "anthropic", messages=grown) is t1

    @pytest.mark.parametrize(
        "rewrite",
        [
            lambda h: [{"role": "user", "content": "[summary of the conversation so far]"}],
            lambda h: [h[0], {"role": "assistant", "content": "EDITED"}, *h[2:]],
            lambda h: h[:-2],
        ],
        ids=["compacted", "middle-edited", "truncated"],
    )
    def test_rewritten_history_gets_fresh_tracker(self, store, rewrite):
        """A rewritten history (client-side /compact, edits, truncation) means
        the provider cache line is gone anyway: start a fresh lineage, keep the
        old tracker until TTL, and never touch the session id."""
        sid = "shared"
        history = self._history("A", 3)
        tracker = store.resolve_tracker(sid, "anthropic", messages=history)
        fresh = store.resolve_tracker(sid, "anthropic", messages=rewrite(history))
        assert fresh is not tracker
        assert store.active_sessions == 2

    @pytest.mark.parametrize("messages", [None, []], ids=["none", "empty"])
    def test_resolve_without_messages_matches_legacy_get_or_create(self, store, messages):
        tracker = store.resolve_tracker("sid", "anthropic", messages=messages)
        assert tracker is store.get_or_create("sid", "anthropic")

    def test_resolve_with_freeze_disabled_matches_legacy_get_or_create(self):
        """With prefix freeze off there is no frozen state to protect — skip
        lineage bookkeeping entirely."""
        store = SessionTrackerStore(PrefixFreezeConfig(enabled=False))
        t_a = store.resolve_tracker("sid", "anthropic", messages=self._history("A", 1))
        t_b = store.resolve_tracker("sid", "anthropic", messages=self._history("B", 1))
        assert t_a is t_b
        assert t_a is store.get_or_create("sid", "anthropic")

    def test_over_cap_conversations_share_one_overflow_tracker(self):
        """A fan-out storm on one session id must not grow trackers unbounded:
        past the cap, new conversations share one overflow tracker instead of
        evicting an established lineage."""
        store = SessionTrackerStore(PrefixFreezeConfig(max_lineages_per_session=4))
        sid = "storm"
        overflow = set()
        for i in range(9):
            tracker = store.resolve_tracker(
                sid, "anthropic", messages=[{"role": "user", "content": f"task {i} " + "z" * 100}]
            )
            if i >= 4:
                overflow.add(id(tracker))
        assert store.active_sessions == 5  # 4 lineages + 1 shared overflow
        assert len(overflow) == 1

    def test_established_lineages_survive_cap_overflow(self):
        """Filling the family must never evict an established conversation —
        under round-robin any eviction victim is the next requester, which
        would degrade EVERY conversation to a cold tracker per turn."""
        store = SessionTrackerStore(PrefixFreezeConfig(max_lineages_per_session=2))
        sid = "shared"
        parent_history = self._history("parent", 4)
        parent = store.resolve_tracker(sid, "anthropic", messages=parent_history)
        shorty = store.resolve_tracker(sid, "anthropic", messages=self._history("shorty", 1))
        # Third divergent conversation lands on the overflow tracker; both
        # established lineages keep their trackers.
        newcomer = store.resolve_tracker(sid, "anthropic", messages=self._history("new", 1))
        assert newcomer is not parent
        assert newcomer is not shorty
        grown = parent_history + [
            {"role": "assistant", "content": "[parent] assistant 3 " + "y" * 200},
            {"role": "user", "content": "[parent] user 4 " + "x" * 200},
        ]
        assert store.resolve_tracker(sid, "anthropic", messages=grown) is parent
        assert (
            store.resolve_tracker(sid, "anthropic", messages=self._history("shorty", 2)) is shorty
        )

    def test_no_cliff_at_cap_plus_one_round_robin(self):
        """cap+1 conversations round-robining turns: the in-cap conversations
        keep their trackers on every round (no eviction churn); only the
        over-cap tail shares the overflow tracker."""
        store = SessionTrackerStore(PrefixFreezeConfig(max_lineages_per_session=3))
        sid = "shared"
        trackers: dict[str, PrefixCacheTracker] = {}
        for turn in range(1, 4):
            for name in ("A", "B", "C", "D"):
                tracker = store.resolve_tracker(
                    sid, "anthropic", messages=self._history(name, turn)
                )
                if turn == 1:
                    trackers[name] = tracker
                elif name != "D":
                    assert tracker is trackers[name], f"[{name}] turn {turn} lost its tracker"
                else:
                    assert tracker is trackers["D"]  # stable overflow tracker

    def test_empty_canonical_history_falls_back_to_legacy(self):
        """A history whose every message projects away (pure directive
        content) carries no lineage signal — behave like get_or_create."""
        store = SessionTrackerStore()
        tracker = store.resolve_tracker("sid", "anthropic", messages=[{}])
        assert tracker is store.get_or_create("sid", "anthropic")

    def test_nan_in_tool_payload_does_not_split_lineage(self):
        """json.loads accepts bare NaN, and NaN != NaN — a resent history
        containing one must still read as the same conversation."""
        store = SessionTrackerStore()
        sid = "shared"
        turn1 = [
            {"role": "user", "content": "run the tool " + "x" * 200},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "score", "input": {"v": float("nan")}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            },
        ]
        t1 = store.resolve_tracker(sid, "anthropic", messages=turn1)
        turn2 = turn1 + [
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "next"},
        ]
        assert store.resolve_tracker(sid, "anthropic", messages=turn2) is t1

    def test_expired_lineages_are_cleaned_up(self):
        config = PrefixFreezeConfig(session_ttl_seconds=1)
        store = SessionTrackerStore(default_config=config)
        for name in ("A", "B"):
            tracker = store.resolve_tracker("sid", "anthropic", messages=self._history(name, 1))
            tracker._last_activity = time.time() - 2

        store._last_cleanup = 0
        store._maybe_cleanup()
        assert store.active_sessions == 0

        # The lineage index must not resurrect evicted trackers: extending an
        # evicted conversation starts cold.
        fresh = store.resolve_tracker("sid", "anthropic", messages=self._history("A", 2))
        assert fresh._turn_number == 0

    def test_shared_session_id_is_not_rotated(self, store):
        """Composition guard: lineage resolution must not leak into session-id
        derivation — beta stickiness and the compression cache key on it."""

        class MockRequest:
            headers = {}

        msgs_a = [{"role": "system", "content": "S"}, *self._history("A", 1)]
        msgs_b = [{"role": "system", "content": "S"}, *self._history("B", 1)]
        id_a = store.compute_session_id(MockRequest(), "claude-3", msgs_a)
        id_b = store.compute_session_id(MockRequest(), "claude-3", msgs_b)
        assert id_a == id_b


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
