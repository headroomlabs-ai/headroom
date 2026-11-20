"""Cache-safe CCR proactive-expansion and memory injection gating (#2186).

Both proactive-expansion and memory injection append into an existing user
message (``_append_context_to_latest_non_frozen_user_turn`` on Anthropic, the
``append_text_to_latest_user_chat_message`` sibling on OpenAI). Before this
fix, that append ran unconditionally in token mode:

  (a) ``overlay_cached_prefix`` replays last turn's already-injected bytes
      into the leading positions it proved stable, then the handler appends
      into one of those positions AGAIN -> the segment carries two copies,
      diverges from what the provider hashed, and busts from there on.
  (b) ``execute_expansions`` had no per-session dedup, so the same
      compressed-content hash could be re-injected across requests.
      Selection is an atomic claim (reserve on select, release when the
      append does not happen) — a read-only filter plus a record-after-
      append let two concurrent turns of one session both win the same hash.
  (c) Claude Code's ``/compact`` starts a new session (new conversation-
      scoped session id, phase 2), but the process-wide ``ContextTracker``
      is workspace-scoped only, so old compressed content tracked under the
      pre-compact session could surface right back into the fresh one.
      Ownership is a SET of sessions: ``_contexts`` is keyed by content hash
      alone, so a second session tracking identical content must not
      overwrite the first's ownership.

This covers all three at the unit level: the shared guard
(``injection_target_already_forwarded``) together with the index finders that
tell it WHICH position each provider's append helper will mutate, the
per-session expansion dedup tracker, and ``ContextTracker``'s session-scoped
``analyze_query``.
"""

from __future__ import annotations

from headroom.cache.prefix_tracker import PrefixCacheTracker, overlay_cached_prefix
from headroom.ccr.context_tracker import ContextTracker, ContextTrackerConfig
from headroom.proxy.ccr_session_tracker import SessionExpansionDedupTracker
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.helpers import (
    append_text_to_latest_user_chat_message,
    injection_target_already_forwarded,
    latest_non_frozen_user_turn_index,
    latest_user_chat_message_index,
)

CTX = "MEMORY CONTEXT"


def _tracker_with_forwarded_turn(
    forwarded: list[dict[str, object]],
    originals: list[dict[str, object]],
) -> PrefixCacheTracker:
    """A tracker that just forwarded ``forwarded`` for client ``originals``."""
    tracker = PrefixCacheTracker("anthropic")
    tracker.update_from_response(
        cache_read_tokens=0,
        cache_write_tokens=10,
        messages=forwarded,
        original_messages=originals,
    )
    return tracker


# ─── (a) injection_target_already_forwarded ───────────────────────────────


def test_fresh_tail_message_allows_injection() -> None:
    """(a) A genuinely new turn's tail was never forwarded -> inject."""
    turn1 = [{"role": "user", "content": "hello"}]
    tracker = _tracker_with_forwarded_turn(turn1, turn1)

    turn2 = [*turn1, {"role": "assistant", "content": "hi"}, {"role": "user", "content": "more"}]
    assert (
        injection_target_already_forwarded(
            turn2,
            prefix_tracker=tracker,
            target_index=latest_non_frozen_user_turn_index(turn2, frozen_message_count=0),
        )
        is False
    )


def test_same_messages_re_request_blocks_injection() -> None:
    """(a) Re-sending the exact same messages -> tail already forwarded, skip."""
    turn1 = [{"role": "user", "content": "hello"}]
    tracker = _tracker_with_forwarded_turn(turn1, turn1)

    # Identical re-request: same messages, same originals.
    assert (
        injection_target_already_forwarded(
            turn1,
            prefix_tracker=tracker,
            target_index=latest_non_frozen_user_turn_index(turn1, frozen_message_count=0),
        )
        is True
    )


def test_diverged_history_allows_injection() -> None:
    """(a) Client rewrote history -> nothing was replayed there, inject."""
    turn1 = [{"role": "user", "content": "hello"}]
    tracker = _tracker_with_forwarded_turn(turn1, turn1)

    diverged = [{"role": "user", "content": "totally different opener"}]
    assert (
        injection_target_already_forwarded(
            diverged,
            prefix_tracker=tracker,
            target_index=latest_non_frozen_user_turn_index(diverged, frozen_message_count=0),
        )
        is False
    )


def test_no_prior_turn_allows_injection() -> None:
    """(a) Cold tracker (first-ever turn) -> nothing to double-inject into."""
    tracker = PrefixCacheTracker("anthropic")
    turn1 = [{"role": "user", "content": "hello"}]
    assert (
        injection_target_already_forwarded(
            turn1,
            prefix_tracker=tracker,
            target_index=latest_non_frozen_user_turn_index(turn1, frozen_message_count=0),
        )
        is False
    )


def test_guard_is_noop_when_append_helper_has_no_target() -> None:
    """(a) ``target_index < 0`` (nothing would be mutated) -> guard stays out."""
    turn1 = [{"role": "user", "content": "hello"}]
    tracker = _tracker_with_forwarded_turn(turn1, turn1)

    assert (
        injection_target_already_forwarded(turn1, prefix_tracker=tracker, target_index=-1) is False
    )


def test_guard_covers_block_append_merge_shape() -> None:
    """(a) The overlay's block-append merge replays last turn's blocks and
    appends this turn's after them; that position must still be guarded."""
    turn1_orig = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    turn1_fwd = [{"role": "user", "content": [{"type": "text", "text": f"hello\n\n{CTX}"}]}]
    tracker = _tracker_with_forwarded_turn(turn1_fwd, turn1_orig)

    # Same message, one appended block (Claude Code's tool_result growth shape).
    turn2 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "and one more thing"},
            ],
        }
    ]
    merged = overlay_cached_prefix(turn2, turn2, turn1_orig, turn1_fwd)
    assert merged[0]["content"][0]["text"].endswith(CTX), "overlay replayed the injected block"

    assert (
        injection_target_already_forwarded(
            merged,
            prefix_tracker=tracker,
            target_index=latest_non_frozen_user_turn_index(merged, frozen_message_count=0),
        )
        is True
    )


# ─── (a) handler-shaped regressions: the target is not always the tail ─────


def test_anthropic_assistant_prefill_does_not_reinject_into_earlier_user_turn() -> None:
    """(a) Anthropic, assistant-prefill tail: the append helper has no live-zone
    target, so already-injected user content is never appended to twice."""
    turn1_orig = [{"role": "user", "content": "hello"}]
    turn1_fwd = [{"role": "user", "content": f"hello\n\n{CTX}"}]
    tracker = _tracker_with_forwarded_turn(turn1_fwd, turn1_orig)

    # Append-only next turn whose newest message is an assistant prefill.
    turn2_orig = [*turn1_orig, {"role": "assistant", "content": "partial answer"}]
    post_overlay = overlay_cached_prefix(list(turn2_orig), turn2_orig, turn1_orig, turn1_fwd)
    assert post_overlay[0]["content"].endswith(CTX), "overlay replayed the injected user turn"

    target = latest_non_frozen_user_turn_index(post_overlay, frozen_message_count=0)
    assert target == -1, "assistant tail is not an injection target"
    assert (
        injection_target_already_forwarded(
            post_overlay, prefix_tracker=tracker, target_index=target
        )
        is False
    )

    # And the helper itself refuses, so the replayed user turn keeps ONE copy.
    result = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        post_overlay, CTX, frozen_message_count=0
    )
    assert result is post_overlay
    assert result[0]["content"].count(CTX) == 1


def test_openai_non_user_tail_blocks_reinjection_into_replayed_user_message() -> None:
    """(a) OpenAI, non-user tail: the append target is an EARLIER user message
    that the overlay already replayed, so the guard must fire."""
    turn1_orig = [{"role": "user", "content": "hello"}]
    turn1_fwd = [{"role": "user", "content": f"hello\n\n{CTX}"}]
    tracker = _tracker_with_forwarded_turn(turn1_fwd, turn1_orig)

    # Append-only next turn ending on a tool result (newest message is not user).
    turn2_orig = [*turn1_orig, {"role": "tool", "tool_call_id": "c1", "content": "42"}]
    post_overlay = overlay_cached_prefix(list(turn2_orig), turn2_orig, turn1_orig, turn1_fwd)
    assert post_overlay[0]["content"].endswith(CTX), "overlay replayed the injected user turn"

    target = latest_user_chat_message_index(post_overlay)
    assert target == 0, "OpenAI appends to the latest USER message, not the tail"
    assert (
        injection_target_already_forwarded(
            post_overlay, prefix_tracker=tracker, target_index=target
        )
        is True
    )

    # Without the guard the append lands a second copy in that exact position.
    unguarded, appended = append_text_to_latest_user_chat_message(post_overlay, CTX)
    assert appended > 0
    assert unguarded[0]["content"].count(CTX) == 2


def test_openai_fresh_user_tail_after_non_user_history_still_injects() -> None:
    """(a) The guard must not over-block: a genuinely new user turn behind the
    same already-injected history still receives injection."""
    turn1_orig = [{"role": "user", "content": "hello"}]
    turn1_fwd = [{"role": "user", "content": f"hello\n\n{CTX}"}]
    tracker = _tracker_with_forwarded_turn(turn1_fwd, turn1_orig)

    turn2_orig = [
        *turn1_orig,
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "next question"},
    ]
    post_overlay = overlay_cached_prefix(list(turn2_orig), turn2_orig, turn1_orig, turn1_fwd)

    target = latest_user_chat_message_index(post_overlay)
    assert target == 2
    assert (
        injection_target_already_forwarded(
            post_overlay, prefix_tracker=tracker, target_index=target
        )
        is False
    )

    injected, appended = append_text_to_latest_user_chat_message(post_overlay, CTX)
    assert appended > 0
    assert injected[2]["content"].endswith(CTX)
    assert injected[0]["content"].count(CTX) == 1, "replayed prefix untouched"


def test_anthropic_frozen_boundary_excludes_target() -> None:
    """(a) A tail inside the frozen prefix is not a target, so the guard is a
    no-op there and the frozen bytes are never mutated."""
    messages = [{"role": "user", "content": "frozen and final"}]
    assert latest_non_frozen_user_turn_index(messages, frozen_message_count=1) == -1
    assert (
        AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
            messages, CTX, frozen_message_count=1
        )
        is messages
    )


# ─── (b) SessionExpansionDedupTracker ─────────────────────────────────────


def test_expansion_dedup_blocks_repeat_hash_same_session() -> None:
    """(b) Same hash proposed twice in one session -> injected only once."""
    dedup = SessionExpansionDedupTracker(max_sessions=10)

    first = dedup.claim("session-a", ["hash1", "hash2"])
    assert first == ["hash1", "hash2"]

    second = dedup.claim("session-a", ["hash1", "hash2", "hash3"])
    assert second == ["hash3"]


def test_expansion_dedup_is_per_session_not_global() -> None:
    """(b) Different sessions may legitimately need the same expansion."""
    dedup = SessionExpansionDedupTracker(max_sessions=10)
    dedup.claim("session-a", ["hash1"])

    assert dedup.claim("session-b", ["hash1"]) == ["hash1"]
    assert dedup.claim("session-a", ["hash1"]) == []


def test_expansion_dedup_lru_eviction() -> None:
    """(b) Oldest session ages out once the bound is exceeded.

    Probe the surviving session first: a claim is a write, so at
    ``max_sessions=1`` asking about the evicted session re-admits it and
    evicts the survivor in turn.
    """
    dedup = SessionExpansionDedupTracker(max_sessions=1)
    dedup.claim("session-a", ["hash1"])
    dedup.claim("session-b", ["hash2"])  # evicts session-a

    assert dedup.claim("session-b", ["hash2"]) == []
    assert dedup.claim("session-a", ["hash1"]) == ["hash1"]


def test_expansion_dedup_read_refreshes_recency() -> None:
    """(b) A session that is only *queried* still counts as used, otherwise it
    is evicted while active and its hashes get expanded a second time."""
    dedup = SessionExpansionDedupTracker(max_sessions=2)
    dedup.claim("session-a", ["hash1"])
    dedup.claim("session-b", ["hash2"])

    assert dedup.claim("session-a", ["hash1"]) == []  # touches session-a
    dedup.claim("session-c", ["hash3"])  # evicts the LRU (session-b)

    assert dedup.claim("session-a", ["hash1"]) == []
    assert dedup.claim("session-b", ["hash2"]) == ["hash2"]


def test_claim_hides_hash_from_a_concurrent_request_immediately() -> None:
    """(b) Deterministic interleaving: A claims, pauses BEFORE appending, B
    tries the same hash. Exactly one may proceed.

    The check-then-act shape this replaces (read the seen-set, release the
    lock, append, record afterwards) let both requests pass selection and
    both append the same hash — the duplicate the dedup exists to prevent.
    """
    dedup = SessionExpansionDedupTracker(max_sessions=10)

    # Request A: selection happens, then A is suspended mid-flight (expansion
    # execution / guards / append all still ahead of it).
    a_claim = dedup.claim("session-a", ["hash1"])
    assert a_claim == ["hash1"]

    # Request B arrives for the same session while A is still paused.
    b_claim = dedup.claim("session-a", ["hash1"])
    assert b_claim == []

    assert len(a_claim) + len(b_claim) == 1  # exactly one winner


def test_release_makes_a_skipped_claim_eligible_again() -> None:
    """(b) A claim that never reached the wire must not retire the hash.

    Covers every skip path in the handler — expansion returned nothing,
    cache mode, the already-forwarded target guard, an ineligible tail, or
    the body raising — all of which release in the `finally`.
    """
    dedup = SessionExpansionDedupTracker(max_sessions=10)

    claimed = dedup.claim("session-a", ["hash1", "hash2"])
    assert claimed == ["hash1", "hash2"]
    assert dedup.claim("session-a", ["hash1", "hash2"]) == []  # hidden while held

    dedup.release("session-a", claimed)  # A skipped the append

    assert dedup.claim("session-a", ["hash1", "hash2"]) == ["hash1", "hash2"]


def test_release_returns_only_the_unused_subset_of_a_claim() -> None:
    """(b) Partial expansion: the appended hash stays retired, the rest of the
    claim goes back. ``execute_expansions`` may return fewer entries than were
    recommended, and the handler releases exactly the difference."""
    dedup = SessionExpansionDedupTracker(max_sessions=10)

    claimed = dedup.claim("session-a", ["appended", "dropped"])
    assert claimed == ["appended", "dropped"]

    dedup.release("session-a", ["dropped"])

    assert dedup.claim("session-a", ["appended", "dropped"]) == ["dropped"]


def test_release_of_an_unknown_session_is_a_noop() -> None:
    """(b) An evicted session must not resurrect an empty entry on release."""
    dedup = SessionExpansionDedupTracker(max_sessions=10)

    dedup.release("session-gone", ["hash1"])

    assert dedup.claim("session-gone", ["hash1"]) == ["hash1"]


# ─── (c) ContextTracker session-scoped analyze_query (/compact repro) ──────


def test_compact_new_session_does_not_see_old_sessions_expansion() -> None:
    """(c) A fresh session (post-/compact) must not get old content surfaced
    back into it, even though the workspace is unchanged."""
    tracker = ContextTracker(ContextTrackerConfig(enabled=True, proactive_expansion=True))
    tracker.track_compression(
        hash_key="abc123",
        turn_number=1,
        tool_name="Bash",
        original_count=100,
        compressed_count=10,
        workspace_key="ws-repo",
        session_id="session-before-compact",
        query_context="find auth files",
        sample_content="auth_middleware.py handles authentication",
    )

    # New session, same workspace, a query that would otherwise match.
    recs = tracker.analyze_query(
        "what about the authentication middleware?",
        current_turn=1,
        workspace_key="ws-repo",
        session_id="session-after-compact",
    )
    assert recs == []

    # The ORIGINAL session still sees its own tracked content.
    recs_same_session = tracker.analyze_query(
        "what about the authentication middleware?",
        current_turn=1,
        workspace_key="ws-repo",
        session_id="session-before-compact",
    )
    assert len(recs_same_session) == 1
    assert recs_same_session[0].hash_key == "abc123"


def test_analyze_query_without_session_id_keeps_legacy_workspace_only_scoping() -> None:
    """(c) Backward compatibility: omitting session_id on both sides (legacy
    callers, existing tests) preserves workspace-only scoping."""
    tracker = ContextTracker(ContextTrackerConfig(enabled=True, proactive_expansion=True))
    tracker.track_compression(
        hash_key="abc123",
        turn_number=1,
        tool_name="Bash",
        original_count=100,
        compressed_count=10,
        workspace_key="ws-repo",
        query_context="find auth files",
        sample_content="auth_middleware.py handles authentication",
    )

    recs = tracker.analyze_query(
        "what about the authentication middleware?",
        current_turn=1,
        workspace_key="ws-repo",
    )
    assert len(recs) == 1


def test_same_content_in_two_sessions_keeps_both_owners_eligible() -> None:
    """(c) ``_contexts`` is keyed by content hash alone, so two sessions that
    compress identical content share one entry. Tracking it in session B must
    ADD B as an owner, not overwrite A — otherwise A silently loses
    eligibility for a context it still holds."""
    tracker = ContextTracker(ContextTrackerConfig(enabled=True, proactive_expansion=True))
    for session in ("session-a", "session-b"):
        tracker.track_compression(
            hash_key="same-content",  # content-addressed: identical bytes -> one entry
            turn_number=1,
            tool_name="Bash",
            original_count=100,
            compressed_count=10,
            workspace_key="ws-repo",
            session_id=session,
            query_context="find auth files",
            sample_content="auth_middleware.py handles authentication",
        )

    query = "what about the authentication middleware?"
    for session in ("session-a", "session-b"):
        recs = tracker.analyze_query(
            query, current_turn=1, workspace_key="ws-repo", session_id=session
        )
        assert [r.hash_key for r in recs] == ["same-content"], f"{session} lost its own context"

    # A third, unrelated session still gets nothing (the filter still filters).
    assert (
        tracker.analyze_query(
            query, current_turn=1, workspace_key="ws-repo", session_id="session-c"
        )
        == []
    )


# ─── handler wiring tripwire ──────────────────────────────────────────────


def test_anthropic_handler_wires_expansion_dedup_and_forwarded_guard() -> None:
    """The dedup and double-inject guards must stay wired into the handler.

    The unit guards above cannot detect the handler simply not calling them —
    in that case the same context is proactively re-injected on every turn of
    a session (the injection is transient, so each next request looks
    identical) and net savings degrade to zero. Assert the anthropic handler
    source still references every guard, and that the guard is asked about the
    index the append helper actually mutates.
    """
    import inspect

    import headroom.proxy.handlers.anthropic as anthropic_handler

    src = inspect.getsource(anthropic_handler)
    for needle in (
        "get_session_expansion_dedup_tracker",
        ".claim(",
        ".release(",
        "injection_target_already_forwarded",
        "target_index=latest_non_frozen_user_turn_index(",
    ):
        assert needle in src, f"anthropic handler lost expansion-guard wiring: {needle}"


def test_openai_handler_asks_the_guard_about_its_own_append_target() -> None:
    """The OpenAI append target is the latest USER message, not the tail; the
    handler must pass that index so non-user tails stay guarded."""
    import inspect

    import headroom.proxy.handlers.openai as openai_handler

    src = inspect.getsource(openai_handler)
    for needle in (
        "injection_target_already_forwarded",
        "target_index=latest_user_chat_message_index(",
    ):
        assert needle in src, f"openai handler lost injection-guard wiring: {needle}"
