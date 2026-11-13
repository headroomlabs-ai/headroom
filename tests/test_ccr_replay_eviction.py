"""History replay must not retain an ineligible event ahead of live context."""

import time

import pytest

from headroom.ccr.context_tracker import ContextTracker, ContextTrackerConfig


def _track(tracker, key, turn, created_at):
    tracker.track_compression(
        hash_key=key,
        turn_number=turn,
        tool_name="Read",
        original_count=10,
        compressed_count=1,
        workspace_key="workspace",
        query_context="authentication middleware",
        sample_content="authentication middleware",
        compression_created_at=created_at,
    )


def _recommendations(tracker):
    return {
        recommendation.hash_key
        for recommendation in tracker.analyze_query(
            "authentication middleware", current_turn=5, workspace_key="workspace"
        )
    }


@pytest.mark.parametrize("expiry", ["turns", "seconds"])
def test_stale_history_replay_does_not_evict_live_context(expiry):
    tracker = ContextTracker(
        ContextTrackerConfig(
            max_tracked_contexts=2,
            max_turn_distance=3,
            relevance_threshold=0.01,
        )
    )
    now = time.time()
    created_at = now - 400 if expiry == "seconds" else now
    _track(tracker, "old", 1, created_at)
    _track(tracker, "live", 4, now)
    assert _recommendations(tracker) == {"live"}

    _track(tracker, "old", 5, created_at)
    _track(tracker, "new", 5, now)

    assert _recommendations(tracker) == {"live", "new"}


def test_eligible_history_replay_still_protects_recently_used_context():
    tracker = ContextTracker(
        ContextTrackerConfig(
            max_tracked_contexts=2,
            max_turn_distance=3,
            relevance_threshold=0.01,
        )
    )
    now = time.time()
    _track(tracker, "replayed", 3, now)
    _track(tracker, "unused", 4, now)
    _track(tracker, "replayed", 5, now)
    _track(tracker, "new", 5, now)

    assert _recommendations(tracker) == {"replayed", "new"}
