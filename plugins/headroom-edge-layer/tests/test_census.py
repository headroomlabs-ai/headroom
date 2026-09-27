from __future__ import annotations

from pathlib import Path

import census

from .fixtures import build_basic_session


def test_census_groups_by_tool_name_not_content_type(tmp_path: Path) -> None:
    build_basic_session(tmp_path)
    cost_tracker = census.CostTracker() if census.CostTracker is not None else None
    result = census.CensusResult()

    for session_file in sorted(tmp_path.glob("*/*.jsonl")):
        replay = census.load_session_replay(session_file)
        result.sessions_scanned += 1
        assert replay is not None
        splits = census._read_cache_creation_splits(session_file)
        census.census_session(replay, result, cost_tracker, splits)

    assert set(result.bucket_stats) == {"file_read", "grep", "web", "other"}
    # The fixture's file_read bucket holds the JSON-file-shaped content it
    # would have if the read target were JSON — grouping by tool name (Read)
    # keeps it there rather than sorting it into a content-type bucket.
    assert result.bucket_stats["file_read"].occurrences == 2
    assert result.bucket_stats["file_read"].repeat_reads == 1
    assert result.bucket_stats["grep"].occurrences == 1
    assert result.bucket_stats["grep"].oversized_outputs == 0  # 60 lines, cap is >100
    assert result.bucket_stats["web"].oversized_outputs == 1  # > 2000 tokens


def test_lifetime_cost_weight_favors_early_turns(tmp_path: Path) -> None:
    """A result read on turn 0 of a 5-turn session should be weighted higher
    than a fresh-token-identical result read on the last turn (1.25 + 0.1 *
    turns_left — the doc's "~7x its size early, ~1.25x on the last turn").
    """
    build_basic_session(tmp_path)
    cost_tracker = census.CostTracker() if census.CostTracker is not None else None
    result = census.CensusResult()
    session_file = next(tmp_path.glob("*/*.jsonl"))
    replay = census.load_session_replay(session_file)
    assert replay is not None
    splits = census._read_cache_creation_splits(session_file)
    census.census_session(replay, result, cost_tracker, splits)

    file_read = result.bucket_stats["file_read"]
    # Two occurrences, same token size (identical repeat read) — the
    # lifetime-weighted total must exceed 1.25x the fresh total, since at
    # least the first occurrence has turns_left > 0.
    assert file_read.lifetime_weighted_tokens > 1.25 * file_read.fresh_tokens


def test_fallback_pricing_used_for_unknown_model() -> None:
    estimate = census.estimate_request_cost(
        None,  # no cost tracker at all
        model="claude-sonnet-4-5-unknown-suffix",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=0,
        cache_write_5m_tokens=0,
        cache_write_1h_tokens=0,
    )
    assert estimate.source == "fallback"
    assert estimate.usd == 3.0 + 15.0  # claude-sonnet fallback: $3 in + $15 out per 1M


def test_unpriced_model_is_flagged_not_zero() -> None:
    estimate = census.estimate_request_cost(
        None,
        model="totally-unknown-provider-xyz",
        input_tokens=1000,
        output_tokens=100,
        cache_read_tokens=0,
        cache_write_5m_tokens=0,
        cache_write_1h_tokens=0,
    )
    assert estimate.source == "unpriced"
    assert estimate.usd is None  # never a silent $0


def test_cache_creation_split_is_read_from_raw_jsonl(tmp_path: Path) -> None:
    session_file = build_basic_session(tmp_path)
    splits = census._read_cache_creation_splits(session_file)
    assert splits["req2"] == (30, 20)  # 5m, 1h — from the fixture's grep turn
