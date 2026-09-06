"""Tests for the x-headroom-keep-last-turns feature (issue #2858).

Covers the apply_keep_last_turns() helper that both anthropic.py and
openai.py call before the optimized_messages assignment.
"""

from __future__ import annotations

from headroom.proxy.helpers import apply_keep_last_turns

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _turns(n: int) -> list[dict]:
    """Build a plausible conversation: n user+assistant pairs + trailing user."""
    msgs: list[dict] = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    msgs.append({"role": "user", "content": "final question"})
    return msgs


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------


def test_empty_messages_returns_unchanged():
    result, dropped = apply_keep_last_turns([], n=0)
    assert result == []
    assert dropped == 0


def test_n_negative_is_noop():
    msgs = _turns(3)
    result, dropped = apply_keep_last_turns(msgs, n=-1)
    assert result is msgs
    assert dropped == 0


def test_n_larger_than_history_is_noop():
    """When n >= existing turns no messages are dropped."""
    msgs = _turns(2)  # 5 messages total
    result, dropped = apply_keep_last_turns(msgs, n=10)
    assert result is msgs
    assert dropped == 0


def test_exact_turns_match_is_noop():
    """Asking for exactly the number of turns present keeps everything."""
    msgs = _turns(3)  # 3 turns → 7 messages
    result, dropped = apply_keep_last_turns(msgs, n=3)
    assert result is msgs
    assert dropped == 0


# ---------------------------------------------------------------------------
# Trimming cases
# ---------------------------------------------------------------------------


def test_n0_keeps_only_final_user_message():
    """n=0 means 'no prior turns' → only the trailing user message survives."""
    msgs = _turns(3)  # 7 messages
    result, dropped = apply_keep_last_turns(msgs, n=0)
    assert len(result) == 1
    assert result[0]["content"] == "final question"
    assert dropped == 6


def test_n1_keeps_one_prior_turn_plus_current():
    """n=1 keeps the most recent user+assistant pair plus the current user message."""
    msgs = _turns(3)  # q0 a0 q1 a1 q2 a2 q_final (7 msgs)
    result, dropped = apply_keep_last_turns(msgs, n=1)
    # Keeps: a2, q_final = indices 5, 6 → but formula gives tail = 7-1-1*2=4
    # messages[4:] = q2 a2 q_final  (3 messages)
    assert len(result) == 3
    assert result[-1]["content"] == "final question"
    assert dropped == 4


def test_n2_keeps_two_prior_turns_plus_current():
    msgs = _turns(4)  # 9 messages
    result, dropped = apply_keep_last_turns(msgs, n=2)
    # tail = 9-1-2*2 = 4; messages[4:] = 5 messages
    assert len(result) == 5
    assert result[-1]["content"] == "final question"
    assert dropped == 4


def test_trailing_user_message_always_present():
    """The current (final) user message is never dropped regardless of n."""
    for n in range(5):
        msgs = _turns(5)
        result, _ = apply_keep_last_turns(msgs, n=n)
        assert result[-1] == {"role": "user", "content": "final question"}, f"n={n}"


def test_total_dropped_plus_kept_equals_original():
    msgs = _turns(5)  # 11 messages
    for n in range(6):
        result, dropped = apply_keep_last_turns(msgs, n=n)
        assert len(result) + dropped == len(msgs), f"n={n}"


# ---------------------------------------------------------------------------
# Single-message conversation
# ---------------------------------------------------------------------------


def test_single_message_conversation_n0():
    """A single-message conversation (no history) is never trimmed."""
    msgs = [{"role": "user", "content": "hello"}]
    result, dropped = apply_keep_last_turns(msgs, n=0)
    assert result is msgs
    assert dropped == 0


def test_single_message_conversation_large_n():
    msgs = [{"role": "user", "content": "hello"}]
    result, dropped = apply_keep_last_turns(msgs, n=100)
    assert result is msgs
    assert dropped == 0


# ---------------------------------------------------------------------------
# Return-value contract
# ---------------------------------------------------------------------------


def test_no_trim_returns_original_object():
    """When nothing is dropped the original list is returned (identity, not copy)."""
    msgs = _turns(2)
    result, dropped = apply_keep_last_turns(msgs, n=100)
    assert result is msgs
    assert dropped == 0


def test_trim_returns_new_slice():
    """When trimming, a new slice (not the original object) is returned."""
    msgs = _turns(3)
    result, dropped = apply_keep_last_turns(msgs, n=0)
    assert result is not msgs
    assert dropped > 0
