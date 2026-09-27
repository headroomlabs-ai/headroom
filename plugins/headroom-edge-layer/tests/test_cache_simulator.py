from __future__ import annotations

from cache_simulator import longest_common_message_prefix, simulate_turn_cache


def test_longest_common_prefix_stops_at_first_difference() -> None:
    a = [{"role": "user", "content": "1"}, {"role": "assistant", "content": "2"}, {"role": "user", "content": "3"}]
    b = [{"role": "user", "content": "1"}, {"role": "assistant", "content": "2"}, {"role": "user", "content": "DIFFERENT"}]
    assert longest_common_message_prefix(a, b) == 2


def test_first_turn_has_no_cache_read() -> None:
    result = simulate_turn_cache(None, [{"role": "user", "content": "hello"}], previous_timestamp=None, current_timestamp=0.0)
    assert result.cache_read_tokens == 0
    assert result.cache_write_tokens > 0


def test_identical_prefix_becomes_cache_read() -> None:
    prev = [{"role": "user", "content": "1"}, {"role": "assistant", "content": "2"}]
    current = prev + [{"role": "user", "content": "3"}]
    result = simulate_turn_cache(prev, current, previous_timestamp=0.0, current_timestamp=1.0)
    assert result.cache_read_tokens > 0
    assert result.cache_write_tokens > 0  # the new tail message


def test_ttl_expiry_resets_to_full_cache_write() -> None:
    prev = [{"role": "user", "content": "1"}]
    current = prev + [{"role": "user", "content": "2"}]
    result = simulate_turn_cache(prev, current, previous_timestamp=0.0, current_timestamp=1000.0, ttl_seconds=300)
    assert result.cache_read_tokens == 0


def test_cache_read_share_is_a_fraction() -> None:
    prev = [{"role": "user", "content": "1"}]
    current = prev + [{"role": "user", "content": "2"}]
    result = simulate_turn_cache(prev, current, previous_timestamp=0.0, current_timestamp=1.0)
    assert 0.0 <= result.cache_read_share <= 1.0
