from __future__ import annotations

from headroom.proxy.helpers import hash_query_for_log as helper_hash_query_for_log
from headroom.proxy.query_log_policy import QUERY_LOG_HASH_BYTES, hash_query_for_log


def test_hash_query_for_log_is_stable_short_hex() -> None:
    query_hash = hash_query_for_log("find user memory about project x")

    assert query_hash == hash_query_for_log("find user memory about project x")
    assert len(query_hash) == QUERY_LOG_HASH_BYTES * 2
    assert all(char in "0123456789abcdef" for char in query_hash)


def test_hash_query_for_log_changes_with_query_content() -> None:
    assert hash_query_for_log("alpha") != hash_query_for_log("beta")


def test_hash_query_for_log_handles_unpaired_surrogates() -> None:
    query_hash = hash_query_for_log("bad-surrogate-\ud800")

    assert len(query_hash) == QUERY_LOG_HASH_BYTES * 2


def test_helpers_hash_query_for_log_delegates_to_policy() -> None:
    assert helper_hash_query_for_log("same") == hash_query_for_log("same")
