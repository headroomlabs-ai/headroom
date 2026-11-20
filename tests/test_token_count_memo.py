"""Parity tests for TokenCountMemo (perf review F2).

The memo's correctness hinges on: tokenizer.count_messages(messages) ==
sum(tokenizer.count_message(m) for m in messages) + tokenizer.REPLY_OVERHEAD.
These tests prove the memoized count matches the direct count exactly across
message shapes the Claude Code route actually sends (text, tool_use,
tool_result string content, tool_result list-of-blocks content, images), using
EstimatingTokenCounter — the only tokenizer the proxy uses for Claude models
(tokenizers/registry.py `_create_anthropic`).
"""

from __future__ import annotations

import threading

import pytest

from headroom.cache.token_count_memo import TokenCountMemo, count_messages_memoized
from headroom.tokenizers.estimator import EstimatingTokenCounter

FIXTURES = [
    pytest.param(
        [
            {"role": "user", "content": "hello there"},
            {"role": "assistant", "content": "hi, how can I help?"},
        ],
        id="plain_text",
    ),
    pytest.param(
        [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "my_tool", "input": {"a": 1}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "tool output data"}
                ],
            },
        ],
        id="tool_use_and_result_string",
    ),
    pytest.param(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "list-of-blocks tool output"}],
                    }
                ],
            },
        ],
        id="tool_result_list_of_blocks",
    ),
    pytest.param(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                    },
                ],
            },
        ],
        id="text_and_image_block",
    ),
    pytest.param(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi", "name": "alice"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "search", "arguments": '{"q": "test"}'},
                    }
                ],
            },
        ],
        id="system_name_and_tool_calls",
    ),
    pytest.param([], id="empty_messages"),
    pytest.param([{"role": "user", "content": "solo message"}], id="single_message"),
]


@pytest.mark.parametrize("messages", FIXTURES)
def test_memoized_count_matches_direct_count(messages) -> None:
    tokenizer = EstimatingTokenCounter()
    memo = TokenCountMemo()

    direct = tokenizer.count_messages(messages)
    memoized = count_messages_memoized(memo, tokenizer, messages)

    assert memoized == direct


def test_memoized_count_stable_across_repeated_calls_cold_and_warm() -> None:
    """Same messages, memo cold then warm — count must not drift."""
    tokenizer = EstimatingTokenCounter()
    memo = TokenCountMemo()
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "output"}],
        },
    ]
    direct = tokenizer.count_messages(messages)

    cold = count_messages_memoized(memo, tokenizer, messages)
    warm = count_messages_memoized(memo, tokenizer, messages)

    assert cold == direct
    assert warm == direct


def test_memoized_count_handles_append_only_delta() -> None:
    """Only the new tail message should require a fresh count; the frozen
    prefix's per-message entries must already be memoized (cache hits)."""
    tokenizer = EstimatingTokenCounter()
    memo = TokenCountMemo()
    turn1 = [{"role": "user", "content": "turn one"}]
    turn2 = turn1 + [{"role": "assistant", "content": "turn two reply"}]

    count_messages_memoized(memo, tokenizer, turn1)
    prefix_key = TokenCountMemo.message_hash(turn1[0])
    assert memo.get(prefix_key, tokenizer=tokenizer) is not None

    memoized_turn2 = count_messages_memoized(memo, tokenizer, turn2)
    assert memoized_turn2 == tokenizer.count_messages(turn2)


def test_message_hash_differs_for_different_messages() -> None:
    h1 = TokenCountMemo.message_hash({"role": "user", "content": "a"})
    h2 = TokenCountMemo.message_hash({"role": "user", "content": "b"})
    assert h1 != h2


def test_message_hash_same_for_identical_messages() -> None:
    m = {"role": "user", "content": "same content"}
    assert TokenCountMemo.message_hash(dict(m)) == TokenCountMemo.message_hash(dict(m))


def test_forced_digest_collision_never_aliases_unequal_messages(monkeypatch) -> None:
    """Canonical verification protects correctness even if the digest collides."""

    class _CountingTokenizer:
        ADDITIVE_COUNTS = True
        REPLY_OVERHEAD = 0

        def count_message(self, message) -> int:  # noqa: ANN001
            return len(message["content"])

        def count_messages(self, messages) -> int:  # noqa: ANN001
            return sum(self.count_message(message) for message in messages)

    monkeypatch.setattr(TokenCountMemo, "message_hash", staticmethod(lambda _message: "collision"))
    memo = TokenCountMemo()
    tokenizer = _CountingTokenizer()

    assert count_messages_memoized(memo, tokenizer, [{"content": "a"}]) == 1
    assert count_messages_memoized(memo, tokenizer, [{"content": "much longer"}]) == 11


class TestTokenCountMemoEviction:
    def test_eviction_at_max_entries(self) -> None:
        memo = TokenCountMemo(max_entries=3)
        tok = object()  # get/put require the bound tokenizer's identity
        memo.bind_or_reset(tok)
        memo.put("a", 1, tokenizer=tok)
        memo.put("b", 2, tokenizer=tok)
        memo.put("c", 3, tokenizer=tok)
        memo.get("a", tokenizer=tok)  # touch "a" so it's not the least-recently-used
        memo.put("d", 4, tokenizer=tok)  # should evict "b" (oldest untouched)

        assert memo.get("a", tokenizer=tok) == 1
        assert memo.get("b", tokenizer=tok) is None
        assert memo.get("c", tokenizer=tok) == 3
        assert memo.get("d", tokenizer=tok) == 4

    def test_get_stats_reports_entry_count(self) -> None:
        memo = TokenCountMemo()
        tok = object()
        memo.bind_or_reset(tok)
        memo.put("a", 1, tokenizer=tok)
        memo.put("b", 2, tokenizer=tok)
        assert memo.get_stats()["entries"] == 2


class TestTokenizerCapabilityGuards:
    """Runtime enforcement of the additive-counts precondition and the
    memo↔tokenizer binding (review findings 3 and 4)."""

    def test_non_additive_tokenizer_bypasses_memo(self) -> None:
        """A tokenizer flagged non-additive (HuggingFace/Mistral chat-template
        counting) must get a plain count_messages call — never per-message
        memoization, which would be systematically wrong for it."""

        class _JointTokenizer:
            ADDITIVE_COUNTS = False
            REPLY_OVERHEAD = 3

            def count_message(self, message) -> int:  # noqa: ANN001
                raise AssertionError("per-message path must not be used")

            def count_messages(self, messages) -> int:  # noqa: ANN001
                return 1234

        memo = TokenCountMemo()
        assert (
            count_messages_memoized(memo, _JointTokenizer(), [{"role": "user", "content": "x"}])
            == 1234
        )
        assert memo.get_stats()["entries"] == 0

    def test_unflagged_tokenizer_bypasses_memo(self) -> None:
        """Duck-typed tokenizers without the ADDITIVE_COUNTS flag get the
        conservative (correct, unmemoized) path."""

        class _Unflagged:
            REPLY_OVERHEAD = 3

            def count_message(self, message) -> int:  # noqa: ANN001
                raise AssertionError("per-message path must not be used")

            def count_messages(self, messages) -> int:  # noqa: ANN001
                return 99

        memo = TokenCountMemo()
        assert count_messages_memoized(memo, _Unflagged(), [{"role": "user", "content": "x"}]) == 99
        assert memo.get_stats()["entries"] == 0

    def test_tokenizer_swap_clears_memo(self) -> None:
        """Counts computed under one tokenizer must not be served after a
        swap (the count fail-open path downgrades to a fresh estimator
        mid-session): the memo rebinds and recounts."""
        messages = [{"role": "user", "content": "hello world " * 50}]
        a = EstimatingTokenCounter()
        b = EstimatingTokenCounter(chars_per_token=1.0)  # very different ratio

        memo = TokenCountMemo()
        count_a = count_messages_memoized(memo, a, messages)
        count_b = count_messages_memoized(memo, b, messages)

        assert count_b == b.count_messages(messages)
        assert count_b != count_a  # stale count under `a` was not reused

    def test_same_tokenizer_keeps_memo_bound(self) -> None:
        messages = [{"role": "user", "content": "hello world " * 50}]
        tokenizer = EstimatingTokenCounter()
        memo = TokenCountMemo()
        count_messages_memoized(memo, tokenizer, messages)
        entries = memo.get_stats()["entries"]
        count_messages_memoized(memo, tokenizer, messages)
        assert memo.get_stats()["entries"] == entries  # no clear, warm hits


class _GatedTokenizer:
    """Additive tokenizer with a fixed per-message count and an interleaving hook.

    ``per_message`` differs between the two instances so a cross-tokenizer leak
    shows up as a wrong number rather than a coincidence.
    """

    ADDITIVE_COUNTS = True
    REPLY_OVERHEAD = 0

    def __init__(self, per_message: int, before_count: object = None) -> None:
        self.per_message = per_message
        self._before_count = before_count

    def count_message(self, message: dict) -> int:  # noqa: ARG002
        if self._before_count is not None:
            self._before_count()
        return self.per_message

    def count_messages(self, messages: list[dict]) -> int:
        return sum(self.count_message(m) for m in messages) + self.REPLY_OVERHEAD


class TestCrossTokenizerRebindRace:
    """The per-session memo is shared by concurrent requests, and the count
    fail-open path (``proxy/token_counting._count_offloaded``) hands each
    request a *fresh* ``EstimatingTokenCounter``. So request A can be mid-count
    under tokenizer A while request B rebinds the memo to tokenizer B. A's
    counts must never reach B.
    """

    def test_put_after_concurrent_rebind_never_leaks_to_new_binding(self) -> None:
        messages = [{"role": "user", "content": "shared prefix message"}]
        key = TokenCountMemo.message_hash(messages[0])
        canonical = TokenCountMemo.canonical_message(messages[0])

        a_is_counting = threading.Event()
        b_has_rebound = threading.Event()

        def _pause_a() -> None:
            # A has already called bind_or_reset and missed the cache; hold it
            # here so B's whole rebind+count lands before A's put.
            a_is_counting.set()
            assert b_has_rebound.wait(timeout=10), "B never rebound"

        a = _GatedTokenizer(per_message=7, before_count=_pause_a)
        b = _GatedTokenizer(per_message=31)
        memo = TokenCountMemo()

        a_total: list[int] = []
        a_thread = threading.Thread(
            target=lambda: a_total.append(count_messages_memoized(memo, a, messages)),
            daemon=True,
        )
        a_thread.start()
        assert a_is_counting.wait(timeout=10), "A never started counting"

        # B rebinds (clearing A's era) and populates its own count.
        b_total = count_messages_memoized(memo, b, messages)
        b_has_rebound.set()
        a_thread.join(timeout=10)
        assert not a_thread.is_alive()

        # A's own total stays exact under tokenizer A — the fix degrades A to
        # uncached counting, it does not hand A B's numbers.
        assert a_total == [7]
        assert b_total == 31

        # The proof: A's post-rebind put was dropped, so the memo still holds
        # only B's count for the shared message and a fresh B read agrees.
        assert memo.get(key, canonical, tokenizer=b) == 31
        assert count_messages_memoized(memo, b, messages) == 31

    def test_get_from_obsolete_binding_misses_instead_of_reading_new_counts(self) -> None:
        """Mirror direction: a stale binding must not read the rebinder's
        counts either (the swap guarantee stays symmetric)."""
        messages = [{"role": "user", "content": "shared prefix message"}]
        key = TokenCountMemo.message_hash(messages[0])
        canonical = TokenCountMemo.canonical_message(messages[0])

        a = _GatedTokenizer(per_message=7)
        b = _GatedTokenizer(per_message=31)
        memo = TokenCountMemo()

        count_messages_memoized(memo, a, messages)
        count_messages_memoized(memo, b, messages)  # rebinds to b

        assert memo.get(key, canonical, tokenizer=a) is None
        assert memo.get(key, canonical, tokenizer=b) == 31
