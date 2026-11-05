from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from headroom import utils


class FakeProvider:
    def __init__(self, result: str | None) -> None:
        self.result = result
        self.calls: list[tuple[int, int, str, int]] = []

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, model: str, cached_tokens: int = 0
    ) -> str | None:
        self.calls.append((input_tokens, output_tokens, model, cached_tokens))
        return self.result


def test_hash_helpers_and_request_id() -> None:
    request_id = utils.generate_request_id()
    assert len(request_id) == 36
    assert request_id.count("-") == 4

    assert (
        utils.compute_hash("hello")
        == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    assert utils.compute_hash("hi\ud800") == utils.compute_hash(
        "hi\ud800".encode("utf-8", "surrogatepass")
    )
    assert utils.compute_short_hash("hello", length=8) == "2cf24dba"


def test_extract_user_query_and_message_hashes() -> None:
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "skip"},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "  latest question  "}],
        },
    ]
    assert utils.extract_user_query(messages) == "latest question"
    assert utils.extract_user_query([{"role": "assistant", "content": "skip"}]) == ""


@pytest.mark.parametrize("latest_only", [False, True])
def test_extract_user_query_combines_text_blocks(latest_only: bool) -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "  Review incidents  "},
                {"type": "image", "source": "example"},
                {"type": "text", "text": " "},
                {"type": "text", "text": "  Focus on Kafka replication  "},
            ],
        }
    ]
    assert utils.extract_user_query(messages, latest_user_turn_only=latest_only) == (
        "Review incidents\nFocus on Kafka replication"
    )


def test_extract_user_query_skips_system_reminder_blocks() -> None:
    messages = [
        {"role": "user", "content": "older question"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<system-reminder>injected instructions</system-reminder>",
                },
                {"type": "text", "text": "real question"},
                {"type": "text", "text": "<system-reminder>more instructions</system-reminder>"},
            ],
        },
    ]
    assert utils.extract_user_query(messages) == "real question"
    assert utils.extract_user_query(messages, latest_user_turn_only=True) == "real question"

    messages[-1]["content"] = [
        {"type": "text", "text": "<system-reminder>injected</system-reminder>"}
    ]
    assert utils.extract_user_query(messages) == "older question"
    assert utils.extract_user_query(messages, latest_user_turn_only=True) == ""


@pytest.mark.parametrize("as_blocks", [False, True])
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("<system-reminder>context</system-reminder>\nFind Kafka lag", "Find Kafka lag"),
        ("Find <system-reminder>context</system-reminder> Kafka lag", "Find  Kafka lag"),
        (
            "<system-reminder id='1'>a</system-reminder>Find <system-reminder>b</system-reminder>Kafka",
            "Find Kafka",
        ),
        ("<System-Reminder>context</System-Reminder>Find Kafka", "Find Kafka"),
        ("İ<system-reminder>x</system-reminder>Kafka", "İ Kafka"),
        ("Find <system-reminder>unfinished", "Find <system-reminder>unfinished"),
        (
            "Find <system-reminder-extra>example</system-reminder> Kafka",
            "Find <system-reminder-extra>example</system-reminder> Kafka",
        ),
    ],
)
def test_extract_user_query_keeps_text_around_reminders(
    content: str, expected: str, as_blocks: bool
) -> None:
    body = [{"type": "text", "text": content}] if as_blocks else content
    messages = [{"role": "user", "content": body}]
    assert utils.extract_user_query(messages, latest_user_turn_only=True) == expected


@pytest.mark.parametrize("as_blocks", [False, True])
def test_reminder_only_turn_keeps_user_query_fallback(as_blocks: bool) -> None:
    reminder = "<system-reminder>context</system-reminder>"
    content = [{"type": "text", "text": reminder}] if as_blocks else reminder
    messages = [{"role": "user", "content": "older question"}, {"role": "user", "content": content}]
    assert utils.extract_user_query(messages) == "older question"
    assert utils.extract_user_query(messages, latest_user_turn_only=True) == ""


def test_extract_user_query_latest_user_turn_only() -> None:
    """The flag stops at the newest user turn instead of resurrecting an older one."""
    # A tool_result continuation: the newest user turn carries no text at all.
    messages = [
        {"role": "user", "content": "the original question"},
        {"role": "assistant", "content": "calling a tool"},
        {"role": "user", "content": [{"type": "tool_result", "content": "output"}]},
    ]

    # Default walks back and finds the older turn's question.
    assert utils.extract_user_query(messages) == "the original question"
    # Scoped to this turn, there is no question to score against.
    assert utils.extract_user_query(messages, latest_user_turn_only=True) == ""


def test_extract_user_query_latest_turn_only_agrees_on_normal_turns() -> None:
    """With text present on the newest user turn, the flag changes nothing."""
    messages = [
        {"role": "user", "content": "older question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": [{"type": "text", "text": " newest question "}]},
    ]

    assert utils.extract_user_query(messages) == "newest question"
    assert utils.extract_user_query(messages, latest_user_turn_only=True) == "newest question"

    hash_one = utils.compute_messages_hash(messages)
    hash_two = utils.compute_messages_hash(list(messages))
    assert hash_one == hash_two
    assert len(hash_one) == 16

    prefix_default = utils.compute_prefix_hash(
        [
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]
    )
    prefix_explicit = utils.compute_prefix_hash(
        [
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ],
        prefix_count=3,
    )
    assert prefix_default == prefix_explicit
    assert utils.compute_prefix_hash([]) == utils.compute_short_hash("")


def test_timestamp_marker_and_json_helpers() -> None:
    ts = utils.format_timestamp(datetime(2026, 4, 23, 6, 0, 0))
    assert ts == "2026-04-23T06:00:00Z"
    assert utils.parse_timestamp(ts) == datetime(2026, 4, 23, 6, 0, 0)
    assert utils.parse_timestamp("2026-04-23T06:00:00") == datetime(2026, 4, 23, 6, 0, 0)


def test_format_timestamp_normalizes_aware_datetimes() -> None:
    # Shipped integrations pass datetime.now(timezone.utc) (aware). Appending a
    # bare "Z" to an aware isoformat used to emit "...+00:00Z" — invalid ISO
    # 8601 that datetime.fromisoformat itself rejects.
    aware_utc = datetime(2026, 4, 23, 6, 0, 0, tzinfo=timezone.utc)
    ts = utils.format_timestamp(aware_utc)
    assert ts == "2026-04-23T06:00:00Z"
    # The stored string must round-trip through a strict parser.
    assert datetime.fromisoformat(ts.rstrip("Z")) == datetime(2026, 4, 23, 6, 0, 0)

    # A non-UTC offset must be converted to UTC, not just stamped "Z" onto the
    # local wall-clock time (which would mislabel the instant).
    east = datetime(2026, 4, 23, 6, 0, 0, tzinfo=timezone(timedelta(hours=5)))
    assert utils.format_timestamp(east) == "2026-04-23T01:00:00Z"

    marker = utils.create_marker("tool_digest", sha256="abc", count="2")
    assert marker == '<headroom:tool_digest sha256="abc" count="2">'
    assert utils.create_tool_digest_marker("abc") == '<headroom:tool_digest sha256="abc">'


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("fast_hash", ("hello", 8), "5d41402a"),
        ("fast_hash", (b"hello",), "5d41402abc4b2a76"),
        (
            "create_dropped_context_marker",
            ("budget",),
            '<headroom:dropped_context reason="budget">',
        ),
        (
            "create_dropped_context_marker",
            ("budget", 4),
            '<headroom:dropped_context reason="budget" count="4">',
        ),
        (
            "create_truncated_marker",
            (100, 25),
            '<headroom:truncated original="100" truncated_to="25">',
        ),
        (
            "extract_markers",
            (
                'x <headroom:tool_digest sha256="abc"> y '
                '<headroom:dropped_context reason="budget" count="2">',
            ),
            [
                {"type": "tool_digest", "attributes": {"sha256": "abc"}},
                {"type": "dropped_context", "attributes": {"reason": "budget", "count": "2"}},
            ],
        ),
        ("safe_json_loads", ('{"ok": true}',), ({"ok": True}, True)),
        ("safe_json_loads", ("{bad",), (None, False)),
        ("safe_json_dumps", ({"emoji": "café"},), '{"emoji":"café"}'),
    ],
)
def test_deprecated_helpers_warn_and_still_work(name: str, args: tuple, expected: object) -> None:
    with pytest.warns(DeprecationWarning, match=f"{name} is deprecated"):
        assert getattr(utils, name)(*args) == expected


def test_cost_formatting_and_deep_copy() -> None:
    provider = FakeProvider("1.25")
    assert utils.estimate_cost(100, 50, "gpt-4o", cached_tokens=10, provider=provider) == 1.25
    assert provider.calls == [(100, 50, "gpt-4o", 10)]
    assert utils.estimate_cost(1, 1, "gpt-4o", provider=None) is None

    none_provider = FakeProvider(None)
    assert utils.estimate_cost(1, 1, "gpt-4o", provider=none_provider) is None

    assert utils.format_cost(0.0099) == "$0.0099"
    assert utils.format_cost(1.234) == "$1.23"

    messages = [{"role": "user", "content": {"nested": ["a"]}}]
    copied = utils.deep_copy_messages(messages)
    copied[0]["content"]["nested"].append("b")
    assert messages == [{"role": "user", "content": {"nested": ["a"]}}]
