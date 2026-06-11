from __future__ import annotations

from datetime import datetime

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
    assert utils.fast_hash("hello", length=8) == "5d41402a"


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

    marker = utils.create_marker("tool_digest", sha256="abc", count="2")
    assert marker == '<headroom:tool_digest sha256="abc" count="2">'
    assert utils.create_tool_digest_marker("abc") == '<headroom:tool_digest sha256="abc">'
    assert utils.create_dropped_context_marker("budget") == (
        '<headroom:dropped_context reason="budget">'
    )
    assert utils.create_dropped_context_marker("budget", count=4) == (
        '<headroom:dropped_context reason="budget" count="4">'
    )
    assert utils.create_truncated_marker(100, 25) == (
        '<headroom:truncated original="100" truncated_to="25">'
    )

    extracted = utils.extract_markers(
        'x <headroom:tool_digest sha256="abc"> y <headroom:dropped_context reason="budget" count="2">'
    )
    assert extracted == [
        {"type": "tool_digest", "attributes": {"sha256": "abc"}},
        {"type": "dropped_context", "attributes": {"reason": "budget", "count": "2"}},
    ]

    assert utils.safe_json_loads('{"ok": true}') == ({"ok": True}, True)
    assert utils.safe_json_loads("{bad") == (None, False)
    assert utils.safe_json_dumps({"emoji": "café"}) == '{"emoji":"café"}'


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
