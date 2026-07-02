"""Chat-Completions path must not re-compress `headroom_retrieve` outputs.

Regression test for the CCR reentrancy bug: the Responses path tracks retrieve
`call_id`s and skips compressing their outputs, but the Chat-Completions path had
no equivalent. A retrieved original (the expanded content the model explicitly
asked for) therefore got re-compressed back into a `<<ccr:...>>` marker on the
next turn — marker-in / marker-out, so retrieval never actually returned content.

These tests pin the capture/restore guard that fixes it. They fail before the fix
(the helpers don't exist) and pass after.
"""

from __future__ import annotations

from headroom.proxy.handlers.openai import (
    _headroom_retrieve_tool_call_ids,
    capture_headroom_retrieve_outputs,
    restore_headroom_retrieve_outputs,
)

ORIGINAL = "RETRIEVED ORIGINAL\n" + "\n".join(
    f"2026-06-19T10:{i:02d}:00Z level=ERROR user=u{i} trace=abc{i} backend=10.0.0.{i}"
    for i in range(40)
)


def _messages(retrieve_output, other_output):
    """Chat history: an assistant headroom_retrieve call + a normal read_file call."""
    return [
        {"role": "user", "content": "expand the dump"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_keep",
                    "type": "function",
                    "function": {"name": "headroom_retrieve", "arguments": '{"hash":"h1"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_keep", "content": retrieve_output},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_other",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_other", "content": other_output},
    ]


def test_detects_only_headroom_retrieve_call_ids():
    ids = _headroom_retrieve_tool_call_ids(_messages(ORIGINAL, ORIGINAL))
    assert ids == {"call_keep"}


def test_namespaced_retrieve_name_is_detected():
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "mcp__headroom__headroom_retrieve", "arguments": "{}"},
                }
            ],
        }
    ]
    assert _headroom_retrieve_tool_call_ids(msgs) == {"c1"}


def test_capture_snapshots_only_retrieve_outputs():
    protected = capture_headroom_retrieve_outputs(_messages(ORIGINAL, ORIGINAL))
    assert set(protected) == {"call_keep"}
    assert protected["call_keep"] == ORIGINAL


def test_capture_is_empty_without_retrieve_calls():
    assert (
        capture_headroom_retrieve_outputs([{"role": "tool", "tool_call_id": "x", "content": "y"}])
        == {}
    )


def test_restore_recovers_byte_identical_original():
    # Simulate the bug: compression mangled BOTH tool outputs before forwarding.
    protected = capture_headroom_retrieve_outputs(_messages(ORIGINAL, ORIGINAL))
    compressed = "<<ccr:deadbeef,log,1234>>"
    post = _messages(compressed, compressed)

    n = restore_headroom_retrieve_outputs(post, protected)

    assert n == 1
    # retrieve output is restored byte-for-byte ...
    assert post[2]["content"] == ORIGINAL
    # ... while the non-retrieve output stays compressed (no over-protection).
    assert post[4]["content"] == compressed


def test_restore_is_idempotent_noop_when_already_pristine():
    protected = capture_headroom_retrieve_outputs(_messages(ORIGINAL, ORIGINAL))
    post = _messages(ORIGINAL, ORIGINAL)
    assert restore_headroom_retrieve_outputs(post, protected) == 0


def test_restore_handles_empty_protection_map():
    assert restore_headroom_retrieve_outputs(_messages(ORIGINAL, ORIGINAL), {}) == 0
