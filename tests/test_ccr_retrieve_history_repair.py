"""Regression tests for the CCR headroom_retrieve history repair (#2814).

A passthrough side-request (prompt-type Stop hook evaluator, /compact) replays a
transcript whose history contains a ``headroom_retrieve`` tool_use, but the
request's own ``tools`` array does not declare that tool. Anthropic 400s on the
dangling reference. The repair neutralizes those history blocks so the 400 is
structurally impossible, without breaking user/assistant alternation.
"""

from __future__ import annotations

from headroom.ccr.tool_injection import CCR_TOOL_NAME
from headroom.proxy.helpers import strip_unsupported_ccr_retrieve_blocks


def _transcript_with_retrieve() -> list[dict]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "read config"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me expand that."},
                {
                    "type": "tool_use",
                    "id": "toolu_ccr_1",
                    "name": CCR_TOOL_NAME,
                    "input": {"hash": "abc123def456abc123def456"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_ccr_1",
                    "content": "the full expanded config text",
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]


def test_noop_when_tool_is_declared() -> None:
    messages = _transcript_with_retrieve()
    tools = [{"name": CCR_TOOL_NAME}]
    out, n = strip_unsupported_ccr_retrieve_blocks(messages, tools)
    assert n == 0
    assert out is messages  # identity: caller skips the write-back


def test_noop_when_no_retrieve_history() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]
    out, n = strip_unsupported_ccr_retrieve_blocks(messages, tools=[])
    assert n == 0
    assert out is messages


def test_neutralizes_retrieve_reference_when_tool_absent() -> None:
    messages = _transcript_with_retrieve()
    out, n = strip_unsupported_ccr_retrieve_blocks(messages, tools=[{"name": "Bash"}])

    assert n == 2  # tool_use + its paired tool_result
    # No dangling references survive anywhere.
    for message in out:
        for block in message["content"]:
            assert block.get("type") != "tool_use" or block.get("name") != CCR_TOOL_NAME
            assert not (
                block.get("type") == "tool_result" and block.get("tool_use_id") == "toolu_ccr_1"
            )

    # Message count and roles are unchanged (alternation intact).
    assert [m["role"] for m in out] == [m["role"] for m in messages]
    # The assistant's own text survives; the tool_use became a text block.
    assistant = out[1]["content"]
    assert assistant[0] == {"type": "text", "text": "Let me expand that."}
    assert assistant[1]["type"] == "text"
    # The retrieved result text is preserved, not dropped.
    assert out[2]["content"][0] == {
        "type": "text",
        "text": "the full expanded config text",
    }


def test_leaves_foreign_tool_use_untouched() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_bash_1", "name": "Bash", "input": {"cmd": "ls"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_bash_1", "content": "file.txt"}
            ],
        },
    ]
    out, n = strip_unsupported_ccr_retrieve_blocks(messages, tools=[{"name": "Bash"}])
    assert n == 0
    assert out is messages


def test_result_falls_back_to_placeholder_without_text() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": CCR_TOOL_NAME, "input": {}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": []}]},
    ]
    out, n = strip_unsupported_ccr_retrieve_blocks(messages, tools=[])
    assert n == 2
    assert out[1]["content"][0] == {"type": "text", "text": "[headroom_retrieve result omitted]"}
