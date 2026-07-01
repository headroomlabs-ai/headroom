"""Live-agent compression mode (``compress(mode='agent')`` / ``densify``).

Agent mode is the densify-only regime for compressing tool results an agent
reads back: no CCR/removal, no ML text compression, a verified losslessness
guarantee, and deterministic (prompt-cache-safe) output. These tests assert
those invariants and that the default ``compress()`` path is unchanged.
"""

from __future__ import annotations

import json

import headroom
from headroom.transforms.compaction_codec import expand_compacted, is_compacted


def _tool_result_messages(records: list[dict]) -> list[dict]:
    return [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "q", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": json.dumps(records)}
            ],
        },
        {"role": "user", "content": "summarize"},
    ]


def _records(n: int = 80) -> list[dict]:
    return [
        {"id": i, "owner": f"user{i % 5}", "status": "active", "note": f"row padding text {i} here"}
        for i in range(n)
    ]


def test_densify_compresses_losslessly_without_markers() -> None:
    res = headroom.densify(_tool_result_messages(_records()))
    out = json.dumps(res.messages)
    assert res.tokens_saved > 0
    assert res.lossless is True
    assert res.reverted_messages == 0
    assert "<<ccr:" not in out  # no removal markers
    assert "__dropped:" not in out  # no dropped rows


def test_densified_tool_result_round_trips() -> None:
    records = _records()
    res = headroom.densify(_tool_result_messages(records))
    content = res.messages[2]["content"][0]["content"]
    assert is_compacted(content)
    assert expand_compacted(content) == records


def test_opaque_blob_is_preserved_not_pointerized() -> None:
    blob = "X" * 2000  # well over the 256B opaque threshold
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t2", "name": "read", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t2",
                    "content": json.dumps({"file": "x", "body": blob}),
                }
            ],
        },
    ]
    res = headroom.densify(msgs)
    out = json.dumps(res.messages)
    assert res.lossless is True
    assert "<<ccr:" not in out
    assert blob in out  # the blob the agent needs is still there verbatim


def test_deterministic_output_for_cache_safety() -> None:
    msgs = _tool_result_messages(_records())
    a = headroom.densify(msgs)
    b = headroom.densify(_tool_result_messages(_records()))
    assert json.dumps(a.messages) == json.dumps(b.messages)


def test_value_factoring_applied_in_agent_mode() -> None:
    paths = ["./pkg/module_one.py", "./pkg/module_two.py", "./pkg/module_three.py"]
    records = [
        {"path": paths[i % len(paths)], "line": 10 + i, "content": f"call_{i}()"} for i in range(40)
    ]
    res = headroom.densify(_tool_result_messages(records))
    content = res.messages[2]["content"][0]["content"]
    assert "__dict:" in content  # legend present
    assert expand_compacted(content) == records  # lossless


def test_default_compress_is_unchanged() -> None:
    res = headroom.compress(_tool_result_messages(_records()), kompress_model="disabled")
    # Default regime does not run lossless verification.
    assert res.lossless is None
    assert res.reverted_messages == 0
    assert res.tokens_saved > 0


def test_verify_lossless_flag_without_agent_mode() -> None:
    res = headroom.compress(
        _tool_result_messages(_records()), kompress_model="disabled", verify_lossless=True
    )
    assert res.lossless is True


def test_empty_and_passthrough_inputs() -> None:
    assert headroom.densify([]).messages == []
    short = [{"role": "user", "content": "hi"}]
    assert headroom.densify(short).messages == short


def test_openai_format_tool_message_is_densified_losslessly() -> None:
    # Generic harness: OpenAI chat format (role='tool', string content) — not
    # Anthropic content blocks. Agent mode must handle it the same way.
    records = [{"id": i, "tag": f"t{i % 4}", "note": f"value padding text {i}"} for i in range(60)]
    msgs = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "q", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps(records)},
    ]
    res = headroom.densify(msgs)
    assert res.lossless is True
    assert res.tokens_saved > 0
    content = res.messages[2]["content"]
    assert is_compacted(content)
    assert expand_compacted(content) == records


def test_reverse_helpers_are_public() -> None:
    # A consumer of densified output must be able to reverse it from the
    # top-level package, not only a deep submodule path.
    from headroom import expand_compacted as top_expand
    from headroom import is_compacted as top_is

    records = [{"id": i, "tag": f"t{i % 3}"} for i in range(30)]
    res = headroom.densify(_tool_result_messages(records))
    content = res.messages[2]["content"][0]["content"]
    assert top_is(content)
    assert top_expand(content) == records
