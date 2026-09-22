"""Compact machine-generated JSON must not evade compression (token-estimate bug).

Whitespace-split token counting made a compact JSON payload (no spaces —
the default output of json.dumps with separators, JSON.stringify, boto3)
count as ~1 token, so compression ratios computed as ~1.0 and the
min_ratio gate rejected SmartCrusher's real output.
"""

from __future__ import annotations

import json
import random

import pytest

from headroom.transforms.content_router import (
    ContentRouter,
    ContentRouterConfig,
    _estimate_tokens,
)


def test_estimate_tokens_monotone_on_compact_json() -> None:
    small = json.dumps([{"a": 1}] * 5, separators=(",", ":"))
    large = json.dumps([{"a": 1}] * 500, separators=(",", ":"))
    assert _estimate_tokens(large) > _estimate_tokens(small) > 1


def test_compact_json_tool_result_compresses() -> None:
    tokenizer = pytest.importorskip("headroom.tokenizers.estimator")
    from headroom.tokenizer import Tokenizer

    tok = Tokenizer(tokenizer.EstimatingTokenCounter())
    random.seed(42)
    rows = [
        {
            "serviceArn": f"arn:aws:ecs:us-east-1:123456789012:service/x/svc-{s:03d}",
            "serviceName": f"svc-{s:03d}",
            "status": "ACTIVE",
            "desiredCount": random.randint(1, 6),
            "runningCount": random.randint(0, 6),
        }
        for s in range(150)
    ]
    payload = json.dumps(rows, separators=(",", ":"))
    assert " " not in payload[:200]  # genuinely compact
    messages = [
        {"role": "user", "content": "Investigate."},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "toolu_1", "name": "list_services", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": payload}],
        },
        {"role": "user", "content": "Summarize in one sentence."},
    ]
    router = ContentRouter(ContentRouterConfig(skip_user_messages=False))
    before = tok.count_messages(messages)
    result = router.apply(
        [json.loads(json.dumps(m)) for m in messages],
        tok,
        context="Summarize",
        frozen_message_count=0,
    )
    after = tok.count_messages(result.messages)
    assert after < before * 0.9, (before, after, result.transforms_applied[:5])


# ---------------------------------------------------------------------------
# Issue #3634: SmartCrusher must not drop object fields (tags/subtasks)
# ---------------------------------------------------------------------------

# Exact MCP `saga-mcp_task_get` payload from the issue (813 bytes, original
# key order, tags encoded as a JSON string). Do not pretty-print or reorder.
MCP_TASK_GET_JSON = (
    '{"id":79,"epic_id":9,"title":"test","description":null,'
    '"status":"done","priority":"medium","sort_order":0,'
    '"assigned_to":null,"estimated_hours":null,"actual_hours":null,'
    '"due_date":null,"source_ref":null,"metadata":"{}",'
    '"created_at":"2026-09-15 20:32:06","updated_at":"2026-09-16 11:59:27",'
    '"description_locked":0,"is_deleted":0,"deleted_at":null,'
    '"deleted_by":null,"delete_reason":null,"epic_name":"latency-routing",'
    '"tags":"[\\"cherry-pick\\",\\"dedicated branch\\"]",'
    '"subtasks":[{"id":163,"task_id":79,"title":"test2","status":"todo",'
    '"sort_order":1,"created_at":"2026-09-15 20:32:13",'
    '"updated_at":"2026-09-16 14:02:26"},'
    '{"id":165,"task_id":79,"title":"test4","status":"todo","sort_order":2,'
    '"created_at":"2026-09-15 20:38:47","updated_at":"2026-09-17 20:29:04"}],'
    '"notes":[],"comments":[],"depends_on":[],"dependents":[]}'
)

_TOOL_CALL_ID = "call_task_get_79"
_TOOL_NAME = "saga-mcp_task_get"


def _offline_tokenizer():
    tokenizer = pytest.importorskip("headroom.tokenizers.estimator")
    from headroom.tokenizer import Tokenizer

    return Tokenizer(tokenizer.EstimatingTokenCounter())


def _object_preserving_router() -> ContentRouter:
    # Real ContentRouter + Rust SmartCrusher. Disable unrelated ML / log
    # fallbacks so a no-savings crush cannot be rewritten by Kompress.
    return ContentRouter(
        ContentRouterConfig(
            skip_user_messages=False,
            enable_kompress=False,
            enable_log_compressor=False,
            enable_search_compressor=False,
            relevance_split=False,
            protect_analysis_context=False,
            exclude_tools=set(),
        )
    )


def _openai_task_messages(payload: str) -> list[dict]:
    return [
        {"role": "user", "content": "read task 79"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": _TOOL_CALL_ID,
                    "type": "function",
                    "function": {"name": _TOOL_NAME, "arguments": '{"id":79}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": _TOOL_CALL_ID,
            "content": payload,
        },
    ]


def _anthropic_task_messages(payload: str) -> list[dict]:
    return [
        {"role": "user", "content": "read task 79"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": _TOOL_CALL_ID,
                    "name": _TOOL_NAME,
                    "input": {"id": 79},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": _TOOL_CALL_ID,
                    "content": payload,
                }
            ],
        },
    ]


def _apply(messages: list[dict]):
    router = _object_preserving_router()
    tok = _offline_tokenizer()
    return router.apply(
        [json.loads(json.dumps(m)) for m in messages],
        tok,
        context="read task 79",
        frozen_message_count=0,
        protect_recent=0,
        protect_analysis_context=False,
    )


def _openai_tool_content(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id") == _TOOL_CALL_ID:
            content = msg.get("content")
            assert isinstance(content, str), content
            return content
    raise AssertionError(f"OpenAI tool message {_TOOL_CALL_ID} missing")


def _anthropic_tool_content(messages: list[dict]) -> str:
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_result" and block.get("tool_use_id") == _TOOL_CALL_ID:
                inner = block.get("content")
                assert isinstance(inner, str), inner
                return inner
    raise AssertionError(f"Anthropic tool_result {_TOOL_CALL_ID} missing")


def _assert_task_payload_preserved(original: str, compressed: str) -> dict:
    assert "<<ccr:" not in compressed, compressed[:200]
    parsed = json.loads(compressed)
    expected = json.loads(original)
    assert parsed == expected, (sorted(expected), sorted(parsed))
    assert "tags" in parsed
    assert "subtasks" in parsed
    assert parsed["tags"] == '["cherry-pick","dedicated branch"]'
    assert len(parsed["subtasks"]) == 2
    assert parsed["subtasks"][0]["title"] == "test2"
    assert parsed["subtasks"][1]["title"] == "test4"
    return parsed


def _wide_renamed_task_json() -> str:
    """Renamed, oversized variant so a below-threshold skip cannot hide key drop.

    Field names are arbitrary (not tags/subtasks) and expensive values sit at
    the beginning, middle, and end so a boundary/stride selector cannot
    accidentally keep them all.
    """
    original = json.loads(MCP_TASK_GET_JSON)
    expensive = {
        "nested": {"label": "child", "note": "n" * 180},
        "items": [],
        "ok": True,
        "missing": None,
    }
    wide: dict = {
        "leading_blob": expensive,
        "label_bundle": original.pop("tags"),
        "child_records": original.pop("subtasks"),
    }
    wide.update(original)
    for i in range(20):
        wide[f"extra_field_{i:02d}"] = (
            f"this is a relatively long value string for entry number {i} with content"
        )
    wide["trailing_blob"] = {
        "nested": {"label": "tail", "note": "t" * 180},
        "items": [],
        "ok": False,
        "missing": None,
    }
    payload = json.dumps(wide, separators=(",", ":"))
    assert len(payload) > 1500
    assert len(wide) > 8
    return payload


def test_mcp_task_fixture_is_the_issue_payload() -> None:
    assert len(MCP_TASK_GET_JSON) == 813
    parsed = json.loads(MCP_TASK_GET_JSON)
    assert list(parsed)[21] == "tags"
    assert list(parsed)[22] == "subtasks"
    assert parsed["tags"] == '["cherry-pick","dedicated branch"]'


def test_mcp_task_object_preserved_openai_tool_role() -> None:
    messages = _openai_task_messages(MCP_TASK_GET_JSON)
    result = _apply(messages)
    content = _openai_tool_content(result.messages)
    assert content == MCP_TASK_GET_JSON
    _assert_task_payload_preserved(MCP_TASK_GET_JSON, content)
    # Tool-call pairing stays intact.
    tool_msg = next(m for m in result.messages if m.get("role") == "tool")
    assistant = next(m for m in result.messages if m.get("role") == "assistant")
    assert tool_msg["tool_call_id"] == _TOOL_CALL_ID
    assert assistant["tool_calls"][0]["id"] == _TOOL_CALL_ID
    assert assistant["tool_calls"][0]["function"]["name"] == _TOOL_NAME
    assert not any("object:adaptive" in t for t in result.transforms_applied)


def test_mcp_task_object_preserved_anthropic_tool_result() -> None:
    messages = _anthropic_task_messages(MCP_TASK_GET_JSON)
    result = _apply(messages)
    content = _anthropic_tool_content(result.messages)
    assert content == MCP_TASK_GET_JSON
    _assert_task_payload_preserved(MCP_TASK_GET_JSON, content)
    assistant = next(m for m in result.messages if m.get("role") == "assistant")
    assert assistant["content"][0]["id"] == _TOOL_CALL_ID
    user_blocks = next(
        m["content"]
        for m in result.messages
        if m.get("role") == "user" and isinstance(m.get("content"), list)
    )
    assert user_blocks[0]["tool_use_id"] == _TOOL_CALL_ID
    assert not any("object:adaptive" in t for t in result.transforms_applied)


def test_wide_renamed_object_fields_preserved_openai_tool_role() -> None:
    payload = _wide_renamed_task_json()
    result = _apply(_openai_task_messages(payload))
    content = _openai_tool_content(result.messages)
    assert "<<ccr:" not in content
    parsed = json.loads(content)
    expected = json.loads(payload)
    assert parsed == expected
    assert "label_bundle" in parsed
    assert "child_records" in parsed
    assert "leading_blob" in parsed
    assert "trailing_blob" in parsed
    assert parsed["label_bundle"] == '["cherry-pick","dedicated branch"]'
    assert len(parsed["child_records"]) == 2
    assert parsed["leading_blob"]["nested"]["note"] == "n" * 180
    assert parsed["trailing_blob"]["nested"]["note"] == "t" * 180


def test_malformed_json_tool_result_passes_through() -> None:
    payload = "{" + "not-json, still a tool result. " * 40
    assert len(payload) > 500
    result = _apply(_openai_task_messages(payload))
    content = _openai_tool_content(result.messages)
    assert content == payload
    assert "<<ccr:" not in content
