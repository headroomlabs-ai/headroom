"""Regression for #2410: the buffered-CCR Responses -> SSE reconstruction must
replay the incremental output-item/text events, not just response.created +
response.completed, so AI-SDK / OpenCode clients render the output."""

from __future__ import annotations

import json

from headroom.proxy.handlers.openai import _openai_responses_to_sse


def _parse(events: list[bytes]) -> list[dict]:
    out: list[dict] = []
    for e in events:
        s = e.decode()
        if s.startswith("data: [DONE]"):
            out.append({"type": "[DONE]"})
            continue
        out.append(json.loads(s.split("data: ", 1)[1]))
    return out


def test_responses_sse_replays_incremental_output_text() -> None:
    resp = {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "model": "gpt-5.3-codex",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello world", "annotations": []}],
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }

    parsed = _parse(_openai_responses_to_sse(resp))
    types = [p["type"] for p in parsed]

    assert types[0] == "response.created"
    assert types[1] == "response.in_progress"
    assert types[-2] == "response.completed"
    assert types[-1] == "[DONE]"

    # The visible assistant text is streamed as an output_text.delta.
    deltas = [p for p in parsed if p["type"] == "response.output_text.delta"]
    assert len(deltas) == 1
    assert deltas[0]["delta"] == "Hello world"
    assert deltas[0]["output_index"] == 1
    assert deltas[0]["content_index"] == 0

    # The message item gets the full content-part sequence; the reasoning item
    # gets add/done with no content parts.
    assert types.count("response.output_item.added") == 2
    assert types.count("response.output_item.done") == 2
    assert "response.content_part.added" in types
    assert "response.output_text.done" in types
    assert "response.content_part.done" in types

    # created / in_progress carry an empty output; completed carries the full one.
    created = next(p for p in parsed if p["type"] == "response.created")
    assert created["response"]["output"] == []
    completed = next(p for p in parsed if p["type"] == "response.completed")
    assert completed["response"]["output"] == resp["output"]

    # Sequence numbers are contiguous from 0.
    seqs = [p["sequence_number"] for p in parsed if p["type"] != "[DONE]"]
    assert seqs == list(range(len(seqs)))


def test_responses_sse_empty_output_still_valid() -> None:
    resp = {"id": "resp_2", "status": "completed", "output": [], "usage": {}}
    types = [p["type"] for p in _parse(_openai_responses_to_sse(resp))]
    assert types == ["response.created", "response.in_progress", "response.completed", "[DONE]"]


def test_responses_sse_non_message_item_added_and_done() -> None:
    resp = {
        "id": "resp_3",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "c1",
                "name": "grep",
                "arguments": "{}",
            }
        ],
        "usage": {},
    }
    parsed = _parse(_openai_responses_to_sse(resp))
    types = [p["type"] for p in parsed]
    assert types == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.completed",
        "[DONE]",
    ]
    # The function_call item is preserved whole on added and done.
    done = next(p for p in parsed if p["type"] == "response.output_item.done")
    assert done["item"]["name"] == "grep"
