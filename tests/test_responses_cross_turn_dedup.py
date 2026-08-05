"""Cross-turn dedup on the OpenAI Responses path (Codex ``function_call_output``).

Fixtures mirror a REAL Codex run captured through the headroom proxy: a file read
returns as ``{"type":"function_call_output","call_id":...,"output":"Chunk ID: …\\n
Wall time: …\\nProcess exited with code 0\\nOriginal token count: …\\nOutput:\\n
<FILE BODY>\\n"}``. The ``Chunk ID`` / ``Wall time`` header varies per call, so a
whole-block match never fires — longest-span matching must fold the identical
body and leave the varying header verbatim.
"""

from __future__ import annotations

from headroom.proxy.handlers.openai import (
    _RESPONSES_OUTPUT_ITEM_TYPES,
    _dedup_responses_output_items,
)

BODY = (
    "def paginate_orders(items, page, page_size):\n"
    '    """Return one page of orders."""\n'
    "    start = page * page_size\n"
    "    end = start + page_size + 1  # off-by-one: should be start + page_size\n"
    "    return items[start:end]\n"
    "\n\n"
    'SERVICE_TAG = "svc-03e8-tag"\n'
    "\n\n"
    "def compute_overdraft(business_id, amount):\n"
    "    fee = amount * 0.05\n"
    '    return {"business_id": business_id, "fee": fee, "tag": SERVICE_TAG}\n'
)


def _wrap(chunk_id: str, wall: str) -> str:
    # Codex's exec_command wrapper — the header lines vary call to call.
    return (
        f"Chunk ID: {chunk_id}\n"
        f"Wall time: {wall} seconds\n"
        "Process exited with code 0\n"
        "Original token count: 97\n"
        "Output:\n"
    )


def _read_output(call_id: str, chunk_id: str, wall: str) -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": _wrap(chunk_id, wall) + BODY,
    }


def _read_call(call_id: str) -> dict:
    return {
        "type": "function_call",
        "name": "exec_command",
        "arguments": '{"cmd":"cat buggy.py","workdir":"/tmp"}',
        "call_id": call_id,
    }


def test_repeated_codex_read_folds_body_keeps_varying_header():
    items = [
        {"role": "user", "content": "find the bug"},
        _read_call("c1"),
        _read_output("c1", "492f0f", "0.0000"),  # read #1 (reference)
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "re-reading"}],
        },
        _read_call("c2"),
        _read_output("c2", "a1b2c3", "0.0100"),  # read #2 (duplicate) -> body folds
    ]
    folded, saved = _dedup_responses_output_items(
        items, _RESPONSES_OUTPUT_ITEM_TYPES, count_tokens=len
    )

    assert folded == 1
    assert saved > 0
    # earliest read: byte-identical (reference target, sits in the cached prefix)
    assert items[2]["output"] == _wrap("492f0f", "0.0000") + BODY
    # later read: identical body folded to a pointer; the per-call header stays verbatim
    later = items[5]["output"]
    assert "[↑" in later
    assert later.startswith("Chunk ID: a1b2c3\nWall time: 0.0100")
    assert "def paginate_orders" not in later  # body folded away
    # lossless: the folded body is still fully present earlier in the request
    assert "def paginate_orders" in items[2]["output"]


def test_single_read_does_not_fold():
    items = [_read_call("c1"), _read_output("c1", "492f0f", "0.0000")]
    folded, saved = _dedup_responses_output_items(
        items, _RESPONSES_OUTPUT_ITEM_TYPES, count_tokens=len
    )
    assert folded == 0 and saved == 0
    assert items[1]["output"] == _wrap("492f0f", "0.0000") + BODY


def test_protected_websearch_outputs_do_not_fold():
    items = [
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": '{\n  "results": [\n    {"title": "Headroom"}\n  ]\n}',
        },
        {
            "type": "function_call_output",
            "call_id": "c2",
            "output": '{\n  "results": [\n    {"title": "Headroom"}\n  ]\n}',
        },
    ]
    folded, saved = _dedup_responses_output_items(
        items,
        _RESPONSES_OUTPUT_ITEM_TYPES,
        count_tokens=len,
        protected_call_ids={"c1", "c2"},
    )

    assert folded == 0
    assert saved == 0
    assert items[0]["output"].endswith('{"title": "Headroom"}\n  ]\n}')
    assert items[1]["output"].endswith('{"title": "Headroom"}\n  ]\n}')


def test_non_output_items_untouched():
    # A duplicated MESSAGE (not a tool output) must never fold — only output
    # items are eligible.
    msg = {"role": "user", "content": BODY}
    items = [dict(msg), {"type": "message", "role": "assistant", "content": "ok"}, dict(msg)]
    folded, _ = _dedup_responses_output_items(items, _RESPONSES_OUTPUT_ITEM_TYPES)
    assert folded == 0
    assert items[2]["content"] == BODY


def test_never_raises_on_malformed():
    # Defensive: junk items must not blow up the request path.
    items = [{"type": "function_call_output"}, {"type": "function_call_output", "output": None}, 42]
    folded, saved = _dedup_responses_output_items(items, _RESPONSES_OUTPUT_ITEM_TYPES)  # type: ignore[arg-type]
    assert (folded, saved) == (0, 0)
