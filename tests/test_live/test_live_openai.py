"""Live OpenAI API tests for the Codex (role="tool" message) path.

Same contract as the Anthropic live tests: the real API must accept our
transformed message shapes (lifecycle markers inside role="tool" results),
and the model must read them correctly.

Skipped without OPENAI_API_KEY. Costs: a few hundred gpt-4o-mini tokens/run.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

MODEL = "gpt-4o-mini"
API_URL = "https://api.openai.com/v1/chat/completions"

FILE_CONTENT = (
    '     1\tdef answer():\n     2\t    """Returns the magic number."""\n     3\t    return 42\n'
) + "".join(f"    {i}\t# padding line {i}\n" for i in range(4, 40))

STALE_MARKER = (
    "[Read content stale: /src/magic.py was modified after this read — "
    "re-read the file for current content. "
    "Retrieve original: hash=abc123def456abc123def456]"
)

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
}


def call_openai(messages: list[dict]) -> str:
    resp = httpx.post(
        API_URL,
        json={
            "model": MODEL,
            "max_tokens": 150,
            "tools": [READ_TOOL],
            "messages": messages,
        },
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        timeout=60,
    )
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:500]}"
    return resp.json()["choices"][0]["message"]["content"] or ""


def read_roundtrip(tc_id: str, content: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "/src/magic.py"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": tc_id, "content": content},
    ]


class TestLifecycleMarkerLiveOpenAI:
    def test_api_accepts_stale_marker_shape(self):
        """A stale-Read marker inside a role='tool' message must be a
        valid body and must be read as 'this content is outdated'."""
        messages = [{"role": "user", "content": "Read /src/magic.py"}]
        messages += read_roundtrip("call_r1", FILE_CONTENT)
        messages += [
            {"role": "assistant", "content": "Read it. Anything else?"},
            {"role": "user", "content": "Check it once more."},
        ]
        messages += read_roundtrip("call_r2", STALE_MARKER)
        messages.append(
            {
                "role": "user",
                "content": "Is the latest read of /src/magic.py in this conversation "
                "current or stale? One word.",
            }
        )
        reply = call_openai(messages)
        assert "stale" in reply.lower(), f"model misread the marker: {reply!r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
