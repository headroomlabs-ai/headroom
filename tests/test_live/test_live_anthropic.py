"""Live Anthropic API tests for the Claude Code (tool_result block) path.

Validates the two claims unit tests cannot: the real API accepts our
transformed message shapes, and the model can still answer correctly from
them — the no-accuracy-loss contract, end to end.

Skipped without ANTHROPIC_API_KEY. Costs: a few hundred haiku tokens/run.
"""

from __future__ import annotations

import os

import httpx
import pytest

from headroom.transforms.search_compressor import SearchCompressor, SearchCompressorConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

MODEL = "claude-haiku-4-5-20251001"
API_URL = "https://api.anthropic.com/v1/messages"

FILE_CONTENT = (
    '     1\tdef answer():\n     2\t    """Returns the magic number."""\n     3\t    return 42\n'
) + "".join(f"    {i}\t# padding line {i}\n" for i in range(4, 40))

READ_TOOL = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    },
}


def call_anthropic(messages: list[dict], system: str | None = None) -> str:
    body: dict = {
        "model": MODEL,
        "max_tokens": 150,
        "tools": [READ_TOOL],
        "messages": messages,
    }
    if system:
        body["system"] = system
    resp = httpx.post(
        API_URL,
        json=body,
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
        timeout=60,
    )
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:500]}"
    return "".join(
        block.get("text", "") for block in resp.json()["content"] if block.get("type") == "text"
    )


def read_roundtrip(tc_id: str, content: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tc_id,
                    "name": "Read",
                    "input": {"file_path": "/src/magic.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tc_id, "content": content}],
        },
    ]


class TestLifecycleMarkerLive:
    def test_api_accepts_stale_marker_shape(self):
        """A stale-Read marker (file edited after read) must be a valid
        message body and not confuse the model into inventing content."""
        messages = [{"role": "user", "content": "Read /src/magic.py"}]
        messages += read_roundtrip("toolu_r1", FILE_CONTENT)
        messages += [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_e1",
                        "name": "Read",  # registered tool; the marker is what matters
                        "input": {"file_path": "/src/magic.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_e1",
                        "content": "[Read content stale: /src/magic.py was modified after this "
                        "read — re-read the file for current content. "
                        "Retrieve original: hash=abc123def456abc123def456]",
                    }
                ],
            },
            {
                "role": "user",
                "content": "Is the content of /src/magic.py in this conversation current or "
                "stale? One word.",
            },
        ]
        reply = call_anthropic(messages)
        assert "stale" in reply.lower(), f"model misread the marker: {reply!r}"


class TestGroupedSearchLive:
    def test_model_reads_grouped_format(self):
        raw = "\n".join(
            [
                "src/payments/processor.py:12:def charge(amount):",
                "src/payments/processor.py:45:def refund(amount):",
                "src/users/auth.py:7:def login(user):",
            ]
        )
        grouped = SearchCompressor(SearchCompressorConfig(group_by_file=True)).compress(raw)
        messages = [
            {
                "role": "user",
                "content": "Here are grep results for 'def ':\n\n"
                + grouped.compressed
                + "\n\nWhich file defines refund()? Reply with just the path.",
            }
        ]
        reply = call_anthropic(messages)
        assert "src/payments/processor.py" in reply, f"grouped format misread: {reply!r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
