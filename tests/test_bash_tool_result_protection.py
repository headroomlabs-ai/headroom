"""Bash tool results are excluded from lossy compression by default (#3652).

Columnar shell output (`ls -la`, `git status`) classifies as PLAIN_TEXT and
used to lose unmarked fields to Kompress. Bash is in DEFAULT_EXCLUDE_TOOLS so
those bytes stay intact the same way Read/Grep/view results do.
"""

from __future__ import annotations

import random

from headroom.config import DEFAULT_EXCLUDE_TOOLS, is_tool_excluded
from headroom.transforms.content_detector import ContentType
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
    RoutingDecision,
)


class _Tokenizer:
    def count_text(self, text: str) -> int:
        return max(1, len(text) // 4)


def _messages(tool_name: str, payload: str) -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": tool_name,
                    "input": {"command": "ls -la"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "content": payload,
                }
            ],
        },
    ]


def _router() -> ContentRouter:
    router = ContentRouter(ContentRouterConfig())
    router.config.min_section_tokens = 1
    router.config.min_chars_for_block_compression = 1
    return router


def _tool_text(result) -> str:
    block = result.messages[1]["content"][0]
    return str(block["content"])


def _ls_la_payload() -> str:
    rng = random.Random(1)
    rows = [
        f"-rw-r--r--  1 tejas staff {rng.randint(1000, 99999)} Sep {d} 09:{d:02d} file_{d}.py"
        for d in range(1, 60)
    ]
    return "total 480\n" + "\n".join(rows)


def _git_status_payload() -> str:
    lines = ["On branch main", "Changes not staged for commit:"]
    lines.extend(f"\tmodified:   src/module_{i}.py" for i in range(30))
    return "\n".join(lines)


def test_bash_is_a_default_exclusion() -> None:
    assert {"Bash", "bash"} <= DEFAULT_EXCLUDE_TOOLS
    assert is_tool_excluded("Bash", DEFAULT_EXCLUDE_TOOLS) is True
    assert is_tool_excluded("bash", DEFAULT_EXCLUDE_TOOLS) is True


def test_bash_tool_result_bypasses_lossy_compressor() -> None:
    router = _router()
    calls = 0

    def fake_compress(*args: object, **kwargs: object) -> RouterCompressionResult:
        nonlocal calls
        calls += 1
        content = str(args[0])
        return RouterCompressionResult(
            compressed="mutated",
            original=content,
            strategy_used=CompressionStrategy.TEXT,
            routing_log=[
                RoutingDecision(ContentType.PLAIN_TEXT, CompressionStrategy.TEXT, 100, 10)
            ],
        )

    router.compress = fake_compress  # type: ignore[method-assign]
    payload = _ls_la_payload()

    for tool_name in ("Bash", "bash"):
        calls = 0
        result = router.apply(_messages(tool_name, payload), _Tokenizer())
        assert calls == 0
        assert any(t.startswith("router:excluded:") for t in result.transforms_applied)
        assert _tool_text(result) != "mutated"


def test_ls_la_owner_fields_survive_default_router() -> None:
    payload = _ls_la_payload()
    result = _router().apply(_messages("Bash", payload), _Tokenizer())
    out = _tool_text(result)

    assert out.count("tejas") == 59
    assert out.count("-rw-r--r--") == 59
    assert "words compressed to" not in out
    assert any(t.startswith("router:excluded:") for t in result.transforms_applied)


def test_git_status_paths_survive_default_router() -> None:
    payload = _git_status_payload()
    result = _router().apply(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Bash",
                        "input": {"command": "git status"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": payload,
                    }
                ],
            },
        ],
        _Tokenizer(),
    )
    out = _tool_text(result)

    assert out.count("modified:") == 30
    for i in range(30):
        assert f"src/module_{i}.py" in out
    assert "words compressed to" not in out
    assert any(t.startswith("router:excluded:") for t in result.transforms_applied)
