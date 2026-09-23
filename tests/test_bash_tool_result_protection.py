"""Raw-shell tool results are excluded from lossy compression by default (#3652).

Columnar shell output (`ls -la`, `git status`) classifies as PLAIN_TEXT and
used to lose unmarked fields to Kompress. Every name in
``ContentRouterConfig.bash_tool_names`` (``bash``, ``shell``, ``local_shell``),
plus Claude Code's ``Bash`` spelling, is in DEFAULT_EXCLUDE_TOOLS. Matching is
case-insensitive, the same way the router lowercases the tool name, so those
bytes stay intact the same way Read/Grep/view results do.
"""

from __future__ import annotations

import random

from headroom.config import DEFAULT_BASH_TOOL_NAMES, DEFAULT_EXCLUDE_TOOLS, is_tool_excluded
from headroom.transforms.content_detector import ContentType
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
    RoutingDecision,
)

# Spellings the router accepts: lowercase bash_tool_names, Claude's Bash, and
# the mixed-case forms tool_name.lower() folds onto those entries.
_SHELL_TOOL_NAMES = (
    "Bash",
    "bash",
    "shell",
    "local_shell",
    "Shell",
    "Local_Shell",
)


class _Tokenizer:
    def count_text(self, text: str) -> int:
        return max(1, len(text) // 4)


def _messages(tool_name: str, payload: str, command: str = "ls -la") -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": tool_name,
                    "input": {"command": command},
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


def test_every_raw_shell_name_is_a_default_exclusion() -> None:
    """Router shell names and their case-folded spellings skip lossy compression."""
    configured = ContentRouterConfig().bash_tool_names
    assert configured == DEFAULT_BASH_TOOL_NAMES
    assert configured == frozenset({"bash", "shell", "local_shell"})
    # Every router shell name is excluded, not only the spellings listed above.
    assert DEFAULT_BASH_TOOL_NAMES <= DEFAULT_EXCLUDE_TOOLS
    assert "Bash" in DEFAULT_EXCLUDE_TOOLS
    for name in (
        *_SHELL_TOOL_NAMES,
        "SHELL",
        "LOCAL_SHELL",
        "Local_shell",
    ):
        assert is_tool_excluded(name, DEFAULT_EXCLUDE_TOOLS) is True, name


def test_raw_shell_tool_result_bypasses_lossy_compressor() -> None:
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

    for tool_name in _SHELL_TOOL_NAMES:
        calls = 0
        result = router.apply(_messages(tool_name, payload), _Tokenizer())
        assert calls == 0, tool_name
        assert any(t.startswith("router:excluded:") for t in result.transforms_applied), tool_name
        assert _tool_text(result) != "mutated"


def test_ls_la_owner_fields_survive_default_router() -> None:
    payload = _ls_la_payload()
    for tool_name in ("shell", "local_shell", "Bash", "bash", "Shell", "Local_Shell"):
        result = _router().apply(_messages(tool_name, payload), _Tokenizer())
        out = _tool_text(result)

        assert out == payload, tool_name
        assert out.count("tejas") == 59, tool_name
        assert out.count("-rw-r--r--") == 59, tool_name
        assert "words compressed to" not in out
        assert any(t.startswith("router:excluded:") for t in result.transforms_applied), tool_name


def test_git_status_paths_survive_default_router() -> None:
    payload = _git_status_payload()
    for tool_name in ("shell", "local_shell", "Bash", "bash", "Shell", "Local_Shell"):
        result = _router().apply(
            _messages(tool_name, payload, command="git status"),
            _Tokenizer(),
        )
        out = _tool_text(result)

        assert out == payload, tool_name
        assert out.count("modified:") == 30, tool_name
        for i in range(30):
            assert f"src/module_{i}.py" in out
        assert "words compressed to" not in out
        assert any(t.startswith("router:excluded:") for t in result.transforms_applied), tool_name
