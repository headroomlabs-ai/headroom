"""Tests for the --force-kompress-all / HEADROOM_FORCE_KOMPRESS_ALL flag.

force_kompress_all routes ALL compressible content through Kompress, bypassing
per-type compressor selection. Critically it must NOT change protection: excluded
tools (Read/Glob/...) stay verbatim. These tests verify the config -> runtime
wiring and that the Read/Glob carve-out still holds with the flag on. They use an
excluded tool's output so no Kompress model load is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from headroom.config import DEFAULT_EXCLUDE_TOOLS
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

if TYPE_CHECKING:
    from headroom.tokenizer import Tokenizer


def _tokenizer() -> Tokenizer:
    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer

    provider = OpenAIProvider()
    return Tokenizer(provider.get_token_counter("gpt-4o"), "gpt-4o")


def _read_messages() -> list[dict]:
    """A Read tool_result. Read is in DEFAULT_EXCLUDE_TOOLS, so it is never compressed."""
    file_dump = "\n".join(f"line {i}: contents of a file that Read returned" for i in range(80))
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_read_1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_read_1", "content": file_dump},
    ]


def test_read_is_a_default_excluded_tool() -> None:
    """Guards the carve-out's premise: Read ships in DEFAULT_EXCLUDE_TOOLS."""
    assert "Read" in DEFAULT_EXCLUDE_TOOLS


def test_config_sets_runtime_force_kompress() -> None:
    """force_kompress_all=True in config resolves the runtime flag on; default off."""
    pytest.importorskip("tiktoken")
    tokenizer = _tokenizer()

    on = ContentRouter(ContentRouterConfig(force_kompress_all=True))
    on.apply(_read_messages(), tokenizer)
    assert on._runtime_force_kompress is True

    off = ContentRouter(ContentRouterConfig())
    off.apply(_read_messages(), tokenizer)
    assert off._runtime_force_kompress is False


def test_per_request_kwarg_overrides_config() -> None:
    """An explicit force_kompress kwarg still wins over the config default."""
    pytest.importorskip("tiktoken")
    tokenizer = _tokenizer()

    router = ContentRouter(ContentRouterConfig(force_kompress_all=True))
    router.apply(_read_messages(), tokenizer, force_kompress=False)
    assert router._runtime_force_kompress is False


def test_read_output_verbatim_under_force_kompress_all() -> None:
    """The carve-out: with force_kompress_all on, Read output (an excluded tool)
    is passed through verbatim — never routed to Kompress."""
    pytest.importorskip("tiktoken")
    tokenizer = _tokenizer()

    messages = _read_messages()
    original = messages[1]["content"]
    router = ContentRouter(ContentRouterConfig(force_kompress_all=True, min_section_tokens=10))
    result = router.apply(messages, tokenizer)

    tool_msg = next(m for m in result.messages if m.get("tool_call_id") == "call_read_1")
    assert tool_msg["content"] == original, "Read tool_result must stay verbatim"
    assert "router:excluded:tool" in result.transforms_applied
