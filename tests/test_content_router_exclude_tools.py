"""Tests for --protect-tool-results / HEADROOM_PROTECT_TOOL_RESULTS.

Three behavioral tests:
1. protect_tool_results merges into the exclude set so named tools are never
   lossy-compressed.
2. _parse_csv_tools parses CSV strings without merging HEADROOM_EXCLUDE_TOOLS.
3. ContentRouter with Bash in exclude_tools passes Bash tool_result verbatim.
"""

from __future__ import annotations

import pytest

from headroom.config import DEFAULT_EXCLUDE_TOOLS
from headroom.proxy.server import (
    HeadroomProxy,
    ProxyConfig,
    _parse_csv_tools,
)


def _build(**overrides: object) -> HeadroomProxy:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        code_aware_enabled=False,
        **overrides,
    )
    return HeadroomProxy(config)


def _router(proxy: HeadroomProxy):
    # ContentRouter is the last transform in the Anthropic pipeline.
    return proxy.anthropic_pipeline.transforms[-1]


# ---------------------------------------------------------------------------
# Test 1: protect_tool_results merges into exclude set
# ---------------------------------------------------------------------------


def test_protect_tool_results_merges_into_exclude_set() -> None:
    """Bash added via protect_tool_results must appear in exclude_tools alongside
    the built-in defaults (e.g. Read), base-fails / head-passes.

    The frozenset is merged as-is; lowercase normalization is handled by
    _parse_exclude_tools on the CLI/env path (tested separately).
    """
    proxy = _build(protect_tool_results=frozenset({"Bash", "bash"}))
    exclude = _router(proxy).config.exclude_tools

    assert exclude is not None, "exclude_tools must be set when protect_tool_results is non-empty"
    assert "Bash" in exclude, "Bash must be in exclude_tools after protect_tool_results merges"
    assert "bash" in exclude, (
        "lowercase bash must be in exclude_tools after protect_tool_results merges"
    )
    assert "Read" in exclude, "Read (built-in default) must still be in exclude_tools"


def test_protect_tool_results_disables_age_decay_in_token_mode() -> None:
    """In token mode, protect_tool_results forces protect_recent_reads_fraction to 0.0
    so protected tools are never compressed by age-decay."""
    proxy = _build(protect_tool_results=frozenset({"Bash", "bash"}), mode="token")
    assert _router(proxy).config.protect_recent_reads_fraction == 0.0


# ---------------------------------------------------------------------------
# Test 2: CSV env var / CLI string parsing
# ---------------------------------------------------------------------------


def test_protect_tool_results_env_var_csv() -> None:
    """_parse_csv_tools parses a comma-separated value into both original-case
    and lowercase entries without merging HEADROOM_EXCLUDE_TOOLS."""
    result = _parse_csv_tools("Bash,WebFetch")

    assert "Bash" in result
    assert "bash" in result
    assert "WebFetch" in result
    assert "webfetch" in result


# ---------------------------------------------------------------------------
# Test 3: Bash tool_result passthrough when protected
# ---------------------------------------------------------------------------


def test_bash_tool_result_passthrough_when_protected() -> None:
    """When Bash is in exclude_tools (via protect_tool_results), its tool_result
    content passes through the ContentRouter verbatim without lossy compression.
    Base-fails / head-passes."""
    pytest.importorskip("tiktoken")  # needed for OpenAI tokenizer

    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    provider = OpenAIProvider()
    token_counter = provider.get_token_counter("gpt-4o")
    tokenizer = Tokenizer(token_counter, "gpt-4o")

    # Build router with Bash explicitly in exclude_tools
    config = ContentRouterConfig(
        min_section_tokens=10,
        exclude_tools=set(DEFAULT_EXCLUDE_TOOLS) | {"Bash", "bash"},
    )
    router = ContentRouter(config)

    bash_output = "\n".join(
        f"line {i}: some output from a bash command that is long enough to compress"
        for i in range(80)
    )
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_bash_1",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_bash_1",
            "content": bash_output,
        },
    ]

    result = router.apply(messages, tokenizer)

    # Bash tool_result must pass through unchanged
    tool_msg = next(m for m in result.messages if m.get("tool_call_id") == "call_bash_1")
    assert tool_msg["content"] == bash_output, (
        "Bash tool_result must be verbatim when Bash is in exclude_tools"
    )
    assert "router:excluded:tool" in result.transforms_applied


# ---------------------------------------------------------------------------
# Test 4: protect_tool_results sentinel survives a profile-derived
# read_protection_window kwarg, even when the protected output is old
# ---------------------------------------------------------------------------


def test_protect_tool_results_survives_runtime_read_protection_window_kwarg() -> None:
    """A profile-derived `read_protection_window` kwarg (e.g. from
    AgentSavingsProfile.protect_recent=2, threaded in via
    proxy_pipeline_kwargs()) must not shrink protection below what
    protect_recent_reads_fraction == 0.0 (the --protect-tool-results
    sentinel) already guarantees for the whole conversation.

    Regression test for the precedence bug: content_router.py used to apply
    the runtime kwarg unconditionally, so a Bash tool_result more than
    `read_protection_window` messages old fell through to lossy compression
    even though --protect-tool-results promised it would never compress
    "regardless of conversation depth" (see PR #1374)."""
    pytest.importorskip("tiktoken")  # needed for OpenAI tokenizer

    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer

    provider = OpenAIProvider()
    token_counter = provider.get_token_counter("gpt-4o")
    tokenizer = Tokenizer(token_counter, "gpt-4o")

    proxy = _build(protect_tool_results=frozenset({"Bash", "bash"}), mode="token")
    router = _router(proxy)

    bash_output = "\n".join(
        f"line {i}: some output from a bash command that is long enough to compress"
        for i in range(80)
    )
    messages: list[dict[str, object]] = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_bash_1",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_bash_1",
            "content": bash_output,
        },
    ]
    # Pad with enough intervening turns that the Bash tool_result above
    # falls outside a read_protection_window=2 (it's ~9-10 messages from
    # the end once padding is added).
    for i in range(8):
        messages.append({"role": "user", "content": f"follow-up turn {i}"})
        messages.append({"role": "assistant", "content": f"reply {i}"})

    # Simulate the profile-derived kwarg the proxy threads into every
    # request via proxy_pipeline_kwargs() (AgentSavingsProfile("coding")
    # sets protect_recent=2).
    result = router.apply(messages, tokenizer, read_protection_window=2)

    tool_msg = next(m for m in result.messages if m.get("tool_call_id") == "call_bash_1")
    assert tool_msg["content"] == bash_output, (
        "Bash tool_result must stay verbatim: protect_recent_reads_fraction == 0.0 "
        "(set by --protect-tool-results) must not be weakened by a profile-derived "
        "read_protection_window kwarg"
    )
    assert "router:excluded:tool" in result.transforms_applied


# ---------------------------------------------------------------------------
# Baseline: Bash NOT in DEFAULT_EXCLUDE_TOOLS (unchanged by this PR)
# ---------------------------------------------------------------------------


def test_bash_not_in_default_exclude_tools() -> None:
    """Bash must remain absent from DEFAULT_EXCLUDE_TOOLS; protect_tool_results
    is the opt-in path."""
    assert "Bash" not in DEFAULT_EXCLUDE_TOOLS
    assert "bash" not in DEFAULT_EXCLUDE_TOOLS
