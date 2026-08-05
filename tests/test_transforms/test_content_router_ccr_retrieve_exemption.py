"""Regression tests: ContentRouter must not re-compress headroom_retrieve results.

Companion to ``test_smart_crusher_ccr_retrieve_exemption.py`` (issue #1077).
SmartCrusher.apply() already guards this exact failure mode, but ContentRouter
— the transform actually registered in the default/proxy pipeline (see
``transforms/pipeline.py``) — calls ``SmartCrusher.crush()`` directly, bypassing
that guard entirely. Without this fix, a ``headroom_retrieve`` tool result sent
back to the model on the next turn gets swept up by ContentRouter's own
compression routing like any other large tool output, producing a fresh
``<<ccr:hash>>`` marker the agent can never redeem (an unresolvable retrieval
loop — the original reported bug).

Tests use the fully-qualified MCP tool name (``mcp__headroom__headroom_retrieve``)
that Claude Code actually sends when connected to `headroom mcp serve` — not the
bare name — since routing the guard through ``is_tool_excluded()`` (alias-aware)
rather than a bare string comparison is the whole point of the fix.
"""

from __future__ import annotations

import json

from headroom.transforms.content_router import ContentRouter, ContentRouterConfig


def _get_tokenizer():
    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer

    provider = OpenAIProvider()
    token_counter = provider.get_token_counter("gpt-4o")
    return Tokenizer(token_counter, "gpt-4o")


def _big_json() -> str:
    """A JSON array large enough to clear the compression threshold."""
    return json.dumps([{"id": i, "value": "x" * 20, "active": i % 2 == 0} for i in range(60)])


def _anthropic_messages(tool_name: str, content: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_ccr_1",
                    "name": tool_name,
                    "input": {"hash": "abc123def456"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_ccr_1",
                    "content": content,
                }
            ],
        },
    ]


def _openai_messages(tool_name: str, content: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_ccr_1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": '{"hash":"abc123def456"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_ccr_1", "content": content},
    ]


def _tool_role_top_level_text_messages(tool_name: str, content: str) -> list[dict]:
    """Some harnesses normalize a tool-role message's content to a top-level
    list of {"type": "text"} blocks, without the {"type": "tool_result", ...}
    wrapper the primary Anthropic-shape tests exercise. This is a distinct
    code path in _process_content_blocks (the generic `text` block branch)."""
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_ccr_toplevel",
                    "name": tool_name,
                    "input": {"hash": "abc123def456"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_ccr_toplevel",
            "content": [{"type": "text", "text": content}],
        },
    ]


def _legacy_function_messages(tool_name: str, content: str) -> list[dict]:
    """Legacy (pre-parallel-tool-calls) OpenAI function-calling: the request
    uses a singular top-level `function_call` field (not `tool_calls`), and
    the response is a role:"function" message carrying the name directly --
    this shape has no call id at all, unlike role:"tool"."""
    return [
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": tool_name, "arguments": '{"hash":"abc123def456"}'},
        },
        {"role": "function", "name": tool_name, "content": content},
    ]


def _anthropic_list_form_messages(tool_name: str, content: str) -> list[dict]:
    """litellm/OpenAI-compatible gateways sometimes wrap tool_result content
    as a list of {"type": "text"} blocks even in Anthropic-shaped messages
    (see content_router.py's _tr_list_form_early flatten, fix-7)."""
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_ccr_listform",
                    "name": tool_name,
                    "input": {"hash": "abc123def456"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_ccr_listform",
                    "content": [{"type": "text", "text": content}],
                }
            ],
        },
    ]


class TestContentRouterCcrRetrieveExemption:
    def test_anthropic_qualified_mcp_name_not_recompressed(self):
        """The MCP-qualified name Claude Code actually sends must be exempted."""
        content = _big_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = _anthropic_messages("mcp__headroom__headroom_retrieve", content)
        result = router.apply(messages, tokenizer)

        tool_result_block = result.messages[1]["content"][0]
        assert tool_result_block["content"] == content, (
            "headroom_retrieve result was recompressed via ContentRouter "
            "(unresolvable retrieval loop)"
        )
        assert "<<ccr:" not in tool_result_block["content"]
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_openai_qualified_mcp_name_not_recompressed(self):
        """Same guarantee for the OpenAI/litellm tool-message shape."""
        content = _big_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = _openai_messages("mcp__headroom__headroom_retrieve", content)
        result = router.apply(messages, tokenizer)

        tool_msg = next(m for m in result.messages if m.get("tool_call_id") == "call_ccr_1")
        assert tool_msg["content"] == content
        assert "<<ccr:" not in tool_msg["content"]
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_bare_tool_name_also_protected(self):
        """The proxy's own internally-injected retrieval tool uses the bare
        name (not MCP-qualified) — must be protected too, matching the
        existing SmartCrusher.apply() coverage for this call shape."""
        content = _big_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = _anthropic_messages("headroom_retrieve", content)
        result = router.apply(messages, tokenizer)

        tool_result_block = result.messages[1]["content"][0]
        assert tool_result_block["content"] == content
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_unconditional_even_with_empty_exclude_tools(self):
        """The guard is config-independent: even a caller that explicitly
        empties exclude_tools (disabling every default exclusion) must not
        be able to recompress a headroom_retrieve result. This is the
        defense-in-depth half of the fix — config.py's DEFAULT_EXCLUDE_TOOLS/
        DEFAULT_VERBATIM_EXCLUDE_TOOLS additions alone would not survive this
        override, since ContentRouter replaces (not merges) exclude_tools
        when the caller sets it explicitly."""
        content = _big_json()
        router = ContentRouter(
            ContentRouterConfig(min_section_tokens=10, exclude_tools=frozenset())
        )
        tokenizer = _get_tokenizer()

        messages = _anthropic_messages("mcp__headroom__headroom_retrieve", content)
        result = router.apply(messages, tokenizer)

        tool_result_block = result.messages[1]["content"][0]
        assert tool_result_block["content"] == content, (
            "headroom_retrieve must stay protected even when exclude_tools is "
            "explicitly emptied — the guard must not depend on config"
        )
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_non_retrieve_tool_still_compressed(self):
        """Exemption is narrow: an ordinary tool with the same large JSON
        content is still compressed — this isn't a blanket JSON passthrough."""
        content = _big_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = _anthropic_messages("Bash", content)
        result = router.apply(messages, tokenizer)

        assert "router:excluded:ccr_retrieve" not in result.transforms_applied
        tool_result_block = result.messages[1]["content"][0]
        assert tool_result_block["content"] != content or result.tokens_after < result.tokens_before

    def test_top_level_text_block_under_tool_role_not_recompressed(self):
        """A harness that normalizes tool-role content to a top-level text
        block (not wrapped in tool_result) must be protected too -- this is
        a distinct code path (_process_content_blocks' generic `text` branch)
        from the tool_result branch the other Anthropic-shape tests exercise."""
        content = _big_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = _tool_role_top_level_text_messages("mcp__headroom__headroom_retrieve", content)
        result = router.apply(messages, tokenizer)

        tool_msg = next(m for m in result.messages if m.get("role") == "tool")
        assert tool_msg["content"][0]["text"] == content
        assert "<<ccr:" not in tool_msg["content"][0]["text"]
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_legacy_openai_function_role_not_recompressed(self):
        """Legacy OpenAI function-calling (role:"function", no tool_call_id --
        the tool name is carried directly on the message) must be protected
        too. This shape predates parallel tool_calls and bypasses
        _build_tool_name_map's id-keyed lookup entirely."""
        content = _big_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = _legacy_function_messages("headroom_retrieve", content)
        result = router.apply(messages, tokenizer)

        fn_msg = next(m for m in result.messages if m.get("role") == "function")
        assert fn_msg["content"] == content
        assert "<<ccr:" not in fn_msg["content"]
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_litellm_list_form_tool_result_not_recompressed(self):
        """litellm/OpenAI-style list-form content (a list of {"type":"text"}
        blocks) nested inside an Anthropic tool_result block must be
        protected too -- distinct from the plain-string tool_result content
        the other Anthropic-shape tests exercise (see fix-7 flatten)."""
        content = _big_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = _anthropic_list_form_messages("mcp__headroom__headroom_retrieve", content)
        result = router.apply(messages, tokenizer)

        tool_result_block = result.messages[1]["content"][0]
        assert tool_result_block["content"] == [{"type": "text", "text": content}]
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_mixed_retrieve_and_normal_in_one_turn_only_normal_compressed(self):
        """Per-block precision: a single turn carrying both a headroom_retrieve
        tool_result and a normal tool_result must protect only the former --
        parity with test_smart_crusher_ccr_retrieve_exemption.py's
        test_mixed_retrieve_and_normal_only_normal_compressed."""
        ccr_content = _big_json()
        normal_content = _big_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_ccr_mixed",
                        "name": "mcp__headroom__headroom_retrieve",
                        "input": {"hash": "abc123def456"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_normal_mixed",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_ccr_mixed",
                        "content": ccr_content,
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_normal_mixed",
                        "content": normal_content,
                    },
                ],
            },
        ]
        result = router.apply(messages, tokenizer)

        blocks = result.messages[1]["content"]
        ccr_block = next(b for b in blocks if b["tool_use_id"] == "toolu_ccr_mixed")
        normal_block = next(b for b in blocks if b["tool_use_id"] == "toolu_normal_mixed")

        assert ccr_block["content"] == ccr_content
        assert normal_block["content"] != normal_content
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_small_headroom_retrieve_content_still_marked_excluded(self):
        """The guard is size-independent: even content well below the
        compression floor must still get the router:excluded:ccr_retrieve
        marker, proving the exemption fires unconditionally rather than
        happening to survive because it was too small to compress anyway."""
        content = "tiny retrieved value"
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = _anthropic_messages("mcp__headroom__headroom_retrieve", content)
        result = router.apply(messages, tokenizer)

        tool_result_block = result.messages[1]["content"][0]
        assert tool_result_block["content"] == content
        assert "router:excluded:ccr_retrieve" in result.transforms_applied
