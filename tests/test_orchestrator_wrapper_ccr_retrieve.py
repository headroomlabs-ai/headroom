"""Regression tests: a headroom_retrieve result delivered through an orchestrator
wrapper must keep the ccr_retrieve exemption (#3563).

OpenCode V2 Code Mode runs every MCP call inside its built-in ``execute`` tool,
and Codex code mode sends calls as ``exec`` / ``functions.exec`` custom tool
calls whose payload is JavaScript. On the wire the tool name is the wrapper --
the inner name never reaches the proxy -- so the exemption that keys on the tool
name missed, SmartCrusher re-offloaded the retrieved bytes into a fresh
``<<ccr:hash>>`` marker, and the model could never redeem it (the same
unresolvable retrieval loop class as #1077 / #2698).

The fix resolves such a call to the retrieval tool's own name when -- and only
when -- the call's own payload invokes ``headroom_retrieve``. The wrapper is
never exempted as a whole, and the result content is never consulted (an
``original_content`` property is user-controllable data, not recovery proof).
"""

from __future__ import annotations

import json

from headroom.config import unwrap_tool_call
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig


def _get_tokenizer():
    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer

    provider = OpenAIProvider()
    token_counter = provider.get_token_counter("gpt-4o")
    return Tokenizer(token_counter, "gpt-4o")


def _big_retrieve_json() -> str:
    """The JSON object the retrieval tool returns: big enough to clear the
    compression threshold, shaped like the #3563 repro (``original_content``)."""
    rows = "\n".join(
        f"Row {i:03d}: synthetic recovery check; status=healthy; payload=harmless sample."
        for i in range(90)
    )
    return json.dumps({"original_content": rows})


def _orchestrator_messages(wrapper_name: str, code: str, content: str) -> list[dict]:
    """OpenAI-shape conversation: an outer wrapper call, then its tool result."""
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_ccr_exec",
                    "type": "function",
                    "function": {"name": wrapper_name, "arguments": json.dumps({"code": code})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_ccr_exec", "content": content},
    ]


RETRIEVE_CODE = (
    'const r = await tools.headroom.headroom_retrieve({hash: "abc123def456"});\nreturn r;'
)
NON_RETRIEVE_CODE = 'const r = await tools.bash.bash({command: "ls"});\nreturn r;'
# A real newline between the callee and its `(`. On the OpenAI wire the payload
# is JSON, so this arrives as the two escaped characters `\n` and a scan of the
# raw JSON text misses it; the call must be recognized on the decoded script.
RETRIEVE_CODE_MULTILINE = (
    'const r = await tools.headroom.headroom_retrieve\n({hash: "abc123def456"});\nreturn r;'
)


class TestOrchestratorUnwrap:
    """Unit coverage for the identity derivation itself."""

    def test_opencode_execute_wrapper_resolves_to_retrieve(self):
        name, _args = unwrap_tool_call("execute", json.dumps({"code": RETRIEVE_CODE}))
        assert name == "headroom_retrieve"

    def test_qualified_mcp_name_inside_script_resolves(self):
        code = 'const r = await tools.mcp__headroom__headroom_retrieve({hash: "abc"});'
        assert unwrap_tool_call("execute", json.dumps({"code": code}))[0] == "headroom_retrieve"

    def test_bracket_access_inside_script_resolves(self):
        code = 'const r = await tools["headroom_retrieve"]({hash: "abc"});'
        assert unwrap_tool_call("execute", json.dumps({"code": code}))[0] == "headroom_retrieve"

    def test_execute_wrapper_without_retrieve_call_keeps_name(self):
        assert unwrap_tool_call("execute", json.dumps({"code": NON_RETRIEVE_CODE}))[0] == "execute"

    def test_non_wrapper_mentioning_retrieve_keeps_name(self):
        # A non-orchestrator tool whose arguments merely name the tool is not
        # recovery output; only recognized wrappers resolve.
        args = '{"command": "grep -n headroom_retrieve( README.md"}'
        assert unwrap_tool_call("bash", args)[0] == "bash"

    def test_wrapper_payload_mentioning_retrieve_without_a_call_keeps_name(self):
        assert unwrap_tool_call("execute", json.dumps({"code": "// see headroom_retrieve"}))[0] == (
            "execute"
        )

    def test_json_encoded_newline_before_call_resolves(self):
        """The OpenAI wire JSON-encodes the payload, so the script's real
        newline arrives as the two characters ``\\n``; resolving must happen on
        the decoded script text, not on the raw JSON text (#3915 review)."""
        code = 'const r = await tools.headroom.headroom_retrieve\n({hash: "abc"});'
        assert unwrap_tool_call("execute", json.dumps({"code": code}))[0] == "headroom_retrieve"

    def test_json_encoded_tab_before_call_resolves(self):
        code = 'const r = await tools.headroom.headroom_retrieve\t({hash: "abc"});'
        assert unwrap_tool_call("execute", json.dumps({"code": code}))[0] == "headroom_retrieve"

    def test_json_encoded_newline_before_bracket_call_resolves(self):
        code = 'const r = await tools["headroom_retrieve"]\n({hash: "abc"});'
        assert unwrap_tool_call("execute", json.dumps({"code": code}))[0] == "headroom_retrieve"

    def test_anthropic_dict_payload_with_newline_resolves(self):
        """The Anthropic wire hands over a decoded dict; it must be scanned
        directly, without a json.dumps round-trip re-encoding the newline."""
        args = {"code": 'const r = await tools.headroom.headroom_retrieve\n({hash: "abc"});'}
        assert unwrap_tool_call("execute", args)[0] == "headroom_retrieve"

    def test_nested_containers_in_dict_payload_resolve(self):
        args = {"meta": {"attempt": 1}, "calls": [{"code": RETRIEVE_CODE_MULTILINE}]}
        assert unwrap_tool_call("execute", args)[0] == "headroom_retrieve"

    def test_raw_javascript_payload_with_newline_resolves(self):
        """A custom_tool_call payload that is not JSON is scanned as-is."""
        code = 'const r = await tools.headroom.headroom_retrieve\n({hash: "abc"});'
        assert unwrap_tool_call("execute", code)[0] == "headroom_retrieve"

    def test_string_values_without_the_name_are_skipped(self):
        args = {"note": "no tool call here", "code": RETRIEVE_CODE_MULTILINE}
        assert unwrap_tool_call("execute", args)[0] == "headroom_retrieve"

    def test_dict_payload_without_string_scripts_keeps_name(self):
        args = {"code": 42, "meta": {"attempt": 1}, "flags": [True, None]}
        assert unwrap_tool_call("execute", args)[0] == "execute"

    def test_non_string_non_dict_arguments_fail_closed(self):
        for args in (None, 42, ["headroom_retrieve("]):
            assert unwrap_tool_call("execute", args)[0] == "execute"

    def test_json_scalar_arguments_fail_closed(self):
        assert unwrap_tool_call("execute", "42")[0] == "execute"

    def test_unrelated_call_chain_keeps_name(self):
        # The mention satisfies the substring prefilter but no chain resolves.
        code = "// see headroom_retrieve docs\nrun(1);"
        assert unwrap_tool_call("execute", json.dumps({"code": code}))[0] == "execute"

    def test_unrelated_bracket_access_keeps_name(self):
        code = '// see headroom_retrieve docs\ntools["other"]();'
        assert unwrap_tool_call("execute", json.dumps({"code": code}))[0] == "execute"

    def test_optional_invocation_resolves(self):
        """`retrieve?.(...)` is a valid JavaScript call; its bytes need the
        exemption like a plain invocation (#3915 review)."""
        code = 'const r = await tools.headroom.headroom_retrieve?.({hash: "abc"});'
        assert unwrap_tool_call("execute", json.dumps({"code": code}))[0] == "headroom_retrieve"

    def test_optional_property_access_resolves(self):
        code = 'const r = await tools?.headroom.headroom_retrieve({hash: "abc"});'
        assert unwrap_tool_call("execute", json.dumps({"code": code}))[0] == "headroom_retrieve"

    def test_optional_bracket_invocation_resolves(self):
        code = 'const r = await tools["headroom_retrieve"]?.({hash: "abc"});'
        assert unwrap_tool_call("execute", json.dumps({"code": code}))[0] == "headroom_retrieve"

    def test_optional_access_around_bracket_call_resolves(self):
        code = 'const r = await tools?.["headroom_retrieve"]?.({hash: "abc"});'
        assert unwrap_tool_call("execute", json.dumps({"code": code}))[0] == "headroom_retrieve"

    def test_optional_chain_without_retrieve_keeps_name(self):
        code = 'const r = await tools?.bash.bash?.({command: "ls"});'
        assert unwrap_tool_call("execute", json.dumps({"code": code}))[0] == "execute"

    def test_hermes_bridge_over_orchestrator_retrieve_resolves(self):
        """`tool_call` wrapping an `execute` that runs the retrieval tool: the
        name must resolve through the nested orchestrator, not stop at
        `execute` (#3915 review)."""
        args = json.dumps({"name": "execute", "arguments": {"code": RETRIEVE_CODE}})
        name, inner = unwrap_tool_call("tool_call", args)
        assert name == "headroom_retrieve"
        assert inner == {"code": RETRIEVE_CODE}

    def test_hermes_bridge_over_orchestrator_without_retrieve_keeps_name(self):
        args = {"name": "execute", "arguments": {"code": NON_RETRIEVE_CODE}}
        assert unwrap_tool_call("tool_call", args)[0] == "execute"


class TestOrchestratorWrapperCcrRetrieveExemption:
    def test_execute_wrapper_retrieve_result_not_recompressed(self):
        """OpenCode V2 Code Mode: the execute wrapper's retrieval result must
        pass through verbatim instead of being re-offloaded."""
        content = _big_retrieve_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = _orchestrator_messages("execute", RETRIEVE_CODE, content)
        result = router.apply(messages, tokenizer)

        tool_msg = next(m for m in result.messages if m.get("role") == "tool")
        assert tool_msg["content"] == content, (
            "headroom_retrieve result behind the execute wrapper was recompressed "
            "(unresolvable retrieval loop, #3563)"
        )
        assert "<<ccr:" not in tool_msg["content"]
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_functions_exec_wrapper_retrieve_result_not_recompressed(self):
        """Codex code-mode shape: same guarantee under `functions.exec`."""
        content = _big_retrieve_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        code = 'const r = await tools.mcp__headroom__headroom_retrieve({hash: "abc"});\nreturn r;'
        messages = _orchestrator_messages("functions.exec", code, content)
        result = router.apply(messages, tokenizer)

        tool_msg = next(m for m in result.messages if m.get("role") == "tool")
        assert tool_msg["content"] == content
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_execute_wrapper_multiline_retrieve_result_not_recompressed(self):
        """Codex code-mode shapes can break the call line before the `(`; the
        decoded script must still resolve and keep the exemption."""
        content = _big_retrieve_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = _orchestrator_messages("execute", RETRIEVE_CODE_MULTILINE, content)
        result = router.apply(messages, tokenizer)

        tool_msg = next(m for m in result.messages if m.get("role") == "tool")
        assert tool_msg["content"] == content, (
            "multiline headroom_retrieve call behind the execute wrapper was recompressed "
            "(unresolvable retrieval loop, #3563)"
        )
        assert "<<ccr:" not in tool_msg["content"]
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_anthropic_execute_wrapper_multiline_retrieve_not_recompressed(self):
        """Anthropic-shape orchestrator call: the tool_use input is a decoded
        dict carrying a real newline, with no JSON text to scan."""
        content = _big_retrieve_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_exec_multiline",
                        "name": "execute",
                        "input": {"code": RETRIEVE_CODE_MULTILINE},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_exec_multiline",
                        "content": content,
                    }
                ],
            },
        ]
        result = router.apply(messages, tokenizer)

        tool_result_block = result.messages[1]["content"][0]
        assert tool_result_block["content"] == content
        assert "<<ccr:" not in tool_result_block["content"]
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_execute_wrapper_multiline_without_retrieve_still_compressed(self):
        """Negative control: a multiline script that does not call the
        retrieval tool keeps the wrapper name and stays compressible."""
        content = _big_retrieve_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        code = "const r = await tools.bash.bash\n({command: 'ls'});\nreturn r;"
        messages = _orchestrator_messages("execute", code, content)
        result = router.apply(messages, tokenizer)

        tool_msg = next(m for m in result.messages if m.get("role") == "tool")
        assert "router:excluded:ccr_retrieve" not in result.transforms_applied
        assert tool_msg["content"] != content or result.tokens_after < result.tokens_before

    def test_execute_wrapper_without_retrieve_call_still_compressed(self):
        """The exemption stays narrow: an execute call that does not invoke the
        retrieval tool has ordinary, compressible output."""
        content = _big_retrieve_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = _orchestrator_messages("execute", NON_RETRIEVE_CODE, content)
        result = router.apply(messages, tokenizer)

        tool_msg = next(m for m in result.messages if m.get("role") == "tool")
        assert "router:excluded:ccr_retrieve" not in result.transforms_applied
        assert tool_msg["content"] != content or result.tokens_after < result.tokens_before

    def test_hermes_bridge_over_execute_retrieve_result_not_recompressed(self):
        """A Hermes `tool_call` over an `execute` running the retrieval tool:
        the paired output must keep the exemption (nested wrapper, #3915
        review)."""
        content = _big_retrieve_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_ccr_hermes",
                        "type": "function",
                        "function": {
                            "name": "tool_call",
                            "arguments": json.dumps(
                                {"name": "execute", "arguments": {"code": RETRIEVE_CODE}}
                            ),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_ccr_hermes", "content": content},
        ]
        result = router.apply(messages, tokenizer)

        tool_msg = next(m for m in result.messages if m.get("role") == "tool")
        assert tool_msg["content"] == content, (
            "nested Hermes/execute retrieval result was recompressed "
            "(unresolvable retrieval loop, #3563)"
        )
        assert "<<ccr:" not in tool_msg["content"]
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_execute_wrapper_optional_call_retrieve_not_recompressed(self):
        """Optional-chaining invocation (`retrieve?.(...)`) keeps the exemption
        through the router (#3915 review)."""
        content = _big_retrieve_json()
        router = ContentRouter(ContentRouterConfig(min_section_tokens=10))
        tokenizer = _get_tokenizer()

        code = 'const r = await tools.headroom.headroom_retrieve?.({hash: "abc"});\nreturn r;'
        messages = _orchestrator_messages("execute", code, content)
        result = router.apply(messages, tokenizer)

        tool_msg = next(m for m in result.messages if m.get("role") == "tool")
        assert tool_msg["content"] == content
        assert "router:excluded:ccr_retrieve" in result.transforms_applied
