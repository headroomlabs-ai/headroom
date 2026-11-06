"""Small shell results must stay verbatim through the lossy router path.

Issue #3590: ``headroom wrap claude`` with a low ``HEADROOM_MIN_TOKENS``
compressed a short ``ls -la`` result into a numbers-only blob. The output
still looked structured, so the model treated it as data instead of
retrying. File reads and searches are already protected; shell output is
deliberately compressible, but a small result is not worth the accuracy
cost. Large build/test logs must keep going through the lossy path.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from headroom.config import TransformResult
from headroom.providers import OpenAIProvider
from headroom.tokenizer import Tokenizer
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    ContentType,
    RouterCompressionResult,
    RoutingDecision,
    _is_small_shell_result,
    _shell_result_min_chars,
)


def _tokenizer() -> Tokenizer:
    provider = OpenAIProvider()
    return Tokenizer(provider.get_token_counter("gpt-4o"), "gpt-4o")


def _lossy_result(original: str) -> RouterCompressionResult:
    """Stand-in for the ML compressor, including its CCR recovery marker."""
    return RouterCompressionResult(
        compressed="128 448 09:26 1760 09:28 11980 16767 2394 1262\n<<ccr:deadbeef>>",
        original=original,
        strategy_used=CompressionStrategy.KOMPRESS,
        strategy_chain=["kompress"],
        routing_log=[
            RoutingDecision(
                ContentType.PLAIN_TEXT,
                CompressionStrategy.KOMPRESS,
                original_tokens=400,
                compressed_tokens=12,
            )
        ],
    )


def _fake_compress(content: str, **kwargs: object) -> RouterCompressionResult:
    return _lossy_result(content)


def _conversation(
    tool_name: str, tool_output: str, *, list_form: bool = False
) -> list[dict[str, object]]:
    """A conversation old enough that the tool result is compressible."""
    tool_content: object = [{"type": "text", "text": tool_output}] if list_form else tool_output
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "run the command"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
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
                    "tool_use_id": "tool_1",
                    "content": tool_content,
                }
            ],
        },
    ]
    for i in range(10):
        messages.append({"role": "assistant", "content": f"step {i}"})
        messages.append({"role": "user", "content": f"continue {i}"})
    return messages


def _route(tool_name: str, tool_output: str, *, list_form: bool = False) -> TransformResult:
    router = ContentRouter(ContentRouterConfig())
    router.compress = _fake_compress  # type: ignore[assignment]
    return router.apply(
        _conversation(tool_name, tool_output, list_form=list_form),
        _tokenizer(),
        target_ratio=0.4,
    )


def _openai_conversation(tool_name: str, tool_output: str) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "run the command"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": tool_output},
    ]
    for i in range(10):
        messages.append({"role": "assistant", "content": f"step {i}"})
        messages.append({"role": "user", "content": f"continue {i}"})
    return messages


def _route_openai(tool_name: str, tool_output: str) -> TransformResult:
    router = ContentRouter(ContentRouterConfig())
    router.compress = _fake_compress  # type: ignore[assignment]
    return router.apply(
        _openai_conversation(tool_name, tool_output), _tokenizer(), target_ratio=0.4
    )


def _tail(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    for i in range(10):
        messages.append({"role": "assistant", "content": f"step {i}"})
        messages.append({"role": "user", "content": f"continue {i}"})
    return messages


def _function_conversation(tool_name: str, tool_output: str) -> list[dict[str, object]]:
    """Legacy OpenAI shape: role:function with the tool name on the message."""
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "run the command"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "function", "name": tool_name, "content": tool_output},
    ]
    return _tail(messages)


def _route_function(tool_name: str, tool_output: str) -> TransformResult:
    router = ContentRouter(ContentRouterConfig())
    router.compress = _fake_compress  # type: ignore[assignment]
    return router.apply(
        _function_conversation(tool_name, tool_output), _tokenizer(), target_ratio=0.4
    )


def _text_block_conversation(tool_name: str, tool_output: str) -> list[dict[str, object]]:
    """OpenAI shape: a tool message whose content is a list of text blocks."""
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "run the command"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "text", "text": tool_output}],
        },
    ]
    return _tail(messages)


def _route_text_block(tool_name: str, tool_output: str) -> TransformResult:
    router = ContentRouter(ContentRouterConfig())
    router.compress = _fake_compress  # type: ignore[assignment]
    return router.apply(
        _text_block_conversation(tool_name, tool_output), _tokenizer(), target_ratio=0.4
    )


def _ls_output(rows: int = 18) -> str:
    files = "\n".join(
        f"-rw-r--r--  1 user  staff  {1000 + i:>6} Oct 22 09:{i:02d} report_{i:02d}_final_v2.md"
        for i in range(rows)
    )
    return (
        "total 128\n"
        "drwxr-xr-x  12 user  staff    384 Oct 22 09:26 .\n"
        "drwxr-xr-x   5 user  staff    160 Oct 22 09:25 ..\n"
        f"{files}\n"
    )


def _padded_output(length: int) -> str:
    """A shell-like payload of exactly ``length`` characters."""
    base = "line: some structured shell output\n"
    return (base * (length // len(base) + 1))[:length]


def test_small_bash_tool_result_stays_verbatim() -> None:
    """A short shell result keeps its structure instead of becoming numbers."""
    output = _ls_output()
    assert len(output) < 2000

    result = _route("Bash", output)

    assert result.messages[3]["content"][0]["content"] == output
    assert "router:shell_small_result_verbatim" in result.transforms_applied


def test_large_bash_tool_result_is_still_compressed() -> None:
    """The guard must not disable compression for large build/test logs."""
    output = _ls_output(rows=80)
    assert len(output) > 2000

    result = _route("Bash", output)

    assert result.messages[3]["content"][0]["content"] != output
    assert "router:tool_result:kompress" in result.transforms_applied


def test_small_non_shell_tool_result_is_still_compressed() -> None:
    """The guard is scoped to shell tools, not every small tool result."""
    output = "a short reference payload\n" * 30

    result = _route("SomeOtherTool", output)

    assert result.messages[3]["content"][0]["content"] != output
    assert "router:tool_result:kompress" in result.transforms_applied


def test_small_openai_tool_result_stays_verbatim() -> None:
    """The OpenAI role:tool shape carries the same guard as Anthropic blocks."""
    output = _ls_output()

    result = _route_openai("Bash", output)

    assert result.messages[3]["content"] == output
    assert "router:shell_small_result_verbatim" in result.transforms_applied


def test_shell_result_min_chars_can_disable_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """The escape hatch restores the previous behaviour for operators."""
    monkeypatch.setenv("HEADROOM_SHELL_RESULT_MIN_CHARS", "0")
    output = _ls_output()

    result = _route("Bash", output)

    assert result.messages[3]["content"][0]["content"] != output
    assert "router:tool_result:kompress" in result.transforms_applied


def test_invalid_shell_result_min_chars_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-numeric override keeps the documented default instead of crashing."""
    monkeypatch.setenv("HEADROOM_SHELL_RESULT_MIN_CHARS", "junk")

    assert _shell_result_min_chars() == 2000


def test_zero_shell_result_min_chars_is_legitimate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero is a real value ("no floor"), not an invalid one."""
    monkeypatch.setenv("HEADROOM_SHELL_RESULT_MIN_CHARS", "0")

    assert _shell_result_min_chars() == 0


def test_negative_shell_result_min_chars_disables_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_SHELL_RESULT_MIN_CHARS", "-1")
    assert _shell_result_min_chars() == 0
    result = _route("Bash", _ls_output())
    assert "router:shell_small_result_verbatim" not in result.transforms_applied


@pytest.mark.parametrize("text", [None, 123, ""])
def test_empty_or_non_string_values_are_not_small_shell_results(text: object) -> None:
    assert not _is_small_shell_result("Bash", text, frozenset({"bash"}))


def test_shell_result_floor_boundary_is_exclusive() -> None:
    """The floor is "shorter than", so exactly 2000 chars stays compressible."""
    at_floor = _padded_output(2000)
    below_floor = _padded_output(1999)

    at_result = _route("Bash", at_floor)
    below_result = _route("Bash", below_floor)

    assert at_result.messages[3]["content"][0]["content"] != at_floor
    assert below_result.messages[3]["content"][0]["content"] == below_floor
    assert "router:shell_small_result_verbatim" in below_result.transforms_applied


@pytest.mark.parametrize(
    "tool_name",
    ["Bash", "shell", "local_shell", "run_shell_command", "execute_command"],
)
def test_recognized_shell_aliases_stay_verbatim(tool_name: str) -> None:
    """Harness aliases from the shared tool-name map get the same protection."""
    output = _ls_output()

    result = _route(tool_name, output)

    assert result.messages[3]["content"][0]["content"] == output
    assert "router:shell_small_result_verbatim" in result.transforms_applied


def test_unrecognized_shell_like_name_is_still_compressed() -> None:
    """A similar-looking name outside the configured set is not protected."""
    output = _ls_output()

    result = _route("run_remote_shell", output)

    assert result.messages[3]["content"][0]["content"] != output
    assert "router:tool_result:kompress" in result.transforms_applied


def test_small_list_form_tool_result_stays_verbatim() -> None:
    """The Anthropic list-form content wrapper is flattened and protected."""
    output = _ls_output()

    result = _route("Bash", output, list_form=True)

    assert result.messages[3]["content"][0]["content"] == [{"type": "text", "text": output}]
    assert "router:shell_small_result_verbatim" in result.transforms_applied


def test_small_legacy_function_result_stays_verbatim() -> None:
    """Legacy role:function resolves the shell name from message["name"]."""
    output = _ls_output()

    result = _route_function("Bash", output)

    assert result.messages[3]["content"] == output
    assert "router:shell_small_result_verbatim" in result.transforms_applied


def test_small_top_level_text_block_stays_verbatim() -> None:
    """A tool message normalized to top-level text blocks gets the guard."""
    output = _ls_output()

    result = _route_text_block("Bash", output)

    assert result.messages[3]["content"] == [{"type": "text", "text": output}]
    assert "router:shell_small_result_verbatim" in result.transforms_applied


@pytest.mark.parametrize("role", ["tool", "function"])
@pytest.mark.parametrize("length", [1999, 2000, 2400])
def test_split_shell_result_uses_total_text_length(role: str, length: int) -> None:
    """The floor applies to the whole result, not each provider text fragment."""
    messages = _text_block_conversation("Bash", "")
    if role == "function":
        messages[3] = {"role": "function", "name": "Bash"}
    blocks = [
        {"type": "text", "text": _padded_output(length // 2)},
        {"type": "text", "text": _padded_output(length - length // 2)},
    ]
    messages[3]["content"] = blocks
    router = ContentRouter(ContentRouterConfig())
    router.compress = _fake_compress  # type: ignore[assignment]

    result = router.apply(messages, _tokenizer(), target_ratio=0.4)

    assert (result.messages[3]["content"] == blocks) is (length < 2000)
    assert ("router:shell_small_result_verbatim" in result.transforms_applied) is (length < 2000)


@pytest.mark.parametrize("route", [_route_openai, _route_function, _route_text_block])
def test_shell_floor_override_applies_to_wire_shapes(
    monkeypatch: pytest.MonkeyPatch, route: Callable[[str, str], TransformResult]
) -> None:
    """A positive override protects below the floor and compresses at it."""
    monkeypatch.setenv("HEADROOM_SHELL_RESULT_MIN_CHARS", "1000")
    below = route("Bash", _padded_output(999))
    at = route("Bash", _padded_output(1000))
    assert "router:shell_small_result_verbatim" in below.transforms_applied
    assert "router:shell_small_result_verbatim" not in at.transforms_applied


def test_custom_shell_names_are_normalized_and_protected() -> None:
    """An operator's custom mixed-case shell name gets the same floor."""
    router = ContentRouter(ContentRouterConfig(bash_tool_names=frozenset({"MyShell"})))
    router.compress = _fake_compress  # type: ignore[assignment]
    output = _ls_output()
    result = router.apply(_openai_conversation("mYsHeLl", output), _tokenizer(), target_ratio=0.4)
    assert result.messages[3]["content"] == output
    assert "router:shell_small_result_verbatim" in result.transforms_applied


@pytest.mark.parametrize("route", [_route_openai, _route_function, _route_text_block])
def test_non_shell_results_remain_compressible_in_all_wire_shapes(
    route: Callable[[str, str], TransformResult],
) -> None:
    output = _ls_output()
    result = route("SomeOtherTool", output)
    assert "router:shell_small_result_verbatim" not in result.transforms_applied
    assert result.messages[3]["content"] not in (output, [{"type": "text", "text": output}])


@pytest.mark.parametrize("route", [_route_openai, _route_function, _route_text_block])
def test_shell_floor_can_be_disabled_in_all_wire_shapes(
    monkeypatch: pytest.MonkeyPatch, route: Callable[[str, str], TransformResult]
) -> None:
    monkeypatch.setenv("HEADROOM_SHELL_RESULT_MIN_CHARS", "0")
    output = _ls_output()
    result = route("Bash", output)
    assert "router:shell_small_result_verbatim" not in result.transforms_applied
    assert result.messages[3]["content"] not in (output, [{"type": "text", "text": output}])


@pytest.mark.parametrize("conversation", [_openai_conversation, _function_conversation])
def test_small_shell_result_diagnostics_explain_why_it_was_preserved(
    conversation: Callable[[str, str], list[dict[str, object]]],
) -> None:
    output = _ls_output()
    router = ContentRouter(ContentRouterConfig())
    router.compress = _fake_compress  # type: ignore[assignment]
    result = router.apply(
        conversation("Bash", output), _tokenizer(), target_ratio=0.4, collect_diagnostics=True
    )
    decision = next(item for item in result.message_decisions if item.message_index == 3)
    assert decision.action == "protected:shell_small_result"
    assert decision.tokens_before == decision.tokens_after
    assert result.messages[3]["content"] == output


@pytest.mark.parametrize("shape", ["anthropic", "tool", "function"])
def test_small_shell_blocks_without_optional_route_counters(shape: str) -> None:
    """Direct block callers can omit counters without losing the shell guard."""
    output = _ls_output()
    if shape == "anthropic":
        message = _conversation("Bash", output)[3]
    elif shape == "function":
        message = {
            "role": "function",
            "name": "Bash",
            "content": [{"type": "text", "text": output}],
        }
    else:
        message = _text_block_conversation("Bash", output)[3]
    transforms: list[str] = []
    router = ContentRouter(ContentRouterConfig())
    router.compress = _fake_compress  # type: ignore[assignment]

    result = router._process_content_blocks(
        message=message,
        content_blocks=message["content"],
        context="",
        transforms_applied=transforms,
        excluded_tool_ids=set(),
        ccr_retrieve_tool_ids=set(),
        tool_name_map={"tool_1": "Bash", "call_1": "Bash"},
        route_counts=None,
        messages_from_end=20,
    )

    assert result is message
    assert transforms == ["router:shell_small_result_verbatim"]
