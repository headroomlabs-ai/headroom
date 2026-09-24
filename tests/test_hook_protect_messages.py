"""``CompressionHooks.protect_messages`` — the hard per-message veto.

``compute_biases`` is a soft multiplier, and several strategies clamp it
against their own floors or ignore it outright: measured against the log and
JSON routes, bias 0.1 and bias 1000 produce byte-identical output. A hook that
has concluded a specific message must survive verbatim therefore cannot say so
with a bias, and this seam is how it says so instead.
"""

from __future__ import annotations

import json

import pytest

from headroom import compress
from headroom.hooks import CompressionHooks
from headroom.tokenizers import get_tokenizer
from headroom.transforms.content_router import ContentRouter

MODEL = "gpt-4o"
LOG = "\n".join(
    f"2026-09-2{i % 9} 10:{i % 60:02d}:00 INFO worker={i} processed batch {i} in {i * 3}ms"
    for i in range(300)
)
BLOB = json.dumps([{"id": i, "name": f"item-{i}", "size": i * 7} for i in range(200)])


def openai_messages(body: str) -> list[dict]:
    return [
        {"role": "user", "content": "look"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "run_terminal_cmd", "arguments": '{"command": "cat x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": body},
        {"role": "user", "content": "now summarise"},
    ]


def anthropic_messages(body: str) -> list[dict]:
    return [
        {"role": "user", "content": "look"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "cat x"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": body}],
        },
        {"role": "user", "content": "now summarise"},
    ]


def _text_at(messages: list[dict], index: int) -> str:
    content = messages[index]["content"]
    if isinstance(content, str):
        return content
    return "".join(
        block.get("content", "") if isinstance(block.get("content"), str) else ""
        for block in content
    )


@pytest.mark.parametrize(
    "build", [openai_messages, anthropic_messages], ids=["openai", "anthropic"]
)
@pytest.mark.parametrize("body", [LOG, BLOB], ids=["log", "json"])
def test_a_protected_message_survives_verbatim(build, body) -> None:
    """Both content shapes, and routes that ignore bias entirely."""
    tokenizer = get_tokenizer(MODEL)

    compressed = ContentRouter().apply(build(body), tokenizer)
    assert _text_at(compressed.messages, 2) != body, "fixture must actually be compressible"

    protected = ContentRouter().apply(build(body), tokenizer, protect={2})
    assert _text_at(protected.messages, 2) == body
    assert "router:protected:hook" in protected.transforms_applied


@pytest.mark.parametrize(
    "protect", [None, set(), {99}, frozenset()], ids=["none", "empty", "miss", "frozen"]
)
def test_the_seam_is_inert_unless_used(protect) -> None:
    """The default path must be byte-identical to not passing the kwarg."""
    tokenizer = get_tokenizer(MODEL)
    baseline = ContentRouter().apply(openai_messages(LOG), tokenizer)
    result = ContentRouter().apply(openai_messages(LOG), tokenizer, protect=protect)
    assert _text_at(result.messages, 2) == _text_at(baseline.messages, 2)
    assert "router:protected:hook" not in result.transforms_applied


def test_protection_beats_every_bias() -> None:
    """The point of the seam: no bias can do this."""
    tokenizer = get_tokenizer(MODEL)
    for bias in (0.1, 1.0, 8.0, 1000.0):
        biased = ContentRouter().apply(openai_messages(LOG), tokenizer, biases={2: bias})
        assert _text_at(biased.messages, 2) != LOG
    protected = ContentRouter().apply(openai_messages(LOG), tokenizer, protect={2})
    assert _text_at(protected.messages, 2) == LOG


def test_the_default_hook_protects_nothing() -> None:
    assert CompressionHooks().protect_messages(openai_messages(LOG), None) == set()


def test_the_seam_is_reachable_from_the_sdk() -> None:
    """Wiring check: a hook installed through ``compress(hooks=...)`` reaches
    the router. Without this the seam exists and nobody can use it."""

    class Protecting(CompressionHooks):
        def protect_messages(self, messages, ctx):
            return {2}

    plain = compress(openai_messages(LOG), model=MODEL, hooks=CompressionHooks())
    protecting = compress(openai_messages(LOG), model=MODEL, hooks=Protecting())

    assert _text_at(plain.messages, 2) != LOG
    assert _text_at(protecting.messages, 2) == LOG
    assert protecting.tokens_after > plain.tokens_after


def test_hooks_that_do_not_implement_it_still_work() -> None:
    """Hooks predating this method are duck-typed subclasses in the wild."""

    class Legacy(CompressionHooks):
        def compute_biases(self, messages, ctx):
            return {2: 1.5}

    result = compress(openai_messages(LOG), model=MODEL, hooks=Legacy())
    assert result.tokens_after < result.tokens_before
