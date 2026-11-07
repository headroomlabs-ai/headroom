"""Regressions for defects found while reviewing the Copilot bridge.

Each test below corresponds to a failure reproduced either against the live
GitHub Copilot API or against the installed litellm transform. They are grouped
by the mechanism that produces them rather than by module, because the bugs are
about how two libraries meet rather than about either one alone.
"""

from __future__ import annotations

from typing import Any

import pytest

from headroom.proxy.handlers.openai import (
    _chat_completion_json_to_responses_json,
    _responses_body_to_chat_completion_body,
    normalize_copilot_model_id,
    resolve_copilot_model_id,
)

COPILOT = "https://api.githubcopilot.com"


# ---------------------------------------------------------------------------
# Upstream gating -- the blocking review finding on the original PR
# ---------------------------------------------------------------------------


def test_casual_label_is_corrected_for_copilot_upstreams() -> None:
    assert (
        resolve_copilot_model_id("Claude Opus 4.8", upstream_base_url=COPILOT) == "claude-opus-4.8"
    )


@pytest.mark.parametrize(
    "upstream",
    [
        "https://api.openai.com",
        "https://my-gateway.internal/v1",
        "http://127.0.0.1:11434",
    ],
)
def test_non_copilot_upstreams_are_never_rewritten(upstream: str) -> None:
    """The Copilot alias table must not leak onto other upstreams.

    A custom gateway may legitimately accept ``"Sol"`` or ``"Claude Opus 4.8"``
    as its own model name; silently rewriting those to Copilot ids would send
    the operator's traffic to a model they did not ask for.
    """
    for label in ("Sol", "Claude Opus 4.8", "GPT Sol", "Gemini 3.1 Pro (Preview)"):
        assert resolve_copilot_model_id(label, upstream_base_url=upstream) == label


def test_missing_upstream_is_treated_as_not_copilot() -> None:
    assert resolve_copilot_model_id("Sol", upstream_base_url=None) == "Sol"


def test_ambiguous_names_still_pass_through_untouched() -> None:
    assert resolve_copilot_model_id("Gemini", upstream_base_url=COPILOT) == "Gemini"


def test_static_normalizer_is_unchanged_for_canonical_ids() -> None:
    assert normalize_copilot_model_id("claude-opus-4.8") == "claude-opus-4.8"


# ---------------------------------------------------------------------------
# Body bridge: field-name and type defects
# ---------------------------------------------------------------------------


def test_bridge_emits_max_completion_tokens_not_max_tokens() -> None:
    """``gpt-5.4`` rejects ``max_tokens`` outright.

    Live: ``400 Unsupported parameter: 'max_tokens' is not supported with this
    model. Use 'max_completion_tokens' instead.`` A sweep of the ten models this
    bridge can target found ``max_completion_tokens`` accepted by all of them,
    so it is strictly the safer field. litellm's transform emits the older name.
    """
    body = _responses_body_to_chat_completion_body(
        "gpt-5.4",
        {"input": "hi", "max_output_tokens": 4096},
    )
    assert "max_tokens" not in body
    assert body["max_completion_tokens"] == 4096


def test_bridge_never_emits_a_dict_reasoning_effort() -> None:
    """litellm returns the whole ``reasoning`` object when a summary is present.

    ``{"effort": "high", "summary": "detailed"}`` on a field whose wire type is
    a string enum. Harmless only while the key is dropped unconditionally; this
    guards the moment retention becomes conditional.
    """
    body = _responses_body_to_chat_completion_body(
        "claude-opus-4.8",
        {"input": "hi", "reasoning": {"effort": "high", "summary": "detailed"}},
    )
    assert not isinstance(body.get("reasoning_effort"), dict)


def test_bridge_forces_non_streaming_upstream() -> None:
    body = _responses_body_to_chat_completion_body("claude-opus-4.8", {"input": "hi"})
    assert body["stream"] is False


# ---------------------------------------------------------------------------
# Body bridge: usage accounting
# ---------------------------------------------------------------------------


def _chat_reply(usage: Any = ...) -> dict[str, Any]:
    reply: dict[str, Any] = {
        "id": "chatcmpl-1",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
        ],
    }
    if usage is not ...:
        reply["usage"] = usage
    return reply


@pytest.mark.parametrize("usage", [..., None])
def test_absent_upstream_usage_does_not_become_zero(usage: Any) -> None:
    """Fabricated zeros silently zero out cost and savings accounting.

    litellm invents ``{"input_tokens": 0, "output_tokens": 0,
    "total_tokens": 0}`` when the reply has no usage. The proxy reads those with
    a fallback that only fires on a *missing or unparseable* value -- a real
    ``0`` parses fine and wins, so the request records zero tokens into metrics,
    the cost tracker, the savings ledger and the request log. Dropping the
    fabricated block restores "upstream said nothing", which the fallback
    already handles correctly.
    """
    translated = _chat_completion_json_to_responses_json(
        responses_api_request={"input": "hi"},
        chat_completion_json=_chat_reply(usage),
    )
    assert "usage" not in translated or any(
        translated["usage"].get(k) for k in ("input_tokens", "output_tokens", "total_tokens")
    )


def test_real_upstream_usage_is_preserved() -> None:
    translated = _chat_completion_json_to_responses_json(
        responses_api_request={"input": "hi"},
        chat_completion_json=_chat_reply(
            {"prompt_tokens": 120, "completion_tokens": 7, "total_tokens": 127}
        ),
    )
    assert translated["usage"]["input_tokens"] == 120
    assert translated["usage"]["output_tokens"] == 7


def test_reply_still_translates_to_responses_shape() -> None:
    translated = _chat_completion_json_to_responses_json(
        responses_api_request={"input": "hi"},
        chat_completion_json=_chat_reply(
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        ),
    )
    assert translated.get("object") == "response"
    assert isinstance(translated.get("output"), list)
