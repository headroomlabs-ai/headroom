"""Live probe: what does Anthropic's thinking-block ``signature`` actually cover?

This test is the empirical foundation for the #3124 relaxation. That change lets
Headroom forward its compression edits on a request that carries signed thinking
blocks, instead of discarding every edit (the #2254 blanket lock, which cost
~34% of Claude Code requests all of their savings). It is only correct if the
signature seals *the thinking block*, not the surrounding request.

Nothing in Anthropic's public docs states the scope, so it is pinned here by
experiment. Each test mutates exactly one part of a replayed turn that holds a
real signed thinking block and asserts the request is still accepted.

``test_forged_signature_is_rejected`` is the **negative control** and the most
important test in the file: without it, a wall of passing tests would be equally
consistent with "Anthropic never validates signatures on this shape", which
would make every other assertion here vacuous.

Opt-in: requires a real key and is gated behind ``pytest.mark.live``.
Run with ``pytest -m live tests/test_thinking_signature_scope_live.py``.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

import pytest

from tests._dotenv import autouse_apply_env, load_env_overrides

_env = load_env_overrides()
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY") or _env.get("ANTHROPIC_API_KEY", "")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set"),
]

apply_dotenv = autouse_apply_env(_env)

MODEL = os.environ.get("HEADROOM_LIVE_THINKING_MODEL", "claude-sonnet-4-6")

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location.",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    }
]
SYSTEM = "You are a helpful assistant. Use tools when they are relevant."
# Deliberately requires reasoning: on adaptive-thinking models (Claude 5) a
# trivial prompt makes the model skip thinking entirely and the probe has
# nothing to test.
PROMPT = (
    "I have 3 meetings in San Francisco tomorrow starting at 9:00am, 1:00pm and "
    "4:30pm. Each runs 90 minutes and I need 25 minutes of travel between "
    "consecutive meetings. Reason carefully about whether that schedule has any "
    "conflicts, then call get_weather for San Francisco so I know what to wear."
)


def _think_cfg(model: str) -> dict[str, Any]:
    """Claude 5 replaced ``budget_tokens`` thinking with adaptive + effort."""
    if model in ("claude-opus-5", "claude-sonnet-5", "claude-fable-5"):
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
    return {"thinking": {"type": "enabled", "budget_tokens": 2000}}


def _post(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    import httpx

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        json=body,
        headers={
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        timeout=120.0,
    )
    return resp.status_code, resp.json()


@pytest.fixture(scope="module")
def signed_turn() -> dict[str, Any]:
    """Obtain one genuine signed thinking block, and the replay body around it."""
    think = _think_cfg(MODEL)
    status, resp = _post(
        {
            "model": MODEL,
            "max_tokens": 3000,
            "system": SYSTEM,
            "tools": TOOLS,
            **think,
            "messages": [{"role": "user", "content": PROMPT}],
        }
    )
    if status != 200:
        pytest.skip(f"could not obtain a thinking turn ({status}): {json.dumps(resp)[:200]}")

    content = resp["content"]
    thinking_idx = next(
        (i for i, b in enumerate(content) if b["type"] in ("thinking", "redacted_thinking")),
        None,
    )
    if thinking_idx is None:
        pytest.skip(f"{MODEL} returned no thinking block for the probe prompt")
    tool_idx = next((i for i, b in enumerate(content) if b["type"] == "tool_use"), None)
    if tool_idx is None:
        pytest.skip(f"{MODEL} did not call the tool; the replay shape needs a tool_use")

    followup = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": content[tool_idx]["id"],
                    "content": "62F, foggy, wind 12mph. Forecast: fog clearing by noon.",
                }
            ],
        }
    ]
    return {
        "thinking_idx": thinking_idx,
        "tool_idx": tool_idx,
        "signature": content[thinking_idx].get("signature", ""),
        "body": {
            "model": MODEL,
            "max_tokens": 3000,
            "system": SYSTEM,
            "tools": TOOLS,
            **think,
            "messages": [
                {"role": "user", "content": PROMPT},
                {"role": "assistant", "content": content},
            ]
            + followup,
        },
    }


def _assistant(body: dict[str, Any]) -> list[dict[str, Any]]:
    return body["messages"][1]["content"]


def _expect_accepted(body: dict[str, Any], what: str) -> None:
    status, resp = _post(body)
    assert status == 200, (
        f"Anthropic rejected a request after {what}, so the thinking signature "
        f"covers more than the block itself and the #3124 relaxation is unsafe "
        f"for this mutation. Response: {json.dumps(resp)[:300]}"
    )


def test_exact_replay_is_accepted(signed_turn):
    """Control: the unmodified replay must work, or every other test is noise."""
    _expect_accepted(copy.deepcopy(signed_turn["body"]), "no modification at all")


def test_compressing_a_tool_result_is_accepted(signed_turn):
    """The mutation Headroom actually makes on Claude Code traffic."""
    body = copy.deepcopy(signed_turn["body"])
    block = body["messages"][2]["content"][0]
    body["messages"][2]["content"][0] = {**block, "content": "62F foggy"}
    _expect_accepted(body, "compressing a tool_result in a later user message")


def test_modifying_a_sibling_block_in_the_thinking_message_is_accepted(signed_turn):
    """The gap the fingerprint cannot close by inspection.

    ``thinking_blocks_survived_mutation`` proves the thinking blocks are
    byte-identical, but says nothing about their siblings in the same assistant
    message. If the seal covered the whole assistant turn, a compressed sibling
    would break it and the fingerprint would wave it through.
    """
    body = copy.deepcopy(signed_turn["body"])
    blocks = _assistant(body)
    tool_idx = signed_turn["tool_idx"]
    blocks[tool_idx] = {**blocks[tool_idx], "input": {"location": "San Francisco, CA"}}
    text_idx = next((i for i, b in enumerate(blocks) if b["type"] == "text"), None)
    if text_idx is not None:
        blocks[text_idx] = {**blocks[text_idx], "text": "compressed sibling text"}
    _expect_accepted(body, "modifying sibling blocks inside the thinking message")


def test_modifying_top_level_system_and_tools_is_accepted(signed_turn):
    """Tool-schema compaction and tool-search deferral edit these fields."""
    body = copy.deepcopy(signed_turn["body"])
    body["system"] = "Assistant. Use tools."
    body["tools"] = copy.deepcopy(TOOLS)
    body["tools"][0]["description"] = "Weather."
    _expect_accepted(body, "rewriting top-level system and tool descriptions")


def test_canonical_reserialization_is_accepted(signed_turn):
    """#2254 blamed a plain re-encode for the 400s. It is not the cause."""
    body = copy.deepcopy(signed_turn["body"])
    body["messages"][1] = json.loads(
        json.dumps({"content": _assistant(body), "role": "assistant"}, sort_keys=True)
    )
    _expect_accepted(body, "re-serializing the body with reordered keys")


def test_forged_signature_is_rejected(signed_turn):
    """NEGATIVE CONTROL — the load-bearing test in this file.

    If a forged signature is *accepted*, Anthropic is not validating signatures
    on this request shape at all, and every acceptance above proves nothing.
    """
    body = copy.deepcopy(signed_turn["body"])
    idx = signed_turn["thinking_idx"]
    blocks = _assistant(body)
    blocks[idx] = {**blocks[idx], "signature": "A" * len(signed_turn["signature"])}
    status, resp = _post(body)
    assert status == 400, (
        "A forged thinking signature was ACCEPTED. Signature validation is not "
        "active on this shape, so the acceptances asserted by the other tests in "
        f"this module carry no information. Response: {json.dumps(resp)[:300]}"
    )
    assert "signature" in json.dumps(resp).lower()
