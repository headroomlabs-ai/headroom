"""Generalized cross-turn prefix canonicalizer (`_canonicalize_for_prefix_compare`).

The delta path decides "is this turn an append-only extension of the last?" by
comparing the canonicalized prefix. Clients attach non-semantic annotations that
vary turn-to-turn (cache_control moved to the newest block, litellm `caller`,
provider_specific_fields, AI-SDK providerMetadata, streaming `index`, string vs
block content). The canonicalizer must ignore all of those, while NEVER dropping a
semantic field (which would mask a real divergence -> stale replay).

Two messages canonicalize-equal IFF they are semantically identical. These tests
pin: (1) each noise field is ignored, across Anthropic/OpenAI/Bedrock shapes;
(2) semantic differences are still detected; (3) reasoning signatures are kept;
(4) opaque tool payloads (input/arguments/json) are compared verbatim so user data
containing keys like `state`/`index` is never corrupted.
"""

from headroom.cache.prefix_tracker import _canonicalize_for_prefix_compare as C


def eq(a, b):
    return C(a) == C(b)


# ── noise is ignored (equal despite it) ───────────────────────────────────────
def test_cache_control_ignored_anthropic():
    a = {
        "role": "user",
        "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
    }
    b = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    assert eq(a, b)


def test_cachepoint_and_caller_ignored():
    a = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "t1",
                "name": "bash",
                "input": {"cmd": "ls"},
                "caller": {"type": "direct"},
            },
            {"cachePoint": {"type": "default"}},
        ],
    }
    b = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
            {"cachePoint": {"type": "default"}},
        ],
    }
    assert eq(a, b)


def test_litellm_and_aisdk_noise_ignored():
    a = {
        "role": "assistant",
        "content": "ok",
        "provider_specific_fields": {"x": 1},
        "reasoning_content": "...",
        "annotations": [{"u": "url"}],
        "system_fingerprint": "fp_1",
        "service_tier": "default",
    }
    b = {
        "role": "assistant",
        "content": "ok",
        "provider_specific_fields": {"x": 999},
        "system_fingerprint": "fp_2",
    }
    assert eq(a, b)


def test_streaming_index_and_state_ignored_at_block_level():
    a = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "t1",
                "name": "b",
                "input": {"c": 1},
                "index": 2,
                "state": "output-available",
                "providerMetadata": {"a": 1},
            }
        ],
    }
    b = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "t1", "name": "b", "input": {"c": 1}}],
    }
    assert eq(a, b)


def test_string_content_normalized_to_block():
    assert eq(
        {"role": "user", "content": "hello"},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    )


def test_tool_result_string_vs_block_equal():
    a = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "out"}],
    }
    b = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [{"type": "text", "text": "out"}],
            }
        ],
    }
    assert eq(a, b)


# ── semantic differences ARE detected (not masked) ────────────────────────────
def test_different_text_detected():
    assert not eq({"role": "user", "content": "A"}, {"role": "user", "content": "B"})


def test_different_tool_input_detected():
    a = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "t1", "name": "b", "input": {"cmd": "ls"}}],
    }
    b = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "t1", "name": "b", "input": {"cmd": "rm -rf /"}}],
    }
    assert not eq(a, b)


def test_different_role_detected():
    assert not eq({"role": "user", "content": "x"}, {"role": "assistant", "content": "x"})


def test_reasoning_signature_preserved_and_compared():
    # Same thinking text, DIFFERENT signature -> genuinely different (must not equate).
    a = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "", "signature": "SIG_A"}],
    }
    b = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "", "signature": "SIG_B"}],
    }
    assert not eq(a, b)
    # Same signature but cache_control noise differs -> equal.
    c = {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "",
                "signature": "SIG_A",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }
    assert eq(a, c)


def test_thinking_present_absent_flip_detected():
    # The litellm/opencode persistence bug: a thinking block dropped on a later turn
    # is a REAL divergence and must fail the compare (raw fallback, never stale replay).
    a = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "S"},
            {"type": "text", "text": "ok"},
        ],
    }
    b = {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}
    assert not eq(a, b)


# ── the opaque-payload safety trap: noise-named keys inside user data ──────────
def test_opaque_input_with_colliding_keys_not_corrupted():
    # `state`/`index` are noise keys at BLOCK level, but here they are legitimate
    # tool-input DATA. They must be compared verbatim, so different inputs differ.
    a = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "t1", "name": "set", "input": {"state": "CA", "index": 3}}
        ],
    }
    b = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "t1", "name": "set", "input": {"state": "NY", "index": 3}}
        ],
    }
    assert not eq(a, b), "tool input with keys named like noise must NOT be stripped/equated"


def test_opaque_arguments_string_verbatim():
    a = {
        "role": "assistant",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"index": 1}'}}
        ],
        "content": None,
    }
    b = {
        "role": "assistant",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"index": 2}'}}
        ],
        "content": None,
    }
    assert not eq(a, b)


def test_bedrock_toolresult_json_payload_verbatim():
    a = {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "t1",
                    "content": [{"json": {"state": "ok", "n": 1}}],
                    "status": "success",
                }
            }
        ],
    }
    b = {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "t1",
                    "content": [{"json": {"state": "ok", "n": 2}}],
                    "status": "success",
                }
            }
        ],
    }
    assert not eq(a, b)


def test_bedrock_cachepoint_and_reasoning_signature():
    # cachePoint (noise) ignored; reasoningText.signature (semantic) compared.
    a = {
        "role": "assistant",
        "content": [
            {"reasoningContent": {"reasoningText": {"text": "r", "signature": "BSIG"}}},
            {"cachePoint": {"type": "default"}},
        ],
    }
    b = {
        "role": "assistant",
        "content": [{"reasoningContent": {"reasoningText": {"text": "r", "signature": "BSIG"}}}],
    }
    assert eq(a, b)
    c = {
        "role": "assistant",
        "content": [
            {"reasoningContent": {"reasoningText": {"text": "r", "signature": "DIFFERENT"}}}
        ],
    }
    assert not eq(a, c)
