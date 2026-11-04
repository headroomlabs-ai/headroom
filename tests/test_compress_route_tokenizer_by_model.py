"""`/v1/compress` must count tokens for ANY wire shape and ANY model family.

The route does no format conversion — callers send whichever shape they already
use. Pinning one provider's token counter for the whole route silently reported
zero savings for Anthropic-shaped payloads: ``OpenAITokenCounter.count_message``
walks list content for ``text`` and ``image_url`` only, and has no else branch,
so an Anthropic ``tool_result`` block contributed literally nothing. A real
request that removed 235 characters reported ``tokens_saved: 0``.

The derived pipelines are built with ``provider=None`` so ``TransformPipeline``
resolves the tokenizer from the per-model registry instead. Every registry
tokenizer derives from ``BaseTokenizer``, whose ``_count_content_parts`` ends in
a serialize-and-count catch-all, so no block type counts as zero and there is no
per-provider block-type list to keep in sync.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app

# Big enough that any real tokenizer must report hundreds of tokens, and
# compressible so the router actually folds it (repeated grep-shaped lines).
_GREP = "\n".join(
    f"src/module_{i}.py:{i * 7}:    result = compute_value(item_{i}, flag=True)" for i in range(60)
)


@pytest.fixture
def client():
    app = create_app(
        ProxyConfig(
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )
    # /v1/compress is loopback-gated (#1227).
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as c:
        yield c


def _anthropic_messages() -> list[dict]:
    """Anthropic native shape: tool_use / tool_result content blocks."""
    return [
        {"role": "user", "content": [{"type": "text", "text": "find compute_value"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "c1", "name": "grep", "input": {"pattern": "compute"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c1", "content": _GREP}],
        },
    ]


def _openai_messages() -> list[dict]:
    """OpenAI native shape: tool_calls + role=tool."""
    return [
        {"role": "user", "content": "find compute_value"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "grep", "arguments": '{"pattern":"compute"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": _GREP},
    ]


# Model names spanning every routing path a gateway realistically sends, including
# a custom alias that matches no known vendor pattern.
_MODELS = [
    "claude-sonnet-4-6",
    "bedrock/anthropic.claude-3-5-sonnet",
    "vertex_ai/claude-sonnet-4@20250514",
    "gemini-2.5-pro",
    "deepseek/deepseek-v4",
    "moonshotai/kimi-k2",
    "my-gateway/big-model",
    "gpt-4o",
]


@pytest.mark.parametrize("model", _MODELS)
def test_anthropic_shape_is_counted_for_every_model_family(client, model):
    """No model name may produce a zero token count for Anthropic content blocks."""
    response = client.post("/v1/compress", json={"messages": _anthropic_messages(), "model": model})

    assert response.status_code == 200, response.text
    body = response.json()
    # The payload is ~4 KB of text. Any honest tokenizer reports hundreds; the
    # pinned OpenAI counter reported 28 for this exact request.
    assert body["tokens_before"] > 500, f"{model} undercounted: {body['tokens_before']}"
    assert body["tokens_saved"] > 0, f"{model} reported no savings: {body}"
    assert body["compression_ratio"] < 1.0


@pytest.mark.parametrize("model", _MODELS)
def test_openai_shape_still_counted_for_every_model_family(client, model):
    """The OpenAI-shaped path must not regress while fixing the Anthropic one."""
    response = client.post("/v1/compress", json={"messages": _openai_messages(), "model": model})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tokens_before"] > 500, f"{model} undercounted: {body['tokens_before']}"
    assert body["tokens_saved"] > 0, f"{model} reported no savings: {body}"


@pytest.mark.parametrize("mode", [None, "ccr", "lossy_inline", "lossless_then_lossy"])
def test_every_mode_counts_anthropic_shape(client, mode):
    """mode="ccr" used to share the request pipeline, which pinned the OpenAI counter."""
    payload: dict = {"messages": _anthropic_messages(), "model": "claude-sonnet-4-6"}
    if mode is not None:
        payload["config"] = {"mode": mode}

    response = client.post("/v1/compress", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tokens_before"] > 500, f"mode={mode} undercounted: {body['tokens_before']}"
    assert body["tokens_saved"] > 0, f"mode={mode} reported no savings: {body}"


def test_response_preserves_the_anthropic_wire_shape(client):
    """Passthrough contract: no format conversion in either direction."""
    response = client.post(
        "/v1/compress",
        json={"messages": _anthropic_messages(), "model": "claude-sonnet-4-6"},
    )

    assert response.status_code == 200
    block = response.json()["messages"][2]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "c1"
    # Content was folded, not dropped or restructured.
    assert 0 < len(block["content"]) < len(_GREP)


def test_tokenizer_choice_does_not_move_the_context_limit(client):
    """Regression guard: the two resolutions must stay independent.

    `model_limit` feeds context_pressure -> min_ratio, so letting a tokenizer
    decision pick the limit table changes compression aggressiveness. gpt-4-32k
    answered by the Anthropic table is 8,192 instead of 32,768 — a 4x
    under-estimate — even though its payload needs a non-OpenAI tokenizer.
    """
    seen: dict = {}
    proxy = client.app.state.proxy
    original = proxy._no_ccr_pipeline().apply

    def spy(**kwargs):
        seen.update(kwargs)
        return original(**kwargs)

    proxy._no_ccr_pipeline().apply = spy
    try:
        response = client.post(
            "/v1/compress",
            json={"messages": _anthropic_messages(), "model": "gpt-4-32k"},
        )
    finally:
        proxy._no_ccr_pipeline().apply = original

    assert response.status_code == 200
    # OpenAI's table, because the MODEL is an OpenAI model — regardless of the
    # Anthropic-shaped body that drives tokenizer selection.
    assert seen["model_limit"] == 32_768


# ── The documented multi-turn recipe ──────────────────────────────────────────
# The endpoint is stateless: unlike the proxy's own request path it runs no
# CacheAligner and tracks no provider cache state, so keeping the prefix stable is
# the caller's job. docs/content/docs/proxy.mdx documents the loop; these two tests
# pin both halves of it so the guidance cannot rot.


def _turn(i: int) -> list[dict]:
    body = "\n".join(f"src/mod_{i}_{j}.py:{j}: match compute_value(x{j})" for j in range(40))
    return [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": f"c{i}", "name": "grep", "input": {"p": "x"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": f"c{i}", "content": body}],
        },
    ]


def _compress(client, messages: list[dict], frozen: int | None = None) -> list[dict]:
    body: dict = {"messages": messages, "model": "claude-sonnet-4-6"}
    if frozen is not None:
        body["config"] = {"frozen_message_count": frozen}
    response = client.post("/v1/compress", json=body)
    assert response.status_code == 200, response.text
    return response.json()["messages"]


def test_system_and_tools_are_accepted_and_ignored(client):
    """Documented contract: only messages/model/token_budget/config are read.

    Anthropic sends `system` and `tools` out of band. The endpoint takes them
    without complaint and returns neither, so neither is compressed — callers must
    keep carrying them. Pinned because the silence is the hazard: a caller has no
    signal that the fields did nothing. If this ever starts returning them, the
    contract changed and docs/content/docs/proxy.mdx needs updating with it.
    """
    response = client.post(
        "/v1/compress",
        json={
            "messages": _anthropic_messages(),
            "model": "claude-sonnet-4-6",
            "system": "You are a coding agent. " * 200,
            "tools": [
                {
                    "name": "read",
                    "description": "Read a file from disk. " * 20,
                    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "system" not in body
    assert "tools" not in body
    # Messages are still compressed normally alongside the ignored fields.
    assert body["tokens_saved"] > 0


def test_documented_loop_keeps_the_cached_prefix_byte_identical(client):
    """Feed back previous OUTPUT + frozen_message_count -> stable prefix every turn."""
    import json

    forwarded = _compress(client, [{"role": "user", "content": [{"type": "text", "text": "go"}]}])
    for i in range(1, 6):
        previous = forwarded
        forwarded = _compress(client, previous + _turn(i), frozen=len(previous))
        replayed = forwarded[: len(previous)]
        assert [json.dumps(m, sort_keys=True) for m in replayed] == [
            json.dumps(m, sort_keys=True) for m in previous
        ], f"turn {i} rewrote the cached prefix"


def test_frozen_prefix_replays_what_you_sent_not_what_you_forwarded(client):
    """Why re-sending pristine originals busts the cache — the documented trap.

    `frozen_message_count` returns the leading messages *exactly as passed in*. So
    the bytes you get back depend entirely on which version you sent: feed it your
    previous OUTPUT and the prefix matches what the provider cached; feed it the
    pristine ORIGINALS and you hand the provider different bytes for a message it
    already cached, paying for compression and a cache miss at once.
    """
    import json

    base = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    originals = base + _turn(1)

    forwarded = _compress(client, originals)
    # Precondition: compression actually changed the prefix, so the two candidate
    # inputs for next turn genuinely differ.
    assert json.dumps(forwarded) != json.dumps(originals)

    # Correct: previous output in, same bytes back.
    good = _compress(client, forwarded + _turn(2), frozen=len(forwarded))
    assert good[: len(forwarded)] == forwarded

    # The trap: pristine originals in, pristine originals back — which is NOT what
    # was forwarded last turn, so the provider's cached prefix no longer matches.
    trap = _compress(client, originals + _turn(2), frozen=len(originals))
    assert trap[: len(originals)] == originals
    assert trap[: len(forwarded)] != forwarded
