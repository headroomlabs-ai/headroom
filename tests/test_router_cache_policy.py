"""Request policy must scope every tier of the router's content cache."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    content_cache_key,
)

TEXT = "The service reports useful observations about the current deployment. " * 80


class Tokenizer:
    def count_text(self, text: str) -> int:
        return max(1, len(str(text)) // 4)


def messages(shape, tool_name="inspect", arguments="{}"):
    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": tool_name, "arguments": arguments},
            }
        ],
    }
    if shape == "tool_result":
        result = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": TEXT}],
        }
    else:
        content = TEXT if shape == "string" else [{"type": "text", "text": TEXT}]
        result = {"role": "tool", "tool_call_id": "call_1", "content": content}
    return [call, result]


def output(result, shape):
    content = result.messages[-1]["content"]
    if shape == "string":
        return content
    return content[0]["content" if shape == "tool_result" else "text"]


@pytest.fixture
def router(monkeypatch):
    # Only the compressor is replaced: apply(), admission, all cache tiers,
    # runtime policy binding and frozen verdict handling remain real.
    router = ContentRouter(ContentRouterConfig(ccr_inject_marker=False))
    monkeypatch.setattr(router, "_kompress_model_ready", lambda: True)
    return router


def install_compressor(monkeypatch, router, *, skip_first=False, marked=False):
    calls = []

    def compress(content, context="", bias=1.0, **kwargs):
        calls.append((router._runtime_kompress_model, context, bias))
        disabled = router._runtime_kompress_model == "disabled" or router._runtime_skip_kompress
        skip = disabled or (skip_first and len(calls) == 1)
        compressed = content if skip else f"compressed under policy {len(calls)}"
        if marked and not skip:
            compressed += " <<ccr:0123456789abcdef>>"
        return SimpleNamespace(
            original=content,
            compressed=compressed,
            compression_ratio=1.0 if skip else 0.1,
            strategy_used=CompressionStrategy.KOMPRESS,
            strategy_chain=[],
        )

    monkeypatch.setattr(router, "compress", compress)
    return calls


@pytest.mark.parametrize("shape", ["string", "tool_result", "text"])
@pytest.mark.parametrize("freeze", [False, True])
@pytest.mark.parametrize(
    "changed",
    [
        {"kompress_model": "disabled"},
        {"kompress_model": "another-model"},
        {"skip_kompress": True},
        {"force_kompress": True},
        {"context": "a different question"},
        {"target_ratio": 0.3},
    ],
)
def test_apply_reuses_results_only_under_the_same_policy(
    monkeypatch, router, shape, freeze, changed
):
    monkeypatch.setenv("HEADROOM_FREEZE_BLOCK_DECISION", str(int(freeze)))
    calls = install_compressor(monkeypatch, router, marked=freeze)
    payload = messages(shape)
    first = output(router.apply(payload, Tokenizer()), shape)
    assert first != TEXT
    assert output(router.apply(payload, Tokenizer()), shape) == first
    assert len(calls) == 1

    second = output(router.apply(payload, Tokenizer(), **changed), shape)
    assert len(calls) == 2, "changed policy must not reuse the previous result"
    assert second != first
    if changed.get("kompress_model") == "disabled" or changed.get("skip_kompress"):
        assert second == TEXT
    assert output(router.apply(payload, Tokenizer(), **changed), shape) == second
    assert len(calls) == 2
    # Returning to the original policy still finds its own cached output.
    assert output(router.apply(payload, Tokenizer()), shape) == first
    assert len(calls) == 2


@pytest.mark.parametrize("shape", ["string", "tool_result", "text"])
def test_frozen_verdicts_are_recorded_separately_for_each_policy(monkeypatch, router, shape):
    monkeypatch.setenv("HEADROOM_FREEZE_BLOCK_DECISION", "1")
    # Lossy output needs a retrieval marker before a verdict is safe to freeze.
    calls = install_compressor(monkeypatch, router, marked=True)
    payload = messages(shape)

    first = output(router.apply(payload, Tokenizer()), shape)
    initial_verdicts = dict(router._frozen_verdicts)
    assert len(initial_verdicts) == 1
    assert list(initial_verdicts.values()) == [True]
    assert output(router.apply(payload, Tokenizer()), shape) == first
    assert router._frozen_verdicts == initial_verdicts
    assert len(calls) == 1

    second = output(router.apply(payload, Tokenizer(), kompress_model="another-model"), shape)
    assert second != first
    assert len(calls) == 2
    assert len(router._frozen_verdicts) == 2
    assert all(verdict is True for verdict in router._frozen_verdicts.values())
    assert initial_verdicts.items() <= router._frozen_verdicts.items()
    assert output(router.apply(payload, Tokenizer()), shape) == first
    assert len(calls) == 2


@pytest.mark.parametrize("shape", ["string", "tool_result", "text"])
@pytest.mark.parametrize("changed", [{"kompress_model": "another-model"}, {"force_kompress": True}])
def test_a_skip_under_one_policy_does_not_suppress_another(monkeypatch, router, shape, changed):
    calls = install_compressor(monkeypatch, router, skip_first=True)
    payload = messages(shape)
    assert output(router.apply(payload, Tokenizer()), shape) == TEXT
    assert output(router.apply(payload, Tokenizer()), shape) == TEXT
    assert len(calls) == 1
    assert output(router.apply(payload, Tokenizer(), **changed), shape) != TEXT
    assert len(calls) == 2


def test_hook_bias_partitions_string_results(monkeypatch, router):
    calls = install_compressor(monkeypatch, router)
    payload = messages("string")
    first = output(router.apply(payload, Tokenizer()), "string")
    second = output(router.apply(payload, Tokenizer(), biases={1: 2.0}), "string")
    assert second != first
    assert [call[2] for call in calls] == [1.0, 2.0]


@pytest.mark.parametrize("shape", ["string", "tool_result"])
def test_effective_tool_bias_partitions_results(monkeypatch, router, shape):
    router.config.tool_profiles = {
        "inspect": SimpleNamespace(bias=1.0),
        "summarize": SimpleNamespace(bias=2.0),
    }
    calls = install_compressor(monkeypatch, router)
    first = output(router.apply(messages(shape), Tokenizer()), shape)
    second = output(router.apply(messages(shape, tool_name="summarize"), Tokenizer()), shape)
    assert second != first
    assert [call[2] for call in calls] == [1.0, 2.0]


def test_tool_result_relevance_uses_the_enriched_context(monkeypatch, router):
    router.config.relevance_split = True
    calls = install_compressor(monkeypatch, router)
    first = router.apply(messages("tool_result", arguments='{"query":"alpha"}'), Tokenizer())
    second = router.apply(messages("tool_result", arguments='{"query":"beta"}'), Tokenizer())
    assert output(first, "tool_result") != output(second, "tool_result")
    assert len(calls) == 2
    assert calls[0][1] != calls[1][1]


def test_policy_keys_are_deterministic_across_processes():
    options = {
        "kompress_model": "custom|model",
        "force_kompress": True,
        "skip_kompress": True,
        "bias": 1.5,
        "context": "question|with\x00delimiters",
    }
    expected = content_cache_key(TEXT, 0.5, gated=True, **options)
    code = (
        "from headroom.transforms.content_router import content_cache_key;"
        f"print(content_cache_key({TEXT!r}, 0.5, gated=True, **{options!r}))"
    )
    for seed in ("1", "2"):
        found = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        assert found == expected


def test_context_and_content_have_unambiguous_boundaries():
    assert content_cache_key("b|c", None, context="a") != content_cache_key(
        "c", None, context="a|b"
    )


@pytest.mark.parametrize("shape", ["string", "tool_result", "text"])
def test_reopened_disk_results_are_scoped_to_the_request_policy(tmp_path, monkeypatch, shape):
    monkeypatch.setenv("HEADROOM_COMPRESSION_CACHE_PATH", str(tmp_path / "policy.db"))
    config = ContentRouterConfig(ccr_inject_marker=False, result_cache_disk=True)
    first = ContentRouter(config)
    calls = install_compressor(monkeypatch, first)
    payload = messages(shape)
    original = output(first.apply(payload, Tokenizer()), shape)
    assert len(calls) == 1

    reopened = ContentRouter(config)
    new_calls = install_compressor(monkeypatch, reopened)
    assert output(reopened.apply(payload, Tokenizer()), shape) == original
    assert new_calls == []
    assert reopened._cache.stats["cache_disk_hits"] == 1
    assert output(reopened.apply(payload, Tokenizer(), kompress_model="disabled"), shape) == TEXT
    assert len(new_calls) == 1
