"""Information-preserving compaction for EXCLUDED tool output.

Excluded tools (Read/Grep/Glob/Write/Edit) are protected from *lossy*
compression for accuracy. This feature still compacts them by detected shape,
using only reversible / data-preserving transforms:

* SEARCH (grep)  -> ripgrep --heading fold   [byte-lossless]
* LOG            -> ANSI strip + run-collapse [byte-lossless modulo ANSI color]
* JSON           -> whitespace-minify         [data-lossless; same object, NOT byte-exact]

Source code and glob path-lists match nothing -> untouched. Always on
(information-preserving, so it needs no feature gate) in every path.
"""

from __future__ import annotations

import json

import pytest

from headroom.providers import OpenAIProvider
from headroom.tokenizer import Tokenizer
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
from headroom.transforms.lossless_compaction import expand_runs, search_unheading, strip_ansi
from headroom.transforms.lossless_provider import (
    get_lossless_provider,
    set_lossless_provider,
)

GREP = "".join(
    f"src/module_{f}.py:{ln * 3}:matched occurrence with some real content here\n"
    for f in range(6)
    for ln in range(15)
)
LOG = "".join(
    f"\x1b[32m2026-07-03 INFO worker {i % 3} processing job batch\x1b[0m\n" for i in range(40)
)
LOG += "".join("2026-07-03 WARN transient retry, backing off\n" for _ in range(25))
JSON = json.dumps(
    {"users": [{"id": i, "name": f"user{i}", "active": i % 2 == 0} for i in range(40)]},
    indent=2,
)
CODE = "def foo(x):\n    return x + 1\n\nclass Bar:\n    value = 42\n" * 30
GLOB = "\n".join(f"src/module_{i}.py" for i in range(60)) + "\n"


@pytest.fixture
def tokenizer():
    provider = OpenAIProvider()
    return Tokenizer(provider.get_token_counter("gpt-4o"), "gpt-4o")


def _compact(content: str):
    router = ContentRouter(ContentRouterConfig())
    return router._lossless_compact_excluded(content)


# --- helper: right transform per shape, right guarantee ---


def test_grep_search_fold_is_byte_lossless():
    out, kind = _compact(GREP)
    assert kind == "search"
    assert len(out) < len(GREP)
    assert search_unheading(out) == GREP  # byte-exact


def test_log_compaction_recovers_modulo_ansi():
    out, kind = _compact(LOG)
    assert kind == "log"
    assert len(out) < len(LOG)
    assert expand_runs(out) == strip_ansi(LOG)  # recover the lines (ANSI dropped)


def test_json_minify_is_data_lossless():
    out, kind = _compact(JSON)
    assert kind == "json"
    assert len(out) < len(JSON)
    assert json.loads(out) == json.loads(JSON)  # same object; NOT byte-exact


def test_source_and_glob_untouched():
    assert _compact(CODE) is None
    assert _compact(GLOB) is None


# --- end-to-end through the router pipeline (excluded tools) ---


def _run(content: str, tool: str, tokenizer):
    router = ContentRouter(ContentRouterConfig())
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": tool, "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": content},
    ]
    result = router.apply(messages, tokenizer, compress_user_messages=True)
    return result.messages[1]["content"], result.transforms_applied


def test_pipeline_folds_grep_and_recovers(tokenizer):
    out, transforms = _run(GREP, "grep", tokenizer)
    assert "router:excluded:lossless_search" in transforms
    assert search_unheading(out) == GREP


def test_pipeline_compacts_log_read(tokenizer):
    out, transforms = _run(LOG, "read", tokenizer)
    assert "router:excluded:lossless_log" in transforms
    assert expand_runs(out) == strip_ansi(LOG)


def test_pipeline_minifies_json_read(tokenizer):
    out, transforms = _run(JSON, "read", tokenizer)
    assert "router:excluded:lossless_json" in transforms
    assert json.loads(out) == json.loads(JSON)  # data-lossless (same object)


def test_pipeline_leaves_source_read_untouched(tokenizer):
    out, _ = _run(CODE, "read", tokenizer)
    assert out == CODE


# --- pluggable lossless provider seam ---------------------------------------


@pytest.fixture(autouse=True)
def _reset_provider():
    """Never leak a registered provider between tests."""
    yield
    set_lossless_provider(None)


def test_default_no_provider_uses_builtin():
    # Unset (default) → built-in folds run; GREP compacts via search-heading.
    assert get_lossless_provider() is None
    out, kind = _compact(GREP)
    assert kind == "search" and search_unheading(out) == GREP


def test_registered_provider_is_authoritative():
    # A registered provider fully owns excluded-tool compaction; the built-in
    # search fold does NOT run (we'd get "search", not our sentinel).
    set_lossless_provider(lambda content: ("<<folded>>", "custom"))
    assert _compact(GREP) == ("<<folded>>", "custom")
    # Authoritative on None too: provider says "leave it" → no built-in fallback.
    set_lossless_provider(lambda content: None)
    assert _compact(GREP) is None


def test_provider_exception_falls_back_to_builtin():
    def boom(content):
        raise RuntimeError("provider blew up")

    set_lossless_provider(boom)
    # Falls back to the built-in fold rather than crashing or passing through raw.
    out, kind = _compact(GREP)
    assert kind == "search" and search_unheading(out) == GREP
