"""Determinism regression test for the compression pipeline.

Prefix caching at Anthropic/OpenAI is byte-exact: turn N+2's cache hit
requires the bytes for turn-N-and-earlier tool results to be identical
across requests. That holds iff every compressor in the pipeline is
deterministic — same input bytes in, same output bytes out, with no
dependence on wall clock, RNG, or process-local state.

This test pins that invariant against a small fixture of representative
tool-output shapes. If any compressor sneaks in non-determinism (e.g. a
timestamp, a uuid, an iteration-order dependency), this test fails
before the change ships and silently busts cache hit rates in
production.
"""

from __future__ import annotations

import json

from headroom.transforms.compression_units import (
    CompressionUnit,
    compress_unit_with_router,
)
from headroom.transforms.content_router import (
    ContentRouter,
    ContentRouterConfig,
)


class _WhitespaceTokenizer:
    """Stand-in tokenizer — matches the production token-counter protocol
    used by `compress_unit_with_router`. Deterministic by construction;
    real tokenizers (tiktoken, anthropic) are also deterministic for the
    same input + model."""

    def count_text(self, text: str) -> int:
        return len(text.split())


_FIXTURES: dict[str, str] = {
    "git_diff_wrapped": (
        "Chunk ID: 904f13\n"
        "Wall time: 0.0000 seconds\n"
        "Process exited with code 0\n"
        "Original token count: 1996\n"
        "Output:\n"
        "headroom/proxy/handlers/openai.py | 12 ++++++++++++\n"
        " 1 file changed, 12 insertions(+)\n\n"
        "--- Changes ---\n\n"
        "diff --git a/headroom/proxy/handlers/openai.py b/headroom/proxy/handlers/openai.py\n"
        "@@ -10,6 +10,18 @@\n"
        " def handle():\n"
        "+    # twelve lines of added context\n" * 6 + "     return None\n"
    ),
    "jsonl_log_lines": "\n".join(
        json.dumps(
            {
                "ts": f"2026-05-10T14:13:{seconds:02d}",
                "level": "INFO",
                "event": "codex_compression_units",
                "request_id": f"hr_1778447324_{seconds:06d}",
                "model": "gpt-5.5",
                "tokens_before": 1234 + seconds,
                "tokens_after": 567 + seconds,
            },
            separators=(",", ":"),
        )
        for seconds in range(30)
    ),
    "search_results_grep": "\n".join(
        f"src/foo/bar/{n:03d}.py:{n * 7}:    def function_{n}(self, arg):" for n in range(40)
    ),
    "plain_long_text": " ".join(["headroom"] * 400),
}


def _compress(content: str, *, router: ContentRouter) -> str:
    """Run one canonical compression round-trip through the unit layer.

    Uses a fresh router so this exercises the full detection +
    strategy-selection path each call (no result_cache priming from a
    prior call leaking the answer)."""

    unit = CompressionUnit(
        text=content,
        provider="openai",
        endpoint="responses",
        role="tool",
        item_type="function_call_output",
        cache_zone="live",
        mutable=True,
        min_bytes=64,
    )
    result = compress_unit_with_router(
        unit,
        router=router,
        tokenizer=_WhitespaceTokenizer(),
    )
    return result.compressed


def test_compression_pipeline_is_byte_deterministic() -> None:
    """Two independent runs of every fixture must produce identical
    bytes. Fresh `ContentRouter` instances avoid the in-process result
    cache short-circuiting the second call — we want the *compression*
    to be deterministic, not just memoized."""

    for name, content in _FIXTURES.items():
        router_a = ContentRouter(ContentRouterConfig())
        router_b = ContentRouter(ContentRouterConfig())

        first = _compress(content, router=router_a)
        second = _compress(content, router=router_b)

        assert first == second, (
            f"Non-deterministic compression for fixture {name!r}: "
            f"len(first)={len(first)} len(second)={len(second)}"
        )


def test_compression_result_cache_returns_identical_bytes() -> None:
    """Within one router, two calls on the same content must return
    identical bytes. Catches a result-cache that stores partial state
    or re-runs the compressor with different seeds on cache miss vs
    cache hit."""

    for name, content in _FIXTURES.items():
        router = ContentRouter(ContentRouterConfig())

        first = _compress(content, router=router)
        second = _compress(content, router=router)

        assert first == second, (
            f"Result-cache returned different bytes for fixture {name!r}: first_hash≠second_hash"
        )


def test_protected_roles_pass_through_unchanged() -> None:
    """Companion guarantee to determinism: protected roles never see
    any compressor at all, regardless of size. If this regresses, the
    prefix-cache invariant for user/system/assistant content is gone."""

    router = ContentRouter(ContentRouterConfig())
    payload = _FIXTURES["plain_long_text"]

    for role in ("user", "system", "developer", "assistant"):
        result = compress_unit_with_router(
            CompressionUnit(
                text=payload,
                provider="openai",
                endpoint="responses",
                role=role,
                item_type="message",
                min_bytes=64,
            ),
            router=router,
            tokenizer=_WhitespaceTokenizer(),
        )
        assert result.modified is False, f"role={role!r} was modified"
        assert result.compressed == payload, f"role={role!r} bytes changed"
