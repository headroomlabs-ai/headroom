"""Lossless JSON precedence and real no-saving fallback boundaries."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    _estimate_tokens,
)


@pytest.mark.parametrize("pretty", [True, False], ids=["minification", "compact-fallback"])
def test_json_minification_precedes_fallback_without_losing_records(pretty: bool) -> None:
    records = [
        {
            "id": i,
            "status": "ok",
            "level": "ERROR" if i == 20 else "INFO",
            "message": f"CRITICAL_{i} transaction completed",
        }
        for i in range(60)
    ]
    content = (
        json.dumps(records, indent=2) if pretty else json.dumps(records, separators=(",", ":"))
    )
    router = ContentRouter(
        ContentRouterConfig(
            smart_crusher_with_compaction=False,
            smart_crusher_max_items_after_crush=1000,
            enable_kompress=False,
            ccr_inject_marker=False,
        )
    )
    compressed, tokens, chain, _reversible = router._apply_strategy_to_content(
        content, CompressionStrategy.SMART_CRUSHER, "CRITICAL_20"
    )

    assert json.loads(compressed) == records
    if pretty:
        assert chain == [CompressionStrategy.JSON_MINIFIER.value]
        assert tokens < _estimate_tokens(content)
    else:
        assert compressed == content


def test_truncated_json_is_not_a_compression_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """JSON-shaped content that does not parse has nothing to minify.

    The detector routes truncated tool output to SmartCrusher by shape, and
    SmartCrusher returns it unchanged. ``minify_json`` then rejects it, which
    surfaced as a WARNING ("Compression with smart_crusher failed: invalid
    input for json_minifier: EOF while parsing ...") although nothing failed:
    the zero-savings Kompress fallback still runs. It must stay quiet, and
    the fallback must keep running.
    """
    records = [{"id": i, "status": "ok", "message": f"transaction {i}"} for i in range(60)]
    pretty = json.dumps(records, indent=2)
    truncated = pretty[: len(pretty) // 2]
    router = ContentRouter(ContentRouterConfig(enable_kompress=False, ccr_inject_marker=False))
    kompress_calls: list[str] = []

    def fake_kompress(content: str, context: str, question: Any) -> tuple[str, int]:
        kompress_calls.append(content)
        return content[: len(content) // 2], 0

    router._try_ml_compressor = fake_kompress  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING, logger="headroom.transforms.content_router"):
        compressed, _tokens, chain, _reversible = router._apply_strategy_to_content(
            truncated, CompressionStrategy.SMART_CRUSHER, ""
        )

    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert chain == [CompressionStrategy.SMART_CRUSHER.value, CompressionStrategy.KOMPRESS.value]
    assert kompress_calls == [truncated]
    assert compressed == truncated[: len(truncated) // 2]
