"""Lossless JSON precedence and real no-saving fallback boundaries."""

from __future__ import annotations

import json

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
