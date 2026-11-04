"""``HeadroomConfig.smart_crusher`` must reach the default pipeline's ContentRouter.

``TransformPipeline`` built its ContentRouter from the router's own defaults,
so the documented knobs (``config.smart_crusher.max_items_after_crush``,
``config.smart_crusher.enabled = False``) were silently ignored by
``HeadroomClient`` and every integration that builds a pipeline from a
``HeadroomConfig``.
"""

from __future__ import annotations

import json
from typing import Any

from headroom.config import HeadroomConfig
from headroom.transforms.cache_aligner import CacheAligner
from headroom.transforms.content_router import ContentRouter
from headroom.transforms.pipeline import TransformPipeline

# A JSON string array takes SmartCrusher's row-capping path even with
# lossless compaction on (arrays of objects usually compact losslessly and
# keep every row, so the cap never shows).
_LOG_LINES = [f"GET /api/items/{i} 200 {10 + i % 50}ms" for i in range(300)]

_ROWS = [
    {"id": i, "service": f"svc-{i % 7}", "status": "ok", "latency_ms": 100 + (i * 37) % 400}
    for i in range(300)
]


def _messages(tool_content: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "Show the access log."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_access_log", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": tool_content},
        {"role": "user", "content": "Summarize."},
    ]


def _compressed_tool_content(pipeline: TransformPipeline, tool_content: str) -> str:
    result = pipeline.apply(_messages(tool_content), model="gpt-4o", model_limit=128_000)
    return next(m for m in result.messages if m.get("role") == "tool")["content"]


def _lines_kept(content: str) -> int:
    return sum(f'"{line}"' in content for line in _LOG_LINES)


def test_max_items_after_crush_reaches_compression() -> None:
    """The troubleshooting recipe: keep more items. It kept 16 regardless."""
    default = _compressed_tool_content(
        TransformPipeline(config=HeadroomConfig()), json.dumps(_LOG_LINES)
    )

    config = HeadroomConfig()
    config.smart_crusher.max_items_after_crush = 50
    keep_more = _compressed_tool_content(TransformPipeline(config=config), json.dumps(_LOG_LINES))

    assert _lines_kept(default) <= 20
    assert _lines_kept(keep_more) >= 50


def test_disabled_smart_crusher_leaves_json_arrays_alone() -> None:
    """The troubleshooting recipe: disable SmartCrusher. It still crushed."""
    config = HeadroomConfig()
    config.smart_crusher.enabled = False
    content = json.dumps(_LOG_LINES)

    assert _compressed_tool_content(TransformPipeline(config=config), content) == content


def test_default_config_compresses_exactly_as_before() -> None:
    """Callers that never touch ``smart_crusher`` see no change.

    The oracle is the pre-fix pipeline: a ContentRouter built from its own
    defaults, behind the same CacheAligner.
    """
    config = HeadroomConfig()
    pre_fix = TransformPipeline(
        config=config,
        transforms=[CacheAligner(config.cache_aligner), ContentRouter()],
    )
    for content in (json.dumps(_LOG_LINES), json.dumps(_ROWS)):
        assert _compressed_tool_content(
            TransformPipeline(config=config), content
        ) == _compressed_tool_content(pre_fix, content)


def test_router_does_not_share_the_callers_config_object() -> None:
    config = HeadroomConfig()
    router = next(
        t for t in TransformPipeline(config=config).transforms if isinstance(t, ContentRouter)
    )

    assert router.config.smart_crusher == config.smart_crusher
    assert router.config.smart_crusher is not config.smart_crusher
