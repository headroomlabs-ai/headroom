"""Unavailable HTML extraction must preserve tool output."""

from types import SimpleNamespace

import pytest

from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
)


def test_html_extract_none_falls_through_to_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    router = ContentRouter(ContentRouterConfig(relevance_split=False, lossless_then_lossy=False))
    monkeypatch.setattr(
        router,
        "_get_html_extractor",
        lambda: SimpleNamespace(extract=lambda content: SimpleNamespace(extracted=None)),
    )
    content = "<html><body><script>no extractable article body</script></body></html>"
    out, _tokens, _chain, _reversible = router._apply_strategy_to_content(
        content, CompressionStrategy.HTML, ""
    )
    assert out == content
