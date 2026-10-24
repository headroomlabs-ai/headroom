"""A selected external compressor that passes a block through must not swallow it.

``CompressOutput.compressed=False`` is the contract's passthrough signal: the
compressor declined this block and returned it unchanged. The router has to
treat that like any other fail-open case and run its built-in dispatch, exactly
as if no external compressor were selected. Before the fix the router adopted
the unchanged block as the external result (chain ``["external:<name>"]``), so
every declined block skipped Headroom's own compressors.
"""

from __future__ import annotations

import pytest

from headroom.cache.compression_store import reset_compression_store
from headroom.transforms.compressor_registry import (
    CompressInput,
    CompressorDescriptor,
    CompressOutput,
)
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
)

# Routes to SMART_CRUSHER (application/json) and is not touched by the STAGE-0
# lossless fold, so it reaches the external dispatch branch.
_JSON_ARRAY = (
    "["
    + ",".join(
        f'{{"id":{i},"name":"item-{i}","status":"active","value":{i * 7},'
        f'"note":"a fairly long descriptive field number {i} to add bulk"}}'
        for i in range(40)
    )
    + "]"
)


@pytest.fixture
def _memory_ccr(monkeypatch):
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "python")
    reset_compression_store()
    yield
    reset_compression_store()


class _DecliningExternal:
    """External compressor that always declines with ``compressed=False``."""

    def __init__(self) -> None:
        self.calls: list[CompressInput] = []

    @property
    def descriptor(self) -> CompressorDescriptor:
        return CompressorDescriptor(
            name="ext_decline",
            content_types=["application/json"],
            lossless=False,
            cost_tier="ml",
            recoverable=True,
        )

    def compress(self, inp: CompressInput) -> CompressOutput:
        self.calls.append(inp)
        return CompressOutput(
            content=inp.content,
            tokens_before=len(inp.content.split()),
            tokens_after=len(inp.content.split()),
            lossless=True,
            compressed=False,
        )


def _router(selection: list[str] | None, comp: _DecliningExternal | None = None) -> ContentRouter:
    router = ContentRouter(
        ContentRouterConfig(enable_kompress=False, active_external_compressors=selection)
    )
    if comp is not None:
        router.compressor_registry.register(comp, replace=True)
        router._active_external_compressors = router._resolve_active_external_compressors()
    return router


def test_passthrough_falls_back_to_builtin_dispatch(_memory_ccr):
    comp = _DecliningExternal()
    with_external = _router(["ext_decline"], comp)
    without_external = _router(None)

    got = with_external._apply_strategy_to_content(
        _JSON_ARRAY, CompressionStrategy.SMART_CRUSHER, ""
    )
    expected = without_external._apply_strategy_to_content(
        _JSON_ARRAY, CompressionStrategy.SMART_CRUSHER, ""
    )

    assert comp.calls, "the external compressor should have been consulted"
    assert "external:ext_decline" not in got[2]
    # Byte-identical to the path with no external compressor selected.
    assert got == expected


def test_passthrough_does_not_short_circuit_compress(_memory_ccr):
    comp = _DecliningExternal()
    router = _router(["ext_decline"], comp)

    result = router.compress(_JSON_ARRAY)

    assert comp.calls
    assert "external:ext_decline" not in result.strategy_chain
