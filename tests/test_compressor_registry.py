"""Tests for the pluggable compressor registry and entry-point discovery."""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import asdict, dataclass

import pytest

from headroom.transforms import compressor_registry
from headroom.transforms.compressor_registry import (
    ENTRY_POINT_GROUP,
    CompressInput,
    CompressorDescriptor,
    CompressorRegistry,
    CompressOutput,
)


class FakeCompressor:
    """Minimal in-memory compressor implementing the Compressor protocol."""

    def __init__(
        self,
        name: str = "fake",
        *,
        content_types: list[str] | None = None,
        lossless: bool = True,
        cost_tier: str = "fast",
        recoverable: bool = True,
        raise_on_compress: bool = False,
    ) -> None:
        self._descriptor = CompressorDescriptor(
            name=name,
            content_types=content_types or ["text/plain"],
            lossless=lossless,
            cost_tier=cost_tier,
            recoverable=recoverable,
        )
        self._raise_on_compress = raise_on_compress

    @property
    def descriptor(self) -> CompressorDescriptor:
        return self._descriptor

    def compress(self, inp: CompressInput) -> CompressOutput:
        if self._raise_on_compress:
            raise RuntimeError("compress must not run during discovery/selection")
        half = inp.content[: max(1, len(inp.content) // 2)]
        return CompressOutput(
            content=half,
            tokens_before=len(inp.content),
            tokens_after=len(half),
            lossless=self._descriptor.lossless,
            markers=[f"fake:{inp.content_type}"],
            recoverable={"deadbeef": inp.content} if self._descriptor.recoverable else {},
            warnings=[],
        )


@dataclass
class FakeEntryPoint:
    """Stand-in for ``importlib.metadata.EntryPoint`` used in discovery tests."""

    name: str
    value: object

    def load(self) -> object:
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def test_registry_starts_empty() -> None:
    reg = CompressorRegistry()
    assert reg.names() == []
    assert reg.get("fake") is None
    assert reg.active(None) == []
    assert reg.active({"*"}) == []


def test_descriptor_round_trips() -> None:
    desc = CompressorDescriptor(
        name="fake",
        content_types=["text/plain", "text/markdown"],
        lossless=True,
        cost_tier="fast",
        recoverable=True,
    )
    assert desc.name == "fake"
    assert desc.content_types == ["text/plain", "text/markdown"]
    assert desc.lossless is True
    assert desc.cost_tier in compressor_registry.COST_TIERS
    assert desc.recoverable is True
    # Pure-data: fully representable as a plain dict for a cross-language boundary.
    assert asdict(desc) == {
        "name": "fake",
        "content_types": ["text/plain", "text/markdown"],
        "lossless": True,
        "cost_tier": "fast",
        "recoverable": True,
    }


def test_register_and_get_by_name() -> None:
    reg = CompressorRegistry()
    comp = FakeCompressor("fake")
    assert reg.register(comp) == "fake"
    assert reg.get("fake") is comp
    assert reg.names() == ["fake"]
    assert [d.name for d in reg.descriptors()] == ["fake"]


def test_register_rejects_empty_name_and_duplicates() -> None:
    reg = CompressorRegistry()
    with pytest.raises(ValueError):
        reg.register(FakeCompressor(""))

    reg.register(FakeCompressor("fake"))
    with pytest.raises(ValueError):
        reg.register(FakeCompressor("fake"))

    replacement = FakeCompressor("fake")
    assert reg.register(replacement, replace=True) == "fake"
    assert reg.get("fake") is replacement


def test_selection_is_opt_in_none_and_empty_select_nothing(caplog) -> None:
    reg = CompressorRegistry()
    reg.register(FakeCompressor("a"))
    reg.register(FakeCompressor("b"))

    with caplog.at_level(logging.INFO, logger=compressor_registry.log.name):
        assert reg.select(None) == set()
        assert reg.select(set()) == set()
    assert reg.active(None) == []
    # Opt-in default is surfaced.
    assert "none selected (opt-in)" in caplog.text


def test_selection_wildcard_selects_all() -> None:
    reg = CompressorRegistry()
    reg.register(FakeCompressor("a"))
    reg.register(FakeCompressor("b"))

    assert reg.select({"*"}) == {"a", "b"}
    active = reg.active({"*"})
    assert [c.descriptor.name for c in active] == ["a", "b"]


def test_selection_by_name_and_unknown_is_skipped(caplog) -> None:
    reg = CompressorRegistry()
    reg.register(FakeCompressor("a"))
    reg.register(FakeCompressor("b"))

    assert reg.select({"a"}) == {"a"}
    assert [c.descriptor.name for c in reg.active({"a", "b"})] == ["a", "b"]

    with caplog.at_level(logging.WARNING, logger=compressor_registry.log.name):
        assert reg.select({"a", "nope"}) == {"a"}
    assert "requested but not registered: nope" in caplog.text


def test_selection_normalizes_whitespace_and_empties() -> None:
    reg = CompressorRegistry()
    reg.register(FakeCompressor("a"))
    assert reg.select({" a ", "", "  "}) == {"a"}


def test_compress_round_trips_input_to_output() -> None:
    reg = CompressorRegistry()
    reg.register(FakeCompressor("fake"))
    comp = reg.active({"fake"})[0]

    inp = CompressInput(
        content="hello world, this is content",
        content_type="text/plain",
        query="summarize",
        config={"aggressive": True},
        budget={"target_ratio": 0.5, "time_ms": 10, "max_items": 3},
    )
    out = comp.compress(inp)

    assert isinstance(out, CompressOutput)
    assert out.tokens_before == len(inp.content)
    assert out.tokens_after == len(out.content)
    assert out.tokens_after < out.tokens_before
    assert out.lossless is True
    assert out.markers == ["fake:text/plain"]
    assert out.recoverable == {"deadbeef": inp.content}
    assert out.warnings == []


def test_discovery_registers_selects_but_does_not_run_compress(monkeypatch, caplog) -> None:
    # A compressor whose compress() would raise proves discovery never calls it.
    guarded = FakeCompressor("guarded", raise_on_compress=True)

    class NeedsInit:
        def __init__(self) -> None:
            raise RuntimeError("bad init")

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group=None: (
            [
                FakeEntryPoint("guarded-instance", guarded),
                FakeEntryPoint("working-class", FakeCompressor),
                FakeEntryPoint("bad-load", RuntimeError("bad load")),
                FakeEntryPoint("bad-init", NeedsInit),
            ]
            if group == ENTRY_POINT_GROUP
            else []
        ),
    )

    reg = CompressorRegistry()
    with caplog.at_level(logging.WARNING, logger=compressor_registry.log.name):
        discovered = reg.discover()

    # Only the loadable/constructible compressors are registered.
    assert sorted(discovered) == ["fake", "guarded"]
    assert reg.get("guarded") is guarded
    # Failures are logged and skipped, not raised.
    assert "bad-load" in caplog.text
    assert "bad-init" in caplog.text

    # Discovery did not run compress; selection alone does not either.
    active = reg.active({"guarded"})
    assert active == [guarded]
    # compress only runs when the caller invokes it.
    with pytest.raises(RuntimeError):
        active[0].compress(CompressInput(content="x", content_type="text/plain"))


def test_discovery_handles_enumeration_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group=None: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    reg = CompressorRegistry()
    assert reg.discover() == []
    assert reg.names() == []
