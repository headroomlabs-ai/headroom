"""Tests for compressor selection (swap-OUT built-ins) + registry inventory.

Covers the two behavior-safe halves of the router/registry integration:

  1. Selection surface: ``ProxyConfig.compressors`` mapped onto the router's
     ``enable_*`` flags at the proxy seam (``_apply_compressor_selection``).
     Default (``None``) must leave every flag at its dataclass default so the
     request path is byte-identical to today.
  2. Registry inventory: ``ContentRouter`` builds a ``CompressorRegistry`` with a
     metadata-only entry for every built-in plus opt-in entry-point discovery,
     without changing dispatch.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest

from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import (
    BUILTIN_COMPRESSOR_FLAGS,
    _apply_compressor_selection,
)
from headroom.transforms import compressor_registry as cr_module
from headroom.transforms.compressor_registry import (
    CompressInput,
    CompressorDescriptor,
    CompressOutput,
)
from headroom.transforms.content_router import (
    _BUILTIN_COMPRESSOR_DESCRIPTORS,
    ContentRouter,
    ContentRouterConfig,
    _build_compressor_registry,
)

# The nine recognized built-in selection names.
_BUILTIN_NAMES = {
    "smart_crusher",
    "kompress",
    "code_aware",
    "search",
    "log",
    "tabular",
    "config",
    "html",
    "image",
}


def _enable_flags(config: ContentRouterConfig) -> dict[str, bool]:
    return {flag: getattr(config, flag) for flag in BUILTIN_COMPRESSOR_FLAGS.values()}


# ─────────────────────────── selection mapping ───────────────────────────────


def test_flag_map_covers_the_nine_recognized_names() -> None:
    assert set(BUILTIN_COMPRESSOR_FLAGS) == _BUILTIN_NAMES


def test_flag_map_targets_real_config_fields() -> None:
    fields = {f.name for f in dataclasses.fields(ContentRouterConfig)}
    for flag in BUILTIN_COMPRESSOR_FLAGS.values():
        assert flag in fields, f"{flag} is not a ContentRouterConfig field"


def test_none_selection_is_a_noop_byte_identical_defaults() -> None:
    """Default (no --compressor) must leave every enable_* at its default."""
    baseline = _enable_flags(ContentRouterConfig())
    config = ContentRouterConfig()
    _apply_compressor_selection(config, None)
    assert _enable_flags(config) == baseline


def test_empty_selection_is_a_noop() -> None:
    baseline = _enable_flags(ContentRouterConfig())
    config = ContentRouterConfig()
    _apply_compressor_selection(config, set())
    assert _enable_flags(config) == baseline


def test_whitespace_only_selection_is_a_noop() -> None:
    baseline = _enable_flags(ContentRouterConfig())
    config = ContentRouterConfig()
    _apply_compressor_selection(config, {"", "  "})
    assert _enable_flags(config) == baseline


def test_single_selection_enables_only_that_builtin() -> None:
    config = ContentRouterConfig()
    _apply_compressor_selection(config, {"kompress"})
    flags = _enable_flags(config)
    assert flags["enable_kompress"] is True
    for name, flag in BUILTIN_COMPRESSOR_FLAGS.items():
        if name != "kompress":
            assert flags[flag] is False, f"{flag} should be disabled"


def test_multi_selection_enables_exactly_those() -> None:
    config = ContentRouterConfig()
    _apply_compressor_selection(config, {"smart_crusher", "log"})
    flags = _enable_flags(config)
    enabled = {name for name, flag in BUILTIN_COMPRESSOR_FLAGS.items() if flags[flag]}
    assert enabled == {"smart_crusher", "log"}


def test_selecting_code_aware_enables_it_even_though_it_defaults_off() -> None:
    # enable_code_aware defaults to False; an explicit selection turns it on.
    config = ContentRouterConfig()
    assert config.enable_code_aware is False
    _apply_compressor_selection(config, {"code_aware"})
    assert config.enable_code_aware is True


def test_wildcard_enables_all_builtins() -> None:
    config = ContentRouterConfig()
    _apply_compressor_selection(config, {"*"})
    assert all(_enable_flags(config).values())


def test_selection_strips_whitespace() -> None:
    config = ContentRouterConfig()
    _apply_compressor_selection(config, {" kompress ", " search"})
    flags = _enable_flags(config)
    enabled = {name for name, flag in BUILTIN_COMPRESSOR_FLAGS.items() if flags[flag]}
    assert enabled == {"kompress", "search"}


def test_only_external_name_disables_all_builtins() -> None:
    # Selecting only a non-builtin (an external/registry name) disables every
    # recognized built-in — the opt-in "exactly these" contract.
    config = ContentRouterConfig()
    _apply_compressor_selection(config, {"my_external_compressor"})
    assert not any(_enable_flags(config).values())


def test_unrecognized_names_are_ignored_but_recognized_still_apply() -> None:
    config = ContentRouterConfig()
    _apply_compressor_selection(config, {"kompress", "does_not_exist"})
    flags = _enable_flags(config)
    assert flags["enable_kompress"] is True
    # No crash / no spurious flag creation for the unknown name.


# ────────────────── unmatched-name warning (#2384, no behavior change) ────────


def test_only_unmatched_selection_warns_that_builtins_are_disabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A typo'd selection must be diagnosable from the startup log."""
    config = ContentRouterConfig()
    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        _apply_compressor_selection(config, {"smart_krusher"})
    # Behavior is unchanged — the opt-in "exactly these" contract still holds.
    assert not any(_enable_flags(config).values())
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "smart_krusher" in text
    assert "disabled" in text.lower()
    assert "smart_crusher" in text  # valid names listed for the typo case


def test_mixed_selection_warns_only_about_unmatched_names(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = ContentRouterConfig()
    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        _apply_compressor_selection(config, {"kompress", "does_not_exist"})
    assert _enable_flags(config)["enable_kompress"] is True
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "does_not_exist" in text
    assert "disabled" not in text.lower()  # built-ins were not all turned off


def test_matched_selection_emits_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    for selection in ({"kompress"}, {"*"}):
        config = ContentRouterConfig()
        with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
            _apply_compressor_selection(config, selection)
    assert not caplog.records


# ─────────────────────────── ProxyConfig field ───────────────────────────────


def test_proxyconfig_compressors_defaults_to_none() -> None:
    assert ProxyConfig().compressors is None


def test_proxyconfig_accepts_compressor_set() -> None:
    config = ProxyConfig(compressors={"kompress", "smart_crusher"})
    assert config.compressors == {"kompress", "smart_crusher"}


# ─────────────────────────── registry inventory ──────────────────────────────


def test_builtin_descriptors_cover_the_nine_names() -> None:
    names = {d.name for d in _BUILTIN_COMPRESSOR_DESCRIPTORS}
    assert names == _BUILTIN_NAMES


def test_builtin_descriptor_names_match_selection_flag_map() -> None:
    names = {d.name for d in _BUILTIN_COMPRESSOR_DESCRIPTORS}
    assert names == set(BUILTIN_COMPRESSOR_FLAGS)


def test_builtin_descriptors_have_valid_cost_tiers() -> None:
    for d in _BUILTIN_COMPRESSOR_DESCRIPTORS:
        assert d.cost_tier in cr_module.COST_TIERS


def test_build_registry_registers_all_builtins() -> None:
    registry = _build_compressor_registry()
    assert set(registry.names()) >= _BUILTIN_NAMES


def test_content_router_exposes_populated_registry() -> None:
    router = ContentRouter(ContentRouterConfig())
    assert set(router.compressor_registry.names()) >= _BUILTIN_NAMES


def test_registry_inventory_does_not_enable_selection() -> None:
    # Inventory is metadata only: with no selection resolved, nothing is active.
    registry = _build_compressor_registry()
    assert registry.active(None) == []


def test_builtin_entry_compress_delegates_via_router() -> None:
    # The built-in entries now expose a WORKING (non-raising) compress that
    # delegates to the router's own dispatch path. kompress with ML disabled is a
    # passthrough (no model load), proving the entry runs without raising.
    router = ContentRouter(ContentRouterConfig(enable_kompress=False))
    entry = router.compressor_registry.get("kompress")
    assert entry is not None
    out = entry.compress(CompressInput(content="hello world", content_type="text/plain"))
    assert isinstance(out, CompressOutput)
    assert out.content == "hello world"  # ML disabled → passthrough, never raises


def test_builtin_entry_without_router_is_inert_passthrough() -> None:
    # A registry built with no bound router (module-level inventory use) has
    # nothing to delegate to, so compress is an inert passthrough — still working
    # (non-raising), never a fabricated result.
    registry = _build_compressor_registry()
    entry = registry.get("kompress")
    assert entry is not None
    out = entry.compress(CompressInput(content="x", content_type="text/plain"))
    assert isinstance(out, CompressOutput)
    assert out.content == "x"


def test_discovery_merges_external_compressor(monkeypatch: pytest.MonkeyPatch) -> None:
    """A discovered `headroom.compressor` entry point joins the built-in inventory."""

    class _FakeExternal:
        @property
        def descriptor(self) -> CompressorDescriptor:
            return CompressorDescriptor(
                name="fake_external",
                content_types=["text/plain"],
                lossless=True,
                cost_tier="fast",
                recoverable=False,
            )

        def compress(self, inp: CompressInput) -> CompressOutput:
            return CompressOutput(
                content=inp.content,
                tokens_before=1,
                tokens_after=1,
                lossless=True,
            )

    class _FakeEntry:
        name = "fake_external"

        def load(self) -> type[_FakeExternal]:
            return _FakeExternal

    def _fake_entry_points(*, group: str) -> list[_FakeEntry]:
        assert group == cr_module.ENTRY_POINT_GROUP
        return [_FakeEntry()]

    monkeypatch.setattr(cr_module.importlib.metadata, "entry_points", _fake_entry_points)

    registry = _build_compressor_registry()
    assert "fake_external" in registry.names()
    assert set(registry.names()) >= _BUILTIN_NAMES
    # The external one is selectable/active; built-ins are inventory-only.
    active = registry.active({"fake_external"})
    assert [c.descriptor.name for c in active] == ["fake_external"]


def test_router_construction_is_unchanged_by_default_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even if discovery raises, the router still constructs with an empty-but-
    # present registry (fail-open) — never breaking the request path.
    def _boom(*, group: str) -> list[object]:
        raise RuntimeError("discovery blew up")

    monkeypatch.setattr(cr_module.importlib.metadata, "entry_points", _boom)
    router = ContentRouter(ContentRouterConfig())
    # discover() itself is fail-open, so built-ins are still registered.
    assert set(router.compressor_registry.names()) >= _BUILTIN_NAMES
