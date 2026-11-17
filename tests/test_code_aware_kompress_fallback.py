"""The code_aware Kompress fallback honours ``enable_kompress=False`` (#4719).

``CodeAwareCompressor`` falls back to Kompress for code it cannot parse (unknown
language, Perl, no tree-sitter, AST failure). That fallback instantiates its own
``KompressCompressor``, so ``ContentRouter._get_kompress()`` returning ``None``
was not enough: the router must also switch the code compressor's fallback off.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import headroom.transforms.code_compressor as cc
import headroom.transforms.kompress_compressor as kc
from headroom.transforms.code_compressor import CodeLanguage, detect_language
from headroom.transforms.compressor_registry import CompressInput
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

# Pseudo-assembly that detect_language() cannot identify (the issue's repro).
UNKNOWN_CODE = "\n".join(
    f"step {i}: load register r{i % 8} <- mem[{i * 4}]; jump-if-zero r{i % 8} label_{i}"
    for i in range(80)
)


@pytest.fixture(autouse=True)
def _tree_sitter_present(monkeypatch: pytest.MonkeyPatch) -> None:
    # The router only builds the code compressor when tree-sitter parses; the
    # unknown-language fallback never reaches the parser, so stub the probe to
    # exercise the router path in environments without the ``code`` extra.
    monkeypatch.setattr(cc, "_check_tree_sitter_available", lambda: True)


@pytest.fixture()
def kompress_spies(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    available = MagicMock(return_value=True)
    compressor = MagicMock()
    compressor.return_value.compress.return_value = SimpleNamespace(
        compressed="kompressed",
        original_tokens=100,
        compressed_tokens=1,
        compression_ratio=0.01,
    )
    monkeypatch.setattr(kc, "is_kompress_available", available)
    monkeypatch.setattr(kc, "KompressCompressor", compressor)
    return available, compressor


def test_disabled_kompress_turns_off_code_aware_fallback() -> None:
    router = ContentRouter(ContentRouterConfig(enable_code_aware=True, enable_kompress=False))

    compressor = router._get_code_compressor()

    assert compressor is not None
    assert compressor.config.fallback_to_kompress is False


def test_enabled_kompress_keeps_code_aware_fallback() -> None:
    router = ContentRouter(ContentRouterConfig(enable_code_aware=True))

    compressor = router._get_code_compressor()

    assert compressor is not None
    assert compressor.config.fallback_to_kompress is True


def test_unknown_language_code_passes_through_without_kompress(
    kompress_spies: tuple[MagicMock, MagicMock],
) -> None:
    assert detect_language(UNKNOWN_CODE)[0] == CodeLanguage.UNKNOWN
    router = ContentRouter(ContentRouterConfig(enable_code_aware=True, enable_kompress=False))
    entry = router.compressor_registry.get("code_aware")
    assert entry is not None

    out = entry.compress(CompressInput(content=UNKNOWN_CODE, content_type="text/x-code"))

    available, compressor = kompress_spies
    available.assert_not_called()
    compressor.assert_not_called()
    assert out.content == UNKNOWN_CODE
