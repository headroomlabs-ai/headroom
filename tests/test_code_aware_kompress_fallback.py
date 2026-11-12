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


def test_enabled_kompress_still_turns_off_code_aware_fallback() -> None:
    # Kompress may be on for prose, but it must never compress code (#4214):
    # the router's code compressor leaves blocks it cannot handle unchanged.
    router = ContentRouter(ContentRouterConfig(enable_code_aware=True))

    compressor = router._get_code_compressor()

    assert compressor is not None
    assert compressor.config.fallback_to_kompress is False


PERL_CODE = "\n".join(
    ["#!/usr/bin/perl", "use strict;", "use warnings;"]
    + [
        f'my $v{i} = $ARGV[{i % 3}] // {i}; print "$v{i}\\n" if $v{i} =~ /^\\d+$/;'
        for i in range(60)
    ]
)

PYTHON_CODE = "\n".join(
    ["import json", ""]
    + [
        f"def handler_{i}(event):\n    if event.get('k{i}'):\n        return json.dumps(event)\n    return None\n"
        for i in range(40)
    ]
)


def _code_aware_entry(enable_kompress: bool):
    router = ContentRouter(
        ContentRouterConfig(enable_code_aware=True, enable_kompress=enable_kompress)
    )
    entry = router.compressor_registry.get("code_aware")
    assert entry is not None
    return router, entry


@pytest.mark.parametrize("enable_kompress", [False, True])
def test_unknown_language_code_passes_through_without_kompress(
    kompress_spies: tuple[MagicMock, MagicMock], enable_kompress: bool
) -> None:
    assert detect_language(UNKNOWN_CODE)[0] == CodeLanguage.UNKNOWN
    _, entry = _code_aware_entry(enable_kompress)

    out = entry.compress(CompressInput(content=UNKNOWN_CODE, content_type="text/x-code"))

    available, compressor = kompress_spies
    available.assert_not_called()
    compressor.assert_not_called()
    assert out.content == UNKNOWN_CODE


@pytest.mark.parametrize("enable_kompress", [False, True])
def test_perl_code_passes_through_without_kompress(
    kompress_spies: tuple[MagicMock, MagicMock], enable_kompress: bool
) -> None:
    # detect_language() reports Perl-dominant code as UNKNOWN on purpose, so the
    # Perl branch is reached through an explicit language hint.
    _, entry = _code_aware_entry(enable_kompress)

    out = entry.compress(
        CompressInput(content=PERL_CODE, content_type="text/x-code", config={"language": "perl"})
    )

    available, compressor = kompress_spies
    available.assert_not_called()
    compressor.assert_not_called()
    assert out.content == PERL_CODE


@pytest.mark.parametrize("enable_kompress", [False, True])
def test_ast_failure_passes_through_without_kompress(
    kompress_spies: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
    enable_kompress: bool,
) -> None:
    router, entry = _code_aware_entry(enable_kompress)
    code_compressor = router._get_code_compressor()
    assert code_compressor is not None

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated AST failure")

    monkeypatch.setattr(code_compressor, "_compress_with_ast", _boom)

    out = entry.compress(CompressInput(content=PYTHON_CODE, content_type="text/x-code"))

    available, compressor = kompress_spies
    available.assert_not_called()
    compressor.assert_not_called()
    assert out.content == PYTHON_CODE
