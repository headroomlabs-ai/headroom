"""Byte-identical differential tests for registry-resolved built-in dispatch.

The content router now dispatches the SIMPLE built-in strategies (CONFIG, LOG,
SEARCH, TABULAR) through the compressor registry instead of a hardcoded direct
``self._get_*().compress(...)`` call in ``_apply_strategy_to_content``. Each
built-in adapter delegates to the SAME ``_get_*`` getter+method with the SAME
arguments, so registry-resolved dispatch must be byte-identical to the historical
direct dispatch: same compressed content, same token count, same single-entry
``strategy_chain``.

Each FLIPPED strategy has a differential test comparing the router's dispatch
output to the built-in's direct output obtained via its ``_get_*`` getter — i.e.
"registry dispatch == old dispatch". SMART_CRUSHER is now also flipped (its
``.crush`` primary invocation goes through the registry while the shared
Kompress→Log fallback block stays direct); KOMPRESS/TEXT now dispatch via the
registry too — the ``kompress`` adapter delegates to the SAME ``_try_ml_compressor``
call with ``question`` forwarded via ``config['question']``, so CONTENT is
preserved and only the reported token metric changed. See
``test_router_registry_smartcrusher.py`` for the full SMART_CRUSHER fallback-chain
and KOMPRESS/TEXT differential coverage.

Offline guardrails:
  * No real ML/ONNX/HF inference — the KOMPRESS ML boundary is mocked.
  * The flipped strategies shrink their representative content, so no zero-savings
    Kompress fallback fires (that would touch the ML boundary and append KOMPRESS
    to the chain).
  * ``lossless_then_lossy`` and ``relevance_split`` are off so the if/elif branch
    is the terminal path; STAGE 0 (``_lossless_first``) is neutralized so search/
    log folds don't return before the branch. Both are shared, unchanged code.
  * The broad ``content_router``/``compression`` -k selection is NOT exercised
    (it hangs on HF-Hub/ONNX).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from headroom.transforms.content_router import (
    _BUILTIN_COMPRESSOR_DESCRIPTORS,
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    _estimate_tokens,
)


def _router() -> ContentRouter:
    """Router whose if/elif branch is the terminal dispatch path.

    ``relevance_split`` off (no LOG/SEARCH relevance split) and
    ``lossless_then_lossy`` off (no lossy layer on top of a strategy result) so a
    successful strategy result returns directly. ``ccr_inject_marker`` off makes
    the compressed output deterministic and marker-free; it is applied identically
    to the direct reference and the dispatch router, so the differential holds
    regardless of its value.
    """
    return ContentRouter(
        ContentRouterConfig(
            relevance_split=False,
            lossless_then_lossy=False,
            ccr_inject_marker=False,
        )
    )


def _isolate_branch(monkeypatch: pytest.MonkeyPatch, router: ContentRouter) -> None:
    """Neutralize STAGE 0 so the if/elif branch under test is exercised.

    ``_lossless_first`` runs unconditionally and can fold search/log content,
    returning before the if/elif. It is shared, unchanged code (the flip only
    touches the branch bodies), so forcing it to a no-op isolates what the flip
    actually changed without altering the branch semantics.
    """
    monkeypatch.setattr(router, "_lossless_first", lambda content, strategy: (content, None))


# Representative content per type. The flipped strategies must SHRINK this so the
# fallback-eligible strategies (TABULAR/CONFIG) don't trip the zero-savings
# Kompress fallback (which would append KOMPRESS to the chain).
_SEARCH = "\n".join(f"src/file{i}.py:{i}: def func{i}(): return {i}" for i in range(30))
_LOG = (
    "\n".join(f"2024-01-01 12:00:{i:02d} INFO task {i}" for i in range(30))
    + "\n"
    + "\n".join("identical repeated line" for _ in range(25))
)
# A markdown table the tabular compressor actually shrinks (schema-fold), and a
# repetitive INI the config compressor actually shrinks (block-fold) — so these
# fallback-eligible strategies produce a real token saving and the shared
# zero-savings Kompress fallback does NOT fire (chain stays single-entry).
_TABULAR = "| id | name | status | score |\n|----|------|--------|-------|\n" + "\n".join(
    f"| {i} | row{i} | ok | {i * 3} |" for i in range(60)
)
_CONFIG = "\n".join(
    f"[section_{i}]\nname = svc{i}\ntimeout = 30\nretries = 3\nverbose = false\nregion = us-east-1"
    for i in range(40)
)


# ───────────────────────── flipped (registry dispatch) ────────────────────────


def test_search_router_dispatch_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _router()
    _isolate_branch(monkeypatch, router)
    context, bias = "func", 1.0
    # OLD dispatch reference: same getter + method the branch used before the flip.
    direct = (
        _router()._get_search_compressor().compress(_SEARCH, context=context, bias=bias).compressed
    )
    out, tokens, chain = router._apply_strategy_to_content(
        _SEARCH, CompressionStrategy.SEARCH, context, bias=bias
    )
    assert out == direct
    assert tokens == _estimate_tokens(direct)
    assert chain == [CompressionStrategy.SEARCH.value]


def test_log_router_dispatch_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _router()
    _isolate_branch(monkeypatch, router)
    bias = 1.0
    direct = _router()._get_log_compressor().compress(_LOG, bias=bias).compressed
    out, tokens, chain = router._apply_strategy_to_content(
        _LOG, CompressionStrategy.LOG, "", bias=bias
    )
    assert out == direct
    assert tokens == _estimate_tokens(direct)
    assert chain == [CompressionStrategy.LOG.value]


def test_tabular_router_dispatch_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _router()
    _isolate_branch(monkeypatch, router)
    context, bias = "q", 1.0
    direct = (
        _router()
        ._get_tabular_compressor()
        .compress(_TABULAR, context=context, bias=bias)
        .compressed
    )
    out, tokens, chain = router._apply_strategy_to_content(
        _TABULAR, CompressionStrategy.TABULAR, context, bias=bias
    )
    assert out == direct
    assert tokens == _estimate_tokens(direct)
    # Fallback-eligible, but a real shrink means no zero-savings Kompress fallback.
    assert chain == [CompressionStrategy.TABULAR.value]
    assert len(out) < len(_TABULAR)


def test_config_router_dispatch_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_CONFIG_COMPRESSION", "1")
    router = _router()
    _isolate_branch(monkeypatch, router)
    context, bias = "q", 1.0
    direct = (
        _router()._get_config_compressor().compress(_CONFIG, context=context, bias=bias).compressed
    )
    out, tokens, chain = router._apply_strategy_to_content(
        _CONFIG, CompressionStrategy.CONFIG, context, bias=bias
    )
    assert out == direct
    # CONFIG's historical metric was len(text.split()) — a WORD count, which this
    # assertion was written to preserve across the registry-dispatch flip. That
    # was right for the flip and wrong as a metric: the value is divided by
    # `original_tokens`, which comes from _estimate_tokens, so a word numerator
    # over a token denominator made a byte-identical no-op score ~0.36 and get
    # accepted as a large saving. Now measured in the denominator's unit.
    assert tokens == _estimate_tokens(direct)
    assert chain == [CompressionStrategy.CONFIG.value]
    assert len(out) < len(_CONFIG)


# ─────────────────────────── deferred (unchanged) ────────────────────────────


def test_smart_crusher_router_dispatch_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    # SMART_CRUSHER is now FLIPPED: its primary ``.crush`` invocation routes
    # through the registry "smart_crusher" adapter (which delegates to the SAME
    # getter + ``.crush(query=..., bias=...)``), while the shared Kompress→Log
    # fallback block stays direct. Registry dispatch must be byte-identical to the
    # historical direct crush. A JSON array shrinks, so no fallback fires and the
    # chain stays single.
    router = _router()
    _isolate_branch(monkeypatch, router)
    content = json.dumps(
        [{"id": i, "status": "ok", "level": "INFO", "value": i * 2} for i in range(40)]
    )
    direct = _router()._get_smart_crusher().crush(content, query="q", bias=1.0).compressed
    out, _tokens, chain = router._apply_strategy_to_content(
        content, CompressionStrategy.SMART_CRUSHER, "q", bias=1.0
    )
    assert out == direct
    assert chain == [CompressionStrategy.SMART_CRUSHER.value]


def _fallback_router() -> ContentRouter:
    """Router with CODE_AWARE routing enabled and the if/elif branch terminal.

    Same isolation as :func:`_router` (no relevance split, no lossy layer, markers
    off) but with ``enable_code_aware=True`` since that flag gates CODE_AWARE
    ROUTING (default off) rather than the getter. HTML routing is on by default.
    """
    return ContentRouter(
        ContentRouterConfig(
            relevance_split=False,
            lossless_then_lossy=False,
            ccr_inject_marker=False,
            enable_code_aware=True,
        )
    )


# ─────────────── flipped fallback strategies: CODE_AWARE / HTML ───────────────


def test_code_aware_router_dispatch_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    # CODE_AWARE-SUCCEEDS: the flip routes through the registry "code_aware"
    # adapter, which delegates to the SAME getter+method with language via config.
    # A deterministic fake getter keeps this offline (no tree-sitter) and records
    # the exact call args, proving content/language/context flow through unchanged.
    router = _fallback_router()
    _isolate_branch(monkeypatch, router)
    seen: dict[str, object] = {}
    shrunk = "def foo(): ...  # compressed body"

    def _fake_compress(content: str, language: object = None, context: str = "") -> SimpleNamespace:
        seen.update(content=content, language=language, context=context)
        return SimpleNamespace(compressed=shrunk)

    monkeypatch.setattr(
        router, "_get_code_compressor", lambda: SimpleNamespace(compress=_fake_compress)
    )
    # Sentinel ML: if the flip WRONGLY dropped into a Kompress fallback we'd see
    # KOMPRESS appended to the chain; the chain assertion catches it.
    monkeypatch.setattr(
        router, "_try_ml_compressor", lambda *a, **k: ("KOMPRESS_SENTINEL", 999_999)
    )

    content = "def foo():\n    " + "x = 1\n    " * 40 + "return x\n"
    out, tokens, chain = router._apply_strategy_to_content(
        content, CompressionStrategy.CODE_AWARE, "q", language="python", bias=1.0
    )
    assert out == shrunk
    # CODE_AWARE's historical token metric is len(compressed.split()), NOT _estimate_tokens.
    assert tokens == len(shrunk.split())
    assert chain == [CompressionStrategy.CODE_AWARE.value]
    # The flip forwarded content, language, and context through the registry adapter.
    assert seen == {"content": content, "language": "python", "context": "q"}


def test_code_aware_unavailable_falls_back_to_kompress(monkeypatch: pytest.MonkeyPatch) -> None:
    # CODE_AWARE-RETURNS-NONE: with the code compressor UNAVAILABLE (tree-sitter
    # missing) the branch's local `compressed` stays None, so the EXISTING inline
    # Kompress fallback runs — the flip touches none of that logic. Mock
    # _try_ml_compressor so NO real ML runs, and assert the SAME strategy_chain
    # ([code_aware, kompress]) the historical direct dispatch produced.
    router = _fallback_router()
    _isolate_branch(monkeypatch, router)
    monkeypatch.setattr(router, "_get_code_compressor", lambda: None)
    monkeypatch.setattr(
        router,
        "_try_ml_compressor",
        lambda content, ctx, q: ("KOMPRESSED::" + content, 3),
    )
    content = "def foo():\n    return 1\n"
    out, tokens, chain = router._apply_strategy_to_content(
        content, CompressionStrategy.CODE_AWARE, "ctx", language=None, bias=1.0
    )
    assert out == "KOMPRESSED::" + content
    assert tokens == 3
    assert chain == [CompressionStrategy.CODE_AWARE.value, CompressionStrategy.KOMPRESS.value]


def test_html_router_dispatch_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    # HTML-EXTRACT-SUCCEEDS: the flip routes through the registry "html" adapter,
    # which delegates to the SAME getter + extract(). A deterministic fake getter
    # keeps this offline (no trafilatura) and records the call arg.
    router = _fallback_router()
    _isolate_branch(monkeypatch, router)
    extracted = "Extracted article body text that trafilatura would return."
    seen: dict[str, object] = {}

    def _fake_extract(content: str) -> SimpleNamespace:
        seen["content"] = content
        return SimpleNamespace(extracted=extracted)

    monkeypatch.setattr(
        router, "_get_html_extractor", lambda: SimpleNamespace(extract=_fake_extract)
    )
    content = "<html><body><article><p>hello world</p></article></body></html>"
    out, tokens, chain = router._apply_strategy_to_content(
        content, CompressionStrategy.HTML, "", bias=1.0
    )
    assert out == extracted
    assert tokens == _estimate_tokens(extracted)
    assert chain == [CompressionStrategy.HTML.value]
    assert seen == {"content": content}


def test_html_extract_none_falls_through_to_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    # HTML-EXTRACT-NONE: when extraction yields None the adapter reports
    # compressed=False, the branch's local `compressed` collapses to None, and the
    # function falls through to the bottom passthrough exactly as the historical
    # `result.extracted is None` path — chain [html, passthrough], content verbatim.
    router = _fallback_router()
    _isolate_branch(monkeypatch, router)
    monkeypatch.setattr(
        router,
        "_get_html_extractor",
        lambda: SimpleNamespace(extract=lambda content: SimpleNamespace(extracted=None)),
    )
    # Sentinel ML so we'd notice if the None path wrongly reached a lossy compressor.
    monkeypatch.setattr(router, "_try_ml_compressor", lambda *a, **k: ("KOMPRESS_SENTINEL", 1))
    content = "<html><body><script>no extractable article body</script></body></html>"
    out, tokens, chain = router._apply_strategy_to_content(
        content, CompressionStrategy.HTML, "", bias=1.0
    )
    assert out == content
    assert tokens == _estimate_tokens(content)
    assert chain == [CompressionStrategy.HTML.value, CompressionStrategy.PASSTHROUGH.value]


def test_diff_deferred_no_registry_entry() -> None:
    # DIFF is DEFERRED: there is no "diff" built-in adapter/descriptor in the
    # registry inventory, so registry resolution would return None and fall back
    # to raw content — NOT byte-identical. It stays on its direct dispatch until a
    # diff built-in adapter lands (PR-A scope).
    router = _router()
    assert router.compressor_registry.get("diff") is None
    assert "diff" not in {d.name for d in _BUILTIN_COMPRESSOR_DESCRIPTORS}


def test_kompress_registry_dispatch_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    # KOMPRESS now dispatches via the registry "kompress" adapter, which still
    # delegates to _try_ml_compressor (the ML boundary). Mock the underlying model
    # so no real ONNX/HF inference runs, and assert the compressed CONTENT is
    # byte-identical to the direct _try_ml_compressor call (question is None here).
    router = _router()
    _isolate_branch(monkeypatch, router)
    fake = SimpleNamespace(
        is_ready=lambda: True,
        ensure_background_load=lambda: None,
        compress=lambda text, **kwargs: SimpleNamespace(
            compressed="KOMPRESSED::" + text, compressed_tokens=7
        ),
    )
    monkeypatch.setattr(router, "_get_kompress", lambda: fake)
    content = "some plain text that the ML model would compress. " * 4
    out, _tokens, chain = router._apply_strategy_to_content(
        content, CompressionStrategy.KOMPRESS, "", bias=1.0
    )
    assert out == "KOMPRESSED::" + content
    assert chain == [CompressionStrategy.KOMPRESS.value]
    # CONTENT is byte-identical to the direct ML call (registry round-trip preserves
    # content); the approved token-metric change is covered in
    # test_router_registry_smartcrusher.py.
    assert out == router._try_ml_compressor(content, "", None)[0]
