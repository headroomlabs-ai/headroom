"""Differential/faithfulness tests for the built-in Compressor adapters.

Each built-in registry entry now delegates ``compress`` to the SAME underlying
built-in method the content router invokes in ``_apply_strategy_to_content``
(reached through the router's own ``_get_*`` getter so config flows through
identically). These tests feed representative content per content type and assert
the adapter output matches what the built-in produces DIRECTLY, and that every
built-in registry entry exposes a working (non-raising) ``compress``.

Guardrails honored:
  * Kompress is mocked — no real ONNX/HF model inference (which hangs here).
  * Only the additive registry path is exercised; the router's dispatch
    (``_apply_strategy_to_content``) is never called, so these prove the adapter
    capability in isolation without touching routing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from headroom.transforms.compressor_registry import CompressInput, CompressOutput
from headroom.transforms.content_router import (
    _BUILTIN_COMPRESSOR_DESCRIPTORS,
    ContentRouter,
    ContentRouterConfig,
    _BuiltinCompressorEntry,
    _estimate_tokens,
)


def _router() -> ContentRouter:
    # enable_code_aware defaults off (that flag gates ROUTING, not the getter);
    # turn it on so nothing about the code getter differs in this test env. The
    # adapters use the getters directly regardless of the enable flags.
    return ContentRouter(ContentRouterConfig(enable_code_aware=True))


def _entry(router: ContentRouter, name: str) -> _BuiltinCompressorEntry:
    entry = router.compressor_registry.get(name)
    assert isinstance(entry, _BuiltinCompressorEntry), name
    return entry


def _assert_output_contract(
    out: CompressOutput, inp: CompressInput, entry: _BuiltinCompressorEntry
) -> None:
    assert isinstance(out, CompressOutput)
    # Token counts use the router's own estimator, over the pure-data content.
    assert out.tokens_before == _estimate_tokens(inp.content)
    assert out.tokens_after == _estimate_tokens(out.content)
    # lossless mirrors the descriptor; adapters emit no markers/warnings and no
    # recovery map (built-ins mirror hash -> original into the CCR store as a
    # side effect of their own compress call, not on the returned result).
    assert out.lossless == entry.descriptor.lossless
    assert out.markers == []
    assert out.recoverable == {}
    assert out.warnings == []


# ─────────────────────── differential (per built-in) ─────────────────────────


def test_smart_crusher_adapter_matches_builtin() -> None:
    router = _router()
    content = json.dumps(
        [{"id": i, "status": "ok", "level": "INFO", "value": i * 2} for i in range(40)]
    )
    direct = router._get_smart_crusher().crush(content, query="q", bias=1.0).compressed
    entry = _entry(router, "smart_crusher")
    inp = CompressInput(content=content, content_type="application/json", query="q")
    out = entry.compress(inp)
    assert out.content == direct
    assert out.tokens_after < out.tokens_before  # representative JSON actually shrinks
    _assert_output_contract(out, inp, entry)


def test_tabular_adapter_matches_builtin() -> None:
    router = _router()
    content = "id,name,status,score\n" + "\n".join(f"{i},row{i},ok,{i * 3}" for i in range(50))
    direct = router._get_tabular_compressor().compress(content, context="q", bias=1.0).compressed
    entry = _entry(router, "tabular")
    inp = CompressInput(content=content, content_type="text/csv", query="q")
    out = entry.compress(inp)
    assert out.content == direct
    _assert_output_contract(out, inp, entry)


def test_log_adapter_matches_builtin() -> None:
    router = _router()
    content = (
        "\n".join(f"2024-01-01 12:00:{i:02d} INFO task {i}" for i in range(30))
        + "\n"
        + "\n".join("identical repeated line" for _ in range(25))
    )
    direct = router._get_log_compressor().compress(content, bias=1.0).compressed
    entry = _entry(router, "log")
    inp = CompressInput(content=content, content_type="text/x-log", query="")
    out = entry.compress(inp)
    assert out.content == direct
    assert out.tokens_after < out.tokens_before  # repetitive log actually shrinks
    _assert_output_contract(out, inp, entry)


def test_search_adapter_matches_builtin() -> None:
    router = _router()
    content = "\n".join(f"src/file{i}.py:{i}: def func{i}(): return {i}" for i in range(30))
    direct = router._get_search_compressor().compress(content, context="func", bias=1.0).compressed
    entry = _entry(router, "search")
    inp = CompressInput(content=content, content_type="text/x-search-results", query="func")
    out = entry.compress(inp)
    assert out.content == direct
    _assert_output_contract(out, inp, entry)


def test_config_adapter_matches_builtin() -> None:
    router = _router()
    content = "\n".join(f"key{i} = value{i}" for i in range(30)) + "\n# comment\n\n# another\n"
    direct = router._get_config_compressor().compress(content, context="q", bias=1.0).compressed
    entry = _entry(router, "config")
    inp = CompressInput(content=content, content_type="text/x-config", query="q")
    out = entry.compress(inp)
    assert out.content == direct
    _assert_output_contract(out, inp, entry)


def test_code_aware_adapter_matches_builtin() -> None:
    router = _router()
    content = (
        "def foo(x):\n"
        "    # a comment\n"
        "    return x + 1\n\n\n"
        "class Bar:\n"
        "    def baz(self):\n"
        "        return 42\n"
    )
    compressor = router._get_code_compressor()
    if compressor is None:
        pytest.skip("code compressor (tree-sitter) unavailable in this environment")
    direct = compressor.compress(content, language=None, context="q").compressed
    entry = _entry(router, "code_aware")
    inp = CompressInput(content=content, content_type="text/x-code", query="q")
    out = entry.compress(inp)
    assert out.content == direct
    _assert_output_contract(out, inp, entry)


def test_html_adapter_matches_builtin() -> None:
    router = _router()
    content = (
        "<html><head><title>T</title></head><body><nav>menu</nav>"
        "<article><h1>Hi</h1><p>Hello world, this is the article body content "
        "that trafilatura should extract from the surrounding chrome.</p></article>"
        "</body></html>"
    )
    extractor = router._get_html_extractor()
    if extractor is None:
        pytest.skip("html extractor (trafilatura) unavailable in this environment")
    direct = extractor.extract(content).extracted
    entry = _entry(router, "html")
    inp = CompressInput(content=content, content_type="text/html", query="")
    out = entry.compress(inp)
    # Adapter maps empty/None extraction to passthrough, matching the router.
    assert out.content == (direct if direct is not None else content)
    _assert_output_contract(out, inp, entry)


def test_kompress_adapter_maps_mocked_result(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _router()
    # Mock the underlying kompress compressor so NO real ONNX/HF inference runs.
    fake = SimpleNamespace(
        is_ready=lambda: True,
        ensure_background_load=lambda: None,
        compress=lambda text, **kwargs: SimpleNamespace(
            compressed="KOMPRESSED::" + text, compressed_tokens=7
        ),
    )
    monkeypatch.setattr(router, "_get_kompress", lambda: fake)

    content = "some plain text that the ML model would compress. " * 4
    entry = _entry(router, "kompress")
    inp = CompressInput(content=content, content_type="text/plain", query="")
    out = entry.compress(inp)
    # The adapter delegates to _try_ml_compressor (the router's kompress path);
    # with no protected tags the result is the fake's compressed text verbatim.
    assert out.content == "KOMPRESSED::" + content
    # ... which is exactly what the router's own kompress dispatch produces.
    assert out.content == router._try_ml_compressor(content, "", None)[0]
    _assert_output_contract(out, inp, entry)


def test_kompress_adapter_forwards_question_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # The kompress adapter reads ``question`` from ``CompressInput.config`` and
    # forwards it into _try_ml_compressor (fixing the latent bug where it hardcoded
    # None). A question-aware fake model embeds the question, so the adapter output
    # differs when the config question differs — proving end-to-end forwarding.
    router = _router()
    seen: dict[str, object] = {}

    def _compress(text: str, *, question: object = None, **kwargs: object) -> SimpleNamespace:
        seen["question"] = question
        return SimpleNamespace(compressed=f"Q[{question}]::{text}", compressed_tokens=5)

    monkeypatch.setattr(
        router,
        "_get_kompress",
        lambda: SimpleNamespace(
            is_ready=lambda: True, ensure_background_load=lambda: None, compress=_compress
        ),
    )
    content = "plain text conditioned on the question " * 4
    entry = _entry(router, "kompress")
    out = entry.compress(
        CompressInput(content=content, content_type="text/plain", config={"question": "why"})
    )
    assert seen["question"] == "why"
    assert out.content == f"Q[why]::{content}"
    # No question in config forwards None (the historical no-question path).
    seen.clear()
    out_none = entry.compress(CompressInput(content=content, content_type="text/plain"))
    assert seen["question"] is None
    assert out_none.content == f"Q[None]::{content}"


def test_kompress_adapter_passthrough_when_ml_disabled() -> None:
    # With ML disabled the router's kompress path is a passthrough (no model load
    # / network), so the adapter returns the content unchanged, never raising.
    router = ContentRouter(ContentRouterConfig(enable_kompress=False))
    content = "plain text with nothing special to compress"
    entry = _entry(router, "kompress")
    inp = CompressInput(content=content, content_type="text/plain", query="")
    out = entry.compress(inp)
    assert out.content == content
    _assert_output_contract(out, inp, entry)


def test_image_adapter_is_documented_passthrough() -> None:
    # ImageCompressor operates on message image blocks (list[dict]) via
    # ImageCompressor.compress(messages), not on the str-based CompressInput
    # contract, and images are never routed through _apply_strategy_to_content.
    # There is no faithful str -> str delegation, so the adapter passes str
    # content through unchanged (documented, never a fabricated compression).
    router = _router()
    entry = _entry(router, "image")
    content = "arbitrary str content that is never an image payload"
    inp = CompressInput(content=content, content_type="image/png", query="")
    out = entry.compress(inp)
    assert out.content == content
    _assert_output_contract(out, inp, entry)


# ───────────────────── compressed-signal (did-compress flag) ─────────────────


def test_default_compress_output_flag_is_true() -> None:
    # Additive contract: `compressed` defaults True so existing/external
    # compressors that never set it are unaffected (treated as having compressed).
    out = CompressOutput(content="x", tokens_before=1, tokens_after=1, lossless=True)
    assert out.compressed is True


def test_adapter_reports_compressed_true_when_builtin_returns_content() -> None:
    # A built-in that returns real content → compressed=True and that content.
    descriptor = _BUILTIN_COMPRESSOR_DESCRIPTORS[0]
    entry = _BuiltinCompressorEntry(descriptor, router=object(), invoke=lambda r, inp: "SHRUNK")
    out = entry.compress(
        CompressInput(content="a longer original block", content_type="text/plain")
    )
    assert out.compressed is True
    assert out.content == "SHRUNK"


def test_adapter_reports_compressed_false_when_builtin_returns_none() -> None:
    # A built-in that returns None (unavailable / not applicable to this str
    # input) → compressed=False and the ORIGINAL content passed through unchanged.
    descriptor = _BUILTIN_COMPRESSOR_DESCRIPTORS[0]
    entry = _BuiltinCompressorEntry(descriptor, router=object(), invoke=lambda r, inp: None)
    out = entry.compress(CompressInput(content="original", content_type="text/plain"))
    assert out.compressed is False
    assert out.content == "original"


def test_adapter_reports_compressed_false_when_no_router() -> None:
    # No bound router → nothing to delegate to → compressed=False, passthrough.
    descriptor = _BUILTIN_COMPRESSOR_DESCRIPTORS[0]
    entry = _BuiltinCompressorEntry(descriptor, router=None)
    out = entry.compress(CompressInput(content="original", content_type="text/plain"))
    assert out.compressed is False
    assert out.content == "original"


def test_image_builtin_adapter_reports_not_compressed() -> None:
    # The image built-in never compresses str content (documented passthrough) →
    # compressed=False, original content unchanged.
    router = _router()
    entry = _entry(router, "image")
    out = entry.compress(CompressInput(content="not an image payload", content_type="image/png"))
    assert out.compressed is False
    assert out.content == "not an image payload"


# ──────────────────────── registry-wide invariants ───────────────────────────


def test_every_builtin_registry_entry_has_working_compress() -> None:
    # ML disabled so kompress is a passthrough (no model load / network); every
    # other built-in returns unchanged on this non-matching plain-text block.
    router = ContentRouter(ContentRouterConfig(enable_kompress=False, enable_code_aware=True))
    names = {d.name for d in _BUILTIN_COMPRESSOR_DESCRIPTORS}
    assert names, "expected built-in descriptors to be registered"
    content = "hello world\nsecond line\n"
    for name in sorted(names):
        entry = _entry(router, name)
        out = entry.compress(CompressInput(content=content, content_type="text/plain"))
        assert isinstance(out, CompressOutput), name
        assert isinstance(out.content, str), name
        assert out.tokens_before == _estimate_tokens(content), name
        assert out.tokens_after == _estimate_tokens(out.content), name
        assert out.lossless == entry.descriptor.lossless, name


def test_adapters_never_expand_or_blank_representative_content() -> None:
    # Adapter outputs must be usable blocks: never blank when input is non-blank,
    # never longer than the input (mirrors the router's own external-dispatch
    # guards, computed here from the built-in's real output).
    router = _router()
    cases = {
        "smart_crusher": json.dumps([{"id": i, "v": i} for i in range(30)]),
        "log": "\n".join("repeated identical log line" for _ in range(40)),
        "search": "\n".join(f"a/b{i}.py:{i}: match here" for i in range(30)),
        "config": "\n".join(f"k{i}=v{i}" for i in range(30)),
    }
    for name, content in cases.items():
        entry = _entry(router, name)
        out = entry.compress(CompressInput(content=content, content_type="text/plain", query=""))
        assert out.content.strip(), name
        assert len(out.content) <= len(content), name
