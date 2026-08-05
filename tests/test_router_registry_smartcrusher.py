"""Differential tests for the SMART_CRUSHER / KOMPRESS / TEXT registry flip.

SMART_CRUSHER flips the PRIMARY compressor invocation to the compressor registry
(``_registry_compress("smart_crusher", ...)``), while the shared post-strategy
Kompress -> Log fallback block stays a direct dispatch. KOMPRESS and TEXT now ALSO
dispatch via the registry (``_registry_compress("kompress", ...)``): the
``kompress`` built-in adapter delegates to the SAME ``_try_ml_compressor(content,
context, question)`` the router historically called, with ``question`` forwarded
via ``CompressInput.config['question']`` (previously the adapter dropped it,
passing ``None`` — that latent bug is fixed here). Content is therefore PRESERVED
byte-for-byte against the direct ``_try_ml_compressor`` call.

The ONE approved, non-byte-identical change is the token metric: KOMPRESS/TEXT now
report ``_estimate_tokens(output.content)`` (the router's calibrated estimate)
instead of ``_try_ml_compressor``'s tuple token count (Kompress's own
``compressed_tokens``). No content/routing/fallback/lossless-then-lossy decision
reads that metric for these two branches, so only the reported number changes.

Every path asserts registry-dispatch CONTENT == the historical direct-dispatch
content (and, via recorded call args, that query/bias/question flow through), and
that the token metric equals ``_estimate_tokens(output.content)``.

Offline guardrails:
  * No real ML/ONNX/HF inference — the ML boundary is mocked at
    ``_try_ml_compressor`` (SMART_CRUSHER fallback tests) or ``_get_kompress``
    (KOMPRESS/TEXT tests, which exercise the real ``_try_ml_compressor`` with a
    fake in-memory model).
  * SmartCrusher itself is pure-Python (no ML), so the SMART_CRUSHER success path
    runs the real crusher on a small JSON array.
  * ``relevance_split`` / ``lossless_then_lossy`` off and STAGE 0
    (``_lossless_first``) neutralized so the if/elif branch under test is the
    terminal path. All of that is shared, unchanged code.
  * The broad ``content_router``/``compression`` -k selection is NOT exercised
    (it hangs on HF-Hub/ONNX).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    _estimate_tokens,
)


def _router() -> ContentRouter:
    """Router whose if/elif branch is the terminal dispatch path.

    ``relevance_split`` off, ``lossless_then_lossy`` off, and
    ``ccr_inject_marker`` off make the compressed output deterministic and
    marker-free; the same config is applied to the direct reference and the
    dispatch router, so the differential holds regardless.
    """
    return ContentRouter(
        ContentRouterConfig(
            relevance_split=False,
            lossless_then_lossy=False,
            ccr_inject_marker=False,
        )
    )


def _isolate_branch(monkeypatch: pytest.MonkeyPatch, router: ContentRouter) -> None:
    """Neutralize STAGE 0 (``_lossless_first``) so the if/elif branch is exercised.

    ``_lossless_first`` is shared, unchanged code (the flip only touches the branch
    body), so forcing it to a no-op isolates what the flip actually changed.
    """
    monkeypatch.setattr(router, "_lossless_first", lambda content, strategy: (content, None))


# A JSON array the real SmartCrusher shrinks (so the fallback chain stays single).
_JSON = json.dumps([{"id": i, "status": "ok", "level": "INFO", "value": i * 2} for i in range(40)])


# ─────────────────────── SMART_CRUSHER: flipped (registry) ────────────────────


def test_smart_crusher_success_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    # SMART_CRUSHER-SUCCEEDS: the flip routes the primary ``.crush`` through the
    # registry "smart_crusher" adapter, which delegates to the SAME getter +
    # ``.crush(query=..., bias=...)``. Differential: registry dispatch == a direct
    # crush on an independent router. A real shrink means no fallback fires.
    router = _router()
    _isolate_branch(monkeypatch, router)
    context, bias = "q", 1.0
    direct = _router()._get_smart_crusher().crush(_JSON, query=context, bias=bias).compressed
    out, tokens, chain = router._apply_strategy_to_content(
        _JSON, CompressionStrategy.SMART_CRUSHER, context, bias=bias
    )
    assert out == direct
    assert tokens == _estimate_tokens(direct)
    assert chain == [CompressionStrategy.SMART_CRUSHER.value]
    assert len(out) < len(_JSON)


def test_smart_crusher_forwards_query_and_bias(monkeypatch: pytest.MonkeyPatch) -> None:
    # The flip must forward content, query (==context), and bias to ``.crush``
    # unchanged. A fake crusher records the exact call args, proving the registry
    # adapter's ``query=inp.query`` / ``bias=budget['bias']`` reproduce the direct
    # ``crush(content, query=context, bias=bias)`` call.
    router = _router()
    _isolate_branch(monkeypatch, router)
    seen: dict[str, object] = {}

    def _crush(content: str, query: str = "", bias: float = 1.0) -> SimpleNamespace:
        seen.update(content=content, query=query, bias=bias)
        return SimpleNamespace(compressed="CRUSHED " + content[:5])

    monkeypatch.setattr(router, "_get_smart_crusher", lambda: SimpleNamespace(crush=_crush))
    # Sentinel ML: if the flip WRONGLY dropped into the Kompress fallback we'd see
    # KOMPRESS appended to the chain; the chain assertion catches it.
    monkeypatch.setattr(
        router, "_try_ml_compressor", lambda *a, **k: ("KOMPRESS_SENTINEL", 999_999)
    )

    content = "x" * 200
    out, tokens, chain = router._apply_strategy_to_content(
        content, CompressionStrategy.SMART_CRUSHER, "myquery", bias=0.7
    )
    assert out == "CRUSHED " + content[:5]
    assert tokens == _estimate_tokens("CRUSHED " + content[:5])
    assert chain == [CompressionStrategy.SMART_CRUSHER.value]
    assert seen == {"content": content, "query": "myquery", "bias": 0.7}


def test_smart_crusher_kompress_fallback_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    # SMART_CRUSHER-NO-SHRINK -> KOMPRESS: the crusher returns the content
    # unchanged (no net saving), so the SHARED, UNCHANGED post-strategy block runs
    # the Kompress fallback. The flip only sets ``compressed``/``compressed_tokens``
    # on entry — identically to the direct crush — so the fallback fires exactly as
    # before: chain [smart_crusher, kompress], Kompress output adopted.
    router = _router()
    _isolate_branch(monkeypatch, router)
    monkeypatch.setattr(
        router,
        "_get_smart_crusher",
        lambda: SimpleNamespace(crush=lambda c, query="", bias=1.0: SimpleNamespace(compressed=c)),
    )
    # Kompress shrinks (fallback_tokens < compressed_tokens) so it is adopted.
    monkeypatch.setattr(router, "_try_ml_compressor", lambda c, ctx, q: ("KOMPRESSED::" + c, 3))
    out, tokens, chain = router._apply_strategy_to_content(
        _JSON, CompressionStrategy.SMART_CRUSHER, "ctx", bias=1.0
    )
    assert out == "KOMPRESSED::" + _JSON
    assert tokens == 3
    assert chain == [
        CompressionStrategy.SMART_CRUSHER.value,
        CompressionStrategy.KOMPRESS.value,
    ]


def test_smart_crusher_log_fallback_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    # SMART_CRUSHER-NO-SHRINK -> KOMPRESS-NO-SHRINK -> LOG: crusher passes content
    # through, Kompress fails to shrink (fallback_tokens NOT < compressed_tokens),
    # and — because the content is valid JSON and the log compressor is enabled —
    # the last-ditch Log fallback runs and shrinks. All of that is the shared,
    # unchanged block; the flip only feeds it identical entry state. Chain
    # [smart_crusher, kompress, log].
    router = _router()
    _isolate_branch(monkeypatch, router)
    monkeypatch.setattr(
        router,
        "_get_smart_crusher",
        lambda: SimpleNamespace(crush=lambda c, query="", bias=1.0: SimpleNamespace(compressed=c)),
    )
    # Kompress returns content unchanged with a huge token count -> NOT a shrink,
    # so the else branch (Log fallback) is taken.
    monkeypatch.setattr(router, "_try_ml_compressor", lambda c, ctx, q: (c, 10**9))
    monkeypatch.setattr(
        router,
        "_get_log_compressor",
        lambda: SimpleNamespace(
            compress=lambda c, bias=1.0: SimpleNamespace(compressed="LOG_FOLDED")
        ),
    )
    out, tokens, chain = router._apply_strategy_to_content(
        _JSON, CompressionStrategy.SMART_CRUSHER, "ctx", bias=1.0
    )
    assert out == "LOG_FOLDED"
    assert tokens == _estimate_tokens("LOG_FOLDED")
    assert chain == [
        CompressionStrategy.SMART_CRUSHER.value,
        CompressionStrategy.KOMPRESS.value,
        CompressionStrategy.LOG.value,
    ]


# ─────────────── KOMPRESS / TEXT: flipped (registry, ML boundary) ──────────────


def _fake_kompress(seen: dict[str, object]) -> SimpleNamespace:
    """In-memory Kompress model: records compress kwargs, no ONNX/HF."""

    def _compress(text: str, **kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        # ``compressed_tokens`` is the model's OWN count (7), deliberately unequal
        # to ``_estimate_tokens`` of the output. After the flip the branch reports
        # ``_estimate_tokens(output.content)`` and DISCARDS this tuple value — the
        # single approved metric change — so the tests below assert the metric is
        # the estimate, not 7.
        return SimpleNamespace(compressed="KOMPRESSED::" + text, compressed_tokens=7)

    return SimpleNamespace(
        is_ready=lambda: True,
        ensure_background_load=lambda: None,
        compress=_compress,
    )


def test_kompress_registry_dispatch_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    # KOMPRESS now dispatches through the registry "kompress" adapter, which
    # delegates to _try_ml_compressor with ``question`` forwarded via config.
    router = _router()
    _isolate_branch(monkeypatch, router)
    seen: dict[str, object] = {}
    monkeypatch.setattr(router, "_get_kompress", lambda: _fake_kompress(seen))

    content = "some plain text that the ML model would compress. " * 4
    out, tokens, chain = router._apply_strategy_to_content(
        content, CompressionStrategy.KOMPRESS, "ctx", question="my question", bias=1.0
    )
    # (a) CONTENT preserved — byte-identical to the direct _try_ml_compressor call
    # with the SAME question (so question is forwarded through the adapter).
    assert out == "KOMPRESSED::" + content
    assert chain == [CompressionStrategy.KOMPRESS.value]
    direct, direct_tokens = router._try_ml_compressor(content, "ctx", "my question")
    assert out == direct
    # The real ``question`` reached the model (config['question'] -> adapter ->
    # _try_ml_compressor -> compressor.compress), fixing the latent adapter bug
    # that dropped it (passed None).
    assert seen["question"] == "my question"
    # (b) APPROVED metric change: the token count is now _estimate_tokens(output),
    # NOT the model's own tuple value (7) that the direct call returns.
    assert tokens == _estimate_tokens(out)
    assert tokens != direct_tokens
    assert direct_tokens == 7


def test_text_registry_dispatch_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    # TEXT shares the "kompress" adapter — same guarantees as the KOMPRESS branch.
    router = _router()
    _isolate_branch(monkeypatch, router)
    seen: dict[str, object] = {}
    monkeypatch.setattr(router, "_get_kompress", lambda: _fake_kompress(seen))

    content = "plain prose the text strategy sends straight to kompress. " * 4
    out, tokens, chain = router._apply_strategy_to_content(
        content, CompressionStrategy.TEXT, "ctx", question="q2", bias=1.0
    )
    assert out == "KOMPRESSED::" + content
    assert chain == [CompressionStrategy.TEXT.value]
    direct, direct_tokens = router._try_ml_compressor(content, "ctx", "q2")
    assert out == direct
    assert seen["question"] == "q2"
    assert tokens == _estimate_tokens(out)
    assert tokens != direct_tokens
    assert direct_tokens == 7


def test_kompress_question_forwarding_changes_content(monkeypatch: pytest.MonkeyPatch) -> None:
    # QA-differential: a question-aware fake model embeds the question in its
    # output, so a DIFFERENT ``question`` yields DIFFERENT content. Proves the
    # router forwards ``question`` end-to-end via the registry adapter
    # (config['question'] -> _invoke_kompress -> _try_ml_compressor ->
    # compressor.compress) — the fix for the adapter that previously dropped it.
    router = _router()
    _isolate_branch(monkeypatch, router)

    def _qa_model() -> SimpleNamespace:
        def _compress(text: str, *, question: object = None, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(compressed=f"Q[{question}]::{text}", compressed_tokens=5)

        return SimpleNamespace(
            is_ready=lambda: True,
            ensure_background_load=lambda: None,
            compress=_compress,
        )

    monkeypatch.setattr(router, "_get_kompress", _qa_model)
    content = "the body the model compresses conditioned on the question. " * 3

    out_a, _tok_a, _ = router._apply_strategy_to_content(
        content, CompressionStrategy.KOMPRESS, "ctx", question="alpha", bias=1.0
    )
    out_b, _tok_b, _ = router._apply_strategy_to_content(
        content, CompressionStrategy.KOMPRESS, "ctx", question="beta", bias=1.0
    )
    assert out_a != out_b
    assert out_a.startswith("Q[alpha]::")
    assert out_b.startswith("Q[beta]::")
    # Each matches the direct _try_ml_compressor call with the SAME question.
    assert out_a == router._try_ml_compressor(content, "ctx", "alpha")[0]
    assert out_b == router._try_ml_compressor(content, "ctx", "beta")[0]
