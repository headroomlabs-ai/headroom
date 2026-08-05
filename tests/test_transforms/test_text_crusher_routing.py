"""Phase 2 (#1171): the kompress size-gate routes oversized text to TextCrusher
when HEADROOM_TEXT_CRUSHER is enabled (real prose savings), instead of the
LogCompressor (which yields ~0 on prose) or ModernBERT (slow)."""

from __future__ import annotations

from headroom.transforms.content_router import ContentRouter, _estimate_tokens


def _prose() -> str:
    return " ".join(
        f"Sentence {i} about distributed systems and authentication tokens expiring soon."
        for i in range(300)
    )


def test_gate_routes_to_text_crusher_when_enabled(monkeypatch):
    monkeypatch.setenv("HEADROOM_TEXT_CRUSHER", "1")
    router = ContentRouter()
    router._kompress_max_tokens = 50  # tiny ceiling so the gate fires

    def _boom():
        raise AssertionError("kompress must not run for gated input")

    monkeypatch.setattr(router, "_get_kompress", _boom)

    prose = _prose()
    out, ntok = router._try_ml_compressor(prose, "authentication tokens")

    assert router._kompress_gate_fires == 1
    # TextCrusher actually compressed (LogCompressor ~0). Compare like-for-like
    # against the same token estimator the compressor reports in, not a raw
    # word count, so the assertion tracks real token reduction.
    assert ntok < _estimate_tokens(prose)
    assert set(out.split()) <= set(prose.split())  # extractive: no invented words


def test_text_crusher_disabled_by_default(monkeypatch):
    monkeypatch.delenv("HEADROOM_TEXT_CRUSHER", raising=False)
    router = ContentRouter()
    assert router._get_text_crusher() is None
