"""Lossless-then-lossy dispatch.

In lossy mode (``lossless=False``) with ``lossless_then_lossy`` on, a foldable
block is FIRST byte-folded losslessly and THEN handed to the aggressive lossy
compressor (Kompress) on the folded remainder. The lossy result is kept only
when it saves at least ``lossy_min_extra_savings`` MORE than the fold already
did; otherwise the pure byte-exact fold is kept, so it is never worse than the
plain fold.

DIFF content is never lossy-chained (Kompressing hunks breaks ``git apply``).
Kompress is mocked so these run without the ModernBERT model.
"""

from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    _estimate_tokens,
)
from headroom.transforms.lossless_compaction import search_unheading


def _grep_block() -> str:
    # Repeated path prefixes → search_heading folds byte-exact (word count flat).
    paths = [
        "src/services/wallet/overdraft/automated_overdraft_initiation.py",
        "src/services/wallet/overdraft/capacity_limits.py",
    ]
    return (
        "\n".join(
            f"{p}:{ln}:    result = compute_overdraft_capacity(business_id, amount)"
            for p in paths
            for ln in range(1, 40)
        )
        + "\n"
    )


def _diff_block() -> str:
    files = ["foo", "bar", "baz"]
    return (
        "\n".join(
            f"diff --git a/{f}.py b/{f}.py\n"
            f"index 1111111aaaaaaa..2222222bbbbbbb 100644\n"
            f"--- a/{f}.py\n+++ b/{f}.py\n"
            f"@@ -1,3 +1,3 @@\n-    old_{f} = 1\n+    new_{f} = 2\n     unchanged_{f}"
            for f in files
        )
        + "\n"
    )


def _router(*, lossless_then_lossy, lossless=False, ccr=False, kompress=None):
    r = ContentRouter(
        ContentRouterConfig(
            lossless=lossless,
            lossless_then_lossy=lossless_then_lossy,
            ccr_inject_marker=ccr,
        )
    )
    calls: list[str] = []

    def _fake_kompress(content, context, question=None):
        calls.append(content)
        out = kompress(content) if kompress else content
        return out, len(out.split())

    r._try_ml_compressor = _fake_kompress  # type: ignore[method-assign]
    return r, calls


def _run(r, content):
    tr, rc = [], {}
    out, was = r._compress_block_content(
        content,
        hash(content),
        "",
        1.0,
        1.0,
        None,
        tr,
        rc,
        [],
        "tool_result",
        "tool",
        True,
    )
    return out, was, tr, rc


def test_lossy_after_fold_chains_when_it_helps():
    block = _grep_block()
    # kompress removes far more than the min-extra-savings floor → lossy kept.
    r, calls = _router(lossless_then_lossy=True, kompress=lambda c: "TINY")
    out, was, tr, rc = _run(r, block)
    assert was is True
    assert out == "TINY"
    assert len(calls) == 1  # kompress ran on the folded remainder
    assert rc.get("lossless_then_lossy_accept") == 1
    assert rc.get("lossless_accept", 0) == 0
    assert tr == ["router:tool_result:lossless_search+kompress"]


def test_keeps_pure_fold_when_lossy_marginal():
    block = _grep_block()
    # kompress returns the fold unchanged (no gain) → pure byte-exact fold kept.
    r, calls = _router(lossless_then_lossy=True, kompress=lambda c: c)
    out, was, tr, rc = _run(r, block)
    assert was is True
    assert len(calls) == 1  # kompress was attempted...
    assert rc.get("lossless_accept") == 1  # ...but the pure fold won
    assert rc.get("lossless_then_lossy_accept", 0) == 0
    assert search_unheading(out) == block  # fully recoverable (byte-exact)


def test_never_kompresses_diff():
    block = _diff_block()
    # kompress would mangle a diff; the lossy pass must never touch diff content.
    r, calls = _router(lossless_then_lossy=True, kompress=lambda c: "MANGLED")
    out, was, tr, rc = _run(r, block)
    assert was is True
    assert calls == []  # lossy stage never touched the diff
    assert "MANGLED" not in out
    assert out.count("@@ ") == block.count("@@ ")  # every hunk header preserved
    assert "new_foo = 2" in out  # hunk bodies intact → still applies


def test_lossy_after_fold_off_is_pure_fold():
    block = _grep_block()
    r, calls = _router(lossless_then_lossy=False, kompress=lambda c: "TINY")
    out, was, tr, rc = _run(r, block)
    assert was is True
    assert calls == []  # no lossy pass when lossless-then-lossy is disabled
    assert rc.get("lossless_accept") == 1
    assert search_unheading(out) == block


def test_lossy_after_fold_never_worse_than_pure_fold():
    block = _grep_block()
    r_fold, _ = _router(lossless_then_lossy=False, kompress=lambda c: "TINY")
    r_chain, _ = _router(lossless_then_lossy=True, kompress=lambda c: "TINY")
    out_fold, _, _, _ = _run(r_fold, block)
    out_chain, _, _, _ = _run(r_chain, block)
    assert len(out_chain) <= len(out_fold)  # chaining never loses to the pure fold


def test_lossy_gate_boundary_default(monkeypatch):
    monkeypatch.delenv("HEADROOM_LOSSY_MIN_EXTRA_SAVINGS", raising=False)
    block = _grep_block()
    r, _ = _router(lossless_then_lossy=True)
    assert abs(r._lossy_min_extra_savings - 0.05) < 1e-9  # default: require >=5% extra
    # Measure the fold with the SAME estimator the router's gate uses
    # (_estimate_tokens), not len(split()) — the two diverge, and the gate at
    # content_router.py compares _komp_tokens against _estimate_tokens(fold).
    fold_tok = _estimate_tokens(r._lossless_first(block, CompressionStrategy.SEARCH)[0])
    # Kompress that keeps 94% of fold tokens -> saves 6% >= the 5% floor -> chained.
    keep = int(fold_tok * 0.94)
    r._try_ml_compressor = lambda c, ctx, q=None: (" ".join(["w"] * keep), keep)  # type: ignore
    _, was, tr, rc = _run(r, block)
    assert was is True and rc.get("lossless_then_lossy_accept") == 1


def test_lossy_gate_env_override(monkeypatch):
    monkeypatch.setenv("HEADROOM_LOSSY_MIN_EXTRA_SAVINGS", "0.20")
    r, _ = _router(lossless_then_lossy=True)
    assert abs(r._lossy_min_extra_savings - 0.20) < 1e-9  # env override wins
    block = _grep_block()
    # Same estimator as the router gate (_estimate_tokens), not len(split()).
    fold_tok = _estimate_tokens(r._lossless_first(block, CompressionStrategy.SEARCH)[0])
    keep = int(fold_tok * 0.90)  # saves 10%: passes the 5% default but FAILS the 20% override
    r._try_ml_compressor = lambda c, ctx, q=None: (" ".join(["w"] * keep), keep)  # type: ignore
    _, was, tr, rc = _run(r, block)
    assert rc.get("lossless_accept") == 1  # kept pure fold under the stricter 20% gate
    assert rc.get("lossless_then_lossy_accept", 0) == 0


def test_lossy_after_fold_noop_in_lossless_only_mode():
    block = _grep_block()
    # lossless-only mode never emits lossy even if lossless-then-lossy is set.
    r, calls = _router(lossless_then_lossy=True, lossless=True, kompress=lambda c: "TINY")
    out, was, tr, rc = _run(r, block)
    assert was is True
    assert calls == []  # no lossy in lossless-only mode
    assert search_unheading(out) == block
