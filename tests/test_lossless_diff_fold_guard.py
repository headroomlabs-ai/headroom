"""The lossless `diff` fold is purely subtractive with no inverse check, so it
must only run on diff-shaped content — never on arbitrary text that happens to
contain an `index <hex>..<hex>` line."""

from __future__ import annotations

from headroom.transforms.content_router import CompressionStrategy, ContentRouter


def _lossless_first(content: str, strategy: CompressionStrategy):
    # _lossless_first only uses the (static) _looks_like_diff and a lazy import,
    # so a bare instance exercises it without the full router init.
    router = object.__new__(ContentRouter)
    return router._lossless_first(content, strategy)


def test_diff_fold_does_not_drop_index_line_from_non_diff_text():
    content = "Here are the object refs:\nindex 0123abc..def4567\nAll done.\n"

    out, label = _lossless_first(content, CompressionStrategy.PASSTHROUGH)

    # The git-blob-index-shaped line must survive; nothing should fold.
    assert "index 0123abc..def4567" in out
    assert out == content
    assert label is None


def test_diff_fold_still_applies_to_real_diffs():
    diff = (
        "diff --git a/x b/x\nindex 1111111..2222222 100644\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    )

    out, label = _lossless_first(diff, CompressionStrategy.DIFF)

    # A genuine diff still gets its index bookkeeping folded (semantic-lossless
    # for `git apply`).
    assert "index 1111111..2222222" not in out
    assert label == "lossless_diff"
