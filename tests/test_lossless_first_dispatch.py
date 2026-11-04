"""Lossless-first dispatch (intended design).

Lossless folds run FIRST for every tool-output block, regardless of the
``lossless`` flag, and are accepted on a real byte reduction even when the word
count is flat (the case the word-ratio gate used to reject). In lossless-only
mode (flag on, no CCR) foldable content folds and non-foldable content is left
verbatim (no lossy drop). In CCR mode (flag off) a foldable block still keeps
its byte-exact fold — the lossless floor is never discarded by a later lossy
stage.
"""

from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
)
from headroom.transforms.lossless_compaction import search_unheading
from headroom.transforms.lossless_provider import (
    get_lossless_provider,
    get_lossless_verifier,
    set_lossless_provider,
)


def _grep_block() -> str:
    # Long, repeated path prefixes → search_heading collapses to --heading form.
    # Word count stays flat/rises while bytes drop a lot (heading adds path words).
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


def _code_block() -> str:
    return (
        "\n".join(
            f"    def method_{i}(self, arg_{i}):\n        return self.reg[{i}] + arg_{i} * {i}"
            for i in range(40)
        )
        + "\n"
    )


def _compress(content: str, *, lossless: bool):
    router = ContentRouter(ContentRouterConfig(lossless=lossless))
    tr: list[str] = []
    out, was = router._compress_block_content(
        content,
        hash((content, lossless)),
        "",
        1.0,
        1.0,
        None,
        tr,
        {},
        [],
        "tool_result",
        "tool",
        True,
    )
    return out, was, tr


def test_flag_on_search_folds_lossless_byte_exact():
    block = _grep_block()
    out, was, tr = _compress(block, lossless=True)
    assert was is True
    assert tr == ["router:tool_result:lossless_search"]
    assert len(out) < len(block)
    # word count is flat/higher -> the old word-ratio gate would have rejected it
    assert len(out.split()) >= len(block.split())
    # fully recoverable
    assert search_unheading(out) == block


def test_flag_on_search_fold_is_deterministic():
    block = _grep_block()
    out1, _, _ = _compress(block, lossless=True)
    out2, _, _ = _compress(block, lossless=True)
    assert out1 == out2  # pure function of content -> prefix-cache safe


def test_flag_on_leaves_non_foldable_code_verbatim():
    # Lossless-only mode must never emit a lossy / marker-free drop.
    out, was, tr = _compress(_code_block(), lossless=True)
    assert was is False
    assert tr == []


def test_flag_off_still_keeps_lossless_floor_for_foldable():
    block = _grep_block()
    out, was, tr = _compress(block, lossless=False)
    assert was is True
    assert tr == ["router:tool_result:lossless_search"]
    assert search_unheading(out) == block


def test_has_lossless_fold_admits_small_block_below_size_floor():
    # A <500-char search block must be admitted — lossless has NO size floor
    # (the min_chars floor guards the lossy path only).
    router = ContentRouter(ContentRouterConfig(lossless=True))
    small = "\n".join(f"pkg/mod/long_filename.py:{n}:value = {n}" for n in range(1, 8)) + "\n"
    assert len(small) < 500
    assert router._has_lossless_fold(small) is True
    # non-foldable tiny code must NOT be admitted (stays "small")
    assert router._has_lossless_fold("def f():\n    return 1\n") is False


def test_registered_provider_competes_on_the_general_path():
    # A registered lossless provider must be consulted for ANY block, not only
    # excluded-tool output — that gate made external folds inert for gateway
    # traffic (/v1/compress), where tool names are the caller's own. The smaller
    # output wins, so a provider can only improve on the built-in folds.
    from headroom.transforms.lossless_provider import set_lossless_provider

    block = _grep_block()
    baseline, _, _ = _compress(block, lossless=True)
    better = "SHORTER-THAN-ANY-BUILTIN-FOLD\n"
    assert len(better) < len(baseline)

    try:
        set_lossless_provider(lambda content: (better, "plugin"))
        out, was, tr = _compress(block + "\n", lossless=True)  # fresh cache key
        assert was is True
        assert out == better
        assert tr == ["router:tool_result:lossless_plugin"]

        # A provider that loses to the built-in fold is ignored, not adopted.
        set_lossless_provider(lambda content: (content + "padding" * 100, "plugin"))
        out2, _, tr2 = _compress(block + "\n\n", lossless=True)
        assert tr2 == ["router:tool_result:lossless_search"]
        assert len(out2) < len(block)

        # A raising provider must not break routing.
        set_lossless_provider(_raise)
        out3, was3, _ = _compress(block + "\n\n\n", lossless=True)
        assert was3 is True
        assert len(out3) < len(block)
    finally:
        set_lossless_provider(None)


def _raise(content: str):
    raise RuntimeError("broken provider")


def test_lossless_mode_non_foldable_is_lossless_noop_not_ratio_too_high():
    # In lossless-only mode, code with no byte-lossless fold is left verbatim.
    # That is NOT a rejected compression, so it must not be bucketed as
    # ratio_too_high (which means "a lossy attempt didn't shrink enough").
    router = ContentRouter(ContentRouterConfig(lossless=True))
    code = "\n".join(f"    x{i} = compute_value({i}, offset={i * 3})" for i in range(60)) + "\n"
    rc: dict = {}
    out, was = router._compress_block_content(
        code,
        hash(code),
        "",
        1.0,
        1.0,
        None,
        [],
        rc,
        [],
        "tool_result",
        "tool",
        True,
    )
    assert was is False
    assert rc.get("lossless_noop", 0) >= 1
    assert rc.get("ratio_too_high", 0) == 0


# ── Hostile / third-party provider input ──────────────────────────────────────
# The provider is arbitrary out-of-tree code called on the request path, and
# `_lossless_first`'s caller (`TransformPipeline.apply`) re-raises. Nothing a
# provider returns may fail a request or silently destroy a block.

_SHORT = "SHORTER-THAN-ANY-BUILTIN-FOLD\n"

_MALFORMED_RESULTS = [
    (_SHORT, "plugin", "extra"),  # 3-tuple -> would raise on unpack
    _SHORT,  # bare string -> would unpack into 2 chars, or raise
    (_SHORT, 42),  # non-str kind
    (42, "plugin"),  # non-str candidate
    {"compacted": _SHORT},  # non-tuple entirely
    17,  # not even iterable
]


def test_malformed_provider_result_is_ignored_not_raised():
    block = _grep_block()
    for i, bad in enumerate(_MALFORMED_RESULTS):
        try:
            set_lossless_provider(lambda content, bad=bad: bad)
            # Distinct content per shape so nothing is served from a cache/memo.
            out, was, tr = _compress(block + "\n" * (i + 1), lossless=True)
        finally:
            set_lossless_provider(None)
        # No exception, and the built-in fold is still the answer.
        assert was is True, bad
        assert tr == ["router:tool_result:lossless_search"], bad
        assert len(out) < len(block), bad


def test_empty_provider_result_is_rejected():
    # "" beats every candidate on length, so an unchecked empty result would
    # win and silently delete the block's content.
    block = _grep_block()
    for i, blank in enumerate(("", "   ", "\n\t \n")):
        try:
            set_lossless_provider(lambda content, blank=blank: (blank, "plugin"))
            out, was, tr = _compress(block + "x" * (i + 1) + "\n", lossless=True)
        finally:
            set_lossless_provider(None)
        assert was is True, repr(blank)
        assert tr == ["router:tool_result:lossless_search"], repr(blank)
        assert out.strip(), repr(blank)


def test_provider_is_never_offered_diff_content():
    # Diff folding is subtractive with no inverse check and a reflowed hunk
    # breaks `git apply`, so a third-party fold must never see a diff — the
    # same reason the built-in "diff" fold is strategy-gated.
    diff = (
        "diff --git a/x b/x\nindex 1111111..2222222 100644\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    )
    router = ContentRouter(ContentRouterConfig(lossless=True))
    baseline_diff = router._lossless_first(diff, CompressionStrategy.DIFF)
    baseline_pass = router._lossless_first(diff, CompressionStrategy.PASSTHROUGH)

    seen: list[str] = []

    def recording(content: str):
        seen.append(content)
        return (_SHORT, "plugin")

    try:
        set_lossless_provider(recording)
        fresh = ContentRouter(ContentRouterConfig(lossless=True))
        # Both the DIFF strategy and diff-*shaped* content under another
        # strategy must bypass the provider entirely.
        assert fresh._lossless_first(diff, CompressionStrategy.DIFF) == baseline_diff
        assert fresh._lossless_first(diff, CompressionStrategy.PASSTHROUGH) == baseline_pass
        assert seen == []
    finally:
        set_lossless_provider(None)


def test_provider_kind_label_is_sanitized():
    # `kind` reaches transforms_applied and the per-strategy metric/timing keys
    # (Prometheus label values), so an unbounded caller-controlled string is a
    # cardinality bomb. Only ^[a-z0-9_]{1,32}$ survives.
    block = _grep_block()
    # "log\n" is the subtle one: Python's `$` matches before a trailing newline,
    # so a `re.match`-based check would let a newline into a metric label.
    bogus = ["UPPER", "has space", "punct!", "x" * 200, "", "kebab-case", "log\n"]
    for i, kind in enumerate(bogus):
        try:
            set_lossless_provider(lambda content, kind=kind: (_SHORT, kind))
            out, was, tr = _compress(block + "\n" * (i + 1), lossless=True)
        finally:
            set_lossless_provider(None)
        assert was is True, kind
        assert out == _SHORT, kind
        assert tr == ["router:tool_result:lossless_provider"], kind

    # A clean label is preserved verbatim.
    try:
        set_lossless_provider(lambda content: (_SHORT, "log_fold_2"))
        _out, _was, tr = _compress(block + "\t", lossless=True)
        assert tr == ["router:tool_result:lossless_log_fold_2"]
    finally:
        set_lossless_provider(None)


def test_lossless_mode_verifier_gates_the_provider_candidate():
    # In --lossless mode STAGE 0's output IS the answer: no CCR marker, no
    # retrieval. The built-in folds self-verify; a provider only does if the
    # operator registered a verifier.
    block = _grep_block()
    provider = lambda content: (_SHORT, "plugin")  # noqa: E731

    def _run(verifier, suffix):
        try:
            set_lossless_provider(provider, verifier=verifier)
            return _compress(block + suffix, lossless=True)
        finally:
            set_lossless_provider(None)

    # No verifier -> documented trust, unchanged behaviour.
    _out, was, tr = _run(None, "\n")
    assert was is True
    assert tr == ["router:tool_result:lossless_plugin"]

    # Verifier says yes -> adopted.
    out, was, tr = _run(lambda original, compacted: True, "\n\n")
    assert out == _SHORT
    assert tr == ["router:tool_result:lossless_plugin"]

    # Verifier says no -> rejected, built-in fold stands.
    out, was, tr = _run(lambda original, compacted: False, "\n\n\n")
    assert tr == ["router:tool_result:lossless_search"]
    assert len(out) < len(block)

    # Verifier raises -> "unverified" -> rejected, never fatal.
    def _boom(original, compacted):
        raise RuntimeError("verifier exploded")

    out, was, tr = _run(_boom, "\n\n\n\n")
    assert tr == ["router:tool_result:lossless_search"]
    assert len(out) < len(block)


def test_memo_does_not_go_stale_when_the_provider_changes():
    # The STAGE 0 memo caches a value that depends on the registered provider,
    # so its key includes the registration generation. Without that, a provider
    # registered (or cleared) after a block was already folded is ignored for
    # that exact block forever — invisible in production (extensions register at
    # startup) but wrong, and a trap for tests and any hot-reload path.
    from headroom.transforms.content_router import CompressionStrategy

    router = ContentRouter(ContentRouterConfig(lossless=True))
    block = _grep_block()
    tiny = "TINY\n"

    try:
        set_lossless_provider(None)
        baseline, baseline_label = router._lossless_first(block, CompressionStrategy.SEARCH)
        assert baseline_label == "lossless_search"

        # Register AFTER the block was already folded once.
        set_lossless_provider(lambda content: (tiny, "plugin"))
        out, label = router._lossless_first(block, CompressionStrategy.SEARCH)
        assert out == tiny, "provider registered after first fold was ignored (stale memo)"
        assert label == "lossless_plugin"

        # Clearing it must take effect too.
        set_lossless_provider(None)
        out, label = router._lossless_first(block, CompressionStrategy.SEARCH)
        assert out == baseline, "cleared provider still served from the memo"
        assert label == "lossless_search"
    finally:
        set_lossless_provider(None)


def test_clearing_the_provider_also_clears_the_verifier():
    # A verifier with no provider is dead state that would silently start
    # gating the *next* provider someone registers.
    try:
        set_lossless_provider(lambda content: None, verifier=lambda o, c: True)
        assert get_lossless_verifier() is not None
        set_lossless_provider(None)
        assert get_lossless_provider() is None
        assert get_lossless_verifier() is None
    finally:
        set_lossless_provider(None)


def test_provider_is_called_once_per_block_not_twice():
    # `_has_lossless_fold` (the small-block admission probe) and STAGE 0 both
    # run `_lossless_first` over the same block, under *different* strategies.
    # Memoizing the provider's answer by content collapses that to one call
    # into third-party code.
    calls: list[str] = []

    def counting(content: str):
        calls.append(content)
        return None

    small = "\n".join(f"pkg/mod/long_filename.py:{n}:value = {n}" for n in range(1, 8)) + "\n"
    try:
        set_lossless_provider(counting)
        router = ContentRouter(ContentRouterConfig(lossless=True))
        assert router._has_lossless_fold(small) is True
        assert len(calls) == 1
        out, was = router._compress_block_content(
            small,
            hash(small),
            "",
            1.0,
            1.0,
            None,
            [],
            {},
            [],
            "tool_result",
            "tool",
            True,
        )
        assert was is True
        assert len(calls) == 1
    finally:
        set_lossless_provider(None)
