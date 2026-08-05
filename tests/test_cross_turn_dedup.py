"""Cross-turn dedup: cache-safety (prefix-monotonicity) + accuracy (info-preserving)."""

import re

from headroom.transforms.cross_turn_dedup import (
    DedupBlock,
    _num_and_key,
    dedup_blocks,
    is_prefix_monotonic,
)

# Compact fold pointer: ``[↑<N>L same as msg <ref>[ <±delta>L]: '<anchor>']`` —
# span length + referenced msg + optional line-number offset + a truncated
# first-line anchor (no explicit line range; recovery locates the span by anchor).
_FOLD_RE = re.compile(r"\[↑(\d+)L same as msg (\d+)(?: ([+-]\d+)L)?: '([^']*)'\]")


def _blk(text, turn, protected=False):
    return DedupBlock(text=text, turn=turn, protected=protected)


def _code(prefix, n):
    # A realistic, non-trivial multi-line source span.
    return "\n".join(
        f"{prefix}    result_{i} = compute_overdraft(business_id={i}, amount={i * 100})"
        for i in range(n)
    )


def _reconstruct(orig_blocks, out_blocks):
    """Replace each fold pointer with the referenced msg's original lines and assert
    it reproduces the original block — proves references are faithful & in-context.

    The compact pointer names the ref msg + span length + a first-line anchor (not
    an explicit line range), so recovery locates the span by its anchor in the
    referenced message and takes ``<N>`` lines. (All test spans are unnumbered, so
    delta is always absent; a non-zero delta would renumber on the way out.)"""
    by_turn = {b.turn: b.text.split("\n") for b in orig_blocks}

    def _content(line):
        return _num_and_key(line)[2].strip()

    for orig, out in zip(orig_blocks, out_blocks):
        if orig.protected:
            assert out.text == orig.text
            continue
        rebuilt = []
        for line in out.text.split("\n"):
            m = _FOLD_RE.search(line)
            if m and line.lstrip().startswith("[↑"):
                assert m.group(3) is None, "unexpected delta for an unnumbered span"
                n, ref, anchor = int(m.group(1)), int(m.group(2)), m.group(4)
                assert ref < orig.turn, "reference must point to an EARLIER msg"
                core = anchor[:-3] if anchor.endswith("...") else anchor
                ref_lines = by_turn[ref]
                idx = next(i for i, rl in enumerate(ref_lines) if _content(rl).startswith(core))
                rebuilt.extend(ref_lines[idx : idx + n])
            else:
                rebuilt.append(line)
        assert "\n".join(rebuilt) == orig.text, f"turn {orig.turn} not faithfully reconstructable"


def test_verbatim_reread_is_folded_keep_earliest():
    span = _code("", 8)
    blocks = [_blk(f"cat merge.py\n{span}\ntail", 1), _blk(f"sed run\n{span}\nmore", 5)]
    out, stats = dedup_blocks(blocks)
    assert out[0].text == blocks[0].text  # earliest untouched
    assert "[↑" in out[1].text  # later occurrence folded
    assert stats["spans_folded"] == 1
    _reconstruct(blocks, out)


def test_cache_safety_prefix_monotonic():
    span = _code("x", 10)
    blocks = [
        _blk("intro line one\nintro line two\n" + span, 1),
        _blk("unrelated diff output\n@@ -1 +1 @@\n-a\n+b", 2),
        _blk("here again:\n" + span, 3),
        _blk("and once more\n" + span + "\ntrailer", 4),
    ]
    assert is_prefix_monotonic(blocks) is True


def test_below_min_lines_not_folded():
    span = _code("", 2)  # below min_lines (3)
    blocks = [_blk(span, 1), _blk(span, 2)]
    out, stats = dedup_blocks(blocks)
    assert stats["spans_folded"] == 0
    assert out[1].text == blocks[1].text


def test_trivial_repeated_lines_not_folded():
    junk = "\n".join(["}"] * 20)  # trivial lines only
    blocks = [_blk(junk, 1), _blk(junk, 2)]
    out, stats = dedup_blocks(blocks)
    assert stats["spans_folded"] == 0


def test_deterministic():
    span = _code("z", 9)
    blocks = [_blk(span, 1), _blk("mid\n" + span, 2), _blk(span, 3)]
    a, _ = dedup_blocks(blocks)
    b, _ = dedup_blocks(blocks)
    assert [x.text for x in a] == [x.text for x in b]


def test_protected_block_not_rewritten_but_is_reference_target():
    span = _code("", 8)
    blocks = [
        _blk(span, 1, protected=True),  # cache_control block — never rewritten
        _blk("later:\n" + span, 2),  # should still fold against the protected one
    ]
    out, stats = dedup_blocks(blocks)
    assert out[0].text == blocks[0].text
    assert "[↑" in out[1].text
    _reconstruct(blocks, out)


def test_info_preserving_reconstruction_multiref():
    s1 = _code("a", 7)
    s2 = _code("b", 8)
    blocks = [
        _blk("h1\n" + s1, 1),
        _blk("h2\n" + s2, 2),
        _blk("mix\n" + s1 + "\n---\n" + s2, 3),  # two folds in one block
    ]
    out, stats = dedup_blocks(blocks)
    assert stats["spans_folded"] == 2
    _reconstruct(blocks, out)


# --------------------------------------------------------------------------
# Numbered (renumber-fold) path: a leading line-number lets the same content
# fold across a uniform shift, with the offset carried in the pointer so the
# original bytes recover as ``str(int(number) + delta)``. That recovery is
# byte-exact ONLY for UNPADDED numbers: a leading-zero prefix (a timestamped
# log row, ``08:00:01``) loses its pad on renumber (``int("08") + 1`` -> ``"9"``,
# not ``"09"``), so it must not fold under a delta. The helper below renumbers
# exactly as the module documents, so the round-trip assertion is faithful.
# --------------------------------------------------------------------------
def _reconstruct_numbered(orig_blocks, out_blocks):
    """Like ``_reconstruct`` but honours a non-zero delta: recover each folded
    span from the referenced msg, renumbering leading line numbers by the stated
    offset (``str(int(number) + delta) + key``), then assert byte-exact bytes."""
    by_turn = {b.turn: b.text.split("\n") for b in orig_blocks}

    def _content(line):
        return _num_and_key(line)[2].strip()

    for orig, out in zip(orig_blocks, out_blocks):
        if orig.protected:
            assert out.text == orig.text
            continue
        rebuilt = []
        for line in out.text.split("\n"):
            m = _FOLD_RE.search(line)
            if m and line.lstrip().startswith("[↑"):
                n, ref = int(m.group(1)), int(m.group(2))
                delta = int(m.group(3)) if m.group(3) else 0
                anchor = m.group(4)
                assert ref < orig.turn, "reference must point to an EARLIER msg"
                core = anchor[:-3] if anchor.endswith("...") else anchor
                ref_lines = by_turn[ref]
                idx = next(i for i, rl in enumerate(ref_lines) if _content(rl).startswith(core))
                for rl in ref_lines[idx : idx + n]:
                    num, key, _c = _num_and_key(rl)  # key keeps the separator
                    rebuilt.append(f"{num + delta}{key}" if (num is not None and delta) else rl)
            else:
                rebuilt.append(line)
        assert "\n".join(rebuilt) == orig.text, f"turn {orig.turn} not faithfully reconstructable"


def _log(prefixes):
    # Timestamped probe rows: identical content, distinct LEADING-ZERO hour.
    return "\n".join(f"{p}:00:01 probe ok latency=12ms region=us-east-1" for p in prefixes)


def test_zero_padded_prefix_not_folded_lossily():
    # A leading-zero numeric prefix shifted by a uniform +1 looks like a renumber
    # (keys match, delta is uniform), but recovery via int(number)+delta drops the
    # pad, so the fold would NOT round-trip. It must be left verbatim instead.
    blocks = [
        _blk("probe window A\n" + _log(["06", "07", "08"]) + "\ndone A", 1),
        _blk("probe window B\n" + _log(["07", "08", "09"]) + "\ndone B", 5),
    ]
    out, stats = dedup_blocks(blocks)
    assert stats["spans_folded"] == 0
    assert out[1].text == blocks[1].text and "[↑" not in out[1].text
    _reconstruct_numbered(blocks, out)  # trivially exact: nothing folded


def test_unpadded_renumber_still_folds_and_recovers_exactly():
    # The intended feature: an UNPADDED grep -n / sed -n read re-displayed after an
    # edit shifted every line number by a constant still folds, and the pointer's
    # delta recovers the exact numbered bytes. The fix must not regress this.
    span = [
        "    result_0 = compute_overdraft(business_id=0, amount=0)",
        "    result_1 = compute_overdraft(business_id=1, amount=100)",
        "    result_2 = compute_overdraft(business_id=2, amount=200)",
    ]
    b1 = "read A\n" + "\n".join(f"{10 + i}:{s}" for i, s in enumerate(span)) + "\nend A"
    b2 = "read B\n" + "\n".join(f"{15 + i}:{s}" for i, s in enumerate(span)) + "\nend B"
    blocks = [_blk(b1, 1), _blk(b2, 3)]
    out, stats = dedup_blocks(blocks)
    assert stats["spans_folded"] == 1
    assert "+5L" in out[1].text  # uniform +5 shift carried in the pointer
    _reconstruct_numbered(blocks, out)


def test_padded_content_exact_redisplay_still_folds():
    # Surgical-scope guard: the fix only blocks the LOSSY renumbered fold. The same
    # zero-padded rows re-displayed VERBATIM (delta 0) are still a byte-identical
    # fold and must continue to compress.
    log = _log(["06", "07", "08"])
    blocks = [
        _blk("probe window A\n" + log + "\ndone A", 1),
        _blk("re-check\n" + log + "\ntail", 4),
    ]
    out, stats = dedup_blocks(blocks)
    assert stats["spans_folded"] == 1
    m = _FOLD_RE.search(out[1].text)
    assert m is not None and m.group(3) is None  # folded, delta-free (byte-identical)
    _reconstruct_numbered(blocks, out)


# --------------------------------------------------------------------------
# Integration: full router.apply() path (content-block tool_result format)
# --------------------------------------------------------------------------
def _mk_tok():
    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer

    return Tokenizer(OpenAIProvider().get_token_counter("gpt-4o"), "gpt-4o")


def _toolmsg(text, tid):
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}],
    }


def _apply_fresh(messages):
    # Fresh router per call: tests the pure-function (prefix-monotonic) property,
    # not cross-call cache state.
    import copy

    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    r = ContentRouter(ContentRouterConfig(lossless=True, enable_cross_turn_dedup=True))
    return r.apply(copy.deepcopy(messages), _mk_tok()).messages


def test_apply_dedups_reread_and_keeps_prefix_stable():
    span = "\n".join(
        f"    result_{i} = compute_overdraft(business_id={i}, amount={i * 100})" for i in range(12)
    )
    m1 = [
        {"role": "user", "content": "fix the overdraft bug"},
        {"role": "assistant", "content": "cat merge.py"},
        _toolmsg(f"$ cat merge.py\n{span}\n# end", "t1"),
    ]
    m2 = m1 + [
        {"role": "assistant", "content": "sed -n range"},
        _toolmsg(f"$ sed -n 1,20p merge.py\n{span}\n# more", "t2"),
    ]
    out1 = _apply_fresh(m1)
    out2 = _apply_fresh(m2)

    # Dedup fired on the later re-read (turn t2), earliest (t1) untouched.
    later = out2[-1]["content"][0]["content"]
    earlier = out2[2]["content"][0]["content"]
    assert "[↑" in later
    assert "[↑" not in earlier and span in earlier

    # CACHE-SAFETY at the router level: appending turn t2 did NOT change any
    # earlier message's emitted bytes → the prompt-cache prefix is stable.
    def _tool_texts(msgs):
        return [
            b["content"]
            for m in msgs
            if isinstance(m.get("content"), list)
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"
        ]

    assert _tool_texts(out2)[:1] == _tool_texts(out1)  # t1 block byte-identical


def test_apply_no_dedup_when_flag_off():
    span = "\n".join(f"    v_{i} = f({i})" for i in range(12))
    import copy

    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    msgs = [
        _toolmsg(f"a\n{span}", "t1"),
        _toolmsg(f"b\n{span}", "t2"),
    ]
    r = ContentRouter(ContentRouterConfig(lossless=True, enable_cross_turn_dedup=False))
    out = r.apply(copy.deepcopy(msgs), _mk_tok()).messages
    joined = "".join(b["content"] for m in out for b in m["content"] if isinstance(b, dict))
    assert "[↑" not in joined


def test_apply_dedup_runs_in_ccr_mode_too():
    # Dedup is no longer gated to lossless mode: with lossless=False (CCR) and
    # the flag on, an exact re-read still folds to an in-context pointer.
    span = "\n".join(f"    total_{i} = reconcile(entry_id={i}, ledger=book_{i})" for i in range(12))
    import copy

    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    msgs = [
        _toolmsg(f"$ cat ledger.py\n{span}\n# eof", "t1"),
        {"role": "assistant", "content": "re-check"},
        _toolmsg(f"$ cat ledger.py\n{span}\n# eof", "t2"),  # exact re-run
    ]
    r = ContentRouter(ContentRouterConfig(lossless=False, enable_cross_turn_dedup=True))
    out = r.apply(copy.deepcopy(msgs), _mk_tok()).messages
    joined = "".join(
        b["content"]
        for m in out
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict)
    )
    assert "[↑" in joined  # dedup fired despite lossless=False


# --------------------------------------------------------------------------
# No dangling reference: dedup folds only against content present in the array
# it processes (it runs last, over the final sent messages). If compaction
# already removed the original, there is nothing earlier to reference → verbatim.
# --------------------------------------------------------------------------
def test_no_fold_when_original_absent_fallback():
    span = _code("", 8)
    # Only the LATER read survives; its original was compacted out of the array.
    blocks = [_blk("unrelated log\n" + _code("z", 8), 1), _blk(f"sed\n{span}\nmore", 5)]
    out, stats = dedup_blocks(blocks)
    assert stats["spans_folded"] == 0
    assert out[1].text == blocks[1].text and "[↑" not in out[1].text


# --------------------------------------------------------------------------
# Shape-agnostic CODE-READ coverage: a file read gets deduped wherever it lands
# — role:tool, role:function, or a text-harness role:user string — keyed off the
# read OUTCOME, never ordinary user prose.
# --------------------------------------------------------------------------
def _dedup_only(messages):
    """Run ONLY the cross-turn dedup pass (no per-block compression) on a raw
    message array, isolating extraction + fold across message shapes."""
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    r = ContentRouter(ContentRouterConfig(lossless=True, enable_cross_turn_dedup=True))
    return r._cross_turn_dedup_messages(messages, 0, [], None)


def _readspan():
    return "\n".join(f"    result_{i} = compute_overdraft(business_id={i})" for i in range(10))


def test_dedup_folds_user_string_read_observation():
    # Text-harness shape: the read output arrives as a role:user STRING after an
    # assistant fenced `cat`. The later identical read folds; earliest stays intact.
    span = _readspan()
    asst = {"role": "assistant", "content": "```bash\ncat report.py\n```"}
    msgs = [
        asst,
        {"role": "user", "content": span},  # read #1 (reference target)
        asst,
        {"role": "user", "content": span},  # read #2 (duplicate) -> folds
    ]
    out = _dedup_only(msgs)
    assert "[↑" in out[3]["content"]
    assert out[1]["content"] == span


def test_dedup_does_not_fold_plain_user_prose():
    # A duplicated ORDINARY user message (no preceding read command) must stay
    # verbatim — user intent is never folded.
    prose = "\n".join(f"please also make sure case {i} is handled carefully" for i in range(10))
    msgs = [
        {"role": "user", "content": prose},
        {"role": "assistant", "content": "understood"},
        {"role": "user", "content": prose},  # duplicate prose -> must NOT fold
    ]
    out = _dedup_only(msgs)
    assert "[↑" not in out[2]["content"] and out[2]["content"] == prose


def test_dedup_folds_role_function_output():
    # Legacy OpenAI role:function tool output — same operation, different label.
    span = _readspan()
    msgs = [
        {"role": "function", "name": "read_file", "content": span},
        {"role": "assistant", "content": "let me re-check"},
        {"role": "function", "name": "read_file", "content": span},  # dup -> folds
    ]
    out = _dedup_only(msgs)
    assert "[↑" in out[2]["content"]
