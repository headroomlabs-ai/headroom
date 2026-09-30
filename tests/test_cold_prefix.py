"""Cold-prefix recompaction: the cross-turn dedup fold must be path-aware.

``cold_recompact_messages`` builds its own lossless ContentRouter with
``enable_cross_turn_dedup=True``. The fold rewrites repeated tool-output spans
to bare ``[↑NL same as msg M]`` in-context pointers — unresolvable on paths
where no CCR retrieval tool can be injected and the client never shows the
model numbered messages (OpenAI chat-completions streaming, wrap copilot).
The recompaction therefore takes ``cross_turn_dedup_recoverable`` and forwards
it to the router gate: unrecoverable paths keep the bytes verbatim, the
Anthropic cache-mode caller (default True) keeps folding.
"""

from headroom.transforms.cold_prefix import cold_recompact_messages


def _mk_tok():
    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer

    return Tokenizer(OpenAIProvider().get_token_counter("gpt-4o"), "gpt-4o")


def _toolmsg(text, tid):
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}],
    }


def _conversation():
    span = "\n".join(f"    result_{i} = compute_overdraft(business_id={i})" for i in range(12))
    return [
        {"role": "user", "content": "fix the overdraft bug"},
        {"role": "assistant", "content": "cat merge.py"},
        _toolmsg(f"$ cat merge.py\n{span}\n# end", "t1"),
        {"role": "assistant", "content": "sed -n range"},
        _toolmsg(f"$ sed -n 1,20p merge.py\n{span}\n# more", "t2"),
    ]


def test_cold_recompact_folds_by_default():
    # Anthropic cache-mode path (the only caller today): unchanged — the
    # repeated span still folds to an in-context pointer.
    msgs = _conversation()
    out, transforms = cold_recompact_messages(msgs, tokenizer=_mk_tok())
    later = out[-1]["content"][0]["content"]
    assert "[↑" in later
    assert any("cross_turn_dedup" in t for t in transforms)


def test_cold_recompact_unrecoverable_path_keeps_verbatim_bytes():
    # Unresolvable-pointer path (OpenAI chat streaming shape): the fold is
    # skipped, the repeated span stays byte-verbatim, and no pointer is
    # emitted — while the recompaction itself still runs (message count and
    # order unchanged).
    msgs = _conversation()
    out, transforms = cold_recompact_messages(
        msgs, tokenizer=_mk_tok(), cross_turn_dedup_recoverable=False
    )
    later = out[-1]["content"][0]["content"]
    assert "[↑" not in later
    assert (
        later
        == "$ sed -n 1,20p merge.py\n"
        + "\n".join(f"    result_{i} = compute_overdraft(business_id={i})" for i in range(12))
        + "\n# more"
    )
    assert not any("cross_turn_dedup" in t for t in transforms)
    assert len(out) == len(msgs)
