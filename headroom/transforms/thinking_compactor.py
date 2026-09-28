"""Convert prior-turn extended-thinking blocks into Kompressed ``text`` blocks.

On Claude 4.6+ models, prior-turn thinking is re-sent as input and **billed**
(verified live: opus-4-6 +995 tok/block, sonnet-4-6 +688; pre-4.6 models strip
it server-side, so this transform is a no-op there — gate on model generation at
the call site). Two findings dictate the mechanism:

1. **Editing a thinking block in place is futile** — Anthropic pins the original
   via the block ``signature`` and re-expands it server-side, ignoring whatever
   text you send (verified: 835==835). The only ways to actually shrink thinking
   are to *drop* the block or *convert it to a plain ``text`` block* (no
   signature → the shorter text is billed as-is; verified 716<835). This
   transform does the latter, running each block's text through Kompress.

2. **Cache safety comes from determinism, not from "only touch the delta".** The
   client re-sends the *original* thinking every turn, but the prompt cache holds
   the *compacted* form we forwarded last turn. So we map original→compacted
   deterministically (memoized by content hash) every turn — the forwarded prefix
   is then byte-stable and the cache still hits. The last ``keep_last_turns``
   assistant turns keep their thinking intact (the active reasoning the model
   needs); a turn aging out of that window is the only byte change, a bounded
   recent-region re-write.

Flag-gated at the call site (``HEADROOM_THINKING_COMPACT``). Fail-open: any
Kompress error leaves the original block untouched. ``keep_last_turns`` is the
quality knob.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

log = logging.getLogger(__name__)

# original-thinking-hash -> compacted text. Deterministic memo so the same
# thinking always yields the same bytes (cache stability), even if Kompress is
# nondeterministic. Bounded; ONNX Kompress is deterministic so eviction+recompute
# is byte-identical anyway. ponytail: crude LRU cap, fine given determinism.
_COMPACT_CACHE: OrderedDict[str, str] = OrderedDict()
_CACHE_CAP = 8192

# Prefix on the emitted text block so the compaction is legible to the model
# (and greppable in logs). Kept short; the token cost is negligible vs the block.
_MARKER = "[prior reasoning, compressed]"


def bills_prior_thinking(model: str) -> bool:
    """True if ``model`` re-bills prior-turn thinking as input (so compaction pays).

    Claude 4.6+ (and the 5 family) keep prior-turn thinking in context and bill it;
    pre-4.6 (sonnet-4-5, haiku-4-5, 3.x) strip it server-side. Verified live:
    opus-4-6/sonnet-4-6 bill, sonnet-4-5/haiku-4-5 strip. **Conservative** — returns
    False unless the version is confidently >= 4.6, because compacting on a stripping
    model would turn free (stripped) thinking into billed text. (Opus 4.5 reportedly
    bills too, but is excluded here pending verification — costs only missed savings.)
    """
    nums: list[int] = []
    for part in model.lower().split("-"):
        if part.isdigit():
            nums.append(int(part))
        elif nums:
            break  # version digits are contiguous; stop at the family/date boundary
    if not nums:
        return False
    major = nums[0]
    minor = nums[1] if len(nums) > 1 else 0
    return major >= 5 or (major, minor) >= (4, 6)


def _memo_compact(text: str, kompress: Any) -> str | None:
    """Deterministically compact ``text`` via Kompress; None if no gain/failure."""
    key = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
    cached = _COMPACT_CACHE.get(key)
    if cached is not None:
        _COMPACT_CACHE.move_to_end(key)
        return cached
    try:
        # allow_download=False: never block the request thread on a cold model
        # (mirrors the proxy convention in ContentRouter._get_kompress callers).
        result = kompress.compress(text, allow_download=False)
        compacted = result.compressed
    except Exception as e:  # fail OPEN — never break the proxy on a bad compressor
        log.warning("thinking compaction failed (%s); leaving block untouched", e)
        return None
    if not isinstance(compacted, str) or not compacted:
        return None
    _COMPACT_CACHE[key] = compacted
    _COMPACT_CACHE.move_to_end(key)
    while len(_COMPACT_CACHE) > _CACHE_CAP:
        _COMPACT_CACHE.popitem(last=False)
    return compacted


def compact_thinking_to_text(
    messages: list[dict[str, Any]],
    *,
    kompress: Any,
    keep_last_turns: int = 1,
    min_words: int = 40,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Replace ``thinking`` blocks with Kompressed ``text`` blocks.

    Every assistant turn except the last ``keep_last_turns`` (which keep their
    thinking verbatim) has each of its ``thinking`` blocks converted to a
    ``text`` block holding the Kompressed summary. Deterministic per content, so
    the forwarded prefix stays cache-stable across turns. Never mutates the input
    list or its message dicts in place. Never raises.

    Args:
        messages: provider-native Anthropic messages.
        kompress: an object with ``compress(text, allow_download=False) ->
            KompressResult`` (local ``KompressCompressor`` or remote
            ``RemoteKompressCompressor``; caller picks per HEADROOM_KOMPRESS_ENDPOINT).
        keep_last_turns: number of most-recent assistant turns whose thinking is
            left intact (the active reasoning). 0 compacts everything.
        min_words: thinking blocks below this word count are left as-is (not worth
            a Kompress call).

    Returns:
        (new_messages, stats) where stats has ``turns_compacted``, ``blocks``,
        ``words_before``, ``words_after``.
    """
    stats = {"turns_compacted": 0, "blocks": 0, "words_before": 0, "words_after": 0}
    if kompress is None:
        return messages, stats

    asst_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    keep = set(asst_indices[-keep_last_turns:]) if keep_last_turns > 0 else set()

    out: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        content = m.get("content")
        if (
            m.get("role") != "assistant"
            or i in keep
            or not isinstance(content, list)
            or not any(isinstance(b, dict) and b.get("type") == "thinking" for b in content)
        ):
            out.append(m)
            continue

        new_content: list[Any] = []
        turn_compacted = False
        for block in content:
            if not (isinstance(block, dict) and block.get("type") == "thinking"):
                new_content.append(block)
                continue
            text = block.get("thinking", "")
            words = len(text.split())
            if words < min_words:
                new_content.append(block)
                continue
            compacted = _memo_compact(text, kompress)
            # Skip if compaction failed or didn't actually shrink the block.
            if compacted is None or len(compacted.split()) >= words:
                new_content.append(block)
                continue
            text_block: dict[str, Any] = {"type": "text", "text": f"{_MARKER} {compacted}"}
            # Preserve a cache breakpoint if one happened to sit on the thinking
            # block (rare — breakpoints usually sit on the last block of a message),
            # so we never silently drop a cache_control marker.
            if "cache_control" in block:
                text_block["cache_control"] = block["cache_control"]
            new_content.append(text_block)
            turn_compacted = True
            stats["blocks"] += 1
            stats["words_before"] += words
            stats["words_after"] += len(compacted.split())

        if turn_compacted:
            stats["turns_compacted"] += 1
            nm = dict(m)
            nm["content"] = new_content
            out.append(nm)
        else:
            out.append(m)

    return out, stats


def _compact_think_spans(
    content: str, kompress: Any, min_words: int, *, drop: bool = False
) -> tuple[str, int, int, int]:
    """Compact each ``<think>…</think>`` span (GLM / DeepSeek-R1 inline reasoning).

    ``drop=False`` (warm) Kompresses the inner text; ``drop=True`` (cold hook)
    removes the whole span. Returns (new_content, blocks, words_before,
    words_after). String-scan (the tags are a fixed literal delimiter, not a
    heuristic). Leaves unmatched/short spans untouched.
    """
    open_tag, close_tag = "<think>", "</think>"
    blocks = wb = wa = 0
    parts: list[str] = []
    pos = 0
    while True:
        start = content.find(open_tag, pos)
        if start == -1:
            parts.append(content[pos:])
            break
        end = content.find(close_tag, start + len(open_tag))
        if end == -1:  # unterminated — leave the remainder as-is
            parts.append(content[pos:])
            break
        parts.append(content[pos:start])
        inner = content[start + len(open_tag) : end]
        words = len(inner.split())
        if words >= min_words and drop:
            # Cold hook: drop the span entirely (append nothing).
            blocks += 1
            wb += words
            pos = end + len(close_tag)
            continue
        new_inner = inner
        if words >= min_words:
            comp = _memo_compact(inner, kompress)
            if comp is not None and len(comp.split()) < words:
                new_inner = comp
                blocks += 1
                wb += words
                wa += len(comp.split())
        parts.append(f"{open_tag}{new_inner}{close_tag}")
        pos = end + len(close_tag)
    return "".join(parts), blocks, wb, wa


def compact_reasoning_openai_chat(
    messages: list[dict[str, Any]],
    *,
    kompress: Any,
    keep_last_turns: int = 1,
    min_words: int = 40,
    drop: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Compact plain-text reasoning in OpenAI-chat messages (Kimi / GLM / DeepSeek-R1).

    Unlike Anthropic thinking / OpenAI reasoning (encrypted handles), these models
    resend reasoning as PLAIN TEXT billed as input — so we can actually shrink it
    (verified: Kimi K2.7 resends ``reasoning_content`` at +1,558 input tok/block).
    Two shapes, both handled:

    * **Kimi:** the assistant message's ``reasoning_content`` field.
    * **GLM / DeepSeek-R1:** inline ``<think>…</think>`` in string content.

    ``drop=False`` (warm): Kompress the reasoning (deterministic → cache-stable).
    ``drop=True`` (cold-prefix hook): remove the reasoning outright — the full
    block, not just ~15% — safe because the cold turn re-caches from scratch.
    Shape-driven — no model gate; no-ops when no plain-text reasoning is present
    (OpenAI's encrypted models, Kimi k2.6). Keeps the last ``keep_last_turns``
    assistant turns intact (the active reasoning the model uses). Never raises.
    """
    stats = {"turns_compacted": 0, "blocks": 0, "words_before": 0, "words_after": 0}
    if kompress is None and not drop:  # drop needs no compressor
        return messages, stats
    asst_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    keep = set(asst_indices[-keep_last_turns:]) if keep_last_turns > 0 else set()

    out: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant" or i in keep:
            out.append(m)
            continue
        changed = False
        nm = dict(m)
        # (1) Kimi: reasoning_content field (plain text, no signature → editable)
        rc = m.get("reasoning_content")
        if isinstance(rc, str) and len(rc.split()) >= min_words:
            if drop:  # cold hook: drop the whole reasoning block
                nm["reasoning_content"] = ""
                changed = True
                stats["blocks"] += 1
                stats["words_before"] += len(rc.split())
            else:
                comp = _memo_compact(rc, kompress)
                if comp is not None and len(comp.split()) < len(rc.split()):
                    nm["reasoning_content"] = comp
                    changed = True
                    stats["blocks"] += 1
                    stats["words_before"] += len(rc.split())
                    stats["words_after"] += len(comp.split())
        # (2) GLM / DeepSeek-R1: inline <think>…</think> in string content
        c = m.get("content")
        if isinstance(c, str) and "<think>" in c:
            new_c, b, wb, wa = _compact_think_spans(c, kompress, min_words, drop=drop)
            if b:
                nm["content"] = new_c
                changed = True
                stats["blocks"] += b
                stats["words_before"] += wb
                stats["words_after"] += wa
        if changed:
            stats["turns_compacted"] += 1
            out.append(nm)
        else:
            out.append(m)
    return out, stats


def _demo() -> None:
    """Self-check: assert conversion, keep-last, tool_use preservation, determinism."""

    class _FakeKompress:
        calls = 0

        def compress(self, text: str, allow_download: bool = True) -> Any:  # noqa: ARG002
            _FakeKompress.calls += 1
            return type("R", (), {"compressed": "short summary"})()

    k = _FakeKompress()
    long = " ".join(["reasoning"] * 60)  # 60 words > min_words
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {  # old assistant turn: thinking + tool_use -> thinking should become text
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": long, "signature": "sig1"},
                {"type": "tool_use", "id": "t1", "name": "calc", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        },
        {  # last assistant turn: must be KEPT full
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": long, "signature": "sig2"}],
        },
    ]
    out, stats = compact_thinking_to_text(msgs, kompress=k, keep_last_turns=1)

    old = out[1]["content"]
    assert old[0] == {"type": "text", "text": f"{_MARKER} short summary"}, old[0]
    assert old[1]["type"] == "tool_use", "tool_use must be preserved"
    assert out[3]["content"][0]["type"] == "thinking", "last turn must stay thinking"
    assert stats["turns_compacted"] == 1 and stats["blocks"] == 1, stats
    assert msgs[1]["content"][0]["type"] == "thinking", "input must not be mutated"

    # determinism: same thinking text -> memoized, no second Kompress call
    calls_before = _FakeKompress.calls
    out2, _ = compact_thinking_to_text(msgs, kompress=k, keep_last_turns=1)
    assert out2[1]["content"][0] == old[0], "must be byte-identical across runs"
    assert _FakeKompress.calls == calls_before, "identical thinking must hit the memo"

    # keep_last_turns=0 compacts the final turn too
    out3, stats3 = compact_thinking_to_text(msgs, kompress=k, keep_last_turns=0)
    assert out3[3]["content"][0]["type"] == "text", "keep_last_turns=0 compacts all"
    assert stats3["turns_compacted"] == 2, stats3

    # model gate: 4.6+ / 5.x bill (compact); pre-4.6 strip (skip)
    assert bills_prior_thinking("claude-opus-4-6")
    assert bills_prior_thinking("claude-sonnet-4-6")
    assert bills_prior_thinking("claude-opus-4-8")
    assert bills_prior_thinking("claude-sonnet-5")
    assert not bills_prior_thinking("claude-sonnet-4-5-20250929")
    assert not bills_prior_thinking("claude-haiku-4-5-20251001")
    assert not bills_prior_thinking("claude-3-5-sonnet-20241022")

    # cache_control on a thinking block is carried to the emitted text block
    msgs_cc: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": long,
                    "signature": "s",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": "next"},
    ]
    out_cc, _ = compact_thinking_to_text(msgs_cc, kompress=k, keep_last_turns=0)
    assert out_cc[1]["content"][0].get("cache_control") == {"type": "ephemeral"}, out_cc[1]

    # --- OpenAI-chat shapes (Kimi reasoning_content + GLM inline <think>) ---
    oc_msgs: list[dict[str, Any]] = [
        {"role": "user", "content": "q"},
        # Kimi: reasoning_content field on an OLD assistant turn -> compacted
        {"role": "assistant", "content": "kimi answer", "reasoning_content": long},
        {"role": "user", "content": "q2"},
        # GLM: inline <think> on an OLD assistant turn -> inner compacted, wrapper kept
        {"role": "assistant", "content": f"<think>{long}</think> glm answer"},
        {"role": "user", "content": "q3"},
        # OpenAI encrypted case: no plain-text reasoning -> must be a no-op
        {"role": "assistant", "content": "plain answer, no reasoning"},
    ]
    oc_out, oc_stats = compact_reasoning_openai_chat(oc_msgs, kompress=k, keep_last_turns=1)
    assert oc_out[1]["reasoning_content"] == "short summary", oc_out[1]
    assert oc_out[3]["content"] == "<think>short summary</think> glm answer", oc_out[3]
    assert oc_out[5]["content"] == "plain answer, no reasoning", "no-reasoning must no-op"
    assert oc_stats["turns_compacted"] == 2 and oc_stats["blocks"] == 2, oc_stats
    # keep_last_turns protects the final assistant turn's reasoning
    oc_msgs2: list[dict[str, Any]] = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": long},
    ]
    oc_out2, _ = compact_reasoning_openai_chat(oc_msgs2, kompress=k, keep_last_turns=1)
    assert oc_out2[1]["reasoning_content"] == long, "last turn reasoning must be kept"

    # drop mode (cold-prefix hook): reasoning removed outright, no compressor needed
    oc_drop, ds = compact_reasoning_openai_chat(
        oc_msgs, kompress=None, keep_last_turns=1, drop=True
    )
    assert oc_drop[1]["reasoning_content"] == "", "Kimi reasoning must be dropped"
    assert oc_drop[3]["content"] == " glm answer", oc_drop[3]  # <think> span removed
    assert oc_drop[5]["content"] == "plain answer, no reasoning", "no-op on encrypted"
    assert ds["turns_compacted"] == 2 and ds["words_after"] == 0, ds

    print("thinking_compactor self-check OK")


if __name__ == "__main__":
    _demo()
