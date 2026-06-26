"""Tests that ContentRouter.apply() gates the role="tool" STRING path against
lossy-unrecoverable compression (#1307).

This exercises the REAL proxy path: ContentRouter.apply() is what the pipeline
runs, and a role="tool" string message routes through Pass-1 -> pending_tasks ->
self.compress() (Pass-2) -> result merge (Pass-3). The fix gates that merge so a
lossy summarizer (kompress/text/code) that did not store the original (no CCR
retrieve marker) cannot replace verbatim tool output.

The Kompress ML model is unavailable offline (it falls back to passthrough), so
the compression *result* is forced via monkeypatch — the seam is self.compress(),
the method apply() actually calls. apply() itself runs unmocked, so this proves
the live path, not an isolated unit (the gap PR #1363's apply()-direct tests had).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
)

# Realistic grep ground truth — comfortably > 50 word-tokens so it clears the
# min_tokens small-skip and reaches the compression path (otherwise apply()
# skips it before compress() is ever called). Every file:line is factual; a
# lossy reconstruction would fabricate paths the agent then acts on as fact.
GREP_OUTPUT = (
    'headroom/transforms/content_router.py:1375:            if role in ("tool", "assistant"):\n'
    'headroom/transforms/smart_crusher.py:1010:            if msg.get("role") == "tool":\n'
    'headroom/transforms/content_router.py:2653:            if role == "tool":\n'
    'headroom/proxy/handlers/anthropic.py:44:                elif block.get("type") == "tool_result":\n'
    'headroom/cache/prefix_tracker.py:88:            if message.get("role") == "tool":\n'
    'headroom/proxy/helpers.py:102:        if msg.get("role") == "tool":\n'
    "headroom/transforms/pipeline.py:132:        transforms.append(ContentRouter())\n"
    "headroom/transforms/kompress_compressor.py:1376:            result = self.compress(content)\n"
    'headroom/transforms/content_router.py:1483:                    compressor_name = "KompressCompressor"\n'
    'headroom/transforms/content_router.py:1568:                compressor_name = "KompressCompressor"\n'
    "headroom/transforms/content_router.py:2667:                bias = self._get_tool_bias(tool_name)\n"
    "headroom/transforms/content_router.py:3317:        result = self.compress(content, context=context)\n"
    "headroom/transforms/content_router.py:3331:                and not CCR_RETRIEVAL_MARKER_RE.search(result.compressed)\n"
    "headroom/proxy/handlers/openai.py:697:    def _compress_openai_responses_live_text_units(self)\n"
    "headroom/transforms/compression_units.py:204:    def compress_unit_with_router(self, unit)\n"
    "headroom/config.py:676:class TransformResult:  # messages, tokens_before, tokens_after\n"
)

LOSSY_SUMMARY = "grep found 8 matches across config and proxy modules (kompressed)."
CCR_MARKER_SUMMARY = "grep matches (kompressed) <<ccr:9f3a21>>"

LOSSY_STRATEGIES = [
    CompressionStrategy.KOMPRESS,
    CompressionStrategy.TEXT,
    CompressionStrategy.CODE_AWARE,
]
STRUCTURED_STRATEGIES = [
    CompressionStrategy.SMART_CRUSHER,
    CompressionStrategy.LOG,
    CompressionStrategy.SEARCH,
    CompressionStrategy.DIFF,
]


class _WordTokenizer:
    """Word-count tokenizer stub — no model, deterministic, offline-safe."""

    def count_text(self, text: object) -> int:
        return len(str(text).split())

    def count_messages(self, messages: list[dict]) -> int:
        return sum(self.count_text(m.get("content", "")) for m in messages)


def _force_result(strategy: CompressionStrategy, compressed: str) -> SimpleNamespace:
    """A RouterCompressionResult stand-in: apply() reads strategy_used, compressed,
    and compression_ratio. ratio 0.3 < any min_ratio so it takes the 'compressed'
    branch and reaches the reversibility gate."""
    return SimpleNamespace(
        compressed=compressed,
        original="",
        strategy_used=strategy,
        compression_ratio=0.3,
    )


def _tool_msg(content: str) -> dict:
    # tool_call_id with no matching assistant tool_calls -> not in the exclude
    # map -> not protected by the Read/Glob/Grep/Write/Edit window, so it reaches
    # compression (matches Bash/shell output, which is never excluded).
    return {"role": "tool", "tool_call_id": "call_bash_1", "content": content}


def _run(monkeypatch, message: dict, strategy: CompressionStrategy, compressed: str):
    router = ContentRouter()
    monkeypatch.setattr(router, "compress", lambda *a, **k: _force_result(strategy, compressed))
    # protect_recent / analysis protections are orthogonal to the reversibility
    # gate and would preempt compression for recent code-like content. Disabling
    # them isolates the gate and mirrors the real "aged-out tool output reaches
    # compression" case that #1307 is about.
    return router.apply(
        [message], _WordTokenizer(), protect_recent=0, protect_analysis_context=False
    )


def test_tool_role_lossy_unmarked_kept_verbatim(monkeypatch) -> None:
    """role=tool + lossy strategy + no CCR marker -> original preserved bit-for-bit."""
    result = _run(monkeypatch, _tool_msg(GREP_OUTPUT), CompressionStrategy.KOMPRESS, LOSSY_SUMMARY)
    assert result.messages[0]["content"] == GREP_OUTPUT


def test_tool_role_lossy_with_ccr_marker_accepted(monkeypatch) -> None:
    """role=tool + lossy strategy WITH a CCR marker -> compressed accepted (recoverable)."""
    result = _run(
        monkeypatch, _tool_msg(GREP_OUTPUT), CompressionStrategy.KOMPRESS, CCR_MARKER_SUMMARY
    )
    assert result.messages[0]["content"] == CCR_MARKER_SUMMARY


def test_assistant_role_lossy_still_compressed(monkeypatch) -> None:
    """Same lossy-unmarked result on role=assistant -> still compressed.

    Proves the gate is scoped to tool ground truth and does not regress
    assistant-text compression effectiveness."""
    msg = {"role": "assistant", "content": GREP_OUTPUT}
    result = _run(monkeypatch, msg, CompressionStrategy.KOMPRESS, LOSSY_SUMMARY)
    assert result.messages[0]["content"] == LOSSY_SUMMARY


@pytest.mark.parametrize("strategy", LOSSY_STRATEGIES, ids=lambda s: s.value)
def test_tool_role_lossy_strategies_all_gated(monkeypatch, strategy) -> None:
    """Every lossy-unmarked strategy is gated for tool role."""
    result = _run(monkeypatch, _tool_msg(GREP_OUTPUT), strategy, LOSSY_SUMMARY)
    assert result.messages[0]["content"] == GREP_OUTPUT


@pytest.mark.parametrize("strategy", STRUCTURED_STRATEGIES, ids=lambda s: s.value)
def test_tool_role_structured_strategies_accepted(monkeypatch, strategy) -> None:
    """Structured strategies are lossless/self-marking -> not gated, compressed kept."""
    result = _run(monkeypatch, _tool_msg(GREP_OUTPUT), strategy, LOSSY_SUMMARY)
    assert result.messages[0]["content"] == LOSSY_SUMMARY
