"""Offloaded token-count helpers shared by proxy handlers.

Tokenizer resolution can be expensive on first use (HuggingFace backends may
download vocab files) and counting a full Claude Code conversation is CPU-bound,
so both run on the caller's compression executor bounded by
``COMPRESSION_TIMEOUT_SECONDS`` (GH #1701: an unbounded on-loop load froze the
whole server). On timeout, error, or a missing executor this fails open to
character-based estimation.

Shared by every provider handler mixin (Anthropic, OpenAI, Gemini): the OpenAI
``/v1/chat/completions`` and ``/v1/responses`` endpoints are multi-provider
passthroughs, so an HF-routed model (qwen, deepseek, llama, ...) can reach them
and trigger the same cold load.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

logger = logging.getLogger("headroom.proxy")


def _record_fallback_model(owner: Any, model: Any, message: str) -> None:
    fallback_models = getattr(owner, "_token_count_fallback_models", None)
    if fallback_models is None:
        fallback_models = set()
        owner._token_count_fallback_models = fallback_models
    if model not in fallback_models:
        fallback_models.add(model)
        logger.warning(message)


async def _count_offloaded(owner: Any, model: Any, count: Callable[[Any], int]) -> tuple[Any, int]:
    """Resolve a tokenizer and apply ``count`` off the event loop when possible.

    ``count`` maps a resolved tokenizer to a token total. Returns
    ``(tokenizer, total)``; the returned tokenizer is fully initialized, so later
    counts on it are pure CPU work. Fails open to ``EstimatingTokenCounter`` when
    the owner has no compression executor, or on timeout/error.
    """
    from headroom.proxy.helpers import COMPRESSION_TIMEOUT_SECONDS
    from headroom.tokenizers import EstimatingTokenCounter, get_tokenizer

    runner = getattr(owner, "_run_compression_in_executor", None)
    if runner is None:
        estimator = EstimatingTokenCounter()
        return estimator, count(estimator)

    def _resolve_and_count() -> tuple[Any, int]:
        tokenizer = get_tokenizer(model)
        return tokenizer, count(tokenizer)

    try:
        result = await runner(_resolve_and_count, timeout=float(COMPRESSION_TIMEOUT_SECONDS))
        return cast(tuple[Any, int], result)
    except Exception as e:  # fail open — includes asyncio.TimeoutError
        _record_fallback_model(
            owner,
            model,
            f"Token counting for model {model} failed or timed out "
            f"({e.__class__.__name__}); falling back to estimation",
        )
        estimator = EstimatingTokenCounter()
        return estimator, count(estimator)


async def count_tokens_offloaded(owner: Any, model: Any, messages: Any) -> tuple[Any, int]:
    """Resolve a tokenizer and count ``messages`` off the event loop when possible."""
    return await _count_offloaded(owner, model, lambda counter: counter.count_messages(messages))


async def count_texts_offloaded(owner: Any, model: Any, texts: Any) -> tuple[Any, int]:
    """Resolve a tokenizer and count text fragments off the event loop when possible."""
    text_list = list(texts)
    return await _count_offloaded(
        owner, model, lambda counter: sum(counter.count_text(text) for text in text_list)
    )


def gemini_output_tokens(usage_meta: dict[str, Any]) -> int:
    """Output-token count for a Gemini ``usageMetadata``, including thinking tokens.

    Gemini reports ``candidatesTokenCount`` sometimes inclusive of the
    ``thoughtsTokenCount`` (2.5-family reasoning) and sometimes exclusive of it.
    When ``promptTokenCount + candidatesTokenCount != totalTokenCount`` the
    thinking tokens are a separate bucket and must be added, or the output cost
    (billed at the output rate) is undercounted. Mirrors litellm's
    ``is_candidate_token_count_inclusive`` rule. Robust to missing/null fields.
    """

    def _int(value: Any) -> int:
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return 0

    candidates = _int(usage_meta.get("candidatesTokenCount"))
    thoughts = _int(usage_meta.get("thoughtsTokenCount"))
    if thoughts <= 0:
        return candidates
    prompt = _int(usage_meta.get("promptTokenCount"))
    total = _int(usage_meta.get("totalTokenCount"))
    # Inclusive iff prompt + candidates already equals total; otherwise the
    # thinking tokens are a separate bucket that belongs in the output count.
    if prompt + candidates == total:
        return candidates
    return candidates + thoughts
