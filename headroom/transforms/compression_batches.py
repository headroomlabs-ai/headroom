"""Bounded batching for small provider-extracted compression units."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .compression_units import (
    _CCR_MARKER_RE,
    _LOSSY_UNMARKED_STRATEGIES,
    CompressionStrategy,
    ContentRouter,
    RoutedCompressionUnit,
    TokenCounterLike,
    UnitCompressionResult,
    _is_structured_shell_output,
)
from .content_router import RouterCompressionResult
from .tag_protector import protect_tags, restore_tags

DEFAULT_MAX_BATCH_BYTES = 2048
DEFAULT_MAX_BATCH_UNITS = 16


@dataclass(frozen=True)
class CompressionBatchEntry:
    """One provider slot with a stable batch-local identifier."""

    entry_id: str
    routed: RoutedCompressionUnit


@dataclass(frozen=True)
class CompressionBatch:
    """Compatible small units that share one future router invocation."""

    entries: tuple[CompressionBatchEntry, ...]
    text_bytes: int


def _text_bytes(text: str) -> int:
    return len(text.encode("utf-8", errors="replace"))


def _compatibility_key(entry: CompressionBatchEntry) -> tuple[object, ...]:
    unit = entry.routed.unit
    return (
        unit.provider,
        unit.endpoint,
        unit.role,
        unit.cache_zone,
        unit.mutable,
        unit.context,
        unit.question,
        unit.bias,
    )


def build_compression_batches(
    entries: list[CompressionBatchEntry],
    *,
    min_batch_bytes: int,
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    max_batch_units: int = DEFAULT_MAX_BATCH_UNITS,
) -> tuple[list[CompressionBatch], list[CompressionBatchEntry]]:
    """Greedily group compatible small units and skip under-floor tails.

    Callers retain the skipped entries as normal ``size_floor`` results. The
    function deliberately does not turn a unit larger than the configured
    batch ceiling into a singleton batch; those units belong to the existing
    independent compression path.
    """

    if min_batch_bytes <= 0:
        raise ValueError("min_batch_bytes must be positive")
    if max_batch_bytes < min_batch_bytes:
        raise ValueError("max_batch_bytes must be at least min_batch_bytes")
    if max_batch_units <= 0:
        raise ValueError("max_batch_units must be positive")

    batches: list[CompressionBatch] = []
    skipped: list[CompressionBatchEntry] = []
    pending: list[CompressionBatchEntry] = []
    pending_bytes = 0
    pending_key: tuple[object, ...] | None = None

    def flush() -> None:
        nonlocal pending, pending_bytes, pending_key
        if not pending:
            return
        if pending_bytes >= min_batch_bytes:
            batches.append(CompressionBatch(entries=tuple(pending), text_bytes=pending_bytes))
        else:
            skipped.extend(pending)
        pending = []
        pending_bytes = 0
        pending_key = None

    for entry in entries:
        entry_bytes = _text_bytes(entry.routed.unit.text)
        entry_key = _compatibility_key(entry)
        if entry_bytes >= min_batch_bytes or entry_bytes > max_batch_bytes:
            flush()
            skipped.append(entry)
            continue
        if pending and (
            entry_key != pending_key
            or len(pending) >= max_batch_units
            or pending_bytes + entry_bytes > max_batch_bytes
        ):
            flush()
        pending.append(entry)
        pending_bytes += entry_bytes
        pending_key = entry_key
        if len(pending) == max_batch_units or pending_bytes == max_batch_bytes:
            flush()

    flush()
    return batches, skipped


def _batch_nonce(batch: CompressionBatch) -> str:
    digest = hashlib.sha256()
    for entry in batch.entries:
        digest.update(entry.entry_id.encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(entry.routed.unit.text.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def _batch_envelope(batch: CompressionBatch, nonce: str, texts: list[str]) -> str:
    return "\n".join(
        (
            f"<headroom-batch-{nonce}-{entry.entry_id}>"
            f"{text}"
            f"</headroom-batch-{nonce}-{entry.entry_id}>"
        )
        for entry, text in zip(batch.entries, texts, strict=True)
    )


def _protect_ccr_markers(
    batch: CompressionBatch, nonce: str
) -> tuple[list[str], dict[str, tuple[int, str]]]:
    """Replace retrieval markers with unique tokens before the single router call."""

    protected_texts: list[str] = []
    marker_blocks: dict[str, tuple[int, str]] = {}
    for entry_index, entry in enumerate(batch.entries):
        marker_index = 0

        def replace_marker(match: re.Match[str], entry_index: int = entry_index) -> str:
            nonlocal marker_index
            placeholder = f"[[HEADROOM_BATCH_CCR_{nonce}_{entry_index}_{marker_index}]]"
            marker_index += 1
            marker_blocks[placeholder] = (entry_index, match.group(0))
            return placeholder

        protected_texts.append(_CCR_MARKER_RE.sub(replace_marker, entry.routed.unit.text))
    return protected_texts, marker_blocks


def _parse_batch_envelope(
    text: str,
    batch: CompressionBatch,
    nonce: str,
) -> list[str] | None:
    """Return ordered entry bodies only when every expected tag is intact."""

    cursor = 0
    values: list[str] = []
    for entry in batch.entries:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        tag_name = f"headroom-batch-{nonce}-{entry.entry_id}"
        pattern = re.compile(
            rf"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>",
            flags=re.DOTALL,
        )
        match = pattern.match(text, cursor)
        if match is None:
            return None
        values.append(match.group(1))
        cursor = match.end()
    if text[cursor:].strip():
        return None
    return values


def _passthrough_batch_results(
    batch: CompressionBatch,
    *,
    tokenizer: TokenCounterLike,
    reason: str,
    router_result: RouterCompressionResult | None = None,
) -> list[tuple[object, UnitCompressionResult]]:
    strategy = (
        router_result.strategy_used.value
        if router_result
        else CompressionStrategy.PASSTHROUGH.value
    )
    return [
        (
            entry.routed.slot,
            UnitCompressionResult(
                original=entry.routed.unit.text,
                compressed=entry.routed.unit.text,
                modified=False,
                tokens_before=tokenizer.count_text(entry.routed.unit.text),
                tokens_after=tokenizer.count_text(entry.routed.unit.text),
                tokens_saved=0,
                transforms_applied=[],
                strategy=strategy,
                reason=reason,
                router_result=router_result,
                text_bytes=_text_bytes(entry.routed.unit.text),
                min_bytes=entry.routed.unit.min_bytes,
                reason_category=reason,
            ),
        )
        for entry in batch.entries
    ]


def compress_batch_with_router(
    batch: CompressionBatch,
    *,
    router: ContentRouter,
    tokenizer: TokenCounterLike,
    target_ratio: float | None = None,
) -> list[tuple[object, UnitCompressionResult]]:
    """Compress one tagged batch and split only structurally valid output."""

    nonce = _batch_nonce(batch)
    batch_texts, marker_blocks = _protect_ccr_markers(batch, nonce)
    envelope = _batch_envelope(batch, nonce, batch_texts)
    protected, protected_blocks = protect_tags(envelope, compress_tagged_content=True)
    prior_target_ratio = getattr(router, "_runtime_target_ratio", None)
    if target_ratio is not None:
        router._runtime_target_ratio = target_ratio
    try:
        router_result = router.compress(
            protected,
            context=batch.entries[0].routed.unit.context,
            question=batch.entries[0].routed.unit.question,
            bias=batch.entries[0].routed.unit.bias,
        )
    except Exception:
        return _passthrough_batch_results(
            batch,
            tokenizer=tokenizer,
            reason="batch_router_error",
        )
    finally:
        if target_ratio is not None:
            router._runtime_target_ratio = prior_target_ratio

    compressed = router_result.compressed
    if not compressed or compressed == protected:
        return _passthrough_batch_results(
            batch,
            tokenizer=tokenizer,
            reason="router_no_change",
            router_result=router_result,
        )
    protected_placeholders = [placeholder for placeholder, _ in protected_blocks]
    protected_placeholders.extend(marker_blocks)
    if any(compressed.count(placeholder) != 1 for placeholder in protected_placeholders):
        return _passthrough_batch_results(
            batch,
            tokenizer=tokenizer,
            reason="batch_invalid",
            router_result=router_result,
        )

    restored = restore_tags(compressed, protected_blocks)
    replacements = _parse_batch_envelope(restored, batch, nonce)
    if replacements is None:
        return _passthrough_batch_results(
            batch,
            tokenizer=tokenizer,
            reason="batch_invalid",
            router_result=router_result,
        )
    if any(
        replacements[entry_index].count(placeholder) != 1
        for placeholder, (entry_index, _marker) in marker_blocks.items()
    ):
        return _passthrough_batch_results(
            batch,
            tokenizer=tokenizer,
            reason="batch_invalid",
            router_result=router_result,
        )
    for placeholder, (entry_index, marker) in marker_blocks.items():
        replacements[entry_index] = replacements[entry_index].replace(placeholder, marker)

    results: list[tuple[object, UnitCompressionResult]] = []
    strategy = router_result.strategy_used.value
    for entry, replacement in zip(batch.entries, replacements, strict=True):
        unit = entry.routed.unit
        tokens_before = tokenizer.count_text(unit.text)
        tokens_after = tokenizer.count_text(replacement)
        if (
            unit.role == "tool"
            and unit.item_type == "local_shell_call_output"
            and _is_structured_shell_output(unit.text)
            and strategy in _LOSSY_UNMARKED_STRATEGIES
            and not _CCR_MARKER_RE.search(replacement)
        ):
            result = UnitCompressionResult(
                original=unit.text,
                compressed=replacement,
                modified=False,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                tokens_saved=0,
                transforms_applied=[],
                strategy=strategy,
                reason="lossy_unrecoverable_tool_output",
                router_result=router_result,
                text_bytes=_text_bytes(unit.text),
                min_bytes=unit.min_bytes,
                reason_category="other",
            )
        elif tokens_after >= tokens_before:
            result = UnitCompressionResult(
                original=unit.text,
                compressed=replacement,
                modified=False,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                tokens_saved=0,
                transforms_applied=[],
                strategy=strategy,
                reason="rejected_not_smaller",
                router_result=router_result,
                text_bytes=_text_bytes(unit.text),
                min_bytes=unit.min_bytes,
                reason_category="rejected_not_smaller",
            )
        else:
            result = UnitCompressionResult(
                original=unit.text,
                compressed=replacement,
                modified=True,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                tokens_saved=tokens_before - tokens_after,
                transforms_applied=[
                    f"router:{unit.provider}:{unit.endpoint}:{unit.item_type}:{strategy}",
                    strategy,
                ],
                strategy=strategy,
                router_result=router_result,
                text_bytes=_text_bytes(unit.text),
                min_bytes=unit.min_bytes,
                reason_category="applied",
            )
        results.append((entry.routed.slot, result))
    return results
