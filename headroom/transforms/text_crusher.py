"""TextCrusher — fast deterministic extractive prose compressor (Phase 2, #1171).

Thin Python wrapper over the native ``headroom._core.TextCrusher``. The
algorithm (sentence scoring with the SHARED BM25 relevance scorer + global
word-shingle near-dup suppression) lives in Rust (``crates/headroom-core``),
reusing the same scorer SmartCrusher uses rather than reimplementing it. This
wrapper only keeps the Python-facing interface stable for ContentRouter + tests.

TextCrusher is the request-path-safe alternative to ModernBERT (kompress) for
large plain text: ~milliseconds instead of minutes. Extractive -- the kept
sentences are verbatim words (each segment trimmed, re-joined with newlines),
never paraphrased; it selects, it does not rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from headroom._core import TextCrusher as _RustTextCrusher
from headroom._core import TextCrusherConfig as _RustTextCrusherConfig


@dataclass
class TextCrusherConfig:
    target_ratio: float = 0.5
    w_recency: float = 1.0
    w_relevance: float = 2.0
    w_salience: float = 1.5
    min_segment_chars: int = 12
    near_dup_threshold: float = 0.85
    min_segments_for_crush: int = 6


class TextCrusher:
    """Extractive prose compressor. ``compress`` returns a result whose
    ``compressed`` text is the kept input sentences verbatim (each trimmed,
    re-joined with newlines) in original order -- selection, not rewriting.
    Backed by the Rust core."""

    def __init__(self, config: TextCrusherConfig | None = None) -> None:
        cfg = config or TextCrusherConfig()
        self._rust = _RustTextCrusher(
            _RustTextCrusherConfig(
                target_ratio=cfg.target_ratio,
                w_recency=cfg.w_recency,
                w_relevance=cfg.w_relevance,
                w_salience=cfg.w_salience,
                min_segment_chars=cfg.min_segment_chars,
                near_dup_threshold=cfg.near_dup_threshold,
                min_segments_for_crush=cfg.min_segments_for_crush,
            )
        )

    def compress(self, content: str, context: str = "", target_ratio: float | None = None) -> Any:
        """Returns a ``TextCrusherResult`` (Rust pyclass) with ``.compressed``,
        ``.original_tokens``, ``.compressed_tokens``, ``.compression_ratio``,
        ``.kept_segments``, ``.total_segments``."""
        return self._rust.compress(content, context, target_ratio)
