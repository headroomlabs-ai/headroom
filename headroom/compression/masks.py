"""Structure mask system for compression.

A StructureMask identifies which parts of content are "structural" (should be
preserved) vs "compressible" (can be compressed by Kompress).

This separates the concerns of:
1. Structure detection (handlers) - What tokens are navigational?
2. Content compression (Kompress) - What tokens can be removed?

The mask is content-agnostic - it's just a boolean array aligned to tokens.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field


@dataclass
class StructureMask:
    """A mask identifying structural vs compressible tokens.

    The mask is aligned to a token sequence. True means "preserve this token"
    (it's structural/navigational), False means "compressible" (Kompress can
    potentially remove it).

    Attributes:
        tokens: The tokenized content (list of strings or token IDs).
        mask: Boolean array, True = preserve, False = compressible.
        metadata: Optional handler-specific metadata.
    """

    tokens: Sequence[str | int]
    mask: list[bool]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate mask alignment."""
        if len(self.tokens) != len(self.mask):
            raise ValueError(
                f"Mask length ({len(self.mask)}) must match tokens length ({len(self.tokens)})"
            )

    @property
    def preservation_ratio(self) -> float:
        """Fraction of tokens marked for preservation."""
        if not self.mask:
            return 0.0
        return sum(self.mask) / len(self.mask)

    @property
    def structural_count(self) -> int:
        """Number of structural (preserved) tokens."""
        return sum(self.mask)

    @property
    def compressible_count(self) -> int:
        """Number of compressible tokens."""
        return len(self.mask) - sum(self.mask)

    def get_structural_tokens(self) -> list[str | int]:
        """Get list of tokens marked as structural."""
        return [t for t, m in zip(self.tokens, self.mask) if m]

    def get_compressible_tokens(self) -> list[str | int]:
        """Get list of tokens marked as compressible."""
        return [t for t, m in zip(self.tokens, self.mask) if not m]

    @classmethod
    def empty(cls, tokens: Sequence[str | int]) -> StructureMask:
        """Create a mask with no structural tokens (all compressible)."""
        return cls(tokens=tokens, mask=[False] * len(tokens))

    @classmethod
    def full(cls, tokens: Sequence[str | int]) -> StructureMask:
        """Create a mask preserving all tokens (nothing compressible)."""
        return cls(tokens=tokens, mask=[True] * len(tokens))

    def union(self, other: StructureMask) -> StructureMask:
        """Combine masks - preserve if EITHER mask says preserve.

        Useful when combining multiple structure detection strategies.

        Args:
            other: Another mask to combine with.

        Returns:
            New mask with union of preserved tokens.

        Raises:
            ValueError: If masks have different lengths.
        """
        if len(self.mask) != len(other.mask):
            raise ValueError("Cannot union masks of different lengths")

        return StructureMask(
            tokens=self.tokens,
            mask=[a or b for a, b in zip(self.mask, other.mask)],
            metadata={"source": "union", **self.metadata, **other.metadata},
        )

    def intersection(self, other: StructureMask) -> StructureMask:
        """Combine masks - preserve only if BOTH masks say preserve.

        Useful for being more aggressive with compression.

        Args:
            other: Another mask to combine with.

        Returns:
            New mask with intersection of preserved tokens.

        Raises:
            ValueError: If masks have different lengths.
        """
        if len(self.mask) != len(other.mask):
            raise ValueError("Cannot intersect masks of different lengths")

        return StructureMask(
            tokens=self.tokens,
            mask=[a and b for a, b in zip(self.mask, other.mask)],
            metadata={"source": "intersection", **self.metadata, **other.metadata},
        )


@dataclass
class MaskSpan:
    """A contiguous span in the mask.

    Useful for applying different compression strategies to different
    parts of the content.
    """

    start: int
    end: int
    is_structural: bool
    label: str = ""  # Optional label (e.g., "key", "value", "signature")

    @property
    def length(self) -> int:
        """Length of the span."""
        return self.end - self.start


def mask_to_spans(mask: StructureMask) -> list[MaskSpan]:
    """Convert a mask to a list of contiguous spans.

    This is useful for processing structural and compressible regions
    separately.

    Args:
        mask: The structure mask.

    Returns:
        List of MaskSpan objects representing contiguous regions.

    Example:
        >>> tokens = ["def", " ", "foo", "(", ")", ":", " ", "pass"]
        >>> mask = StructureMask(tokens, [True, True, True, True, True, True, False, False])
        >>> spans = mask_to_spans(mask)
        >>> [(s.start, s.end, s.is_structural) for s in spans]
        [(0, 6, True), (6, 8, False)]
    """
    if not mask.mask:
        return []

    spans = []
    current_start = 0
    current_structural = mask.mask[0]

    for i, is_structural in enumerate(mask.mask[1:], start=1):
        if is_structural != current_structural:
            spans.append(
                MaskSpan(
                    start=current_start,
                    end=i,
                    is_structural=current_structural,
                )
            )
            current_start = i
            current_structural = is_structural

    # Don't forget the last span
    spans.append(
        MaskSpan(
            start=current_start,
            end=len(mask.mask),
            is_structural=current_structural,
        )
    )

    return spans


def apply_mask_to_text(
    text: str,
    mask: StructureMask,
    compress_fn: Callable[[str], str],
    tokenizer_decode: Callable[[Sequence[str | int]], str] | None = None,
) -> str:
    """Apply compression to non-structural regions of text.

    This is the core function that enables structure-preserving compression.
    Structural regions are kept verbatim, non-structural regions are
    passed to the compression function.

    Args:
        text: Original text.
        mask: Structure mask aligned to tokens.
        compress_fn: Function to compress text (e.g., Kompress).
        tokenizer_decode: Optional function to decode tokens to text.
            If not provided, assumes tokens are strings and joins them.

    Returns:
        Text with non-structural regions compressed.
    """
    spans = mask_to_spans(mask)
    result_parts = []

    if tokenizer_decode is None:
        # Default: assume tokens are strings
        def tokenizer_decode(tokens: Sequence[str | int]) -> str:
            return "".join(str(t) for t in tokens)

    for span in spans:
        span_tokens = mask.tokens[span.start : span.end]
        span_text = tokenizer_decode(span_tokens)

        if span.is_structural:
            # Keep structural regions verbatim
            result_parts.append(span_text)
        else:
            # Compress non-structural regions
            compressed = compress_fn(span_text)
            result_parts.append(compressed)

    return "".join(result_parts)


@dataclass
class EntropyScore:
    """Entropy-based preservation signal.

    High entropy content (UUIDs, hashes, random strings) should generally
    be preserved because:
    1. They're information-dense (can't be reconstructed)
    2. They're often identifiers (semantically important)
    3. Token-level compressors may mangle them

    This is a self-signal - no external classifier needed.
    """

    value: float  # 0.0 to 1.0, normalized entropy
    should_preserve: bool  # True if entropy above threshold

    @classmethod
    def compute(cls, text: str, threshold: float = 0.85) -> EntropyScore:
        """Compute entropy score for text.

        Args:
            text: Text to analyze.
            threshold: Entropy threshold for preservation (0.0-1.0).
                Higher = more selective.

        Returns:
            EntropyScore with value and preservation recommendation.
        """
        if not text:
            return cls(value=0.0, should_preserve=False)

        # Calculate character entropy
        import math
        from collections import Counter

        # Count character frequencies
        counter = Counter(text)
        total = len(text)

        # Calculate Shannon entropy
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # Normalize to 0-1 range
        # Maximum possible entropy for this alphabet size
        max_entropy = math.log2(len(counter)) if len(counter) > 1 else 1.0
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        return cls(
            value=normalized,
            should_preserve=normalized >= threshold,
        )


def compute_entropy_mask(
    tokens: Sequence[str],
    threshold: float = 0.85,
    min_token_length: int = 8,
) -> StructureMask:
    """Create a mask preserving high-entropy tokens.

    This is a self-signal that doesn't require content classification.
    High-entropy tokens (UUIDs, hashes, etc.) are marked for preservation.

    Args:
        tokens: List of string tokens.
        threshold: Entropy threshold (0.0-1.0). Higher = more selective.
        min_token_length: Only check tokens this long or longer.
            Short tokens rarely have meaningful entropy.

    Returns:
        StructureMask with high-entropy tokens marked for preservation.

    Example:
        >>> tokens = ["user", ":", " ", "8f14e45f-ceea-4123-8f14-e45fceea4123"]
        >>> mask = compute_entropy_mask(tokens)
        >>> mask.mask
        [False, False, False, True]  # UUID preserved
    """
    mask = []

    for token in tokens:
        if isinstance(token, int):
            # Token ID, can't compute entropy
            mask.append(False)
            continue

        token_str = str(token)

        # Skip short tokens
        if len(token_str) < min_token_length:
            mask.append(False)
            continue

        # Compute entropy
        score = EntropyScore.compute(token_str, threshold)
        mask.append(score.should_preserve)

    return StructureMask(
        tokens=tokens,
        mask=mask,
        metadata={"source": "entropy", "threshold": threshold},
    )
