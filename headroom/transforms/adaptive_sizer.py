"""Adaptive compression sizing via information saturation detection.

Instead of hardcoded max_items/max_matches, this module statistically determines
how many items to keep by finding the "knee point" — where adding more items
stops providing meaningful new information.

Algorithm: Track unique bigrams as items are added in importance order. Build a
cumulative coverage curve. Find the knee (Kneedle algorithm) where marginal
information gain drops sharply. That's the optimal K.

Per-tool profiles apply a bias multiplier on the statistically-determined K:
- conservative (bias=1.5): keep 50% more than mathematically needed
- moderate (bias=1.0): trust the statistics
- aggressive (bias=0.7): compress harder
"""

from __future__ import annotations

import hashlib
import logging
import zlib
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def compute_optimal_k(
    items: Sequence[str],
    bias: float = 1.0,
    min_k: int = 3,
    max_k: int | None = None,
) -> int:
    """Compute the optimal number of items to keep using information saturation.

    Three-tier decision system:
      Tier 1 (fast path): trivial cases, near-duplicate detection
      Tier 2 (standard):  Kneedle on unique bigram coverage curve
      Tier 3 (validation): zlib compression ratio sanity check

    Args:
        items: Sequence of string representations of items (in importance order).
        bias: Multiplier on the knee point. >1 = keep more, <1 = keep fewer.
        min_k: Never return fewer than this.
        max_k: Never return more than this (None = no cap).

    Returns:
        Optimal number of items to keep.
    """
    n = len(items)
    effective_max = max_k if max_k is not None else n

    # Tier 1: Fast path
    if n <= 8:
        return n

    # Check for near-total redundancy
    unique_count = count_unique_simhash(items)
    if unique_count <= 3:
        k = max(min_k, unique_count)
        return min(k, effective_max)

    # Tier 2: Kneedle on unique bigram coverage
    curve = compute_unique_bigram_curve(items)
    knee = find_knee(curve)

    # Diversity ratio: what fraction of items are genuinely unique?
    # 1.0 = every item is distinct, 0.1 = mostly near-duplicates.
    diversity_ratio = unique_count / n

    if knee is None:
        # No saturation found — each item adds new information.
        # Scale keep-fraction continuously with diversity:
        #   diversity ~1.0 → keep 100%  (all unique — dropping any loses info)
        #   diversity ~0.5 → keep ~65%  (moderate)
        #   diversity ~0.2 → keep ~44%  (low-ish)
        #   diversity ~0.0 → keep ~30%  (mostly dupes, same as old default)
        # No arbitrary cap — if items are all unique, keep them all.
        keep_fraction = 0.3 + 0.7 * diversity_ratio
        knee = max(min_k, int(n * keep_fraction))
    else:
        # Knee found, but if diversity is very high the knee may be
        # a weak signal (e.g., minor bigram overlap causing a shallow
        # curve bend).  Don't drop below a diversity floor.
        if diversity_ratio > 0.7:
            diversity_floor = max(min_k, int(n * (0.3 + 0.7 * diversity_ratio)))
            knee = max(knee, diversity_floor)

    # Apply bias multiplier
    k = max(min_k, int(knee * bias))
    k = min(k, effective_max)

    # Tier 3: Validate with zlib compression ratio
    k = _validate_with_zlib(items, k, effective_max)

    k = max(min_k, min(k, effective_max))

    logger.debug(
        "adaptive_sizer: n=%d unique=%d diversity=%.2f knee=%s bias=%.1f → k=%d",
        n,
        unique_count,
        diversity_ratio,
        knee,
        bias,
        k,
    )
    return k


def find_knee(curve: list[int]) -> int | None:
    """Find the knee point in a monotonically increasing curve.

    Uses the Kneedle algorithm: normalize to [0,1], compute the difference
    from the y=x diagonal, return the index of maximum difference.

    Args:
        curve: List of cumulative values (e.g., unique bigram counts).

    Returns:
        Index of the knee point, or None if no clear knee exists.
    """
    n = len(curve)
    if n < 3:
        return None

    # Normalize x and y to [0, 1]
    x_min, x_max = 0, n - 1
    y_min, y_max = curve[0], curve[-1]

    if y_max == y_min:
        # Flat curve — all items are identical
        return 1

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Compute difference from the diagonal (y = x in normalized space)
    max_diff = -1.0
    knee_idx = None

    for i in range(n):
        x_norm = (i - x_min) / x_range
        y_norm = (curve[i] - y_min) / y_range
        diff = y_norm - x_norm  # For concave curves, knee is where this is maximized
        if diff > max_diff:
            max_diff = diff
            knee_idx = i

    # Require a meaningful deviation from diagonal
    if max_diff < 0.05:
        return None

    # Knee is at knee_idx, but we want to include items up to and including the knee
    # Add 1 because we're converting from 0-indexed to count
    return knee_idx + 1 if knee_idx is not None else None


def compute_unique_bigram_curve(items: Sequence[str]) -> list[int]:
    """Build cumulative unique bigram coverage curve.

    For each item (in order), extracts word-level bigrams, adds them to a
    running set, and records the total unique count.

    Args:
        items: Sequence of string items in importance order.

    Returns:
        List where curve[k] = number of unique bigrams after seeing items[0:k+1].
    """
    seen_bigrams: set[tuple[str, str]] = set()
    curve: list[int] = []

    for item in items:
        words = item.lower().split()
        if len(words) < 2:
            # For single-word items, use the word itself as a "unigram bigram"
            seen_bigrams.add((words[0] if words else "", ""))
        else:
            for j in range(len(words) - 1):
                seen_bigrams.add((words[j], words[j + 1]))
        curve.append(len(seen_bigrams))

    return curve


def _simhash(text: str) -> int:
    """Compute a 64-bit SimHash fingerprint for a text string.

    Uses character 4-grams hashed to 64-bit values, then aggregates
    via weighted bit voting.

    Args:
        text: Input text.

    Returns:
        64-bit integer fingerprint.
    """
    v = [0] * 64
    text_lower = text.lower()

    # Character 4-grams
    for i in range(max(1, len(text_lower) - 3)):
        gram = text_lower[i : i + 4]
        h = int(hashlib.md5(gram.encode(), usedforsecurity=False).hexdigest()[:16], 16)  # nosec B324
        for j in range(64):
            if h & (1 << j):
                v[j] += 1
            else:
                v[j] -= 1

    fingerprint = 0
    for j in range(64):
        if v[j] > 0:
            fingerprint |= 1 << j
    return fingerprint


def _hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two 64-bit integers."""
    return bin(a ^ b).count("1")


def count_unique_simhash(items: Sequence[str], threshold: int = 3) -> int:
    """Count items with distinct content using SimHash.

    Groups items by SimHash fingerprint similarity (Hamming distance <= threshold).
    Returns the number of distinct groups.

    Args:
        items: Sequence of string items.
        threshold: Max Hamming distance to consider items as duplicates.

    Returns:
        Number of unique content groups.
    """
    if not items:
        return 0

    # Compute fingerprints
    fingerprints = [_simhash(item) for item in items]

    # Greedy clustering: assign each item to the first matching cluster
    clusters: list[int] = []  # Representative fingerprint per cluster
    for fp in fingerprints:
        matched = False
        for rep in clusters:
            if _hamming_distance(fp, rep) <= threshold:
                matched = True
                break
        if not matched:
            clusters.append(fp)

    return len(clusters)


def _validate_with_zlib(
    items: Sequence[str],
    k: int,
    max_k: int,
    tolerance: float = 0.15,
) -> int:
    """Validate K using zlib compression ratio comparison.

    If the compression ratio of the selected subset differs significantly
    from the full set, increase K.

    Args:
        items: All items.
        k: Currently proposed K.
        max_k: Maximum allowed K.
        tolerance: Max allowed ratio difference (default 15%).

    Returns:
        Adjusted K (may be increased if validation fails).
    """
    if k >= len(items) or k >= max_k:
        return k

    full_text = "\n".join(items).encode()
    subset_text = "\n".join(items[:k]).encode()

    # Skip validation for very small content (zlib overhead dominates)
    if len(full_text) < 200:
        return k

    full_compressed = len(zlib.compress(full_text, level=1))
    subset_compressed = len(zlib.compress(subset_text, level=1))

    full_ratio = full_compressed / len(full_text) if full_text else 1.0
    subset_ratio = subset_compressed / len(subset_text) if subset_text else 1.0

    # If subset compresses much better than full, it's missing diverse content
    # A lower ratio means more redundancy. If subset ratio is much lower,
    # it means the subset is more redundant than the full set — we're missing info.
    ratio_diff = abs(full_ratio - subset_ratio)

    if ratio_diff > tolerance:
        # Increase K by 20% to capture more diversity
        adjusted_k = min(int(k * 1.2), max_k)
        logger.debug(
            "zlib validation: ratio_diff=%.3f > %.3f, adjusting k=%d → %d",
            ratio_diff,
            tolerance,
            k,
            adjusted_k,
        )
        return adjusted_k

    return k
