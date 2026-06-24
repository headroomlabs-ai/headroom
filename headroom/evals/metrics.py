"""Evaluation metrics for comparing LLM outputs.

These metrics determine whether compression preserved accuracy
by comparing responses from original vs compressed context.
"""

from __future__ import annotations

import re
from collections import Counter


def normalize_text(text: str) -> str:
    """Normalize text for comparison.

    - Lowercase
    - Remove extra whitespace
    - Remove punctuation (optional)
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    """Simple word tokenization."""
    # Split on whitespace and punctuation
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


def compute_exact_match(response_a: str, response_b: str) -> bool:
    """Check if two responses are exactly the same (after normalization)."""
    return normalize_text(response_a) == normalize_text(response_b)


def compute_f1(response_a: str, response_b: str) -> float:
    """Compute token-level F1 score between two responses.

    F1 = 2 * (precision * recall) / (precision + recall)

    This measures how much overlap there is between the two responses.
    A score of 1.0 means identical tokens, 0.0 means no overlap.
    """
    tokens_a = tokenize(response_a)
    tokens_b = tokenize(response_b)

    if not tokens_a or not tokens_b:
        return 0.0

    # Count token occurrences
    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)

    # Find common tokens (considering counts)
    common = sum((counter_a & counter_b).values())

    if common == 0:
        return 0.0

    precision = common / len(tokens_a)
    recall = common / len(tokens_b)

    return 2 * precision * recall / (precision + recall)


def compute_bleu(response_a: str, response_b: str, max_n: int = 4) -> float:
    """Compute BLEU-like score between two responses.

    Uses n-gram precision up to max_n.
    """
    tokens_a = tokenize(response_a)
    tokens_b = tokenize(response_b)

    if not tokens_a or not tokens_b:
        return 0.0

    precisions = []

    for n in range(1, max_n + 1):
        # Get n-grams
        ngrams_a = [tuple(tokens_a[i : i + n]) for i in range(len(tokens_a) - n + 1)]
        ngrams_b = [tuple(tokens_b[i : i + n]) for i in range(len(tokens_b) - n + 1)]

        if not ngrams_a:
            break

        counter_a = Counter(ngrams_a)
        counter_b = Counter(ngrams_b)

        # Clipped count
        clipped = sum((counter_a & counter_b).values())
        total = len(ngrams_a)

        if total > 0:
            precisions.append(clipped / total)
        else:
            precisions.append(0.0)

    if not precisions or all(p == 0 for p in precisions):
        return 0.0

    # Geometric mean of precisions
    import math

    nonzero_precisions = [p for p in precisions if p > 0]
    log_sum = sum(math.log(p) for p in nonzero_precisions) / len(nonzero_precisions)
    return math.exp(log_sum)


def compute_rouge_l(response_a: str, response_b: str) -> float:
    """Compute ROUGE-L score (longest common subsequence).

    Measures the longest common subsequence between two responses.
    """
    tokens_a = tokenize(response_a)
    tokens_b = tokenize(response_b)

    if not tokens_a or not tokens_b:
        return 0.0

    # LCS using dynamic programming
    m, n = len(tokens_a), len(tokens_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens_a[i - 1] == tokens_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    if lcs_length == 0:
        return 0.0

    precision = lcs_length / m
    recall = lcs_length / n

    return 2 * precision * recall / (precision + recall)


def compute_semantic_similarity(
    response_a: str,
    response_b: str,
    model_name: str | None = None,
) -> float:
    """Compute semantic similarity using sentence embeddings.

    Requires sentence-transformers package.

    Args:
        response_a: First response
        response_b: Second response
        model_name: Sentence transformer model to use. Uses config default if None.

    Returns:
        Cosine similarity between embeddings (0.0 to 1.0)
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "sentence-transformers required for semantic similarity. "
            "Install with: pip install sentence-transformers"
        ) from e

    # Use centralized registry for shared model instances
    from headroom.models.ml_models import MLModelRegistry

    model = MLModelRegistry.get_sentence_transformer(model_name)

    embeddings = model.encode([response_a, response_b])
    embedding_a, embedding_b = embeddings[0], embeddings[1]

    # Cosine similarity
    dot_product = np.dot(embedding_a, embedding_b)
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def compute_answer_equivalence(
    response_a: str,
    response_b: str,
    ground_truth: str | None = None,
    semantic_threshold: float = 0.85,
    f1_threshold: float = 0.7,
) -> dict[str, float | bool | None]:
    """Comprehensive answer equivalence check.

    Combines multiple metrics to determine if two responses
    are functionally equivalent.

    Args:
        response_a: Original response
        response_b: Compressed response
        ground_truth: Optional ground truth answer
        semantic_threshold: Threshold for semantic similarity
        f1_threshold: Threshold for F1 score

    Returns:
        Dictionary with metrics and overall verdict
    """
    exact_match = compute_exact_match(response_a, response_b)
    f1_score = compute_f1(response_a, response_b)
    rouge_l = compute_rouge_l(response_a, response_b)
    semantic_similarity: float | None = None
    ground_truth_in_a: bool | None = None
    ground_truth_in_b: bool | None = None

    # Try semantic similarity
    try:
        semantic_similarity = compute_semantic_similarity(response_a, response_b)
    except ImportError:
        pass

    # Check ground truth
    if ground_truth:
        gt_lower = ground_truth.lower()
        ground_truth_in_a = gt_lower in response_a.lower()
        ground_truth_in_b = gt_lower in response_b.lower()

    # Determine equivalence
    # Equivalent if ANY of:
    # 1. Exact match
    # 2. High F1 score
    # 3. High semantic similarity
    # 4. Both contain ground truth
    equivalent = (
        exact_match
        or f1_score >= f1_threshold
        or (semantic_similarity is not None and semantic_similarity >= semantic_threshold)
        or (ground_truth_in_a is True and ground_truth_in_b is True)
    )

    result: dict[str, float | bool | None] = {
        "exact_match": exact_match,
        "f1_score": f1_score,
        "rouge_l": rouge_l,
        "semantic_similarity": semantic_similarity,
        "ground_truth_in_a": ground_truth_in_a,
        "ground_truth_in_b": ground_truth_in_b,
        "equivalent": equivalent,
    }

    return result


def compute_information_recall(
    original_context: str,
    compressed_context: str,
    probe_facts: list[str],
) -> dict:
    """Test if specific facts are preserved after compression.

    Args:
        original_context: Original context before compression
        compressed_context: Context after compression
        probe_facts: List of facts/strings that should be retrievable

    Returns:
        Dictionary with recall metrics
    """
    original_lower = original_context.lower()
    compressed_lower = compressed_context.lower()

    facts_in_original = []
    facts_in_compressed = []
    facts_lost = []

    for fact in probe_facts:
        fact_lower = fact.lower()
        in_original = fact_lower in original_lower
        in_compressed = fact_lower in compressed_lower

        if in_original:
            facts_in_original.append(fact)
            if in_compressed:
                facts_in_compressed.append(fact)
            else:
                facts_lost.append(fact)

    recall = len(facts_in_compressed) / len(facts_in_original) if facts_in_original else 1.0

    return {
        "total_probes": len(probe_facts),
        "facts_in_original": len(facts_in_original),
        "facts_preserved": len(facts_in_compressed),
        "facts_lost": facts_lost,
        "recall": recall,
    }
