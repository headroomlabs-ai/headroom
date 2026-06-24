"""LLM-as-judge scoring for memory evaluation.

Uses an LLM to evaluate answer quality by comparing predictions
against ground truth answers. More nuanced than token-level metrics
like F1 and exact match.

The judge scores answers on a 1-5 scale:
- 5: Perfect match (semantically equivalent)
- 4: Mostly correct (minor details differ)
- 3: Partially correct (key info present, some errors)
- 2: Mostly incorrect (some relevant info, major errors)
- 1: Completely wrong (irrelevant or contradictory)
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Prompt template for LLM judge
JUDGE_PROMPT = """You are evaluating a memory-based question answering system.

Given a question, the ground truth answer, and the system's predicted answer,
score the prediction on a scale of 1-5:

5 = Perfect: The predicted answer is semantically equivalent to the ground truth
4 = Mostly correct: The prediction captures the main point with minor differences
3 = Partially correct: The prediction has some correct information but is incomplete or has errors
2 = Mostly incorrect: The prediction has minimal relevant information or significant errors
1 = Completely wrong: The prediction is irrelevant or contradicts the ground truth

Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {prediction}

First, provide a brief reasoning (1-2 sentences), then give your score.

Format your response EXACTLY as:
Reasoning: <your reasoning>
Score: <number 1-5>"""


def create_openai_judge(
    model: str = "gpt-4o",
    api_key: str | None = None,
) -> Callable[[str, str, str], tuple[float, str]]:
    """Create an LLM judge using OpenAI's API.

    Args:
        model: OpenAI model to use (default: gpt-4o).
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided).

    Returns:
        A judge function that takes (question, ground_truth, prediction)
        and returns (score, reasoning).

    Example:
        judge_fn = create_openai_judge(model="gpt-4o-mini")
        score, reasoning = judge_fn(
            "What is Alice's favorite color?",
            "Blue",
            "Alice prefers blue"
        )
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "OpenAI package required for LLM judge. Install with: pip install openai"
        ) from e

    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def judge(question: str, ground_truth: str, prediction: str) -> tuple[float, str]:
        prompt = JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Deterministic scoring
            max_tokens=200,
        )

        text = response.choices[0].message.content or ""
        return _parse_judge_response(text)

    return judge


def create_anthropic_judge(
    model: str = "claude-sonnet-4-20250514",
    api_key: str | None = None,
) -> Callable[[str, str, str], tuple[float, str]]:
    """Create an LLM judge using Anthropic's API.

    Args:
        model: Anthropic model to use (default: claude-sonnet-4-20250514).
        api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided).

    Returns:
        A judge function that takes (question, ground_truth, prediction)
        and returns (score, reasoning).

    Example:
        judge_fn = create_anthropic_judge()
        score, reasoning = judge_fn(
            "When did Bob start his new job?",
            "March 2024",
            "Bob began working at his new position in early March of 2024"
        )
    """
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "Anthropic package required for LLM judge. Install with: pip install anthropic"
        ) from e

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def judge(question: str, ground_truth: str, prediction: str) -> tuple[float, str]:
        prompt = JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
        )

        response = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )

        text = ""
        if response.content and hasattr(response.content[0], "text"):
            text = response.content[0].text
        return _parse_judge_response(text)

    return judge


def create_litellm_judge(
    model: str = "gpt-4o",
) -> Callable[[str, str, str], tuple[float, str]]:
    """Create an LLM judge using LiteLLM for any supported provider.

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514", "ollama/llama3").

    Returns:
        A judge function that takes (question, ground_truth, prediction)
        and returns (score, reasoning).

    Example:
        # Use Ollama for local evaluation
        judge_fn = create_litellm_judge(model="ollama/llama3")
        score, reasoning = judge_fn(question, ground_truth, prediction)
    """
    try:
        import litellm
    except ImportError as e:
        raise ImportError(
            "LiteLLM package required for LLM judge. Install with: pip install litellm"
        ) from e

    def judge(question: str, ground_truth: str, prediction: str) -> tuple[float, str]:
        prompt = JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
        )

        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )

        text = response.choices[0].message.content or ""
        return _parse_judge_response(text)

    return judge


def _parse_judge_response(text: str) -> tuple[float, str]:
    """Parse the judge's response to extract score and reasoning.

    Args:
        text: Raw response from the LLM judge.

    Returns:
        Tuple of (score, reasoning).
    """
    reasoning = ""
    score = 3.0  # Default to middle score if parsing fails

    lines = text.strip().split("\n")

    for line in lines:
        line = line.strip()

        if line.lower().startswith("reasoning:"):
            reasoning = line[len("reasoning:") :].strip()

        elif line.lower().startswith("score:"):
            score_text = line[len("score:") :].strip()
            try:
                # Extract the first number from the score text
                import re

                match = re.search(r"(\d+(?:\.\d+)?)", score_text)
                if match:
                    score = float(match.group(1))
                    # Clamp to valid range
                    score = max(1.0, min(5.0, score))
            except ValueError:
                logger.warning(f"Could not parse score from: {score_text}")

    # If no explicit reasoning found, use the whole text
    if not reasoning:
        reasoning = text.strip()

    return score, reasoning


def create_batch_judge(
    judge_fn: Callable[[str, str, str], tuple[float, str]],
    max_concurrent: int = 5,
) -> Callable[[list[tuple[str, str, str]]], list[tuple[float, str]]]:
    """Create a batch judge function for parallel evaluation.

    Args:
        judge_fn: Single-item judge function.
        max_concurrent: Maximum concurrent API calls.

    Returns:
        A function that takes a list of (question, ground_truth, prediction)
        and returns a list of (score, reasoning).
    """
    from concurrent.futures import ThreadPoolExecutor

    def batch_judge(
        items: list[tuple[str, str, str]],
    ) -> list[tuple[float, str]]:
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(judge_fn, q, gt, pred) for q, gt, pred in items]
            return [f.result() for f in futures]

    return batch_judge


# Convenience function for simple scoring without LLM
def simple_judge(
    question: str,
    ground_truth: str,
    prediction: str,
) -> tuple[float, str]:
    """Simple rule-based judge using F1 score.

    Useful for quick evaluation without API calls.

    Args:
        question: The question (unused but kept for API compatibility).
        ground_truth: Expected answer.
        prediction: Predicted answer.

    Returns:
        Tuple of (score 1-5, reasoning).
    """
    from headroom.evals.metrics import compute_exact_match, compute_f1

    # Check exact match first
    if compute_exact_match(prediction, ground_truth):
        return 5.0, "Exact match with ground truth"

    # Calculate F1 score
    f1 = compute_f1(prediction, ground_truth)

    # Map F1 to 1-5 scale
    if f1 >= 0.9:
        score = 5.0
        reasoning = f"Very high token overlap (F1={f1:.2f})"
    elif f1 >= 0.7:
        score = 4.0
        reasoning = f"High token overlap (F1={f1:.2f})"
    elif f1 >= 0.5:
        score = 3.0
        reasoning = f"Moderate token overlap (F1={f1:.2f})"
    elif f1 >= 0.3:
        score = 2.0
        reasoning = f"Low token overlap (F1={f1:.2f})"
    else:
        score = 1.0
        reasoning = f"Very low token overlap (F1={f1:.2f})"

    return score, reasoning
