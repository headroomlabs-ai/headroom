"""Prompt Comparison System using LLM-as-Judge.

This module provides tools to verify that Headroom preserves the semantic
meaning of prompts by comparing prompts before and after processing.

This is CRITICAL for proving that Headroom doesn't change the meaning of prompts
when they go through the proxy.

Usage:
    from headroom.evals.prompt_comparison import compare_prompts, PromptComparer

    # Simple comparison
    result = compare_prompts(
        original_prompt="What is the capital of France?",
        headroom_modified_prompt="What is the capital of France?",
    )
    print(f"Equivalent: {result.are_equivalent}")

    # Full comparer with logging
    comparer = PromptComparer()
    result = comparer.compare(original, modified)
    comparer.log_result(result)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Judge prompt template for semantic equivalence checking
SEMANTIC_EQUIVALENCE_JUDGE_PROMPT = """You are an expert evaluator assessing whether two prompts are semantically equivalent.

Your task is to determine if a modified prompt preserves the exact same meaning, intent, and information as the original prompt.

CRITICAL: Even small changes in meaning matter. You must be extremely precise.

Consider these aspects:
1. **Core Intent**: Does the modified prompt ask for the same thing?
2. **Context Preservation**: Is all contextual information preserved?
3. **Constraints**: Are all constraints, requirements, and specifications preserved?
4. **Tone and Formality**: Are there significant tone changes that might affect the response?
5. **Information Content**: Is all factual information in the original still present?
6. **Ambiguity**: Does the modification introduce or remove ambiguity?

ORIGINAL PROMPT:
```
{original_prompt}
```

MODIFIED PROMPT:
```
{modified_prompt}
```

Analyze both prompts carefully and provide your assessment.

Respond in EXACTLY this format:
EQUIVALENT: <YES or NO>
CONFIDENCE: <HIGH, MEDIUM, or LOW>
DIFFERENCES: <List specific differences found, or "None" if equivalent>
REASONING: <Brief explanation of your assessment>
CONCERN_LEVEL: <NONE, LOW, MEDIUM, HIGH, or CRITICAL>"""


@dataclass
class PromptComparisonResult:
    """Result of comparing two prompts for semantic equivalence.

    Attributes:
        original: The original prompt before Headroom processing.
        headroom_modified: The prompt after Headroom processing.
        are_equivalent: Whether the prompts are semantically equivalent.
        confidence: Confidence level of the judgment (HIGH, MEDIUM, LOW).
        differences: List of specific differences found between prompts.
        reasoning: The judge's explanation for the assessment.
        concern_level: Level of concern about the differences (NONE to CRITICAL).
        judge_model: The model used for judging.
        timestamp: When the comparison was made.
        raw_judge_response: The full response from the judge LLM.
        metadata: Additional metadata about the comparison.
    """

    original: str
    headroom_modified: str
    are_equivalent: bool
    confidence: str = "MEDIUM"
    differences: list[str] = field(default_factory=list)
    reasoning: str = ""
    concern_level: str = "NONE"
    judge_model: str = "gpt-4o"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    raw_judge_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "original": self.original,
            "headroom_modified": self.headroom_modified,
            "are_equivalent": self.are_equivalent,
            "confidence": self.confidence,
            "differences": self.differences,
            "reasoning": self.reasoning,
            "concern_level": self.concern_level,
            "judge_model": self.judge_model,
            "timestamp": self.timestamp,
            "raw_judge_response": self.raw_judge_response,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptComparisonResult:
        """Create result from dictionary."""
        return cls(
            original=data["original"],
            headroom_modified=data["headroom_modified"],
            are_equivalent=data["are_equivalent"],
            confidence=data.get("confidence", "MEDIUM"),
            differences=data.get("differences", []),
            reasoning=data.get("reasoning", ""),
            concern_level=data.get("concern_level", "NONE"),
            judge_model=data.get("judge_model", "gpt-4o"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            raw_judge_response=data.get("raw_judge_response", ""),
            metadata=data.get("metadata", {}),
        )

    def is_concerning(self) -> bool:
        """Check if the result indicates concerning differences."""
        return not self.are_equivalent or self.concern_level in ["HIGH", "CRITICAL"]

    def summary(self) -> str:
        """Generate a human-readable summary."""
        status = "PASS" if self.are_equivalent else "FAIL"
        lines = [
            f"=== Prompt Comparison Result: {status} ===",
            f"Equivalent: {self.are_equivalent}",
            f"Confidence: {self.confidence}",
            f"Concern Level: {self.concern_level}",
            f"Judge Model: {self.judge_model}",
        ]

        if self.differences:
            lines.append("Differences:")
            for diff in self.differences:
                lines.append(f"  - {diff}")

        if self.reasoning:
            lines.append(f"Reasoning: {self.reasoning}")

        return "\n".join(lines)


def _parse_judge_response(response_text: str) -> dict[str, Any]:
    """Parse the judge's response to extract structured fields.

    Args:
        response_text: Raw response from the judge LLM.

    Returns:
        Dictionary with parsed fields.
    """
    result = {
        "are_equivalent": False,
        "confidence": "MEDIUM",
        "differences": [],
        "reasoning": "",
        "concern_level": "MEDIUM",
    }

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()

        if line.upper().startswith("EQUIVALENT:"):
            value = line[len("EQUIVALENT:") :].strip().upper()
            result["are_equivalent"] = value == "YES"

        elif line.upper().startswith("CONFIDENCE:"):
            value = line[len("CONFIDENCE:") :].strip().upper()
            if value in ["HIGH", "MEDIUM", "LOW"]:
                result["confidence"] = value

        elif line.upper().startswith("DIFFERENCES:"):
            diff_text = line[len("DIFFERENCES:") :].strip()
            if diff_text.lower() != "none":
                # Parse differences - could be comma-separated or a single item
                if "," in diff_text:
                    result["differences"] = [d.strip() for d in diff_text.split(",")]
                elif diff_text:
                    result["differences"] = [diff_text]

        elif line.upper().startswith("REASONING:"):
            result["reasoning"] = line[len("REASONING:") :].strip()

        elif line.upper().startswith("CONCERN_LEVEL:"):
            value = line[len("CONCERN_LEVEL:") :].strip().upper()
            if value in ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                result["concern_level"] = value

    return result


def compare_prompts(
    original_prompt: str,
    headroom_modified_prompt: str,
    judge_model: str = "gpt-4o",
    api_key: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> PromptComparisonResult:
    """Compare two prompts using an LLM judge to verify semantic equivalence.

    This function sends both prompts to GPT-4o (or another specified model)
    to determine if they are semantically equivalent.

    Args:
        original_prompt: The original prompt before Headroom processing.
        headroom_modified_prompt: The prompt after Headroom processing.
        judge_model: The OpenAI model to use as judge (default: gpt-4o).
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided).
        metadata: Optional metadata to attach to the result.

    Returns:
        PromptComparisonResult with the comparison outcome.

    Raises:
        ImportError: If openai package is not installed.
        ValueError: If API key is not provided and not in environment.

    Example:
        result = compare_prompts(
            original_prompt="Explain quantum computing in simple terms.",
            headroom_modified_prompt="Explain quantum computing in simple terms.",
        )
        if not result.are_equivalent:
            print(f"WARNING: Prompts differ! {result.differences}")
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "OpenAI package required for prompt comparison. Install with: pip install openai"
        ) from e

    # Get API key
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )

    client = OpenAI(api_key=resolved_api_key)

    # Build the judge prompt
    judge_prompt = SEMANTIC_EQUIVALENCE_JUDGE_PROMPT.format(
        original_prompt=original_prompt,
        modified_prompt=headroom_modified_prompt,
    )

    # Call the judge
    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,  # Deterministic for consistent evaluation
        max_tokens=500,
    )

    raw_response = response.choices[0].message.content or ""

    # Parse the response
    parsed = _parse_judge_response(raw_response)

    return PromptComparisonResult(
        original=original_prompt,
        headroom_modified=headroom_modified_prompt,
        are_equivalent=parsed["are_equivalent"],
        confidence=parsed["confidence"],
        differences=parsed["differences"],
        reasoning=parsed["reasoning"],
        concern_level=parsed["concern_level"],
        judge_model=judge_model,
        raw_judge_response=raw_response,
        metadata=metadata or {},
    )


class PromptComparer:
    """A class for comparing prompts with logging and persistence.

    This class provides a stateful interface for comparing prompts,
    with support for logging results to files and tracking history.

    Attributes:
        judge_model: The model to use for judging.
        log_dir: Directory for storing comparison logs.
        comparison_history: List of all comparisons made.

    Example:
        comparer = PromptComparer(log_dir="./comparison_logs")

        # Compare prompts
        result = comparer.compare(original, modified)

        # Log the result
        comparer.log_result(result)

        # Get summary of all comparisons
        print(comparer.summary())
    """

    def __init__(
        self,
        judge_model: str = "gpt-4o",
        api_key: str | None = None,
        log_dir: str | Path | None = None,
    ):
        """Initialize the PromptComparer.

        Args:
            judge_model: The OpenAI model to use as judge.
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided).
            log_dir: Optional directory for storing comparison logs.
        """
        self.judge_model = judge_model
        self.api_key = api_key
        self.log_dir = Path(log_dir) if log_dir else None
        self.comparison_history: list[PromptComparisonResult] = []

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def compare(
        self,
        original_prompt: str,
        headroom_modified_prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> PromptComparisonResult:
        """Compare two prompts and track the result.

        Args:
            original_prompt: The original prompt before Headroom processing.
            headroom_modified_prompt: The prompt after Headroom processing.
            metadata: Optional metadata to attach to the result.

        Returns:
            PromptComparisonResult with the comparison outcome.
        """
        result = compare_prompts(
            original_prompt=original_prompt,
            headroom_modified_prompt=headroom_modified_prompt,
            judge_model=self.judge_model,
            api_key=self.api_key,
            metadata=metadata,
        )

        self.comparison_history.append(result)
        return result

    def log_result(self, result: PromptComparisonResult) -> None:
        """Log a comparison result.

        Args:
            result: The comparison result to log.
        """
        # Log to standard logger
        if result.are_equivalent:
            logger.info(
                f"Prompt comparison PASSED - Confidence: {result.confidence}, "
                f"Concern: {result.concern_level}"
            )
        else:
            logger.warning(
                f"Prompt comparison FAILED - Differences: {result.differences}, "
                f"Concern: {result.concern_level}"
            )

        # Log to file if configured
        if self.log_dir:
            log_file = self.log_dir / "comparisons.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(result.to_dict()) + "\n")

    def compare_and_log(
        self,
        original_prompt: str,
        headroom_modified_prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> PromptComparisonResult:
        """Compare two prompts and immediately log the result.

        Args:
            original_prompt: The original prompt before Headroom processing.
            headroom_modified_prompt: The prompt after Headroom processing.
            metadata: Optional metadata to attach to the result.

        Returns:
            PromptComparisonResult with the comparison outcome.
        """
        result = self.compare(original_prompt, headroom_modified_prompt, metadata)
        self.log_result(result)
        return result

    def get_concerning_results(self) -> list[PromptComparisonResult]:
        """Get all results with concerning differences.

        Returns:
            List of results where prompts were not equivalent or had high concern.
        """
        return [r for r in self.comparison_history if r.is_concerning()]

    def summary(self) -> str:
        """Generate a summary of all comparisons.

        Returns:
            Human-readable summary string.
        """
        if not self.comparison_history:
            return "No comparisons performed yet."

        total = len(self.comparison_history)
        equivalent = sum(1 for r in self.comparison_history if r.are_equivalent)
        concerning = len(self.get_concerning_results())

        lines = [
            "=== Prompt Comparison Summary ===",
            f"Total comparisons: {total}",
            f"Equivalent: {equivalent} ({equivalent / total * 100:.1f}%)",
            f"Non-equivalent: {total - equivalent}",
            f"Concerning: {concerning}",
        ]

        if concerning > 0:
            lines.append("\nConcerning comparisons:")
            for result in self.get_concerning_results():
                lines.append(f"  - {result.concern_level}: {result.differences}")

        return "\n".join(lines)

    def export_history(self, path: str | Path) -> None:
        """Export comparison history to a JSON file.

        Args:
            path: Path to save the history.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(
                {
                    "judge_model": self.judge_model,
                    "total_comparisons": len(self.comparison_history),
                    "comparisons": [r.to_dict() for r in self.comparison_history],
                    "summary": {
                        "equivalent_count": sum(
                            1 for r in self.comparison_history if r.are_equivalent
                        ),
                        "concerning_count": len(self.get_concerning_results()),
                    },
                },
                f,
                indent=2,
            )


def compare_messages(
    original_messages: list[dict[str, Any]],
    modified_messages: list[dict[str, Any]],
    judge_model: str = "gpt-4o",
    api_key: str | None = None,
) -> PromptComparisonResult:
    """Compare two message lists (OpenAI/Anthropic format) for semantic equivalence.

    This is useful for comparing full conversation histories, not just single prompts.

    Args:
        original_messages: Original message list before Headroom.
        modified_messages: Modified message list after Headroom.
        judge_model: The OpenAI model to use as judge.
        api_key: OpenAI API key.

    Returns:
        PromptComparisonResult with the comparison outcome.
    """

    # Convert messages to string representation for comparison
    def messages_to_string(messages: list[dict[str, Any]]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Handle content that's a list (multimodal)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            text_parts.append("[IMAGE]")
                        elif item.get("type") == "image":
                            text_parts.append("[IMAGE]")
                    else:
                        text_parts.append(str(item))
                content = " ".join(text_parts)

            parts.append(f"[{role}]: {content}")

        return "\n\n".join(parts)

    original_str = messages_to_string(original_messages)
    modified_str = messages_to_string(modified_messages)

    result = compare_prompts(
        original_prompt=original_str,
        headroom_modified_prompt=modified_str,
        judge_model=judge_model,
        api_key=api_key,
        metadata={
            "comparison_type": "messages",
            "original_message_count": len(original_messages),
            "modified_message_count": len(modified_messages),
        },
    )

    return result


def batch_compare_prompts(
    prompt_pairs: list[tuple[str, str]],
    judge_model: str = "gpt-4o",
    api_key: str | None = None,
    max_concurrent: int = 5,
) -> list[PromptComparisonResult]:
    """Compare multiple prompt pairs in parallel.

    Args:
        prompt_pairs: List of (original, modified) prompt pairs.
        judge_model: The OpenAI model to use as judge.
        api_key: OpenAI API key.
        max_concurrent: Maximum concurrent API calls.

    Returns:
        List of comparison results in the same order as input pairs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[PromptComparisonResult | None] = [None] * len(prompt_pairs)

    def compare_pair(
        index: int, original: str, modified: str
    ) -> tuple[int, PromptComparisonResult]:
        result = compare_prompts(
            original_prompt=original,
            headroom_modified_prompt=modified,
            judge_model=judge_model,
            api_key=api_key,
            metadata={"batch_index": index},
        )
        return index, result

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [
            executor.submit(compare_pair, i, original, modified)
            for i, (original, modified) in enumerate(prompt_pairs)
        ]

        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result

    # Filter out None values (shouldn't happen, but for type safety)
    return [r for r in results if r is not None]


def verify_headroom_preservation(
    original_messages: list[dict[str, Any]],
    headroom_messages: list[dict[str, Any]],
    judge_model: str = "gpt-4o",
    api_key: str | None = None,
    fail_on_difference: bool = False,
) -> PromptComparisonResult:
    """Verify that Headroom preserves the semantic meaning of a request.

    This is the main entry point for verifying Headroom's accuracy preservation.
    It compares the original messages with those modified by Headroom and
    raises an error if differences are found (when fail_on_difference=True).

    Args:
        original_messages: Messages before Headroom processing.
        headroom_messages: Messages after Headroom processing.
        judge_model: The model to use for judging.
        api_key: OpenAI API key.
        fail_on_difference: If True, raise an error when prompts differ.

    Returns:
        PromptComparisonResult with the verification outcome.

    Raises:
        ValueError: If fail_on_difference=True and prompts are not equivalent.

    Example:
        # Capture messages before and after Headroom
        original = [{"role": "user", "content": "Hello, how are you?"}]
        after_headroom = [{"role": "user", "content": "Hello, how are you?"}]

        result = verify_headroom_preservation(
            original, after_headroom, fail_on_difference=True
        )
    """
    result = compare_messages(
        original_messages=original_messages,
        modified_messages=headroom_messages,
        judge_model=judge_model,
        api_key=api_key,
    )

    if fail_on_difference and not result.are_equivalent:
        raise ValueError(
            f"Headroom modified prompt semantics! "
            f"Differences: {result.differences}. "
            f"Reasoning: {result.reasoning}"
        )

    return result
