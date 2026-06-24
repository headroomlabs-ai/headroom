"""LoCoMo evaluation runner for memory system benchmarking.

This module implements the evaluation pipeline for testing memory systems
against the LoCoMo benchmark. It stores conversations as memories, queries
with questions, and scores the answers.

Metrics:
- F1 score (token overlap)
- Exact match
- LLM-as-judge (optional)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from headroom.evals.memory.locomo import (
    LOCOMO_CATEGORIES,
    LoCoMoCase,
    LoCoMoConversation,
    load_locomo,
)
from headroom.evals.metrics import compute_exact_match, compute_f1
from headroom.memory import HierarchicalMemory, MemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class MemoryEvalConfig:
    """Configuration for memory evaluation.

    Attributes:
        n_conversations: Number of conversations to evaluate (None = all).
        categories: LoCoMo question categories to include (1-5).
        skip_adversarial: Skip category 5 (unanswerable questions).
        top_k_memories: Number of memories to retrieve for each question (0 = all).
        llm_judge_enabled: Whether to use LLM-as-judge scoring.
        llm_judge_model: Model to use for LLM-as-judge (e.g., "gpt-4o").
        memory_config: Configuration for the memory system.
        batch_size: Batch size for memory storage operations.
        f1_threshold: F1 score threshold for "correct" answer.
        progress_callback: Optional callback for progress updates.
        extract_memories: Use LLM to extract facts from dialogue (recommended).
        extraction_model: Model for memory extraction (e.g., "gpt-4o-mini").
        pass_all_memories: Pass ALL memories to LLM instead of retrieval (Path A).
        parallel_workers: Number of parallel workers for LLM calls.
        debug: Enable debug logging.
    """

    n_conversations: int | None = None
    categories: list[int] | None = None
    skip_adversarial: bool = True
    top_k_memories: int = 10
    llm_judge_enabled: bool = False
    llm_judge_model: str = "gpt-4o"
    memory_config: MemoryConfig | None = None
    batch_size: int = 50
    f1_threshold: float = 0.5
    progress_callback: Callable[[str, int, int], None] | None = None
    extract_memories: bool = True  # Use LLM extraction by default
    extraction_model: str = "gpt-4o-mini"
    pass_all_memories: bool = False  # Path A: pass all memories, no retrieval
    parallel_workers: int = 10  # Parallel LLM calls
    debug: bool = False  # Debug logging


@dataclass
class MemoryEvalResult:
    """Result from evaluating a single LoCoMo case."""

    case: LoCoMoCase
    predicted_answer: str
    retrieved_memories: list[str]
    retrieval_scores: list[float]

    # Core metrics
    f1_score: float
    exact_match: bool
    is_correct: bool

    # Optional LLM judge
    llm_judge_score: float | None = None
    llm_judge_reasoning: str | None = None

    # Timing
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.case.question,
            "ground_truth": self.case.answer,
            "predicted": self.predicted_answer,
            "category": self.case.category_name,
            "category_id": self.case.category,
            "conversation_id": self.case.conversation_id,
            "f1_score": self.f1_score,
            "exact_match": self.exact_match,
            "is_correct": self.is_correct,
            "llm_judge_score": self.llm_judge_score,
            "llm_judge_reasoning": self.llm_judge_reasoning,
            "num_memories_retrieved": len(self.retrieved_memories),
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
        }


@dataclass
class MemoryEvalSuiteResult:
    """Aggregated results from LoCoMo evaluation."""

    total_cases: int
    correct_cases: int
    accuracy: float

    # Aggregate metrics
    avg_f1_score: float
    exact_match_rate: float
    avg_llm_judge_score: float | None

    # Per-category metrics
    metrics_by_category: dict[str, dict[str, float]]

    # Individual results
    results: list[MemoryEvalResult] = field(default_factory=list)

    # Timing
    total_duration_seconds: float = 0.0
    avg_retrieval_latency_ms: float = 0.0
    avg_generation_latency_ms: float = 0.0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "correct_cases": self.correct_cases,
            "accuracy": self.accuracy,
            "avg_f1_score": self.avg_f1_score,
            "exact_match_rate": self.exact_match_rate,
            "avg_llm_judge_score": self.avg_llm_judge_score,
            "metrics_by_category": self.metrics_by_category,
            "total_duration_seconds": self.total_duration_seconds,
            "avg_retrieval_latency_ms": self.avg_retrieval_latency_ms,
            "avg_generation_latency_ms": self.avg_generation_latency_ms,
            "timestamp": self.timestamp,
            "config": self.config,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: Path | str) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "LoCoMo Memory Evaluation Results",
            "=" * 60,
            f"Total Cases: {self.total_cases}",
            f"Accuracy: {self.accuracy:.1%} ({self.correct_cases}/{self.total_cases})",
            f"Average F1 Score: {self.avg_f1_score:.3f}",
            f"Exact Match Rate: {self.exact_match_rate:.1%}",
        ]

        if self.avg_llm_judge_score is not None:
            lines.append(f"Average LLM Judge Score: {self.avg_llm_judge_score:.2f}/5")

        lines.append("")
        lines.append("Results by Category:")
        lines.append("-" * 40)

        for cat_name, metrics in sorted(self.metrics_by_category.items()):
            lines.append(
                f"  {cat_name}: {metrics['accuracy']:.1%} accuracy, "
                f"{metrics['avg_f1']:.3f} F1 ({metrics['count']:.0f} questions)"
            )

        lines.append("")
        lines.append(f"Total Duration: {self.total_duration_seconds:.1f}s")
        lines.append(f"Avg Retrieval Latency: {self.avg_retrieval_latency_ms:.1f}ms")
        lines.append(f"Avg Generation Latency: {self.avg_generation_latency_ms:.1f}ms")

        return "\n".join(lines)


class LoCoMoEvaluator:
    """Evaluator for LoCoMo memory benchmark.

    This class orchestrates the full evaluation pipeline:
    1. Load LoCoMo conversations
    2. Store conversation dialogues as memories
    3. Query with questions and retrieve relevant memories
    4. Generate answers using an LLM
    5. Score answers against ground truth

    Usage:
        evaluator = LoCoMoEvaluator(
            answer_fn=my_llm_answer_function,
            config=MemoryEvalConfig(n_conversations=5),
        )
        result = await evaluator.run()
        print(result.summary())
    """

    def __init__(
        self,
        answer_fn: Callable[[str, list[str]], str] | None = None,
        llm_judge_fn: Callable[[str, str, str], tuple[float, str]] | None = None,
        config: MemoryEvalConfig | None = None,
    ):
        """Initialize the LoCoMo evaluator.

        Args:
            answer_fn: Function that takes (question, memories) and returns an answer.
                       If None, uses a simple retrieval-based answerer.
            llm_judge_fn: Function that takes (question, ground_truth, prediction)
                          and returns (score 0-5, reasoning). Optional.
            config: Evaluation configuration.
        """
        self.answer_fn = answer_fn or self._default_answer_fn
        self.llm_judge_fn = llm_judge_fn
        self.config = config or MemoryEvalConfig()
        self.memory: HierarchicalMemory | None = None
        # Store all memories per conversation for Path A (pass all memories)
        self._all_memories: dict[str, list[str]] = {}
        self._debug_logs: list[dict] = []

    def _default_answer_fn(self, question: str, memories: list[str]) -> str:
        """Default answer function that concatenates relevant memories.

        This is a simple baseline - real evaluations should use an LLM.
        """
        if not memories:
            return "I don't have information about that."

        # Return the most relevant memory as the answer
        # In practice, you'd pass this to an LLM
        return memories[0]

    async def _setup_memory(self) -> HierarchicalMemory:
        """Create and configure the memory system."""
        config = self.config.memory_config or MemoryConfig()
        return await HierarchicalMemory.create(config)

    def _extract_memories_from_session(
        self,
        session_text: str,
        session_datetime: str,
        speaker_a: str,
        speaker_b: str,
    ) -> list[dict[str, str]]:
        """Use LLM to extract key facts from a session.

        Args:
            session_text: Full session dialogue text.
            session_datetime: When the session occurred.
            speaker_a: Name of first speaker.
            speaker_b: Name of second speaker.

        Returns:
            List of extracted memory dicts with 'content' and 'category'.
        """
        try:
            import litellm
        except ImportError:
            logger.warning("litellm not available, falling back to raw dialogue storage")
            return []

        prompt = f"""Extract key facts from this conversation. This is critical for answering questions later.

SESSION DATE: {session_datetime}
SPEAKERS: {speaker_a} and {speaker_b}

CONVERSATION:
{session_text}

EXTRACTION RULES:
1. **DATES ARE CRITICAL**:
   - If a specific date is mentioned (e.g., "7 May", "January 15th"), ALWAYS include it exactly
   - Convert ALL relative dates to ABSOLUTE dates using the session date ({session_datetime}):
     * "last year" → calculate the year (if session is 2023, last year = 2022)
     * "yesterday" → calculate the exact date
     * "next month" → calculate the month and year
     * "last Saturday" → calculate the exact date
   - NEVER use relative terms like "last year", "next month", "yesterday" in your output

2. **COMPLETE FACTS**: Each memory must be self-contained with:
   - WHO (use their name: {speaker_a} or {speaker_b})
   - WHAT happened or what the fact is
   - WHEN (specific date/time if it's an event)

3. **WHAT TO EXTRACT**:
   - Personal info (identity, job, relationships, age, location)
   - Events with dates (when did something happen)
   - Plans and intentions (what they plan to do and when)
   - Preferences and opinions
   - Experiences (places visited, things done)

OUTPUT FORMAT (JSON only):
{{"memories": [
  {{"content": "On 7 May 2023, Caroline attended an LGBTQ support group.", "category": "event"}},
  {{"content": "Melanie painted a sunrise in 2022.", "category": "event"}},
  {{"content": "Jon lost his job as a banker on 19 January 2023.", "category": "event"}}
]}}

IMPORTANT: Every event MUST have a specific date. If you cannot determine the date, state it as "around {session_datetime}"."""

        try:
            response = litellm.completion(
                model=self.config.extraction_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
            )
            content = response.choices[0].message.content or ""

            # Parse JSON from response
            import json
            import re

            # Try to find JSON in response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                memories: list[dict[str, str]] = data.get("memories", [])
                return memories
        except Exception as e:
            logger.warning(f"Memory extraction failed: {e}")

        return []

    async def _store_conversation(
        self,
        conversation: LoCoMoConversation,
    ) -> int:
        """Store a conversation's dialogues as memories.

        Args:
            conversation: The LoCoMo conversation to store.

        Returns:
            Number of memories stored.
        """
        if self.memory is None:
            raise RuntimeError("Memory system not initialized")

        memories_data: list[dict[str, Any]] = []
        user_id = f"locomo_{conversation.sample_id}"

        if self.config.extract_memories:
            # Use LLM to extract facts from each session
            for session in conversation.sessions:
                session_id = f"session_{session.session_num}"

                # Get full session text
                session_text = session.text

                # Extract memories using LLM
                extracted = self._extract_memories_from_session(
                    session_text=session_text,
                    session_datetime=session.datetime,
                    speaker_a=conversation.speaker_a,
                    speaker_b=conversation.speaker_b,
                )

                for mem in extracted:
                    memories_data.append(
                        {
                            "content": mem.get("content", ""),
                            "user_id": user_id,
                            "session_id": session_id,
                            "importance": 0.7,  # Extracted facts are more important
                            "metadata": {
                                "session_datetime": session.datetime,
                                "session_num": session.session_num,
                                "extracted": True,
                                "category": mem.get("category", "fact"),
                            },
                        }
                    )

            logger.info(
                f"Extracted {len(memories_data)} memories from {len(conversation.sessions)} sessions"
            )
        else:
            # Fallback: store raw dialogue turns
            for session in conversation.sessions:
                session_id = f"session_{session.session_num}"

                for dialogue in session.dialogues:
                    date_prefix = f"[{session.datetime}] " if session.datetime else ""
                    content = f"{date_prefix}{dialogue.to_message_format()}"

                    metadata = {
                        "speaker": dialogue.speaker,
                        "session_datetime": session.datetime,
                        "session_num": session.session_num,
                        "dia_id": dialogue.dia_id,
                    }

                    if dialogue.image_url:
                        metadata["has_image"] = True
                        metadata["image_caption"] = dialogue.image_caption

                    memories_data.append(
                        {
                            "content": content,
                            "user_id": user_id,
                            "session_id": session_id,
                            "importance": 0.5,
                            "metadata": metadata,
                        }
                    )

        # Store in batches
        total_stored = 0
        for i in range(0, len(memories_data), self.config.batch_size):
            batch = memories_data[i : i + self.config.batch_size]
            await self.memory.add_batch(batch)
            total_stored += len(batch)

        # For Path A: store all memories in dict for direct access
        self._all_memories[user_id] = [m["content"] for m in memories_data]

        if self.config.debug:
            self._debug_logs.append(
                {
                    "event": "memories_stored",
                    "conversation_id": conversation.sample_id,
                    "num_memories": total_stored,
                    "sample_memories": [m["content"][:100] for m in memories_data[:5]],
                }
            )

        logger.info(f"Stored {total_stored} memories for conversation {conversation.sample_id}")
        return total_stored

    async def _retrieve_memories(
        self,
        question: str,
        conversation_id: str,
    ) -> tuple[list[str], list[float], float]:
        """Retrieve relevant memories for a question.

        Args:
            question: The question to answer.
            conversation_id: The conversation ID (for scoping).

        Returns:
            Tuple of (memory_contents, similarity_scores, latency_ms).
        """
        user_id = f"locomo_{conversation_id}"
        start = time.time()

        # Path A: Return ALL memories (no retrieval bottleneck)
        if self.config.pass_all_memories:
            all_mems = self._all_memories.get(user_id, [])
            latency_ms = (time.time() - start) * 1000

            if self.config.debug:
                self._debug_logs.append(
                    {
                        "event": "retrieve_all_memories",
                        "question": question[:100],
                        "conversation_id": conversation_id,
                        "num_memories": len(all_mems),
                    }
                )

            # Return all memories with score 1.0 (no ranking)
            return all_mems, [1.0] * len(all_mems), latency_ms

        # Path B: Use vector retrieval (original approach)
        if self.memory is None:
            raise RuntimeError("Memory system not initialized")

        results = await self.memory.search(
            query=question,
            user_id=user_id,
            top_k=self.config.top_k_memories,
        )
        latency_ms = (time.time() - start) * 1000

        memories = [r.memory.content for r in results]
        scores = [r.similarity for r in results]

        if self.config.debug:
            self._debug_logs.append(
                {
                    "event": "retrieve_top_k",
                    "question": question[:100],
                    "conversation_id": conversation_id,
                    "num_retrieved": len(memories),
                    "top_scores": scores[:3] if scores else [],
                    "top_memories": [m[:80] for m in memories[:3]],
                }
            )

        return memories, scores, latency_ms

    async def _evaluate_case(
        self,
        case: LoCoMoCase,
    ) -> MemoryEvalResult:
        """Evaluate a single QA case.

        Args:
            case: The LoCoMo case to evaluate.

        Returns:
            Evaluation result with metrics.
        """
        # Retrieve relevant memories
        memories, scores, retrieval_latency = await self._retrieve_memories(
            case.question, case.conversation_id
        )

        # Generate answer
        start = time.time()
        predicted_answer = self.answer_fn(case.question, memories)
        generation_latency = (time.time() - start) * 1000

        # Compute metrics
        ground_truth = str(case.answer) if case.answer is not None else ""
        f1_score = compute_f1(predicted_answer, ground_truth)
        exact_match = compute_exact_match(predicted_answer, ground_truth)

        # Determine correctness
        is_correct = f1_score >= self.config.f1_threshold

        # LLM judge scoring (optional)
        llm_judge_score = None
        llm_judge_reasoning = None

        if self.config.llm_judge_enabled and self.llm_judge_fn:
            try:
                llm_judge_score, llm_judge_reasoning = self.llm_judge_fn(
                    case.question, ground_truth, predicted_answer
                )
                # Use LLM judge for correctness if available
                is_correct = llm_judge_score >= 3.0  # Score 3+ out of 5 = correct
            except Exception as e:
                logger.warning(f"LLM judge failed: {e}")

        # Debug logging
        if self.config.debug:
            self._debug_logs.append(
                {
                    "event": "evaluate_case",
                    "question": case.question,
                    "ground_truth": ground_truth,
                    "predicted": predicted_answer[:200],
                    "category": case.category_name,
                    "num_memories": len(memories),
                    "f1_score": f1_score,
                    "llm_judge_score": llm_judge_score,
                    "is_correct": is_correct,
                }
            )

        return MemoryEvalResult(
            case=case,
            predicted_answer=predicted_answer,
            retrieved_memories=memories,
            retrieval_scores=scores,
            f1_score=f1_score,
            exact_match=exact_match,
            is_correct=is_correct,
            llm_judge_score=llm_judge_score,
            llm_judge_reasoning=llm_judge_reasoning,
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=generation_latency,
        )

    def _aggregate_results(
        self,
        results: list[MemoryEvalResult],
        duration_seconds: float,
    ) -> MemoryEvalSuiteResult:
        """Aggregate individual results into suite result."""
        if not results:
            return MemoryEvalSuiteResult(
                total_cases=0,
                correct_cases=0,
                accuracy=0.0,
                avg_f1_score=0.0,
                exact_match_rate=0.0,
                avg_llm_judge_score=None,
                metrics_by_category={},
                total_duration_seconds=duration_seconds,
            )

        # Overall metrics
        correct = sum(1 for r in results if r.is_correct)
        total = len(results)

        avg_f1 = sum(r.f1_score for r in results) / total
        exact_match_count = sum(1 for r in results if r.exact_match)

        # LLM judge scores
        llm_scores = [r.llm_judge_score for r in results if r.llm_judge_score is not None]
        avg_llm_judge = sum(llm_scores) / len(llm_scores) if llm_scores else None

        # Per-category metrics
        metrics_by_category: dict[str, dict[str, float]] = {}
        for cat_id, cat_name in LOCOMO_CATEGORIES.items():
            cat_results = [r for r in results if r.case.category == cat_id]
            if cat_results:
                cat_correct = sum(1 for r in cat_results if r.is_correct)
                cat_f1 = sum(r.f1_score for r in cat_results) / len(cat_results)
                metrics_by_category[cat_name] = {
                    "count": len(cat_results),
                    "accuracy": cat_correct / len(cat_results),
                    "avg_f1": cat_f1,
                    "correct": cat_correct,
                }

        # Timing
        avg_retrieval = sum(r.retrieval_latency_ms for r in results) / total
        avg_generation = sum(r.generation_latency_ms for r in results) / total

        return MemoryEvalSuiteResult(
            total_cases=total,
            correct_cases=correct,
            accuracy=correct / total,
            avg_f1_score=avg_f1,
            exact_match_rate=exact_match_count / total,
            avg_llm_judge_score=avg_llm_judge,
            metrics_by_category=metrics_by_category,
            results=results,
            total_duration_seconds=duration_seconds,
            avg_retrieval_latency_ms=avg_retrieval,
            avg_generation_latency_ms=avg_generation,
            config={
                "n_conversations": self.config.n_conversations,
                "categories": self.config.categories,
                "top_k_memories": self.config.top_k_memories,
                "f1_threshold": self.config.f1_threshold,
                "llm_judge_enabled": self.config.llm_judge_enabled,
            },
        )

    async def run(
        self,
        conversations: list[LoCoMoConversation] | None = None,
    ) -> MemoryEvalSuiteResult:
        """Run the full LoCoMo evaluation.

        Args:
            conversations: Optional pre-loaded conversations. If None, loads from dataset.

        Returns:
            Aggregated evaluation results.
        """
        start_time = time.time()

        # Load conversations if not provided
        if conversations is None:
            conversations = load_locomo(
                n_conversations=self.config.n_conversations,
                categories=self.config.categories,
                skip_adversarial=self.config.skip_adversarial,
            )

        # Initialize memory system
        logger.info("Initializing memory system...")
        self.memory = await self._setup_memory()

        # Store all conversations
        logger.info(f"Storing {len(conversations)} conversations...")
        total_memories = 0
        for i, conv in enumerate(conversations):
            memories_stored = await self._store_conversation(conv)
            total_memories += memories_stored

            if self.config.progress_callback:
                self.config.progress_callback("storing", i + 1, len(conversations))

        logger.info(f"Stored {total_memories} total memories")

        # Collect all QA cases
        all_cases: list[LoCoMoCase] = []
        for conv in conversations:
            all_cases.extend(conv.qa_cases)

        logger.info(f"Evaluating {len(all_cases)} QA cases...")

        # Evaluate cases (with parallelization if configured)
        results: list[MemoryEvalResult] = []

        if self.config.parallel_workers > 1:
            # Parallel evaluation using semaphore to limit concurrency
            import asyncio

            semaphore = asyncio.Semaphore(self.config.parallel_workers)
            completed = 0

            async def eval_with_semaphore(case: LoCoMoCase) -> MemoryEvalResult:
                nonlocal completed
                async with semaphore:
                    result = await self._evaluate_case(case)
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"Evaluated {completed}/{len(all_cases)} cases")
                    return result

            # Run all evaluations in parallel (limited by semaphore)
            results = await asyncio.gather(*[eval_with_semaphore(c) for c in all_cases])
            results = list(results)
        else:
            # Sequential evaluation
            for i, case in enumerate(all_cases):
                result = await self._evaluate_case(case)
                results.append(result)

                if self.config.progress_callback:
                    self.config.progress_callback("evaluating", i + 1, len(all_cases))

                if (i + 1) % 10 == 0:
                    logger.info(f"Evaluated {i + 1}/{len(all_cases)} cases")

        duration = time.time() - start_time

        # Aggregate results
        suite_result = self._aggregate_results(results, duration)

        # Save debug logs if enabled
        if self.config.debug and self._debug_logs:
            suite_result.config["debug_logs"] = self._debug_logs

        logger.info(f"Evaluation complete in {duration:.1f}s")
        logger.info(f"Accuracy: {suite_result.accuracy:.1%}")

        return suite_result


async def run_locomo_eval(
    answer_fn: Callable[[str, list[str]], str],
    config: MemoryEvalConfig | None = None,
    llm_judge_fn: Callable[[str, str, str], tuple[float, str]] | None = None,
    output_path: Path | str | None = None,
) -> MemoryEvalSuiteResult:
    """Convenience function to run LoCoMo evaluation.

    Args:
        answer_fn: Function that takes (question, memories) and returns answer.
        config: Evaluation configuration.
        llm_judge_fn: Optional LLM judge function.
        output_path: Optional path to save results JSON.

    Returns:
        Evaluation results.

    Example:
        def my_answer_fn(question: str, memories: list[str]) -> str:
            # Use your LLM to answer based on retrieved memories
            context = "\\n".join(memories)
            return llm.complete(f"Context: {context}\\n\\nQuestion: {question}")

        result = await run_locomo_eval(my_answer_fn)
        print(result.summary())
    """
    evaluator = LoCoMoEvaluator(
        answer_fn=answer_fn,
        llm_judge_fn=llm_judge_fn,
        config=config,
    )

    result = await evaluator.run()

    if output_path:
        result.save(output_path)
        logger.info(f"Results saved to {output_path}")

    return result


# Synchronous wrapper for convenience
def run_locomo_eval_sync(
    answer_fn: Callable[[str, list[str]], str],
    config: MemoryEvalConfig | None = None,
    llm_judge_fn: Callable[[str, str, str], tuple[float, str]] | None = None,
    output_path: Path | str | None = None,
) -> MemoryEvalSuiteResult:
    """Synchronous wrapper for run_locomo_eval."""
    return asyncio.run(run_locomo_eval(answer_fn, config, llm_judge_fn, output_path))
