"""LoCoMo Evaluator V3 - Tests retrieval quality of memory systems.

This evaluator reflects how real-world memory systems work:
1. Store raw conversation turns (no LLM extraction)
2. Retrieve relevant turns for each question
3. Measure retrieval recall against ground truth evidence
4. Pass retrieved context to LLM for answer synthesis

Key insight: The memory system's job is RETRIEVAL, not extraction.
The LLM handles comprehension given the right context.

Metrics:
- Retrieval Recall@k: % of evidence turns found in top-k retrieved
- Retrieval MRR: Mean reciprocal rank of first evidence turn
- End-to-end Accuracy: Using LLM-as-judge for semantic correctness
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm

from headroom.evals.memory.locomo import (
    LOCOMO_CATEGORIES,
    LoCoMoCase,
    LoCoMoConversation,
    load_locomo,
)
from headroom.memory.backends.local import LocalBackend, LocalBackendConfig
from headroom.memory.backends.mem0 import Mem0Backend, Mem0Config
from headroom.memory.models import Memory
from headroom.memory.ports import MemorySearchResult, VectorSearchResult

logger = logging.getLogger(__name__)


@dataclass
class EvalConfigV3:
    """Configuration for V3 memory evaluation.

    This evaluator tests retrieval quality by:
    1. Storing raw dialogue turns as memories
    2. Measuring if evidence turns are retrieved for each question
    """

    n_conversations: int | None = None
    categories: list[int] | None = None
    skip_adversarial: bool = True

    # Backend selection
    backend_type: str = "local"  # "local" or "mem0"

    # Mem0 settings (when backend_type="mem0")
    mem0_api_key: str | None = None  # For Mem0 cloud mode
    mem0_mode: str = "cloud"  # "local" or "cloud"
    mem0_enable_graph: bool = True  # Enable graph storage (Neo4j)

    # Retrieval settings
    top_k: int = 10  # How many turns to retrieve per question
    use_hybrid_search: bool = False  # Combine vector + text search (local only)
    vector_weight: float = 0.5  # Weight for vector similarity (hybrid mode)
    text_weight: float = 0.5  # Weight for text/BM25 matching (hybrid mode)

    # Answer generation
    answer_model: str = "gpt-4o-mini"
    use_llm_judge: bool = True
    judge_model: str = "gpt-4o-mini"

    # Debug
    debug: bool = False
    db_path: str | None = None


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality."""

    recall_at_k: float = 0.0  # % of evidence turns in top-k
    mrr: float = 0.0  # Mean reciprocal rank
    precision_at_k: float = 0.0  # % of retrieved that are evidence

    # Per-category breakdown
    recall_by_category: dict[str, float] = field(default_factory=dict)


@dataclass
class CaseResultV3:
    """Result for a single QA case."""

    case: LoCoMoCase

    # Retrieval results
    retrieved_turn_ids: list[str]
    evidence_turn_ids: list[str]
    retrieval_recall: float  # What % of evidence was retrieved
    retrieval_rank: int | None  # Rank of first evidence (None if not found)

    # Answer results
    predicted_answer: str
    is_correct: bool
    judge_score: float | None = None
    judge_reasoning: str | None = None

    # Context used
    retrieved_context: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.case.question,
            "ground_truth": self.case.answer,
            "predicted": self.predicted_answer,
            "category": self.case.category_name,
            "evidence_turns": self.evidence_turn_ids,
            "retrieved_turns": self.retrieved_turn_ids,
            "retrieval_recall": self.retrieval_recall,
            "retrieval_rank": self.retrieval_rank,
            "is_correct": self.is_correct,
            "judge_score": self.judge_score,
        }


@dataclass
class EvalResultV3:
    """Aggregated evaluation results."""

    total_cases: int

    # Retrieval metrics (the main thing we're testing)
    avg_retrieval_recall: float
    avg_mrr: float
    retrieval_by_category: dict[str, dict[str, float]]

    # End-to-end accuracy
    accuracy: float
    accuracy_by_category: dict[str, float]

    # Individual results
    results: list[CaseResultV3] = field(default_factory=list)

    # Metadata
    config: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "LoCoMo V3 Evaluation Results",
            "(Retrieval-Focused Memory Evaluation)",
            "=" * 60,
            "",
            "RETRIEVAL QUALITY (Memory System Performance):",
            f"  Average Recall@{self.config.get('top_k', 10)}: {self.avg_retrieval_recall:.1%}",
            f"  Mean Reciprocal Rank: {self.avg_mrr:.3f}",
            "",
            "Retrieval by Category:",
        ]

        for cat_name, metrics in sorted(self.retrieval_by_category.items()):
            lines.append(
                f"  {cat_name}: {metrics['recall']:.1%} recall ({metrics['count']:.0f} questions)"
            )

        lines.extend(
            [
                "",
                "END-TO-END ACCURACY (LLM + Retrieval):",
                f"  Overall: {self.accuracy:.1%}",
                "",
                "Accuracy by Category:",
            ]
        )

        for cat_name, acc in sorted(self.accuracy_by_category.items()):
            lines.append(f"  {cat_name}: {acc:.1%}")

        lines.extend(
            [
                "",
                f"Total Duration: {self.duration_seconds:.1f}s",
                f"Total Cases: {self.total_cases}",
            ]
        )

        return "\n".join(lines)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        with open(path, "w") as f:
            json.dump(
                {
                    "total_cases": self.total_cases,
                    "avg_retrieval_recall": self.avg_retrieval_recall,
                    "avg_mrr": self.avg_mrr,
                    "accuracy": self.accuracy,
                    "retrieval_by_category": self.retrieval_by_category,
                    "accuracy_by_category": self.accuracy_by_category,
                    "config": self.config,
                    "results": [r.to_dict() for r in self.results],
                },
                f,
                indent=2,
            )


class LoCoMoEvaluatorV3:
    """Evaluator focused on retrieval quality.

    This evaluator answers the question:
    "Can the memory system retrieve the dialogue turns that contain the answer?"

    Process:
    1. Store each dialogue turn as a separate memory (raw text, no extraction)
    2. For each question, retrieve top-k turns via semantic search
    3. Check if the evidence turns (ground truth) are in the retrieved set
    4. Optionally: generate answer from retrieved context, judge correctness
    """

    def __init__(self, config: EvalConfigV3 | None = None):
        self._config = config or EvalConfigV3()
        self._backend: LocalBackend | Mem0Backend | None = None
        self._turn_id_to_content: dict[str, str] = {}  # Map dia_id -> memory content
        self._debug_logs: list[dict] = []

    async def run(self) -> EvalResultV3:
        start_time = time.time()

        # Load conversations
        conversations = load_locomo(
            n_conversations=self._config.n_conversations,
            categories=self._config.categories,
            skip_adversarial=self._config.skip_adversarial,
        )

        logger.info(f"Loaded {len(conversations)} conversations")

        all_results: list[CaseResultV3] = []

        for conv_idx, conv in enumerate(conversations):
            logger.info(
                f"Processing conversation {conv_idx + 1}/{len(conversations)}: {conv.sample_id}"
            )

            # Fresh backend for each conversation
            if self._config.backend_type == "mem0":
                mem0_config = Mem0Config(
                    mode=self._config.mem0_mode,
                    api_key=self._config.mem0_api_key,
                    enable_graph=self._config.mem0_enable_graph,
                )
                self._backend = Mem0Backend(mem0_config)
                graph_status = "with graph" if self._config.mem0_enable_graph else "vector-only"
                logger.info(f"Using Mem0 backend ({self._config.mem0_mode} mode, {graph_status})")
            else:
                db_dir = tempfile.gettempdir()
                db_path = self._config.db_path or f"{db_dir}/locomo_v3_{uuid.uuid4().hex[:8]}.db"
                self._backend = LocalBackend(LocalBackendConfig(db_path=db_path))
                logger.info("Using LocalBackend")

            self._turn_id_to_content = {}

            try:
                # Phase 1: Store raw dialogue turns
                await self._store_dialogue_turns(conv)

                # Phase 2: Evaluate each question
                for case in conv.qa_cases:
                    result = await self._evaluate_case(case, conv)
                    all_results.append(result)

            finally:
                if self._backend and hasattr(self._backend, "close"):
                    await self._backend.close()  # type: ignore[union-attr]

        # Aggregate results
        duration = time.time() - start_time
        return self._aggregate_results(all_results, duration)

    async def _store_dialogue_turns(self, conv: LoCoMoConversation) -> int:
        """Store each dialogue turn as a memory.

        No LLM extraction - just raw dialogue with metadata.
        """
        assert self._backend is not None, "Backend must be initialized"
        stored = 0
        user_id = f"locomo_{conv.sample_id}"

        for session in conv.sessions:
            for dialogue in session.dialogues:
                # Format: include session date for temporal context
                content = f"[{session.datetime}] {dialogue.speaker}: {dialogue.text}"

                # Store mapping for retrieval checking
                self._turn_id_to_content[dialogue.dia_id] = content

                metadata = {
                    "dia_id": dialogue.dia_id,
                    "speaker": dialogue.speaker,
                    "session_num": session.session_num,
                    "session_datetime": session.datetime,
                }

                # Save to memory system (different interface for Mem0 vs Local)
                try:
                    if isinstance(self._backend, Mem0Backend):
                        memory = Memory(
                            content=content,
                            user_id=user_id,
                            importance=0.5,
                            metadata=metadata,
                        )
                        await self._backend.save_memory(memory)
                    else:
                        await self._backend.save_memory(
                            content=content,
                            user_id=user_id,
                            importance=0.5,
                            metadata=metadata,
                        )
                    stored += 1
                except Exception as e:
                    logger.error(f"Failed to store dialogue {dialogue.dia_id}: {e}")
                    logger.error(f"Content was: {content[:200]}...")
                    raise  # Re-raise to see full traceback

        logger.info(f"Stored {stored} dialogue turns")
        return stored

    async def _evaluate_case(self, case: LoCoMoCase, conv: LoCoMoConversation) -> CaseResultV3:
        """Evaluate a single QA case."""
        assert self._backend is not None, "Backend must be initialized"
        user_id = f"locomo_{conv.sample_id}"

        # Retrieve relevant turns - handle different backend interfaces
        results: list[MemorySearchResult] | list[VectorSearchResult]
        if isinstance(self._backend, Mem0Backend):
            # Mem0 has its own optimized search with graph expansion
            results = await self._backend.search_memories(
                query=case.question,
                user_id=user_id,
                limit=self._config.top_k,
            )
        elif self._config.use_hybrid_search:
            # LocalBackend hybrid search (vector + BM25)
            results = await self._backend.hybrid_search(
                query=case.question,
                user_id=user_id,
                top_k=self._config.top_k,
                vector_weight=self._config.vector_weight,
                text_weight=self._config.text_weight,
            )
        else:
            # LocalBackend vector-only search
            results = await self._backend.search_memories(
                query=case.question,
                user_id=user_id,
                top_k=self._config.top_k,
            )

        # Extract retrieved turn IDs from metadata
        # Note: Mem0 may store our metadata in "custom_metadata", Local stores in "metadata"
        retrieved_ids = []
        retrieved_contents = []
        for r in results:
            metadata = r.memory.metadata or {}
            dia_id = metadata.get("dia_id")
            if dia_id:
                retrieved_ids.append(dia_id)
                retrieved_contents.append(r.memory.content)

        # Calculate retrieval metrics
        evidence_ids = case.evidence or []

        # Recall: what % of evidence turns were retrieved?
        if evidence_ids:
            found = sum(1 for e in evidence_ids if e in retrieved_ids)
            retrieval_recall = found / len(evidence_ids)
        else:
            retrieval_recall = 1.0  # No evidence needed

        # Rank of first evidence turn
        retrieval_rank = None
        for i, rid in enumerate(retrieved_ids):
            if rid in evidence_ids:
                retrieval_rank = i + 1  # 1-indexed
                break

        # Generate answer from retrieved context
        context = "\n".join(retrieved_contents)
        predicted, judge_score, judge_reasoning = await self._generate_answer(
            case.question,
            str(case.answer),
            context,
            conv.speaker_a,
            conv.speaker_b,
        )

        # Determine correctness
        is_correct = judge_score is not None and judge_score >= 0.7

        return CaseResultV3(
            case=case,
            retrieved_turn_ids=retrieved_ids,
            evidence_turn_ids=evidence_ids,
            retrieval_recall=retrieval_recall,
            retrieval_rank=retrieval_rank,
            predicted_answer=predicted,
            is_correct=is_correct,
            judge_score=judge_score,
            judge_reasoning=judge_reasoning,
            retrieved_context=context[:500],  # Truncate for storage
        )

    async def _generate_answer(
        self,
        question: str,
        ground_truth: str,
        context: str,
        speaker_a: str,
        speaker_b: str,
    ) -> tuple[str, float | None, str | None]:
        """Generate answer from context and judge correctness."""

        # Generate answer
        answer_prompt = f"""Based on the following conversation excerpts, answer the question.
If the answer is not in the context, say "Information not found".

Give a SHORT, DIRECT answer - just the specific information requested.

Context:
{context}

Question: {question}

Answer:"""

        try:
            response = await asyncio.to_thread(
                litellm.completion,
                model=self._config.answer_model,
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            predicted = response.choices[0].message.content or "Information not found"
        except Exception as e:
            logger.warning(f"Answer generation failed: {e}")
            predicted = "Error generating answer"

        # Judge correctness
        judge_score = None
        judge_reasoning = None

        if self._config.use_llm_judge:
            judge_prompt = f"""You are evaluating if a predicted answer is semantically correct.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {predicted}

Score the predicted answer from 0.0 to 1.0:
- 1.0: Completely correct, matches ground truth semantically
- 0.7-0.9: Mostly correct, captures the key information
- 0.4-0.6: Partially correct, some relevant information
- 0.1-0.3: Mostly incorrect but shows some understanding
- 0.0: Completely wrong or "not found" when answer exists

Respond in JSON format:
{{"score": 0.0, "reasoning": "brief explanation"}}"""

            try:
                response = await asyncio.to_thread(
                    litellm.completion,
                    model=self._config.judge_model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0.0,
                    max_tokens=150,
                )
                content = response.choices[0].message.content or ""

                # Parse JSON response
                import re

                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    judge_score = float(data.get("score", 0))
                    judge_reasoning = data.get("reasoning", "")
            except Exception as e:
                logger.warning(f"Judge failed: {e}")

        return predicted, judge_score, judge_reasoning

    def _aggregate_results(self, results: list[CaseResultV3], duration: float) -> EvalResultV3:
        """Aggregate individual results."""
        if not results:
            return EvalResultV3(
                total_cases=0,
                avg_retrieval_recall=0.0,
                avg_mrr=0.0,
                retrieval_by_category={},
                accuracy=0.0,
                accuracy_by_category={},
                config={"top_k": self._config.top_k},
                duration_seconds=duration,
            )

        # Overall retrieval metrics
        avg_recall = sum(r.retrieval_recall for r in results) / len(results)

        # MRR: mean of 1/rank for first evidence found
        mrr_sum = 0.0
        for r in results:
            if r.retrieval_rank is not None:
                mrr_sum += 1.0 / r.retrieval_rank
        avg_mrr = mrr_sum / len(results)

        # Per-category retrieval
        retrieval_by_category = {}
        for cat_id, cat_name in LOCOMO_CATEGORIES.items():
            cat_results = [r for r in results if r.case.category == cat_id]
            if cat_results:
                cat_recall = sum(r.retrieval_recall for r in cat_results) / len(cat_results)
                retrieval_by_category[cat_name] = {
                    "recall": cat_recall,
                    "count": len(cat_results),
                }

        # End-to-end accuracy
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / len(results)

        # Per-category accuracy
        accuracy_by_category = {}
        for cat_id, cat_name in LOCOMO_CATEGORIES.items():
            cat_results = [r for r in results if r.case.category == cat_id]
            if cat_results:
                cat_correct = sum(1 for r in cat_results if r.is_correct)
                accuracy_by_category[cat_name] = cat_correct / len(cat_results)

        return EvalResultV3(
            total_cases=len(results),
            avg_retrieval_recall=avg_recall,
            avg_mrr=avg_mrr,
            retrieval_by_category=retrieval_by_category,
            accuracy=accuracy,
            accuracy_by_category=accuracy_by_category,
            results=results,
            config={
                "top_k": self._config.top_k,
                "answer_model": self._config.answer_model,
                "use_llm_judge": self._config.use_llm_judge,
            },
            duration_seconds=duration,
        )


async def run_locomo_eval_v3(
    config: EvalConfigV3 | None = None,
    output_path: Path | str | None = None,
) -> EvalResultV3:
    """Run V3 evaluation."""
    evaluator = LoCoMoEvaluatorV3(config)
    result = await evaluator.run()

    if output_path:
        result.save(output_path)
        logger.info(f"Results saved to {output_path}")

    return result


def run_locomo_eval_v3_sync(
    config: EvalConfigV3 | None = None,
    output_path: Path | str | None = None,
) -> EvalResultV3:
    """Synchronous wrapper."""
    return asyncio.run(run_locomo_eval_v3(config, output_path))
