"""LoCoMo Evaluator V2 - Tests LLM-controlled memory with tools.

This evaluator tests the new architecture where:
1. LLM decides what to save (memory_save tool)
2. LLM decides when to search (memory_search tool)
3. Graph relationships enable multi-hop reasoning

The key difference from V1 is that instead of using heuristics or
explicit extraction, the LLM autonomously decides what memories
are worth saving and how to search for relevant information.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
import uuid
from collections.abc import Callable
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
from headroom.evals.metrics import compute_exact_match, compute_f1
from headroom.memory.backends.local import LocalBackend, LocalBackendConfig
from headroom.memory.models import Memory
from headroom.memory.system import MemoryBackend
from headroom.memory.tools import MEMORY_TOOLS

logger = logging.getLogger(__name__)


# =============================================================================
# Metrics Dataclasses
# =============================================================================


@dataclass
class EvalMetrics:
    """Metrics tracking for V2 evaluation.

    Tracks accuracy, memory save/search behavior, and graph operations.
    """

    # Accuracy metrics
    total_questions: int = 0
    correct_answers: int = 0
    accuracy_by_category: dict[str, float] = field(default_factory=dict)

    # Memory save metrics
    total_turns: int = 0
    saves_attempted: int = 0
    saves_successful: int = 0
    save_precision: float = 0.0  # Did LLM save the right things?
    save_recall: float = 0.0  # Did LLM save all important things?

    # Search metrics
    total_searches: int = 0
    search_latency_ms: list[float] = field(default_factory=list)
    searches_with_results: int = 0
    avg_results_per_search: float = 0.0

    # Graph metrics
    graph_expansions: int = 0
    multi_hop_accuracy: float = 0.0
    entities_created: int = 0
    relationships_created: int = 0

    # LLM call metrics
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_llm_latency_ms: float = 0.0

    def summary(self) -> str:
        """Return formatted summary of metrics."""
        lines = [
            "=" * 60,
            "LoCoMo V2 Evaluation Metrics",
            "=" * 60,
            "",
            "Accuracy Metrics:",
            f"  Total Questions: {self.total_questions}",
            f"  Correct Answers: {self.correct_answers}",
            f"  Overall Accuracy: {self.correct_answers / max(1, self.total_questions):.1%}",
            "",
            "Memory Save Metrics:",
            f"  Total Turns Processed: {self.total_turns}",
            f"  Save Attempts: {self.saves_attempted}",
            f"  Successful Saves: {self.saves_successful}",
            f"  Save Precision: {self.save_precision:.1%}",
            f"  Save Recall: {self.save_recall:.1%}",
            "",
            "Search Metrics:",
            f"  Total Searches: {self.total_searches}",
            f"  Searches with Results: {self.searches_with_results}",
            f"  Avg Results per Search: {self.avg_results_per_search:.1f}",
            f"  Avg Search Latency: {sum(self.search_latency_ms) / max(1, len(self.search_latency_ms)):.1f}ms",
            "",
            "Graph Metrics:",
            f"  Entities Created: {self.entities_created}",
            f"  Relationships Created: {self.relationships_created}",
            f"  Graph Expansions: {self.graph_expansions}",
            f"  Multi-hop Accuracy: {self.multi_hop_accuracy:.1%}",
            "",
            "LLM Metrics:",
            f"  Total LLM Calls: {self.total_llm_calls}",
            f"  Total Input Tokens: {self.total_input_tokens:,}",
            f"  Total Output Tokens: {self.total_output_tokens:,}",
            f"  Avg LLM Latency: {self.total_llm_latency_ms / max(1, self.total_llm_calls):.0f}ms",
            "",
        ]

        if self.accuracy_by_category:
            lines.append("Accuracy by Category:")
            for cat_name, acc in sorted(self.accuracy_by_category.items()):
                lines.append(f"  {cat_name}: {acc:.1%}")

        return "\n".join(lines)


@dataclass
class MemoryEvalConfigV2:
    """Configuration for V2 memory evaluation.

    Attributes:
        n_conversations: Number of conversations to evaluate (None = all).
        categories: LoCoMo question categories to include (1-5).
        skip_adversarial: Skip category 5 (unanswerable questions).
        llm_judge_enabled: Whether to use LLM-as-judge scoring.
        llm_judge_model: Model to use for LLM-as-judge.
        f1_threshold: F1 score threshold for "correct" answer.
        parallel_workers: Number of parallel workers for LLM calls.
        debug: Enable debug logging.
        save_model: Model for deciding what to save.
        answer_model: Model for answering questions.
        max_search_results: Maximum memories to retrieve per search.
        include_graph_expansion: Whether to expand search via graph.
        db_path: Path to memory database (use temp by default).
        backend_factory: Optional factory callable to create custom backends.
            Takes a user_id string and returns a MemoryBackend instance.
            If None, defaults to creating a LocalBackend.
    """

    n_conversations: int | None = None
    categories: list[int] | None = None
    skip_adversarial: bool = True
    llm_judge_enabled: bool = False
    llm_judge_model: str = "gpt-4o"
    f1_threshold: float = 0.5
    parallel_workers: int = 5
    debug: bool = False
    save_model: str = "gpt-4o-mini"
    answer_model: str = "gpt-4o"
    max_search_results: int = 10
    include_graph_expansion: bool = True
    db_path: str | None = None  # None = use temp file
    backend_factory: Callable[[str], MemoryBackend] | None = None  # user_id -> backend


@dataclass
class MemoryEvalResultV2:
    """Result from evaluating a single LoCoMo case with V2."""

    case: LoCoMoCase
    predicted_answer: str
    searched_memories: list[str]
    search_queries: list[str]

    # Core metrics
    f1_score: float
    exact_match: bool
    is_correct: bool

    # Optional LLM judge
    llm_judge_score: float | None = None
    llm_judge_reasoning: str | None = None

    # Timing
    search_latency_ms: float = 0.0
    answer_latency_ms: float = 0.0

    # Debug info
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
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
            "num_memories_searched": len(self.searched_memories),
            "search_queries": self.search_queries,
            "search_latency_ms": self.search_latency_ms,
            "answer_latency_ms": self.answer_latency_ms,
            "tool_calls": self.tool_calls,
        }


@dataclass
class MemoryEvalSuiteResultV2:
    """Aggregated results from LoCoMo V2 evaluation."""

    total_cases: int
    correct_cases: int
    accuracy: float

    # Aggregate metrics
    avg_f1_score: float
    exact_match_rate: float
    avg_llm_judge_score: float | None

    # Per-category metrics
    metrics_by_category: dict[str, dict[str, float]]

    # Detailed metrics
    metrics: EvalMetrics

    # Individual results
    results: list[MemoryEvalResultV2] = field(default_factory=list)

    # Timing
    total_duration_seconds: float = 0.0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: dict[str, Any] = field(default_factory=dict)

    # Debug logs
    debug_logs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "correct_cases": self.correct_cases,
            "accuracy": self.accuracy,
            "avg_f1_score": self.avg_f1_score,
            "exact_match_rate": self.exact_match_rate,
            "avg_llm_judge_score": self.avg_llm_judge_score,
            "metrics_by_category": self.metrics_by_category,
            "total_duration_seconds": self.total_duration_seconds,
            "timestamp": self.timestamp,
            "config": self.config,
            "results": [r.to_dict() for r in self.results],
            "debug_logs": self.debug_logs,
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
            "LoCoMo V2 Memory Evaluation Results",
            "(LLM-Controlled Memory Architecture)",
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

        # Add detailed metrics summary
        lines.append("")
        lines.append(self.metrics.summary())

        return "\n".join(lines)


# =============================================================================
# V2 Evaluator
# =============================================================================


class LoCoMoEvaluatorV2:
    """
    Evaluates memory system with LLM-controlled tools.

    Process:
    1. For each conversation, replay turns letting LLM save memories
    2. For each question, let LLM search memories to answer
    3. Compare answers to ground truth
    4. Track metrics (accuracy, save precision, search latency)

    The key innovation is that the LLM decides autonomously:
    - WHAT to save (via memory_save tool)
    - WHEN to search (via memory_search tool)
    - HOW to answer based on retrieved memories
    """

    def __init__(
        self,
        backend: MemoryBackend | None = None,
        answer_model: str = "gpt-4o",
        config: MemoryEvalConfigV2 | None = None,
    ):
        """Initialize the V2 evaluator.

        Args:
            backend: Memory backend to use. If None, creates backend per conversation.
            answer_model: LLM model for answering questions.
            config: Evaluation configuration.
        """
        self._backend: MemoryBackend | LocalBackend | None = backend
        self._answer_model = answer_model
        self._config = config or MemoryEvalConfigV2()
        self._metrics = EvalMetrics()
        self._debug_logs: list[dict[str, Any]] = []
        self._current_user_id: str = ""

    async def run(self) -> MemoryEvalSuiteResultV2:
        """Run the full evaluation.

        Returns:
            Aggregated evaluation results with detailed metrics.
        """
        start_time = time.time()

        # Load LoCoMo data
        conversations = load_locomo(
            n_conversations=self._config.n_conversations,
            categories=self._config.categories,
            skip_adversarial=self._config.skip_adversarial,
        )

        logger.info(
            f"Loaded {len(conversations)} conversations with "
            f"{sum(len(c.qa_cases) for c in conversations)} questions"
        )

        all_results: list[MemoryEvalResultV2] = []

        # Process each conversation
        for conv_idx, conversation in enumerate(conversations):
            logger.info(
                f"Processing conversation {conv_idx + 1}/{len(conversations)}: "
                f"{conversation.sample_id}"
            )

            # Create fresh backend for this conversation
            self._current_user_id = f"locomo_{conversation.sample_id}"

            if self._config.backend_factory is not None:
                # Use custom backend factory
                self._backend = self._config.backend_factory(self._current_user_id)
            else:
                # Default to LocalBackend
                db_dir = tempfile.gettempdir()
                db_path = self._config.db_path or f"{db_dir}/locomo_v2_{uuid.uuid4().hex[:8]}.db"
                backend_config = LocalBackendConfig(db_path=db_path)
                self._backend = LocalBackend(backend_config)

            try:
                # Phase 1: Replay conversation, letting LLM save memories
                saved_memories = await self._replay_conversation(conversation)
                logger.info(f"  Saved {len(saved_memories)} memories from conversation")

                # Phase 2: Answer questions using memory tools
                conv_results = await self._evaluate_questions(conversation)
                all_results.extend(conv_results)

                logger.info(
                    f"  Answered {len(conv_results)} questions, "
                    f"{sum(1 for r in conv_results if r.is_correct)} correct"
                )

            finally:
                # Clean up backend
                if self._backend and hasattr(self._backend, "close"):
                    await self._backend.close()  # type: ignore[union-attr]

        # Calculate final metrics
        duration = time.time() - start_time
        suite_result = self._aggregate_results(all_results, duration)

        logger.info(f"Evaluation complete in {duration:.1f}s")
        logger.info(f"Overall accuracy: {suite_result.accuracy:.1%}")

        return suite_result

    async def _replay_conversation(
        self,
        conversation: LoCoMoConversation,
    ) -> list[Memory]:
        """
        Replay a conversation, letting LLM decide what to save.

        For each session/turn, we present the dialogue to the LLM with
        the memory_save tool available. The LLM autonomously decides
        whether to save information and how to categorize it.

        Args:
            conversation: The LoCoMo conversation to replay.

        Returns:
            List of saved memories.
        """
        saved_memories: list[Memory] = []

        # System prompt for the memory extraction phase
        system_prompt = f"""You are an AI assistant processing a conversation between {conversation.speaker_a} and {conversation.speaker_b}.

Your task is to identify and save important information that would be useful for answering questions later.

IMPORTANT GUIDELINES:
1. Save facts about people, events, dates, preferences, and relationships
2. Each memory should be self-contained and include:
   - WHO (name of person)
   - WHAT (the fact or event)
   - WHEN (specific date if mentioned)
3. For dates: Convert relative dates to absolute dates when possible
   - If session date is "7 May 2023" and someone says "last week", save as "around late April 2023"
4. Save relationships between people (e.g., "Alice is Bob's manager")
5. DO NOT save trivial greetings or small talk
6. Use the memory_save tool to store each important fact

Categories to use:
- fact: Factual information about people or events
- preference: Likes, dislikes, preferences
- entity: Information defining a person/place/organization
- decision: Decisions made
- insight: Inferred patterns or insights"""

        # Get just the save tool
        save_tool = next(t for t in MEMORY_TOOLS if t["function"]["name"] == "memory_save")

        for session in conversation.sessions:
            # Format session as dialogue
            session_text = f"\n[Session Date: {session.datetime}]\n"
            for dialogue in session.dialogues:
                session_text += f"{dialogue.speaker}: {dialogue.text}\n"
                if dialogue.image_caption:
                    session_text += f"  [Shared image: {dialogue.image_caption}]\n"

            self._metrics.total_turns += len(session.dialogues)

            # Call LLM to process this session
            start = time.time()
            try:
                response = await asyncio.to_thread(
                    litellm.completion,
                    model=self._config.save_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Process this conversation session and save any important facts:\n{session_text}",
                        },
                    ],
                    tools=[save_tool],
                    tool_choice="auto",
                    temperature=0.0,
                    max_tokens=2000,
                )

                latency_ms = (time.time() - start) * 1000
                self._metrics.total_llm_calls += 1
                self._metrics.total_llm_latency_ms += latency_ms

                if hasattr(response, "usage") and response.usage:
                    self._metrics.total_input_tokens += response.usage.prompt_tokens or 0
                    self._metrics.total_output_tokens += response.usage.completion_tokens or 0

                # Process tool calls
                message = response.choices[0].message
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        if tool_call.function.name == "memory_save":
                            try:
                                args = json.loads(tool_call.function.arguments)
                                memory = await self._execute_save(args)
                                if memory:
                                    saved_memories.append(memory)
                                    self._metrics.saves_successful += 1
                                self._metrics.saves_attempted += 1
                            except Exception as e:
                                logger.warning(f"Failed to save memory: {e}")
                                self._metrics.saves_attempted += 1

                            if self._config.debug:
                                self._debug_logs.append(
                                    {
                                        "event": "memory_save",
                                        "session": session.session_num,
                                        "args": args if "args" in dir() else None,
                                        "success": "memory" in dir() and memory is not None,
                                    }
                                )

            except Exception as e:
                logger.error(f"Error processing session {session.session_num}: {e}")
                if self._config.debug:
                    self._debug_logs.append(
                        {
                            "event": "session_error",
                            "session": session.session_num,
                            "error": str(e),
                        }
                    )

        return saved_memories

    async def _execute_save(self, args: dict[str, Any]) -> Memory | None:
        """Execute a memory_save tool call.

        Args:
            args: Arguments from the tool call.

        Returns:
            The saved Memory object, or None if save failed.
        """
        if not self._backend:
            return None

        content = args.get("content", "")
        if not content:
            return None

        importance = args.get("importance", 0.5)
        entities = args.get("entities", [])
        relationships = args.get("relationships", [])

        # Convert relationships to expected format
        formatted_rels = []
        for rel in relationships:
            formatted_rels.append(
                {
                    "source": rel.get("source", ""),
                    "target": rel.get("target", ""),
                    "type": rel.get("relation", "related_to"),
                }
            )

        try:
            memory = await self._backend.save_memory(
                content=content,
                user_id=self._current_user_id,
                importance=importance,
                entities=entities,
                relationships=formatted_rels,
            )

            # Track graph metrics
            self._metrics.entities_created += len(entities)
            self._metrics.relationships_created += len(relationships)

            return memory
        except Exception as e:
            logger.warning(f"Failed to save memory: {e}")
            return None

    async def _evaluate_questions(
        self,
        conversation: LoCoMoConversation,
    ) -> list[MemoryEvalResultV2]:
        """Evaluate all questions for a conversation.

        Uses parallel workers to speed up evaluation.

        Args:
            conversation: The conversation with QA cases.

        Returns:
            List of evaluation results.
        """
        results: list[MemoryEvalResultV2] = []

        if self._config.parallel_workers > 1:
            # Parallel evaluation
            semaphore = asyncio.Semaphore(self._config.parallel_workers)

            async def eval_with_semaphore(case: LoCoMoCase) -> MemoryEvalResultV2:
                async with semaphore:
                    return await self._answer_question(case, conversation)

            tasks = [eval_with_semaphore(case) for case in conversation.qa_cases]
            results = await asyncio.gather(*tasks)
            results = list(results)
        else:
            # Sequential evaluation
            for case in conversation.qa_cases:
                result = await self._answer_question(case, conversation)
                results.append(result)

        return results

    async def _answer_question(
        self,
        case: LoCoMoCase,
        conversation: LoCoMoConversation,
    ) -> MemoryEvalResultV2:
        """
        Answer a question using memory tools.

        The LLM can call memory_search to find relevant information,
        then formulates an answer based on retrieved memories.

        Args:
            case: The LoCoMo QA case.
            conversation: The conversation context.

        Returns:
            Evaluation result with answer and metrics.
        """
        # Get search tool
        search_tool = next(t for t in MEMORY_TOOLS if t["function"]["name"] == "memory_search")

        # System prompt for answering
        system_prompt = f"""You are answering questions about a conversation between {conversation.speaker_a} and {conversation.speaker_b}.

You have access to a memory system containing facts from their conversations. Use the memory_search tool to find relevant information before answering.

CRITICAL INSTRUCTIONS:
1. ALWAYS search for relevant memories before answering
2. Base your answer ONLY on information found in memories
3. If you cannot find the answer in memories, say "Information not found"

ANSWER FORMAT - EXTREMELY IMPORTANT:
- Give the SHORTEST possible answer that directly answers the question
- DO NOT add context, explanations, or elaboration
- DO NOT repeat the question or use full sentences unless necessary
- Match these formats EXACTLY:

Question types and answer formats:
- "What is X's job?" -> "Software engineer" (NOT "X works as a software engineer")
- "When did X happen?" -> "5 July 2023" (NOT "X happened on 5 July 2023")
- "Who is X?" -> "Alice's manager" (NOT "X is Alice's manager")
- "What did X do?" -> "Went to Paris" (NOT "X went to Paris")
- "What is X's relationship status?" -> "Single" (NOT "X is single")
- "What does X like?" -> "Italian food" (NOT "X likes Italian food")

The answer should contain ONLY the specific information requested, nothing more."""

        searched_memories: list[str] = []
        search_queries: list[str] = []
        tool_calls_log: list[dict[str, Any]] = []
        total_search_latency = 0.0
        answer_start = time.time()

        # Allow multiple tool call rounds
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {case.question}"},
        ]

        max_rounds = 3
        final_answer = "Information not found"

        for round_num in range(max_rounds):
            try:
                response = await asyncio.to_thread(
                    litellm.completion,
                    model=self._config.answer_model,
                    messages=messages,
                    tools=[search_tool],
                    tool_choice="auto" if round_num == 0 else "none",  # Force search on first round
                    temperature=0.0,
                    max_tokens=500,
                )

                self._metrics.total_llm_calls += 1
                if hasattr(response, "usage") and response.usage:
                    self._metrics.total_input_tokens += response.usage.prompt_tokens or 0
                    self._metrics.total_output_tokens += response.usage.completion_tokens or 0

                message = response.choices[0].message

                # Check for tool calls
                if message.tool_calls:
                    # Add assistant message with tool calls
                    messages.append(
                        {
                            "role": "assistant",
                            "content": message.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in message.tool_calls
                            ],
                        }
                    )

                    for tool_call in message.tool_calls:
                        if tool_call.function.name == "memory_search":
                            args = json.loads(tool_call.function.arguments)
                            query = args.get("query", "")
                            search_queries.append(query)

                            # Execute search
                            search_start = time.time()
                            memories = await self._execute_search(args)
                            search_latency = (time.time() - search_start) * 1000
                            total_search_latency += search_latency
                            self._metrics.search_latency_ms.append(search_latency)

                            searched_memories.extend(memories)

                            # Format results for LLM
                            if memories:
                                result_text = "Found memories:\n" + "\n".join(
                                    f"- {m}" for m in memories
                                )
                                self._metrics.searches_with_results += 1
                            else:
                                result_text = "No relevant memories found."

                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": result_text,
                                }
                            )

                            tool_calls_log.append(
                                {
                                    "tool": "memory_search",
                                    "query": query,
                                    "num_results": len(memories),
                                    "latency_ms": search_latency,
                                }
                            )

                            self._metrics.total_searches += 1
                else:
                    # No tool calls - this is the final answer
                    final_answer = message.content or "Information not found"
                    break

            except Exception as e:
                logger.error(f"Error answering question: {e}")
                if self._config.debug:
                    self._debug_logs.append(
                        {
                            "event": "answer_error",
                            "question": case.question,
                            "error": str(e),
                        }
                    )
                break

        answer_latency = (time.time() - answer_start) * 1000

        # Calculate avg results per search
        if self._metrics.total_searches > 0:
            self._metrics.avg_results_per_search = (
                len(searched_memories) / self._metrics.total_searches
            )

        # Compute metrics
        ground_truth = str(case.answer) if case.answer is not None else ""
        f1_score = compute_f1(final_answer, ground_truth)
        exact_match = compute_exact_match(final_answer, ground_truth)
        is_correct = f1_score >= self._config.f1_threshold

        # LLM judge scoring (optional)
        llm_judge_score = None
        llm_judge_reasoning = None

        if self._config.llm_judge_enabled:
            try:
                from headroom.evals.memory.judge import create_litellm_judge

                judge_fn = create_litellm_judge(model=self._config.llm_judge_model)
                llm_judge_score, llm_judge_reasoning = judge_fn(
                    case.question, ground_truth, final_answer
                )
                is_correct = llm_judge_score >= 3.0
            except Exception as e:
                logger.warning(f"LLM judge failed: {e}")

        self._metrics.total_questions += 1
        if is_correct:
            self._metrics.correct_answers += 1

        if self._config.debug:
            self._debug_logs.append(
                {
                    "event": "answer_complete",
                    "question": case.question,
                    "ground_truth": ground_truth,
                    "predicted": final_answer,
                    "f1_score": f1_score,
                    "is_correct": is_correct,
                    "search_queries": search_queries,
                    "num_memories": len(searched_memories),
                }
            )

        return MemoryEvalResultV2(
            case=case,
            predicted_answer=final_answer,
            searched_memories=searched_memories,
            search_queries=search_queries,
            f1_score=f1_score,
            exact_match=exact_match,
            is_correct=is_correct,
            llm_judge_score=llm_judge_score,
            llm_judge_reasoning=llm_judge_reasoning,
            search_latency_ms=total_search_latency,
            answer_latency_ms=answer_latency,
            tool_calls=tool_calls_log,
        )

    async def _execute_search(self, args: dict[str, Any]) -> list[str]:
        """Execute a memory_search tool call.

        Args:
            args: Arguments from the tool call.

        Returns:
            List of memory contents matching the search.
        """
        if not self._backend:
            return []

        query = args.get("query", "")
        if not query:
            return []

        entities = args.get("entities")
        include_related = args.get("include_related", self._config.include_graph_expansion)
        top_k = args.get("top_k", self._config.max_search_results)

        if include_related:
            self._metrics.graph_expansions += 1

        try:
            results = await self._backend.search_memories(
                query=query,
                user_id=self._current_user_id,
                top_k=top_k,
                entities=entities,
                include_related=include_related,
            )

            return [r.memory.content for r in results]
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            return []

    def _aggregate_results(
        self,
        results: list[MemoryEvalResultV2],
        duration_seconds: float,
    ) -> MemoryEvalSuiteResultV2:
        """Aggregate individual results into suite result."""
        if not results:
            return MemoryEvalSuiteResultV2(
                total_cases=0,
                correct_cases=0,
                accuracy=0.0,
                avg_f1_score=0.0,
                exact_match_rate=0.0,
                avg_llm_judge_score=None,
                metrics_by_category={},
                metrics=self._metrics,
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

                # Store in metrics for summary
                self._metrics.accuracy_by_category[cat_name] = cat_correct / len(cat_results)

        # Calculate multi-hop accuracy specifically
        multi_hop_results = [r for r in results if r.case.category == 3]
        if multi_hop_results:
            multi_hop_correct = sum(1 for r in multi_hop_results if r.is_correct)
            self._metrics.multi_hop_accuracy = multi_hop_correct / len(multi_hop_results)

        # Calculate save precision/recall (approximate)
        # Precision: Were the saved memories actually useful?
        if self._metrics.saves_successful > 0:
            # Use searched memories as proxy for usefulness
            self._metrics.save_precision = min(
                1.0, self._metrics.searches_with_results / max(1, self._metrics.total_searches)
            )

        # Recall: Did we save enough? (Hard to measure without ground truth labels)
        # Approximate using accuracy as proxy
        self._metrics.save_recall = self._metrics.correct_answers / max(
            1, self._metrics.total_questions
        )

        return MemoryEvalSuiteResultV2(
            total_cases=total,
            correct_cases=correct,
            accuracy=correct / total,
            avg_f1_score=avg_f1,
            exact_match_rate=exact_match_count / total,
            avg_llm_judge_score=avg_llm_judge,
            metrics_by_category=metrics_by_category,
            metrics=self._metrics,
            results=results,
            total_duration_seconds=duration_seconds,
            config={
                "n_conversations": self._config.n_conversations,
                "categories": self._config.categories,
                "save_model": self._config.save_model,
                "answer_model": self._config.answer_model,
                "f1_threshold": self._config.f1_threshold,
                "llm_judge_enabled": self._config.llm_judge_enabled,
                "include_graph_expansion": self._config.include_graph_expansion,
            },
            debug_logs=self._debug_logs if self._config.debug else [],
        )


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_locomo_eval_v2(
    config: MemoryEvalConfigV2 | None = None,
    output_path: Path | str | None = None,
) -> MemoryEvalSuiteResultV2:
    """Convenience function to run LoCoMo V2 evaluation.

    Args:
        config: Evaluation configuration.
        output_path: Optional path to save results JSON.

    Returns:
        Evaluation results.

    Example:
        config = MemoryEvalConfigV2(
            n_conversations=3,
            answer_model="gpt-4o",
            save_model="gpt-4o-mini",
        )
        result = await run_locomo_eval_v2(config)
        print(result.summary())
    """
    config = config or MemoryEvalConfigV2()

    evaluator = LoCoMoEvaluatorV2(
        answer_model=config.answer_model,
        config=config,
    )

    result = await evaluator.run()

    if output_path:
        result.save(output_path)
        logger.info(f"Results saved to {output_path}")

    return result


def run_locomo_eval_v2_sync(
    config: MemoryEvalConfigV2 | None = None,
    output_path: Path | str | None = None,
) -> MemoryEvalSuiteResultV2:
    """Synchronous wrapper for run_locomo_eval_v2."""
    return asyncio.run(run_locomo_eval_v2(config, output_path))
