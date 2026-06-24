"""Memory evaluation framework for Headroom.

Benchmarks for evaluating memory system quality using industry-standard
datasets like LoCoMo.

Evaluates:
- Memory extraction quality
- Semantic retrieval accuracy
- End-to-end QA performance
- Hierarchical scoping
- Temporal versioning

Example:
    from headroom.evals.memory import LoCoMoEvaluator, MemoryEvalConfig

    async def my_answer_fn(question: str, memories: list[str]) -> str:
        # Your LLM-based answerer here
        return "..."

    evaluator = LoCoMoEvaluator(
        answer_fn=my_answer_fn,
        config=MemoryEvalConfig(n_conversations=5),
    )
    result = await evaluator.run()
    print(result.summary())
"""

from headroom.evals.memory.judge import (
    create_anthropic_judge,
    create_litellm_judge,
    create_openai_judge,
    simple_judge,
)
from headroom.evals.memory.locomo import (
    CATEGORY_DESCRIPTIONS,
    LOCOMO_CATEGORIES,
    LoCoMoCase,
    LoCoMoConversation,
    LoCoMoResult,
    get_locomo_stats,
    load_locomo,
)
from headroom.evals.memory.runner import (
    LoCoMoEvaluator,
    MemoryEvalConfig,
    MemoryEvalResult,
    MemoryEvalSuiteResult,
    run_locomo_eval,
    run_locomo_eval_sync,
)
from headroom.evals.memory.runner_v2 import (
    EvalMetrics,
    LoCoMoEvaluatorV2,
    MemoryEvalConfigV2,
    MemoryEvalResultV2,
    MemoryEvalSuiteResultV2,
    run_locomo_eval_v2,
    run_locomo_eval_v2_sync,
)

__all__ = [
    # Dataset loading
    "load_locomo",
    "get_locomo_stats",
    # Data models
    "LoCoMoConversation",
    "LoCoMoCase",
    "LoCoMoResult",
    # Constants
    "LOCOMO_CATEGORIES",
    "CATEGORY_DESCRIPTIONS",
    # V1 Evaluation (explicit extraction)
    "LoCoMoEvaluator",
    "MemoryEvalConfig",
    "MemoryEvalResult",
    "MemoryEvalSuiteResult",
    "run_locomo_eval",
    "run_locomo_eval_sync",
    # V2 Evaluation (LLM-controlled tools)
    "LoCoMoEvaluatorV2",
    "MemoryEvalConfigV2",
    "MemoryEvalResultV2",
    "MemoryEvalSuiteResultV2",
    "EvalMetrics",
    "run_locomo_eval_v2",
    "run_locomo_eval_v2_sync",
    # LLM Judge
    "create_openai_judge",
    "create_anthropic_judge",
    "create_litellm_judge",
    "simple_judge",
]
