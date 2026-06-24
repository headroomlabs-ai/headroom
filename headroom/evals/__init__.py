"""Headroom Evaluation Framework.

Prove that compression doesn't impact LLM accuracy through:
1. Before/After comparisons on identical queries
2. Ground truth benchmarks (HotpotQA, BFCL, SQuAD, etc.)
3. Information retrieval probes
4. Statistical significance testing
5. Batch API compression accuracy testing

Install with: pip install headroom-ai[evals]

Quick start:
    from headroom.evals import run_quick_eval
    results = run_quick_eval(n_samples=5)
    print(results.summary())

Batch compression eval:
    from headroom.evals import run_batch_compression_eval
    results = run_batch_compression_eval(provider="anthropic", n_samples=10)
    print(results.summary())

Available datasets:
    - RAG: hotpotqa, natural_questions, triviaqa, msmarco, squad
    - Long Context: longbench, narrativeqa
    - Tool Use: bfcl, toolbench, tool_outputs
    - Code: codesearchnet, humaneval
"""

from headroom.evals.batch_compression_eval import (
    BatchCompressionEvaluator,
    BatchEvalResult,
    BatchEvalSuiteResult,
    BatchRequest,
    BatchTestCase,
    TestCategory,
    TokenCountAccuracyResult,
    evaluate_token_counting_accuracy,
    get_all_test_cases,
    run_batch_compression_eval,
    run_quick_batch_eval,
)
from headroom.evals.core import (
    CompressionEvaluator,
    EvalCase,
    EvalMode,
    EvalResult,
    EvalSuite,
    EvalSuiteResult,
)
from headroom.evals.datasets import (
    DATASET_REGISTRY,
    list_available_datasets,
    load_bfcl,
    load_codesearchnet,
    load_custom_dataset,
    load_dataset_by_name,
    load_hotpotqa,
    load_humaneval,
    load_longbench,
    load_msmarco,
    load_narrativeqa,
    load_natural_questions,
    load_squad,
    load_tool_output_samples,
    load_toolbench,
    load_triviaqa,
)
from headroom.evals.metrics import (
    compute_answer_equivalence,
    compute_exact_match,
    compute_f1,
    compute_information_recall,
    compute_rouge_l,
    compute_semantic_similarity,
)
from headroom.evals.runners.before_after import (
    BeforeAfterRunner,
    LLMConfig,
    run_quick_eval,
)
from headroom.transforms.content_router import ContentRouterConfig

__all__ = [
    # Core classes
    "EvalCase",
    "EvalResult",
    "EvalSuite",
    "EvalSuiteResult",
    "EvalMode",
    "CompressionEvaluator",
    # Runner
    "BeforeAfterRunner",
    "LLMConfig",
    "ContentRouterConfig",
    "run_quick_eval",
    # Batch compression eval
    "BatchCompressionEvaluator",
    "BatchEvalResult",
    "BatchEvalSuiteResult",
    "BatchRequest",
    "BatchTestCase",
    "TestCategory",
    "TokenCountAccuracyResult",
    "evaluate_token_counting_accuracy",
    "get_all_test_cases",
    "run_batch_compression_eval",
    "run_quick_batch_eval",
    # Metrics
    "compute_f1",
    "compute_exact_match",
    "compute_semantic_similarity",
    "compute_answer_equivalence",
    "compute_rouge_l",
    "compute_information_recall",
    # Dataset loaders
    "load_hotpotqa",
    "load_natural_questions",
    "load_triviaqa",
    "load_msmarco",
    "load_squad",
    "load_longbench",
    "load_narrativeqa",
    "load_bfcl",
    "load_toolbench",
    "load_codesearchnet",
    "load_humaneval",
    "load_tool_output_samples",
    "load_custom_dataset",
    "load_dataset_by_name",
    "list_available_datasets",
    "DATASET_REGISTRY",
]
