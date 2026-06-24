"""Dataset loaders for evaluation benchmarks.

Loads real data from established sources for comprehensive compression evaluation:

RAG/Retrieval:
- HotpotQA: Multi-hop QA with Wikipedia passages
- Natural Questions: Google's real search questions
- TriviaQA: Large-scale trivia QA
- MS MARCO: Microsoft's real search queries
- SQuAD: Reading comprehension

Long Context:
- LongBench: Long context understanding benchmark
- NarrativeQA: Story comprehension

Tool Use:
- BFCL: Berkeley Function Calling Leaderboard
- ToolBench: API tool usage benchmark

Code:
- CodeSearchNet: Code search and understanding
- HumanEval: Code generation benchmark

Custom:
- Tool output samples: Built-in realistic tool outputs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from headroom.evals.core import EvalCase, EvalSuite


def _check_datasets_installed() -> None:
    """Check if HuggingFace datasets is installed."""
    try:
        import datasets  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "HuggingFace datasets required for this loader. "
            "Install with: pip install headroom-ai[evals]"
        ) from e


# =============================================================================
# RAG / RETRIEVAL DATASETS
# =============================================================================


def load_hotpotqa(
    n: int = 100,
    split: str = "validation",
) -> EvalSuite:
    """Load HotpotQA dataset for multi-hop QA evaluation.

    HotpotQA contains questions requiring reasoning over multiple
    Wikipedia passages, with verified ground truth answers.

    Dataset: https://huggingface.co/datasets/hotpotqa/hotpot_qa

    Args:
        n: Number of samples to load
        split: Dataset split ("train", "validation")

    Returns:
        EvalSuite with HotpotQA cases
    """
    _check_datasets_installed()
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=split)

    cases: list[EvalCase] = []
    for i, item in enumerate(ds):
        if i >= n:
            break

        # Build context from supporting facts
        context_parts = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            context_parts.append(f"## {title}\n" + "\n".join(sentences))

        context = "\n\n".join(context_parts)

        cases.append(
            EvalCase(
                id=f"hotpot_{i}",
                context=context,
                query=item["question"],
                ground_truth=item["answer"],
                metadata={
                    "source": "HotpotQA",
                    "type": item.get("type", "unknown"),
                    "level": item.get("level", "unknown"),
                },
            )
        )

    return EvalSuite(name="HotpotQA", cases=cases)


def load_natural_questions(
    n: int = 100,
    split: str = "validation",
) -> EvalSuite:
    """Load Google's Natural Questions dataset.

    Real questions from Google search with long-form Wikipedia answers.
    Excellent for testing compression on factual retrieval.

    Dataset: https://huggingface.co/datasets/google-research-datasets/natural_questions

    Args:
        n: Number of samples to load
        split: Dataset split ("train", "validation")

    Returns:
        EvalSuite with Natural Questions cases
    """
    _check_datasets_installed()
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/natural_questions", "default", split=split)

    cases: list[EvalCase] = []
    for i, item in enumerate(ds):
        if len(cases) >= n:
            break

        # Get document text (can be very long)
        doc_tokens = item.get("document", {}).get("tokens", {})
        if not doc_tokens:
            continue

        tokens = doc_tokens.get("token", [])
        is_html = doc_tokens.get("is_html", [])

        # Filter out HTML tokens, keep text only
        text_tokens = [t for t, h in zip(tokens, is_html) if not h]
        context = " ".join(text_tokens[:2000])  # Limit context size

        if not context.strip():
            continue

        # Get question
        question = item.get("question", {}).get("text", "")
        if not question:
            continue

        # Get short answer if available
        annotations = item.get("annotations", {})
        short_answers = annotations.get("short_answers", [[]])
        ground_truth = None
        if short_answers and short_answers[0]:
            first_answer = short_answers[0][0]
            start = first_answer.get("start_token", 0)
            end = first_answer.get("end_token", 0)
            if end > start:
                ground_truth = " ".join(tokens[start:end])

        cases.append(
            EvalCase(
                id=f"nq_{i}",
                context=context,
                query=question,
                ground_truth=ground_truth,
                metadata={
                    "source": "Natural Questions",
                    "has_short_answer": ground_truth is not None,
                },
            )
        )

    return EvalSuite(name="Natural_Questions", cases=cases)


def load_triviaqa(
    n: int = 100,
    split: str = "validation",
    subset: str = "rc",
) -> EvalSuite:
    """Load TriviaQA dataset.

    Large-scale trivia QA with evidence documents.
    Good for testing factoid question answering.

    Dataset: https://huggingface.co/datasets/trivia_qa

    Args:
        n: Number of samples to load
        split: Dataset split ("train", "validation")
        subset: Dataset subset ("rc" for reading comprehension, "unfiltered")

    Returns:
        EvalSuite with TriviaQA cases
    """
    _check_datasets_installed()
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", subset, split=split)

    cases: list[EvalCase] = []
    for i, item in enumerate(ds):
        if len(cases) >= n:
            break

        # Get search results as context
        search_results = item.get("search_results", {})
        search_contexts = search_results.get("search_context", [])

        if not search_contexts:
            # Try entity pages
            entity_pages = item.get("entity_pages", {})
            wiki_contexts = entity_pages.get("wiki_context", [])
            if wiki_contexts:
                context = "\n\n".join(wiki_contexts[:3])  # Top 3 wiki pages
            else:
                continue
        else:
            context = "\n\n".join(search_contexts[:5])  # Top 5 search results

        if not context.strip():
            continue

        question = item.get("question", "")
        if not question:
            continue

        # Ground truth answer
        answer = item.get("answer", {})
        ground_truth = answer.get("value") or answer.get("normalized_value")

        cases.append(
            EvalCase(
                id=f"triviaqa_{i}",
                context=context[:10000],  # Limit context size
                query=question,
                ground_truth=ground_truth,
                metadata={
                    "source": "TriviaQA",
                    "subset": subset,
                    "aliases": answer.get("aliases", []),
                },
            )
        )

    return EvalSuite(name=f"TriviaQA_{subset}", cases=cases)


def load_msmarco(
    n: int = 100,
    split: str = "validation",
) -> EvalSuite:
    """Load MS MARCO passage ranking dataset.

    Real Bing search queries with relevant passages.
    Excellent for testing RAG context compression.

    Dataset: https://huggingface.co/datasets/microsoft/ms_marco

    Args:
        n: Number of samples to load
        split: Dataset split ("train", "validation", "test")

    Returns:
        EvalSuite with MS MARCO cases
    """
    _check_datasets_installed()
    from datasets import load_dataset

    ds = load_dataset("microsoft/ms_marco", "v2.1", split=split)

    cases: list[EvalCase] = []
    for i, item in enumerate(ds):
        if len(cases) >= n:
            break

        # Build context from passages
        passages = item.get("passages", {})
        passage_texts = passages.get("passage_text", [])
        is_selected = passages.get("is_selected", [])

        if not passage_texts:
            continue

        # Combine passages as context
        context_parts = []
        for j, (text, selected) in enumerate(zip(passage_texts, is_selected)):
            prefix = "[RELEVANT] " if selected else ""
            context_parts.append(f"{prefix}Passage {j + 1}: {text}")

        context = "\n\n".join(context_parts)

        query = item.get("query", "")
        if not query:
            continue

        # Get answers
        answers = item.get("answers", [])
        ground_truth = answers[0] if answers else None

        cases.append(
            EvalCase(
                id=f"msmarco_{i}",
                context=context,
                query=query,
                ground_truth=ground_truth,
                metadata={
                    "source": "MS_MARCO",
                    "query_type": item.get("query_type", "unknown"),
                    "num_passages": len(passage_texts),
                },
            )
        )

    return EvalSuite(name="MS_MARCO", cases=cases)


def load_squad(
    n: int = 100,
    split: str = "validation",
) -> EvalSuite:
    """Load SQuAD v2 dataset for reading comprehension.

    SQuAD contains paragraphs with questions and extractive answers.

    Dataset: https://huggingface.co/datasets/rajpurkar/squad_v2

    Args:
        n: Number of samples to load
        split: Dataset split

    Returns:
        EvalSuite with SQuAD cases
    """
    _check_datasets_installed()
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split=split)

    cases: list[EvalCase] = []
    for i, item in enumerate(ds):
        if len(cases) >= n:
            break

        # Skip unanswerable questions
        if not item["answers"]["text"]:
            continue

        cases.append(
            EvalCase(
                id=f"squad_{i}",
                context=item["context"],
                query=item["question"],
                ground_truth=item["answers"]["text"][0],  # First answer
                metadata={
                    "source": "SQuAD_v2",
                    "title": item.get("title", ""),
                },
            )
        )

    return EvalSuite(name="SQuAD_v2", cases=cases)


# =============================================================================
# LONG CONTEXT DATASETS
# =============================================================================


def load_longbench(
    n: int = 50,
    task: str = "qasper",
) -> EvalSuite:
    """Load LongBench dataset for long context evaluation.

    LongBench tests understanding of very long documents (4K-128K tokens).
    Critical for testing compression on long contexts.

    Dataset: https://huggingface.co/datasets/THUDM/LongBench

    Available tasks:
    - qasper: Scientific paper QA
    - multifieldqa_en: Multi-field QA
    - narrativeqa: Story comprehension
    - gov_report: Government report summarization
    - qmsum: Meeting summarization
    - multi_news: Multi-document summarization

    Args:
        n: Number of samples to load
        task: LongBench task name

    Returns:
        EvalSuite with LongBench cases
    """
    _check_datasets_installed()
    from datasets import load_dataset

    try:
        ds = load_dataset("THUDM/LongBench", task, split="test")
    except Exception as e:
        raise ValueError(f"Failed to load LongBench task '{task}': {e}") from e

    cases: list[EvalCase] = []
    for i, item in enumerate(ds):
        if i >= n:
            break

        context = item.get("context", "")
        if not context:
            continue

        query = item.get("input", "")
        if not query:
            continue

        # Ground truth (list of answers for some tasks)
        answers = item.get("answers", [])
        ground_truth = answers[0] if answers else None

        cases.append(
            EvalCase(
                id=f"longbench_{task}_{i}",
                context=context,
                query=query,
                ground_truth=ground_truth,
                metadata={
                    "source": "LongBench",
                    "task": task,
                    "context_length": len(context),
                },
            )
        )

    return EvalSuite(name=f"LongBench_{task}", cases=cases)


def load_narrativeqa(
    n: int = 100,
    split: str = "test",
) -> EvalSuite:
    """Load NarrativeQA dataset for story comprehension.

    Questions about books and movie scripts requiring understanding
    of narrative structure and long-range dependencies.

    Dataset: https://huggingface.co/datasets/deepmind/narrativeqa

    Args:
        n: Number of samples to load
        split: Dataset split

    Returns:
        EvalSuite with NarrativeQA cases
    """
    _check_datasets_installed()
    from datasets import load_dataset

    ds = load_dataset("deepmind/narrativeqa", split=split)

    cases: list[EvalCase] = []
    for i, item in enumerate(ds):
        if len(cases) >= n:
            break

        # Get summary as context (full text is very long)
        document = item.get("document", {})
        summary = document.get("summary", {})
        context = summary.get("text", "")

        if not context:
            continue

        question = item.get("question", {}).get("text", "")
        if not question:
            continue

        # Multiple reference answers
        answers = item.get("answers", [])
        answer_texts = [a.get("text", "") for a in answers if a.get("text")]
        ground_truth = answer_texts[0] if answer_texts else None

        cases.append(
            EvalCase(
                id=f"narrativeqa_{i}",
                context=context,
                query=question,
                ground_truth=ground_truth,
                metadata={
                    "source": "NarrativeQA",
                    "document_kind": document.get("kind", "unknown"),
                    "all_answers": answer_texts,
                },
            )
        )

    return EvalSuite(name="NarrativeQA", cases=cases)


# =============================================================================
# TOOL USE / FUNCTION CALLING DATASETS
# =============================================================================


def load_bfcl(
    n: int = 100,
    category: str = "simple",
) -> EvalSuite:
    """Load Berkeley Function Calling Leaderboard dataset.

    BFCL contains real API schemas with ground truth function calls,
    ideal for testing tool output compression.

    Dataset: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard

    Available categories:
    - simple: Single function calls
    - multiple: Multiple function selection
    - parallel: Parallel function calls
    - exec_simple: Executable simple functions
    - exec_multiple: Executable multiple functions
    - exec_parallel: Executable parallel functions

    Args:
        n: Number of samples to load
        category: BFCL category

    Returns:
        EvalSuite with BFCL cases
    """
    import urllib.request

    base_url = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
    data_file = f"BFCL_v3_{category}.json"
    gt_file = f"possible_answer/BFCL_v3_{category}.json"

    # Download questions + function schemas (JSONL)
    try:
        raw = urllib.request.urlopen(f"{base_url}/{data_file}").read().decode("utf-8")  # nosec B310
        items = [json.loads(line) for line in raw.strip().split("\n") if line.strip()]
    except Exception as e:
        raise ValueError(f"Failed to download BFCL dataset '{data_file}': {e}") from e

    # Download ground truth (JSONL, keyed by id)
    gt_by_id: dict[str, str] = {}
    try:
        gt_raw = urllib.request.urlopen(f"{base_url}/{gt_file}").read().decode("utf-8")  # nosec B310
        for line in gt_raw.strip().split("\n"):
            if line.strip():
                obj = json.loads(line)
                gt_by_id[obj["id"]] = json.dumps(obj.get("ground_truth", []))
    except Exception:
        pass  # Ground truth is optional

    cases: list[EvalCase] = []
    for i, item in enumerate(items):
        if i >= n:
            break

        item_id = item.get("id", f"bfcl_{category}_{i}")

        # Extract question from nested structure: [[{"role":"user","content":"..."}]]
        question = ""
        if item.get("question"):
            try:
                question = item["question"][0][0].get("content", "")
            except (IndexError, KeyError, TypeError):
                question = str(item.get("question", ""))

        # Functions as context (this is what we'd compress)
        functions = item.get("function", [])
        context = json.dumps(functions, indent=2) if functions else ""

        if not context or len(context) < 10:
            continue

        # Ground truth
        gt_str = gt_by_id.get(item_id)

        cases.append(
            EvalCase(
                id=item_id,
                context=context,
                query=question,
                ground_truth=gt_str,
                metadata={
                    "source": "BFCL",
                    "category": category,
                    "num_functions": len(functions) if isinstance(functions, list) else 0,
                },
            )
        )

    return EvalSuite(name=f"BFCL_{category}", cases=cases)


def load_toolbench(
    n: int = 100,
    category: str = "G1",
) -> EvalSuite:
    """Load ToolBench dataset for API tool usage.

    Real-world API scenarios with multiple tools and complex reasoning.

    Dataset: https://huggingface.co/datasets/ToolBench/ToolBench

    Categories:
    - G1: Single-tool single-step
    - G2: Single-tool multi-step
    - G3: Multi-tool single-step

    Args:
        n: Number of samples to load
        category: ToolBench category (G1, G2, G3)

    Returns:
        EvalSuite with ToolBench cases
    """
    _check_datasets_installed()
    from datasets import load_dataset

    try:
        ds = load_dataset("ToolBench/ToolBench", category, split="test")
    except Exception as e:
        raise ValueError(f"Failed to load ToolBench category '{category}': {e}") from e

    cases: list[EvalCase] = []
    for i, item in enumerate(ds):
        if len(cases) >= n:
            break

        # Get tool definitions as context
        tools = item.get("api_list", [])
        if not tools:
            continue

        # Format tools as JSON context
        tool_defs = []
        for tool in tools:
            tool_defs.append(
                {
                    "name": tool.get("api_name", ""),
                    "description": tool.get("api_description", ""),
                    "parameters": tool.get("required_parameters", [])
                    + tool.get("optional_parameters", []),
                }
            )

        context = json.dumps(tool_defs, indent=2)

        query = item.get("query", "")
        if not query:
            continue

        # Expected answer/trajectory
        answer = item.get("answer", "")

        cases.append(
            EvalCase(
                id=f"toolbench_{category}_{i}",
                context=context,
                query=query,
                ground_truth=answer if answer else None,
                metadata={
                    "source": "ToolBench",
                    "category": category,
                    "num_tools": len(tools),
                },
            )
        )

    return EvalSuite(name=f"ToolBench_{category}", cases=cases)


# =============================================================================
# CODE DATASETS
# =============================================================================


def load_codesearchnet(
    n: int = 100,
    language: str = "python",
    split: str = "test",
) -> EvalSuite:
    """Load CodeSearchNet dataset for code understanding.

    Code snippets with natural language descriptions.
    Tests compression on code without losing semantic meaning.

    Dataset: https://huggingface.co/datasets/code_search_net

    Languages: python, java, javascript, go, ruby, php

    Args:
        n: Number of samples to load
        language: Programming language
        split: Dataset split

    Returns:
        EvalSuite with CodeSearchNet cases
    """
    _check_datasets_installed()
    from datasets import load_dataset

    try:
        ds = load_dataset("code_search_net", language, split=split)
    except Exception as e:
        raise ValueError(f"Failed to load CodeSearchNet for '{language}': {e}") from e

    cases: list[EvalCase] = []
    for i, item in enumerate(ds):
        if len(cases) >= n:
            break

        code = item.get("func_code_string", "") or item.get("whole_func_string", "")
        if not code:
            continue

        # Use docstring as query (find code from description)
        docstring = item.get("func_documentation_string", "")
        if not docstring:
            continue

        cases.append(
            EvalCase(
                id=f"codesearchnet_{language}_{i}",
                context=code,
                query="What does this code do? Describe its functionality.",
                ground_truth=docstring,
                metadata={
                    "source": "CodeSearchNet",
                    "language": language,
                    "func_name": item.get("func_name", ""),
                    "repo": item.get("repository_name", ""),
                },
            )
        )

    return EvalSuite(name=f"CodeSearchNet_{language}", cases=cases)


def load_humaneval(
    n: int = 164,  # Total size is 164
) -> EvalSuite:
    """Load HumanEval dataset for code generation.

    Hand-crafted programming problems with test cases.
    Tests if compression preserves enough info for code generation.

    Dataset: https://huggingface.co/datasets/openai_humaneval

    Args:
        n: Number of samples to load (max 164)

    Returns:
        EvalSuite with HumanEval cases
    """
    _check_datasets_installed()
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")

    cases: list[EvalCase] = []
    for i, item in enumerate(ds):
        if i >= n:
            break

        # Prompt contains function signature and docstring
        prompt = item.get("prompt", "")
        if not prompt:
            continue

        # Canonical solution
        canonical = item.get("canonical_solution", "")

        # Test cases for verification
        test = item.get("test", "")

        # Use the prompt as context, ask to complete
        cases.append(
            EvalCase(
                id=f"humaneval_{item.get('task_id', i)}",
                context=prompt,
                query="Complete this function implementation.",
                ground_truth=canonical,
                metadata={
                    "source": "HumanEval",
                    "task_id": item.get("task_id", ""),
                    "entry_point": item.get("entry_point", ""),
                    "test": test,
                },
            )
        )

    return EvalSuite(name="HumanEval", cases=cases)


# =============================================================================
# BUILT-IN TOOL OUTPUT SAMPLES
# =============================================================================


def load_tool_output_samples() -> EvalSuite:
    """Load built-in tool output samples for testing.

    These are realistic tool outputs that headroom is designed to compress:
    - API responses (JSON)
    - Log outputs
    - Code files
    - Database query results
    - Kubernetes events
    - Error tracebacks
    """
    cases = [
        # GitHub API response
        EvalCase(
            id="github_search_001",
            context=json.dumps(
                {
                    "total_count": 3,
                    "items": [
                        {
                            "id": 12345,
                            "name": "headroom",
                            "full_name": "anthropic/headroom",
                            "description": "Context optimization for LLM applications",
                            "stargazers_count": 1250,
                            "language": "Python",
                            "topics": ["llm", "compression"],
                        },
                        {
                            "id": 23456,
                            "name": "llm-cache",
                            "full_name": "openai/llm-cache",
                            "description": "High-performance caching for LLMs",
                            "stargazers_count": 890,
                            "language": "Python",
                            "topics": ["caching", "llm"],
                        },
                        {
                            "id": 34567,
                            "name": "prompt-optimizer",
                            "full_name": "google/prompt-optimizer",
                            "description": "Automatic prompt optimization with RL",
                            "stargazers_count": 2100,
                            "language": "Python",
                            "topics": ["prompt-engineering", "rlhf"],
                        },
                    ],
                },
                indent=2,
            ),
            query="Which repository has the most stars?",
            ground_truth="prompt-optimizer",
            metadata={"source": "tool_output", "tool": "github_search"},
        ),
        # Kubernetes events
        EvalCase(
            id="k8s_events_001",
            context="""NAMESPACE     LAST SEEN   TYPE      REASON              OBJECT                                MESSAGE
default       2m          Warning   FailedScheduling    pod/nginx-deployment-5d8b9c7f4-x2k9j   0/3 nodes are available: 3 Insufficient memory
default       5m          Normal    Scheduled           pod/redis-master-0                     Successfully assigned default/redis-master-0 to node-2
kube-system   1h          Warning   NodeNotReady        node/node-3                            Node node-3 status is now: NodeNotReady
default       30s         Normal    Pulled              pod/api-server-7f8d9c8b5-m4n2p         Container image "api-server:v2.1.0" already present""",
            query="What is the error with the nginx deployment?",
            ground_truth="Insufficient memory",
            metadata={"source": "tool_output", "tool": "kubectl_events"},
        ),
        # Python traceback
        EvalCase(
            id="traceback_001",
            context="""Traceback (most recent call last):
  File "/app/services/payment.py", line 127, in process_payment
    result = stripe.PaymentIntent.create(
  File "/usr/local/lib/python3.11/site-packages/stripe/api_resources/payment_intent.py", line 87, in create
    return cls._static_request("post", url, params=params)
stripe.error.CardError: Your card was declined. This transaction requires authentication.
Request ID: req_a1b2c3d4e5f6g7h8
Error Code: card_declined
Decline Code: authentication_required""",
            query="What is the error code?",
            ground_truth="card_declined",
            metadata={"source": "tool_output", "tool": "error_logs"},
        ),
        # Database query result
        EvalCase(
            id="db_query_001",
            context=json.dumps(
                [
                    {
                        "user_id": 1,
                        "name": "Alice",
                        "email": "alice@example.com",
                        "role": "admin",
                        "created_at": "2024-01-15",
                    },
                    {
                        "user_id": 2,
                        "name": "Bob",
                        "email": "bob@example.com",
                        "role": "user",
                        "created_at": "2024-02-20",
                    },
                    {
                        "user_id": 3,
                        "name": "Charlie",
                        "email": "charlie@example.com",
                        "role": "user",
                        "created_at": "2024-03-10",
                    },
                    {
                        "user_id": 4,
                        "name": "Diana",
                        "email": "diana@example.com",
                        "role": "moderator",
                        "created_at": "2024-04-05",
                    },
                    {
                        "user_id": 5,
                        "name": "Eve",
                        "email": "eve@example.com",
                        "role": "user",
                        "created_at": "2024-05-01",
                    },
                ],
                indent=2,
            ),
            query="Who is the admin user?",
            ground_truth="Alice",
            metadata={"source": "tool_output", "tool": "database_query"},
        ),
        # Metrics data
        EvalCase(
            id="metrics_001",
            context=json.dumps(
                {
                    "service": "api-gateway",
                    "period": "last_hour",
                    "metrics": {
                        "requests_total": 125432,
                        "requests_success": 124890,
                        "requests_failed": 542,
                        "latency_p50_ms": 45,
                        "latency_p99_ms": 230,
                        "error_rate_percent": 0.43,
                        "cpu_usage_percent": 67.5,
                        "memory_usage_mb": 2048,
                    },
                },
                indent=2,
            ),
            query="What is the p99 latency?",
            ground_truth="230",
            metadata={"source": "tool_output", "tool": "metrics_api"},
        ),
        # Git log output
        EvalCase(
            id="git_log_001",
            context="""commit a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0
Author: Alice Developer <alice@example.com>
Date:   Mon Jan 15 10:30:00 2024 -0800

    Fix critical security vulnerability in authentication

    - Patched SQL injection in login endpoint
    - Added input sanitization
    - Updated tests

commit b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1
Author: Bob Engineer <bob@example.com>
Date:   Sun Jan 14 15:45:00 2024 -0800

    Add user profile feature

    - New profile page component
    - Avatar upload functionality
    - Bio field with markdown support

commit c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2
Author: Alice Developer <alice@example.com>
Date:   Sat Jan 13 09:00:00 2024 -0800

    Refactor database connection pooling""",
            query="Who fixed the security vulnerability?",
            ground_truth="Alice Developer",
            metadata={"source": "tool_output", "tool": "git_log"},
        ),
        # AWS CLI output
        EvalCase(
            id="aws_ec2_001",
            context=json.dumps(
                {
                    "Reservations": [
                        {
                            "Instances": [
                                {
                                    "InstanceId": "i-0abc123def456789a",
                                    "InstanceType": "t3.large",
                                    "State": {"Name": "running"},
                                    "PrivateIpAddress": "10.0.1.100",
                                    "Tags": [{"Key": "Name", "Value": "web-server-1"}],
                                },
                                {
                                    "InstanceId": "i-0def456ghi789012b",
                                    "InstanceType": "t3.xlarge",
                                    "State": {"Name": "stopped"},
                                    "PrivateIpAddress": "10.0.1.101",
                                    "Tags": [{"Key": "Name", "Value": "web-server-2"}],
                                },
                                {
                                    "InstanceId": "i-0ghi789jkl012345c",
                                    "InstanceType": "r5.2xlarge",
                                    "State": {"Name": "running"},
                                    "PrivateIpAddress": "10.0.2.50",
                                    "Tags": [{"Key": "Name", "Value": "database-primary"}],
                                },
                            ]
                        }
                    ]
                },
                indent=2,
            ),
            query="Which instance is stopped?",
            ground_truth="web-server-2",
            metadata={"source": "tool_output", "tool": "aws_ec2_describe"},
        ),
        # Large JSON API response with nested data
        EvalCase(
            id="complex_api_001",
            context=json.dumps(
                {
                    "status": "success",
                    "data": {
                        "organization": {
                            "id": "org_123",
                            "name": "Acme Corp",
                            "plan": "enterprise",
                        },
                        "projects": [
                            {
                                "id": "proj_001",
                                "name": "Backend API",
                                "status": "active",
                                "team_size": 5,
                                "budget_remaining": 15000,
                            },
                            {
                                "id": "proj_002",
                                "name": "Mobile App",
                                "status": "active",
                                "team_size": 8,
                                "budget_remaining": 28500,
                            },
                            {
                                "id": "proj_003",
                                "name": "Data Pipeline",
                                "status": "paused",
                                "team_size": 3,
                                "budget_remaining": 5000,
                            },
                        ],
                        "total_budget": 100000,
                        "spent": 51500,
                    },
                },
                indent=2,
            ),
            query="Which project has the highest budget remaining?",
            ground_truth="Mobile App",
            metadata={"source": "tool_output", "tool": "project_api"},
        ),
    ]

    return EvalSuite(name="ToolOutputSamples", cases=cases)


# =============================================================================
# CUSTOM DATASET LOADERS
# =============================================================================


def load_custom_dataset(path: Path | str) -> EvalSuite:
    """Load a custom evaluation dataset from JSONL file.

    Expected format (one JSON object per line):
    {"id": "case_001", "context": "...", "query": "...", "ground_truth": "..."}

    Args:
        path: Path to JSONL file

    Returns:
        EvalSuite with loaded cases
    """
    return EvalSuite.from_jsonl(path)


def generate_retrieval_probes(
    context: str,
    n_probes: int = 5,
) -> list[str]:
    """Generate retrieval probes from a context.

    Extracts key facts/entities that should be retrievable
    after compression.

    Args:
        context: The context to analyze
        n_probes: Number of probes to generate

    Returns:
        List of fact strings to probe for
    """
    import re

    probes = []

    # Look for specific patterns
    patterns = [
        r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # Names (e.g., "John Smith")
        r"\b\d{4}-\d{2}-\d{2}\b",  # Dates (e.g., "2024-01-15")
        r"\b[A-Z]{2,}\b",  # Acronyms (e.g., "API", "HTTP")
        r"\b\d+\.\d+%?\b",  # Numbers (e.g., "99.9%", "123.45")
        r'"[^"]{5,50}"',  # Quoted strings
        r"\b[a-z_]+_[a-z_]+\b",  # Snake case identifiers
    ]

    for pattern in patterns:
        matches = re.findall(pattern, context)
        for match in matches[:2]:  # Take up to 2 per pattern
            if match not in probes:
                probes.append(match.strip('"'))
            if len(probes) >= n_probes:
                return probes

    return probes


# =============================================================================
# DATASET REGISTRY & UTILITIES
# =============================================================================


DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    # RAG/Retrieval
    "hotpotqa": {
        "loader": load_hotpotqa,
        "description": "Multi-hop QA requiring reasoning over multiple Wikipedia passages",
        "category": "rag",
        "default_n": 100,
    },
    "natural_questions": {
        "loader": load_natural_questions,
        "description": "Real Google search questions with Wikipedia answers",
        "category": "rag",
        "default_n": 100,
    },
    "triviaqa": {
        "loader": load_triviaqa,
        "description": "Large-scale trivia QA with evidence documents",
        "category": "rag",
        "default_n": 100,
    },
    "msmarco": {
        "loader": load_msmarco,
        "description": "Real Bing search queries with relevant passages",
        "category": "rag",
        "default_n": 100,
    },
    "squad": {
        "loader": load_squad,
        "description": "Reading comprehension with extractive answers",
        "category": "rag",
        "default_n": 100,
    },
    # Long Context
    "longbench": {
        "loader": load_longbench,
        "description": "Long context understanding (4K-128K tokens)",
        "category": "long_context",
        "default_n": 50,
    },
    "narrativeqa": {
        "loader": load_narrativeqa,
        "description": "Story comprehension requiring narrative understanding",
        "category": "long_context",
        "default_n": 100,
    },
    # Tool Use
    "bfcl": {
        "loader": load_bfcl,
        "description": "Berkeley Function Calling Leaderboard - API schemas",
        "category": "tool_use",
        "default_n": 100,
    },
    "toolbench": {
        "loader": load_toolbench,
        "description": "Real-world API tool usage scenarios",
        "category": "tool_use",
        "default_n": 100,
    },
    # Code
    "codesearchnet": {
        "loader": load_codesearchnet,
        "description": "Code snippets with natural language descriptions",
        "category": "code",
        "default_n": 100,
    },
    "humaneval": {
        "loader": load_humaneval,
        "description": "Hand-crafted programming problems",
        "category": "code",
        "default_n": 164,
    },
    # Built-in
    "tool_outputs": {
        "loader": load_tool_output_samples,
        "description": "Built-in realistic tool outputs (JSON, logs, etc.)",
        "category": "tool_use",
        "default_n": None,  # Fixed size
    },
}


def list_available_datasets() -> dict[str, list[str]]:
    """List all available datasets by category.

    Returns:
        Dictionary mapping category to list of dataset names
    """
    by_category: dict[str, list[str]] = {}
    for name, info in DATASET_REGISTRY.items():
        category = info["category"]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(name)
    return by_category


def load_dataset_by_name(
    name: str,
    n: int | None = None,
    **kwargs: Any,
) -> EvalSuite:
    """Load a dataset by name from the registry.

    Args:
        name: Dataset name from registry
        n: Number of samples (uses default if not specified)
        **kwargs: Additional arguments for the loader

    Returns:
        EvalSuite with loaded cases
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    info = DATASET_REGISTRY[name]
    loader = info["loader"]

    # Use provided n or default
    if n is None:
        n = info.get("default_n")

    if n is not None:
        result: EvalSuite = loader(n=n, **kwargs)
    else:
        result = loader(**kwargs)
    return result
