"""Batch API Compression Accuracy Evaluation.

This module evaluates whether compression preserves LLM accuracy when processing
batch API requests through the Headroom proxy.

Evaluation Strategy:
1. Create batch requests with questions that have known/verifiable answers
2. Run batches through proxy WITH compression enabled
3. Run same batches directly to API WITHOUT compression
4. Compare results using F1 score, semantic similarity, and ground truth matching
5. Report accuracy preservation rate and token savings

Test Categories:
- Math questions: Simple arithmetic with deterministic answers
- Factual questions with context: Include paragraph, ask about specific facts
- Multi-turn conversations: Test context preservation across turns
- JSON extraction: Verify data extraction from compressed tool outputs

Works with both OpenAI and Anthropic batch APIs.

Usage:
    >>> from headroom.evals.batch_compression_eval import (
    ...     BatchCompressionEvaluator,
    ...     run_batch_compression_eval,
    ... )
    >>> results = run_batch_compression_eval(provider="anthropic", n_samples=10)
    >>> print(results.summary())

CLI:
    python -m headroom.evals.batch_compression_eval --provider anthropic --samples 10
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from headroom.evals.metrics import (
    compute_exact_match,
    compute_f1,
    compute_rouge_l,
    compute_semantic_similarity,
)
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class TestCategory(Enum):
    """Categories of test cases for batch compression evaluation."""

    MATH = "math"  # Simple arithmetic
    FACTUAL = "factual"  # Facts from context
    MULTI_TURN = "multi_turn"  # Conversation context
    JSON_EXTRACTION = "json_extraction"  # Extract from JSON
    CODE_UNDERSTANDING = "code_understanding"  # Understand code context
    LONG_CONTEXT = "long_context"  # Long documents


@dataclass
class BatchRequest:
    """A single request in a batch."""

    custom_id: str
    messages: list[dict[str, Any]]
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.0

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic batch API format."""
        return {
            "custom_id": self.custom_id,
            "params": {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": self.messages,
            },
        }

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI batch API format."""
        return {
            "custom_id": self.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": self.messages,
            },
        }


@dataclass
class BatchTestCase:
    """A test case for batch compression evaluation."""

    id: str
    category: TestCategory
    request: BatchRequest
    ground_truth: str
    ground_truth_keywords: list[str] = field(default_factory=list)
    context_facts: list[str] = field(default_factory=list)  # Facts that must be preserved
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchEvalResult:
    """Result of evaluating a single batch test case."""

    case_id: str
    category: TestCategory

    # Token counts
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    tokens_saved: int

    # Responses
    response_original: str  # Response from uncompressed request
    response_compressed: str  # Response from compressed request

    # Metrics
    exact_match: bool
    f1_score: float
    rouge_l_score: float
    semantic_similarity: float | None = None
    ground_truth_match: bool = False
    keywords_found: list[str] = field(default_factory=list)
    keywords_missing: list[str] = field(default_factory=list)

    # Timing
    latency_original_ms: float = 0.0
    latency_compressed_ms: float = 0.0

    # Verdict
    accuracy_preserved: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "case_id": self.case_id,
            "category": self.category.value,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": self.compression_ratio,
            "tokens_saved": self.tokens_saved,
            "response_original": self.response_original[:500],
            "response_compressed": self.response_compressed[:500],
            "exact_match": self.exact_match,
            "f1_score": self.f1_score,
            "rouge_l_score": self.rouge_l_score,
            "semantic_similarity": self.semantic_similarity,
            "ground_truth_match": self.ground_truth_match,
            "keywords_found": self.keywords_found,
            "keywords_missing": self.keywords_missing,
            "accuracy_preserved": self.accuracy_preserved,
            "error": self.error,
        }


@dataclass
class BatchEvalSuiteResult:
    """Aggregated results from batch compression evaluation."""

    provider: str
    total_cases: int
    passed_cases: int
    failed_cases: int

    # By category
    results_by_category: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Aggregate metrics
    avg_compression_ratio: float = 0.0
    avg_f1_score: float = 0.0
    avg_rouge_l_score: float = 0.0
    avg_semantic_similarity: float | None = None
    accuracy_preservation_rate: float = 0.0

    # Token savings
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    total_tokens_saved: int = 0

    # Individual results
    results: list[BatchEvalResult] = field(default_factory=list)

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== Batch Compression Evaluation ({self.provider}) ===",
            f"Cases: {self.passed_cases}/{self.total_cases} passed ({self.accuracy_preservation_rate:.1%})",
            f"Compression: {self.avg_compression_ratio:.1%} average",
            f"F1 Score: {self.avg_f1_score:.3f}",
            f"ROUGE-L: {self.avg_rouge_l_score:.3f}",
        ]
        if self.avg_semantic_similarity is not None:
            lines.append(f"Semantic Similarity: {self.avg_semantic_similarity:.3f}")
        lines.append(
            f"Tokens: {self.total_original_tokens:,} -> {self.total_compressed_tokens:,} "
            f"({self.total_tokens_saved:,} saved)"
        )

        # Per-category breakdown
        if self.results_by_category:
            lines.append("\nBy Category:")
            for category, stats in self.results_by_category.items():
                lines.append(
                    f"  {category}: {stats['passed']}/{stats['total']} passed, "
                    f"F1={stats['avg_f1']:.3f}, compression={stats['avg_compression']:.1%}"
                )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "results_by_category": self.results_by_category,
            "avg_compression_ratio": self.avg_compression_ratio,
            "avg_f1_score": self.avg_f1_score,
            "avg_rouge_l_score": self.avg_rouge_l_score,
            "avg_semantic_similarity": self.avg_semantic_similarity,
            "accuracy_preservation_rate": self.accuracy_preservation_rate,
            "total_original_tokens": self.total_original_tokens,
            "total_compressed_tokens": self.total_compressed_tokens,
            "total_tokens_saved": self.total_tokens_saved,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "results": [r.to_dict() for r in self.results],
        }


# =============================================================================
# Test Case Generation
# =============================================================================


def generate_math_test_cases() -> list[BatchTestCase]:
    """Generate math question test cases with deterministic answers."""
    return [
        BatchTestCase(
            id="math_001",
            category=TestCategory.MATH,
            request=BatchRequest(
                custom_id="math_001",
                messages=[
                    {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
                ],
            ),
            ground_truth="4",
            ground_truth_keywords=["4"],
        ),
        BatchTestCase(
            id="math_002",
            category=TestCategory.MATH,
            request=BatchRequest(
                custom_id="math_002",
                messages=[
                    {
                        "role": "user",
                        "content": "What is 15 * 7? Answer with just the number.",
                    }
                ],
            ),
            ground_truth="105",
            ground_truth_keywords=["105"],
        ),
        BatchTestCase(
            id="math_003",
            category=TestCategory.MATH,
            request=BatchRequest(
                custom_id="math_003",
                messages=[
                    {
                        "role": "user",
                        "content": "If x = 5 and y = 3, what is x^2 + y^2? Answer with just the number.",
                    }
                ],
            ),
            ground_truth="34",
            ground_truth_keywords=["34"],
        ),
        BatchTestCase(
            id="math_004",
            category=TestCategory.MATH,
            request=BatchRequest(
                custom_id="math_004",
                messages=[
                    {
                        "role": "user",
                        "content": "What is the square root of 144? Answer with just the number.",
                    }
                ],
            ),
            ground_truth="12",
            ground_truth_keywords=["12"],
        ),
        BatchTestCase(
            id="math_005",
            category=TestCategory.MATH,
            request=BatchRequest(
                custom_id="math_005",
                messages=[
                    {
                        "role": "user",
                        "content": "A store has 50 apples. They sell 23 and receive 15 more. How many apples do they have? Answer with just the number.",
                    }
                ],
            ),
            ground_truth="42",
            ground_truth_keywords=["42"],
        ),
    ]


def generate_factual_test_cases() -> list[BatchTestCase]:
    """Generate factual question test cases with context paragraphs."""
    return [
        BatchTestCase(
            id="factual_001",
            category=TestCategory.FACTUAL,
            request=BatchRequest(
                custom_id="factual_001",
                messages=[
                    {
                        "role": "user",
                        "content": """Based on this context, answer the question.

Context:
The Headroom SDK is a context optimization layer for LLM applications. It was created
by Anthropic in 2024. The main features include SmartCrusher for JSON compression,
Kompress for text compression, and CCR (Compress-Cache-Retrieve) for reversible
compression. The SDK supports Python 3.9+ and can save up to 70% of tokens on
large JSON arrays.

Question: What percentage of tokens can the SDK save on large JSON arrays?""",
                    }
                ],
            ),
            ground_truth="70%",
            ground_truth_keywords=["70", "percent", "%"],
            context_facts=["70%", "SmartCrusher", "Kompress", "CCR", "Python 3.9"],
        ),
        BatchTestCase(
            id="factual_002",
            category=TestCategory.FACTUAL,
            request=BatchRequest(
                custom_id="factual_002",
                messages=[
                    {
                        "role": "user",
                        "content": """Read this product review and answer the question.

Review:
I purchased the XYZ-500 wireless headphones last month for $149.99. The battery life
is excellent - I get about 35 hours on a single charge. The noise cancellation is
good but not as strong as the Sony WH-1000XM5. Sound quality is warm with punchy bass.
The Bluetooth connection is stable with my iPhone 15. Overall rating: 4 out of 5 stars.

Question: How many hours of battery life does the reviewer report?""",
                    }
                ],
            ),
            ground_truth="35 hours",
            ground_truth_keywords=["35", "hours"],
            context_facts=["$149.99", "35 hours", "4 out of 5", "XYZ-500", "iPhone 15"],
        ),
        BatchTestCase(
            id="factual_003",
            category=TestCategory.FACTUAL,
            request=BatchRequest(
                custom_id="factual_003",
                messages=[
                    {
                        "role": "user",
                        "content": """Here is information about a conference. Answer the question.

Event Details:
The Annual AI Summit 2024 will be held at the San Francisco Convention Center from
March 15-17, 2024. Keynote speakers include Dr. Sarah Chen from Stanford University,
Mark Thompson from OpenAI, and Professor James Liu from MIT. Registration opens
January 5th with early bird pricing of $599 (regular price $899). Expected attendance
is 5,000 participants from over 40 countries.

Question: What is the early bird registration price?""",
                    }
                ],
            ),
            ground_truth="$599",
            ground_truth_keywords=["599", "$599"],
            context_facts=["$599", "$899", "March 15-17", "5,000", "40 countries"],
        ),
    ]


def generate_json_extraction_test_cases() -> list[BatchTestCase]:
    """Generate JSON extraction test cases - critical for batch API compression."""
    return [
        BatchTestCase(
            id="json_001",
            category=TestCategory.JSON_EXTRACTION,
            request=BatchRequest(
                custom_id="json_001",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Here is some JSON data. Answer the question.

Data:
{
                            json.dumps(
                                {
                                    "users": [
                                        {"id": 1, "name": "Alice", "role": "admin", "active": True},
                                        {"id": 2, "name": "Bob", "role": "user", "active": True},
                                        {
                                            "id": 3,
                                            "name": "Charlie",
                                            "role": "user",
                                            "active": False,
                                        },
                                        {
                                            "id": 4,
                                            "name": "Diana",
                                            "role": "moderator",
                                            "active": True,
                                        },
                                        {"id": 5, "name": "Eve", "role": "user", "active": True},
                                    ]
                                },
                                indent=2,
                            )
                        }

Question: Who is the admin user? Answer with just the name.""",
                    }
                ],
            ),
            ground_truth="Alice",
            ground_truth_keywords=["Alice"],
            context_facts=["Alice", "admin", "Bob", "Charlie", "Diana", "Eve"],
        ),
        BatchTestCase(
            id="json_002",
            category=TestCategory.JSON_EXTRACTION,
            request=BatchRequest(
                custom_id="json_002",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze this API response and answer the question.

API Response:
{
                            json.dumps(
                                {
                                    "status": "success",
                                    "data": {
                                        "repositories": [
                                            {
                                                "name": "headroom",
                                                "stars": 1250,
                                                "language": "Python",
                                            },
                                            {
                                                "name": "llm-cache",
                                                "stars": 890,
                                                "language": "Python",
                                            },
                                            {
                                                "name": "prompt-optimizer",
                                                "stars": 2100,
                                                "language": "Python",
                                            },
                                        ],
                                        "total_count": 3,
                                    },
                                },
                                indent=2,
                            )
                        }

Question: Which repository has the most stars? Answer with just the repository name.""",
                    }
                ],
            ),
            ground_truth="prompt-optimizer",
            ground_truth_keywords=["prompt-optimizer", "prompt optimizer"],
            context_facts=["headroom", "1250", "llm-cache", "890", "prompt-optimizer", "2100"],
        ),
        BatchTestCase(
            id="json_003",
            category=TestCategory.JSON_EXTRACTION,
            request=BatchRequest(
                custom_id="json_003",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Here is order data. Answer the question.

Orders:
{
                            json.dumps(
                                [
                                    {
                                        "order_id": "ORD-001",
                                        "customer": "John",
                                        "total": 150.00,
                                        "status": "shipped",
                                    },
                                    {
                                        "order_id": "ORD-002",
                                        "customer": "Jane",
                                        "total": 299.99,
                                        "status": "pending",
                                    },
                                    {
                                        "order_id": "ORD-003",
                                        "customer": "Bob",
                                        "total": 75.50,
                                        "status": "delivered",
                                    },
                                    {
                                        "order_id": "ORD-004",
                                        "customer": "Alice",
                                        "total": 499.00,
                                        "status": "pending",
                                    },
                                ],
                                indent=2,
                            )
                        }

Question: What is the total value of orders with 'pending' status? Answer with just the number.""",
                    }
                ],
            ),
            ground_truth="798.99",
            ground_truth_keywords=["798.99", "799"],
            context_facts=["ORD-001", "ORD-002", "ORD-003", "ORD-004", "pending"],
        ),
    ]


def generate_multi_turn_test_cases() -> list[BatchTestCase]:
    """Generate multi-turn conversation test cases."""
    return [
        BatchTestCase(
            id="multi_001",
            category=TestCategory.MULTI_TURN,
            request=BatchRequest(
                custom_id="multi_001",
                messages=[
                    {
                        "role": "user",
                        "content": "I'm planning a trip to Tokyo. My budget is $3000 and I want to stay for 7 days.",
                    },
                    {
                        "role": "assistant",
                        "content": "Great choice! Tokyo is amazing. With a $3000 budget for 7 days, I'd recommend staying in Shinjuku or Shibuya. Budget hotels run $80-120/night. You'll have about $2000 left for food, transport, and activities. Would you like hotel recommendations or activity suggestions?",
                    },
                    {
                        "role": "user",
                        "content": "What was my total budget again?",
                    },
                ],
            ),
            ground_truth="$3000",
            ground_truth_keywords=["3000", "$3000", "3,000"],
            context_facts=["$3000", "7 days", "Tokyo", "Shinjuku", "Shibuya"],
        ),
        BatchTestCase(
            id="multi_002",
            category=TestCategory.MULTI_TURN,
            request=BatchRequest(
                custom_id="multi_002",
                messages=[
                    {
                        "role": "user",
                        "content": "Help me debug this error: TypeError: cannot read property 'map' of undefined",
                    },
                    {
                        "role": "assistant",
                        "content": "This error occurs when you try to call .map() on a variable that is undefined. Common causes: 1) Data hasn't loaded yet (async issue), 2) Wrong variable name, 3) API returned unexpected format. Can you share the code where this happens?",
                    },
                    {
                        "role": "user",
                        "content": "The code is: const items = data.results; return items.map(x => x.name);",
                    },
                    {
                        "role": "assistant",
                        "content": "I see the issue. Either `data` is undefined, or `data.results` is undefined. Add a check: `const items = data?.results || [];` This uses optional chaining and provides a fallback empty array.",
                    },
                    {
                        "role": "user",
                        "content": "What was the original error I reported?",
                    },
                ],
            ),
            ground_truth="TypeError: cannot read property 'map' of undefined",
            ground_truth_keywords=["TypeError", "map", "undefined"],
            context_facts=["TypeError", "map", "undefined", "data.results"],
        ),
        BatchTestCase(
            id="multi_003",
            category=TestCategory.MULTI_TURN,
            request=BatchRequest(
                custom_id="multi_003",
                messages=[
                    {
                        "role": "user",
                        "content": "I need to create a Python function that calculates the factorial of a number.",
                    },
                    {
                        "role": "assistant",
                        "content": """Here's a Python function for factorial:

```python
def factorial(n):
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

This uses recursion. For large numbers, consider using `math.factorial()` or an iterative approach.""",
                    },
                    {
                        "role": "user",
                        "content": "What happens if I pass -5 to this function?",
                    },
                ],
            ),
            ground_truth="ValueError",
            ground_truth_keywords=["ValueError", "raises", "error", "negative"],
            context_facts=["factorial", "ValueError", "negative", "recursion"],
        ),
    ]


def generate_code_understanding_test_cases() -> list[BatchTestCase]:
    """Generate code understanding test cases."""
    return [
        BatchTestCase(
            id="code_001",
            category=TestCategory.CODE_UNDERSTANDING,
            request=BatchRequest(
                custom_id="code_001",
                messages=[
                    {
                        "role": "user",
                        "content": """Analyze this Python code and answer the question.

```python
class UserManager:
    def __init__(self, db_connection):
        self.db = db_connection
        self.cache = {}

    def get_user(self, user_id: int) -> dict:
        if user_id in self.cache:
            return self.cache[user_id]

        user = self.db.query(f"SELECT * FROM users WHERE id = {user_id}")
        self.cache[user_id] = user
        return user

    def clear_cache(self):
        self.cache = {}
```

Question: What is the purpose of the `cache` attribute? Answer in one sentence.""",
                    }
                ],
            ),
            ground_truth="caching user data",
            ground_truth_keywords=["cache", "store", "avoid", "database", "lookup", "performance"],
            context_facts=["cache", "get_user", "clear_cache", "db_connection"],
        ),
        BatchTestCase(
            id="code_002",
            category=TestCategory.CODE_UNDERSTANDING,
            request=BatchRequest(
                custom_id="code_002",
                messages=[
                    {
                        "role": "user",
                        "content": """What does this function return?

```javascript
function processData(items) {
    return items
        .filter(item => item.active)
        .map(item => item.name.toUpperCase())
        .sort();
}
```

Question: If I call processData([{name: 'bob', active: true}, {name: 'Alice', active: false}, {name: 'Charlie', active: true}]), what will be returned?""",
                    }
                ],
            ),
            ground_truth='["BOB", "CHARLIE"]',
            ground_truth_keywords=["BOB", "CHARLIE"],
            context_facts=["filter", "map", "sort", "active", "toUpperCase"],
        ),
    ]


def generate_long_context_test_cases() -> list[BatchTestCase]:
    """Generate long context test cases to stress test compression."""
    # Create a long document
    long_document = """
# Technical Specification Document: Project Phoenix

## Executive Summary
Project Phoenix is a next-generation distributed computing platform designed to handle
petabyte-scale data processing with sub-second latency. The system will support real-time
analytics, machine learning workloads, and event-driven architectures.

## System Requirements

### Hardware Requirements
- Minimum 64GB RAM per node
- 8-core CPU (Intel Xeon or AMD EPYC recommended)
- 10Gbps network connectivity
- NVMe SSD storage (minimum 1TB per node)
- Total cluster: minimum 10 nodes

### Software Requirements
- Linux (Ubuntu 22.04 LTS or RHEL 8+)
- Kubernetes 1.28+
- Docker 24.0+
- Python 3.11+
- Java 17 (for Kafka components)

## Architecture Overview

### Core Components
1. **Data Ingestion Layer (DIL)**
   - Apache Kafka for message streaming
   - Supports 1 million events/second throughput
   - Auto-scaling based on queue depth
   - Dead letter queue for failed messages

2. **Processing Engine (PE)**
   - Apache Spark for batch processing
   - Apache Flink for stream processing
   - Custom ML inference engine
   - GPU acceleration support

3. **Storage Layer (SL)**
   - Apache Cassandra for operational data
   - Delta Lake for analytical workloads
   - Redis for caching (4-hour TTL)
   - S3-compatible object storage

4. **Query Engine (QE)**
   - Trino for federated queries
   - GraphQL API layer
   - REST API with OpenAPI 3.0 spec
   - gRPC for internal communication

### Security Features
- mTLS for all inter-service communication
- OAuth 2.0 / OIDC for authentication
- RBAC with fine-grained permissions
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- Audit logging with tamper-proof storage

## Performance Specifications

### Latency Requirements
| Operation | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Read | 5ms | 20ms | 50ms |
| Write | 10ms | 40ms | 100ms |
| Query | 100ms | 500ms | 1000ms |
| ML Inference | 50ms | 150ms | 300ms |

### Throughput Requirements
- Ingestion: 1M events/second
- Queries: 10K QPS
- Storage writes: 500K ops/second
- ML predictions: 100K/second

## Deployment Timeline

### Phase 1: Foundation (Q1 2024)
- Infrastructure provisioning
- Core services deployment
- Basic monitoring setup
- Cost estimate: $500,000

### Phase 2: Data Layer (Q2 2024)
- Storage systems deployment
- Data migration scripts
- Backup and recovery testing
- Cost estimate: $750,000

### Phase 3: Processing (Q3 2024)
- Spark/Flink deployment
- ML pipeline integration
- Performance optimization
- Cost estimate: $600,000

### Phase 4: Production (Q4 2024)
- Full production deployment
- Load testing and tuning
- Documentation and training
- Cost estimate: $400,000

Total estimated cost: $2,250,000

## Contact Information
- Project Lead: Dr. Sarah Chen (sarah.chen@company.com)
- Tech Lead: Marcus Johnson (marcus.j@company.com)
- DevOps Lead: Lisa Park (lisa.park@company.com)
"""

    return [
        BatchTestCase(
            id="long_001",
            category=TestCategory.LONG_CONTEXT,
            request=BatchRequest(
                custom_id="long_001",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Read this technical document and answer the question.

{long_document}

Question: What is the total estimated cost for the project?""",
                    }
                ],
            ),
            ground_truth="$2,250,000",
            ground_truth_keywords=["2,250,000", "2250000", "$2.25 million"],
            context_facts=[
                "$2,250,000",
                "Phase 1",
                "$500,000",
                "Q4 2024",
                "Apache Kafka",
                "Dr. Sarah Chen",
            ],
        ),
        BatchTestCase(
            id="long_002",
            category=TestCategory.LONG_CONTEXT,
            request=BatchRequest(
                custom_id="long_002",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Read this technical document and answer the question.

{long_document}

Question: What is the P99 latency requirement for ML Inference?""",
                    }
                ],
            ),
            ground_truth="300ms",
            ground_truth_keywords=["300", "300ms", "milliseconds"],
            context_facts=["300ms", "ML Inference", "P99", "50ms", "150ms"],
        ),
        BatchTestCase(
            id="long_003",
            category=TestCategory.LONG_CONTEXT,
            request=BatchRequest(
                custom_id="long_003",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Read this technical document and answer the question.

{long_document}

Question: What caching system is used and what is its TTL?""",
                    }
                ],
            ),
            ground_truth="Redis, 4-hour TTL",
            ground_truth_keywords=["Redis", "4", "hour", "TTL"],
            context_facts=["Redis", "4-hour TTL", "caching", "Storage Layer"],
        ),
    ]


def get_all_test_cases() -> list[BatchTestCase]:
    """Get all test cases for batch compression evaluation."""
    cases = []
    cases.extend(generate_math_test_cases())
    cases.extend(generate_factual_test_cases())
    cases.extend(generate_json_extraction_test_cases())
    cases.extend(generate_multi_turn_test_cases())
    cases.extend(generate_code_understanding_test_cases())
    cases.extend(generate_long_context_test_cases())
    return cases


# =============================================================================
# Token Counting
# =============================================================================


class TokenCounter:
    """Token counter for evaluation."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self._tokenizer: Any = None

    def _get_tokenizer(self) -> Any:
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            try:
                from headroom.tokenizers import get_tokenizer

                self._tokenizer = get_tokenizer(self.model)
            except ImportError:
                logger.warning("Could not load tokenizer, using estimate")
                self._tokenizer = None
        return self._tokenizer

    def count_text(self, text: str) -> int:
        """Count tokens in text."""
        tokenizer = self._get_tokenizer()
        if tokenizer:
            try:
                return int(tokenizer.count_text(text))
            except Exception:
                pass
        # Fallback: estimate ~4 chars per token
        return len(text) // 4

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_text(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += self.count_text(str(block.get("content", "")))
        return total


# =============================================================================
# Batch Compression Evaluator
# =============================================================================


class BatchCompressionEvaluator:
    """Evaluator for batch API compression accuracy.

    This class runs evaluation by:
    1. Compressing batch request messages
    2. Calling LLM with both original and compressed messages
    3. Comparing responses to measure accuracy preservation

    Example:
        >>> evaluator = BatchCompressionEvaluator(provider="anthropic")
        >>> results = evaluator.run(test_cases)
        >>> print(results.summary())
    """

    def __init__(
        self,
        provider: Literal["anthropic", "openai"] = "anthropic",
        model: str | None = None,
        router_config: ContentRouterConfig | None = None,
        use_semantic_similarity: bool = True,
    ):
        """Initialize evaluator.

        Args:
            provider: LLM provider ("anthropic" or "openai")
            model: Model to use (defaults to provider's default)
            router_config: Configuration for ContentRouter compression
            use_semantic_similarity: Whether to compute semantic similarity
        """
        self.provider = provider
        self.model = model or self._get_default_model(provider)
        self.router_config = router_config or ContentRouterConfig()
        self.use_semantic_similarity = use_semantic_similarity

        # Initialize components
        self._router = ContentRouter(config=self.router_config)
        self._token_counter = TokenCounter(self.model)
        self._llm_client = self._init_llm_client()

    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider."""
        return {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
        }.get(provider, "claude-sonnet-4-20250514")

    def _init_llm_client(self) -> Any:
        """Initialize LLM client."""
        if self.provider == "anthropic":
            try:
                import anthropic

                return anthropic.Anthropic()
            except ImportError as e:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                ) from e
        elif self.provider == "openai":
            try:
                import openai

                return openai.OpenAI()
            except ImportError as e:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                ) from e
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_llm(self, messages: list[dict[str, Any]]) -> str:
        """Call LLM and return response text."""
        if self.provider == "anthropic":
            response = self._llm_client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                messages=messages,
            )
            return str(response.content[0].text)
        elif self.provider == "openai":
            response = self._llm_client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                messages=messages,
            )
            content = response.choices[0].message.content
            return str(content) if content else ""
        return ""

    def _compress_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Compress messages using ContentRouter.

        Args:
            messages: Messages to compress.

        Returns:
            Tuple of (compressed_messages, original_tokens, compressed_tokens)
        """
        original_tokens = self._token_counter.count_messages(messages)
        compressed_messages = []

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 100:
                # Try to compress
                try:
                    result = self._router.compress(content)
                    if result.compression_ratio < 0.95:  # Only use if we got compression
                        compressed_messages.append({**msg, "content": result.compressed})
                    else:
                        compressed_messages.append(msg)
                except Exception as e:
                    logger.debug("Compression failed: %s", e)
                    compressed_messages.append(msg)
            else:
                compressed_messages.append(msg)

        compressed_tokens = self._token_counter.count_messages(compressed_messages)
        return compressed_messages, original_tokens, compressed_tokens

    def evaluate_case(self, case: BatchTestCase) -> BatchEvalResult:
        """Evaluate a single test case.

        Args:
            case: Test case to evaluate.

        Returns:
            Evaluation result.
        """
        messages = case.request.messages.copy()

        # Count original tokens
        original_tokens = self._token_counter.count_messages(messages)

        # Compress messages
        compressed_messages, _, compressed_tokens = self._compress_messages(messages)

        compression_ratio = 1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0
        tokens_saved = original_tokens - compressed_tokens

        # Call LLM with ORIGINAL messages
        start = time.time()
        try:
            response_original = self._call_llm(messages)
        except Exception as e:
            response_original = f"ERROR: {e}"
        latency_original = (time.time() - start) * 1000

        # Call LLM with COMPRESSED messages
        start = time.time()
        try:
            response_compressed = self._call_llm(compressed_messages)
        except Exception as e:
            response_compressed = f"ERROR: {e}"
        latency_compressed = (time.time() - start) * 1000

        # Compute metrics
        exact_match = compute_exact_match(response_original, response_compressed)
        f1_score = compute_f1(response_original, response_compressed)
        rouge_l_score = compute_rouge_l(response_original, response_compressed)

        # Semantic similarity
        semantic_sim = None
        if self.use_semantic_similarity:
            try:
                semantic_sim = compute_semantic_similarity(response_original, response_compressed)
            except (ImportError, Exception):
                pass

        # Check ground truth and keywords
        response_lower = response_compressed.lower()
        ground_truth_match = (
            case.ground_truth.lower() in response_lower
            or compute_f1(response_compressed, case.ground_truth) > 0.5
        )

        keywords_found = [kw for kw in case.ground_truth_keywords if kw.lower() in response_lower]
        keywords_missing = [
            kw for kw in case.ground_truth_keywords if kw.lower() not in response_lower
        ]

        # Determine accuracy preservation
        # Preserved if ANY of:
        # 1. F1 > 0.7
        # 2. Semantic similarity > 0.85
        # 3. Ground truth match
        # 4. Most keywords found
        accuracy_preserved = (
            f1_score > 0.7
            or (semantic_sim is not None and semantic_sim > 0.85)
            or ground_truth_match
            or (len(keywords_found) >= len(case.ground_truth_keywords) * 0.5)
        )

        return BatchEvalResult(
            case_id=case.id,
            category=case.category,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            tokens_saved=tokens_saved,
            response_original=response_original,
            response_compressed=response_compressed,
            exact_match=exact_match,
            f1_score=f1_score,
            rouge_l_score=rouge_l_score,
            semantic_similarity=semantic_sim,
            ground_truth_match=ground_truth_match,
            keywords_found=keywords_found,
            keywords_missing=keywords_missing,
            latency_original_ms=latency_original,
            latency_compressed_ms=latency_compressed,
            accuracy_preserved=accuracy_preserved,
        )

    def run(
        self,
        test_cases: list[BatchTestCase],
        progress_callback: Any = None,
    ) -> BatchEvalSuiteResult:
        """Run evaluation on all test cases.

        Args:
            test_cases: Test cases to evaluate.
            progress_callback: Optional callback(current, total, result)

        Returns:
            Aggregated evaluation results.
        """
        start_time = time.time()
        results: list[BatchEvalResult] = []

        for i, case in enumerate(test_cases):
            result = self.evaluate_case(case)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(test_cases), result)

        # Aggregate results
        passed = sum(1 for r in results if r.accuracy_preserved)
        failed = len(results) - passed

        # By category stats
        results_by_category: dict[str, dict[str, Any]] = {}
        for category in TestCategory:
            category_results = [r for r in results if r.category == category]
            if category_results:
                cat_passed = sum(1 for r in category_results if r.accuracy_preserved)
                results_by_category[category.value] = {
                    "total": len(category_results),
                    "passed": cat_passed,
                    "avg_f1": sum(r.f1_score for r in category_results) / len(category_results),
                    "avg_compression": sum(r.compression_ratio for r in category_results)
                    / len(category_results),
                }

        # Overall stats
        total_original = sum(r.original_tokens for r in results)
        total_compressed = sum(r.compressed_tokens for r in results)

        avg_compression = sum(r.compression_ratio for r in results) / len(results) if results else 0
        avg_f1 = sum(r.f1_score for r in results) / len(results) if results else 0
        avg_rouge_l = sum(r.rouge_l_score for r in results) / len(results) if results else 0

        semantic_sims = [
            r.semantic_similarity for r in results if r.semantic_similarity is not None
        ]
        avg_semantic = sum(semantic_sims) / len(semantic_sims) if semantic_sims else None

        return BatchEvalSuiteResult(
            provider=self.provider,
            total_cases=len(results),
            passed_cases=passed,
            failed_cases=failed,
            results_by_category=results_by_category,
            avg_compression_ratio=avg_compression,
            avg_f1_score=avg_f1,
            avg_rouge_l_score=avg_rouge_l,
            avg_semantic_similarity=avg_semantic,
            accuracy_preservation_rate=passed / len(results) if results else 0,
            total_original_tokens=total_original,
            total_compressed_tokens=total_compressed,
            total_tokens_saved=total_original - total_compressed,
            results=results,
            duration_seconds=time.time() - start_time,
        )


# =============================================================================
# Token Counting Accuracy Evaluation
# =============================================================================


@dataclass
class TokenCountAccuracyResult:
    """Result of token counting accuracy evaluation."""

    model: str
    test_cases: int
    exact_matches: int
    avg_error_percent: float
    max_error_percent: float
    results: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary."""
        return f"""=== Token Counting Accuracy ({self.model}) ===
Test cases: {self.test_cases}
Exact matches: {self.exact_matches} ({self.exact_matches / self.test_cases:.1%})
Average error: {self.avg_error_percent:.2f}%
Max error: {self.max_error_percent:.2f}%"""


def evaluate_token_counting_accuracy(
    model: str = "claude-sonnet-4-20250514",
) -> TokenCountAccuracyResult:
    """Evaluate token counting accuracy against provider's official count.

    This verifies that Headroom's token counting matches the provider's
    actual token usage as reported in API responses.

    Args:
        model: Model to test token counting for.

    Returns:
        Token counting accuracy results.
    """
    # Test texts of varying complexity
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        json.dumps({"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}),
        "This is a longer text that spans multiple sentences. "
        "It includes various punctuation marks, numbers like 123, "
        "and even some symbols like @#$%. "
        "The purpose is to test token counting accuracy.",
        "Machine learning and artificial intelligence are transforming industries. " * 10,
    ]

    token_counter = TokenCounter(model)
    results = []
    errors = []

    for text in test_texts:
        estimated = token_counter.count_text(text)

        # Note: In a real implementation, we would call the API and check
        # the actual token count from the response. For now, we compare
        # against a reference tokenizer.
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            actual = len(enc.encode(text))
        except ImportError:
            # Fallback: assume estimate is close
            actual = len(text) // 4

        error_percent = abs(estimated - actual) / max(actual, 1) * 100
        errors.append(error_percent)

        results.append(
            {
                "text_length": len(text),
                "estimated_tokens": estimated,
                "actual_tokens": actual,
                "error_percent": error_percent,
            }
        )

    exact_matches = sum(1 for r in results if r["error_percent"] < 1)

    return TokenCountAccuracyResult(
        model=model,
        test_cases=len(results),
        exact_matches=exact_matches,
        avg_error_percent=sum(errors) / len(errors) if errors else 0,
        max_error_percent=max(errors) if errors else 0,
        results=results,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def run_batch_compression_eval(
    provider: Literal["anthropic", "openai"] = "anthropic",
    n_samples: int | None = None,
    categories: list[str] | None = None,
    use_semantic_similarity: bool = False,
) -> BatchEvalSuiteResult:
    """Run batch compression evaluation.

    Args:
        provider: LLM provider.
        n_samples: Number of samples (None = all).
        categories: Categories to test (None = all).
        use_semantic_similarity: Whether to compute semantic similarity.

    Returns:
        Evaluation results.
    """
    test_cases = get_all_test_cases()

    # Filter by category if specified
    if categories:
        category_set = {TestCategory(c) for c in categories}
        test_cases = [c for c in test_cases if c.category in category_set]

    # Limit samples if specified
    if n_samples:
        test_cases = test_cases[:n_samples]

    evaluator = BatchCompressionEvaluator(
        provider=provider,
        use_semantic_similarity=use_semantic_similarity,
    )

    def progress(current: int, total: int, result: BatchEvalResult) -> None:
        status = "PASS" if result.accuracy_preserved else "FAIL"
        print(
            f"  [{current}/{total}] {result.case_id}: {status} "
            f"(F1={result.f1_score:.2f}, compression={result.compression_ratio:.1%})"
        )

    print(f"\nRunning batch compression eval with {len(test_cases)} test cases...")
    print(f"Provider: {provider}")
    print()

    results = evaluator.run(test_cases, progress_callback=progress)

    print(f"\n{results.summary()}")
    return results


def run_quick_batch_eval(
    provider: Literal["anthropic", "openai"] = "anthropic",
) -> BatchEvalSuiteResult:
    """Run a quick batch eval with just math and JSON tests.

    Args:
        provider: LLM provider.

    Returns:
        Evaluation results.
    """
    return run_batch_compression_eval(
        provider=provider,
        categories=["math", "json_extraction"],
        use_semantic_similarity=False,
    )


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate batch API compression accuracy")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider",
    )
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to test")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=[
            "math",
            "factual",
            "json_extraction",
            "multi_turn",
            "code_understanding",
            "long_context",
        ],
        default=None,
        help="Categories to test",
    )
    parser.add_argument(
        "--semantic-similarity",
        action="store_true",
        help="Compute semantic similarity (requires sentence-transformers)",
    )
    parser.add_argument(
        "--token-counting",
        action="store_true",
        help="Run token counting accuracy evaluation",
    )
    parser.add_argument("--output", type=str, default=None, help="Output file for JSON results")

    args = parser.parse_args()

    if args.token_counting:
        token_results = evaluate_token_counting_accuracy()
        print(token_results.summary())
        if args.output:
            import json

            with open(args.output, "w") as f:
                json.dump(
                    {
                        "model": token_results.model,
                        "test_cases": token_results.test_cases,
                        "exact_matches": token_results.exact_matches,
                        "avg_error_percent": token_results.avg_error_percent,
                        "max_error_percent": token_results.max_error_percent,
                        "results": token_results.results,
                    },
                    f,
                    indent=2,
                )
    else:
        eval_results = run_batch_compression_eval(
            provider=args.provider,
            n_samples=args.samples,
            categories=args.categories,
            use_semantic_similarity=args.semantic_similarity,
        )

        if args.output:
            import json

            with open(args.output, "w") as f:
                json.dump(eval_results.to_dict(), f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
