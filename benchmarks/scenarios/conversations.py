"""Conversation generators for benchmark scenarios.

This module provides generators for realistic conversation patterns that
exercise Headroom transforms:

- Agentic conversations: Multi-turn with tool calls (SmartCrusher, RollingWindow)
- RAG conversations: Large context injection (CacheAligner, RollingWindow)

These generators produce conversations that mirror real-world usage patterns
from production agentic systems.
"""

from __future__ import annotations

import json
import random
import uuid
from typing import Any

from .tool_outputs import (
    generate_api_responses,
    generate_database_rows,
    generate_log_entries,
    generate_search_results,
)


def generate_agentic_conversation(
    turns: int,
    tool_calls_per_turn: int = 1,
    items_per_tool_response: int = 50,
) -> list[dict[str, Any]]:
    """Generate a multi-turn agentic conversation with tool calls.

    Simulates a realistic coding assistant or data analysis agent with:
    - System prompt with instructions
    - Multiple user/assistant turns
    - Tool calls with realistic responses
    - Variety of tool types (search, database, API)

    Args:
        turns: Number of user turns to generate.
        tool_calls_per_turn: Average tool calls per assistant response.
        items_per_tool_response: Items in each tool response.

    Returns:
        List of message dictionaries (OpenAI format).

    Example:
        messages = generate_agentic_conversation(50, tool_calls_per_turn=2)
        # System + 50 turns with tool calls = ~250+ messages
    """
    messages = []

    # System prompt
    messages.append(
        {
            "role": "system",
            "content": _generate_system_prompt(),
        }
    )

    # Generate turns
    for turn_idx in range(turns):
        # User message
        user_query = _generate_user_query(turn_idx)
        messages.append(
            {
                "role": "user",
                "content": user_query,
            }
        )

        # Assistant with tool calls
        num_calls = max(1, tool_calls_per_turn + random.randint(-1, 1))
        tool_calls = []

        for call_idx in range(num_calls):
            tool_name, arguments = _generate_tool_call(turn_idx, call_idx)
            call_id = f"call_{uuid.uuid4().hex[:16]}"

            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments),
                    },
                }
            )

        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            }
        )

        # Tool responses
        for tool_call in tool_calls:
            tool_response = _generate_tool_response(
                tool_call["function"]["name"],
                items_per_tool_response,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_response),
                }
            )

        # Assistant summary (most turns, not all)
        if random.random() < 0.8:
            messages.append(
                {
                    "role": "assistant",
                    "content": _generate_assistant_summary(turn_idx, tool_calls),
                }
            )

    return messages


def generate_rag_conversation(
    context_tokens: int,
    num_queries: int = 3,
) -> list[dict[str, Any]]:
    """Generate a RAG conversation with injected context.

    Simulates retrieval-augmented generation patterns with:
    - Large context documents injected into system or user messages
    - Multiple queries against the context
    - Dynamic date information for cache alignment testing

    Args:
        context_tokens: Approximate target tokens for context.
        num_queries: Number of user queries about the context.

    Returns:
        List of message dictionaries (OpenAI format).

    Example:
        messages = generate_rag_conversation(10000, num_queries=5)
        # ~10K tokens of context + 5 Q&A turns
    """
    messages = []

    # System prompt with date (for CacheAligner testing)
    messages.append(
        {
            "role": "system",
            "content": _generate_rag_system_prompt(),
        }
    )

    # Generate context documents
    context_content = _generate_rag_context(context_tokens)

    # Inject context as first user message
    messages.append(
        {
            "role": "user",
            "content": f"Here are the relevant documents for context:\n\n{context_content}\n\nPlease analyze these documents.",
        }
    )

    # Assistant acknowledgment
    messages.append(
        {
            "role": "assistant",
            "content": "I've reviewed the provided documents. I can see information about technical documentation, API specifications, and configuration guides. What would you like to know?",
        }
    )

    # Generate Q&A turns
    for i in range(num_queries):
        question = _generate_rag_question(i)
        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        answer = _generate_rag_answer(i)
        messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

    return messages


def generate_anthropic_agentic_conversation(
    turns: int,
    tool_calls_per_turn: int = 1,
    items_per_tool_response: int = 50,
) -> list[dict[str, Any]]:
    """Generate a multi-turn agentic conversation in Anthropic format.

    Same as generate_agentic_conversation but with Anthropic's content
    block structure for tool_use and tool_result.

    Args:
        turns: Number of user turns to generate.
        tool_calls_per_turn: Average tool calls per assistant response.
        items_per_tool_response: Items in each tool response.

    Returns:
        List of message dictionaries (Anthropic format).
    """
    messages = []

    # System message (Anthropic uses separate system parameter, but we include it)
    messages.append(
        {
            "role": "system",
            "content": _generate_system_prompt(),
        }
    )

    for turn_idx in range(turns):
        # User message
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": _generate_user_query(turn_idx)}],
            }
        )

        # Assistant with tool_use blocks
        num_calls = max(1, tool_calls_per_turn + random.randint(-1, 1))
        content_blocks = []

        for call_idx in range(num_calls):
            tool_name, arguments = _generate_tool_call(turn_idx, call_idx)
            tool_use_id = f"toolu_{uuid.uuid4().hex[:16]}"

            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": tool_name,
                    "input": arguments,
                }
            )

        messages.append(
            {
                "role": "assistant",
                "content": content_blocks,
            }
        )

        # Tool results in user message
        tool_results = []
        for block in content_blocks:
            tool_response = _generate_tool_response(
                block["name"],
                items_per_tool_response,
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": json.dumps(tool_response),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": tool_results,
            }
        )

        # Assistant response
        if random.random() < 0.8:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": _generate_assistant_summary(turn_idx, [])}
                    ],
                }
            )

    return messages


# Helper functions


def _generate_system_prompt() -> str:
    """Generate a realistic system prompt."""
    return """You are an AI assistant with access to various tools for searching, querying, and analyzing data.

Your capabilities include:
- Searching documents and code repositories
- Querying databases for information
- Analyzing logs and metrics
- Making API calls to external services

Guidelines:
1. Always use the most appropriate tool for the task
2. Analyze results thoroughly before responding
3. Be concise but comprehensive in your answers
4. If a query returns many results, summarize the key findings

Current date: 2025-01-06
System version: 2.1.0"""


def _generate_rag_system_prompt() -> str:
    """Generate a RAG-style system prompt with dynamic date."""
    return """You are a helpful assistant that answers questions based on provided context documents.

Rules:
- Only answer based on the provided context
- If the context doesn't contain relevant information, say so
- Cite specific sections when possible
- Be precise and factual

Current date: 2025-01-06
Today is Monday, January 6th, 2025."""


def _generate_user_query(turn_idx: int) -> str:
    """Generate a realistic user query."""
    queries = [
        "Can you search for documentation about authentication?",
        "What are the recent error logs from the API service?",
        "Find all users who signed up in the last week",
        "Query the metrics database for CPU usage patterns",
        "Search for any issues related to timeout errors",
        "Look up the configuration for the payment service",
        "Find all transactions that failed today",
        "What does the documentation say about rate limiting?",
        "Check the logs for any critical errors",
        "Search for code examples of database connections",
        "Find the user with ID 12345",
        "What are the top 10 most frequent errors?",
        "Look up records for UUID 550e8400-e29b-41d4-a716-446655440000",
        "Search for all mentions of memory leaks",
        "Find the deployment history for production",
    ]
    return queries[turn_idx % len(queries)]


def _generate_tool_call(turn_idx: int, call_idx: int) -> tuple[str, dict[str, Any]]:
    """Generate a tool call name and arguments."""
    tools = [
        ("search_documents", {"query": f"search query {turn_idx}", "limit": 50}),
        ("query_database", {"table": "users", "filters": {"status": "active"}, "limit": 100}),
        ("get_logs", {"service": "api", "level": "ERROR", "hours": 24}),
        ("search_code", {"pattern": "def handle_", "language": "python"}),
        ("get_metrics", {"metric": "cpu_usage", "period": "1h"}),
        ("list_api_responses", {"endpoint": "/api/v1/users", "limit": 50}),
        ("get_user", {"user_id": random.randint(1000, 9999)}),
        ("search_errors", {"query": "timeout", "severity": "high"}),
    ]
    return random.choice(tools)


def _generate_tool_response(tool_name: str, n: int) -> list[dict[str, Any]]:
    """Generate appropriate tool response based on tool type."""
    if "search" in tool_name or "document" in tool_name:
        return generate_search_results(n)
    elif "log" in tool_name or "error" in tool_name:
        return generate_log_entries(n)
    elif "database" in tool_name or "query" in tool_name:
        return generate_database_rows(n)
    else:
        return generate_api_responses(n)


def _generate_assistant_summary(turn_idx: int, tool_calls: list) -> str:
    """Generate an assistant summary response."""
    summaries = [
        "Based on the search results, I found several relevant documents. The most relevant ones discuss the authentication flow and API endpoints.",
        "I've analyzed the logs and found some patterns. There were a few errors in the past hour, mostly related to connection timeouts.",
        "The database query returned the requested records. I can see several active users matching your criteria.",
        "Looking at the metrics, I notice some fluctuation in CPU usage. The average is around 45% with occasional spikes.",
        "The search results show multiple code examples. The most relevant implementation uses async patterns for better performance.",
    ]
    return summaries[turn_idx % len(summaries)]


def _generate_rag_context(target_tokens: int) -> str:
    """Generate context documents for RAG scenarios."""
    # Approximate 4 characters per token
    target_chars = target_tokens * 4

    documents = []
    current_chars = 0

    doc_templates = [
        _generate_api_doc,
        _generate_config_doc,
        _generate_tutorial_doc,
        _generate_faq_doc,
    ]

    while current_chars < target_chars:
        generator = random.choice(doc_templates)
        doc = generator()
        documents.append(doc)
        current_chars += len(doc)

    return "\n\n---\n\n".join(documents)


def _generate_api_doc() -> str:
    """Generate a fake API documentation section."""
    endpoints = ["users", "orders", "products", "auth", "payments"]
    endpoint = random.choice(endpoints)
    return f"""## API Reference: /{endpoint}

### GET /api/v1/{endpoint}
Returns a list of {endpoint}.

**Parameters:**
- `limit` (int): Maximum number of results (default: 20)
- `offset` (int): Pagination offset (default: 0)
- `filter` (string): Filter expression

**Response:**
```json
{{
  "data": [...],
  "meta": {{
    "total": 1000,
    "limit": 20,
    "offset": 0
  }}
}}
```

**Rate Limits:**
- 100 requests per minute for standard tier
- 1000 requests per minute for premium tier

**Error Codes:**
- 400: Invalid request parameters
- 401: Authentication required
- 429: Rate limit exceeded
- 500: Internal server error"""


def _generate_config_doc() -> str:
    """Generate a fake configuration documentation."""
    services = ["database", "cache", "queue", "api", "worker"]
    service = random.choice(services)
    return f"""## Configuration: {service.title()} Service

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| {service.upper()}_HOST | Host address | localhost |
| {service.upper()}_PORT | Port number | {random.randint(3000, 9000)} |
| {service.upper()}_TIMEOUT | Timeout in ms | 5000 |
| {service.upper()}_MAX_CONNECTIONS | Max connections | 100 |

### Example Configuration

```yaml
{service}:
  host: ${{{service.upper()}_HOST}}
  port: ${{{service.upper()}_PORT}}
  timeout: ${{{service.upper()}_TIMEOUT}}
  pool:
    min: 10
    max: 100
```

### Best Practices
- Always set explicit timeouts to prevent hanging connections
- Use connection pooling for better performance
- Monitor health endpoints regularly"""


def _generate_tutorial_doc() -> str:
    """Generate a fake tutorial section."""
    topics = ["authentication", "pagination", "error handling", "caching", "webhooks"]
    topic = random.choice(topics)
    return f"""## Tutorial: {topic.title()}

### Overview
This guide covers how to implement {topic} in your application.

### Prerequisites
- API key configured
- SDK version 2.0+
- Python 3.9+

### Step 1: Setup
First, configure your client:
```python
client = Client(api_key=os.environ["API_KEY"])
```

### Step 2: Implementation
Here's the basic pattern for {topic}:
```python
def handle_{topic.replace(" ", "_")}(request):
    # Validate input
    if not request.is_valid:
        raise ValidationError("Invalid request")

    # Process
    result = client.process(request)

    # Return response
    return Response(data=result)
```

### Step 3: Testing
Verify your implementation:
```bash
pytest tests/test_{topic.replace(" ", "_")}.py -v
```

### Common Issues
- Issue: Timeout errors -> Solution: Increase timeout value
- Issue: Rate limiting -> Solution: Implement exponential backoff
- Issue: Invalid tokens -> Solution: Refresh credentials"""


def _generate_faq_doc() -> str:
    """Generate a fake FAQ section."""
    return """## Frequently Asked Questions

### Q: How do I authenticate?
A: Use API key authentication by including your key in the Authorization header:
```
Authorization: Bearer <your-api-key>
```

### Q: What are the rate limits?
A: Standard tier: 100 req/min. Premium: 1000 req/min. Enterprise: Custom.

### Q: How do I handle pagination?
A: Use the `limit` and `offset` parameters. Check `meta.total` for total count.

### Q: What formats are supported?
A: JSON (default), XML (legacy), and Protocol Buffers (beta).

### Q: How do I report issues?
A: Open a ticket at support.example.com or email support@example.com."""


def _generate_rag_question(idx: int) -> str:
    """Generate a question about RAG context."""
    questions = [
        "What are the rate limits for the API?",
        "How do I configure the database connection?",
        "What authentication method should I use?",
        "How do I handle pagination in responses?",
        "What are the common error codes?",
    ]
    return questions[idx % len(questions)]


def _generate_rag_answer(idx: int) -> str:
    """Generate an answer based on RAG context."""
    answers = [
        "According to the documentation, the rate limits are 100 requests per minute for standard tier and 1000 requests per minute for premium tier.",
        "Based on the configuration docs, you should set the DATABASE_HOST and DATABASE_PORT environment variables. Connection pooling is recommended with min=10 and max=100 connections.",
        "The documents indicate that API key authentication is the recommended method. Include your key in the Authorization header as a Bearer token.",
        "For pagination, use the `limit` and `offset` parameters in your requests. The `meta.total` field in the response shows the total count of available records.",
        "Common error codes include: 400 (Invalid request), 401 (Authentication required), 429 (Rate limit exceeded), and 500 (Internal server error).",
    ]
    return answers[idx % len(answers)]
