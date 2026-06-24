# Agno Integration

Headroom integrates with [Agno](https://github.com/agno-agi/agno) (formerly Phidata) to provide automatic context optimization for AI agents. This guide covers model wrapping, observability hooks, and multi-provider support.

---

## Installation

```bash
pip install "headroom-ai[agno]"
```

This installs Headroom with Agno support. You'll also need Agno itself:

```bash
pip install agno
```

---

## Quick Start

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from headroom.integrations.agno import HeadroomAgnoModel

# Wrap your model
model = HeadroomAgnoModel(OpenAIChat(id="gpt-4o"))

# Create agent as usual
agent = Agent(model=model)

# Use exactly like before
response = agent.run("What's the capital of France?")

# Check savings
print(f"Tokens saved: {model.total_tokens_saved}")
print(model.get_savings_summary())
# {'total_requests': 1, 'total_tokens_saved': 245, 'average_savings_percent': 12.3}
```

---

## Integration Patterns

### 1. Basic Model Wrapping

The simplest integration - wrap any Agno model with `HeadroomAgnoModel`:

```python
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.models.google import Gemini
from headroom.integrations.agno import HeadroomAgnoModel

# Works with any Agno model
openai_model = HeadroomAgnoModel(OpenAIChat(id="gpt-4o"))
claude_model = HeadroomAgnoModel(Claude(id="claude-3-5-sonnet-20241022"))
gemini_model = HeadroomAgnoModel(Gemini(id="gemini-2.0-flash"))

# Each automatically uses the correct provider for accurate token counting
```

**Why this matters**: Headroom automatically detects the underlying provider and applies the correct tokenizer for accurate optimization metrics.

### 2. Agent with Observability Hooks

Use hooks for detailed tracking without modifying your model:

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from headroom.integrations.agno import (
    HeadroomAgnoModel,
    HeadroomPreHook,
    HeadroomPostHook,
)

# Model wrapper for optimization
model = HeadroomAgnoModel(OpenAIChat(id="gpt-4o"))

# Hooks for observability
pre_hook = HeadroomPreHook()
post_hook = HeadroomPostHook(token_alert_threshold=10000)

agent = Agent(
    model=model,
    pre_hooks=[pre_hook],
    post_hooks=[post_hook],
)

# Run agent
response = agent.run("Analyze this large dataset...")

# Check metrics from model
print(f"Tokens saved: {model.total_tokens_saved}")

# Check observability from hooks
print(f"Post-hook summary: {post_hook.get_summary()}")
print(f"Alerts triggered: {post_hook.alerts}")
```

**Why this matters**: Hooks provide observability into agent behavior and can alert when token usage exceeds thresholds.

### 3. Convenience Hook Factory

Use `create_headroom_hooks()` to create matched hook pairs:

```python
from headroom.integrations.agno import create_headroom_hooks

pre_hook, post_hook = create_headroom_hooks(
    token_alert_threshold=5000,
    log_level="DEBUG",
)

agent = Agent(
    model=model,
    pre_hooks=[pre_hook],
    post_hooks=[post_hook],
)
```

### 4. Custom Configuration

Pass a `HeadroomConfig` for fine-grained control:

```python
from headroom import HeadroomConfig, HeadroomMode
from headroom.integrations.agno import HeadroomAgnoModel

config = HeadroomConfig(
    default_mode=HeadroomMode.OPTIMIZE,
    # Add other configuration options as needed
)

model = HeadroomAgnoModel(
    wrapped_model=OpenAIChat(id="gpt-4o"),
    config=config,
)
```

### 5. Standalone Message Optimization

Optimize messages without wrapping a model:

```python
from headroom.integrations.agno import optimize_messages

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Analyze this large JSON: " + large_json},
]

optimized_messages, metrics = optimize_messages(messages, model="gpt-4o")

print(f"Tokens saved: {metrics['tokens_saved']}")
print(f"Transforms applied: {metrics['transforms_applied']}")
```

### 6. Async Operations

Full async support for high-throughput applications:

```python
import asyncio
from headroom.integrations.agno import HeadroomAgnoModel

async def process_async():
    model = HeadroomAgnoModel(OpenAIChat(id="gpt-4o"))

    # Async response
    response = await model.aresponse(messages)

    # Async streaming
    async for chunk in model.aresponse_stream(messages):
        print(chunk, end="", flush=True)

    print(f"\nTokens saved: {model.total_tokens_saved}")

asyncio.run(process_async())
```

---

## Real-World Examples

### Example 1: Tool-Heavy Agent

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from headroom.integrations.agno import HeadroomAgnoModel

# Wrap model for optimization
model = HeadroomAgnoModel(OpenAIChat(id="gpt-4o"))

# Agent with search tools
agent = Agent(
    model=model,
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
)

# Tool outputs get compressed automatically
response = agent.run("Research the latest AI developments and summarize")

# Impact: Tool outputs (often 10K+ tokens) compressed by 70-90%
print(f"Tokens saved: {model.total_tokens_saved}")
print(model.get_savings_summary())
```

### Example 2: Multi-Model Routing

```python
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from headroom.integrations.agno import HeadroomAgnoModel

# Different models for different tasks
fast_model = HeadroomAgnoModel(OpenAIChat(id="gpt-4o-mini"))
powerful_model = HeadroomAgnoModel(Claude(id="claude-3-5-sonnet-20241022"))

# Use fast model for simple tasks
simple_agent = Agent(model=fast_model)

# Use powerful model for complex reasoning
complex_agent = Agent(model=powerful_model)

# Each tracks its own metrics
print(f"Fast model saved: {fast_model.total_tokens_saved}")
print(f"Powerful model saved: {powerful_model.total_tokens_saved}")
```

### Example 3: Production Monitoring

```python
from agno.agent import Agent
from headroom.integrations.agno import (
    HeadroomAgnoModel,
    create_headroom_hooks,
)

model = HeadroomAgnoModel(OpenAIChat(id="gpt-4o"))
pre_hook, post_hook = create_headroom_hooks(
    token_alert_threshold=50000,  # Alert on large requests
    log_level="WARNING",
)

agent = Agent(
    model=model,
    pre_hooks=[pre_hook],
    post_hooks=[post_hook],
)

# Run multiple requests
for query in user_queries:
    response = agent.run(query)

# Check for alerts
if post_hook.alerts:
    print(f"WARNING: {len(post_hook.alerts)} requests exceeded threshold")
    for alert in post_hook.alerts:
        print(f"  - {alert}")

# Summary stats
summary = post_hook.get_summary()
print(f"Total requests: {summary['total_requests']}")
print(f"Average tokens: {summary['average_tokens']}")
```

### Example 4: Reset for New Sessions

```python
model = HeadroomAgnoModel(OpenAIChat(id="gpt-4o"))

# Session 1
agent.run("First conversation...")
print(f"Session 1 savings: {model.get_savings_summary()}")

# Reset for new session
model.reset()

# Session 2 - metrics start fresh
agent.run("Second conversation...")
print(f"Session 2 savings: {model.get_savings_summary()}")
```

---

## Supported Providers

HeadroomAgnoModel automatically detects the provider from the wrapped model:

| Provider | Agno Models | Auto-Detected |
|----------|-------------|---------------|
| **OpenAI** | `OpenAIChat`, `OpenAILike` | Yes |
| **Anthropic** | `Claude`, `AwsBedrock` | Yes |
| **Google** | `Gemini`, `VertexAI` | Yes |
| **Cohere** | `Cohere`, `CohereChat` | Yes |
| **Groq** | `Groq` | Yes (OpenAI-compatible) |
| **Mistral** | `Mistral` | Yes (OpenAI-compatible) |
| **Together** | `Together` | Yes (OpenAI-compatible) |
| **Ollama** | `Ollama` | Yes (OpenAI-compatible) |

To disable auto-detection:

```python
model = HeadroomAgnoModel(
    wrapped_model=some_model,
    auto_detect_provider=False,  # Falls back to OpenAI tokenizer
)
```

---

## Feature Coverage

### What's Optimized

HeadroomAgnoModel optimizes messages at the LLM call boundary. This covers:

| Feature | Optimized | Notes |
|---------|-----------|-------|
| **User/Assistant Messages** | ✅ Yes | Full message history compressed |
| **Tool Calls** | ✅ Yes | Tool call arguments optimized |
| **Tool Results** | ✅ Yes | JSON responses compressed 70-90% via SmartCrusher |
| **System Prompts** | ✅ Yes | Included in message optimization |
| **Streaming Responses** | ✅ Yes | Both sync and async |
| **Multi-turn Conversations** | ✅ Yes | Full history available for optimization |

### Known Limitations

The integration operates at the model layer, not the agent layer. Some Agno features operate outside this boundary:

| Agno Feature | Status | Explanation |
|--------------|--------|-------------|
| **Agent Memory** | ⚠️ Partial | Memory content is optimized when it enters messages, but the persistent memory store itself is not compressed. If you're storing large amounts of data in agent memory, consider summarizing before storage. |
| **Knowledge Bases** | ⚠️ Partial | KB retrieval happens before messages reach the model. Retrieved context is optimized as part of the message, but we can't influence KB retrieval itself. |
| **Agent Teams** | ❌ Not supported | Each agent's model is wrapped independently. No cross-agent optimization or team-level coordination. |
| **Tool Definitions** | ⚠️ Not deduplicated | Tool schemas are sent with every request. Future versions may deduplicate repeated tool definitions. |
| **Structured Outputs** | ✅ Supported | `response_model` works normally; optimization doesn't affect output parsing. |
| **Reasoning Models** | ✅ Supported | Extended thinking works; we don't compress reasoning traces. |

### Best Practices for Maximum Savings

1. **Tool-heavy agents see the biggest wins** — Tool results (JSON, logs, search results) compress 70-90%
2. **Long conversations benefit from RollingWindow** — Configure context limits to avoid hitting provider maximums
3. **Wrap at the model level, not agent level** — This ensures all LLM calls go through optimization
4. **Use hooks for observability** — Track token usage patterns to identify optimization opportunities

### Future Improvements

We're tracking these potential enhancements:

- **Memory optimization hooks** — Compress data before it enters agent memory
- **Knowledge base integration** — Optimize retrieved context at the KB layer
- **Tool schema deduplication** — Cache and reference repeated tool definitions
- **Team-level optimization** — Shared context compression across agent teams

Contributions welcome! See [CONTRIBUTING.md](https://github.com/chopratejas/headroom/blob/main/CONTRIBUTING.md).

---

## Configuration Reference

### HeadroomAgnoModel

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wrapped_model` | Any | Required | The Agno model to wrap |
| `config` | `HeadroomConfig` | `None` | Custom configuration |
| `auto_detect_provider` | `bool` | `True` | Auto-detect provider for token counting |

**Properties:**
- `wrapped_model` - Access the underlying Agno model
- `total_tokens_saved` - Running total of tokens saved
- `metrics_history` - List of last 100 `OptimizationMetrics`

**Methods:**
- `response(messages, **kwargs)` - Sync response with optimization
- `response_stream(messages, **kwargs)` - Sync streaming response
- `aresponse(messages, **kwargs)` - Async response
- `aresponse_stream(messages, **kwargs)` - Async streaming
- `get_savings_summary()` - Returns dict with stats
- `reset()` - Clear all metrics

### HeadroomPreHook

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `HeadroomConfig` | `None` | Configuration (for future use) |
| `model` | `str` | `"gpt-4o"` | Model name for estimation |

### HeadroomPostHook

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_level` | `str` | `"INFO"` | Logging level |
| `token_alert_threshold` | `int` | `None` | Alert if tokens exceed this |

**Properties:**
- `total_requests` - Number of requests tracked
- `alerts` - List of alert messages

**Methods:**
- `get_summary()` - Returns dict with request stats
- `reset()` - Clear history and alerts

### create_headroom_hooks()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `HeadroomConfig` | `None` | Config for pre-hook |
| `model` | `str` | `"gpt-4o"` | Model for pre-hook |
| `log_level` | `str` | `"INFO"` | Log level for post-hook |
| `token_alert_threshold` | `int` | `None` | Alert threshold for post-hook |

Returns: `tuple[HeadroomPreHook, HeadroomPostHook]`

---

## Import Reference

```python
# Main integration
from headroom.integrations.agno import HeadroomAgnoModel

# Hooks
from headroom.integrations.agno import HeadroomPreHook
from headroom.integrations.agno import HeadroomPostHook
from headroom.integrations.agno import create_headroom_hooks

# Utilities
from headroom.integrations.agno import optimize_messages
from headroom.integrations.agno import agno_available
from headroom.integrations.agno import get_headroom_provider
from headroom.integrations.agno import get_model_name_from_agno

# Or import everything from parent
from headroom.integrations import (
    HeadroomAgnoModel,
    HeadroomPreHook,
    HeadroomPostHook,
    create_headroom_hooks,
)
```

---

## Troubleshooting

### Check if Agno is Available

```python
from headroom.integrations.agno import agno_available

if agno_available():
    from headroom.integrations.agno import HeadroomAgnoModel
else:
    print("Install agno: pip install agno")
```

### Provider Detection Issues

If auto-detection fails, check the detected provider:

```python
from headroom.integrations.agno import get_headroom_provider, get_model_name_from_agno

model = OpenAIChat(id="gpt-4o")
provider = get_headroom_provider(model)
model_name = get_model_name_from_agno(model)

print(f"Detected provider: {type(provider).__name__}")
print(f"Model name: {model_name}")
```

### Metrics Not Updating

Ensure you're checking the correct object:

```python
# Model metrics (optimization)
print(model.total_tokens_saved)  # Actual savings

# Hook metrics (observability)
print(post_hook.get_summary())  # Request tracking
```

Note: Hooks track request counts, not token savings. Use the model wrapper for optimization metrics.
