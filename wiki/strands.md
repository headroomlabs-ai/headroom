# Strands Integration

Headroom integrates with [Strands Agents](https://github.com/strands-agents/sdk-python) to provide automatic context optimization. Two integration patterns: wrap the model, or hook into tool calls.

---

## Installation

```bash
pip install headroom-ai strands-agents
```

---

## Quick Start

```python
from strands import Agent
from strands.models.bedrock import BedrockModel
from headroom.integrations.strands import HeadroomStrandsModel

# Wrap your model
model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
optimized = HeadroomStrandsModel(wrapped_model=model)

# Create agent as usual
agent = Agent(model=optimized)
response = agent("Investigate the production incident")

# Check savings
print(f"Tokens saved: {optimized.total_tokens_saved}")
```

Every API call the agent makes — including tool result round-trips — gets compressed automatically.

---

## Integration Patterns

### 1. Model Wrapping

Wraps the Strands `Model` interface. Every call to `stream()` compresses the messages before they hit the provider.

```python
from strands.models.bedrock import BedrockModel
from headroom.integrations.strands import HeadroomStrandsModel

model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
optimized = HeadroomStrandsModel(wrapped_model=model)

# Streaming works identically
agent = Agent(model=optimized)
response = agent("Analyze these logs")
```

With custom config:

```python
from headroom import HeadroomConfig

config = HeadroomConfig()
optimized = HeadroomStrandsModel(wrapped_model=model, config=config)
```

### 2. Hook Provider (Tool Output Compression)

Compresses tool call results via Strands' hook system. Uses SmartCrusher on JSON arrays returned by tools.

```python
from strands import Agent
from strands.models.bedrock import BedrockModel
from headroom.integrations.strands import HeadroomHookProvider

model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
hooks = HeadroomHookProvider(
    compress_tool_outputs=True,
    min_tokens_to_compress=200,
    preserve_errors=True,
)

agent = Agent(model=model, hooks=[hooks])
response = agent("Search the database for recent failures")

# Check tool compression savings
print(f"Tokens saved by hooks: {hooks.total_tokens_saved}")
```

The hook preserves:

- Error items (error indicators, exceptions)
- Anomalous values (statistical outliers)
- Items matching the user's query context
- First/last items for boundary context

### 3. Both Together

Model wrapping compresses conversation history. Hooks compress individual tool results. Use both for maximum savings.

```python
from headroom.integrations.strands import HeadroomStrandsModel, HeadroomHookProvider

optimized = HeadroomStrandsModel(wrapped_model=model)
hooks = HeadroomHookProvider(compress_tool_outputs=True)

agent = Agent(model=optimized, hooks=[hooks])
```

---

## Structured Output

HeadroomStrandsModel supports Strands' structured output feature:

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    severity: str
    root_cause: str
    recommendation: str

result = optimized.structured_output(Analysis, messages)
```

---

## Metrics

```python
# Per-request metrics
for m in optimized.metrics_history:
    print(f"  {m.tokens_before} → {m.tokens_after} ({m.tokens_saved} saved)")

# Running total
print(f"Total saved: {optimized.total_tokens_saved}")
```

---

## How It Works

```
Agent decides to call tool
    │
    ▼
Tool executes, returns result
    │
    ▼
HeadroomHookProvider (optional)
    compresses tool result JSON
    │
    ▼
Agent builds next API request
    │
    ▼
HeadroomStrandsModel.stream()
    compresses full message list
    │
    ▼
Provider API (Bedrock, etc.)
```

The model wrapper uses Headroom's full pipeline (CacheAligner → ContentRouter → IntelligentContext). The hook provider uses SmartCrusher directly for fast JSON compression of individual tool results.

---

## Supported Providers

HeadroomStrandsModel auto-detects the provider from the wrapped model:

| Strands Model | Provider Detected |
|--------------|-------------------|
| `BedrockModel` | Anthropic (via Bedrock) |
| `OllamaModel` | OpenAI-compatible |
| Custom `Model` | Falls back to estimation |
